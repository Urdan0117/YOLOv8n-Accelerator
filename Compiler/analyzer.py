import tvm
import tvm.relay as relay

class UniversalLayerAnalyzer(relay.ExprVisitor):
    def __init__(self):
        super().__init__()
        self.ops_to_emit = [] 
        self.current_flags = 0

    def visit_call(self, call):
        super().visit_call(call)
        if isinstance(call.op, tvm.ir.Op):
            op_name = call.op.name

            if op_name in ["sigmoid", "qnn.sigmoid"]: 
                self.current_flags |= 0x1
                self.ops_to_emit.append(("IGNORE", call, 0))
            elif op_name in ["multiply", "qnn.mul"]: 
                self.current_flags |= 0x2
                self.ops_to_emit.append(("IGNORE", call, 0))
            elif op_name in ["nn.relu", "relu", "clip"]: 
                self.current_flags |= 0x4
                self.ops_to_emit.append(("IGNORE", call, 0))

            elif op_name == "qnn.requantize":
                

                def get_core_op(node):
                    if not isinstance(node, relay.Call): return None
                    if node.op.name in ["qnn.conv2d", "nn.conv2d", "qnn.add", "add"]: 
                        return node.op.name

                    if node.op.name in ["nn.bias_add", "qnn.mul", "qnn.sigmoid", "clip"]:
                        return get_core_op(node.args[0])
                    return None
                
                core_op = get_core_op(call.args[0])

                if core_op in ["qnn.conv2d", "nn.conv2d"]:
                    self.ops_to_emit.append(("FUSED_QNN_CONV", call, self.current_flags))
                elif core_op in ["qnn.add", "add"]:
                    self.ops_to_emit.append(("FUSED_QNN_ADD", call, self.current_flags))
                else:
                    self.ops_to_emit.append(("IGNORE", call, 0)) 
                
                self.current_flags = 0


            elif op_name in ["qnn.conv2d", "nn.bias_add", "qnn.add", "nn.conv2d"]:
                self.ops_to_emit.append(("IGNORE", call, 0))
            elif op_name in ["qnn.quantize", "qnn.dequantize", "cast", "right_shift"]:
                self.ops_to_emit.append(("IGNORE", call, 0))

            elif op_name in ["reshape", "split", "strided_slice", "transpose", "squeeze", "expand_dims"]:
                self.ops_to_emit.append(("IGNORE", call, 0))


            elif op_name == "nn.max_pool2d":
                self.ops_to_emit.append(("POOL", call, self.current_flags))
                self.current_flags = 0
            elif "concatenate" in op_name:
                self.ops_to_emit.append(("CONCAT", call, self.current_flags))
                self.current_flags = 0

            # CPU
            elif op_name in ["image.resize2d", "nn.softmax", "divide", "subtract", "qnn.subtract"]:
                self.ops_to_emit.append(("OTHER", call, self.current_flags))
                self.current_flags = 0
            else:
                self.ops_to_emit.append(("OTHER", call, self.current_flags))
                self.current_flags = 0