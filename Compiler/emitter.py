import tvm
import tvm.relay as relay
import numpy as np
from utils import get_tensor_shape, float_to_int32_bits
from analyzer import UniversalLayerAnalyzer

class NPUFullProgramEmitter:
    def __init__(self, params_dict):
        self.params_dict = params_dict
        self.instructions = []
        self.pc_counter = 0 
        self.memory_map = {} 
        self.current_free_addr = 0x00500000 
        self.cpu_op_names = set()
        self.cpu_tasks = []
        self.weight_memory = bytearray(64 * 1024 * 1024)
        self.max_weight_offset = 0 

    def visit(self, expr):
        analyzer = UniversalLayerAnalyzer()
        analyzer.visit(expr)
        
        def get_real_addr(node):
            if node in self.memory_map:
                return self.memory_map[node]
            if isinstance(node, relay.TupleGetItem):
                return get_real_addr(node.tuple_value)
            
            if isinstance(node, relay.Call) and node.op.name in [
                "reshape", "qnn.requantize", "cast", "clip", "split", "strided_slice", 
                "transpose", "qnn.sigmoid", "qnn.mul", "squeeze", "expand_dims", "qnn.quantize", "qnn.dequantize"
            ]:
                if len(node.args) > 0:
                    return get_real_addr(node.args[0])
            
            return self.memory_map.get(node, 0x00000000)

        def get_w_np(node):
            if isinstance(node, relay.Constant): return node.data.numpy()
            if isinstance(node, relay.Var) and node.name_hint in self.params_dict: 
                return self.params_dict[node.name_hint].numpy()
            return None

        for op_label, inner_call, flags in analyzer.ops_to_emit:
            
            if op_label == "IGNORE":
                if len(inner_call.args) > 0:
                    self.memory_map[inner_call] = get_real_addr(inner_call.args[0])
                continue

            wgt_addr = 0x40000000 + self.max_weight_offset
            
            in_shape = get_tensor_shape(inner_call.args[0].checked_type) if hasattr(inner_call, "args") and len(inner_call.args) > 0 else [1, 1, 1, 1]
            out_shape = get_tensor_shape(inner_call.checked_type)
            
            in_c, in_h, in_w = 1, 1, 1
            if len(in_shape) >= 4: in_c, in_h, in_w = in_shape[1], in_shape[2], in_shape[3]
            elif len(in_shape) == 3: in_c, in_h, in_w = in_shape[0], in_shape[1], in_shape[2]
            elif len(in_shape) == 2: in_c, in_h, in_w = in_shape[1], 1, 1
                
            out_c = out_shape[1] if len(out_shape) >= 2 else (out_shape[0] if len(out_shape) == 1 else 1)
            layer_size = int(np.prod(out_shape)) * 1 
            
            in_addr = 0x0
            if len(inner_call.args) > 0:
                in_addr = get_real_addr(inner_call.args[0])

            layer_scale = 1.0 
            stride = 1; pad = 0; kernel = 1

            if op_label == "FUSED_QNN_CONV":

                conv_node = inner_call.args[0]
                while isinstance(conv_node, relay.Call) and conv_node.op.name not in ["qnn.conv2d", "nn.conv2d"]:
                    conv_node = conv_node.args[0]

                in_addr = get_real_addr(conv_node.args[0])
                
                w_np = get_w_np(conv_node.args[1])
                if w_np is not None:
                    w_bytes = w_np.astype(np.int8).tobytes()
                    offset = wgt_addr - 0x40000000
                    self.weight_memory[offset : offset + len(w_bytes)] = w_bytes
                    self.max_weight_offset += (len(w_bytes) + 0xF) & ~0xF
                
                in_scale = inner_call.args[1].data.numpy().item()
                out_scale = inner_call.args[3].data.numpy().item()
                layer_scale = in_scale / out_scale
                
                if hasattr(conv_node.attrs, "strides"): stride = int(conv_node.attrs.strides[0])
                if hasattr(conv_node.attrs, "padding"): pad = int(conv_node.attrs.padding[0])
                if hasattr(conv_node.attrs, "kernel_size"): kernel = int(conv_node.attrs.kernel_size[0])
                
                op_label = "CONV"

            elif op_label == "FUSED_QNN_ADD":

                add_node = inner_call.args[0]
                while isinstance(add_node, relay.Call) and add_node.op.name not in ["qnn.add", "add"]:
                    add_node = add_node.args[0]
                
                in_addr = get_real_addr(add_node.args[0])
                wgt_addr = get_real_addr(add_node.args[1])
                
                in_scale = inner_call.args[1].data.numpy().item()
                out_scale = inner_call.args[3].data.numpy().item()
                layer_scale = in_scale / out_scale
                
                op_label = "ADD"

            elif op_label == "CONCAT":
                tup_node = inner_call.args[0]
                if isinstance(tup_node, relay.Tuple):
                    in_addr = get_real_addr(tup_node.fields[0]) if len(tup_node.fields) > 0 else 0x0
                    wgt_addr = get_real_addr(tup_node.fields[1]) if len(tup_node.fields) > 1 else 0x0


            out_addr = self.current_free_addr
            self.memory_map[inner_call] = out_addr 
            self.current_free_addr = (out_addr + layer_size + 0xF) & ~0xF 

            scale_bits = float_to_int32_bits(layer_scale)
            
            is_cpu = (in_w >= 8400 or in_h >= 8400) or op_label == "OTHER"

            if is_cpu:
                op_name = inner_call.op.name if hasattr(inner_call, "op") else "unknown"
                self.cpu_op_names.add(op_name)
                task_code = f"    elif interrupt_pc == {self.pc_counter}:\n" \
                            f"        cpu_execute_{op_name.replace('.', '_')}(sram, 0x{in_addr:08X}, 0x{out_addr:08X}, {in_h}, {in_w}, {in_c})\n"
                self.cpu_tasks.append(task_code)
                self.instructions.append(f"OP:OTHER  | IN:0x{in_addr:08X} | WGT:0x{wgt_addr:08X} | OUT:0x{out_addr:08X} | FLAGS:0x0 | STRIDE:{stride} | PAD:0 | KERNEL:1")
                self.pc_counter += 1
            else:
                self.instructions.append(f"OP:CONFIG | IN_H:{in_h} | IN_W:{in_w} | IN_C:{in_c} | OUT_C:{out_c} | STRIDE:{stride} | SCALE:0x{scale_bits:08X}")
                self.pc_counter += 1
                self.instructions.append(f"OP:{op_label:6s} | IN:0x{in_addr:08X} | WGT:0x{wgt_addr:08X} | OUT:0x{out_addr:08X} | FLAGS:0x{flags:X} | STRIDE:{stride} | PAD:{pad} | KERNEL:{kernel}")
                self.pc_counter += 1