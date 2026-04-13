import onnx
import tvm
import tvm.relay as relay

# Import components from the refactored compiler pipeline
from emitter import NPUFullProgramEmitter
from assembler import text_to_hex_full

def main():
    # 1. Load the ONNX model
    model_path = "../Model/train/weights/best_int8.onnx"
    onnx_model = onnx.load(model_path)
    mod, params = relay.frontend.from_onnx(onnx_model, {"images": (1, 3, 640, 640)})
    
    #print(mod)
    print("Running optimization passes and INT8 weight quantization")

    # 2. Apply TVM optimization passes
    with tvm.transform.PassContext(opt_level=1):
        mod["main"] = relay.build_module.bind_params_by_name(mod["main"], params)
        mod = relay.transform.InferType()(mod)
        mod = relay.transform.FakeQuantizationToInteger()(mod)
        mod = relay.transform.FoldConstant()(mod) 
        # mod = relay.transform.FuseOps(fuse_opt_level = 1)(mod)
        # mod = relay.transform.InferType()(mod_fused) 
        # print(mod)
        
        # 3. Execute the instruction emitter
        emitter = NPUFullProgramEmitter(params)
        emitter.visit(mod["main"])
        
    emitter.instructions.append("OP:HALT   | IN:0x00000000 | WGT:0x00000000 | OUT:0x00000000 | FLAGS:0x0 | STRIDE:0 | PAD:0 | KERNEL:0")
    
    # 4. Write intermediate instructions and assemble to hex
    with open("../Build/full_instructions.txt", "w") as f:
        for cmd in emitter.instructions: 
            f.write(cmd + "\n")
            
    text_to_hex_full("../Build/full_instructions.txt", "../Build/npu_program.hex")
    
    # 5. Export quantized weights
    final_weight_bytes = emitter.weight_memory[:emitter.max_weight_offset]
    with open("../Build/weights.bin", "wb") as f:
        f.write(final_weight_bytes)

    # 6. Generate CPU fallback runtime for unsupported operations
    print("Generating CPU runtime drivers (cpu_runtime.py)...")
    with open("../Build/cpu_runtime.py", "w", encoding="utf-8") as f:
        f.write("import numpy as np\n\n")
        for op in sorted(emitter.cpu_op_names):
            f.write(f"def cpu_execute_{op.replace('.','_')}(sram, in_addr, out_addr, h, w, c):\n")
            f.write(f"    pass\n\n")

        f.write("def handle_npu_interrupt(sram, interrupt_pc):\n")
        if not emitter.cpu_tasks:
            f.write("    pass\n")
        else:
            f.write("    if interrupt_pc < 0:\n        pass\n")
            for task in emitter.cpu_tasks:
                f.write(task)
                
    print(f"Compilation complete. Weight file size: {len(final_weight_bytes)/1024/1024:.2f} MB")

if __name__ == "__main__": 
    main()