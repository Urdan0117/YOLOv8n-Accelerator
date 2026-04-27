import sys
from pathlib import Path

import torch
import torch.nn as nn
import onnx

project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

from layer_info import (
    ShapeParam,
    Conv2DShapeParam,
    LinearShapeParam,
    MaxPool2DShapeParam,
)

from lib.models.vgg import VGG
from network_parser import torch2onnx

def parse_pytorch(model: nn.Module, input_shape=(1, 3, 32, 32)) -> list[ShapeParam]:
    layers = []
    #! <<<========= Implement here =========>>>
    # 定義 Hook 函數
    def hook_fn(module, inputs, output):
        module_name = type(module).__name__
        
        # 使用字串包含 "Conv" 來判斷，這樣連 ConvReLU2d 或 QConv2d 都能抓到！
        if "Conv" in module_name:
            param = Conv2DShapeParam(
                N=output.shape[0],
                H=inputs[0].shape[2],
                W=inputs[0].shape[3],
                C=module.in_channels,
                M=module.out_channels,
                # 處理 kernel_size / padding / stride 可能是 int 或 tuple 的情況
                R=module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size,
                S=module.kernel_size[1] if isinstance(module.kernel_size, tuple) else module.kernel_size,
                E=output.shape[2],
                F=output.shape[3],
                P=module.padding[0] if isinstance(module.padding, tuple) else module.padding,
                U=module.stride[0] if isinstance(module.stride, tuple) else module.stride
            )
            layers.append(param)
            
        # 使用字串判斷 MaxPool
        elif "MaxPool" in module_name:
            param = MaxPool2DShapeParam(
                N=output.shape[0],
                kernel_size=module.kernel_size,
                stride=module.stride
            )
            layers.append(param)


        # 使用字串判斷Linear
        elif "Linear" in module_name:
            param = LinearShapeParam(
                N=output.shape[0],
                in_features=module.in_features,
                out_features=module.out_features
            )
            layers.append(param)

    # 註冊 hooks
    hooks = []
    for module in model.modules():
        module_name = type(module).__name__
        # 這裡也要用字串判斷，否則 hook 根本不會綁定到融合後的卷積層上！
        if "Conv" in module_name or "MaxPool" in module_name or "Linear" in module_name:
            hooks.append(module.register_forward_hook(hook_fn))

    # 餵入假資料來觸發 forward pass
    dummy_input = torch.randn(input_shape)
    # 如果有使用 GPU，需要確保 dummy_input 和 model 在同一個 device
    #device = next(model.parameters()).device
    p = next(model.parameters(), None)
    if p is None:
        p = next(model.buffers(), torch.tensor(0))
    device = p.device
    model(dummy_input.to(device))

    # 移除 hooks
    for hook in hooks:
        hook.remove()

    return layers



def parse_onnx(model: onnx.ModelProto) -> list[ShapeParam]:
    #! <<<========= Implement here =========>>>
    
    import onnx.shape_inference
    # 先做 Shape Inference，否則 ONNX 裡面可能沒有中間層的尺寸資訊
    inferred_model = onnx.shape_inference.infer_shapes(model)
    graph = inferred_model.graph
    
    # 建立一個名稱到 Shape 的映射字典，方便查詢
    def get_shape(name):
        # 1. 從 value_info, input, 或 output 中找尋
        for info in list(graph.value_info) + list(graph.input) + list(graph.output):
            if info.name == name:
                return [dim.dim_value for dim in info.type.tensor_type.shape.dim]
        
        # 2. 搜尋權重 (Initializer)
        for init in graph.initializer:
            if init.name == name:
                return list(init.dims)

    layers = []
    for node in graph.node:
        # 1. 處理 Conv
        if node.op_type == "Conv":
            input_shape = get_shape(node.input[0])
            output_shape = get_shape(node.output[0])
            weight_shape = get_shape(node.input[1]) # [M, C, R, S]
            
            # 從屬性 (Attribute) 中提取 Padding 和 Stride
            attrs = {attr.name: attr.ints for attr in node.attribute}
            pads = attrs.get("pads", [0, 0])
            strides = attrs.get("strides", [1, 1])

            param = Conv2DShapeParam(
                N=output_shape[0],
                H=input_shape[2],
                W=input_shape[3],
                C=weight_shape[1],
                M=weight_shape[0],
                R=weight_shape[2],
                S=weight_shape[3],
                E=output_shape[2],
                F=output_shape[3],
                P=pads[0],
                U=strides[0]
            )
            layers.append(param)

        # 2. 處理 Gemm (ONNX 的 Linear 通常對應到 Gemm)
        elif node.op_type == "Gemm":
            input_shape = get_shape(node.input[0])
            output_shape = get_shape(node.output[0])
            # Gemm 的 input 通常是 [N, in_features]
            param = LinearShapeParam(
                N=output_shape[0],
                in_features=input_shape[1],
                out_features=output_shape[1]
            )
            layers.append(param)

        # 3. 處理 MaxPool
        elif node.op_type == "MaxPool":
            output_shape = get_shape(node.output[0])
            attrs = {attr.name: attr.ints for attr in node.attribute}
            k_size = attrs.get("kernel_shape", [2, 2])[0]
            stride = attrs.get("strides", [2, 2])[0]
            param = MaxPool2DShapeParam(
                N=output_shape[0],
                kernel_size=k_size,
                stride=stride
            )
            layers.append(param)

    return layers

def compare_layers(answer, layers):
    if len(answer) != len(layers):
        print(
            f"Layer count mismatch: answer has {len(answer)}, but ONNX has {len(layers)}"
        )

    min_len = min(len(answer), len(layers))

    for i in range(min_len):
        ans_layer = vars(answer[i])
        layer = vars(layers[i])

        diffs = {
            k: (ans_layer[k], layer[k])
            for k in ans_layer
            if k in layer and ans_layer[k] != layer[k]
        }

        if diffs:
            print(f"Difference in layer {i + 1} ({type(answer[i]).__name__}):")
            for k, (ans_val, val) in diffs.items():
                print(f"  {k}: answer = {ans_val}, onnx = {val}")

    if len(answer) > len(layers):
        print(f"Extra layers in answer: {answer[len(layers) :]}")
    elif len(layers) > len(answer):
        print(f"Extra layers in yours: {layers[len(answer) :]}")


def run_tests() -> None:
    """Run tests on the network parser functions."""
    answer = [
        Conv2DShapeParam(N=1, H=32, W=32, R=3, S=3, E=32, F=32, C=3, M=64, U=1, P=1),
        MaxPool2DShapeParam(N=1, kernel_size=2, stride=2),
        Conv2DShapeParam(N=1, H=16, W=16, R=3, S=3, E=16, F=16, C=64, M=192, U=1, P=1),
        MaxPool2DShapeParam(N=1, kernel_size=2, stride=2),
        Conv2DShapeParam(N=1, H=8, W=8, R=3, S=3, E=8, F=8, C=192, M=384, U=1, P=1),
        Conv2DShapeParam(N=1, H=8, W=8, R=3, S=3, E=8, F=8, C=384, M=256, U=1, P=1),
        Conv2DShapeParam(N=1, H=8, W=8, R=3, S=3, E=8, F=8, C=256, M=256, U=1, P=1),
        MaxPool2DShapeParam(N=1, kernel_size=2, stride=2),
        LinearShapeParam(N=1, in_features=4096, out_features=256),
        LinearShapeParam(N=1, in_features=256, out_features=128),
        LinearShapeParam(N=1, in_features=128, out_features=10),
    ]

    # Test with the PyTorch model.
    model = VGG()
    layers_pth = parse_pytorch(model)

    # Define the input shape.
    dummy_input = torch.randn(1, 3, 32, 32)
    # Save the model to ONNX.
    torch2onnx.torch2onnx(model, "parser_onnx.onnx", dummy_input)
    # Load the ONNX model.
    model_onnx = onnx.load("parser_onnx.onnx")
    layers_onnx = parse_onnx(model_onnx)

    # Display results.
    print("PyTorch Network Parser:")
    if layers_pth == answer:
        print("Correct!")
    else:
        print("Wrong!")
        compare_layers(answer, layers_pth)

    print("ONNX Network Parser:")
    if layers_onnx == answer:
        print("Correct!")
    else:
        print("Wrong!")
        compare_layers(answer, layers_onnx)


if __name__ == "__main__":
    run_tests()
