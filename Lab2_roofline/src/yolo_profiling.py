import torch
from torch.profiler import profile, record_function, ProfilerActivity
from ultralytics import YOLO

def main():
    print("Loading YOLOv8n model...")
    # 自動下載並載入 YOLOv8n 模型
    model = YOLO("yolov8n.pt").model
    model.eval()

    # YOLOv8 預設的影像輸入大小是 1張圖 x 3通道 x 640 x 640
    print("Preparing dummy input (1x3x640x640)...")
    dummy_input = torch.randn(1, 3, 640, 640)

    print("Starting Profiling...")
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with record_function("model_inference"):
            # 進行一次推論
            model(dummy_input)

    # 印出最花時間的前 15 個 Operator
    print("\n=== Profiling Results ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

if __name__ == "__main__":
    main()