import onnxruntime as ort
import numpy as np
import json
import os

# 時間單位自動轉換器 (微秒 us -> 毫秒 ms)
def format_time(us_val):
    if us_val >= 1000:
        return f"{us_val/1000:.3f}ms"
    else:
        return f"{us_val:.3f}us"

def main():
    onnx_model_path = "yolov8n.onnx " #best_int8_custom.onnx #yolov8n.onnx 
    
    if not os.path.exists(onnx_model_path):
        print(f"❌ 找不到檔案 {onnx_model_path}")
        return

    # 1. 執行 Profiling
    options = ort.SessionOptions()
    options.enable_profiling = True
    session = ort.InferenceSession(onnx_model_path, options)
    
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)

    session.run(None, {input_name: dummy_input})
    session.run(None, {input_name: dummy_input}) # 第二次推論讓時間更穩定

    prof_file = session.end_profiling()

    # 2. 讀取 JSON 並彙整數據
    with open(prof_file, 'r') as f:
        trace_data = json.load(f)

    op_stats = {}
    total_time_us = 0

    for event in trace_data:
        if event.get('cat') == 'Node' and 'dur' in event:
            dur = event['dur']
            op_name = event.get('args', {}).get('op_name', event['name'])
            
            if op_name not in op_stats:
                op_stats[op_name] = {'dur': 0, 'calls': 0}
            
            op_stats[op_name]['dur'] += dur
            op_stats[op_name]['calls'] += 1
            total_time_us += dur

    # 依時間由長到短排序
    sorted_ops = sorted(op_stats.items(), key=lambda x: x[1]['dur'], reverse=True)

    # 3. 完美模仿 PyTorch 輸出格式
    print("\n=== Profiling Results ===")
    
    # 分隔線
    sep = "-"*33 + "  " + "------------  "*8
    print(sep)
    
    # 標題行
    header = "{:>33}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}".format(
        "Name", "Self CPU %", "Self CPU", "CPU total %", "CPU total", "CPU time avg", "CPU Mem", "Self CPU Mem", "# of Calls"
    )
    print(header)
    print(sep)

    # 內容資料行 (印出前 15 名)
    for op, stats in sorted_ops[:15]:
        dur = stats['dur']
        calls = stats['calls']
        percent = (dur / total_time_us) * 100 if total_time_us > 0 else 0
        
        percent_str = f"{percent:.2f}%"
        time_str = format_time(dur)
        avg_time_str = format_time(dur / calls)
        mem_str = "--" # ONNX trace 不含記憶體數據，以 -- 表示
        
        row_str = "{:>33}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}".format(
            op, percent_str, time_str, percent_str, time_str, avg_time_str, mem_str, mem_str, calls
        )
        print(row_str)

    print(sep)
    print(f"Self CPU time total: {format_time(total_time_us)}")
    print()

    # 清除佔空間的 JSON 檔
    os.remove(prof_file)

if __name__ == "__main__":
    main()