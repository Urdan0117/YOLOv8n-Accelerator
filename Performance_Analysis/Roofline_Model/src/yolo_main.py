import os
from pathlib import Path
import onnx
import pandas as pd
from tqdm import tqdm

# 引入 Lab2 的核心工具
from analytical_model import EyerissMapper
from network_parser import parse_onnx
from roofline import plot_roofline_from_df

def main():
    model_path = "yolov8n.onnx"
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 開始解析 ONNX 模型: {model_path} ...")
    try:
        onnx_model = onnx.load(model_path)
        layers = parse_onnx(onnx_model)
        print(f"✅ 成功解析 {len(layers)} 個支援的運算層！")
    except Exception as e:
        print(f"❌ 解析 ONNX 時發生錯誤:\n{e}")
        return

    # 1. 取得所有硬體候選配置 (從你修改過的 mapper.py 讀取)
    base_mapper = EyerissMapper(name="DSE_Base")
    hw_candidates = base_mapper.generate_hardware()
    print(f"🔍 偵測到 {len(hw_candidates)} 組硬體候選配置，開始進行 DSE 探索...\n")

    all_hw_performance = []
    best_hw_index = -1
    min_edp = float('inf') # 改為追蹤最小 EDP
    best_layers_results = []

    # 2. 針對每一組硬體配置進行評估
    for hw_idx, hw in enumerate(hw_candidates):
        print(f"🏗️ 評估硬體組合 {hw_idx + 1}/{len(hw_candidates)}: PE={hw.pe_array_h}x{hw.pe_array_w}, GLB={hw.glb_size//1024}KB, bus_bw={hw.bus_bw}bit")
        
        results = []
        total_latency = 0
        total_energy = 0
        
        # 使用 tqdm 顯示單組硬體的進度條
        for i, layer in enumerate(tqdm(layers, desc="Processing Layers", leave=False)):
            mapper = EyerissMapper(name=f"yolo_layer_{i}")
            try:
                # 指定當前評估的硬體
                mapper.hardware = hw
                # 執行映射與分析
                res = mapper.run(conv2d=layer, num_solutions=1)[0]
                
                # ====== 終極防呆寫法：不管它傳回什麼格式，我們都硬轉成字典 (保留原模板邏輯) ======
                if isinstance(res, dict):
                    res_dict = res.copy()
                elif hasattr(res, "to_dict"):
                    res_dict = res.to_dict()
                else:
                    import dataclasses
                    res_dict = dataclasses.asdict(res)
                # =========================================================================
                
                res_dict["layer"] = f"layer_{i}"
                results.append(res_dict)
                
                total_latency += res_dict.get("latency", 0)
                total_energy += res_dict.get("energy_total", 0)
                
            except Exception as e:
                # 攔截特殊層 (如 1x1 或非 Conv 層) 並印出錯誤，但不中斷程式
                print(f"⚠️ Layer {i} 發生預期外錯誤，已略過: {e}")

        if not results:
            continue

        # 計算評估指標: EDP (Energy-Delay Product)
        total_edp = total_latency * total_energy
        pe_total = hw.pe_array_h * hw.pe_array_w

        # 紀錄這組硬體的總體表現
        hw_perf_entry = {
            "HW_ID": hw_idx,
            "PE_H": hw.pe_array_h,
            "PE_W": hw.pe_array_w,
            "PE_Total": pe_total,
            "GLB_KB": hw.glb_size // 1024,
            "BW": hw.bus_bw,
            "Total_Latency": total_latency,
            "Total_Energy": total_energy,
            "Total_EDP": total_edp,
            "layers_data": results
        }
        all_hw_performance.append(hw_perf_entry)

        # 找出 EDP 最低的霸主作為最佳硬體
        if total_edp < min_edp:
            min_edp = total_edp
            best_hw_index = hw_idx
            best_layers_results = results
            
        print() # 換行美化日誌

    if not all_hw_performance:
        print("❌ 沒有成功估算任何圖層，無法產生結果。")
        return

    # 3. 排行榜輸出 (依照 EDP 由低到高排序)
    all_hw_performance.sort(key=lambda x: x["Total_EDP"])
    
    print("\n🏆 === YOLOv8n 硬體配置排行榜 (依照 EDP 排序 TOP 5) ===")
    print(f" {'HW_ID':>5}  {'PE_Total':>8}  {'Total_Latency':>13}  {'Total_Energy':>12}  {'Total_EDP':>14}")
    for hw_perf in all_hw_performance[:5]:
        print(f" {hw_perf['HW_ID']:>5}  {hw_perf['PE_Total']:>8}  {hw_perf['Total_Latency']:>13}  {hw_perf['Total_Energy']:12.6f} {hw_perf['Total_EDP']:1.6e}")

    # 4. 儲存最佳硬體的詳細結果
    best_hw_summary = all_hw_performance[0]
    print(f"\n🌟 最佳硬體配置為: PE {best_hw_summary['PE_H']}x{best_hw_summary['PE_W']}, GLB={best_hw_summary['GLB_KB']}KB, bus_bw={best_hw_summary['BW']}bit")
    
    df = pd.DataFrame(best_layers_results)
    csv_path = output_dir / "yolo_output.csv"
    df.to_csv(csv_path, index=False)
    print(f"📊 最佳硬體的層級數據已更新至: {csv_path}")
    roofline_path = output_dir / "yolov8_roofline.png"

    # 5. 繪製 Roofline Model (使用原模板函式)
    try:
        plot_roofline_from_df(df, roofline_path)
        print(f"📈 成功繪製最佳硬體的 Roofline Model 至: {output_dir / 'yolov8_roofline.png'}")
    except Exception as e:
        print(f"❌ 繪製 Roofline 時發生錯誤: {e}")

if __name__ == "__main__":
    main()