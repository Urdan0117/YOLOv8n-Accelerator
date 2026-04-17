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
    print(f"🔍 偵測到 {len(hw_candidates)} 組硬體候選配置，開始進行 DSE 探索...")

    all_hw_performance = []
    best_hw_index = -1
    min_latency = float('inf')
    best_layers_results = []

    # 2. 開始 DSE 迴圈：每一組硬體跑一次完整網路
    for hw_idx, hw in enumerate(hw_candidates):
        total_latency = 0
        total_energy = 0
        current_hw_results = []
        
        print(f"\n🏗️ 評估硬體組合 {hw_idx+1}/{len(hw_candidates)}: PE={hw.pe_array_h}x{hw.pe_array_w}, GLB={hw.glb_size//1024}KB")
        
        for i, layer in enumerate(tqdm(layers, desc="Processing Layers", leave=False)):
            mapper = EyerissMapper(name=f"yolo_layer_{i}")
            try:
                # 傳入當前的硬體配置 hw
                mapper.hardware = hw 
            
                solutions = mapper.run(conv2d=layer, num_solutions=1)
                if not solutions:
                    continue
                
                res = solutions[0]
                # 防呆轉換
                if isinstance(res, dict):
                    res_dict = res.copy()
                elif hasattr(res, "to_dict"):
                    res_dict = res.to_dict()
                else:
                    import dataclasses
                    res_dict = dataclasses.asdict(res)
                
                res_dict["layer"] = f"layer_{i}"
                current_hw_results.append(res_dict)
                total_latency += res_dict.get('latency', 0)
                total_energy += res_dict.get('energy_total', 0)
            except Exception as e:
                pass # 略過報錯層

        # 紀錄這組硬體的總表現
        all_hw_performance.append({
            "HW_ID": hw_idx,
            "PE_H": hw.pe_array_h,
            "PE_W": hw.pe_array_w,
            "PE_Total": hw.pe_array_h * hw.pe_array_w,
            "Total_Latency": total_latency,
            "Total_Energy": total_energy,
            "results": current_hw_results
        })

        # 更新最佳硬體
        if total_latency < min_latency and total_latency > 0:
            min_latency = total_latency
            best_hw_index = hw_idx
            best_layers_results = current_hw_results

    # 3. 排序並找出 TOP 5
    df_dse = pd.DataFrame(all_hw_performance)
    top5 = df_dse.sort_values(by="Total_Latency").head(5)

    print("\n🏆 === YOLOv8n 硬體配置排行榜 (TOP 5) ===")
    print(top5[["HW_ID", "PE_H", "PE_W", "PE_Total", "Total_Latency"]].to_string(index=False))
    
    # 儲存 DSE 總表
    df_dse.drop(columns=['results']).to_csv(output_dir / "dse_hardware_results.csv", index=False)

    # 4. 針對 TOP 1 (最佳硬體) 產生 CSV 與 Roofline
    if best_layers_results:
        best_hw = hw_candidates[best_hw_index]
        print(f"\n🌟 最佳硬體配置為: PE {best_hw.pe_array_h}x{best_hw.pe_array_w}")
        
        df_best = pd.DataFrame(best_layers_results)
        csv_path = output_dir / "yolo_output.csv"
        df_best.to_csv(csv_path, index=False)
        print(f"📊 最佳硬體的層級數據已更新至: {csv_path}")

        roofline_path = output_dir / "yolov8_roofline.png"
        try:
            plot_roofline_from_df(df_best, roofline_path)
            print(f"📈 成功繪製最佳硬體的 Roofline Model 至: {roofline_path}")
        except Exception as e:
            print(f"❌ 繪製 Roofline 時發生錯誤: {e}")

if __name__ == "__main__":
    main()

"""
import os
from pathlib import Path
import onnx
import pandas as pd

# 引入 Lab2 的核心工具
from analytical_model import EyerissMapper
from network_parser import parse_onnx
from roofline import plot_roofline_from_df

def main():
    model_path = "yolov8n.onnx"
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 開始解析 ONNX 模型: {model_path} ...")
    
    # 【關鍵點】直接呼叫 ONNX 解析器，避開原本寫死的 VGG 與 torch.load
    try:
        onnx_model = onnx.load(model_path)
        layers = parse_onnx(onnx_model)
        print(f"✅ 成功解析 {len(layers)} 個支援的運算層！")
    except Exception as e:
        print(f"❌ 解析 ONNX 時發生錯誤:\n{e}")
        print("💡 提示：這通常是因為 network_parser.py 還不認識 YOLOv8 的某些特殊節點 (如 Concat, Split, Resize)。")
        return

    print("⚙️ 開始進行硬體架構映射與效能估算 (Eyeriss Mapper)...")
    results = []
    for i, layer in enumerate(layers):
        mapper = EyerissMapper(name=f"yolo_layer_{i}")
        try:
            # 針對每一層執行硬體效能估算
            res = mapper.run(conv2d=layer, num_solutions=1)[0]
            
            # ====== 終極防呆寫法：不管它傳回什麼格式，我們都硬轉成字典 ======
            if isinstance(res, dict):
                res_dict = res.copy()
            elif hasattr(res, "to_dict"):
                res_dict = res.to_dict()
            else:
                import dataclasses
                res_dict = dataclasses.asdict(res)
            # ==========================================================
            
            res_dict["layer"] = f"layer_{i}"
            results.append(res_dict)
        except Exception as e:
            print(f"⚠️ 略過 layer_{i}，因為 Mapper 無法處理: {e}")

    if not results:
        print("❌ 沒有成功估算任何圖層，無法產生 CSV 與 Roofline。")
        return

    # 1. 將結果存成 CSV
    df = pd.DataFrame(results)
    csv_path = output_dir / "yolo_output.csv"
    df.to_csv(csv_path, index=False)
    print(f"📊 成功儲存硬體估算數據至: {csv_path}")

    # 2. 畫出 Roofline Model
    roofline_path = output_dir / "yolov8_roofline.png"
    try:
        plot_roofline_from_df(df, roofline_path)
        print(f"📈 成功繪製 Roofline Model 至: {roofline_path}")
    except Exception as e:
        print(f"❌ 繪製 Roofline 時發生錯誤: {e}")

if __name__ == "__main__":
    main()"""