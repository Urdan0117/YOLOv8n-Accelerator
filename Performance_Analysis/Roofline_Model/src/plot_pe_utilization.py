import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    csv_path = 'output/yolo_output_custom.csv'
    output_path = 'output/pe_utilization_custom.png'
    
    if not os.path.exists(csv_path):
        print(f"❌ 找不到檔案 {csv_path}")
        return

    # 1. 讀取數據
    df = pd.read_csv(csv_path)

    # 2. 計算 PE 使用率 (Utilization)
    # 確保不會除以 0 的安全寫法
    peak_perf = df['peak_performance'].values
    actual_perf = df['performance'].values
    """ -> 怎麼計算 PE 使用率？
    performance (實際效能)：這層網路平均每個 Cycle 實際完成了幾個 MACs 運算。

    peak_performance (峰值算力)：你的 16x16 陣列理論上每個 Cycle 最多能做 256 個 MACs 運算。

    公式： PE 使用率 (%) = (performance / peak_performance) * 100
    """
    utilization = (actual_perf / peak_perf) * 100

    # 3. 準備畫圖
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 設定 X 軸 (Layer 索引)
    x_indices = range(len(df))
    
    # 4. 依照不同模組上色
    # Backbone (0~21), Neck (22~45), Head (46~)
    colors = []
    for i in x_indices:
        if i <= 21:
            colors.append('#1f77b4')  # 藍色 Backbone
        elif i <= 45:
            colors.append('#ff7f0e')  # 橘色 Neck
        else:
            colors.append('#2ca02c')  # 綠色 Head

    # 畫出長條圖
    bars = ax.bar(x_indices, utilization, color=colors, edgecolor='black', linewidth=0.5)

    # 5. 畫一條 100% 的完美極限虛線當作基準
    ax.axhline(y=100, color='red', linestyle='--', linewidth=2, label='100% Theoretical Limit')

    # 6. 圖表美化
    ax.set_title('YOLOv8n PE Utilization per Layer (16x16 PE Array)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Layer Index', fontsize=13)
    ax.set_ylabel('PE Utilization (%)', fontsize=13)
    ax.set_ylim(0, 110) # 留一點頂部空間
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 加上自訂圖例 (Legend)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', edgecolor='black', label='Backbone'),
        Patch(facecolor='#ff7f0e', edgecolor='black', label='Neck'),
        Patch(facecolor='#2ca02c', edgecolor='black', label='Head'),
        plt.Line2D([0], [0], color='red', linestyle='--', lw=2, label='100% Peak')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

    # 7. 儲存圖片
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✅ 成功繪製 PE 使用率圖表至: {output_path}")

if __name__ == "__main__":
    main()