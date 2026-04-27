import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    # 🌟 設定存取與儲存路徑
    csv_path = 'output/yolo_output.csv'
    # 為了區分，我將檔名改為 grouped
    output_path = 'output/yolov8_roofline_grouped.png' 
    
    if not os.path.exists(csv_path):
        print(f"找不到檔案 {csv_path}")
        return

    # 1. 讀取數據
    df = pd.read_csv(csv_path)

    # 🌟 直接使用 CSV 內算好的數值，確保精準
    intensity = df['intensity'].values
    performance = df['performance'].values

    # 🌟 全自動抓取硬體極限 (動態對應最佳硬體)
    PE_MAC_PER_CYCLE = df['peak_performance'].iloc[0]
    BW_PER_CYCLE = df['peak_bandwidth'].iloc[0]

    # 2. 建立單一圖表 (加大figsize以容納所有點)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Roofline 硬體邊界數學計算
    x_line = np.logspace(-2, 3, 100)
    y_roof = np.minimum(np.ones_like(x_line) * PE_MAC_PER_CYCLE, x_line * BW_PER_CYCLE)

    # 3. 繪製屋頂線 ( zorder=1 確保線在點的下方 )
    ax.plot(x_line, y_roof, color='black', linewidth=3, zorder=1, label=f'Roofline Limit ({PE_MAC_PER_CYCLE} PE)')
    
    # 設定座標軸為對數
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # 🌟 修正之前的設定錯誤：放寬視野，確保留白美觀
    ax.set_xlim([10**-1, 10**3])  # 0.1 到 1000
    ax.set_ylim([10**-1, 10**3])  # 0.1 到 1000
    
    # 設定標籤與網格
    ax.set_title('YOLOv8n Roofline Analysis (Grouped Stages)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Operational Intensity (MACs / DRAM Access)', fontsize=13)
    ax.set_ylabel('Performance (MACs / Cycle)', fontsize=13)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    # 4. 定義各模組樣式
    # 分割索引 (Index ranges are from CSV)
    backbone_end = 21
    neck_end = 45

    # 樣式表
    styles = [
        {'range': df.index <= backbone_end, 'label': 'Backbone', 'color': '#1f77b4', 'marker': 'o', 's': 90},
        {'range': (df.index > backbone_end) & (df.index <= neck_end), 'label': 'Neck', 'color': '#ff7f0e', 'marker': 's', 's': 90},
        {'range': df.index > neck_end, 'label': 'Head (Detection)', 'color': '#2ca02c', 'marker': '^', 's': 130} # Head 設大一點
    ]

    # 5. 畫點 ( zorder=2 蓋在線上 )
    for style in styles:
        mask = style['range']
        ax.scatter(
            intensity[mask], 
            performance[mask], 
            color=style['color'], 
            marker=style['marker'], 
            s=style['s'], 
            alpha=0.7, 
            edgecolor='white', 
            zorder=2, 
            label=style['label']
        )

    # 6. 加入圖例 (Legend) 放在右上角
    ax.legend(fontsize=12, loc='upper left', frameon=True, shadow=True)

    plt.tight_layout()
    
    # 儲存圖片
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 成功繪製合併版 Roofline 圖表至: {output_path}")
    plt.close() # 關閉以釋放記憶體

if __name__ == "__main__":
    main()