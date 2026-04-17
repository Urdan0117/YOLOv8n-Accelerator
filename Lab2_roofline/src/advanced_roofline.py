import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    csv_path = 'output/yolo_output.csv'
    
    if not os.path.exists(csv_path):
        print(f"找不到檔案 {csv_path}")
        return

    # 1. 讀取與計算數據
    df = pd.read_csv(csv_path)
    cols = df.columns.str.lower()
    mac_col = df.columns[cols.str.contains('mac')][0]
    dram_col = df.columns[cols.str.contains('dram')][0]
    cycles_col = df.columns[cols.str.contains('cycle|latency')][0]

    intensity = df[mac_col].values / df[dram_col].values
    performance = df[mac_col].values / df[cycles_col].values

    # 2. 建立 1x3 的並排子圖 (設定共用 X 軸與 Y 軸方便比較)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, sharex=True)
    
    # Roofline 硬體邊界設定
    PE_MAC_PER_CYCLE = 192  
    BW_PER_CYCLE = 8        
    x_line = np.logspace(-2, 3, 100)
    y_roof = np.minimum(np.ones_like(x_line) * PE_MAC_PER_CYCLE, x_line * BW_PER_CYCLE)

    # 模組設定
    titles = ['Backbone (Feature Extraction)', 'Neck (C2f / FPN)', 'Head (Detection)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']

    # 3. 初始化三張圖的背景與屋頂線
    for ax, title in zip(axes, titles):
        ax.plot(x_line, y_roof, color='black', linewidth=2, zorder=1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # === 新增這行：強制把 X 軸的視野鎖定在 10^-2 到 10^2 ===
        ax.set_xlim([10**0, 200])  
        ax.set_ylim([10**0, 200])  
        # ====================================================
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Operational Intensity (MACs / DRAM Accesses)', fontsize=11)
        ax.grid(True, which="both", ls="--", alpha=0.3)
    # 4. 依照 Layer Index 將點畫入對應的子圖
    for i in range(len(df)):
        x = intensity[i]
        y = performance[i]
        
        if i <= 21:    # Backbone
            axes[0].scatter(x, y, color=colors[0], marker=markers[0], s=100, alpha=0.7, edgecolor='white', zorder=2)
        elif i <= 45:  # Neck
            axes[1].scatter(x, y, color=colors[1], marker=markers[1], s=100, alpha=0.7, edgecolor='white', zorder=2)
        else:          # Head
            axes[2].scatter(x, y, color=colors[2], marker=markers[2], s=100, alpha=0.7, edgecolor='white', zorder=2)

    plt.tight_layout()
    
    # 儲存圖片
    output_path = 'output/yolov8_roofline_subplots.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 成功繪製拆分版 Roofline 圖表至: {output_path}")

if __name__ == "__main__":
    main()