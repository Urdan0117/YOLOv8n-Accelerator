import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    csv_path = 'output/yolo_output.csv'
    
    if not os.path.exists(csv_path):
        print(f"找不到檔案 {csv_path}")
        return

    # 1. 讀取數據
    df = pd.read_csv(csv_path)

    # 🌟 修正點 1：直接使用 CSV 內官方算好的數值，最精準！
    intensity = df['intensity'].values
    performance = df['performance'].values

    # 🌟 修正點 2：自動從 CSV 抓取當前最佳硬體的極限，不再寫死！
    # 這樣以後你不管換成 192, 256 還是 512 PE，圖表都會自動對應
    PE_MAC_PER_CYCLE = df['peak_performance'].iloc[0]
    BW_PER_CYCLE = df['peak_bandwidth'].iloc[0]

    # 2. 建立 1x3 的並排子圖
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, sharex=True)
    
    # Roofline 硬體邊界數學計算
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
        
        # 🌟 修正點 3：放寬視野範圍，確保屋頂(256)與極端點都能被看見
        ax.set_xlim([5*10**-1, 10**3])  # X軸從 0.1 到 1000
        ax.set_ylim([5*10**-1, 10**3])  # Y軸從 0.1 到 1000
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Operational Intensity (MACs / DRAM Access)', fontsize=11)
        if ax == axes[0]: # 只在最左邊的圖顯示 Y 軸標籤
            ax.set_ylabel('Performance (MACs / Cycle)', fontsize=11)
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
    print(f"✅ 成功繪製全自動修正版 Roofline 圖表至: {output_path}")

if __name__ == "__main__":
    main()