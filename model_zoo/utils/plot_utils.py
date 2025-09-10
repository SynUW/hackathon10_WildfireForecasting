import os
import math
from typing import Optional

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_fill_missing_plots(x_before: torch.Tensor, x_after: torch.Tensor, save_path: str,
                            sample_index: int = 0, max_cols: int = 8,
                            color_before: str = '#d62728', color_after: str = '#1f77b4') -> None:
    """
    将填补缺失值前后的序列绘制在同一张图（每个特征一个子图）。

    参数:
      x_before: [B, L, N] 填补前
      x_after:  [B, L, N] 填补后
      save_path: 保存路径（png）
      sample_index: 选择绘制的batch样本索引
      max_cols: 子图网格每行最大列数
      color_before: 前(红)
      color_after:  后(蓝)
    """
    if x_before is None or x_after is None:
        return
    assert x_before.shape == x_after.shape, "x_before 与 x_after 形状必须一致"
    B, L, N = x_before.shape
    if B == 0 or L == 0 or N == 0:
        return

    b = min(max(sample_index, 0), B - 1)
    xb = x_before[b].detach().float().cpu().numpy()  # [L,N]
    xa = x_after[b].detach().float().cpu().numpy()   # [L,N]
    t = np.arange(L)

    cols = min(max_cols, N)
    rows = int(math.ceil(N / cols))
    fig_w = 3.2 * cols
    fig_h = 2.2 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)

    for i in range(N):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        yb = xb[:, i]
        ya = xa[:, i]
        ax.plot(t, yb, color=color_before, linewidth=1.0, alpha=0.9, label='before')
        ax.plot(t, ya, color=color_after, linewidth=1.0, alpha=0.9, label='after')
        ax.set_title(f'Feat {i+1}', fontsize=9)
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
        if r == rows - 1:
            ax.set_xlabel('t')
        if c == 0:
            ax.set_ylabel('value')
        if i == 0:
            ax.legend(fontsize=8)

    # 关闭多余子图
    for j in range(N, rows * cols):
        r = j // cols
        c = j % cols
        axes[r][c].axis('off')

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


