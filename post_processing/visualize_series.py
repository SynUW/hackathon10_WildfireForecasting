#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
可视化像素级时间序列（GT 正方向，Sigmoid 负方向）

输入目录包含按天输出的 GeoTIFF：
  window{xxx}_{xx}_{yyyymmdd}_{classified|groundtruth|prediction|sigmoid}.tif

功能：
- 解析目录内所有日期，按日期对齐 groundtruth 与 sigmoid 两类栅格
- 对指定像素（行、列）或随机抽样的一组像素，绘制完整时间序列：
  - y 轴正方向为 ground truth 值
  - y 轴负方向为 sigmoid prediction 值（取负号展示）
  - x 轴为时间（按日期排序）
- 忽略无效值 -9999（当作缺失）

用法示例：
  python visualize_series.py \
    --input-dir /path/to/folder \
    --pixels 120,340;200,500 \
    --out plot.png

或随机抽样 9 个像素，生成 3x3 子图：
  python visualize_series.py \
    --input-dir /path/to/folder \
    --random-pixels 9 \
    --grid 3 3 \
    --out plot.png
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

try:
    import rasterio
except ImportError as e:
    print("需要安装 rasterio: pip install rasterio", file=sys.stderr)
    raise


# 文件命名解析：window{xxx}_{xx}_{yyyymmdd}_{type}.tif
FNAME_RE = re.compile(
    r"^window(?P<win>\d{3})_(?P<day>\d{2})_(?P<date>\d{8})_(?P<kind>classified|groundtruth|prediction|sigmoid)\.tif$"
)


@dataclass
class TiffEntry:
    path: str
    date: str  # yyyymmdd
    kind: str  # groundtruth or sigmoid (我们主要使用这两类)


def parse_directory(input_dir: str) -> List[TiffEntry]:
    entries: List[TiffEntry] = []
    for name in os.listdir(input_dir):
        m = FNAME_RE.match(name)
        if not m:
            continue
        kind = m.group('kind')
        # 允许 groundtruth / sigmoid / prediction，忽略 classified
        if kind not in ("groundtruth", "sigmoid", "prediction"):
            continue
        date = m.group('date')
        path = os.path.join(input_dir, name)
        entries.append(TiffEntry(path=path, date=date, kind=kind))
    if not entries:
        raise FileNotFoundError("目录中未找到 groundtruth/sigmoid 的 tiff 文件，或命名不符合约定。")
    return entries


def load_stack(entries: List[TiffEntry]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], List[str]]:
    """
    将 entries 中的 groundtruth / sigmoid / prediction 按日期对齐并堆叠为时间序列。
    返回：gt_stack, sig_stack, pred_stack, dates_sorted
      - gt_stack: [T, H, W]
      - sig_stack: [T, H, W]
      - pred_stack: [T, H, W] 或 None（若不存在 prediction 文件）
      - dates_sorted: 升序日期列表（长度 T）
    缺失（某天缺少某一类）用 NaN 补齐。
    """
    # 收集所有日期
    dates = sorted({e.date for e in entries})
    # 将每种 kind 的文件按日期索引
    by_kind_date: Dict[str, Dict[str, str]] = {"groundtruth": {}, "sigmoid": {}, "prediction": {}}
    for e in entries:
        by_kind_date[e.kind][e.date] = e.path

    # 先读取一个文件确定尺寸
    first_path = entries[0].path
    with rasterio.open(first_path) as src:
        H, W = src.height, src.width

    def read_one(path: Optional[str]) -> np.ndarray:
        if path is None:
            return np.full((H, W), np.nan, dtype=np.float32)
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
        # -9999 视为无效
        arr[arr == -9999] = np.nan
        return arr

    gt_list: List[np.ndarray] = []
    sig_list: List[np.ndarray] = []
    pred_list: List[np.ndarray] = []
    for d in dates:
        gt_list.append(read_one(by_kind_date["groundtruth"].get(d)))
        sig_list.append(read_one(by_kind_date["sigmoid"].get(d)))
        # prediction 可选
        pred_list.append(read_one(by_kind_date["prediction"].get(d)) if by_kind_date["prediction"].get(d) else np.full((H, W), np.nan, dtype=np.float32))

    gt_stack = np.stack(gt_list, axis=0)   # [T,H,W]
    sig_stack = np.stack(sig_list, axis=0) # [T,H,W]
    # 如果 prediction 完全不存在，则返回 None
    if all(np.isnan(x).all() for x in pred_list):
        pred_stack = None
    else:
        pred_stack = np.stack(pred_list, axis=0)
    return gt_stack, sig_stack, pred_stack, dates


def parse_pixels(pixels_arg: Optional[str]) -> List[Tuple[int, int]]:
    if not pixels_arg:
        return []
    pixels: List[Tuple[int, int]] = []
    parts = [p.strip() for p in pixels_arg.split(';') if p.strip()]
    for p in parts:
        try:
            r_str, c_str = p.split(',')
            r = int(r_str)
            c = int(c_str)
            pixels.append((r, c))
        except Exception:
            raise ValueError(f"像素坐标格式错误：{p}，应为 row,col；多个用分号分隔。")
    return pixels


def choose_random_pixels(H: int, W: int, k: int, seed: int = 42) -> List[Tuple[int, int]]:
    rng = np.random.default_rng(seed)
    total = H * W
    k = min(k, total)
    idxs = rng.choice(total, size=k, replace=False)
    pixels: List[Tuple[int, int]] = []
    for idx in idxs:
        r = int(idx // W)
        c = int(idx % W)
        pixels.append((r, c))
    return pixels


def make_grid(n: int, grid_arg: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    if grid_arg is not None:
        return grid_arg
    # 自动布局：近似正方形
    rows = int(np.floor(np.sqrt(n))) or 1
    cols = int(np.ceil(n / rows))
    return rows, cols


def build_balanced_pixel_sets(
    gt_stack: np.ndarray,
    sig_stack: np.ndarray,
    k: int,
    pos_ratio: float = 0.5,
    seed: int = 42,
) -> List[Tuple[int, int]]:
    """
    基于像素全时序的有效性与 GT 定义的正负类别，进行均衡抽样：
      - 有效像素：该像素在所有时间步上，gt 与 sigmoid 都不是 NaN
      - 正样本：GT 在任意时间步 > 0
      - 负样本：GT 在所有时间步 == 0
    返回最多 k 个像素，尽量正负各半。
    """
    T, H, W = gt_stack.shape
    # 有效掩码（全时刻有效）
    valid_all = (~np.isnan(gt_stack) & ~np.isnan(sig_stack)).all(axis=0)  # [H, W]
    # 基于 GT 定义正负
    pos_mask = (gt_stack > 0).any(axis=0) & valid_all
    neg_mask = (gt_stack == 0).all(axis=0) & valid_all

    pos_coords = np.argwhere(pos_mask)
    neg_coords = np.argwhere(neg_mask)

    rng = np.random.default_rng(seed)
    # 目标数量（按比例四舍五入）
    target_pos = int(np.round(k * np.clip(pos_ratio, 0.0, 1.0)))
    target_neg = k - target_pos
    # 按可用上限裁剪
    take_pos = min(target_pos, len(pos_coords))
    take_neg = min(target_neg, len(neg_coords))

    if take_pos == 0 and take_neg == 0:
        # 兜底：退化为随机有效像素
        return choose_random_pixels(H, W, k=min(k, int(valid_all.sum())), seed=seed)

    pos_sel_idx = rng.choice(len(pos_coords), size=take_pos, replace=False) if take_pos > 0 else np.array([], dtype=int)
    neg_sel_idx = rng.choice(len(neg_coords), size=take_neg, replace=False) if take_neg > 0 else np.array([], dtype=int)

    pixels: List[Tuple[int, int]] = []
    for i in pos_sel_idx:
        r, c = map(int, pos_coords[i])
        pixels.append((r, c))
    for i in neg_sel_idx:
        r, c = map(int, neg_coords[i])
        pixels.append((r, c))

    # 如仍有剩余名额，尽量从剩余中补齐（优先使用样本较多的一类）
    remain = max(0, k - len(pixels))
    if remain > 0:
        pos_remaining = [tuple(map(int, pos_coords[i])) for i in range(len(pos_coords)) if i not in set(pos_sel_idx.tolist())]
        neg_remaining = [tuple(map(int, neg_coords[i])) for i in range(len(neg_coords)) if i not in set(neg_sel_idx.tolist())]
        # 先从更多的一类补齐
        buckets = sorted([(pos_remaining, len(pos_remaining)), (neg_remaining, len(neg_remaining))], key=lambda x: -x[1])
        for bucket, size in buckets:
            if remain <= 0:
                break
            if size > 0:
                take = min(remain, size)
                idx = rng.choice(size, size=take, replace=False)
                for j in idx:
                    pixels.append(bucket[int(j)])
                remain -= take

    return pixels


def plot_time_series(
    gt_stack: np.ndarray,
    sig_stack: np.ndarray,
    dates: List[str],
    pixels: List[Tuple[int, int]],
    out_path: str,
    grid: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,
    linewidth: float = 1.2,
    alpha_sigmoid: float = 0.9,
    alpha_gt: float = 0.9,
    pred_stack: Optional[np.ndarray] = None,
    pred_z_clip: float = 3.0,
    focus_window: int = 0,  # 若>0，则围绕GT>0的天数展示前后N天
) -> None:
    T, H, W = gt_stack.shape
    rows, cols = make_grid(len(pixels), grid)

    # 定义一个内部函数：根据GT选择要展示的时间索引（围绕GT>0的日子前后N天）
    def select_time_indices(y_gt_full: np.ndarray) -> np.ndarray:
        if focus_window and focus_window > 0:
            pos_idx = np.where(np.isfinite(y_gt_full) & (y_gt_full > 0))[0]
            if pos_idx.size > 0:
                mask = np.zeros(T, dtype=bool)
                for t in pos_idx.tolist():
                    s = max(0, t - focus_window)
                    e = min(T - 1, t + focus_window)
                    mask[s:e + 1] = True
                return np.where(mask)[0]
            else:
                # 负样本：选择中间的一个窗口以压缩时间尺度
                L = min(T, 2 * focus_window + 1)
                mid = T // 2
                s = max(0, mid - L // 2)
                e = s + L
                return np.arange(s, e)
        # 不聚焦：使用完整序列
        return np.arange(T)

    fig_w = 3.4 * cols
    fig_h = 2.6 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)

    # 统一 y 轴范围（对称），根据所有像素数据估计
    max_abs = 0.0
    for (r, c) in pixels:
        y_gt_full = gt_stack[:, r, c]
        idx = select_time_indices(y_gt_full)
        y_gt = y_gt_full[idx]
        y_sig = (sig_stack[:, r, c] * 100.0)[idx]  # 将 Sigmoid(0..1) 放大到 0..100 再绘制在负轴
        vals = [np.nan_to_num(y_gt, nan=0.0), np.nan_to_num(-y_sig, nan=0.0)]
        # 考虑 prediction z-score（若提供），并进行±pred_z_clip裁剪→映射到[0,1]→再映射到[0,100]
        if pred_stack is not None:
            y_pred = pred_stack[:, r, c][idx]
            # z-score（忽略 NaN）
            mu = np.nanmean(y_pred)
            sigma = np.nanstd(y_pred)
            if np.isfinite(mu) and np.isfinite(sigma) and sigma > 1e-6:
                y_pred_z = (y_pred - mu) / sigma
                # 裁剪到 ±pred_z_clip
                y_pred_z = np.clip(y_pred_z, -pred_z_clip, pred_z_clip)
                # 映射到 [0,1]
                denom = 2.0 * max(pred_z_clip, 1e-6)
                y_pred_01 = (y_pred_z + pred_z_clip) / denom
                # 再映射到 [0,100] 以匹配显示量纲
                y_pred_disp = y_pred_01 * 100.0
                vals.append(np.nan_to_num(-y_pred_disp, nan=0.0))
        y = np.concatenate([
            *vals
        ])
        max_abs = max(max_abs, float(np.nanmax(np.abs(y))))
    # 若未能估计，使用 100 的标准范围（GT 0..100，对称到 负100..100）
    if max_abs <= 0:
        max_abs = 100.0
    # 限制到不小于 100，保证刻度稳定
    max_abs = max(max_abs, 100.0)
    ylim = (-1.05 * max_abs, 1.05 * max_abs)

    for idx_px, (r, c) in enumerate(pixels):
        rr = idx_px // cols
        cc = idx_px % cols
        ax = axes[rr][cc]
        y_gt_full = gt_stack[:, r, c]
        idx = select_time_indices(y_gt_full)
        x = np.arange(len(idx))  # 压缩时间轴
        dates_sel = [dates[i] for i in idx]
        y_gt = y_gt_full[idx]
        y_sig = (sig_stack[:, r, c] * 100.0)[idx]  # 放大到 0..100

        # 画两条线：GT 向上(红)，Sigmoid 向下(蓝，放大到0..100)
        ax.plot(x, y_gt, color="#d62728", linewidth=linewidth, alpha=alpha_gt, label="GT(+)")
        ax.plot(x, -y_sig, color="#1f77b4", linewidth=linewidth, alpha=alpha_sigmoid, label="Sigmoid(-)")
        # 画 prediction z-score→[0,1] 标准化后的负轴线（紫色）
        if pred_stack is not None:
            y_pred = pred_stack[:, r, c][idx]
            mu = np.nanmean(y_pred)
            sigma = np.nanstd(y_pred)
            if np.isfinite(mu) and np.isfinite(sigma) and sigma > 1e-6:
                y_pred_z = (y_pred - mu) / sigma
                y_pred_z = np.clip(y_pred_z, -pred_z_clip, pred_z_clip)
                denom = 2.0 * max(pred_z_clip, 1e-6)
                y_pred_01 = (y_pred_z + pred_z_clip) / denom
                y_pred_disp = y_pred_01 * 100.0
                ax.plot(x, -y_pred_disp, color="#9467bd", linewidth=linewidth, alpha=0.9, label="Pred z01(-)")

        # 在 GT>0 的日期位置画竖直虚线（浅灰）
        try:
            gt_pos = np.where(np.isfinite(y_gt) & (y_gt > 0))[0]
            for px in gt_pos.tolist():
                ax.axvline(px, color="#cfcfcf", linestyle='--', linewidth=0.8, alpha=0.6, zorder=0)
        except Exception:
            pass

        # 文本标注：对于每个 GT>0 的日期，显示 (t-2, t-1, t, t+1, t+2) 的 prediction 归一化值
        # 归一化：z-score→裁剪±pred_z_clip→映射到[0,1]→再映射到[0,100]
        # 使用 OffsetBox 逐token着色，避免叠加重影与mathtext依赖
        if pred_stack is not None:
            y_pred_full = pred_stack[:, r, c]
            # 基于全时序的 z-score（忽略 NaN）
            mu_full = np.nanmean(y_pred_full)
            sigma_full = np.nanstd(y_pred_full)
            if np.isfinite(mu_full) and np.isfinite(sigma_full) and sigma_full > 1e-6:
                y_pred_z_full = (y_pred_full - mu_full) / sigma_full
                y_pred_z_full = np.clip(y_pred_z_full, -pred_z_clip, pred_z_clip)
                denom_full = 2.0 * max(pred_z_clip, 1e-6)
                y_pred_01_full = (y_pred_z_full + pred_z_clip) / denom_full
                y_pred_disp_full = y_pred_01_full * 100.0
                # 查找所有 GT>0 的日期
                y_gt_full = gt_stack[:, r, c]
                pos_days_full = np.where(np.isfinite(y_gt_full) & (y_gt_full > 0))[0]
                if pos_days_full.size > 0:
                    line_boxes: list[AnchoredOffsetbox] = []
                    vlines: list = []
                    vpacker_children: list = []
                    for t0 in pos_days_full.tolist():
                        # 构造一行：日期 + 五个token（-20d, -10d, 0d, +10d, +20d），其中第3个为红色（当天）
                        idxs = [t0 - 20, t0 - 10, t0, t0 + 10, t0 + 20]
                        tokens: list[str] = []
                        for ti in idxs:
                            if 0 <= ti < len(y_pred_disp_full) and np.isfinite(y_pred_disp_full[ti]):
                                tokens.append(f"{float(y_pred_disp_full[ti]):.1f}")
                            else:
                                tokens.append("-")
                        # 逐段 TextArea
                        date_box = TextArea(f"{dates[t0]}: ", textprops=dict(color='black', fontsize=7.5, family='monospace'))
                        toks_boxes = []
                        for j, tstr in enumerate(tokens):
                            color = 'red' if j == 2 else 'black'
                            toks_boxes.append(TextArea(tstr + (" " if j < 4 else ""), textprops=dict(color=color, fontsize=7.5, family='monospace')))
                        hbox = HPacker(children=[date_box] + toks_boxes, pad=0, sep=4, align="baseline")
                        vpacker_children.append(hbox)
                    vbox = VPacker(children=vpacker_children, pad=0, sep=2, align="left")
                    anchored = AnchoredOffsetbox(loc='upper left', child=vbox, frameon=True,
                                                 bbox_to_anchor=(0.02, 0.98), bbox_transform=ax.transAxes, borderpad=0.2)
                    anchored.patch.set_alpha(0.65)
                    anchored.patch.set_facecolor('white')
                    anchored.patch.set_edgecolor('none')
                    ax.add_artist(anchored)

        ax.set_ylim(ylim)
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
        ax.set_title(f"Pixel (r={r}, c={c})", fontsize=10)
        # 稀疏化日期刻度
        # 根据所选时间索引数量设置刻度
        T_sel = len(idx)
        step = max(1, T_sel // 6)
        xticks = list(range(0, T_sel, step))
        ax.set_xticks(xticks)
        ax.set_xticklabels([dates_sel[i] for i in xticks], rotation=30, ha='right', fontsize=8)
        if rr == rows - 1:
            ax.set_xlabel("Date")
        if cc == 0:
            ax.set_ylabel("GT (+0..100) / Sigmoid (-0..100)")
        if idx_px == 0:
            ax.legend(fontsize=8)

    # 关闭多余子图
    total_axes = rows * cols
    for j in range(len(pixels), total_axes):
        rr = j // cols
        cc = j % cols
        axes[rr][cc].axis('off')

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96] if title else None)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="可视化像素级时间序列（GT 正，Sigmoid 负）")
    parser.add_argument('--input-dir', 
                        default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/7to1_focal_withRegressionLoss_withfirms_baseline/s_mamba_org_best_f1/visualizations',
                        help='输入目录，包含按天输出的 tiff 文件')
    parser.add_argument('--pixels', default='', help='像素坐标列表：row,col;row,col;...')
    parser.add_argument('--random-pixels', type=int, default=50, help='随机抽样像素数量（与 --pixels 互斥，>0 生效）')
    parser.add_argument('--grid', type=int, nargs=2, metavar=('ROWS', 'COLS'), help='子图网格(rows cols)，可选')
    parser.add_argument('--out', default='./visualizations', help='输出 PNG 文件路径')
    parser.add_argument('--title', default='', help='图标题，可选')
    parser.add_argument('--seed', type=int, default=42, help='随机像素选择的随机种子')
    parser.add_argument('--pred-z-clip', type=float, default=3.0, help='prediction z-score裁剪阈值(σ)，并映射到±100')
    parser.add_argument('--pos-ratio', type=float, default=0.5, help='随机抽样时正样本比例(0..1)，例如0.8表示80%正样本')
    args = parser.parse_args()

    entries = parse_directory(args.input_dir)
    gt_stack, sig_stack, pred_stack, dates = load_stack(entries)
    T, H, W = gt_stack.shape

    pixels = parse_pixels(args.pixels)
    if not pixels:
        if args.random_pixels and args.random_pixels > 0:
            # 使用按比例的正负抽样（过滤掉任一时刻含无效值的像素）
            pixels = build_balanced_pixel_sets(
                gt_stack,
                sig_stack,
                k=args.random_pixels,
                pos_ratio=args.pos_ratio,
                seed=args.seed,
            )
        else:
            # 默认随机选 9 个像素（不保证均衡，仅兜底）
            pixels = choose_random_pixels(H, W, k=9, seed=args.seed)

    grid = tuple(args.grid) if args.grid is not None else None
    title = args.title if args.title else None

    plot_time_series(
        gt_stack=gt_stack,
        sig_stack=sig_stack,
        dates=dates,
        pixels=pixels,
        out_path=args.out,
        grid=grid,
        title=title,
        pred_stack=pred_stack,
        pred_z_clip=float(max(0.1, args.pred_z_clip)),
    )


if __name__ == '__main__':
    main()

 