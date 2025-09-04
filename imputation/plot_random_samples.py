import os
import argparse
import random
import re
from math import ceil
from typing import List

import h5py
import numpy as np
import matplotlib.pyplot as plt


def parse_indices(spec: str) -> List[int]:
    indices: List[int] = []
    if not spec:
        return indices
    parts = [p.strip() for p in spec.split(',') if p.strip()]
    for p in parts:
        if '-' in p:
            a, b = p.split('-')
            indices.extend(list(range(int(a), int(b) + 1)))
        else:
            indices.append(int(p))
    # de-dup and keep order
    seen = set()
    ordered: List[int] = []
    for x in indices:
        if x not in seen:
            ordered.append(x)
            seen.add(x)
    return ordered


def pick_keys(f: h5py.File, n: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    # 只选择形如 sample 键：包含下划线且数据为二维数组
    keys = []
    for k in f.keys():
        if '_' not in k:
            continue
        obj = f[k]
        if isinstance(obj, h5py.Dataset) and obj.ndim == 2:
            keys.append(k)
    if not keys:
        # 回退：选择所有二维dataset
        for k in f.keys():
            obj = f[k]
            if isinstance(obj, h5py.Dataset) and obj.ndim == 2:
                keys.append(k)
    if len(keys) <= n:
        return keys
    return rng.sample(keys, n)


def sanitize_filename(s: str) -> str:
    s = re.sub(r'[^a-zA-Z0-9_\-]+', '_', s)
    return s[:200]


def parse_row_col_from_key(key: str):
    """从数据集键名解析(row, col)。
    支持："row_col"、"YYYYMMDD_row_col"，或任意包含多个数字的键（取最后两个）。
    失败返回(None, None)。
    """
    m = re.match(r"^(\d+)[_-](\d+)$", key)
    if m:
        return int(m.group(1)), int(m.group(2))
    nums = re.findall(r"\d+", key)
    if len(nums) >= 2:
        try:
            return int(nums[-2]), int(nums[-1])
        except Exception:
            return None, None
    return None, None


def choose_channels(total_c: int, specified: List[int], max_channels: int) -> List[int]:
    if specified:
        valid = [i for i in specified if 0 <= i < total_c]
        return valid[:max_channels] if max_channels > 0 else valid
    # 默认从0开始依次取，最多max_channels个（max_channels<=0表示全取）
    all_idx = list(range(total_c))
    return all_idx[:max_channels] if max_channels > 0 else all_idx


def preprocess_series(ys: np.ndarray, policy: str) -> np.ndarray:
    y = ys.astype(np.float32).copy()
    finite = np.isfinite(y)
    if policy == 'keep':
        return y
    if policy == 'zero':
        y[~finite] = 0.0
        return y
    if policy == 'ffill':
        if not finite.any():
            return y
        y_ff = y.copy()
        # find first valid and back-fill leading
        first_valid = np.argmax(finite)
        if finite[first_valid]:
            if first_valid > 0:
                y_ff[:first_valid] = y[first_valid]
            # forward fill after first_valid
            for i in range(first_valid + 1, y.shape[0]):
                if not np.isfinite(y_ff[i]):
                    y_ff[i] = y_ff[i - 1]
        return y_ff
    # linear interpolation (default)
    if not finite.any():
        return y
    good_x = np.flatnonzero(finite)
    good_y = y[finite]
    xi = np.arange(y.shape[0])
    if good_x.size == 1:
        # only one point -> fill with that constant
        y[~finite] = good_y[0]
        return y
    y[~finite] = np.interp(xi[~finite], good_x, good_y)
    return y


def plot_sample(
    key: str,
    data: np.ndarray,
    out_dir: str,
    channel_indices: List[int],
    layout: str,
    n_cols: int,
    nan_policy: str,
):
    # data shape: [C, T]
    C, T = data.shape
    xs = np.arange(T)
    # 优先用 row_col 命名
    row, col = parse_row_col_from_key(key)
    safe_key = f"{row}_{col}" if (row is not None and col is not None) else sanitize_filename(key)
    out_path = os.path.join(out_dir, f"{safe_key}.png")

    plt.figure(figsize=(16, 9))
    title_id = f"{row}_{col}" if (row is not None and col is not None) else key
    title = f"{title_id}  (C={C}, T={T})"
    if layout == 'overlay':
        # 打印所选通道的数值范围（应用nan_policy后的序列）
        g_min, g_max = None, None
        for idx in channel_indices:
            ys_raw = data[idx].astype(np.float32)
            ys = preprocess_series(ys_raw, nan_policy)
            finite = np.isfinite(ys)
            if finite.any():
                vmin = float(np.min(ys[finite]))
                vmax = float(np.max(ys[finite]))
                g_min = vmin if g_min is None else min(g_min, vmin)
                g_max = vmax if g_max is None else max(g_max, vmax)
                print(f"[{key}] ch{idx:02d} range: [{vmin:.6g}, {vmax:.6g}]  finite={int(finite.sum())}/{ys.size}")
            else:
                print(f"[{key}] ch{idx:02d} range: all-NaN  finite=0/{ys.size}")
            # 若数值范围极小，做视图上的微缩放以便可见
            rng = float(np.nanmax(ys) - np.nanmin(ys)) if ys.size > 0 else 0.0
            if not np.isfinite(rng) or rng == 0.0:
                ys = ys + 1e-6 * (idx + 1)
            plt.plot(xs, ys, label=f"ch{idx:02d}", lw=1.0)
        if g_min is not None and g_max is not None:
            print(f"[{key}] selected-channels global range: [{g_min:.6g}, {g_max:.6g}]")
        plt.title(title)
        plt.xlabel("time index")
        plt.ylabel("value")
        plt.xlim(0, max(0, T - 1))
        if len(channel_indices) <= 20:
            plt.legend(ncol=2, fontsize=8)
        else:
            plt.legend([], [], frameon=False)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return out_path

    # subplots layout
    n = len(channel_indices)
    if n_cols <= 0:
        n_cols = 3
    n_rows = ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, max(6, 3 * n_rows)), squeeze=False)
    g_min, g_max = None, None
    for i, ch in enumerate(channel_indices):
        r = i // n_cols
        c = i % n_cols
        ax = axes[r][c]
        ys_raw = data[ch].astype(np.float32)
        ys = preprocess_series(ys_raw, nan_policy)
        finite = np.isfinite(ys)
        if finite.any():
            vmin = float(np.min(ys[finite]))
            vmax = float(np.max(ys[finite]))
            g_min = vmin if g_min is None else min(g_min, vmin)
            g_max = vmax if g_max is None else max(g_max, vmax)
            print(f"[{key}] ch{ch:02d} range: [{vmin:.6g}, {vmax:.6g}]  finite={int(finite.sum())}/{ys.size}")
        else:
            print(f"[{key}] ch{ch:02d} range: all-NaN  finite=0/{ys.size}")
        rng = float(np.nanmax(ys) - np.nanmin(ys)) if ys.size > 0 else 0.0
        if not np.isfinite(rng) or rng == 0.0:
            ys = ys + 1e-6
        ax.plot(xs, ys, lw=1.0)
        ax.set_title(f"ch{ch:02d}")
        ax.grid(True, ls='--', alpha=0.3)
        ax.set_xlim(0, max(0, T - 1))
    if g_min is not None and g_max is not None:
        print(f"[{key}] selected-channels global range: [{g_min:.6g}, {g_max:.6g}]")
    # hide empty axes
    for i in range(n, n_rows * n_cols):
        r = i // n_cols
        c = i % n_cols
        axes[r][c].axis('off')
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="从imputed H5随机选n个样本，绘制驱动因素折线图")
    parser.add_argument("--h5", type=str, default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/full_datasets_imputed/2005_year_dataset.h5", help="imputed H5文件路径（数据形状一般为[39, T]）")
    parser.add_argument("--n", type=int, default=10, help="随机选择的样本数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="./imputed_plots")
    parser.add_argument("--channel-indices", type=str, default="", help="要绘制的通道索引，如: 0,1,2,5-8。不填则默认按顺序选择")
    parser.add_argument("--max-channels", type=int, default=39, help="最多绘制的通道数；<=0表示绘制全部")
    parser.add_argument("--layout", type=str, default="subplots", choices=["overlay", "subplots"], help="叠加或子图布局")
    parser.add_argument("--n-cols", type=int, default=3, help="子图布局的列数，仅当layout=subplots有效")
    parser.add_argument("--nan-policy", type=str, default="keep", choices=["keep", "zero", "ffill", "linear"], help="NaN处理策略：保留/置零/前向填充/线性插值")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with h5py.File(args.h5, 'r') as f:
        keys = pick_keys(f, args.n, args.seed)
        if not keys:
            print("[WARN] 未在H5中找到合适的数据集键。")
            return
        print(f"将绘制 {len(keys)} 个样本：")
        for k in keys:
            print(f"  - {k}")

        idx_spec = parse_indices(args.channel_indices)
        for k in keys:
            arr = f[k][()]
            if arr.ndim != 2:
                continue
            # 统一为 [C, T]
            C, T = arr.shape
            # 更稳健的转置判断：若时间步数远大于通道数，确保时间维在轴1
            if T < C:
                arr = arr.T
                C, T = arr.shape
            channels = choose_channels(C, idx_spec, args.max_channels)
            out_path = plot_sample(k, arr, args.out_dir, channels, args.layout, args.n_cols, args.nan_policy)
            print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

