"""
基于SAITS的模型，对指定年份的缺失值进行插补，并保存到新的H5文件中
"""
import os
import sys
import argparse
import re
import numpy as np
import h5py
import torch

# Ensure project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from dataload_year import TimeSeriesDataLoader
from imputation.saits import SAITS


def build_model(d_time: int, d_feature: int, device: torch.device) -> SAITS:
    model = SAITS(
        n_groups=1,
        n_group_inner_layers=1,
        d_time=d_time,
        d_feature=d_feature,
        d_model=d_feature,
        d_inner=d_feature,
        n_head=1,
        d_k=d_feature,
        d_v=d_feature,
        dropout=0.1,
        input_with_mask=True,
        param_sharing_strategy="between_group",
        MIT=True,
        device=str(device),
        diagonal_attention_mask=True,
    )
    return model.to(device)


def parse_years(s: str):
    years = []
    for part in s.split(','):
        if '-' in part:
            a, b = part.split('-')
            years.extend(list(range(int(a), int(b) + 1)))
        else:
            years.append(int(part))
    return years


def _parse_dataset_name_to_row_col(name: str):
    """解析数据集名称为 (row, col)。
    兼容以下常见样式：
      - "row_col"
      - "YYYYMMDD_row_col"
      - 含多个数字的名称，取最后两个数字作为 row、col
    失败则返回 (None, None)。
    """
    # 直接匹配 row_col 或 row-col
    m = re.match(r"^(\d+)[_-](\d+)$", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    # 匹配 ..._row_col
    nums = re.findall(r"\d+", name)
    if len(nums) >= 2:
        try:
            return int(nums[-2]), int(nums[-1])
        except Exception:
            return None, None
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Impute missing values and save to a new H5 file")
    parser.add_argument("--h5_dir", type=str, default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/full_datasets",  # year_datasets_h5_masked_10x
                        help="原始按年H5数据根目录。若提供 --in_h5 将忽略此参数")
    parser.add_argument("--in_h5", type=str, default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/full_datasets/cache/samples_5563b8eced09.h5",
                        help="可选：直接基于已缓存的cache.h5进行插补（键形如 YYYYMMDD_row_col，数据形状[39,L]）")
    parser.add_argument("--years", type=str, default="2000-2024")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--ckpt", type=str, default="./runs_imputation/saits_best-39.pt")
    parser.add_argument("--out_h5", type=str, default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/full_datasets/cache/imputed_cache_real.h5")
    parser.add_argument("--seed", type=int, default=42)
    # New: directly impute per-year H5 files and save to a mirrored folder
    parser.add_argument("--impute-year-files", action="store_true",
                        help="对原始按年H5文件逐像素插补并保存为新目录下的同名H5文件")
    parser.add_argument("--out_dir", type=str, default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/full_datasets_imputed",
                        help="保存插补后按年H5文件的目录（与原文件同名），启用 --impute-year-files 时必填")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    from torch.utils.data import Dataset, Subset, DataLoader as TorchDataLoader

    def is_imputed_cache_h5(h5_path: str) -> bool:
        """检测是否为逐样本数据式cache（每个键是一个样本，形如 YYYYMMDD_row_col，数据为[39,L]）。
        若是样本索引cache（如 samples_*.h5），则返回False。"""
        try:
            with h5py.File(h5_path, 'r') as f:
                # 快速规则：存在大量形如包含下划线的dataset键，并且任取一个dataset是2D数组
                keys = list(f.keys())
                if len(keys) == 0:
                    return False
                # 如果是样本索引cache，常见顶层键少且非样本名；这里取前若干尝试
                sample_like = 0
                for k in keys[:50]:
                    if not isinstance(k, str):
                        continue
                    if '_' in k and k.split('_')[0].isdigit():
                        obj = f[k]
                        if isinstance(obj, h5py.Dataset) and obj.ndim == 2:
                            sample_like += 1
                return sample_like > 0
        except Exception:
            return False

    use_in_h5 = bool(args.in_h5) and os.path.isfile(args.in_h5) and is_imputed_cache_h5(args.in_h5)

    # ===== Path 1: Impute whole per-year H5 files and save to new folder =====
    if args.impute_year_files:
        assert args.out_dir, "--out_dir 必须提供，用于保存插补后的H5文件"
        os.makedirs(args.out_dir, exist_ok=True)

        # Discover year files
        year_files = []
        for fn in os.listdir(args.h5_dir):
            if fn.endswith('_year_dataset.h5'):
                try:
                    y = int(fn.split('_')[0])
                    year_files.append((y, os.path.join(args.h5_dir, fn)))
                except Exception:
                    continue
        year_files.sort(key=lambda x: x[0])

        # Filter years
        years_keep = set(parse_years(args.years))
        year_files = [(y, p) for (y, p) in year_files if (not years_keep) or (y in years_keep)]
        if not year_files:
            print(f"[WARN] 未找到年份文件: {args.h5_dir}, years={args.years}")

        # Build models cache by time length
        model_cache = {}

        # Load checkpoint once (state_dict)
        state = None
        if os.path.isfile(args.ckpt):
            ckpt = torch.load(args.ckpt, map_location=device)
            state = ckpt.get("model_state_dict", ckpt)
            print(f"Loaded checkpoint: {args.ckpt}")
        else:
            print(f"[WARN] Checkpoint not found: {args.ckpt}. Proceeding with randomly initialized model.")

        def get_model_for_T(T: int, C: int = 39) -> SAITS:
            # 始终使用与checkpoint一致的时间长度（365）来构建模型，避免权重shape不匹配
            T_model = 365 if T >= 365 else T
            if T_model not in model_cache:
                m = build_model(d_time=T_model, d_feature=C, device=device)
                if state is not None:
                    # 以strict=True加载，保证与365配置完全一致
                    m.load_state_dict(state)
                m.eval()
                model_cache[T_model] = m
            return model_cache[T_model]

        # Process each year file
        for y, src_path in year_files:
            with h5py.File(src_path, 'r') as fin:
                # Infer time length and channels
                T = int(fin.attrs.get('total_time_steps', 365))
                C = int(fin.attrs.get('total_channels', 39))
                model = get_model_for_T(T, C)

                # Prepare output file
                dst_path = os.path.join(args.out_dir, os.path.basename(src_path))
                with h5py.File(dst_path, 'w') as fout:
                    # Copy file-level attrs
                    for k, v in fin.attrs.items():
                        fout.attrs[k] = v
                    fout.attrs['imputed_from'] = src_path
                    fout.attrs['impute_ckpt'] = args.ckpt

                    # Collect dataset names (pixels)
                    pixel_keys = []
                    for k in fin.keys():
                        if '_' not in k:
                            continue
                        obj = fin[k]
                        if isinstance(obj, h5py.Dataset) and obj.ndim == 2:
                            pixel_keys.append(k)
                    # Batch process for speed
                    batch = []
                    meta = []
                    def flush_batch():
                        if not batch:
                            return
                        X = np.stack(batch, axis=0)  # [B, C, T]
                        X = torch.from_numpy(X).to(device=device, dtype=torch.float32)
                        X_orig = X.permute(0, 2, 1).contiguous()  # [B, T, C]
                        # 观测掩码：1 表示原始有效值；排除 NaN/Inf/255/-9999，并按通道处理0值有效性
                        observed_base = (
                            torch.isfinite(X_orig)
                            & (X_orig != 255)
                            & (X_orig != -9999)
                        )
                        # 通道0..12的0视为有效，其余通道的0视为无效
                        ch_idx = torch.arange(X_orig.shape[2], device=X_orig.device)
                        zero_ok = (ch_idx <= 12).view(1, 1, -1)
                        nonzero = (X_orig >= 1e-6)
                        observed = (observed_base & (zero_ok | nonzero)).to(torch.float32)
                        # 再次与isfinite对齐，杜绝观测位非有限
                        observed = (observed * torch.isfinite(X_orig).to(torch.float32)).clamp(0.0, 1.0)
                        # 用where替换缺失值为0，避免 NaN*0 仍为 NaN
                        X_masked = torch.where(observed > 0.5, X_orig, torch.zeros_like(X_orig))
                        # 额外兜底：将任何残余的NaN/Inf转为0
                        X_masked = torch.nan_to_num(X_masked, nan=0.0, posinf=0.0, neginf=0.0)
                        if T == 366:
                            # 使用365天模型插补前365天
                            X_orig_365 = X_masked[:, :365, :]
                            observed_365 = observed[:, :365, :]
                            inputs = {
                                "X": X_orig_365,
                                "missing_mask": observed_365,
                                "X_holdout": X_orig[:, :365, :],
                                "indicating_mask": torch.zeros_like(X_orig_365),
                            }
                            with torch.no_grad():
                                out = model(inputs, stage="test")
                                X_c_365 = out["imputed_data"]  # [B, 365, C]
                                X_c_365 = torch.nan_to_num(X_c_365, nan=0.0, posinf=0.0, neginf=0.0)
                                # 严格替换缺失位置为模型输出
                                X_filled_365 = torch.where(observed_365 > 0.5, X_orig_365, X_c_365)
                            # 处理第366天：若无效则用第365天的插值（直接拷贝前一天的填充值），否则保留原值
                            obs366 = observed[:, 365, :]  # [B, C]
                            x366_orig = X_orig[:, 365, :]  # [B, C]
                            x365_final = X_filled_365[:, 364, :]  # [B, C]
                            x366_final = torch.where(obs366 > 0.5, x366_orig, x365_final)
                            X_filled = torch.cat([X_filled_365, x366_final.unsqueeze(1)], dim=1)  # [B, 366, C]
                        else:
                            inputs = {
                                "X": X_masked,
                                "missing_mask": observed,
                                "X_holdout": X_orig,
                                "indicating_mask": torch.zeros_like(X_orig),
                            }
                            with torch.no_grad():
                                out = model(inputs, stage="test")
                                X_c = out["imputed_data"]  # [B, T, C]
                                X_c = torch.nan_to_num(X_c, nan=0.0, posinf=0.0, neginf=0.0)
                                # 严格替换缺失位置为模型输出
                                X_filled = torch.where(observed > 0.5, X_orig, X_c)

                        X_filled = torch.nan_to_num(X_filled, nan=0.0, posinf=0.0, neginf=0.0)
                        X_filled = X_filled.permute(0, 2, 1).contiguous().detach().cpu().numpy()  # [B, C, T]
                        # Write each dataset；名称采用"row_col"，解析失败则回退原名
                        for i, rc in enumerate(meta):
                            row, col = rc
                            ds_name = None
                            if row is not None and col is not None:
                                ds_name = f"{row}_{col}"
                            else:
                                ds_name = pixel_keys[i] if i < len(pixel_keys) else f"pix_{i}"
                            if ds_name in fout:
                                del fout[ds_name]
                            fout.create_dataset(ds_name, data=X_filled[i], compression="gzip", compression_opts=4, shuffle=True)
                        batch.clear()
                        meta.clear()

                    for name in pixel_keys:
                        arr = fin[name][()]
                        # Ensure shape [C,T]
                        if arr.ndim != 2:
                            continue
                        if arr.shape[0] != C:
                            # transpose if stored [T,C]
                            if arr.shape[1] == C:
                                arr = arr.T
                            else:
                                continue
                        # Append to batch；meta存(row,col)
                        batch.append(arr.astype(np.float32))
                        meta.append(_parse_dataset_name_to_row_col(name))
                        if len(batch) >= args.batch_size:
                            flush_batch()
                    flush_batch()

            print(f"✅ Year {y} imputed -> {dst_path}")

        print("🎉 All year files imputed and saved.")
        return

    # ===== Path 2/3: Original sampler-based or cache-based per-sample imputation =====
    if not use_in_h5:
        # 基于原始年度数据构建加载器
        ts_loader = TimeSeriesDataLoader(
            h5_dir=args.h5_dir,
            positive_ratio=1.0,
            pos_neg_ratio=999999,
            resample_each_epoch=False,
            lookback_seq=365,
            forecast_hor=7,
            verbose_sampling=False,
            enable_performance_optimizations=True,
        )

        # 目前内部数据集已按年份加载；此处直接取全部索引
        indices = list(range(len(ts_loader.dataset)))
        subset = Subset(ts_loader.dataset, indices)
        loader = TorchDataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            collate_fn=ts_loader.dataset.custom_collate_fn,
        )
    else:
        # 基于已有的cache.h5构建数据集（键: YYYYMMDD_row_col，数据: [39, L]）
        if bool(args.in_h5) and os.path.isfile(args.in_h5) and not use_in_h5:
            print(f"[INFO] Detected non-imputed cache file '{args.in_h5}'. Falling back to original loader via --h5_dir.\n"
                  f"       提示：--in_h5 仅支持逐样本数据式cache（每个样本为一个dataset，形如 YYYYMMDD_row_col）。")
            # 回退到原始数据加载路径
            ts_loader = TimeSeriesDataLoader(
                h5_dir=args.h5_dir,
                positive_ratio=1.0,
                pos_neg_ratio=999999,
                resample_each_epoch=False,
                lookback_seq=365,
                forecast_hor=7,
                verbose_sampling=False,
                enable_performance_optimizations=True,
            )
            indices = list(range(len(ts_loader.dataset)))
            subset = Subset(ts_loader.dataset, indices)
            loader = TorchDataLoader(
                subset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=4,
                collate_fn=ts_loader.dataset.custom_collate_fn,
            )
        else:
            years_set = set(parse_years(args.years))

        class H5ImputeDataset(Dataset):
            def __init__(self, h5_path: str, years_keep: set[int]):
                self.h5_path = h5_path
                with h5py.File(self.h5_path, 'r') as f:
                    self.keys = [k for k in f.keys() if isinstance(f[k], h5py.Dataset) and f[k].ndim == 2]

            def __len__(self):
                return len(self.keys)

            def __getitem__(self, idx):
                key = self.keys[idx]
                with h5py.File(self.h5_path, 'r') as f:
                    arr = f[key][()]
                if arr.ndim != 2:
                    raise RuntimeError("dataset should be 2-D")
                # 转为 [C, L] 的float32张量
                if arr.shape[0] <= arr.shape[1]:
                    arr2 = arr
                else:
                    arr2 = arr.T
                x = torch.from_numpy(arr2).to(torch.float32)
                # 解析(row,col)
                row, col = _parse_dataset_name_to_row_col(key)
                meta = (row, col)
                return x, None, meta

        def h5_collate(batch):
            # batch: List[Tuple[x:[C,L] tensor, None, meta_tuple]] -> ([B,C,L], None, List[meta])
            xs = []
            metas = []
            for x, _, meta in batch:
                xs.append(x)
                metas.append(meta)
            xs = torch.stack(xs, dim=0)
            return xs, None, metas

            ds = H5ImputeDataset(args.in_h5, years_set)
            loader = TorchDataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=(args.num_workers > 0),
                prefetch_factor=(4 if args.num_workers > 0 else None),
                collate_fn=h5_collate,
            )

    # Peek batch to infer shapes
    first_batch = next(iter(loader))
    past_batch, _, metadata_list = first_batch
    B0, C0, L0 = past_batch.shape  # [B, 39, L]
    # 始终使用365长度模型（若L0>=365），避免与checkpoint不一致
    d_time_model = 365 if L0 >= 365 else L0
    model = build_model(d_time=d_time_model, d_feature=C0, device=device)
    # Load checkpoint
    if not os.path.isfile(args.ckpt):
        print(f"[WARN] Checkpoint not found: {args.ckpt}. Proceeding with randomly initialized model.")
    else:
        ckpt = torch.load(args.ckpt, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {args.ckpt}")

    # Prepare output H5
    os.makedirs(os.path.dirname(args.out_h5), exist_ok=True)
    h5f = h5py.File(args.out_h5, 'w')
    h5f.attrs['source_h5_dir'] = (args.in_h5 if use_in_h5 else args.h5_dir)
    h5f.attrs['years'] = args.years
    h5f.attrs['lookback_seq'] = 365
    h5f.attrs['note'] = 'Per-sample imputed past windows; dataset key = f"{date_int}_{row}_{col}"; data shape [39, L]'

    with torch.no_grad():
        sample_count = 0
        for batch in loader:
            # 兼容两种数据源的batch结构
            if use_in_h5:
                past_batch, _, metadata_list = batch  # 默认collate：past_batch:[B,C,L], metadata: list[tuple]
            else:
                past_batch, _, metadata_list = batch

            X_orig = past_batch.to(device, non_blocking=True).permute(0, 2, 1).contiguous()  # [B,L,N]
            # print(torch.isfinite(X_orig))
            # print(torch.max(X_orig), torch.min(X_orig))
            # Build masks with zero handling: ch0..12: 0 is valid; others: 0 is invalid
            observed_base = (
                torch.isfinite(X_orig)
                & (X_orig != 255)
                & (X_orig != -9999)
            )
            ch_idx = torch.arange(X_orig.shape[2], device=X_orig.device)
            zero_ok = (ch_idx <= 12).view(1, 1, -1)
            nonzero = (X_orig >= 1e-6)
            observed = (observed_base & (zero_ok | nonzero)).to(torch.float32)
            observed = (observed * torch.isfinite(X_orig).to(torch.float32)).clamp(0.0, 1.0)
            # Mask original missing positions to zero before feeding model
            X_masked = torch.where(observed > 0.5, X_orig, torch.zeros_like(X_orig))
            X_masked = torch.nan_to_num(X_masked, nan=0.0, posinf=0.0, neginf=0.0)
            missing_mask = observed  # no artificial mask here
            indicating = torch.zeros_like(X_orig)
            inputs = {
                "X": X_masked,
                "missing_mask": missing_mask,
                "X_holdout": X_orig,
                "indicating_mask": indicating,
            }
            out = model(inputs, stage="test")
            X_c = out["imputed_data"]  # [B, L_used, N]
            X_c = torch.nan_to_num(X_c, nan=0.0, posinf=0.0, neginf=0.0)
            if L0 == 366 and d_time_model == 365:
                # 先对前365天进行插补
                X_orig_365 = X_orig[:, :365, :]
                miss_365 = missing_mask[:, :365, :]
                X_c_365 = X_c[:, :365, :]
                X_c_365 = torch.nan_to_num(X_c_365, nan=0.0, posinf=0.0, neginf=0.0)
                X_filled_365 = torch.where(miss_365 > 0.5, X_orig_365, X_c_365)
                # 第366天：缺失则用第365天的插补值，否则保留原值
                obs366 = missing_mask[:, 365, :]
                x366_orig = X_orig[:, 365, :]
                x365_final = X_filled_365[:, 364, :]
                x366_final = torch.where(obs366 > 0.5, x366_orig, x365_final)
                X_filled = torch.cat([X_filled_365, x366_final.unsqueeze(1)], dim=1)
            else:
                # Replace only missing positions (excluding chan0 zeros)
                X_filled = torch.where(missing_mask > 0.5, X_orig, X_c)
            X_filled = torch.nan_to_num(X_filled, nan=0.0, posinf=0.0, neginf=0.0)
            # Save each sample，输出名称采用"row_col"
            for b in range(X_filled.shape[0]):
                row, col = metadata_list[b]
                if row is None or col is None:
                    key = f"sample_{b}"
                else:
                    key = f"{int(row)}_{int(col)}"
                data_np = X_filled[b].detach().cpu().numpy().T  # [N,L] -> save as [C,L]
                if key in h5f:
                    del h5f[key]
                h5f.create_dataset(key, data=data_np, compression="gzip")
                sample_count += 1
            if sample_count % 1000 == 0:
                print(f"Saved {sample_count} samples...")

    h5f.close()
    print(f"Done. Saved {sample_count} imputed samples to {args.out_h5}")


if __name__ == "__main__":
    main()

