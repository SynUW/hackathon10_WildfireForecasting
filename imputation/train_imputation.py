import argparse
import os
import time
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Local imports
from saits import SAITS
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)
from dataload_year import TimeSeriesDataLoader


# === Utilities to derive deterministic seeds from metadata ===
def _metadata_to_seed(meta_item: List[int], base_seed: int) -> int:
    # meta: [date_int, row, col]
    date_int, row, col = int(meta_item[0]), int(meta_item[1]), int(meta_item[2])
    # simple hash composition (avoid Python hash randomness)
    return int((base_seed * 1315423911 + date_int * 2654435761 + row * 97531 + col * 31337) & 0x7FFFFFFF)


def sample_artificial_mask_per_batch(
    X_orig: torch.Tensor,
    k_per_feature: int = 30,
    mode: str = "train",
    base_seed: int = 42,
    metadata_list: Optional[List[List[int]]] = None,
    start_feature_one_based: int = 22,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate masks per batch following the user's spec.

    Inputs:
      - X_orig: [B, L, N] original data; invalid values are marked by -9999 or 255 (not 0)
      - k_per_feature: number of artificial masks per feature (default 30)
      - mode: 'train' | 'val' | 'test' (train: resample every call; val/test: deterministic by sample index)
      - base_seed: global seed for deterministic mask in val/test
      - sample_indices: [B] global indices of samples (required for val/test)

    Outputs:
      - X_masked: [B, L, N] original missing and artificial masked positions are zeroed
      - mask_missing_only: [B, L, N] binary: 1 for original observed; 0 for original missing
      - mask_missing_plus_artificial: [B, L, N] binary: 1 observed; 0 for original-missing and artificial-masked
      - indicating_mask: [B, L, N] binary: 1 at artificial masked positions; 0 elsewhere
    """
    B, L, N = X_orig.shape
    device = X_orig.device

    # Only allow artificial masking on features from `start_feature_one_based` to the last (1-based indexing)
    # Clamp to valid range in case N < start_feature_one_based
    start_idx = max(0, min(N, int(start_feature_one_based) - 1))  # convert to 0-based
    # feature gate mask with shape [1, 1, N]
    feature_gate = torch.zeros(1, 1, N, device=device, dtype=X_orig.dtype)
    if start_idx < N:
        feature_gate[:, :, start_idx:] = 1.0

    # Original observed mask (binary): 1 observed, 0 missing
    # Now: invalid values are -9999 or 255 across all channels (including FIRMS)
    observed_mask = ((X_orig != -9999) & (X_orig != 255)).to(torch.float32)

    # Binary masks to build
    mask_missing_only = observed_mask.clone()
    mask_missing_plus_artificial = observed_mask.clone()
    indicating_mask = torch.zeros_like(X_orig)

    if mode == "train":
        # Vectorized random selection using Top-K over L with strong penalty on invalid positions
        obs_counts = observed_mask.sum(dim=1)  # [B, N]
        # random scores in [0,1), subtract large penalty where not observed so they won't be picked
        scores = torch.rand(B, L, N, device=device) - (1.0 - observed_mask) * 1e6
        k = int(k_per_feature)
        k = min(k, max(0, L - 1))
        topk_idx = torch.topk(scores, k=k, dim=1, largest=True, sorted=False).indices  # [B, k, N]
        # Build indicating mask via scatter
        indicating_mask = torch.zeros(B, L, N, device=device)
        indicating_mask.scatter_(dim=1, index=topk_idx, src=torch.ones_like(topk_idx, dtype=indicating_mask.dtype))
        # Remove any selected invalid positions and ensure at least one observed remains
        indicating_mask = indicating_mask * observed_mask
        # Zero-out artificial masks for features before start_idx
        indicating_mask = indicating_mask * feature_gate
        too_small = (obs_counts <= 1.0).to(indicating_mask.dtype).unsqueeze(1)  # [B,1,N]
        indicating_mask = indicating_mask * (1.0 - too_small)
        # Compose final masks
        mask_missing_plus_artificial = observed_mask * (1.0 - indicating_mask)
    else:
        # Deterministic vectorized per-sample selection (val/test) to reduce Python overhead
        if metadata_list is None:
            raise ValueError("metadata_list required for deterministic mask generation in val/test")
        obs_counts = observed_mask.sum(dim=1)  # [B, N]
        k = int(k_per_feature)
        k = min(k, max(0, L - 1))
        for b in range(B):
            seed_val = _metadata_to_seed(metadata_list[b], base_seed)
            gen = torch.Generator(device='cpu').manual_seed(int(seed_val))
            scores_cpu = torch.rand(L, N, generator=gen)
            scores = scores_cpu.to(device=device, dtype=X_orig.dtype)
            scores = scores - (1.0 - observed_mask[b]) * 1e6
            if k > 0:
                idx = torch.topk(scores, k=k, dim=0, largest=True, sorted=False).indices  # [k, N]
                sel = torch.zeros(L, N, device=device, dtype=X_orig.dtype)
                sel.scatter_(dim=0, index=idx, src=torch.ones_like(idx, dtype=X_orig.dtype))
                sel = sel * observed_mask[b]
                # Zero-out artificial masks for features before start_idx
                sel = sel * feature_gate[0]
                too_small = (obs_counts[b] <= 1.0).to(sel.dtype).unsqueeze(0)  # [1,N]
                sel = sel * (1.0 - too_small)
                indicating_mask[b] = sel
                mask_missing_plus_artificial[b] = observed_mask[b] * (1.0 - sel)

    # Build X_masked by zeroing both original-missing and artificial-masked positions
    # Equivalent to multiplying by mask_missing_plus_artificial
    X_masked = X_orig * mask_missing_plus_artificial

    return X_masked, mask_missing_only, mask_missing_plus_artificial, indicating_mask


def build_model(d_time: int, d_feature: int, device: torch.device) -> SAITS:
    # SAITS default-ish hyperparameters (kept simple)
    model = SAITS(
        n_groups=1,
        n_group_inner_layers=1,
        d_time=d_time,
        d_feature=d_feature,
        d_model=d_feature,  # keep default simple
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


def train_one_epoch(
    model: SAITS,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    base_seed: int,
) -> Tuple[float, float]:
    model.train()
    total_recon = 0.0
    total_impute = 0.0
    steps = 0
    import time as _time
    data_wait_total = 0.0
    mask_time_total = 0.0
    fwd_time_total = 0.0
    bwd_time_total = 0.0
    iter_start = _time.time()

    for past_batch, future_batch, metadata_list in loader:
        # data wait time
        now = _time.time()
        data_wait_total += (now - iter_start)
        # past_batch: [B, 39, L]; convert to [B, L, 39]
        X_orig = past_batch.to(device, non_blocking=True).permute(0, 2, 1).contiguous()
        # Generate masks (train mode: resample every step)
        t0 = _time.time()
        X_masked, mask_missing_only, mask_mixed, indicating = sample_artificial_mask_per_batch(
            X_orig, k_per_feature=30, mode="train", base_seed=base_seed, metadata_list=metadata_list,
            start_feature_one_based=22,
        )
        mask_time_total += (_time.time() - t0)
        # observed mask after artificial: 1 observed, 0 missing
        observed_after = mask_mixed  # already float binary 0/1
        # Build reconstruction mask to exclude first 21 features from reconstruction loss
        _, _, N = X_orig.shape
        start_idx = max(0, min(N, 22 - 1))
        feature_gate = torch.zeros(1, 1, N, device=device, dtype=X_orig.dtype)
        if start_idx < N:
            feature_gate[:, :, start_idx:] = 1.0
        reconstruction_mask = observed_after * feature_gate

        inputs = {
            "X": X_masked,
            "missing_mask": observed_after,
            "X_holdout": X_orig,
            "indicating_mask": indicating,
            "reconstruction_mask": reconstruction_mask,
        }
        # forward
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        t1 = _time.time()
        out = model(inputs, stage="train")
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        fwd_time_total += (_time.time() - t1)
        recon = out["reconstruction_loss"]
        impute = out["imputation_loss"]
        loss = recon + impute

        # backward
        t2 = _time.time()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        bwd_time_total += (_time.time() - t2)

        total_recon += float(recon.detach().cpu().item())
        total_impute += float(impute.detach().cpu().item())
        steps += 1
        iter_start = _time.time()

    denom = max(steps, 1)
    print(f"[Timing] data_wait/it={data_wait_total/denom:.4f}s, mask/it={mask_time_total/denom:.4f}s, fwd/it={fwd_time_total/denom:.4f}s, bwd/it={bwd_time_total/denom:.4f}s")
    return total_recon / denom, total_impute / denom


@torch.no_grad()
def evaluate(
    model: SAITS,
    loader: DataLoader,
    device: torch.device,
    split: str,
    base_seed: int,
) -> Tuple[float, float]:
    model.eval()
    total_recon = 0.0
    total_impute = 0.0
    steps = 0
    import time as _time
    data_wait_total = 0.0
    mask_time_total = 0.0
    fwd_time_total = 0.0
    iter_start = _time.time()

    for past_batch, future_batch, metadata_list in loader:
        now = _time.time()
        data_wait_total += (now - iter_start)
        X_orig = past_batch.to(device, non_blocking=True).permute(0, 2, 1).contiguous()
        # Deterministic masks in val/test
        t0 = _time.time()
        X_masked, mask_missing_only, mask_mixed, indicating = sample_artificial_mask_per_batch(
            X_orig, k_per_feature=30, mode=split, base_seed=base_seed, metadata_list=metadata_list,
            start_feature_one_based=22,
        )
        mask_time_total += (_time.time() - t0)
        observed_after = mask_mixed  # already float binary 0/1
        # Build reconstruction mask to exclude first 21 features from reconstruction loss
        _, _, N = X_orig.shape
        start_idx = max(0, min(N, 22 - 1))
        feature_gate = torch.zeros(1, 1, N, device=device, dtype=X_orig.dtype)
        if start_idx < N:
            feature_gate[:, :, start_idx:] = 1.0
        reconstruction_mask = observed_after * feature_gate
        inputs = {
            "X": X_masked,
            "missing_mask": observed_after,
            "X_holdout": X_orig,
            "indicating_mask": indicating,
            "reconstruction_mask": reconstruction_mask,
        }
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        t1 = _time.time()
        out = model(inputs, stage="val" if split == "val" else "test")
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        fwd_time_total += (_time.time() - t1)
        total_recon += float(out["reconstruction_loss"].detach().cpu().item())
        total_impute += float(out["imputation_loss"].detach().cpu().item())
        steps += 1
        iter_start = _time.time()

    denom = max(steps, 1)
    print(f"[Timing {split}] data_wait/it={data_wait_total/denom:.4f}s, mask/it={mask_time_total/denom:.4f}s, fwd/it={fwd_time_total/denom:.4f}s")
    return total_recon / denom, total_impute / denom


def main():
    parser = argparse.ArgumentParser(description="Train SAITS for imputation with artificial masking")
    parser.add_argument(
        "--h5_dir",
        type=str,
        default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/year_datasets_h5_masked_10x",
        help="H5 dataset directory (default aligns with forecasting pipeline)"
    )
    parser.add_argument("--train_years", type=str, default="2000-2020", help="Train years, e.g., 2001-2022")
    parser.add_argument("--val_years", type=str, default="2021-2022", help="Validation years, e.g., 2023")
    parser.add_argument("--test_years", type=str, default="2023-2024", help="Test years, e.g., 2024")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience on val imputation_MAE")
    parser.add_argument("--save_dir", type=str, default="./runs_imputation")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build year-based loader (aligned with forecasting pipeline)
    def parse_years(s: str) -> List[int]:
        years = []
        for part in s.split(','):
            if '-' in part:
                a, b = part.split('-')
                years.extend(list(range(int(a), int(b) + 1)))
            else:
                years.append(int(part))
        return years

    data_loader = TimeSeriesDataLoader(
        h5_dir=args.h5_dir,
        positive_ratio=1.0,
        pos_neg_ratio=2,
        resample_each_epoch=False,
        lookback_seq=365,
        forecast_hor=7,
        verbose_sampling=False,
        enable_performance_optimizations=True,
    )

    train_indices, val_indices, test_indices = data_loader.get_year_based_split(
        train_years=parse_years(args.train_years),
        val_years=parse_years(args.val_years),
        test_years=parse_years(args.test_years),
    )

    train_dataset = Subset(data_loader.dataset, train_indices)
    val_dataset = Subset(data_loader.dataset, val_indices)
    test_dataset = Subset(data_loader.dataset, test_indices)

    # DataLoaders with dataset's collate_fn returning (past, future, metadata)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=data_loader.dataset.custom_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=data_loader.dataset.custom_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=data_loader.dataset.custom_collate_fn,
    )

    # Infer L, N from one batch
    sample_batch = next(iter(train_loader))
    past_sample = sample_batch[0]
    # past_sample: [B, 39, L]
    _, N, L = past_sample.shape

    model = build_model(d_time=L, d_feature=N, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_impute = float("inf")
    best_path = os.path.join(args.save_dir, "saits_best.pt")
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_recon, train_impute = train_one_epoch(model, train_loader, optimizer, device, epoch, base_seed=args.seed)
        val_recon, val_impute = evaluate(model, val_loader, device, split="val", base_seed=args.seed)
        dt = time.time() - t0
        print(
            f"Epoch {epoch:03d} | time {dt:.1f}s | train(recon={train_recon:.5f}, imp={train_impute:.5f}) | val(recon={val_recon:.5f}, imp={val_impute:.5f})"
        )

        # Early stopping on val imputation MAE
        if val_impute < best_val_impute:
            best_val_impute = val_impute
            patience_left = args.patience
            torch.save({"model_state_dict": model.state_dict(), "L": L, "N": N}, best_path)
            print(f"  Saved new best to {best_path} (val_impute={best_val_impute:.6f})")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered.")
                break

    # Load best and evaluate on test
    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best model from {best_path}")

    test_recon, test_impute = evaluate(model, test_loader, device, split="test", base_seed=args.seed)
    print(f"Test: recon={test_recon:.5f}, imp={test_impute:.5f}")


if __name__ == "__main__":
    main()

