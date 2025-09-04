# dataload_year_new.py
# 新版 dataload：缓存数据本身，而不是索引
# -----------------------------------------------------------
# - 首次运行时会遍历 raw Dataset（__getitem__ 返回 (past, future, meta)）
#   并将样本写入分片 .npz 文件
# - 后续运行直接 memory-map 加载 .npz，取数据无需再次扫描原始数据
# - Cache 的目录名由参数签名生成（lookback、forecast、variables...）
#   → 改参数 = 新建 cache；改回去 = 命中旧 cache
# - 保留 TimeSeriesPixelDataset / TimeSeriesDataLoader 接口
#   与你的训练脚本兼容
# -----------------------------------------------------------

import os
import json
import hashlib
import pickle
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

import numpy as np
import torch
from torch.utils.data import Dataset

# -------------------- helpers --------------------

def _json_stable(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

def _hash_sig(d: Dict[str, Any]) -> str:
    return hashlib.sha1(_json_stable(d).encode("utf-8")).hexdigest()[:16]

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

# -------------------- cache signature --------------------

@dataclass
class CacheSignature:
    lookback: int
    forecast: int
    variables: Tuple[str, ...]              # channel names/order
    normalization: str = "none"
    mask_policy: str = "none"
    spatial_patch: Tuple[int, int] = (1, 1) # (H,W) or (1,1) for pixel-level
    target_type: str = "binary"
    dataset_version: str = "v1"
    sampler_version: str = "v1"
    seed: int = 42
    extra: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["variables"] = list(self.variables)
        d["spatial_patch"] = list(self.spatial_patch)
        d["extra"] = d["extra"] or {}
        return d

    def uid(self) -> str:
        core = self.to_dict()
        h = _hash_sig(core)
        return f"lb{self.lookback}_fc{self.forecast}_vars{len(self.variables)}_patch{self.spatial_patch[0]}x{self.spatial_patch[1]}_{h}"

# -------------------- cached dataset --------------------

class _CachedShardsDataset(Dataset):
    """
    Backed by shard .npz files: X [N,...], y [N,...], meta [N,...].
    np.load(mmap_mode='r') for fast random access; returns torch tensors + meta.
    """
    def __init__(self, shard_paths: List[str]) -> None:
        super().__init__()
        assert shard_paths, "No shard paths provided."
        self._paths = shard_paths
        self._shards = []
        self._cum = []
        total = 0
        for p in shard_paths:
            arr = np.load(p, mmap_mode="r", allow_pickle=True)
            n = int(arr["X"].shape[0])
            self._shards.append(arr)
            total += n
            self._cum.append(total)
        self._total = total
        x0 = self._shards[0]["X"]
        y0 = self._shards[0]["y"]
        self.feature_shape = tuple(x0.shape[1:])
        self.target_shape = tuple(y0.shape[1:])
        self.has_meta = "meta" in self._shards[0].files

    def __len__(self) -> int:
        return self._total

    def _locate(self, i: int) -> Tuple[int, int]:
        if i < 0:
            i += self._total
        lo, hi = 0, len(self._cum)-1
        while lo <= hi:
            mid = (lo+hi)//2
            if i < self._cum[mid]:
                hi = mid-1
            else:
                lo = mid+1
        shard_idx = lo
        start = 0 if shard_idx == 0 else self._cum[shard_idx-1]
        off = i - start
        return shard_idx, off

    def __getitem__(self, i: int):
        s, off = self._locate(i)
        sh = self._shards[s]
        x = torch.from_numpy(np.asarray(sh["X"][off]))
        y = torch.from_numpy(np.asarray(sh["y"][off]))
        if self.has_meta:
            m = sh["meta"][off]
            return x, y, m
        else:
            return x, y, None

# -------------------- public dataset --------------------

class TimeSeriesPixelDataset(Dataset):
    """
    Wrapper that builds/loads a *data-level* cache from a base/raw dataset.

    base_dataset: any Dataset whose __getitem__ returns (past, future, meta)
    """
    def __init__(
        self,
        cache_root: str,
        signature: Union[CacheSignature, Dict[str, Any]],
        base_dataset: Optional[Dataset] = None,
        raw_dataset_ctor: Optional[Callable[..., Dataset]] = None,
        raw_kwargs: Optional[Dict[str, Any]] = None,
        shard_size: int = 4096,
        dtype: str = "float32",   # or "float16"
        overwrite: bool = False,
    ) -> None:
        super().__init__()
        self.cache_root = cache_root
        if isinstance(signature, dict):
            signature = CacheSignature(
                lookback=signature.get("lookback", 0),
                forecast=signature.get("forecast", 0),
                variables=tuple(signature.get("variables", [])),
                normalization=signature.get("normalization", "none"),
                mask_policy=signature.get("mask_policy", "none"),
                spatial_patch=tuple(signature.get("spatial_patch", (1,1))),
                target_type=signature.get("target_type", "binary"),
                dataset_version=signature.get("dataset_version", "v1"),
                sampler_version=signature.get("sampler_version", "v1"),
                seed=int(signature.get("seed", 42)),
                extra=signature.get("extra", {}) or {},
            )
        self.signature = signature
        self.uid = signature.uid()
        self.root = os.path.join(cache_root, self.uid)
        self.shards_dir = os.path.join(self.root, "shards")
        self.meta_path = os.path.join(self.root, "meta.pkl")
        _ensure_dir(self.shards_dir)

        # Construct base dataset if not provided
        if base_dataset is None and raw_dataset_ctor is not None:
            base_dataset = raw_dataset_ctor(**(raw_kwargs or {}))
        if base_dataset is None:
            raise ValueError("Either base_dataset or raw_dataset_ctor+raw_kwargs must be provided.")

        # Build or load cache
        if (not overwrite) and os.path.exists(self.meta_path):
            with open(self.meta_path, "rb") as f:
                meta = pickle.load(f)
            shard_files = meta["shard_files"]
            shard_paths = [os.path.join(self.shards_dir, fn) for fn in shard_files]
            self._cached = _CachedShardsDataset(shard_paths)
        else:
            self._cached = self._build_from_base(base_dataset, shard_size, dtype)

    def _build_from_base(self, base: Dataset, shard_size: int, dtype: str) -> _CachedShardsDataset:
        n = len(base)
        if n == 0:
            raise ValueError("Base dataset is empty; cannot build cache.")
        shard_files: List[str] = []
        X_buf: List[np.ndarray] = []
        Y_buf: List[np.ndarray] = []
        M_buf: List[Any] = []

        def flush(k: int):
            if not X_buf:
                return
            X_arr = np.stack(X_buf, axis=0)
            Y_arr = np.stack(Y_buf, axis=0)
            if dtype == "float16":
                X_arr = X_arr.astype(np.float16, copy=False)
            elif dtype == "float32":
                X_arr = X_arr.astype(np.float32, copy=False)
            else:
                raise ValueError("dtype must be 'float32' or 'float16'")
            shard_name = f"shard_{k:05d}.npz"
            shard_path = os.path.join(self.shards_dir, shard_name)
            np.savez(shard_path, X=X_arr, y=Y_arr, meta=np.array(M_buf, dtype=object))
            shard_files.append(shard_name)
            X_buf.clear(); Y_buf.clear(); M_buf.clear()

        k = 0
        for i in range(n):
            past, future, meta = base[i]
            if isinstance(past, torch.Tensor):  past = past.detach().cpu().numpy()
            if isinstance(future, torch.Tensor):future = future.detach().cpu().numpy()
            X_buf.append(past)
            Y_buf.append(future)
            M_buf.append(meta)
            if len(X_buf) >= shard_size:
                flush(k); k += 1
        flush(k)

        meta = {
            "signature": self.signature.to_dict(),
            "num_samples": n,
            "shard_size": shard_size,
            "dtype": dtype,
            "shard_files": shard_files,
        }
        with open(self.meta_path, "wb") as f:
            pickle.dump(meta, f)

        shard_paths = [os.path.join(self.shards_dir, fn) for fn in shard_files]
        return _CachedShardsDataset(shard_paths)

    def __len__(self) -> int:
        return len(self._cached)

    def __getitem__(self, idx: int):
        return self._cached[idx]

    @staticmethod
    def custom_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, Any]]):
        pasts, futures, metas = zip(*batch)
        x = torch.stack(pasts, dim=0)
        y = torch.stack(futures, dim=0)
        return x, y, list(metas)

# -------------------- DataLoader helper --------------------

class TimeSeriesDataLoader:
    """
    Convenience wrapper expected by the training script.
    Exposes:
      - .dataset  (a TimeSeriesPixelDataset that serves cached DATA)
      - .get_year_based_split(...)
    """
    def __init__(
        self,
        h5_dir: Optional[str] = None,
        positive_ratio: float = 0.5,
        pos_neg_ratio: float = 1.0,
        resample_each_epoch: bool = False,

        cache_root: str = "./_cache_data",
        signature: Optional[Union[CacheSignature, Dict[str, Any]]] = None,
        base_dataset: Optional[Dataset] = None,
        raw_dataset_ctor: Optional[Callable[..., Dataset]] = None,
        raw_kwargs: Optional[Dict[str, Any]] = None,
        shard_size: int = 4096,
        dtype: str = "float32",
        overwrite: bool = False,
    ) -> None:
        if signature is None:
            signature = {
                "lookback": 0,
                "forecast": 0,
                "variables": [],
                "dataset_version": "v1",
                "sampler_version": "v1",
                "seed": 42,
                "extra": {"h5_dir": h5_dir, "pos_ratio": positive_ratio, "pn_ratio": pos_neg_ratio}
            }

        self.dataset = TimeSeriesPixelDataset(
            cache_root=cache_root,
            signature=signature,
            base_dataset=base_dataset,
            raw_dataset_ctor=raw_dataset_ctor,
            raw_kwargs=raw_kwargs or {"h5_dir": h5_dir},
            shard_size=shard_size,
            dtype=dtype,
            overwrite=overwrite,
        )

    def get_year_based_split(
        self,
        train_years: List[int],
        val_years: List[int],
        test_years: List[int],
    ) -> Tuple[List[int], List[int], List[int]]:
        years = []
        for i in range(len(self.dataset)):
            _, _, meta = self.dataset[i]
            date_int = int(meta[0]) if isinstance(meta, (list, tuple, np.ndarray)) else int(meta)
            year = date_int // 10000  # assume YYYYMMDD
            years.append(year)

        train_idx = [i for i, y in enumerate(years) if y in train_years]
        val_idx   = [i for i, y in enumerate(years) if y in val_years]
        test_idx  = [i for i, y in enumerate(years) if y in test_years]

        return train_idx, val_idx, test_idx
