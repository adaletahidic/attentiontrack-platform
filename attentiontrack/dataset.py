from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def load_npz_paths(features_dir: str | Path, split: str) -> List[Path]:
    p = Path(features_dir) / split
    if not p.exists():
        raise FileNotFoundError(f"Split folder not found: {p}")
    return sorted([x for x in p.rglob("*.npz") if x.is_file()])


def normalize_and_impute(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """NaN-safe z-score, then impute NaNs to 0."""
    Xn = (X - mean) / (std + 1e-8)
    Xn = np.where(np.isfinite(Xn), Xn, 0.0).astype(np.float32)
    return Xn


class NPZSequenceDataset(Dataset):
    def __init__(
        self,
        paths: List[Path],
        *,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        self.paths = paths
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        npz = np.load(self.paths[idx], allow_pickle=True)
        X = npz["X"].astype(np.float32)  # (T, F)
        length = int(npz["length"])
        y = int(npz["y"])

        if self.mean is not None and self.std is not None:
            X = normalize_and_impute(X, self.mean, self.std)

        return torch.from_numpy(X), torch.tensor(length, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def collate_batch(batch):
    """
    Pads to max T in batch.
    Returns:
      X: (B, T_max, F)
      lengths: (B,)
      y: (B,)
    """
    Xs, lens, ys = zip(*batch)
    lengths = torch.stack(lens, dim=0)
    y = torch.stack(ys, dim=0)

    T_max = max(x.shape[0] for x in Xs)
    F = Xs[0].shape[1]
    X_pad = torch.zeros((len(Xs), T_max, F), dtype=torch.float32)
    for i, x in enumerate(Xs):
        T = x.shape[0]
        X_pad[i, :T] = x
    return X_pad, lengths, y
