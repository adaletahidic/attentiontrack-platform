#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np


def gather_npz_files(root: Path) -> List[Path]:
    files = []
    for split in ["train", "val", "test"]:
        split_dir = root / split
        if split_dir.exists():
            files.extend(sorted(split_dir.rglob("*.npz")))
    return files


def load_labels(paths: List[Path]) -> np.ndarray:
    labels = []
    for p in paths:
        npz = np.load(p, allow_pickle=True)
        y = int(npz["y"])
        labels.append(y)
    return np.asarray(labels, dtype=np.int64)


def stratified_split_indices(
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8, "Ratios must sum to 1."

    rng = np.random.default_rng(seed)

    idx_all = np.arange(len(y))
    idx_pos = idx_all[y == 1]
    idx_neg = idx_all[y == 0]

    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    def split_class_indices(idx_cls: np.ndarray):
        n = len(idx_cls)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_test = n - n_train - n_val

        train_idx = idx_cls[:n_train]
        val_idx = idx_cls[n_train:n_train + n_val]
        test_idx = idx_cls[n_train + n_val:n_train + n_val + n_test]
        return train_idx, val_idx, test_idx

    pos_train, pos_val, pos_test = split_class_indices(idx_pos)
    neg_train, neg_val, neg_test = split_class_indices(idx_neg)

    train_idx = np.concatenate([pos_train, neg_train])
    val_idx = np.concatenate([pos_val, neg_val])
    test_idx = np.concatenate([pos_test, neg_test])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx


def print_split_stats(name: str, labels: np.ndarray) -> None:
    total = len(labels)
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    pos_ratio = pos / max(total, 1)
    print(f"{name}: total={total}, y0={neg}, y1={pos}, pos_ratio={pos_ratio:.4f}")


def save_manifest(
    out_csv: Path,
    files: List[Path],
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    split_map = {}
    for i in train_idx:
        split_map[int(i)] = "train"
    for i in val_idx:
        split_map[int(i)] = "val"
    for i in test_idx:
        split_map[int(i)] = "test"

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "path", "filename", "label", "split"])
        for i, p in enumerate(files):
            writer.writerow([i, str(p), p.name, int(labels[i]), split_map[i]])


def copy_files_to_new_split(
    files: List[Path],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    dst_root: Path,
    flatten: bool = True,
) -> None:
    dst_root.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (dst_root / split).mkdir(parents=True, exist_ok=True)

    def copy_subset(indices: np.ndarray, split_name: str):
        for i in indices:
            src = files[int(i)]
            if flatten:
                dst = dst_root / split_name / src.name
            else:
                dst = dst_root / split_name / src.name
            shutil.copy2(src, dst)

    copy_subset(train_idx, "train")
    copy_subset(val_idx, "val")
    copy_subset(test_idx, "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--features_dir",
        type=str,
        required=True,
        help="Root folder containing existing train/val/test NPZ folders.",
    )
    ap.add_argument(
        "--manifest_csv",
        type=str,
        default="runs_full/resplit/split_manifest.csv",
        help="Where to save the new split manifest CSV.",
    )
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--copy_to_dir",
        type=str,
        default="",
        help="Optional destination root directory for physically copying files into new train/val/test folders.",
    )

    args = ap.parse_args()

    features_dir = Path(args.features_dir)
    manifest_csv = Path(args.manifest_csv)

    files = gather_npz_files(features_dir)
    if not files:
        raise FileNotFoundError(f"No NPZ files found under: {features_dir}")

    print(f"Found {len(files)} NPZ files.")
    labels = load_labels(files)

    print_split_stats("all", labels)

    train_idx, val_idx, test_idx = stratified_split_indices(
        y=labels,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print_split_stats("train", labels[train_idx])
    print_split_stats("val", labels[val_idx])
    print_split_stats("test", labels[test_idx])

    save_manifest(
        out_csv=manifest_csv,
        files=files,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    print(f"Saved manifest: {manifest_csv}")

    if args.copy_to_dir:
        dst_root = Path(args.copy_to_dir)
        copy_files_to_new_split(
            files=files,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            dst_root=dst_root,
            flatten=True,
        )
        print(f"Copied new split to: {dst_root}")


if __name__ == "__main__":
    main()