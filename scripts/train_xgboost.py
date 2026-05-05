#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from xgboost import XGBClassifier

from attentiontrack.dataset import load_npz_paths


def safe_stats_from_sequence(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32)

    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    min_ = np.nanmin(X, axis=0)
    max_ = np.nanmax(X, axis=0)

    feats = np.concatenate([mean, std, min_, max_], axis=0)
    feats = np.where(np.isfinite(feats), feats, 0.0).astype(np.float32)
    return feats


def load_tabular_split(paths: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
    X_list = []
    y_list = []

    for p in paths:
        npz = np.load(p, allow_pickle=True)
        X_seq = npz["X"]
        y = int(npz["y"])

        x_tab = safe_stats_from_sequence(X_seq)
        X_list.append(x_tab)
        y_list.append(y)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y


def binary_confusion(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp, tn, fp, fn


def binary_metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp, tn, fp, fn = binary_confusion(y_true, y_pred)

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2.0 * prec * rec / max(prec + rec, 1e-12)
    spec = tn / max(tn + fp, 1)
    bal_acc = 0.5 * (rec + spec)

    return {
        "acc": float(acc),
        "prec": float(prec),
        "rec": float(rec),
        "f1": float(f1),
        "spec": float(spec),
        "bal_acc": float(bal_acc),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def rankdata_avg(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)

    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    return ranks


def roc_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = rankdata_avg(y_score)
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def pr_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    tp = 0
    fp = 0
    precisions = [1.0]
    recalls = [0.0]

    for yy in y_sorted:
        if yy == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / max(tp + fp, 1))
        recalls.append(tp / n_pos)

    precisions = np.asarray(precisions, dtype=np.float64)
    recalls = np.asarray(recalls, dtype=np.float64)

    area = 0.0
    for i in range(1, len(recalls)):
        area += (recalls[i] - recalls[i - 1]) * precisions[i]
    return float(area)


def evaluate_with_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    metrics = binary_metrics_from_preds(y_true, y_pred)
    metrics["threshold"] = float(threshold)
    metrics["roc_auc"] = roc_auc_score_np(y_true, y_prob)
    metrics["pr_auc"] = pr_auc_score_np(y_true, y_prob)
    return metrics


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_name: str = "bal_acc",
    thr_min: float = 0.20,
    thr_max: float = 0.90,
    thr_steps: int = 15,
):
    best_thr = 0.5
    best_metrics = None
    best_score = -1.0

    for thr in np.linspace(thr_min, thr_max, thr_steps):
        y_pred = (y_prob >= thr).astype(np.int64)
        metrics = binary_metrics_from_preds(y_true, y_pred)
        score = float(metrics[metric_name])

        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_metrics = metrics

    return best_thr, best_metrics


def split_stats(name: str, y: np.ndarray) -> None:
    total = len(y)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    print(f"{name}: total={total}, y0={neg}, y1={pos}, pos_ratio={pos / max(total, 1):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", type=str, required=True)
    ap.add_argument("--save_dir", type=str, default="runs_full/xgboost")
    ap.add_argument("--decision_metric", type=str, default="bal_acc",
                    choices=["bal_acc", "f1", "acc", "prec", "rec", "spec"])
    ap.add_argument("--thr_min", type=float, default=0.20)
    ap.add_argument("--thr_max", type=float, default=0.90)
    ap.add_argument("--thr_steps", type=int, default=15)

    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--max_depth", type=int, default=4)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--subsample", type=float, default=0.9)
    ap.add_argument("--colsample_bytree", type=float, default=0.9)
    ap.add_argument("--random_state", type=int, default=42)

    args = ap.parse_args()

    features_dir = Path(args.features_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_paths = load_npz_paths(features_dir, "train")
    val_paths = load_npz_paths(features_dir, "val")
    test_paths = load_npz_paths(features_dir, "test")

    X_train, y_train = load_tabular_split(train_paths)
    X_val, y_val = load_tabular_split(val_paths)
    X_test, y_test = load_tabular_split(test_paths)

    print(f"Tabular feature dim: {X_train.shape[1]}")
    split_stats("train", y_train)
    split_stats("val", y_val)
    split_stats("test", y_test)

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=args.random_state,
        scale_pos_weight=scale_pos_weight,
        n_jobs=4,
    )

    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    best_thr, _ = find_best_threshold(
        y_val, val_prob,
        metric_name=args.decision_metric,
        thr_min=args.thr_min,
        thr_max=args.thr_max,
        thr_steps=args.thr_steps,
    )

    val_metrics = evaluate_with_threshold(y_val, val_prob, best_thr)

    test_prob = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_with_threshold(y_test, test_prob, best_thr)

    print(f"BEST threshold from val: {best_thr:.3f}")
    print(f"VAL metrics: {val_metrics}")
    print(f"TEST metrics: {test_metrics}")

    with open(save_dir / "xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)

    summary = {
        "feature_dim": int(X_train.shape[1]),
        "best_threshold": best_thr,
        "val": val_metrics,
        "test": test_metrics,
        "params": vars(args),
        "scale_pos_weight": scale_pos_weight,
    }
    (save_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved: {save_dir / 'xgboost_model.pkl'} and {save_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()