#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from attentiontrack.dataset import load_npz_paths

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def safe_stats_from_sequence(X: np.ndarray) -> np.ndarray:
    """
    Convert sequence X of shape [T, F] into aggregated tabular features:
    mean, std, min, max over time for each feature.
    NaNs are handled safely.
    Output shape: [4 * F]
    """
    X = X.astype(np.float32)

    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    min_ = np.nanmin(X, axis=0)
    max_ = np.nanmax(X, axis=0)

    feats = np.concatenate([mean, std, min_, max_], axis=0)

    # If an entire column is NaN across time, nanmean/nanstd/nanmin/nanmax can remain NaN.
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


def build_logreg():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
                random_state=42,
            )),
        ]
    )


def build_random_forest():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=4,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]
    )


def run_model(name: str, model, X_train, y_train, X_val, y_val, X_test, y_test, args):
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    best_thr, _ = find_best_threshold(
        y_val,
        val_prob,
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

    return {
        "best_threshold": best_thr,
        "val": val_metrics,
        "test": test_metrics,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", type=str, required=True)
    ap.add_argument("--save_dir", type=str, default="runs_full/baseline_tabular")
    ap.add_argument(
        "--decision_metric",
        type=str,
        default="bal_acc",
        choices=["bal_acc", "f1", "acc", "prec", "rec", "spec"],
    )
    ap.add_argument("--thr_min", type=float, default=0.20)
    ap.add_argument("--thr_max", type=float, default=0.90)
    ap.add_argument("--thr_steps", type=int, default=15)
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

    results = {}

    logreg = build_logreg()
    results["logistic_regression"] = run_model(
        "LogisticRegression",
        logreg,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        args,
    )

    rf = build_random_forest()
    results["random_forest"] = run_model(
        "RandomForest",
        rf,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        args,
    )

    out = {
        "feature_dim": int(X_train.shape[1]),
        "decision_metric": args.decision_metric,
        "threshold_search": {
            "thr_min": args.thr_min,
            "thr_max": args.thr_max,
            "thr_steps": args.thr_steps,
        },
        "results": results,
    }

    out_path = save_dir / "baseline_metrics.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()