#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from attentiontrack.dataset import load_npz_paths, NPZSequenceDataset, collate_batch


def compute_norm_stats(train_paths: List[Path]) -> Tuple[np.ndarray, np.ndarray, float]:
    sum_ = None
    sumsq = None
    cnt = None
    y_pos = 0
    y_neg = 0

    for p in train_paths:
        npz = np.load(p, allow_pickle=True)
        X = npz["X"].astype(np.float32)
        y = int(npz["y"])

        if y == 1:
            y_pos += 1
        else:
            y_neg += 1

        if sum_ is None:
            feat_dim = X.shape[1]
            sum_ = np.zeros(feat_dim, dtype=np.float64)
            sumsq = np.zeros(feat_dim, dtype=np.float64)
            cnt = np.zeros(feat_dim, dtype=np.float64)

        mask = np.isfinite(X)
        X0 = np.where(mask, X, 0.0).astype(np.float64)

        sum_ += X0.sum(axis=0)
        sumsq += (X0 * X0).sum(axis=0)
        cnt += mask.sum(axis=0)

    mean = sum_ / np.maximum(cnt, 1.0)
    var = sumsq / np.maximum(cnt, 1.0) - mean * mean
    var = np.maximum(var, 1e-6)
    std = np.sqrt(var)

    raw_pos_weight = float(y_neg / max(y_pos, 1))
    return mean.astype(np.float32), std.astype(np.float32), raw_pos_weight


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


@torch.no_grad()
def collect_probs(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys = []
    probs = []

    for X, lengths, y in loader:
        X = X.to(device)
        lengths = lengths.to(device)
        logits = model(X, lengths).squeeze(-1)
        p = torch.sigmoid(logits)

        ys.append(y.cpu().numpy())
        probs.append(p.cpu().numpy())

    return (
        np.concatenate(ys, axis=0).astype(np.int64),
        np.concatenate(probs, axis=0).astype(np.float32),
    )


def evaluate_with_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    metrics = binary_metrics_from_preds(y_true, y_pred)
    metrics["threshold"] = float(threshold)
    metrics["roc_auc"] = roc_auc_score_np(y_true, y_prob)
    metrics["pr_auc"] = pr_auc_score_np(y_true, y_prob)
    return metrics


def find_best_threshold(y_true, y_prob, metric_name="bal_acc", thr_min=0.20, thr_max=0.90, thr_steps=15):
    best_thr = 0.5
    best_score = -1.0
    best_metrics = None

    for thr in np.linspace(thr_min, thr_max, thr_steps):
        y_pred = (y_prob >= thr).astype(np.int64)
        metrics = binary_metrics_from_preds(y_true, y_pred)
        score = float(metrics[metric_name])
        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_metrics = metrics

    return best_thr, best_metrics


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerEncoderClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.3,
        num_classes: int = 1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=512)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, lengths):
        T = x.size(1)
        key_padding_mask = torch.arange(T, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)

        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)

        valid_mask = (~key_padding_mask).float().unsqueeze(-1)
        pooled = (h * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)

        logits = self.head(pooled)
        return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", type=str, required=True)
    ap.add_argument("--save_dir", type=str, default="runs_full/transformer_encoder")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dim_feedforward", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--pos_weight_scale", type=float, default=0.3)
    ap.add_argument("--decision_metric", type=str, default="bal_acc",
                    choices=["bal_acc", "f1", "acc", "prec", "rec", "spec"])
    ap.add_argument("--thr_min", type=float, default=0.20)
    ap.add_argument("--thr_max", type=float, default=0.90)
    ap.add_argument("--thr_steps", type=int, default=15)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    features_dir = Path(args.features_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_paths = load_npz_paths(features_dir, "train")
    val_paths = load_npz_paths(features_dir, "val")
    test_paths = load_npz_paths(features_dir, "test")

    print(f"train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}")

    mean, std, raw_pos_weight = compute_norm_stats(train_paths)
    used_pos_weight = max(1.0, raw_pos_weight * args.pos_weight_scale)

    print(f"raw_pos_weight={raw_pos_weight:.3f}")
    print(f"used_pos_weight={used_pos_weight:.3f}")

    ds_train = NPZSequenceDataset(train_paths, mean=mean, std=std)
    ds_val = NPZSequenceDataset(val_paths, mean=mean, std=std)
    ds_test = NPZSequenceDataset(test_paths, mean=mean, std=std)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    X0, _, _ = ds_train[0]
    input_size = X0.shape[1]
    print(f"Input size: {input_size}")

    model = TransformerEncoderClassifier(
        input_size=input_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_classes=1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([used_pos_weight], dtype=torch.float32, device=device)
    )

    best_score = -1.0
    best_threshold = 0.5
    best_path = save_dir / "best.pt"
    patience_left = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_items = 0

        for X, lengths, y in dl_train:
            X = X.to(device)
            lengths = lengths.to(device)
            y = y.to(device).float()

            optimizer.zero_grad()
            logits = model(X, lengths).squeeze(-1)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item()) * X.size(0)
            n_items += X.size(0)

        train_loss = total_loss / max(n_items, 1)

        y_val_true, y_val_prob = collect_probs(model, dl_val, device)
        thr, _ = find_best_threshold(
            y_val_true, y_val_prob,
            metric_name=args.decision_metric,
            thr_min=args.thr_min,
            thr_max=args.thr_max,
            thr_steps=args.thr_steps,
        )
        val_metrics = evaluate_with_threshold(y_val_true, y_val_prob, thr)
        score = float(val_metrics[args.decision_metric])

        print(
            f"Epoch {epoch:02d} | loss={train_loss:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | val_bal_acc={val_metrics['bal_acc']:.4f} | "
            f"val_spec={val_metrics['spec']:.4f} | val_roc_auc={val_metrics['roc_auc']:.4f} | "
            f"val_pr_auc={val_metrics['pr_auc']:.4f} | thr={thr:.2f}"
        )

        history.append({"epoch": epoch, "train_loss": train_loss, "val": val_metrics})

        if score > best_score + 1e-6:
            best_score = score
            best_threshold = thr
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "best_threshold": best_threshold,
                    "input_size": input_size,
                    "args": vars(args),
                },
                best_path,
            )
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    best_threshold = float(ckpt.get("best_threshold", 0.5))

    y_val_true, y_val_prob = collect_probs(model, dl_val, device)
    final_val_metrics = evaluate_with_threshold(y_val_true, y_val_prob, best_threshold)

    y_test_true, y_test_prob = collect_probs(model, dl_test, device)
    test_metrics = evaluate_with_threshold(y_test_true, y_test_prob, best_threshold)

    print(f"BEST threshold from val: {best_threshold:.3f}")
    print(f"FINAL VAL metrics: {final_val_metrics}")
    print(f"TEST metrics: {test_metrics}")

    (save_dir / "norm.json").write_text(
        json.dumps(
            {"mean": mean.tolist(), "std": std.tolist(), "raw_pos_weight": raw_pos_weight, "used_pos_weight": used_pos_weight},
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "best_score_metric": args.decision_metric,
        "best_score": best_score,
        "best_threshold": best_threshold,
        "final_val": final_val_metrics,
        "test": test_metrics,
        "history": history,
    }
    (save_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved: {best_path}, {save_dir / 'norm.json'}, {save_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()