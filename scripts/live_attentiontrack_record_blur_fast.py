#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import platform
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch

from attentiontrack.feature_extractor import MediaPipeFeatureExtractor
from attentiontrack.model_ns import AttentionLSTM


def load_norm(norm_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = json.loads(norm_path.read_text(encoding="utf-8"))
    mean = np.array(data["mean"], dtype=np.float32)
    std = np.array(data["std"], dtype=np.float32)
    return mean, std


def norm_impute(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = (x - mean) / (std + 1e-8)
    z = np.where(np.isfinite(z), z, 0.0).astype(np.float32)
    return z


def rule_feedback(att_score: int) -> str:
    if att_score == 3:
        return "Focused"
    if att_score == 2:
        return "Mostly focused"
    if att_score == 1:
        return "Attention dropping"
    return "Refocus needed"


def get_attention_reason(
    eye_closed: int,
    eye_drowsy: int,
    yawning: int,
    head_forward: int,
    body_stable: int,
) -> str:
    reasons = []
    if eye_closed or eye_drowsy:
        reasons.append("eye closure / drowsiness")
    if yawning:
        reasons.append("yawning")
    if not head_forward:
        reasons.append("head pose drift")
    if not body_stable:
        reasons.append("body restlessness")

    if not reasons:
        return "general attention drift"
    return ", ".join(reasons[:2])


def get_dashboard_cause(
    eye_closed: int,
    eye_drowsy: int,
    yawning: int,
    head_forward: int,
    body_stable: int,
) -> str:
    if eye_closed or eye_drowsy:
        return "drowsy/eyes"
    if yawning:
        return "yawning"
    if not head_forward:
        return "looking_away/head_pose"
    if not body_stable:
        return "fidgeting/motion"
    return "unknown"


def compute_severity(
    rule_ready: bool,
    att_score: int,
    drift_ready: bool,
    drift_prob: float | None,
    drift_state: str,
) -> int:
    if drift_ready and str(drift_state).upper() == "DRIFT":
        if drift_prob is not None and drift_prob >= 0.75:
            return 3
        if drift_prob is not None and drift_prob >= 0.55:
            return 2
        return 1

    if rule_ready and att_score <= 0:
        return 3
    if rule_ready and att_score == 1:
        return 2
    return 0


def risk_label(v: float) -> str:
    if not np.isfinite(v):
        return "Unknown"
    if v <= 30:
        return "Low"
    if v <= 60:
        return "Moderate"
    return "High"


def compute_ari_raw(
    rule_ready: bool,
    att_score: int,
    drift_ready: bool,
    drift_prob: float | None,
) -> float:
    rule_risk = np.nan
    if rule_ready:
        rule_risk = float(np.clip((3.0 - att_score) / 3.0 * 100.0, 0.0, 100.0))

    drift_risk = np.nan
    if drift_ready and drift_prob is not None and np.isfinite(drift_prob):
        drift_risk = float(np.clip(drift_prob * 100.0, 0.0, 100.0))

    if np.isfinite(rule_risk) and np.isfinite(drift_risk):
        return float(0.6 * rule_risk + 0.4 * drift_risk)
    if np.isfinite(rule_risk):
        return float(rule_risk)
    if np.isfinite(drift_risk):
        return float(drift_risk)
    return float("nan")


def append_event_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


class PopupManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._active = False

    def notify(self, title: str, message: str):
        with self._lock:
            if self._active:
                return
            self._active = True

        def _worker():
            try:
                if platform.system().lower().startswith("win"):
                    import ctypes
                    ctypes.windll.user32.MessageBoxW(0, message, title, 0x1000)
                else:
                    print(f"[{title}] {message}", flush=True)
            finally:
                with self._lock:
                    self._active = False

        threading.Thread(target=_worker, daemon=True).start()


def apply_privacy_blur(
    frame_bgr: np.ndarray,
    blur_mode: str,
    blur_k: int,
    mp_face: Any,
    mp_selfie: Any,
    seg_thr: float = 0.5,
) -> np.ndarray:
    if blur_mode == "none":
        return frame_bgr

    k = int(blur_k)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1

    h, w = frame_bgr.shape[:2]

    if blur_mode == "full":
        return cv2.GaussianBlur(frame_bgr, (k, k), 0)

    if blur_mode == "face":
        if mp_face is None:
            return frame_bgr
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)
        out = frame_bgr.copy()
        if res.detections:
            for det in res.detections:
                bbox = det.location_data.relative_bounding_box
                x1 = max(0, int(bbox.xmin * w))
                y1 = max(0, int(bbox.ymin * h))
                x2 = min(w, x1 + int(bbox.width * w))
                y2 = min(h, y1 + int(bbox.height * h))
                if x2 > x1 and y2 > y1:
                    roi = out[y1:y2, x1:x2]
                    out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
        return out

    if blur_mode in ("background", "person"):
        if mp_selfie is None:
            return frame_bgr
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = mp_selfie.process(rgb)
        mask = res.segmentation_mask > float(seg_thr)
        mask3 = np.repeat(mask[:, :, None], 3, axis=2)
        blurred = cv2.GaussianBlur(frame_bgr, (k, k), 0)

        if blur_mode == "background":
            return np.where(mask3, frame_bgr, blurred)
        return np.where(mask3, blurred, frame_bgr)

    return frame_bgr


def _infer_model_cfg(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    if "model_state" not in ckpt or "input_size" not in ckpt:
        raise KeyError(
            "Checkpoint must contain 'model_state' and 'input_size'. "
            "Use the best.pt produced by train_lstm_4.py."
        )

    state = ckpt["model_state"]
    train_args = ckpt.get("args", {}) or {}

    lstm_ih = state["lstm.weight_ih_l0"]
    hidden_size = int(lstm_ih.shape[0] // 4)
    bidirectional = "lstm.weight_ih_l0_reverse" in state
    num_layers = len(
        [k for k in state.keys() if k.startswith("lstm.weight_ih_l") and not k.endswith("_reverse")]
    )

    cfg = {
        "input_size": int(ckpt["input_size"]),
        "hidden_size": int(train_args.get("hidden_size", hidden_size)),
        "num_layers": int(train_args.get("num_layers", num_layers)),
        "dropout": float(train_args.get("dropout", 0.3)),
        "bidirectional": bool(train_args.get("bidirectional", bidirectional)),
        "num_classes": int(state["classifier.3.weight"].shape[0]),
        "attn_dim": int(state["attention.proj.weight"].shape[0]),
        "fc_hidden": int(state["classifier.0.weight"].shape[0]),
    }
    return cfg


def _load_model_from_ckpt(model_path: Path, device: torch.device):
    ckpt = torch.load(model_path, map_location=device)
    cfg = _infer_model_cfg(ckpt)

    model = AttentionLSTM(**cfg)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, ckpt, cfg


def _open_camera(camera_index: int) -> cv2.VideoCapture:
    if platform.system().lower().startswith("win"):
        return cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(camera_index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--norm_path", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=50)
    ap.add_argument("--target_fps", type=float, default=5.0)
    ap.add_argument(
        "--drift_th",
        type=float,
        default=None,
        help="If omitted, use best_threshold stored in checkpoint.",
    )
    ap.add_argument("--persist_n", type=int, default=3)
    ap.add_argument("--camera_index", type=int, default=0)
    ap.add_argument("--popup_cooldown_sec", type=float, default=8.0)

    ap.add_argument("--events_path", type=str, default="attention_events.jsonl")
    ap.add_argument("--student_id", type=str, default="Student A")
    ap.add_argument("--session_id", type=str, default="")

    ap.add_argument("--record_path", type=str, default="", help="Save anonymized MP4 here (no audio).")
    ap.add_argument(
        "--blur_mode",
        type=str,
        default="background",
        choices=["none", "face", "background", "person", "full"],
    )
    ap.add_argument("--blur_k", type=int, default=31)
    ap.add_argument("--seg_thr", type=float, default=0.5)

    args = ap.parse_args()

    events_path = Path(args.events_path)
    session_id = args.session_id.strip() or datetime.now().strftime("session_%Y%m%d_%H%M%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = load_norm(Path(args.norm_path))

    model, ckpt, model_cfg = _load_model_from_ckpt(Path(args.model_path), device)
    drift_th = float(ckpt.get("best_threshold", 0.3)) if args.drift_th is None else float(args.drift_th)
    input_size = int(model_cfg["input_size"])

    if mean.shape[0] != input_size or std.shape[0] != input_size:
        raise ValueError(
            f"Feature dimension mismatch: norm has {mean.shape[0]} dims, checkpoint expects {input_size}."
        )

    print(f"Device: {device}", flush=True)
    print(f"Loaded model_ns config: {model_cfg}", flush=True)
    print(f"Using drift threshold: {drift_th:.3f}", flush=True)
    print(f"Event log path: {events_path}", flush=True)
    print(f"Student ID: {args.student_id}", flush=True)
    print(f"Session ID: {session_id}", flush=True)

    extractor = MediaPipeFeatureExtractor(target_fps=args.target_fps)
    buffer = deque(maxlen=args.seq_len)
    drift_hits = 0
    ari_hist = deque(maxlen=5)

    popup = PopupManager()
    last_popup_t = -1e9
    last_alert_on = False
    last_logged_bucket = None

    mp_face = None
    mp_selfie = None
    if args.blur_mode == "face":
        mp_face = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6,
        )
    elif args.blur_mode in ("background", "person"):
        mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

    cap = _open_camera(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam (index {args.camera_index}).")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(src_fps) or src_fps < 1.0 or src_fps > 240.0:
        src_fps = 30.0
    frame_interval = max(1, int(round(src_fps / max(args.target_fps, 1e-6))))

    frame_count = 0
    writer = None
    out_fps = float(args.target_fps)

    t_sec = 0.0
    dt = 1.0 / args.target_fps

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1
            if frame_count % frame_interval != 0:
                continue

            frame_raw = frame
            ff = extractor.extract(frame_raw, t_sec=t_sec)
            x_raw = ff.x

            if x_raw.shape[0] != input_size:
                raise ValueError(
                    f"Live feature dimension mismatch: got {x_raw.shape[0]}, expected {input_size}."
                )

            # indices:
            # 13 eye_closed, 14 eye_drowsy, 15 eye_open, 16 yawning,
            # 17 head_forward, 18 body_stable, 19 face_ok, 20 pose_ok
            eye_closed = int(x_raw[13] > 0.5)
            eye_drowsy = int(x_raw[14] > 0.5)
            eye_open = int(x_raw[15] > 0.5)
            yawning = int(x_raw[16] > 0.5)
            head_forward = int(x_raw[17] > 0.5)
            body_stable = int(x_raw[18] > 0.5)
            face_ok = int(x_raw[19] > 0.5)
            pose_ok = int(x_raw[20] > 0.5)

            rule_ready = bool(face_ok and pose_ok)
            if rule_ready:
                att_score = eye_open + head_forward + body_stable
                rule_msg = rule_feedback(att_score)
            else:
                att_score = -1
                rule_msg = "Waiting for stable face/pose"

            x_norm = norm_impute(x_raw, mean, std)
            buffer.append(x_norm)

            drift_prob = None
            drift_state = "N/A"
            drift_ready = False

            if len(buffer) >= args.seq_len:
                drift_ready = True
                X = np.stack(list(buffer), axis=0)
                Xt = torch.from_numpy(X).unsqueeze(0).to(device)
                Lt = torch.tensor([len(buffer)], dtype=torch.long, device=device)

                with torch.no_grad():
                    logit = model(Xt, Lt).squeeze()
                    p = torch.sigmoid(logit).item()

                drift_prob = p
                drift_hits = drift_hits + 1 if p >= drift_th else 0
                drift_state = "DRIFT" if drift_hits >= args.persist_n else "OK"

            reason = get_attention_reason(
                eye_closed=eye_closed,
                eye_drowsy=eye_drowsy,
                yawning=yawning,
                head_forward=head_forward,
                body_stable=body_stable,
            )

            cause = get_dashboard_cause(
                eye_closed=eye_closed,
                eye_drowsy=eye_drowsy,
                yawning=yawning,
                head_forward=head_forward,
                body_stable=body_stable,
            )

            alert_on = False
            if drift_state == "DRIFT":
                alert_on = True
            elif rule_ready and att_score <= 1:
                alert_on = True

            if drift_ready and str(drift_state).upper() == "DRIFT":
                issue = "drift"
            elif rule_ready and att_score <= 1:
                issue = "rule"
            else:
                issue = "none"

            severity = compute_severity(
                rule_ready=rule_ready,
                att_score=att_score,
                drift_ready=drift_ready,
                drift_prob=drift_prob,
                drift_state=drift_state,
            )

            banner = None
            if alert_on:
                if drift_prob is not None and drift_prob >= 0.75:
                    banner = "HIGH ATTENTION RISK"
                else:
                    banner = "ATTENTION DROP"

                popup_due = (t_sec - last_popup_t) >= args.popup_cooldown_sec
                rising_edge = alert_on and not last_alert_on

                if popup_due or rising_edge:
                    msg = (
                        f"{banner}\n\n"
                        f"Reason: {reason}.\n"
                        f"Action: Refocus on the screen/content."
                    )
                    if drift_prob is not None:
                        msg += f"\nDrift probability: {drift_prob:.2f}"
                    popup.notify("AttentionTrack", msg)
                    last_popup_t = t_sec

            last_alert_on = alert_on

            ari_raw = compute_ari_raw(
                rule_ready=rule_ready,
                att_score=att_score,
                drift_ready=drift_ready,
                drift_prob=drift_prob,
            )
            if np.isfinite(ari_raw):
                ari_hist.append(float(ari_raw))
            ari_ma = float(np.mean(ari_hist)) if len(ari_hist) > 0 else float("nan")
            risk_level = risk_label(ari_ma)
            attention_pct = float(np.clip(100.0 - ari_ma, 0.0, 100.0)) if np.isfinite(ari_ma) else float("nan")

            message = None
            if alert_on and banner is not None:
                message = f"{banner}: {reason}. Refocus on the screen/content."

            current_bucket = int(t_sec)
            if current_bucket != last_logged_bucket:
                log_row = {
                    "event_time": datetime.now().isoformat(timespec="seconds"),
                    "student_id": args.student_id,
                    "session_id": session_id,
                    "t": float(t_sec),
                    "rule_ready": int(rule_ready),
                    "att_score": int(att_score),
                    "eye_closed": int(eye_closed),
                    "eye_drowsy": int(eye_drowsy),
                    "eye_open": int(eye_open),
                    "yawning": int(yawning),
                    "head_forward": int(head_forward),
                    "body_stable": int(body_stable),
                    "face_ok": int(face_ok),
                    "pose_ok": int(pose_ok),
                    "drift_ready": int(drift_ready),
                    "drift_prob": None if drift_prob is None else float(drift_prob),
                    "drift_state": str(drift_state),
                    "alert_emitted": int(alert_on),
                    "issue": issue,
                    "cause": cause,
                    "severity": int(severity),
                    "banner": banner,
                    "message": message,
                    "ari_raw": None if not np.isfinite(ari_raw) else float(ari_raw),
                    "ari_ma": None if not np.isfinite(ari_ma) else float(ari_ma),
                    "risk_level": risk_level,
                    "attention_pct": None if not np.isfinite(attention_pct) else float(attention_pct),
                }
                append_event_jsonl(events_path, log_row)
                last_logged_bucket = current_bucket

            frame_vis = frame.copy()
            frame_vis = apply_privacy_blur(
                frame_vis,
                args.blur_mode,
                args.blur_k,
                mp_face,
                mp_selfie,
                args.seg_thr,
            )

            overlay = frame_vis
            y0 = 30

            rule_color = (0, 255, 0) if (rule_ready and att_score >= 2) else (0, 255, 255)
            cv2.putText(
                overlay,
                f"Rule score: {att_score} | {rule_msg}",
                (20, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                rule_color,
                2,
            )

            if drift_prob is None:
                cv2.putText(
                    overlay,
                    f"LSTM: warming up ({len(buffer)}/{args.seq_len} frames)",
                    (20, y0 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    2,
                )
            else:
                lstm_color = (0, 255, 0) if drift_state == "OK" else (0, 255, 255)
                cv2.putText(
                    overlay,
                    f"LSTM drift prob: {drift_prob:.2f} | th: {drift_th:.2f} | state: {drift_state}",
                    (20, y0 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    lstm_color,
                    2,
                )

            cv2.putText(
                overlay,
                f"Reason: {reason}",
                (20, y0 + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (230, 230, 230),
                2,
            )

            if np.isfinite(ari_ma):
                cv2.putText(
                    overlay,
                    f"ARI: {ari_ma:.1f} | Risk: {risk_level}",
                    (20, y0 + 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (230, 230, 230),
                    2,
                )

            if alert_on and banner is not None:
                cv2.putText(
                    overlay,
                    banner,
                    (20, y0 + 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            if args.record_path:
                if writer is None:
                    h, w = overlay.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.record_path, fourcc, out_fps, (w, h))
                writer.write(overlay)

            cv2.imshow("AttentionTrack (Fast BiLSTM + Blur)", overlay)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

            t_sec += dt

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        extractor.close()
        if mp_face is not None:
            mp_face.close()
        if mp_selfie is not None:
            mp_selfie.close()


if __name__ == "__main__":
    main()