from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np

from . import rules


@dataclass
class FrameFeatures:
    x: np.ndarray               # (F,)
    meta: Dict[str, float]      # scalar diagnostics (optional)


class MediaPipeFeatureExtractor:
    """
    Extracts interpretable multimodal features from a single BGR frame:
    - Eye: EAR + eye_state (one-hot)
    - Mouth: MAR + yawning flag
    - Head pose: pitch/yaw/roll + forward/looking-around rule
    - Gaze proxy: normalized iris position within each eye (if refine_landmarks is available)
    - Body motion: shoulder/wrist motion magnitude + stable/fidget heuristic

    Notes:
      - This module is designed so you can re-use exactly the same feature logic for:
        (i) offline DAiSEE feature extraction and (ii) live webcam inference.
      - Missing face/pose detections are represented as NaNs for continuous features and 0 for rule bits.
        Training code should normalize with NaN-safe statistics + imputation.
    """

    def __init__(
        self,
        target_fps: float = 5.0,
        blink_window_sec: float = 3.0,
        motion_th_px: float = 20.0,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        refine_landmarks: bool = True,
    ):
        self.target_fps = float(target_fps)
        self.blink_window_sec = float(blink_window_sec)
        self.motion_th_px = float(motion_th_px)

        import mediapipe as mp  # heavy import; keep here

        self._mp_face_mesh = mp.solutions.face_mesh
        self._mp_pose = mp.solutions.pose

        self.face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.pose = self._mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # stateful trackers (for blink and motion)
        self._prev_ear: Optional[float] = None
        self._ear_below: bool = False
        self._blink_times: deque = deque()  # timestamps (seconds)

        self._prev_shoulders: Optional[np.ndarray] = None  # (4,) [lx,ly,rx,ry]
        self._prev_wrists: Optional[np.ndarray] = None     # (4,) [lx,ly,rx,ry]

    def close(self):
        # mediapipe resources
        if self.face_mesh:
            self.face_mesh.close()
        if self.pose:
            self.pose.close()

    @staticmethod
    def feature_names() -> List[str]:
        return [
            # continuous face
            "ear", "mar", "pitch", "yaw", "roll",
            # gaze proxy (continuous)
            "gaze_lx", "gaze_ly", "gaze_rx", "gaze_ry",
            # motion (continuous)
            "shoulder_motion", "wrist_motion",
            # blink dynamics
            "blink_event", "blink_rate",
            # rule bits (engineered semantic encoding)
            "eye_closed", "eye_drowsy", "eye_open",
            "yawning",
            "head_forward",
            "body_stable",
            # detection flags
            "face_ok", "pose_ok",
        ]

    def extract(self, frame_bgr: np.ndarray, t_sec: float) -> FrameFeatures:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        face_res = self.face_mesh.process(rgb)
        pose_res = self.pose.process(rgb)

        # defaults
        ear = np.nan
        mar = np.nan
        pitch = np.nan
        yaw = np.nan
        roll = np.nan
        gaze_lx = gaze_ly = gaze_rx = gaze_ry = np.nan
        eye_state = None
        yawning = 0
        head_forward = 0
        face_ok = 0

        shoulder_motion = np.nan
        wrist_motion = np.nan
        body_stable = 0
        pose_ok = 0

        # -------------------
        # FACE features
        # -------------------
        if face_res.multi_face_landmarks:
            face_ok = 1
            lm = face_res.multi_face_landmarks[0].landmark

            # EAR
            ear_left = rules.eye_aspect_ratio(lm, rules.LEFT_EYE_IDX, w, h)
            ear_right = rules.eye_aspect_ratio(lm, rules.RIGHT_EYE_IDX, w, h)
            ear = float((ear_left + ear_right) / 2.0)
            eye_state = rules.classify_ear(ear)

            # MAR
            mar = float(rules.mouth_aspect_ratio(lm, w, h))
            yawning = rules.classify_mar(mar)

            # head pose
            angles = rules.compute_head_pose_angles(lm, w, h)
            if angles is not None:
                pitch, yaw, roll = angles
                head_forward = rules.head_forward_rule(pitch, yaw, rules.HEAD_PITCH_YAW_TH)

            # gaze proxy (if iris landmarks exist)
            # MediaPipe FaceMesh (refine_landmarks=True) commonly provides 478 landmarks,
            # with right iris = 469..472, left iris = 474..477.
            if len(lm) >= 478:
                gaze_lx, gaze_ly = self._gaze_proxy(lm, w, h, left=True)
                gaze_rx, gaze_ry = self._gaze_proxy(lm, w, h, left=False)

        # -------------------
        # POSE motion features
        # -------------------
        if pose_res.pose_landmarks:
            pose_ok = 1
            plm = pose_res.pose_landmarks.landmark

            shoulders = self._pose_points(plm, w, h, rules.POSE_LEFT_SHOULDER, rules.POSE_RIGHT_SHOULDER)
            wrists = self._pose_points(plm, w, h, rules.POSE_LEFT_WRIST, rules.POSE_RIGHT_WRIST)

            shoulder_motion = self._motion_mag(self._prev_shoulders, shoulders)
            wrist_motion = self._motion_mag(self._prev_wrists, wrists)

            self._prev_shoulders = shoulders
            self._prev_wrists = wrists

            # simple stability heuristic
            if np.isfinite(shoulder_motion) and shoulder_motion < self.motion_th_px:
                body_stable = 1

        # -------------------
        # blink dynamics
        # -------------------
        blink_event = 0.0
        if np.isfinite(ear):
            blink_event = float(self._update_blink(ear, t_sec))
        blink_rate = float(self._blink_rate(t_sec))

        # -------------------
        # pack features
        # -------------------
        eye_oh = rules.one_hot(eye_state if eye_state is not None else -1, 3)  # (3,)

        x = np.array(
            [
                ear, mar, pitch, yaw, roll,
                gaze_lx, gaze_ly, gaze_rx, gaze_ry,
                shoulder_motion, wrist_motion,
                blink_event, blink_rate,
                eye_oh[0], eye_oh[1], eye_oh[2],
                float(yawning),
                float(head_forward),
                float(body_stable),
                float(face_ok), float(pose_ok),
            ],
            dtype=np.float32,
        )

        meta = {
            "face_ok": float(face_ok),
            "pose_ok": float(pose_ok),
        }
        return FrameFeatures(x=x, meta=meta)

    # -------------------
    # internal helpers
    # -------------------
    @staticmethod
    def _pose_points(plm, w, h, left_idx: int, right_idx: int) -> np.ndarray:
        lx = float(plm[left_idx].x * w)
        ly = float(plm[left_idx].y * h)
        rx = float(plm[right_idx].x * w)
        ry = float(plm[right_idx].y * h)
        return np.array([lx, ly, rx, ry], dtype=np.float32)

    @staticmethod
    def _motion_mag(prev: Optional[np.ndarray], cur: np.ndarray) -> float:
        if prev is None:
            return float("nan")
        return float(np.linalg.norm(cur - prev))

    def _update_blink(self, ear: float, t_sec: float) -> int:
        # hysteresis around EAR_CLOSED_TH
        if ear < rules.EAR_CLOSED_TH and not self._ear_below:
            self._ear_below = True
        if ear >= rules.EAR_OPEN_TH and self._ear_below:
            # blink completed
            self._ear_below = False
            self._blink_times.append(t_sec)
            return 1
        return 0

    def _blink_rate(self, t_sec: float) -> float:
        # keep only last window
        win = self.blink_window_sec
        while self._blink_times and (t_sec - self._blink_times[0]) > win:
            self._blink_times.popleft()
        return len(self._blink_times) / win if win > 0 else 0.0

    @staticmethod
    def _gaze_proxy(lm, w, h, left: bool) -> Tuple[float, float]:
        # iris indices
        if left:
            iris = [474, 475, 476, 477]
            corner_l, corner_r = 33, 133
            lid_u, lid_l = 159, 145
        else:
            iris = [469, 470, 471, 472]
            corner_l, corner_r = 362, 263
            lid_u, lid_l = 386, 374

        iris_xy = np.mean([rules._pt(lm, i, w, h) for i in iris], axis=0)
        c_l = rules._pt(lm, corner_l, w, h)
        c_r = rules._pt(lm, corner_r, w, h)
        u = rules._pt(lm, lid_u, w, h)
        l = rules._pt(lm, lid_l, w, h)

        # normalized within-eye coordinates (0..1 roughly)
        dx = float(c_r[0] - c_l[0])
        dy = float(l[1] - u[1])
        gx = float((iris_xy[0] - c_l[0]) / dx) if dx != 0 else float("nan")
        gy = float((iris_xy[1] - u[1]) / dy) if dy != 0 else float("nan")
        return gx, gy
