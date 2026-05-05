import numpy as np
import cv2

# -----------------------------
# Landmarks (MediaPipe FaceMesh)
# -----------------------------
# EAR eye landmark sets (commonly used with FaceMesh)
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]   # p1..p6
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]  # p1..p6

# MAR mouth landmarks (inner/outer mix to approximate yawning)
MOUTH_CORNER_L = 61
MOUTH_CORNER_R = 291
MOUTH_PAIR_1_U, MOUTH_PAIR_1_L = 13, 14
MOUTH_PAIR_2_U, MOUTH_PAIR_2_L = 81, 178
MOUTH_PAIR_3_U, MOUTH_PAIR_3_L = 311, 402

# Head pose indices (same as your original script)
FACE_3D_INDICES = [33, 263, 1, 61, 291, 199]

# Pose indices (MediaPipe Pose)
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16

# -----------------------------
# Thresholds (as in your rule-based prototype)
# -----------------------------
EAR_CLOSED_TH = 0.2
EAR_DROWSY_TH_LOW = 0.2
EAR_DROWSY_TH_HIGH = 0.3
EAR_OPEN_TH = 0.3

MAR_YAWN_TH = 0.8  # MAR >= 0.8 => yawning

# SRL-oriented head-pose rule (paper text): abs(pitch/yaw) > 10° => looking around
HEAD_PITCH_YAW_TH = 10.0


def _pt(lm, idx, w, h):
    """Return (x,y) pixel coords from normalized landmark."""
    return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float32)


def eye_aspect_ratio(lm, eye_idx, w, h):
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
    eye_idx must map [p1,p2,p3,p4,p5,p6]
    """
    p1 = _pt(lm, eye_idx[0], w, h)
    p2 = _pt(lm, eye_idx[1], w, h)
    p3 = _pt(lm, eye_idx[2], w, h)
    p4 = _pt(lm, eye_idx[3], w, h)
    p5 = _pt(lm, eye_idx[4], w, h)
    p6 = _pt(lm, eye_idx[5], w, h)

    vert1 = np.linalg.norm(p2 - p6)
    vert2 = np.linalg.norm(p3 - p5)
    horiz = np.linalg.norm(p1 - p4)
    if horiz == 0:
        return 0.0
    return (vert1 + vert2) / (2.0 * horiz)


def mouth_aspect_ratio(lm, w, h):
    """
    Simple MAR suitable for yawning:
    MAR = (d(13,14) + d(81,178) + d(311,402)) / (3 * d(61,291))
    """
    left = _pt(lm, MOUTH_CORNER_L, w, h)
    right = _pt(lm, MOUTH_CORNER_R, w, h)

    u1 = _pt(lm, MOUTH_PAIR_1_U, w, h)
    l1 = _pt(lm, MOUTH_PAIR_1_L, w, h)
    u2 = _pt(lm, MOUTH_PAIR_2_U, w, h)
    l2 = _pt(lm, MOUTH_PAIR_2_L, w, h)
    u3 = _pt(lm, MOUTH_PAIR_3_U, w, h)
    l3 = _pt(lm, MOUTH_PAIR_3_L, w, h)

    horiz = np.linalg.norm(left - right)
    if horiz == 0:
        return 0.0

    v = (np.linalg.norm(u1 - l1) + np.linalg.norm(u2 - l2) + np.linalg.norm(u3 - l3)) / 3.0
    return v / horiz


def classify_ear(ear: float) -> int:
    """Return {0:closed, 1:drowsy, 2:open}."""
    if ear < EAR_CLOSED_TH:
        return 0
    if EAR_DROWSY_TH_LOW <= ear <= EAR_DROWSY_TH_HIGH:
        return 1
    if ear > EAR_OPEN_TH:
        return 2
    return 1


def classify_mar(mar: float) -> int:
    """Return {0:not yawning, 1:yawning}."""
    return 1 if mar >= MAR_YAWN_TH else 0


def compute_head_pose_angles(lm, w, h):
    """
    Estimate pitch/yaw/roll in degrees using solvePnP + RQDecomp3x3.
    Returns (pitch, yaw, roll) as floats.
    """
    face_2d, face_3d = [], []
    for idx in FACE_3D_INDICES:
        x = lm[idx].x * w
        y = lm[idx].y * h
        z = lm[idx].z * w
        face_2d.append([x, y])
        face_3d.append([x, y, z])

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    cam_matrix = np.array([[w, 0, w / 2],
                           [0, w, h / 2],
                           [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    yaw, pitch, roll = float(angles[1]), float(angles[0]), float(angles[2])
    return pitch, yaw, roll


def head_forward_rule(pitch: float, yaw: float, th_deg: float = HEAD_PITCH_YAW_TH) -> int:
    """Return 1 if within +-th_deg on both pitch and yaw, else 0 (looking around)."""
    return 1 if (abs(pitch) <= th_deg and abs(yaw) <= th_deg) else 0


def one_hot(idx: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    if 0 <= idx < n:
        v[idx] = 1.0
    return v
