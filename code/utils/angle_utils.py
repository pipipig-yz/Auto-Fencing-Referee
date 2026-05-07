"""Joint angle computation from COCO keypoints."""

import numpy as np


# COCO-17 keypoint indices
KP = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}


def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at point b formed by vectors b→a and b→c, in degrees."""
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return float("nan")
    cos_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _kp(keypoints: np.ndarray, name: str) -> np.ndarray:
    """Return (x, y) for a named keypoint (drops confidence dimension)."""
    return keypoints[KP[name], :2]


def compute_angles(keypoints: np.ndarray, side: str) -> dict:
    """Compute all relevant joint angles for one athlete.

    Args:
        keypoints: (17, 3) array — x, y, confidence per keypoint.
        side: "left" or "right" — which screen side this athlete is on.

    Both fencers are right-handed sabre fencers, so:
    - Sword arm = anatomical RIGHT arm for both athletes.
    - Front leg  = the leg whose ankle is closer to the opponent.
                   Left athlete (faces right)  → larger ankle X = front.
                   Right athlete (faces left)  → smaller ankle X = front.
    Using X-position for front/rear leg avoids relying on YOLO correctly
    labelling anatomical sides in side-view footage.
    """
    # Sword arm — anatomical right for both (both are right-handed)
    sword_shoulder = _kp(keypoints, "right_shoulder")
    sword_elbow    = _kp(keypoints, "right_elbow")
    sword_wrist    = _kp(keypoints, "right_wrist")
    sword_hip      = _kp(keypoints, "right_hip")

    # Front / rear leg — determined by X-position relative to opponent side
    r_ankle = _kp(keypoints, "right_ankle")
    l_ankle = _kp(keypoints, "left_ankle")
    r_knee  = _kp(keypoints, "right_knee")
    l_knee  = _kp(keypoints, "left_knee")
    r_hip   = _kp(keypoints, "right_hip")
    l_hip   = _kp(keypoints, "left_hip")

    if side == "left":
        # Faces right → front ankle has larger X
        if r_ankle[0] >= l_ankle[0]:
            front_ankle, rear_ankle = r_ankle, l_ankle
            front_knee,  rear_knee  = r_knee,  l_knee
            front_hip,   rear_hip   = r_hip,   l_hip
        else:
            front_ankle, rear_ankle = l_ankle, r_ankle
            front_knee,  rear_knee  = l_knee,  r_knee
            front_hip,   rear_hip   = l_hip,   r_hip
    else:
        # Faces left → front ankle has smaller X
        if r_ankle[0] <= l_ankle[0]:
            front_ankle, rear_ankle = r_ankle, l_ankle
            front_knee,  rear_knee  = r_knee,  l_knee
            front_hip,   rear_hip   = r_hip,   l_hip
        else:
            front_ankle, rear_ankle = l_ankle, r_ankle
            front_knee,  rear_knee  = l_knee,  r_knee
            front_hip,   rear_hip   = l_hip,   r_hip

    elbow_angle      = angle_between(sword_shoulder, sword_elbow, sword_wrist)
    shoulder_angle   = angle_between(sword_hip, sword_shoulder, sword_elbow)
    front_knee_angle = angle_between(front_hip, front_knee, front_ankle)
    rear_knee_angle  = angle_between(rear_hip,  rear_knee,  rear_ankle)

    # Angle between the two thighs, measured at the hip midpoint.
    # In a lunge the legs spread wide, pushing this angle well above 90°.
    hip_mid = (front_hip + rear_hip) / 2.0
    inter_thigh_angle = angle_between(front_knee, hip_mid, rear_knee)

    # Trunk lean: angle of the line (rear_hip → front_hip) relative to vertical
    hip_vec = front_hip - rear_hip
    vertical = np.array([0.0, -1.0])  # up in image coords = negative Y
    norm = np.linalg.norm(hip_vec)
    if norm > 1e-6:
        cos_t = np.clip(np.dot(hip_vec / norm, vertical), -1.0, 1.0)
        trunk_lean = float(np.degrees(np.arccos(cos_t)))
    else:
        trunk_lean = float("nan")

    left_hip_pt  = _kp(keypoints, "left_hip")
    right_hip_pt = _kp(keypoints, "right_hip")
    com_x = float((left_hip_pt[0] + right_hip_pt[0]) / 2.0)

    stance_width = float(abs(front_ankle[0] - rear_ankle[0]))

    # Wrist height relative to sword-side hip (positive = wrist above hip)
    wrist_height = float(sword_hip[1] - sword_wrist[1])  # Y increases downward

    return {
        "elbow_angle":       elbow_angle,
        "shoulder_angle":    shoulder_angle,
        "front_knee_angle":  front_knee_angle,
        "rear_knee_angle":   rear_knee_angle,
        "inter_thigh_angle": inter_thigh_angle,
        "trunk_lean":        trunk_lean,
        "stance_width":      stance_width,
        "com_x":             com_x,
        "wrist_height":      wrist_height,
        "front_ankle_x":     float(front_ankle[0]),
        "rear_ankle_x":      float(rear_ankle[0]),
    }


def keypoint_max_velocity(kp_prev: np.ndarray, kp_curr: np.ndarray) -> float:
    """Max displacement (px/frame) across all 17 keypoints between two frames."""
    displacements = np.linalg.norm(kp_curr[:, :2] - kp_prev[:, :2], axis=1)
    return float(np.max(displacements))
