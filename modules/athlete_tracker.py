"""Athlete identity tracker and baseline calibration.

Responsibilities:
- Assign each PoseDetection to Left (L) or Right (R) athlete by bounding box X.
- Handle brief detection loss (carry forward last known data).
- Detect the static on-guard period and capture per-athlete baseline values.
- Each frame: return structured FrameData with angles, velocities, and baseline deltas.
"""

import logging
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import config
from modules.pose_module import PoseDetection
from utils.angle_utils import compute_angles, keypoint_max_velocity

logger = logging.getLogger(__name__)


@dataclass
class AthleteBaseline:
    elbow_angle:       float = math.nan
    shoulder_angle:    float = math.nan
    front_knee_angle:  float = math.nan
    rear_knee_angle:   float = math.nan
    trunk_lean:        float = math.nan
    stance_width:      float = math.nan
    com_x:             float = math.nan
    wrist_height:      float = math.nan
    captured:          bool  = False

    def capture(self, angles: dict) -> None:
        self.elbow_angle      = angles["elbow_angle"]
        self.shoulder_angle   = angles["shoulder_angle"]
        self.front_knee_angle = angles["front_knee_angle"]
        self.rear_knee_angle  = angles["rear_knee_angle"]
        self.trunk_lean       = angles["trunk_lean"]
        self.stance_width     = angles["stance_width"]
        self.com_x            = angles["com_x"]
        self.wrist_height     = angles["wrist_height"]
        self.captured         = True

    def quality_check(self, side: str) -> None:
        for attr, lo, hi, label in [
            ("elbow_angle",      config.BASELINE_ELBOW_MIN, config.BASELINE_ELBOW_MAX, "elbow"),
            ("front_knee_angle", config.BASELINE_KNEE_MIN,  config.BASELINE_KNEE_MAX,  "front knee"),
            ("rear_knee_angle",  config.BASELINE_KNEE_MIN,  config.BASELINE_KNEE_MAX,  "rear knee"),
        ]:
            val = getattr(self, attr)
            if math.isnan(val) or not (lo <= val <= hi):
                logger.warning(
                    "Baseline quality warning — %s athlete %s=%.1f° outside [%d, %d]°. "
                    "Keypoint detection may have failed.",
                    side, label, val, lo, hi,
                )


@dataclass
class AthleteFrame:
    """Per-frame data for one athlete."""
    detected:        bool  = False
    keypoints:       np.ndarray | None = None   # (17, 3)
    angles:          dict  = field(default_factory=dict)
    com_velocity:    float = 0.0   # px/frame, positive = moving right
    front_ankle_vel: float = 0.0   # px/frame, absolute displacement
    rear_ankle_vel:  float = 0.0
    kp_max_velocity: float = 0.0   # max across all 17 keypoints
    miss_frames:     int   = 0     # how many frames carried forward


@dataclass
class FrameData:
    frame_idx: int
    L: AthleteFrame = field(default_factory=AthleteFrame)
    R: AthleteFrame = field(default_factory=AthleteFrame)
    baseline_captured: bool = False


class AthleteTracker:
    """Tracks both athletes across all frames of a video."""

    def __init__(self):
        self.baseline_L = AthleteBaseline()
        self.baseline_R = AthleteBaseline()

        self._prev_kp_L: np.ndarray | None = None
        self._prev_kp_R: np.ndarray | None = None
        self._prev_com_L: float | None = None
        self._prev_com_R: float | None = None
        self._prev_front_ankle_L: float | None = None
        self._prev_front_ankle_R: float | None = None
        self._prev_rear_ankle_L: float | None = None
        self._prev_rear_ankle_R: float | None = None

        self._miss_L: int = 0
        self._miss_R: int = 0
        self._last_det_L: PoseDetection | None = None
        self._last_det_R: PoseDetection | None = None

        # Rolling window of angle dicts used for baseline stability check
        self._still_angles_L: deque[dict] = deque(maxlen=config.BASELINE_MIN_FRAMES)
        self._still_angles_R: deque[dict] = deque(maxlen=config.BASELINE_MIN_FRAMES)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def update(self, frame_idx: int, detections: list[PoseDetection]) -> FrameData:
        """Process one frame and return structured FrameData."""
        det_L, det_R = self._assign(detections)
        data = FrameData(frame_idx=frame_idx)

        data.L = self._build_athlete_frame(det_L, "left",  self._prev_kp_L,
                                           self._prev_com_L,
                                           self._prev_front_ankle_L,
                                           self._prev_rear_ankle_L,
                                           self._miss_L)
        data.R = self._build_athlete_frame(det_R, "right", self._prev_kp_R,
                                           self._prev_com_R,
                                           self._prev_front_ankle_R,
                                           self._prev_rear_ankle_R,
                                           self._miss_R)

        # Update miss counters
        if det_L is None:
            self._miss_L += 1
        else:
            self._miss_L = 0
            self._last_det_L = det_L

        if det_R is None:
            self._miss_R += 1
        else:
            self._miss_R = 0
            self._last_det_R = det_R

        # Store previous keypoints for velocity computation next frame
        if data.L.keypoints is not None:
            self._prev_kp_L = data.L.keypoints
            self._prev_com_L = data.L.angles.get("com_x")
            self._prev_front_ankle_L = data.L.angles.get("front_ankle_x")
            self._prev_rear_ankle_L  = data.L.angles.get("rear_ankle_x")

        if data.R.keypoints is not None:
            self._prev_kp_R = data.R.keypoints
            self._prev_com_R = data.R.angles.get("com_x")
            self._prev_front_ankle_R = data.R.angles.get("front_ankle_x")
            self._prev_rear_ankle_R  = data.R.angles.get("rear_ankle_x")

        # Baseline calibration
        if not (self.baseline_L.captured and self.baseline_R.captured):
            self._try_capture_baseline(data)

        data.baseline_captured = self.baseline_L.captured and self.baseline_R.captured
        return data

    def reset(self) -> None:
        """Reset all state for a new video."""
        self.__init__()

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _assign(self, detections: list[PoseDetection]) -> tuple:
        """Split detections into (left_det, right_det).

        Detections are sorted by center_x. With two people on screen the
        leftmost is L and the rightmost is R. Fewer than two detections →
        fall back to carry-forward for the missing side.
        """
        if len(detections) >= 2:
            return detections[0], detections[-1]

        if len(detections) == 1:
            det = detections[0]
            # Determine which side this single detection belongs to
            if self._last_det_L is not None and self._last_det_R is not None:
                dist_L = abs(det.center_x - self._last_det_L.center_x)
                dist_R = abs(det.center_x - self._last_det_R.center_x)
                if dist_L < dist_R:
                    return det, None
                else:
                    return None, det
            # No history yet — put on whichever side center_x suggests
            frame_mid = det.center_x  # rough guess
            return (det, None) if frame_mid < 500 else (None, det)

        return None, None

    def _build_athlete_frame(
        self,
        det: PoseDetection | None,
        side: str,
        prev_kp,
        prev_com,
        prev_fa,
        prev_ra,
        miss_count: int,
    ) -> AthleteFrame:
        af = AthleteFrame()

        # Carry forward if detection missing and within tolerance
        if det is None:
            last = self._last_det_L if side == "left" else self._last_det_R
            if last is not None and miss_count < config.TRACKER_MAX_MISS_FRAMES:
                det = last
                af.miss_frames = miss_count + 1
            else:
                af.detected = False
                return af

        af.detected   = True
        af.keypoints  = det.keypoints
        af.angles     = det.compute_angles(side)

        # Keypoint max velocity
        if prev_kp is not None:
            af.kp_max_velocity = keypoint_max_velocity(prev_kp, det.keypoints)

        # CoM velocity (signed: positive = moving right on screen)
        if prev_com is not None and af.angles.get("com_x") is not None:
            af.com_velocity = af.angles["com_x"] - prev_com

        # Ankle velocities (absolute displacement per frame)
        if prev_fa is not None and af.angles.get("front_ankle_x") is not None:
            af.front_ankle_vel = abs(af.angles["front_ankle_x"] - prev_fa)
        if prev_ra is not None and af.angles.get("rear_ankle_x") is not None:
            af.rear_ankle_vel = abs(af.angles["rear_ankle_x"] - prev_ra)

        return af

    def _try_capture_baseline(self, data: FrameData) -> None:
        """Accumulate angle history; capture baseline when window is stable.

        Stability criteria (all must pass over the last BASELINE_MIN_FRAMES frames):
          1. Elbow angle mean is within BASELINE_ELBOW_RANGE.
          2. Knee angle mean is within BASELINE_KNEE_RANGE.
          3. Std-dev of elbow angle < BASELINE_STABLE_STD.
          4. Std-dev of front knee angle < BASELINE_STABLE_STD.
        """
        if data.L.detected:
            self._still_angles_L.append(data.L.angles)
        if data.R.detected:
            self._still_angles_R.append(data.R.angles)

        if not self.baseline_L.captured and _is_stable(self._still_angles_L):
            avg = _average_angles(list(self._still_angles_L))
            self.baseline_L.capture(avg)
            self.baseline_L.quality_check("Left")
            logger.info("Baseline captured for LEFT athlete: elbow=%.1f° front_knee=%.1f°",
                        self.baseline_L.elbow_angle, self.baseline_L.front_knee_angle)

        if not self.baseline_R.captured and _is_stable(self._still_angles_R):
            avg = _average_angles(list(self._still_angles_R))
            self.baseline_R.capture(avg)
            self.baseline_R.quality_check("Right")
            logger.info("Baseline captured for RIGHT athlete: elbow=%.1f° front_knee=%.1f°",
                        self.baseline_R.elbow_angle, self.baseline_R.front_knee_angle)


def _is_stable(window: deque) -> bool:
    """Return True if the angle window satisfies baseline stability criteria."""
    if len(window) < config.BASELINE_MIN_FRAMES:
        return False
    elbows = [d.get("elbow_angle",      float("nan")) for d in window]
    knees  = [d.get("front_knee_angle", float("nan")) for d in window]
    elbows = [v for v in elbows if not math.isnan(v)]
    knees  = [v for v in knees  if not math.isnan(v)]
    if len(elbows) < config.BASELINE_MIN_FRAMES // 2:
        return False
    if len(knees)  < config.BASELINE_MIN_FRAMES // 2:
        return False
    e_mean = float(np.mean(elbows))
    k_mean = float(np.mean(knees))
    e_std  = float(np.std(elbows))
    k_std  = float(np.std(knees))
    e_lo, e_hi = config.BASELINE_ELBOW_RANGE
    k_lo, k_hi = config.BASELINE_KNEE_RANGE
    return (
        e_lo <= e_mean <= e_hi and
        k_lo <= k_mean <= k_hi and
        e_std < config.BASELINE_STABLE_STD and
        k_std < config.BASELINE_STABLE_STD
    )


def _average_angles(angle_list: list[dict]) -> dict:
    """Average a list of angle dicts, ignoring NaN values."""
    keys = angle_list[0].keys()
    result = {}
    for k in keys:
        vals = [d[k] for d in angle_list if not math.isnan(d.get(k, math.nan))]
        result[k] = float(np.mean(vals)) if vals else math.nan
    return result
