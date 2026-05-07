"""Piste (fencing strip) detection and athlete filtering.

Responsibilities:
- Detect the piste bounding box from a video frame using HSV colour thresholding.
- Filter YOLO athlete detections to only those standing on the piste.
- Among piste athletes, return the two largest detections (by bounding box area).
- Save a debug image showing the detected piste and filtered athletes.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from modules.pose_module import PoseDetection

logger = logging.getLogger(__name__)

# HSV range for white / silver (high value, low saturation)
_PISTE_S_MAX  = 60    # max saturation
_PISTE_V_MIN  = 160   # min value (brightness)

# Minimum fraction of frame width the piste candidate must span
_PISTE_MIN_WIDTH_RATIO  = 0.40
# The piste height is usually much smaller than its width
_PISTE_MAX_ASPECT_RATIO = 0.25   # height / width must be below this


class PisteDetector:
    """Detects the piste region and filters athlete detections accordingly."""

    def __init__(self):
        self._piste_box: tuple[int, int, int, int] | None = None  # (x, y, w, h)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    @property
    def piste_box(self) -> tuple[int, int, int, int] | None:
        """Last detected piste bounding box (x, y, w, h), or None."""
        return self._piste_box

    def detect_piste(self, frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        """Detect the piste in a single frame.

        Updates and returns self._piste_box as (x, y, w, h).
        Returns None if no suitable region found.
        """
        h, w = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # Mask: low saturation + high value = white / silver
        mask = cv2.inRange(
            hsv,
            np.array([0,           0, _PISTE_V_MIN]),
            np.array([180, _PISTE_S_MAX,         255]),
        )

        # Morphological cleanup to connect fragmented piste surface
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_h)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3)))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.warning("Piste detection: no white/silver contours found.")
            return None

        best = None
        best_area = 0
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw < w * _PISTE_MIN_WIDTH_RATIO:
                continue
            if ch / cw > _PISTE_MAX_ASPECT_RATIO:
                continue
            area = cw * ch
            if area > best_area:
                best_area = area
                best = (x, y, cw, ch)

        if best is None:
            logger.warning("Piste detection: no candidate passed shape filters.")
            return None

        self._piste_box = best
        logger.info("Piste detected: x=%d y=%d w=%d h=%d", *best)
        return best

    def filter_athletes(
        self,
        detections: list[PoseDetection],
        piste_box: tuple[int, int, int, int] | None = None,
    ) -> list[PoseDetection]:
        """Return up to 2 athletes standing on the piste, largest first.

        Falls back to the 2 largest detections overall if no piste is known
        or no detections land on the piste.
        """
        box = piste_box or self._piste_box

        if box is not None:
            px, py, pw, ph = box
            piste_y_min = py
            piste_y_max = py + ph + int(ph * 0.5)  # allow feet slightly below piste edge

            on_piste = [
                d for d in detections
                if _foot_y(d) >= piste_y_min and _foot_y(d) <= piste_y_max
            ]
        else:
            on_piste = detections

        if not on_piste:
            on_piste = detections  # fallback: use all

        # Sort by bounding box area descending; keep the 2 largest
        on_piste.sort(key=lambda d: _box_area(d), reverse=True)
        result = on_piste[:2]

        # Restore left-to-right order
        result.sort(key=lambda d: d.center_x)
        return result

    # ------------------------------------------------------------------ #
    #  Debug visualisation                                                 #
    # ------------------------------------------------------------------ #

    def save_debug_image(
        self,
        frame_bgr: np.ndarray,
        detections: list[PoseDetection],
        filtered: list[PoseDetection],
        out_path: str | Path,
    ) -> None:
        """Draw piste box + all detections + filtered athletes, save to file."""
        out = frame_bgr.copy()

        # Draw piste region
        if self._piste_box is not None:
            px, py, pw, ph = self._piste_box
            cv2.rectangle(out, (px, py), (px + pw, py + ph), (0, 255, 255), 3)
            cv2.putText(out, "PISTE", (px + 5, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw all raw detections (grey)
        for det in detections:
            x1, y1, x2, y2 = map(int, det.box_xyxy)
            cv2.rectangle(out, (x1, y1), (x2, y2), (160, 160, 160), 1)

        # Draw filtered athletes (blue = left, red = right)
        colors = [(255, 80, 80), (80, 80, 255)]
        labels = ["L", "R"]
        for i, det in enumerate(filtered):
            x1, y1, x2, y2 = map(int, det.box_xyxy)
            col = colors[i] if i < len(colors) else (0, 255, 0)
            lbl = labels[i] if i < len(labels) else str(i)
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 3)
            cv2.putText(out, lbl, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
            # Mark foot position
            fy = int(_foot_y(det))
            fx = int(det.center_x)
            cv2.circle(out, (fx, fy), 6, col, -1)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), out)
        logger.info("Piste debug image saved: %s", out_path)


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _foot_y(det: PoseDetection) -> float:
    """Y coordinate of the lower edge of the bounding box (approximate foot level)."""
    return float(det.box_xyxy[3])


def _box_area(det: PoseDetection) -> float:
    x1, y1, x2, y2 = det.box_xyxy
    return float((x2 - x1) * (y2 - y1))
