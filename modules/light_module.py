"""Scoring apparatus light detection.

Detects red / green light blobs in each frame using HSV colour thresholding.
Assigns blobs to Left or Right side by X position.
Requires a blob to persist for LIGHT_CONFIRM_FRAMES consecutive frames
before emitting a confirmed light event.

Light events emitted:
    LIGHT_ON_L    — left apparatus light confirmed on
    LIGHT_ON_R    — right apparatus light confirmed on
    LIGHT_ON_BOTH — both lights confirmed on (closes analysis window)
    LIGHT_OFF     — all lights extinguished
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


class LightEvent(str, Enum):
    LIGHT_ON_L    = "LIGHT_ON_L"
    LIGHT_ON_R    = "LIGHT_ON_R"
    LIGHT_ON_BOTH = "LIGHT_ON_BOTH"
    LIGHT_OFF     = "LIGHT_OFF"
    WINDOW_CLOSE  = "WINDOW_CLOSE"   # emitted LIGHT_WINDOW_CLOSE_DELAY frames after LIGHT_ON_BOTH


@dataclass
class LightState:
    """Tracks confirmation state for one side."""
    pending_frames: int = 0    # consecutive frames blob has been seen
    confirmed: bool     = False

    def see(self) -> bool:
        """Call when blob is detected this frame. Returns True on confirmation."""
        self.pending_frames += 1
        if not self.confirmed and self.pending_frames >= config.LIGHT_CONFIRM_FRAMES:
            self.confirmed = True
            return True
        return False

    def miss(self) -> bool:
        """Call when blob is NOT detected this frame. Returns True if light just went off."""
        was_confirmed = self.confirmed
        self.pending_frames = 0
        self.confirmed = False
        return was_confirmed   # True = light was on, now off


@dataclass
class FrameLightResult:
    frame_idx:  int
    left_on:    bool = False
    right_on:   bool = False
    events:     list[LightEvent] = field(default_factory=list)

    @property
    def both_on(self) -> bool:
        return self.left_on and self.right_on


class LightModule:
    """Per-frame light detection with confirmation logic."""

    def __init__(self, frame_width: int):
        """
        Args:
            frame_width: pixel width of the video frame,
                         used to split left / right halves.
        """
        self._mid_x          = frame_width // 2
        self._state_L        = LightState()
        self._state_R        = LightState()
        self._both_on_frame: int | None = None   # frame when LIGHT_ON_BOTH was emitted
        self._window_closed  = False
        self._history: list[FrameLightResult] = []

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def update(self, frame_idx: int, frame_bgr: np.ndarray) -> FrameLightResult:
        """Process one frame and return light state + any new events."""
        result = FrameLightResult(frame_idx=frame_idx)

        blob_L, blob_R = self._detect_blobs(frame_bgr)

        # Left side
        if blob_L:
            just_confirmed = self._state_L.see()
            if just_confirmed:
                result.events.append(LightEvent.LIGHT_ON_L)
                logger.info("frame %d: LIGHT_ON_L confirmed", frame_idx)
        else:
            just_off = self._state_L.miss()
            if just_off:
                result.events.append(LightEvent.LIGHT_OFF)

        # Right side
        if blob_R:
            just_confirmed = self._state_R.see()
            if just_confirmed:
                result.events.append(LightEvent.LIGHT_ON_R)
                logger.info("frame %d: LIGHT_ON_R confirmed", frame_idx)
        else:
            just_off = self._state_R.miss()
            if just_off and LightEvent.LIGHT_OFF not in result.events:
                result.events.append(LightEvent.LIGHT_OFF)

        result.left_on  = self._state_L.confirmed
        result.right_on = self._state_R.confirmed

        # LIGHT_ON_BOTH fires once when both become simultaneously confirmed
        if result.both_on and self._both_on_frame is None:
            self._both_on_frame = frame_idx
            result.events.append(LightEvent.LIGHT_ON_BOTH)
            logger.info("frame %d: LIGHT_ON_BOTH", frame_idx)

        # WINDOW_CLOSE fires LIGHT_WINDOW_CLOSE_DELAY frames after LIGHT_ON_BOTH
        if (self._both_on_frame is not None
                and not self._window_closed
                and frame_idx >= self._both_on_frame + config.LIGHT_WINDOW_CLOSE_DELAY):
            self._window_closed = True
            result.events.append(LightEvent.WINDOW_CLOSE)
            logger.info("frame %d: WINDOW_CLOSE (light +%d frames)",
                        frame_idx, config.LIGHT_WINDOW_CLOSE_DELAY)

        self._history.append(result)
        return result

    def reset(self) -> None:
        self.__init__(self._mid_x * 2)

    @property
    def history(self) -> list[FrameLightResult]:
        return self._history

    @property
    def window_closed(self) -> bool:
        return self._window_closed

    def first_both_on_frame(self) -> int | None:
        """Return the frame index of LIGHT_ON_BOTH, or None if not yet seen."""
        return self._both_on_frame

    def window_close_frame(self) -> int | None:
        """Return the frame index of WINDOW_CLOSE, or None if not yet reached."""
        if self._both_on_frame is None:
            return None
        return self._both_on_frame + config.LIGHT_WINDOW_CLOSE_DELAY
        return None

    # ------------------------------------------------------------------ #
    #  Internal blob detection                                             #
    # ------------------------------------------------------------------ #

    def _detect_blobs(self, frame_bgr: np.ndarray) -> tuple[bool, bool]:
        """Return (left_blob_present, right_blob_present).

        Only inspects the bottom LIGHT_ROI_Y_RATIO portion of the frame
        to avoid false positives from audience, flags, and scoreboard text.
        """
        h, w = frame_bgr.shape[:2]
        roi_y = int(h * config.LIGHT_ROI_Y_RATIO)
        roi = frame_bgr[roi_y:, :]   # bottom slice only

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Red mask (two HSV ranges because red wraps around 0°/180°)
        red1 = cv2.inRange(hsv,
                           np.array(config.RED_LOWER_1),
                           np.array(config.RED_UPPER_1))
        red2 = cv2.inRange(hsv,
                           np.array(config.RED_LOWER_2),
                           np.array(config.RED_UPPER_2))
        red_mask = cv2.bitwise_or(red1, red2)

        green_mask = cv2.inRange(hsv,
                                 np.array(config.GREEN_LOWER),
                                 np.array(config.GREEN_UPPER))

        light_mask = cv2.bitwise_or(red_mask, green_mask)

        # Small morph cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        light_mask = cv2.morphologyEx(light_mask, cv2.MORPH_CLOSE, kernel)

        left_half  = light_mask[:, :self._mid_x]
        right_half = light_mask[:, self._mid_x:]

        blob_L = _largest_blob_area(left_half)  >= config.LIGHT_MIN_AREA_PX
        blob_R = _largest_blob_area(right_half) >= config.LIGHT_MIN_AREA_PX

        return blob_L, blob_R


def _largest_blob_area(mask: np.ndarray) -> float:
    """Return the area of the largest connected component in the mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    return float(max(cv2.contourArea(c) for c in contours))
