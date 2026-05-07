"""Shared drawing helpers for all debug visualisation scripts."""

import math

import cv2
import numpy as np

import config
from modules.athlete_tracker import AthleteFrame, AthleteBaseline

_FONT   = cv2.FONT_HERSHEY_SIMPLEX
_SCALE  = 1.6     # metric lines
_THICK  = 3
_PAD    = 16
_GAP    = 10
_GREEN  = (0, 230, 0)
_GRAY   = (150, 150, 150)
_WHITE  = (255, 255, 255)
_BG     = (20, 20, 20)

_SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6),
]


def fv(v: float, fmt: str = ".1f") -> str:
    return "nan" if math.isnan(v) else format(v, fmt)


# ── Skeleton ──────────────────────────────────────────────────────────────────

def draw_skeleton(img: np.ndarray, det, color: tuple) -> None:
    kps = det.keypoints
    for i, j in _SKELETON:
        if kps[i, 2] > 0.3 and kps[j, 2] > 0.3:
            cv2.line(img,
                     (int(kps[i, 0]), int(kps[i, 1])),
                     (int(kps[j, 0]), int(kps[j, 1])),
                     color, 2)
    for x, y, c in kps:
        if c > 0.3:
            cv2.circle(img, (int(x), int(y)), 4, color, -1)


# ── Top label ─────────────────────────────────────────────────────────────────

def draw_top_label(
    img:       np.ndarray,
    label:     str,
    frame_idx: int,
    fps:       float,
) -> None:
    """Centered banner at top: '<label>   f####   #.##s'"""
    h, w = img.shape[:2]
    scale = 2.0
    thick = 3
    text  = f"{label}   f{frame_idx:04d}   {frame_idx / fps:.2f}s"

    (tw, th), bl = cv2.getTextSize(text, _FONT, scale, thick)
    box_h = th + bl + _PAD * 2

    cv2.rectangle(img, (0, 0), (w, box_h), _BG, cv2.FILLED)
    x_text = max(0, (w - tw) // 2)
    cv2.putText(img, text, (x_text, _PAD + th),
                _FONT, scale, _WHITE, thick, cv2.LINE_AA)


# ── Per-athlete metric block ──────────────────────────────────────────────────

def draw_athlete_metrics(
    img:  np.ndarray,
    side: str,                  # "L" or "R"
    af:   AthleteFrame,
    bl:   AthleteBaseline,
) -> None:
    """Draw lunge (3 metrics) + attack (2 metrics) block at bottom-left or right.

    Entire LUNGE section is green when all three lunge conditions are met.
    Entire ATTACK section is green when both attack conditions are met.
    """
    h, w = img.shape[:2]

    # ── Extract values ────────────────────────────────────────────────────────
    if af.detected:
        knee     = af.angles.get("front_knee_angle",  math.nan)
        thigh    = af.angles.get("inter_thigh_angle", math.nan)
        stance   = af.angles.get("stance_width",      math.nan)
        elbow    = af.angles.get("elbow_angle",       math.nan)
        shoulder = af.angles.get("shoulder_angle",    math.nan)
    else:
        knee = thigh = stance = elbow = shoulder = math.nan

    ratio = math.nan
    if bl.captured and bl.stance_width > 0 and not math.isnan(stance):
        ratio = stance / bl.stance_width

    thigh_thresh = (config.LUNGE_THIGH_ANGLE_L if side == "L"
                    else config.LUNGE_THIGH_ANGLE_R)

    lunge_ok  = (not math.isnan(knee)     and knee     > config.LUNGE_FRONT_KNEE_ABS
                 and not math.isnan(thigh) and thigh    > thigh_thresh
                 and not math.isnan(ratio) and ratio    > config.LUNGE_WIDTH_RATIO)
    attack_ok = (not math.isnan(elbow)    and elbow    > config.ATTACK_ELBOW_ABS
                 and not math.isnan(shoulder) and shoulder > config.ATTACK_SHOULDER_ABS)

    lc = _GREEN if lunge_ok  else _GRAY
    ac = _GREEN if attack_ok else _GRAY

    # ── Build line list: (text, color, is_header) ────────────────────────────
    lines = [
        (f"{side} — LUNGE",                             lc,   True),
        (f"  knee     {fv(knee)}°",                     lc,   False),
        (f"  thigh    {fv(thigh)}°",                    lc,   False),
        (f"  ratio    {fv(ratio, '.2f')}×",             lc,   False),
        ("",                                             _GRAY, False),   # spacer
        (f"{side} — ATTACK",                            ac,   True),
        (f"  elbow    {fv(elbow)}°",                    ac,   False),
        (f"  shoulder {fv(shoulder)}°",                 ac,   False),
    ]

    # ── Measure box ───────────────────────────────────────────────────────────
    line_heights = []
    max_w = 0
    for text, _, is_hdr in lines:
        if not text:
            line_heights.append(_GAP * 2)
            continue
        sc = _SCALE * 1.1 if is_hdr else _SCALE
        (tw, lh), bl_h = cv2.getTextSize(text, _FONT, sc, _THICK)
        line_heights.append(lh + bl_h + _GAP)
        max_w = max(max_w, tw)

    box_h = _PAD + sum(line_heights) + _PAD
    box_w = max_w + _PAD * 2

    x0 = 0 if side == "L" else w - box_w
    y0 = h - box_h

    cv2.rectangle(img, (x0, y0), (x0 + box_w, h), _BG, cv2.FILLED)

    # ── Draw lines ────────────────────────────────────────────────────────────
    y = y0 + _PAD
    for (text, color, is_hdr), lh in zip(lines, line_heights):
        if text:
            sc = _SCALE * 1.1 if is_hdr else _SCALE
            (_, th), _ = cv2.getTextSize(text, _FONT, sc, _THICK)
            cv2.putText(img, text, (x0 + _PAD, y + th),
                        _FONT, sc, color, _THICK, cv2.LINE_AA)
        y += lh
