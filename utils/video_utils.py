"""Frame reading and video metadata helpers."""

import cv2
import numpy as np
from pathlib import Path


def open_video(path: str | Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    return cap


def get_video_info(cap: cv2.VideoCapture) -> dict:
    return {
        "fps":    cap.get(cv2.CAP_PROP_FPS),
        "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }


def iter_frames(cap: cv2.VideoCapture, target_fps: float | None = None):
    """Yield (frame_index, frame_bgr) pairs.

    If target_fps is given and differs from the video's native fps, frames are
    subsampled uniformly so the effective rate matches target_fps.
    """
    info = get_video_info(cap)
    native_fps = info["fps"]

    if target_fps is None or abs(native_fps - target_fps) < 0.5:
        step = 1
    else:
        step = max(1, round(native_fps / target_fps))

    frame_idx = 0
    output_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            yield output_idx, frame
            output_idx += 1
        frame_idx += 1


def parse_ground_truth(video_path: str | Path) -> str | None:
    """Infer ground-truth scorer from filename convention.

    Returns: "left", "right", "simultaneous", or None if unrecognised.
    """
    name = Path(video_path).stem
    if "右侧" in name:
        return "right"
    if "左侧" in name:
        return "left"
    if "同时" in name:
        return "simultaneous"
    return None


def parse_scenario(video_path: str | Path) -> str | None:
    """Return "coup_double" or "simultaneous" based on folder / filename."""
    parts = Path(video_path).parts
    for p in parts:
        if "互中" in p:
            return "coup_double"
        if "同时" in p:
            return "simultaneous"
    name = Path(video_path).stem
    if "互中" in name:
        return "coup_double"
    if "同时" in name:
        return "simultaneous"
    return None
