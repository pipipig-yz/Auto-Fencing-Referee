"""Batch priority report — runs the full pipeline on every video in 互中/ and 同时/,
captures Priority step log lines and outputs a clean per-video summary.

Usage (from the code/ directory):
    python scripts/batch_priority_report.py
    python scripts/batch_priority_report.py --no-piste
"""

import argparse
import logging
import sys
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

import config
from modules.pose_module import PoseModule
from modules.athlete_tracker import AthleteTracker
from modules.piste_module import PisteDetector
from modules.light_module import LightModule
from modules.event_aggregator import EventAggregator
from utils.video_utils import open_video, get_video_info, iter_frames


# ── Log capture handler ───────────────────────────────────────────────────────

class _CaptureHandler(logging.Handler):
    """Captures log records into a list for post-processing."""
    def __init__(self):
        super().__init__()
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record.getMessage())

    def clear(self) -> None:
        self.records.clear()


# ── Only keep lines that start with these prefixes ───────────────────────────

_KEEP_PREFIXES = (
    "Priority step 1:",
    "Priority step 2:",
    "Priority step 3:",
    "VERDICT:",
)


def _filter(records: list[str]) -> list[str]:
    return [r for r in records if any(r.startswith(p) for p in _KEEP_PREFIXES)]


# ── Single-video runner ───────────────────────────────────────────────────────

def run_video(video_path: Path, use_piste: bool,
              pose: PoseModule, capture: _CaptureHandler) -> str:
    """Run the full pipeline on one video and return a formatted report block."""
    capture.clear()

    tracker = AthleteTracker()
    agg     = EventAggregator()

    cap  = open_video(video_path)
    info = get_video_info(cap)
    fps  = info["fps"]
    lights = LightModule(frame_width=info["width"])

    piste_det = PisteDetector()
    active_piste = use_piste
    if use_piste:
        ret, first_frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if ret:
            box = piste_det.detect_piste(first_frame)
            if not box:
                active_piste = False

    for frame_idx, frame in iter_frames(cap, target_fps=config.TARGET_FPS):
        raw_dets   = pose.infer(frame)
        detections = piste_det.filter_athletes(raw_dets) if active_piste else raw_dets[:2]
        pose_data  = tracker.update(frame_idx, detections)
        light_data = lights.update(frame_idx, frame)
        agg.update(pose_data, light_data, tracker.baseline_L, tracker.baseline_R)

        if lights.window_closed:
            break

    cap.release()
    result = agg.finalize()

    # ── Format report block ───────────────────────────────────────────────────
    sep   = "─" * 60
    lines = [
        sep,
        f"FILE    : {video_path.name}",
        f"VERDICT : {result.verdict}",
        "",
    ]
    priority_lines = _filter(capture.records)
    if priority_lines:
        lines += priority_lines
    else:
        lines.append("(no priority log lines captured)")
    lines.append("")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-piste", action="store_true")
    args = parser.parse_args()

    video_root = Path(config.VIDEO_ROOT)
    folders    = ["互中", "同时"]
    videos: list[Path] = []
    for folder in folders:
        videos += sorted((video_root / folder).glob("*.mp4"))

    if not videos:
        print("No videos found.")
        sys.exit(1)

    # Silence everything first, then attach capture only to event_aggregator
    logging.root.setLevel(logging.WARNING)
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).setLevel(logging.WARNING)

    capture = _CaptureHandler()
    agg_logger = logging.getLogger("modules.event_aggregator")
    agg_logger.setLevel(logging.INFO)
    agg_logger.propagate = False
    agg_logger.addHandler(capture)

    print(f"\nLoading YOLOv8-Pose model …")
    pose = PoseModule()
    print(f"Running {len(videos)} videos\n")

    for video_path in videos:
        report = run_video(video_path, not args.no_piste, pose, capture)
        print(report)

    print("Done.")


if __name__ == "__main__":
    main()
