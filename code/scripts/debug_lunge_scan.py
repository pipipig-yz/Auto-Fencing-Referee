"""Scan all videos and save every frame where an athlete satisfies ALL THREE
lunge conditions simultaneously:
    front_knee_angle  > LUNGE_FRONT_KNEE_ABS  (150°)
    inter_thigh_angle > LUNGE_THIGH_ANGLE_L/R (100°/120°)
    stance_width_ratio > LUNGE_WIDTH_RATIO     (2.0×)

Output: output/lunge_scan/<video_stem>/<frame>_<L|R|LR>.jpg

Usage (from the code/ directory):
    python scripts/debug_lunge_scan.py
    python scripts/debug_lunge_scan.py --no-piste
"""

import argparse
import logging
import math
import sys
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
from utils.debug_utils import draw_skeleton, draw_top_label, draw_athlete_metrics

logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s  %(message)s")

VIDEOS = [
    r"..\video\奥运-标记视频\互中\互中1左侧.mp4",
    r"..\video\奥运-标记视频\互中\互中1右侧.mp4",
    r"..\video\奥运-标记视频\互中\互中2右侧.mp4",
    r"..\video\奥运-标记视频\互中\互中3左侧.mp4",
    r"..\video\奥运-标记视频\互中\互中4右侧.mp4",
    r"..\video\奥运-标记视频\互中\互中5右侧.mp4",
    r"..\video\奥运-标记视频\同时\同时1-1.mp4",
    r"..\video\奥运-标记视频\同时\同时2-1.mp4",
    r"..\video\奥运-标记视频\同时\同时3-1.mp4",
    r"..\video\奥运-标记视频\同时\同时飞剑-1.mp4",
    r"..\video\奥运-标记视频\同时\同时（左侧启动快）-1.mp4",
]

_COLOR_L = (255, 80,  80)
_COLOR_R = (80,  80, 255)


def _is_lunge(af, bl, side):
    if not af.detected or not bl.captured or bl.stance_width <= 0:
        return False
    knee  = af.angles.get("front_knee_angle",  math.nan)
    thigh = af.angles.get("inter_thigh_angle", math.nan)
    stance = af.angles.get("stance_width",     math.nan)
    if math.isnan(knee) or math.isnan(thigh) or math.isnan(stance):
        return False
    ratio        = stance / bl.stance_width
    thigh_thresh = config.LUNGE_THIGH_ANGLE_L if side == "L" else config.LUNGE_THIGH_ANGLE_R
    return (knee  > config.LUNGE_FRONT_KNEE_ABS
            and thigh > thigh_thresh
            and ratio > config.LUNGE_WIDTH_RATIO)


def _save_frame(frame, detections, frame_idx, fps, af_L, af_R, bl_L, bl_R,
                trig_L, trig_R, out_dir):
    img = frame.copy()

    for det, color in zip(detections[:2], [_COLOR_L, _COLOR_R]):
        draw_skeleton(img, det, color)

    sides = ("L" if trig_L else "") + ("R" if trig_R else "")
    draw_top_label(img, f"LUNGE  {sides}", frame_idx, fps)
    draw_athlete_metrics(img, "L", af_L, bl_L)
    draw_athlete_metrics(img, "R", af_R, bl_R)

    tag = sides
    cv2.imwrite(str(out_dir / f"{frame_idx:05d}_{tag}.jpg"), img)


def process_video(video_path: Path, fps: float, use_piste: bool,
                  pose: PoseModule, piste: PisteDetector) -> int:
    out_dir = Path(config.OUTPUT_ROOT) / "lunge_scan" / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.jpg"):
        f.unlink()

    tracker = AthleteTracker()
    agg     = EventAggregator()
    cap     = open_video(video_path)
    info    = get_video_info(cap)
    fps     = info["fps"]
    lights  = LightModule(frame_width=info["width"])

    _use_piste = use_piste
    if _use_piste:
        ret, first = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if ret and not piste.detect_piste(first):
            _use_piste = False

    saved = 0

    for frame_idx, frame in iter_frames(cap, target_fps=config.TARGET_FPS):
        raw_dets   = pose.infer(frame)
        detections = piste.filter_athletes(raw_dets) if _use_piste else raw_dets[:2]
        pose_data  = tracker.update(frame_idx, detections)
        light_data = lights.update(frame_idx, frame)
        agg.update(pose_data, light_data, tracker.baseline_L, tracker.baseline_R)

        if not (agg._window_open and not agg._window_closed):
            if lights.window_closed:
                break
            continue

        trig_L = _is_lunge(pose_data.L, tracker.baseline_L, "L")
        trig_R = _is_lunge(pose_data.R, tracker.baseline_R, "R")

        if trig_L or trig_R:
            _save_frame(frame, detections, frame_idx, fps,
                        pose_data.L, pose_data.R,
                        tracker.baseline_L, tracker.baseline_R,
                        trig_L, trig_R, out_dir)
            saved += 1

        if lights.window_closed:
            break

    cap.release()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-piste", action="store_true")
    args = parser.parse_args()

    print("Loading YOLOv8-Pose model …")
    pose  = PoseModule()
    piste = PisteDetector()

    total = 0
    for rel_path in VIDEOS:
        video_path = Path(rel_path)
        if not video_path.exists():
            print(f"  SKIP (not found): {video_path.name}")
            continue
        print(f"\n{'─'*60}")
        print(f"  {video_path.name}")
        saved = process_video(video_path, config.TARGET_FPS,
                              not args.no_piste, pose, piste)
        out = Path(config.OUTPUT_ROOT) / "lunge_scan" / video_path.stem
        print(f"  {saved} image(s) → {out}")
        total += saved

    print(f"\n{'='*60}")
    print(f"  Done. {total} total images saved.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
