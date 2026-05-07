"""Debug script for EventAggregator — runs the full pipeline on one video,
saves a frame image for every detected event, and prints the event timeline
+ priority verdict.

Output: output/aggregator/<video_stem>/<frame>_<EVENT_TYPE>.jpg

Usage (from the code/ directory):
    python scripts/debug_aggregator.py
    python scripts/debug_aggregator.py --video "..\\video\\奥运-标记视频\\互中\\互中1左侧.mp4"
    python scripts/debug_aggregator.py --no-piste
    python scripts/debug_aggregator.py --lunge
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

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s  %(message)s")

_COLOR_L = (255, 80,  80)
_COLOR_R = (80,  80, 255)
_LUNGE_EVENTS  = {"L_LUNGE", "R_LUNGE", "L_LUNGE_FULL", "R_LUNGE_FULL"}


def _save_event_frame(frame, detections, ev, frame_idx, fps, out_dir,
                      af_L, af_R, bl_L, bl_R):
    img = frame.copy()

    for det, color in zip(detections[:2], [_COLOR_L, _COLOR_R]):
        draw_skeleton(img, det, color)

    draw_top_label(img, ev.event_type, frame_idx, fps)
    draw_athlete_metrics(img, "L", af_L, bl_L)
    draw_athlete_metrics(img, "R", af_R, bl_R)

    cv2.imwrite(str(out_dir / f"{frame_idx:05d}_{ev.event_type}.jpg"), img)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None)
    parser.add_argument("--no-piste", action="store_true")
    parser.add_argument("--lunge", action="store_true",
                        help="Print per-frame lunge table and save only lunge events.")
    args = parser.parse_args()

    video_path = (
        Path(args.video) if args.video
        else Path(config.VIDEO_ROOT) / "互中" / "互中1左侧.mp4"
    )
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}")
        sys.exit(1)

    out_dir = Path(config.OUTPUT_ROOT) / "aggregator" / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.jpg"):
        f.unlink()

    print(f"\n{'='*62}")
    print(f"Video  : {video_path.name}")
    print(f"Output : {out_dir}")
    print(f"{'='*62}\n")

    print("Loading YOLOv8-Pose model …")
    pose    = PoseModule()
    tracker = AthleteTracker()
    piste   = PisteDetector()
    agg     = EventAggregator()

    cap  = open_video(video_path)
    info = get_video_info(cap)
    fps  = info["fps"]
    lights = LightModule(frame_width=info["width"])
    print(f"Video: {info['width']}×{info['height']}  {fps:.1f}fps  "
          f"{info['total_frames']} frames\n")

    use_piste = not args.no_piste
    if use_piste:
        ret, first_frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if ret:
            box = piste.detect_piste(first_frame)
            if box:
                print(f"Piste detected: x={box[0]} y={box[1]} w={box[2]} h={box[3]}\n")
            else:
                print("WARNING: piste not detected — using raw detections.\n")
                use_piste = False

    if args.lunge:
        print(f"{'Frame':>6}  "
              f"{'L-knee':>8} {'L-thigh':>8} {'L-stance':>9} {'L-ratio':>8}  "
              f"{'R-knee':>8} {'R-thigh':>8} {'R-stance':>9} {'R-ratio':>8}")
        print("-" * 90)

    saved = 0

    for frame_idx, frame in iter_frames(cap, target_fps=config.TARGET_FPS):
        raw_dets   = pose.infer(frame)
        detections = piste.filter_athletes(raw_dets) if use_piste else raw_dets[:2]
        pose_data  = tracker.update(frame_idx, detections)
        light_data = lights.update(frame_idx, frame)
        new_events = agg.update(pose_data, light_data, tracker.baseline_L, tracker.baseline_R)

        if args.lunge and pose_data.baseline_captured:
            def _f(v):
                return f"{'nan':>8}" if v != v else f"{v:8.1f}"
            bl_L = tracker.baseline_L
            bl_R = tracker.baseline_R
            L, R = pose_data.L, pose_data.R
            stance_L = L.angles.get("stance_width", float("nan"))
            stance_R = R.angles.get("stance_width", float("nan"))
            ratio_L  = stance_L / bl_L.stance_width if bl_L.stance_width > 0 else float("nan")
            ratio_R  = stance_R / bl_R.stance_width if bl_R.stance_width > 0 else float("nan")
            print(f"{frame_idx:>6}  "
                  f"{_f(L.angles.get('front_knee_angle',  float('nan')))} "
                  f"{_f(L.angles.get('inter_thigh_angle', float('nan')))} "
                  f"{_f(stance_L)}px {ratio_L:8.2f}x  "
                  f"{_f(R.angles.get('front_knee_angle',  float('nan')))} "
                  f"{_f(R.angles.get('inter_thigh_angle', float('nan')))} "
                  f"{_f(stance_R)}px {ratio_R:8.2f}x")

        for ev in new_events:
            if args.lunge and ev.event_type not in _LUNGE_EVENTS:
                continue
            _save_event_frame(
                frame, detections, ev, frame_idx, fps, out_dir,
                pose_data.L, pose_data.R, tracker.baseline_L, tracker.baseline_R,
            )
            saved += 1

        if lights.window_closed:
            print(f"\nWindow closed at frame {frame_idx}. Stopping.\n")
            break

    cap.release()
    result = agg.finalize()

    print(f"{saved} event image(s) → {out_dir}")
    print("\n" + "=" * 62)
    print(f"Ground truth : {video_path.stem}")
    print(f"VERDICT      : {result.verdict}")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
