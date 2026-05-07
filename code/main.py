"""main.py — Batch entry point for the Fencing Auto Referee pipeline.

Processes every labelled video in VIDEO_ROOT (互中/ and 同时/ subfolders),
runs the full analysis pipeline, saves visualizations, and prints an
accuracy summary comparing predicted verdicts against ground-truth labels
encoded in each filename.

Usage:
    python main.py
    python main.py --folders 互中 同时
    python main.py --video "..\video\奥运-标记视频\互中\互中1左侧.mp4"
    python main.py --no-piste
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

import config
from modules.pose_module      import PoseModule
from modules.athlete_tracker  import AthleteTracker
from modules.piste_module     import PisteDetector
from modules.light_module     import LightModule
from modules.event_aggregator import EventAggregator, AggregatorResult
from utils.video_utils        import open_video, get_video_info, iter_frames
from visualization.timeline_plot import save_timeline
from visualization.angle_plot    import save_angle_plot

logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


# ── Ground-truth label parsing ────────────────────────────────────────────────

def _ground_truth(stem: str) -> str:
    """Infer expected verdict from filename stem.

    Convention:
        互中*左侧* → LEFT_SCORES
        互中*右侧* → RIGHT_SCORES
        同时*      → SIMULTANEOUS
        anything else → UNKNOWN
    """
    s = stem
    if "同时" in s:
        return "SIMULTANEOUS"
    if "互中" in s or "防守" in s or "复杂" in s:
        if "左侧" in s:
            return "LEFT_SCORES"
        if "右侧" in s:
            return "RIGHT_SCORES"
    return "UNKNOWN"


# ── Single-video pipeline ─────────────────────────────────────────────────────

def run_video(
    video_path: Path,
    pose:       PoseModule,
    use_piste:  bool,
    out_root:   Path,
) -> tuple[str, str, str]:
    """Run the full pipeline on one video.

    Returns (video_name, ground_truth, verdict_str).
    """
    out_dir = out_root / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = AthleteTracker()
    agg     = EventAggregator()

    cap  = open_video(video_path)
    info = get_video_info(cap)
    fps  = info["fps"]
    lights = LightModule(frame_width=info["width"])

    piste_det   = PisteDetector()
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
    result: AggregatorResult = agg.finalize()

    # ── Save visualizations ───────────────────────────────────────────────────
    gt = _ground_truth(video_path.stem)

    save_timeline(
        result      = result,
        fps         = fps,
        out_path    = out_dir / "timeline.png",
        video_name  = video_path.stem,
        ground_truth= gt,
    )

    save_angle_plot(
        result      = result,
        baseline_L  = tracker.baseline_L,
        baseline_R  = tracker.baseline_R,
        fps         = fps,
        out_path    = out_dir / "joint_angles.png",
        video_name  = video_path.stem,
    )

    # ── Save summary text ─────────────────────────────────────────────────────
    verdict_str = result.verdict.value
    match       = (gt == verdict_str) or (gt == "UNKNOWN")
    summary = (
        f"Video        : {video_path.name}\n"
        f"Ground truth : {gt}\n"
        f"Verdict      : {verdict_str}\n"
        f"Match        : {'YES ✓' if match else 'NO  ✗'}\n"
        f"\n"
        f"t_arm_L    : {result.t_arm_L}\n"
        f"t_arm_R    : {result.t_arm_R}\n"
        f"t_launch_L : {result.t_launch_L}\n"
        f"t_launch_R : {result.t_launch_R}\n"
        f"t_lunge_L  : {result.t_step_L}\n"
        f"t_lunge_R  : {result.t_step_R}\n"
        f"pause_L    : {result.pause_L}\n"
        f"pause_R    : {result.pause_R}\n"
    )
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")

    return video_path.stem, gt, verdict_str


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fencing Auto Referee — batch pipeline")
    parser.add_argument("--video",   default=None,
                        help="Run on a single video path instead of the full batch.")
    parser.add_argument("--folders", nargs="+", default=["互中", "同时"],
                        help="Subfolders of VIDEO_ROOT to process.")
    parser.add_argument("--no-piste", action="store_true",
                        help="Skip piste detection (useful if piste not visible).")
    args = parser.parse_args()

    # ── Collect videos ────────────────────────────────────────────────────────
    if args.video:
        videos = [Path(args.video)]
    else:
        video_root = Path(config.VIDEO_ROOT)
        videos = []
        for folder in args.folders:
            videos += sorted((video_root / folder).glob("*.mp4"))

    if not videos:
        print("No videos found. Check VIDEO_ROOT in config.py.")
        sys.exit(1)

    out_root = Path(config.OUTPUT_ROOT)

    print(f"\n{'='*60}")
    print(f"  Fencing Auto Referee — Batch Run")
    print(f"  Videos : {len(videos)}")
    print(f"  Output : {out_root}")
    print(f"{'='*60}\n")

    print("Loading YOLOv8-Pose model …")
    pose = PoseModule()
    print("Model ready.\n")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    results: list[tuple[str, str, str]] = []
    for i, vp in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}]  {vp.name} …", end=" ", flush=True)
        name, gt, verdict = run_video(vp, pose, not args.no_piste, out_root)
        match = (gt == verdict) or (gt == "UNKNOWN")
        status = "✓" if match else "✗"
        print(f"{verdict}  {status}")
        results.append((name, gt, verdict))

    # ── Accuracy summary ──────────────────────────────────────────────────────
    labelled  = [(n, g, v) for n, g, v in results if g != "UNKNOWN"]
    correct   = sum(1 for _, g, v in labelled if g == v)
    total     = len(labelled)
    accuracy  = (correct / total * 100) if total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  ACCURACY SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Video':<35}  {'GT':<20}  {'Predicted':<20}  {'Match'}")
    print(f"  {'-'*35}  {'-'*20}  {'-'*20}  {'-'*5}")
    for name, gt, verdict in results:
        match  = (gt == verdict) or (gt == "UNKNOWN")
        status = "✓" if match else "✗"
        print(f"  {name:<35}  {gt:<20}  {verdict:<20}  {status}")
    print(f"\n  Labelled videos : {total}")
    print(f"  Correct         : {correct}")
    print(f"  Accuracy        : {accuracy:.1f}%")
    print(f"{'='*60}\n")
    print(f"Output saved to: {out_root}")


if __name__ == "__main__":
    main()
