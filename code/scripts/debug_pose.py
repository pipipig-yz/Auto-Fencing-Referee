"""Debug script — Checkpoint 1, 2 & audio.

Usage (from the code/ directory):
    python scripts/debug_pose.py --video "..\video\...\clip.mp4" --detect-piste --detect-lights --detect-audio
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

import config
from modules.pose_module import PoseModule, PoseDetection
from modules.athlete_tracker import AthleteTracker
from modules.piste_module import PisteDetector
from modules.light_module import LightModule, LightEvent
from modules.audio_module import analyse as analyse_audio
from utils.video_utils import open_video, get_video_info, iter_frames

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

_SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6),
]


def draw_skeleton(frame: np.ndarray, det: PoseDetection, color: tuple) -> np.ndarray:
    kps = det.keypoints
    out = frame.copy()
    for i, j in _SKELETON:
        if kps[i, 2] > 0.3 and kps[j, 2] > 0.3:
            cv2.line(out,
                     (int(kps[i, 0]), int(kps[i, 1])),
                     (int(kps[j, 0]), int(kps[j, 1])),
                     color, 2)
    for idx, (x, y, c) in enumerate(kps):
        if c > 0.3:
            cv2.circle(out, (int(x), int(y)), 4, color, -1)
            cv2.putText(out, str(idx), (int(x) + 4, int(y) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    x1, y1 = int(det.box_xyxy[0]), int(det.box_xyxy[1])
    label = "L" if color == (255, 80, 80) else "R"
    cv2.putText(out, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None)
    parser.add_argument("--frames", type=int, default=999999,
                        help="Max frames to process. Default = entire video.")
    parser.add_argument("--save-overlay", action="store_true")
    parser.add_argument("--overlay-every", type=int, default=10)
    parser.add_argument("--detect-piste", action="store_true")
    parser.add_argument("--detect-lights", action="store_true")
    parser.add_argument("--detect-audio", action="store_true",
                        help="Run audio analysis (buzzer + Allez detection).")
    args = parser.parse_args()

    # ── Pick a video ──────────────────────────────────────────────────────
    if args.video:
        video_path = Path(args.video)
    else:
        root = Path(config.VIDEO_ROOT)
        videos = list(root.rglob("*.mp4"))
        if not videos:
            print(f"No .mp4 files found under {config.VIDEO_ROOT}")
            sys.exit(1)
        video_path = videos[0]

    overlay_dir = Path(config.OUTPUT_ROOT) / "debug_overlay" / video_path.stem
    if args.save_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Video : {video_path.name}")
    print(f"{'='*60}\n")

    # ── Audio analysis (whole-file, done before frame loop) ───────────────
    audio_result = None
    if args.detect_audio:
        print("Running audio analysis …")
        audio_result = analyse_audio(video_path)
        print(f"  Language   : {audio_result.language or 'unknown'}")
        if audio_result.words:
            print("  Speech (all words with timestamps):")
            for w in audio_result.words:
                print(f"    [{w.start:6.2f}s – {w.end:6.2f}s]  {w.word}")
        else:
            print("  Speech     : (none detected)")
        print(f"  Allez      : {audio_result.allez_time_s:.3f}s"
              if audio_result.allez_time_s is not None else "  Allez      : not detected")
        print(f"  Buzzer     : {audio_result.buzzer_time_s:.3f}s"
              if audio_result.buzzer_time_s is not None else "  Buzzer     : not detected")
        print()

    # ── Init modules ──────────────────────────────────────────────────────
    print("Loading YOLOv8-Pose model …")
    pose    = PoseModule()
    tracker = AthleteTracker()
    piste   = PisteDetector()

    cap  = open_video(video_path)
    info = get_video_info(cap)
    fps  = info["fps"]
    lights = LightModule(frame_width=info["width"])
    print(f"Video: {info['width']}×{info['height']}  {fps:.1f}fps  "
          f"{info['total_frames']} frames\n")

    # ── Piste detection on first frame ────────────────────────────────────
    if args.detect_piste:
        ret, first_frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if ret:
            box = piste.detect_piste(first_frame)
            if box:
                print(f"Piste: x={box[0]} y={box[1]} w={box[2]} h={box[3]}")
                overlay_dir.mkdir(parents=True, exist_ok=True)
                raw_dets = pose.infer(first_frame)
                filtered = piste.filter_athletes(raw_dets)
                piste.save_debug_image(first_frame, raw_dets, filtered,
                                       overlay_dir / "piste_detection.jpg")
                print(f"Piste debug image → {overlay_dir / 'piste_detection.jpg'}\n")
            else:
                print("WARNING: Piste not detected on first frame.\n")

    # ── Frame loop ────────────────────────────────────────────────────────
    print(f"{'Frame':>6}  "
          f"{'L-elbow':>8} {'L-knee':>7} {'L-stance':>9} {'L-comVel':>9}  "
          f"{'R-elbow':>8} {'R-knee':>7} {'R-stance':>9} {'R-comVel':>9}  "
          f"{'status':>12}")
    print("-" * 105)

    baseline_frame: dict[str, int] = {}   # tracks when each side's baseline was first captured
    window_open_frame: int | None = None  # first frame after any baseline is captured

    for frame_idx, frame in iter_frames(cap, target_fps=config.TARGET_FPS):
        if frame_idx >= args.frames:
            break

        raw_dets   = pose.infer(frame)
        detections = piste.filter_athletes(raw_dets) if args.detect_piste else raw_dets[:2]
        fd         = tracker.update(frame_idx, detections)

        # Track baseline capture frame
        if tracker.baseline_L.captured and "BASELINE LEFT" not in baseline_frame:
            baseline_frame["BASELINE LEFT"] = frame_idx
        if tracker.baseline_R.captured and "BASELINE RIGHT" not in baseline_frame:
            baseline_frame["BASELINE RIGHT"] = frame_idx

        # WINDOW_OPEN: first frame after any baseline is captured
        if window_open_frame is None and baseline_frame:
            window_open_frame = frame_idx
            print(f"  >>> frame {frame_idx:>4}  WINDOW_OPEN")

        def fmt(val, unit="°"):
            return "    nan " if val != val else f"{val:7.1f}{unit}"

        # Light detection
        if args.detect_lights:
            lr = lights.update(frame_idx, frame)
            if lr.events:
                print(f"  >>> frame {frame_idx:>4}  {' '.join(e.value for e in lr.events)}")

        L, R   = fd.L, fd.R
        status = "BASELINE✓" if fd.baseline_captured else ""

        print(f"{frame_idx:>6}  "
              f"{fmt(L.angles.get('elbow_angle',      float('nan')))} "
              f"{fmt(L.angles.get('front_knee_angle', float('nan')))} "
              f"{fmt(L.angles.get('stance_width',     float('nan')), 'px')} "
              f"{fmt(L.com_velocity, 'px')}  "
              f"{fmt(R.angles.get('elbow_angle',      float('nan')))} "
              f"{fmt(R.angles.get('front_knee_angle', float('nan')))} "
              f"{fmt(R.angles.get('stance_width',     float('nan')), 'px')} "
              f"{fmt(R.com_velocity, 'px')}  "
              f"{status}")

        # Skeleton overlay
        if args.save_overlay and frame_idx % args.overlay_every == 0:
            overlay = frame.copy()
            if args.detect_piste and piste.piste_box:
                px, py, pw, ph = piste.piste_box
                cv2.rectangle(overlay, (px, py), (px+pw, py+ph), (0, 255, 255), 2)
            for i, det in enumerate(detections[:2]):
                overlay = draw_skeleton(overlay, det, [(255, 80, 80), (80, 80, 255)][i])
            cv2.putText(overlay, f"frame {frame_idx}  {status}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imwrite(str(overlay_dir / f"frame_{frame_idx:05d}.jpg"), overlay)

    cap.release()

    # ════════════════════════════════════════════════════════════════
    #  UNIFIED EVENT TIMELINE
    # ════════════════════════════════════════════════════════════════
    timeline: list[tuple[float, str]] = []   # (sort_key_s, display_line)

    def _add(time_s: float | None, frame_n: int | None, label: str, detail: str = ""):
        if time_s is not None:
            fn  = frame_n if frame_n is not None else round(time_s * fps)
            line = f"{label:<22}  frame {fn:>4}  ({time_s:.2f}s)"
            if detail:
                line += f"  — {detail}"
            timeline.append((time_s, line))
        else:
            timeline.append((float("inf"), f"{label:<22}  *** NOT DETECTED ***"))

    # Window open
    if window_open_frame is not None:
        _add(window_open_frame / fps, window_open_frame, "WINDOW_OPEN")
    else:
        timeline.append((float("inf"), f"{'WINDOW_OPEN':<22}  *** NOT REACHED ***"))

    # Baseline
    for key, bl in [("BASELINE LEFT",  tracker.baseline_L),
                    ("BASELINE RIGHT", tracker.baseline_R)]:
        if bl.captured:
            fn = baseline_frame.get(key, 0)
            detail = (f"elbow={bl.elbow_angle:.1f}°  "
                      f"knee={bl.front_knee_angle:.1f}°  "
                      f"stance={bl.stance_width:.1f}px")
            _add(fn / fps, fn, key, detail)
        else:
            timeline.append((float("inf"),
                f"{key:<22}  *** NOT CAPTURED ***"
                "  → raise BASELINE_STABLE_STD or lower BASELINE_MIN_FRAMES"))

    # Lights
    if args.detect_lights:
        light_event_map: dict[LightEvent, int] = {}
        for r in lights.history:
            for e in r.events:
                if e not in light_event_map:
                    light_event_map[e] = r.frame_idx
        for event, label in [
            (LightEvent.LIGHT_ON_L,    "LIGHT_ON_L"),
            (LightEvent.LIGHT_ON_R,    "LIGHT_ON_R"),
            (LightEvent.LIGHT_ON_BOTH, "LIGHT_ON_BOTH"),
            (LightEvent.WINDOW_CLOSE,  "WINDOW_CLOSE"),
        ]:
            fn = light_event_map.get(event)
            _add(fn / fps if fn is not None else None, fn, label)

    # Audio
    if audio_result is not None:
        _add(audio_result.allez_time_s,  None, "ALLEZ")
        _add(audio_result.buzzer_time_s, None, "BUZZER")

    # Print sorted timeline
    timeline.sort(key=lambda x: x[0])
    print("\n" + "="*60)
    print("EVENT TIMELINE")
    print("="*60)
    for _, line in timeline:
        print(f"  {line}")

    if args.save_overlay:
        print(f"\nOverlay images → {overlay_dir}")


if __name__ == "__main__":
    main()
