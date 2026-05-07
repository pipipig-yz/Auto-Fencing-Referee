<div align="right">
  <a href="README.md">🇨🇳 中文</a>
</div>

<h1 align="center">🤺 Fencing Auto Referee</h1>
<p align="center">Automatic priority judgment for sabre fencing using YOLOv8-Pose</p>

---

## Overview

This project performs offline analysis of Olympic-level sabre fencing video to automatically determine which fencer holds **priority** (right-of-way) in each phrase of action.

The system faithfully implements FIE sabre priority rules:

- Per-frame joint angle extraction via pose estimation
- Detection of **attack** (elbow extension), **launch** (CoM surge), and **lunge** (knee extension) events
- Three-step cascade priority judgment algorithm
- Two visualization outputs: event timeline + joint angle curves

**Priority Algorithm (3-step cascade)**

```
STEP 1: arm_gap > 3 frames?  → first mover wins priority
STEP 2: lunge_gap > 3 frames? → first mover wins priority
STEP 3: launch_gap > 10 frames? → first mover wins priority
else → SIMULTANEOUS
```

---

## File Structure

```
Fencing Auto Referee/
├── code/
│   ├── main.py                      # Batch pipeline entry point
│   ├── config.py                    # All tunable parameters
│   ├── requirements.txt
│   ├── PROJECT.md                   # Full design document
│   │
│   ├── modules/
│   │   ├── pose_module.py           # YOLOv8-Pose inference wrapper
│   │   ├── athlete_tracker.py       # L/R assignment + baseline calibration
│   │   ├── piste_module.py          # Piste detection + athlete filtering
│   │   ├── light_module.py          # Scoring light HSV detection
│   │   ├── audio_module.py          # Audio event detection
│   │   └── event_aggregator.py      # Event detection + priority judgment (core)
│   │
│   ├── utils/
│   │   ├── angle_utils.py           # Joint angle computation (COCO-17)
│   │   ├── video_utils.py           # Video I/O + frame sampling
│   │   ├── json_utils.py            # JSON serialisation
│   │   └── debug_utils.py           # Debug drawing helpers
│   │
│   ├── visualization/
│   │   ├── timeline_plot.py         # Plot 1: horizontal swim-lane event timeline
│   │   └── angle_plot.py            # Plot 2: joint angle curves (4 subplots)
│   │
│   └── scripts/
│       ├── debug_aggregator.py      # Single-video debug + event frame images
│       ├── batch_priority_report.py # Batch priority step log
│       ├── debug_attack_scan.py     # Per-frame attack onset debug
│       └── debug_lunge_scan.py      # Per-frame lunge debug
│
├── video/
│   └── 奥运-标记视频/
│       ├── 互中/                    # Mutual attack clips (5 videos, L/R labelled)
│       └── 同时/                    # Simultaneous attack clips (5 videos)
│
├── models/                          # YOLOv8 weights (local only, not tracked)
├── reference/                       # FIE rulebook + reference papers
└── present file/                    # Project slides
```

---

## Installation

**Requirements:** Python 3.10+, virtual environment recommended

```bash
cd code
pip install -r requirements.txt
```

**Download model weights** (ultralytics auto-downloads on first run):

```python
from ultralytics import YOLO
YOLO("yolov8x-pose.pt")   # downloads automatically
```

Move the downloaded `yolov8x-pose.pt` to the `models/` directory, then verify paths in `config.py`:

```python
MODEL_PATH  = r"<project root>\models\yolov8x-pose.pt"
VIDEO_ROOT  = r"<project root>\video\奥运-标记视频"
OUTPUT_ROOT = r"<project root>\code\output"
```

---

## Usage

All commands are run from the `code/` directory.

### `main.py` — Batch Pipeline

Runs the full pipeline on all labelled videos, saves visualizations, and prints an accuracy summary.

```bash
# Process all videos in 互中/ and 同时/
python main.py

# Single video
python main.py --video "..\video\奥运-标记视频\互中\互中1左侧.mp4"

# Specific subfolder only
python main.py --folders 互中

# Skip piste detection (if piste not visible)
python main.py --no-piste
```

**Console output:**

```
============================================================
  Fencing Auto Referee — Batch Run
  Videos : 10
  Output : output/
============================================================

[1/10]  互中1左侧.mp4 … LEFT_SCORES  ✓
[2/10]  互中2右侧.mp4 … RIGHT_SCORES ✓
...

============================================================
  ACCURACY SUMMARY
============================================================
  Video                                GT                    Predicted             Match
  互中1左侧                            LEFT_SCORES           LEFT_SCORES           ✓
  互中2右侧                            RIGHT_SCORES          RIGHT_SCORES          ✓
  ...
  Labelled videos : 10
  Correct         : X
  Accuracy        : X.X%
============================================================
```

**File output (one subdirectory per video):**

```
output/
└── 互中1左侧/
    ├── timeline.png       # Event timeline
    ├── joint_angles.png   # Joint angle curves
    └── summary.txt        # Verdict + key timestamps
```

---

### `scripts/debug_aggregator.py` — Single-Video Event Debug

Runs the full pipeline and **saves an annotated frame image for every detected event**, then prints the priority verdict.

```bash
# Default video (互中1左侧.mp4)
python scripts/debug_aggregator.py

# Specify video
python scripts/debug_aggregator.py --video "..\video\奥运-标记视频\互中\互中1左侧.mp4"

# Print per-frame lunge table, save only lunge event frames
python scripts/debug_aggregator.py --lunge
```

**Output:** Frame images saved to `output/aggregator/<video_stem>/00123_L_ATTACK_START.jpg` etc.

---

### `scripts/batch_priority_report.py` — Batch Priority Step Log

Processes all videos and prints **the decision detail at each step** of the cascade algorithm, useful for threshold tuning.

```bash
python scripts/batch_priority_report.py
```

**Output:**

```
────────────────────────────────────────────────────────────
FILE    : 互中1左侧.mp4
VERDICT : LEFT_SCORES

Priority step 1: arm_gap=8 > threshold=3 → LEFT arm first
VERDICT: LEFT_SCORES (no penalty)
```

---

### `scripts/debug_attack_scan.py` / `debug_lunge_scan.py` — Per-Frame Debug

Prints per-frame attack / lunge detection values inside the analysis window, useful for diagnosing threshold issues.

```bash
python scripts/debug_attack_scan.py
python scripts/debug_lunge_scan.py
```

---

## Output Visualizations

### Plot 1: Event Timeline (`timeline.png`)

Horizontal swim-lane chart showing the timing of all events for both athletes within the analysis window.

- Each lane represents one event type (attack, launch, lunge, pause, lights)
- Vertical markers indicate key timestamps
- Verdict annotated at the top

### Plot 2: Joint Angle Curves (`joint_angles.png`)

Four subplots (top to bottom):

| Subplot | Contents |
|---------|----------|
| L Attack | Left elbow angle (solid) + shoulder angle (dash-dot) |
| L Lunge  | Left front knee (solid) + inter-thigh angle (dashed) + width ratio (right axis) |
| R Attack | Right elbow + shoulder |
| R Lunge  | Right front knee + inter-thigh + width ratio |

- Dashed horizontal lines: baseline values (on-guard pose)
- Dotted horizontal lines: detection thresholds (angle value annotated on y-axis left)
- Vertical markers: baseline / ARM↑ / LAUNCH / LUNGE / LIGHT ON
- Right panel: all threshold values + per-athlete baseline measurements

---

## Video Dataset

| Filename pattern | Ground truth label |
|-----------------|-------------------|
| `互中*左侧*` | LEFT_SCORES |
| `互中*右侧*` | RIGHT_SCORES |
| `同时*` | SIMULTANEOUS |

10 labelled videos total: 5 mutual attack (互中) + 5 simultaneous (同时).

---

## Key Parameters (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ATTACK_ELBOW_ABS` | 140° | Absolute elbow angle threshold for attack detection |
| `ATTACK_SHOULDER_ABS` | 60° | Absolute shoulder angle threshold |
| `LUNGE_FRONT_KNEE_ABS` | 150° | Absolute front knee threshold for lunge detection |
| `LUNGE_THIGH_ANGLE_L` | 100° | Inter-thigh angle threshold for left athlete |
| `LUNGE_THIGH_ANGLE_R` | 120° | Inter-thigh angle threshold for right athlete |
| `LUNGE_WIDTH_RATIO` | 2.0× | Stance width vs. baseline ratio threshold |
| `SIMULTANEOUS_ARM_THRESHOLD` | 3 frames | Arm gap ≤ this → simultaneous |
| `TARGET_FPS` | 30 | Unified processing frame rate |

---

## Dependencies

- [ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv8-Pose
- OpenCV
- NumPy
- Matplotlib
- librosa (audio module)
