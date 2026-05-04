# Fencing Auto Referee System — Project Documentation

**Working directory:** `C:\Users\yz\Desktop\Fencing Auto Referee\`
**Hardware:** NVIDIA RTX 4060 · Windows 11
**Analysis mode:** Offline video batch processing
**Weapon:** Sabre
**Language:** English (all code, output, UI); Chinese permitted in dev discussion
**FIE Rules Reference:** `FIE_Technical_Rules_Dec2025.pdf` (root folder)
**Document version:** v1.2 · 2026-05-04

---

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [FIE Sabre Rules — Key Excerpts](#2-fie-sabre-rules--key-excerpts)
3. [Video Dataset](#3-video-dataset)
4. [Analysis Pipeline & Time Window](#4-analysis-pipeline--time-window)
5. [Baseline Calibration](#5-baseline-calibration)
6. [Detection Parameters & Thresholds](#6-detection-parameters--thresholds)
7. [Event Definitions](#7-event-definitions)
8. [Priority Judgment Algorithm](#8-priority-judgment-algorithm)
9. [Visualization Output](#9-visualization-output)
10. [Config Parameters Reference](#10-config-parameters-reference)
11. [Module Descriptions](#11-module-descriptions)
12. [Directory Structure](#12-directory-structure)
13. [Dependencies](#13-dependencies)
14. [Development Roadmap](#14-development-roadmap)
15. [Key Design Decisions](#15-key-design-decisions)

---

## 1. Project Goal

Automatically analyze sabre fencing bout videos to produce:

| Goal | Description |
|------|-------------|
| **Baseline calibration** | Capture each athlete's individual joint angles and distances from the static on-guard position before "Allez" |
| **Athlete pose detection** | YOLOv8-Pose tracking 17 keypoints per athlete within the active analysis window |
| **Action event recognition** | Rule-based engine classifying actions from joint angle changes relative to each athlete's baseline |
| **Scoring apparatus light detection** | HSV color-domain detection of red/green light activation (defines the end of the analysis window) |
| **Timeline visualization** | Plot 1: per-athlete event timeline with light events |
| **Joint angle visualization** | Plot 2: joint angle curves + event markers for both athletes |

**Scope (current phase):** Scenarios `coup double` (互中) and `simultaneous action` (同时) only.

---

## 2. FIE Sabre Rules — Key Excerpts

Source: FIE Technical Rules, December 2025 (t.96–t.106).
Full document: `FIE_Technical_Rules_Dec2025.pdf`

### 2.1 Method of Making a Hit (t.96)

- The sabre is a weapon for **thrusting and cutting** with both the cutting edge and the back of the blade.
- Hits with the cutting edge, flat, or back of blade are all valid.
- Hitting with the guard is **forbidden** and annuls the hit.
- The **fleche** and any forward movement where the rear foot completely passes the front foot are **forbidden** (t.101.5). Any hit by the offending fencer is annulled.

### 2.2 Valid Target (t.97)

The target is **everything above the hip line** — the entire upper body, both arms, and the head/mask.

### 2.3 Correct Attack — The Priority Rule (t.101–t.102)

An attack is **correctly executed** when:
- The **arm straightens** (cutting edge threatening the valid target) **before** the lunge or step is initiated.
- For a **lunge**: arm extension onset must precede lunge onset; hit must arrive no later than when the front foot lands.
- For a **step-forward-lunge**: arm extension onset must precede the step-forward onset.

> **Critical:** If the arm is NOT extended before the step/lunge begins, the attack is faulty and the opponent's counter-attack gains priority.

### 2.4 Parry and Riposte (t.105)

- A successful parry grants the **right to riposte immediately**.
- A delayed riposte (pause after parry) gives the attacker the right to renew the attack.

### 2.5 Judging Hits When Both Fencers Are Hit (t.106) ← Core of this project

#### Case A — Simultaneous Action (t.106.1)
Both fencers conceived and executed an attack **at the same time**.
- Apparatus: both lights on.
- Result: **no point** awarded to either fencer.

#### Case B — Coup Double (t.106.2–4)
Both lights on, but one fencer made a **clearly faulty action**.
Priority determines who scores.

**Defender is alone counted as hit (attacker scores)** when (t.106.3):
- (a) Defender makes a stop hit on a **simple attack** — stop hit has no priority.
- (b) Defender attempts to **avoid** the hit but fails.
- (c) Defender made a successful parry but **delayed the riposte** — attacker renews.
- (d) Defender makes stop hit during **compound attack** but **not in time**.
- (e) Defender had **point in line**, blade was deflected, then re-attacks instead of parrying.

**Attacker is alone counted as hit (defender scores)** when (t.106.4):
- (a) Attack launched while opponent had **point in line** without deflecting it.
- (b) Tried to find blade, **failed (dérobement)**, continued attack.
- (c) Compound attack — opponent found blade and **riposted immediately**.
- (d) **Bent arm or paused** during compound attack — opponent stop-hits during pause.
- (e) Opponent's **stop hit arrived one fencing time before** attacker's final movement.
- (f) **Remise/redoublement** after opponent's immediate simple riposte.

**Unable to judge:** no clear fault from either side → competitors replaced on guard, no point (t.106.5).

### 2.6 Referee Commands (t.22–23)

| Command | Meaning |
|---------|---------|
| `"On guard!"` | Fencers take guard position |
| `"Are you ready?"` | Check readiness |
| `"Play!" / "Allez"` | Bout starts — **analysis window opens** |
| `"Halt!"` | Bout stops |

---

## 3. Video Dataset

### 3.1 Location

```
C:\Users\yz\Desktop\Fencing Auto Referee\video\奥运-标记视频\
```

### 3.2 Structure (active scope only)

```
奥运-标记视频/
├── 互中/                              # Coup double scenarios
│   ├── 互中1右侧.mp4                  # Coup double — RIGHT fencer scores
│   ├── 互中2右侧.mp4                  # Coup double — RIGHT fencer scores
│   ├── 互中3左侧.mp4                  # Coup double — LEFT fencer scores
│   ├── 互中4右侧.mp4                  # Coup double — RIGHT fencer scores
│   └── 互中5右侧.mp4                  # Coup double — RIGHT fencer scores
└── 同时/                              # Simultaneous action scenarios
    ├── 同时1-1.mp4                    # Simultaneous — no point awarded
    ├── 同时2-1.mp4
    ├── 同时3-1.mp4
    ├── 同时飞剑-1.mp4                 # Simultaneous — involves flying blade
    └── 同时（左侧启动快）-1.mp4       # Simultaneous — left side initiates faster
```

**Out of scope for now:** `复杂进攻右侧-击剑线不成立.mp4`, `防守还击2-1.mp4`

### 3.3 Filename Convention

| Part | Meaning |
|------|---------|
| `互中` | Coup double — apparatus lights both sides, one fencer scores |
| `同时` | Simultaneous action — apparatus lights both sides, no point |
| `右侧` | **Right-side fencer scores** the point |
| `左侧` | **Left-side fencer scores** the point |
| `-1` suffix | Clip index |

### 3.4 Screen Side Convention

- **Left (L):** fencer on the left side of the screen
- **Right (R):** fencer on the right side of the screen

### 3.5 Scenario Comparison

| Scenario | Apparatus | Point |
|----------|-----------|-------|
| **Coup double** (互中) | Both lights on | One fencer scores (priority winner) |
| **Simultaneous** (同时) | Both lights on | Neither scores |

> These two scenarios look identical on the apparatus. The difference is in the **temporal sequence of body movements**, which this project aims to detect.

### 3.6 Future: Piste Distance Calibration

The two fencers' starting lines are exactly **4 metres apart** and are visible as markings on the piste in the video. In a future phase, these lines will be detected to establish a pixel-to-metre conversion factor, enabling real-world distance measurements. For the current phase, all distances use **pixel units**.

---

## 4. Analysis Pipeline & Time Window

### 4.1 Full Per-Video Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│  PHASE 0 — BASELINE CALIBRATION                              │
│                                                              │
│  Condition: both athletes' keypoint velocities < STILL_VEL  │
│             for at least BASELINE_MIN_FRAMES consecutive     │
│             frames (the static on-guard period before Allez) │
│                                                              │
│  Output: per-athlete baseline joint angles + ankle           │
│          separation distance (stored, used as reference)     │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  WINDOW START DETECTION                                      │
│                                                              │
│  After baseline captured:                                    │
│  → Either audio "Allez" is detected (future audio module)    │
│  → Or: first frame where any keypoint velocity exceeds       │
│        MOVEMENT_START_VEL (both athletes were still,         │
│        now one moves)                                        │
│                                                              │
│  T_zero = this frame                                         │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  PHASE 1 — ACTIVE ANALYSIS WINDOW                            │
│                                                              │
│  Start: T_zero (Allez / first movement)                      │
│  End:   T_light (first frame LIGHT_ON_BOTH is detected)      │
│                                                              │
│  All pose detection, angle computation, event recognition,   │
│  and priority judgment operate ONLY within this window.      │
│  Data outside this window is ignored for priority analysis.  │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  PHASE 2 — PRIORITY JUDGMENT & OUTPUT                        │
│                                                              │
│  Apply priority algorithm to events in [T_zero, T_light]     │
│  Compare result with ground truth from filename              │
│  Generate timeline plot, angle plot, JSON log                │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 Time Window Rationale

| Boundary | Logic | Reason |
|----------|-------|--------|
| **Window start** | Allez + athletes static | Prevents spurious events from earlier footwork or warm-up movements from polluting event detection |
| **Window end** | Apparatus light activates | The hit has been registered; any movement after this is irrelevant to priority judgment |

### 4.3 "Both Athletes Static" Detection

```
For each frame in a rolling window of BASELINE_MIN_FRAMES frames:
  velocity_L = max keypoint displacement (px/frame) of left athlete
  velocity_R = max keypoint displacement (px/frame) of right athlete

  if velocity_L < STILL_VEL AND velocity_R < STILL_VEL:
    → frame counts as "still"

If BASELINE_MIN_FRAMES consecutive "still" frames found:
  → baseline calibration runs on this window
  → system ready for window start detection
```

---

## 5. Baseline Calibration

During the static on-guard period, the following values are measured and stored per athlete. All subsequent event thresholds are defined **relative to these baselines**.

### 5.1 Stored Baseline Values

| Variable | Description | Unit |
|----------|-------------|------|
| `baseline_elbow_angle` | Elbow angle in on-guard stance | degrees |
| `baseline_shoulder_angle` | Shoulder angle in on-guard stance | degrees |
| `baseline_front_knee_angle` | Front knee angle in on-guard stance | degrees |
| `baseline_rear_knee_angle` | Rear knee angle in on-guard stance | degrees |
| `baseline_trunk_lean` | Trunk forward lean angle vs. vertical | degrees |
| `baseline_stance_width` | Pixel distance between front and rear ankle | px |
| `baseline_com_x` | Horizontal position of hip midpoint | px |
| `baseline_wrist_height` | Wrist Y relative to hip Y | px |

### 5.2 How Baselines Are Used

All event trigger conditions are expressed as **deltas from baseline** or **ratios to baseline**:

```
Example — ATTACK_START trigger:
  NOT: elbow_angle > 150°
  BUT: (elbow_angle - baseline_elbow_angle) > ATTACK_ELBOW_DELTA
       AND rate_of_change(elbow_angle) > ATTACK_ELBOW_RATE_DEG_PER_FRAME

Example — LUNGE trigger:
  NOT: ankle_separation > 1.4 × some_fixed_value
  BUT: ankle_separation > baseline_stance_width × LUNGE_WIDTH_RATIO
```

### 5.3 Baseline Quality Check

If the measured baseline values fall outside plausible human ranges (e.g., elbow angle < 30° or > 170°), log a warning — keypoint detection may have failed for that athlete in that clip.

---

## 6. Detection Parameters & Thresholds

All values below reference named constants in `config.py`. No values are hardcoded in module logic.

### 6.1 Parameter P1 — Sword Arm Elbow Angle

```
Keypoints:  shoulder → elbow → wrist  (sword arm side)
Baseline:   baseline_elbow_angle  (captured during static period)
Unit:       degrees
```

| Event | Trigger condition | Config variable |
|-------|------------------|-----------------|
| `ATTACK_START` | `elbow_angle - baseline` > `ATTACK_ELBOW_DELTA` AND rate > `ATTACK_ELBOW_RATE_DEG_PER_FRAME` for ≥ `ATTACK_MIN_FRAMES` frames | `ATTACK_ELBOW_DELTA`, `ATTACK_ELBOW_RATE_DEG_PER_FRAME`, `ATTACK_MIN_FRAMES` |
| `ATTACK_FULL` | `elbow_angle - baseline` > `ATTACK_ELBOW_FULL_DELTA` | `ATTACK_ELBOW_FULL_DELTA` |
| `ATTACK_PAUSE` | After `ATTACK_START`: elbow_angle decreases by > `ATTACK_PAUSE_REVERSAL_DEG` over ≥ `PAUSE_MIN_FRAMES` frames | `ATTACK_PAUSE_REVERSAL_DEG`, `PAUSE_MIN_FRAMES` |

**Suggested starting values (tune per video):**
```python
ATTACK_ELBOW_DELTA            = 35    # degrees above baseline
ATTACK_ELBOW_RATE_DEG_PER_FRAME = 4  # °/frame minimum rate
ATTACK_ELBOW_FULL_DELTA       = 55    # degrees above baseline = fully extended
ATTACK_PAUSE_REVERSAL_DEG     = 8     # reversal depth to count as pause
ATTACK_MIN_FRAMES             = 2     # consecutive frames to confirm onset
```

### 6.2 Parameter P2 — Sword Arm Shoulder Angle

```
Keypoints:  hip → shoulder → elbow  (sword arm side)
Baseline:   baseline_shoulder_angle
Unit:       degrees
```

Used to distinguish **thrust** (elbow extends forward, shoulder angle increases) from **cut** (shoulder rotates, elbow angle changes differently). Auxiliary signal — does not independently trigger events but is logged for visualization and future refinement.

### 6.3 Parameter P3 — Centre of Mass Forward Velocity

```
Keypoints:  midpoint of (left_hip, right_hip)  → X coordinate
Computed:   velocity = Δcom_x / frame  (pixels/frame)
Baseline:   com_x near-zero velocity during static period
Unit:       pixels/frame
```

| Event | Trigger condition | Config variable |
|-------|------------------|-----------------|
| `LAUNCH` | `com_velocity_x` > `LAUNCH_COM_VEL_PX` for ≥ `LAUNCH_MIN_FRAMES` frames AND `ATTACK_START` is concurrent or follows within `LAUNCH_ARM_WINDOW_FRAMES` | `LAUNCH_COM_VEL_PX`, `LAUNCH_MIN_FRAMES`, `LAUNCH_ARM_WINDOW_FRAMES` |
| `STEP_FORWARD` | `com_velocity_x` between `STEP_VEL_MIN_PX` and `LAUNCH_COM_VEL_PX` for ≥ `STEP_MIN_FRAMES` frames AND no arm extension concurrent | `STEP_VEL_MIN_PX`, `STEP_MIN_FRAMES` |
| `STEP_BACK` | `com_velocity_x` < `−STEP_VEL_MIN_PX` | — |

**Suggested starting values:**
```python
LAUNCH_COM_VEL_PX             = 8     # pixels/frame for launch
STEP_VEL_MIN_PX               = 2     # pixels/frame minimum for step (vs. noise)
LAUNCH_MIN_FRAMES             = 2
STEP_MIN_FRAMES               = 3
LAUNCH_ARM_WINDOW_FRAMES      = 5     # arm extension must start within N frames of CoM surge
```

### 6.4 Parameter P4 — Front Ankle Forward Displacement

```
Keypoints:  front ankle X coordinate  (frame-to-frame difference)
"Front ankle" = ankle closer to opponent
Baseline:   near-zero displacement during static period
Unit:       pixels/frame
```

| Event | Trigger condition | Config variable |
|-------|------------------|-----------------|
| `STEP_START` | Front ankle displacement > `STEP_ANKLE_MIN_PX` for ≥ 2 frames | `STEP_ANKLE_MIN_PX` |
| `LUNGE_STEP_START` | Front ankle displacement > `LUNGE_ANKLE_MIN_PX` AND rear ankle relatively still (< `REAR_ANKLE_STILL_PX`) | `LUNGE_ANKLE_MIN_PX`, `REAR_ANKLE_STILL_PX` |
| `FRONT_FOOT_LAND` | Front ankle displacement drops below `FOOT_LAND_VEL_PX` after a lunge step | `FOOT_LAND_VEL_PX` |

**Suggested starting values:**
```python
STEP_ANKLE_MIN_PX             = 4     # pixels/frame
LUNGE_ANKLE_MIN_PX            = 10    # pixels/frame (faster than step)
REAR_ANKLE_STILL_PX           = 3     # rear ankle "still" threshold
FOOT_LAND_VEL_PX              = 2     # foot has landed when velocity drops below this
```

### 6.5 Parameter P5 — Front Knee Angle

```
Keypoints:  hip → knee → ankle  (front leg)
Baseline:   baseline_front_knee_angle
Unit:       degrees
```

| Event | Trigger condition | Config variable |
|-------|------------------|-----------------|
| `LUNGE_START` | `front_knee_angle` decreases by > `LUNGE_KNEE_DELTA_DEG` below baseline AND ankle separation > `baseline_stance_width × LUNGE_WIDTH_RATIO` | `LUNGE_KNEE_DELTA_DEG`, `LUNGE_WIDTH_RATIO` |
| `LUNGE_FULL` | `front_knee_angle - baseline` < `LUNGE_KNEE_FULL_DELTA_DEG` (angle is significantly smaller than baseline) | `LUNGE_KNEE_FULL_DELTA_DEG` |

**Suggested starting values:**
```python
LUNGE_KNEE_DELTA_DEG          = -20   # knee must decrease by 20° from baseline to start
LUNGE_KNEE_FULL_DELTA_DEG     = -35   # knee 35° below baseline = full lunge
LUNGE_WIDTH_RATIO             = 1.35  # ankle separation must also be 1.35× baseline
```

### 6.6 Parameter P6 — Ankle Separation Distance

```
Computed:   |front_ankle_x - rear_ankle_x|
Baseline:   baseline_stance_width  (pixels)
Unit:       ratio to baseline_stance_width
```

| Event | Trigger condition | Config variable |
|-------|------------------|-----------------|
| `LUNGE_FULL` (supporting) | `ankle_separation / baseline_stance_width` > `LUNGE_WIDTH_RATIO` | `LUNGE_WIDTH_RATIO` |

### 6.7 Parameter P7 — Attack Pause (Continuity Check)

Applies **only within the active analysis window, and only after LAUNCH has been triggered**.

```
Condition: LAUNCH has occurred (athlete is in forward motion)
Check:     over a rolling window of PAUSE_MIN_FRAMES frames:
             front ankle displacement < PAUSE_FOOT_VEL_PX
             AND com_velocity_x < PAUSE_COM_VEL_PX
If true → ATTACK_PAUSE event triggered
```

| Config variable | Meaning |
|-----------------|---------|
| `PAUSE_MIN_FRAMES` | Number of consecutive "stopped" frames to confirm pause |
| `PAUSE_FOOT_VEL_PX` | Foot velocity below this = foot stopped |
| `PAUSE_COM_VEL_PX` | CoM velocity below this = body stopped |

**Suggested starting values:**
```python
PAUSE_MIN_FRAMES              = 3
PAUSE_FOOT_VEL_PX             = 2     # pixels/frame
PAUSE_COM_VEL_PX              = 2     # pixels/frame
```

---

## 7. Event Definitions

### 7.1 Full Event Type Table

| Event code | Description | Category | Primary trigger |
|------------|-------------|----------|----------------|
| `BASELINE_CAPTURED` | Static baseline stored for both athletes | System | Both still for `BASELINE_MIN_FRAMES` |
| `WINDOW_OPEN` | Analysis window starts | System | After baseline: first movement detected |
| `WINDOW_CLOSE` | Analysis window ends | System | `LIGHT_ON_BOTH` detected |
| `L_LAUNCH` | Left fencer launches (attack initiation) | Action | CoM surge + arm begins extending |
| `R_LAUNCH` | Right fencer launches | Action | — |
| `L_STEP_FWD` | Left step forward | Action | CoM forward, no arm extension, no lunge |
| `R_STEP_FWD` | Right step forward | Action | — |
| `L_STEP_BACK` | Left step back | Action | CoM rearward |
| `R_STEP_BACK` | Right step back | Action | — |
| `L_LUNGE` | Left lunge begins | Action | Front knee drops + ankle separation increases |
| `R_LUNGE` | Right lunge begins | Action | — |
| `L_LUNGE_FULL` | Left lunge complete | Action | Knee fully bent, wide stance |
| `R_LUNGE_FULL` | Right lunge complete | Action | — |
| `L_ATTACK_START` | Left arm begins extending | Action | Elbow angle rising above baseline + delta |
| `R_ATTACK_START` | Right arm begins extending | Action | — |
| `L_ATTACK_FULL` | Left arm fully extended | Action | Elbow angle well above baseline |
| `R_ATTACK_FULL` | Right arm fully extended | Action | — |
| `L_FRONT_FOOT_LAND` | Left front foot lands (lunge end) | Action | Front ankle velocity drops |
| `R_FRONT_FOOT_LAND` | Right front foot lands | Action | — |
| `L_ATTACK_PAUSE` | Left attack interrupted (pause detected) | Action | CoM + foot stopped post-LAUNCH |
| `R_ATTACK_PAUSE` | Right attack interrupted | Action | — |
| `LIGHT_ON_L` | Left apparatus light activates | Light | HSV red/green blob, left side |
| `LIGHT_ON_R` | Right apparatus light activates | Light | HSV red/green blob, right side |
| `LIGHT_ON_BOTH` | Both lights on — closes analysis window | Light | Both sides simultaneously |
| `LIGHT_OFF` | All lights extinguished | Light | Blobs disappear |

### 7.2 Key Event Timestamps Used by Priority Algorithm

| Symbol | Event | Role in algorithm |
|--------|-------|------------------|
| `T_arm_L` | `L_ATTACK_START` frame | Arm extension onset for Left |
| `T_arm_R` | `R_ATTACK_START` frame | Arm extension onset for Right |
| `T_step_L` | `L_LUNGE` or `L_STEP_FWD` frame | Step/lunge onset for Left |
| `T_step_R` | `R_LUNGE` or `R_STEP_FWD` frame | Step/lunge onset for Right |
| `T_launch_L` | `L_LAUNCH` frame | CoM surge onset for Left |
| `T_launch_R` | `R_LAUNCH` frame | CoM surge onset for Right |
| `T_light` | `LIGHT_ON_BOTH` frame | Analysis window end |
| `pause_L` | `L_ATTACK_PAUSE` present? | Boolean: Left attack had pause |
| `pause_R` | `R_ATTACK_PAUSE` present? | Boolean: Right attack had pause |

### 7.3 Typical Event Sequences

**Coup double — right scores:**
```
T=0.00  WINDOW_OPEN
T=0.05  R_ATTACK_START       ← right arm extends first
T=0.08  R_LAUNCH             ← right CoM surges forward
T=0.12  L_ATTACK_START       ← left arm extends later
T=0.15  L_LAUNCH
T=0.30  R_LUNGE              ← right arm was BEFORE right step ✓ (valid attack)
T=0.32  L_LUNGE              ← left arm was AFTER left step ✗ (faulty: arm too late)
T=0.45  R_FRONT_FOOT_LAND
T=0.47  L_FRONT_FOOT_LAND
T=0.50  LIGHT_ON_BOTH → WINDOW_CLOSE
→ Priority: Right (arm first, valid). Left's arm was late (faulty).
→ Result: RIGHT scores.
```

**Simultaneous action:**
```
T=0.00  WINDOW_OPEN
T=0.05  R_ATTACK_START
T=0.05  L_ATTACK_START       ← both arms extend within SIMULTANEOUS_ARM_THRESHOLD
T=0.06  R_LAUNCH
T=0.06  L_LAUNCH             ← both launch within SIMULTANEOUS_STEP_THRESHOLD
T=0.35  R_LUNGE + L_LUNGE
T=0.50  LIGHT_ON_BOTH → WINDOW_CLOSE
→ No pause in either attack, both initiated simultaneously
→ Result: NO POINT (simultaneous).
```

---

## 8. Priority Judgment Algorithm

All timing comparisons use frame counts. All threshold constants are defined in `config.py`.

### 8.1 Core Principle

> A fencer has priority if they **first correctly initiated an attack** — meaning their arm began extending clearly before their opponent's, AND their arm extension preceded their own step/lunge by at least `ARM_BEFORE_STEP_MIN_GAP` frames.

> The analysis window is strictly bounded: from `WINDOW_OPEN` to `WINDOW_CLOSE` (light activation). Events outside this window are ignored.

### 8.2 Validity Check — Is the Attack Correctly Executed?

For each athlete X ∈ {L, R}:

```
valid_attack_X = (T_step_X - T_arm_X) > ARM_BEFORE_STEP_MIN_GAP
                  ↑
                  arm must precede step by more than this gap
                  (not just "arm before step", but clearly before)
```

If `valid_attack_X` is False: the athlete's attack is faulty (arm did not lead the step). The opponent's stop-hit or counter-attack gains priority (t.106.4a/b).

### 8.3 Continuity Check — Did the Attack Pause?

```
continuous_X = (no L_ATTACK_PAUSE / R_ATTACK_PAUSE event for athlete X
                within the analysis window)
```

If `continuous_X` is False: the attack was interrupted. The opponent's action during the pause is valid (t.106.4d).

### 8.4 Priority Decision Tree (Full)

```
BOTH LIGHTS ON  →  WINDOW_CLOSE
│
│  [Step 1] Compare arm extension onset times
│
├─ |T_arm_L - T_arm_R| > SIMULTANEOUS_ARM_THRESHOLD
│   │   (clear time difference in arm extension)
│   │
│   ├─ T_arm_L < T_arm_R  (Left arm earlier)
│   │    first_mover = L,  other = R
│   │    └─→ [Jump to Step 3: Validity & Continuity Check]
│   │
│   └─ T_arm_R < T_arm_L  (Right arm earlier)
│        first_mover = R,  other = L
│        └─→ [Jump to Step 3: Validity & Continuity Check]
│
└─ |T_arm_L - T_arm_R| ≤ SIMULTANEOUS_ARM_THRESHOLD
    │   (arm onsets too close to distinguish)
    │
    │  [Step 2] Fall back: compare launch (CoM / step) onset times
    │
    ├─ |T_launch_L - T_launch_R| > SIMULTANEOUS_STEP_THRESHOLD
    │   │   (clear time difference in launch)
    │   │
    │   ├─ T_launch_L < T_launch_R  →  first_mover = L
    │   └─ T_launch_R < T_launch_L  →  first_mover = R
    │        └─→ [Jump to Step 3: Validity & Continuity Check]
    │
    └─ |T_launch_L - T_launch_R| ≤ SIMULTANEOUS_STEP_THRESHOLD
        │   (launch also simultaneous)
        │
        ├─ continuous_L AND continuous_R
        │      → ══ SIMULTANEOUS — NO POINT ══
        │
        └─ NOT continuous_L  →  L had pause  →  R scores
           NOT continuous_R  →  R had pause  →  L scores
           BOTH paused       →  → SIMULTANEOUS (unable to judge)


[Step 3] Validity & Continuity Check for first_mover
│
├─ valid_attack_first_mover AND continuous_first_mover
│      → ══ PRIORITY: first_mover  →  first_mover SCORES ══
│
├─ NOT valid_attack_first_mover
│      (first mover's arm did not lead the step)
│      → first_mover's attack is faulty
│      → check other fencer:
│        ├─ valid_attack_other → ══ other SCORES ══
│        └─ not valid_attack_other → → SIMULTANEOUS (both faulty)
│
└─ NOT continuous_first_mover  (first mover had a pause)
       → first_mover loses priority
       → check other fencer:
         ├─ continuous_other → ══ other SCORES ══
         └─ not continuous_other → → SIMULTANEOUS (both paused)
```

### 8.5 Time Comparison Parameters

| Config constant | Default | Meaning |
|-----------------|---------|---------|
| `SIMULTANEOUS_ARM_THRESHOLD` | 6 frames (≈200ms @30fps) | Max frame gap for arm onsets to be "simultaneous" |
| `SIMULTANEOUS_STEP_THRESHOLD` | 6 frames | Max frame gap for launch onsets to be "simultaneous" |
| `ARM_BEFORE_STEP_MIN_GAP` | 2 frames | Arm must precede step by at least this many frames to be valid |

> All three constants must be tuned by reviewing the labeled videos. Starting values are based on the FIE "one period of fencing time" (≈200ms) guideline.

---

## 9. Visualization Output

### 9.1 Plot 1 — Event Timeline

```
Time (s) →     0.0   0.1   0.2   0.3   0.4   0.5
               │     │     │     │     │     │
 Left  ─────── │ ────┼──[ATK_START]──[LUNGE]──── │
               │     │     │     │     │     │
 Right ─────── │ [ATK_START]─[LAUNCH]──[LUNGE]── │
               │     │     │     │     │     │
 Light ─────── │     │     │     │     │ ●BOTH
               │     │     │     │     │     │
              T=0                            T=light
           (WINDOW_OPEN)              (WINDOW_CLOSE)
```

**Visual elements:**
- Horizontal axis: time in seconds, zero = `WINDOW_OPEN`
- Gray background: baseline calibration region (before T=0)
- Left athlete events: **blue** bars (durations) or triangles (instants)
- Right athlete events: **orange** bars or triangles
- Light event: **red** vertical line
- Analysis window boundary: shaded region end at light event
- Priority winner shaded: winning athlete's region highlighted

### 9.2 Plot 2 — Joint Angle Curves

Two vertically stacked subplots sharing the time axis:

**Top — Left athlete:**
- Elbow angle (blue solid line)
- Front knee angle (blue dashed line)
- Horizontal dashed reference lines at: `baseline + ATTACK_ELBOW_DELTA`, `baseline + ATTACK_ELBOW_FULL_DELTA`, `baseline_knee + LUNGE_KNEE_FULL_DELTA`
- Vertical event markers: `ATTACK_START` (thin solid), `LUNGE` (thin dashed), `ATTACK_PAUSE` (red)

**Bottom — Right athlete:**
- Same, orange color scheme

Baseline values shown as horizontal annotations.

### 9.3 Output Files

```
code/output/
└── {video_name}/
    ├── timeline.png        # Plot 1: event timeline
    ├── joint_angles.png    # Plot 2: joint angle curves with baselines
    ├── events.json         # Complete timestamped event sequence
    ├── baseline.json       # Stored baseline values for both athletes
    ├── pose_data.csv       # Per-frame keypoint coordinates
    └── summary.txt         # Priority result + ground truth label + match/mismatch
```

---

## 10. Config Parameters Reference

All parameters are defined in `config.py`. The file is organized into labeled sections so thresholds can be adjusted without touching module code.

```python
# ════════════════════════════════════════════════════════════════
#  config.py  —  All tunable parameters for the analysis pipeline
# ════════════════════════════════════════════════════════════════

# ── Paths ────────────────────────────────────────────────────────
VIDEO_ROOT   = r"C:\Users\yz\Desktop\Fencing Auto Referee\video\奥运-标记视频"
OUTPUT_ROOT  = r"C:\Users\yz\Desktop\Fencing Auto Referee\code\output"
MODEL_PATH   = r"C:\Users\yz\Desktop\Fencing Auto Referee\models\yolov8x-pose.pt"

# ── Video processing ─────────────────────────────────────────────
TARGET_FPS          = 30       # resample all videos to this FPS for consistent frame math

# ── Baseline calibration ─────────────────────────────────────────
BASELINE_MIN_FRAMES = 20       # consecutive still frames required to capture baseline
STILL_VEL           = 3.0      # px/frame: max keypoint velocity to count as "still"

# ── Analysis window ──────────────────────────────────────────────
MOVEMENT_START_VEL  = 6.0      # px/frame: CoM velocity that opens the analysis window

# ── Timing thresholds (frames) ───────────────────────────────────
SIMULTANEOUS_ARM_THRESHOLD    = 6    # arm onset gap ≤ this → simultaneous (arm level)
SIMULTANEOUS_STEP_THRESHOLD   = 6    # launch onset gap ≤ this → simultaneous (step level)
ARM_BEFORE_STEP_MIN_GAP       = 2    # arm must precede step by > this to be valid

# ── Attack / arm extension ───────────────────────────────────────
ATTACK_ELBOW_DELTA            = 35   # °  above baseline to trigger ATTACK_START
ATTACK_ELBOW_RATE_DEG_PER_FRAME = 4  # °/frame  minimum rate for onset
ATTACK_ELBOW_FULL_DELTA       = 55   # °  above baseline = ATTACK_FULL
ATTACK_MIN_FRAMES             = 2    # consecutive frames to confirm onset

# ── Attack pause (continuity) ────────────────────────────────────
ATTACK_PAUSE_REVERSAL_DEG     = 8    # elbow drops this much during attack → pause
PAUSE_MIN_FRAMES              = 3    # consecutive "stopped" frames to confirm pause
PAUSE_FOOT_VEL_PX             = 2    # px/frame: foot below this = stopped
PAUSE_COM_VEL_PX              = 2    # px/frame: CoM below this = stopped

# ── Launch / step forward ────────────────────────────────────────
LAUNCH_COM_VEL_PX             = 8    # px/frame: CoM velocity to trigger LAUNCH
LAUNCH_MIN_FRAMES             = 2
LAUNCH_ARM_WINDOW_FRAMES      = 5    # arm extension must start within N frames of CoM surge
STEP_VEL_MIN_PX               = 2    # px/frame: minimum CoM velocity for a step
STEP_MIN_FRAMES               = 3

# ── Lunge detection ──────────────────────────────────────────────
LUNGE_ANKLE_MIN_PX            = 10   # px/frame: front ankle speed to start lunge
REAR_ANKLE_STILL_PX           = 3    # px/frame: rear ankle "still" threshold
LUNGE_KNEE_DELTA_DEG          = -20  # ° below baseline front knee → lunge starting
LUNGE_KNEE_FULL_DELTA_DEG     = -35  # ° below baseline front knee → full lunge
LUNGE_WIDTH_RATIO             = 1.35 # ankle separation > ratio × baseline → lunge
FOOT_LAND_VEL_PX              = 2    # px/frame: front ankle velocity drops → foot landed

# ── Light detection (HSV) ────────────────────────────────────────
LIGHT_MIN_AREA_PX             = 500  # px²: minimum blob area to register as light on
LIGHT_CONFIRM_FRAMES          = 2    # consecutive frames blob must persist to confirm
RED_LOWER_1  = (0,   120, 100)
RED_UPPER_1  = (10,  255, 255)
RED_LOWER_2  = (160, 120, 100)
RED_UPPER_2  = (180, 255, 255)
GREEN_LOWER  = (40,  80,  80)
GREEN_UPPER  = (90,  255, 255)
```

---

## 11. Module Descriptions

### 11.1 Pose Module (`modules/pose_module.py`)

**Model:** `yolov8x-pose.pt`

**Output per frame:** 17 keypoints (x, y, confidence) per detected person.

**COCO keypoint indices:**
```
 0: nose          1: left_eye       2: right_eye
 3: left_ear      4: right_ear      5: left_shoulder
 6: right_shoulder  7: left_elbow   8: right_elbow
 9: left_wrist   10: right_wrist   11: left_hip
12: right_hip    13: left_knee     14: right_knee
15: left_ankle   16: right_ankle
```

Responsibilities:
- Run YOLOv8-Pose on each frame
- Assign detections to Left and Right athlete by bounding box X centre
- Compute all joint angles and derived values per frame
- Flag low-confidence keypoints (< `KEYPOINT_CONF_THRESHOLD`) as unreliable

### 11.2 Athlete Tracker (`modules/athlete_tracker.py`)

- Maintains Left/Right identity across frames using bounding box X
- Handles brief detection loss (carries forward last known assignment for up to `TRACKER_MAX_MISS_FRAMES` frames)
- Detects and stores baseline values during the static calibration phase

### 11.3 Light Detection Module (`modules/light_module.py`)

- Applies HSV thresholds to each frame to find red and green blobs
- Assigns blobs to Left or Right side by X position
- Requires blob area > `LIGHT_MIN_AREA_PX` and persistence for `LIGHT_CONFIRM_FRAMES`
- Outputs `LIGHT_ON_L`, `LIGHT_ON_R`, `LIGHT_ON_BOTH`, `LIGHT_OFF` events with frame timestamps

### 11.4 Event Aggregator (`modules/event_aggregator.py`)

- Receives: per-frame joint angle data (from pose module) + light events
- Enforces analysis window: ignores events outside `[WINDOW_OPEN, WINDOW_CLOSE]`
- Detects all action events using thresholds from `config.py`
- Applies the priority judgment algorithm (Section 8)
- Outputs: structured event list + priority verdict

### 11.5 Visualizer (`visualization/`)

- `timeline_plot.py`: draws Plot 1 from the event list
- `angle_plot.py`: draws Plot 2 from per-frame angle data and baseline values
- Both plots mark the analysis window boundaries and the priority verdict

---

## 12. Directory Structure

```
Fencing Auto Referee/
├── FIE_Technical_Rules_Dec2025.pdf
├── code/
│   ├── PROJECT.md
│   ├── requirements.txt
│   ├── config.py                    # ← all tunable parameters here
│   ├── main.py                      # batch entry point
│   │
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── pose_module.py
│   │   ├── athlete_tracker.py
│   │   ├── light_module.py
│   │   └── event_aggregator.py
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── timeline_plot.py
│   │   └── angle_plot.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── video_utils.py
│   │   ├── angle_utils.py
│   │   └── json_utils.py
│   │
│   └── output/
│
├── video/
│   └── 奥运-标记视频/
│       ├── 互中/
│       └── 同时/
│
└── models/
    └── yolov8x-pose.pt
```

---

## 13. Dependencies

### Python 3.10 or 3.11

### `requirements.txt`

```
ultralytics>=8.2.0
torch>=2.2.0
torchvision>=0.17.0
opencv-python>=4.9.0
numpy>=1.26.0
pandas>=2.2.0
matplotlib>=3.8.0
scipy>=1.12.0
tqdm>=4.66.0
```

### CUDA (RTX 4060)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### System: FFmpeg in PATH (for future audio module)

---

## 14. Development Roadmap

### Phase 1 — Project Setup ✓ (current)

- [x] Project documentation
- [x] FIE rules downloaded and reviewed
- [ ] Create directory skeleton
- [ ] `config.py` — all parameters
- [ ] `requirements.txt`
- [ ] `utils/angle_utils.py`
- [ ] `utils/video_utils.py`

### Phase 2 — Detection Modules

- [ ] `modules/pose_module.py` — inference + angle extraction
- [ ] `modules/athlete_tracker.py` — baseline calibration + L/R assignment
- [ ] `modules/light_module.py` — HSV light + window close detection

### Phase 3 — Event Recognition & Priority

- [ ] `modules/event_aggregator.py` — all events + analysis window enforcement
- [ ] Priority algorithm (Section 8)
- [ ] Validate on labeled videos; tune `config.py` thresholds

### Phase 4 — Visualization

- [ ] `visualization/timeline_plot.py`
- [ ] `visualization/angle_plot.py`

### Phase 5 — Batch Entry Point & Reporting

- [ ] `main.py`
- [ ] Accuracy summary (predicted scorer vs. ground truth filename label)

### Future

- [ ] Audio "Allez" detection for precise window start
- [ ] Piste 4m line detection for pixel-to-metre calibration
- [ ] Extend to parry/riposte scenarios

---

## 15. Key Design Decisions

### Pixel units for distances (current phase)
All distance measurements use raw pixel values. Relative ratios (e.g., stance width) provide enough scale-invariance for the current video set. Future: piste 4m line detection will provide a pixel-to-metre conversion.

### Per-athlete baseline calibration (not fixed ranges)
Each athlete's on-guard angles differ. Using individual baselines eliminates inter-athlete variation and makes thresholds more robust. Baselines are captured from the static period before Allez.

### Analysis window strictly bounded by Allez and light activation
Events before Allez (warm-up movements) and after the light (post-hit reactions) are excluded. This prevents spurious events from corrupting priority judgment.

### Time comparisons require minimum gap (not bare < / >)
To avoid false priority calls from single-frame noise, all "A happened before B" assertions require `(T_B - T_A) > MIN_GAP`. All gap constants are in `config.py`.

### Two-stage simultaneous detection
First tests arm extension timing; if inconclusive, tests launch timing; then checks continuity. This matches how a human referee thinks: the primary signal is who started the arm action, with footwork as a tiebreaker.

### Rule-based engine (not ML classifier)
Too few labeled videos for training. A rule engine is interpretable, directly tied to FIE rules, and tunable. Can be upgraded to a classifier when more data is available.

### No audio module in current phase
Body movement timing is the primary signal. Audio ("Allez") will improve window-start precision when added; the current fallback is first-movement detection.

---

*Updated: 2026-05-04 · FIE Rules: December 2025 (t.96–t.106)*
