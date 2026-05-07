# ════════════════════════════════════════════════════════════════
#  config.py  —  All tunable parameters for the analysis pipeline
# ════════════════════════════════════════════════════════════════

# ── Paths ────────────────────────────────────────────────────────
VIDEO_ROOT  = r"C:\Users\yz\Desktop\Fencing Auto Referee\video\奥运-标记视频"
OUTPUT_ROOT = r"C:\Users\yz\Desktop\Fencing Auto Referee\code\output"
MODEL_PATH  = r"C:\Users\yz\Desktop\Fencing Auto Referee\models\yolov8x-pose.pt"

# ── Video processing ─────────────────────────────────────────────
TARGET_FPS = 30  # resample all videos to this FPS for consistent frame math

# ── Pose detection ───────────────────────────────────────────────
KEYPOINT_CONF_THRESHOLD  = 0.5   # below this confidence → keypoint flagged unreliable
TRACKER_MAX_MISS_FRAMES  = 5     # carry forward last known assignment for up to N frames

# ── Baseline calibration ─────────────────────────────────────────
# Tune BASELINE_MIN_FRAMES to match how long the static on-guard period lasts in your videos.
BASELINE_MIN_FRAMES  = 10            # frames to average; also used as the stability window
BASELINE_STABLE_STD  = 4.0           # °: max std-dev of elbow AND knee over the window
BASELINE_ELBOW_RANGE = (50,  160)    # (min, max) °: plausible on-guard elbow angle
BASELINE_KNEE_RANGE  = (80,  165)    # (min, max) °: plausible on-guard knee angle

# ── Analysis window ──────────────────────────────────────────────
MOVEMENT_START_VEL = 6.0    # px/frame: CoM velocity that opens the analysis window

# ── Timing thresholds (frames) ───────────────────────────────────
SIMULTANEOUS_ARM_THRESHOLD    =  3  # arm onset gap ≤ this → arms simultaneous; check lunge
SIMULTANEOUS_LUNGE_THRESHOLD  =  3  # lunge onset gap ≤ this → lunges simultaneous; check launch
SIMULTANEOUS_LAUNCH_THRESHOLD = 10  # launch gap ≤ this → launches simultaneous; penalty tiebreak only

# ── Attack / arm extension ───────────────────────────────────────
ATTACK_ELBOW_ABS    = 140  # °: absolute elbow angle must exceed this → arm extended
ATTACK_SHOULDER_ABS =  60  # °: shoulder angle (hip→shoulder→elbow) must exceed this
ATTACK_MIN_FRAMES   =   2  # consecutive frames both conditions must hold to confirm

# ── Attack pause (continuity) ────────────────────────────────────
ATTACK_PAUSE_REVERSAL_DEG = 8   # elbow drops this much during attack → pause
PAUSE_MIN_FRAMES          = 4   # consecutive "stopped" frames to confirm pause (>3 frames)
PAUSE_FOOT_VEL_PX         = 2   # px/frame: foot below this = stopped
PAUSE_COM_VEL_PX          = 2   # px/frame: CoM below this = stopped

# ── Launch / step forward ────────────────────────────────────────
LAUNCH_COM_VEL_PX        = 8    # px/frame: CoM velocity to trigger LAUNCH
LAUNCH_MIN_FRAMES        = 2
LAUNCH_ARM_WINDOW_FRAMES = 5    # arm extension must start within N frames of CoM surge
STEP_VEL_MIN_PX          = 2    # px/frame: minimum CoM velocity for a step (vs. noise)
STEP_MIN_FRAMES          = 3

# ── Lunge detection ──────────────────────────────────────────────
STEP_ANKLE_MIN_PX       = 4     # px/frame: front ankle speed to trigger STEP_START
LUNGE_ANKLE_MIN_PX      = 10    # px/frame: front ankle speed to start lunge
REAR_ANKLE_STILL_PX     = 3     # px/frame: rear ankle "still" threshold
LUNGE_FRONT_KNEE_ABS    = 150   # °: front knee angle (absolute) must exceed this → lunge begin
LUNGE_THIGH_ANGLE_L     = 100   # °: inter-thigh angle threshold for LEFT athlete → lunge begin
LUNGE_THIGH_ANGLE_R     = 120   # °: inter-thigh angle threshold for RIGHT athlete → lunge begin
LUNGE_WIDTH_RATIO       = 2.0   # ankle separation > ratio × baseline → lunge begin
LUNGE_PEAK_DROP         = 0.1   # stance ratio must drop by this much from peak to confirm LUNGE_FULL
FOOT_LAND_VEL_PX        = 2     # px/frame: front ankle velocity drops → foot landed

# ── Light detection (HSV) ────────────────────────────────────────
LIGHT_MIN_AREA_PX          = 5000  # px²: minimum blob area — apparatus lights are large
LIGHT_CONFIRM_FRAMES       = 2     # consecutive frames blob must persist to confirm
LIGHT_WINDOW_CLOSE_DELAY   = 10    # frames after LIGHT_ON_BOTH before window closes
# Only search for lights in the bottom portion of the frame (below the piste).
# 0.65 means: only look at pixels whose Y > frame_height * 0.65
LIGHT_ROI_Y_RATIO     = 0.65
RED_LOWER_1  = (0,   150, 150)
RED_UPPER_1  = (10,  255, 255)
RED_LOWER_2  = (160, 150, 150)
RED_UPPER_2  = (180, 255, 255)
GREEN_LOWER  = (40,  100, 100)
GREEN_UPPER  = (90,  255, 255)

# ── Temporal angle smoothing (blur mitigation) ───────────────────
# Inspired by: Lin et al., "Mitigating Blur for Robust 3D Baseball Tracking", MMSports'23
# Fast limb movement causes motion blur → noisy keypoint estimates.
# A short rolling average over ANGLE_SMOOTH_WINDOW frames stabilises
# joint angle readings without introducing perceptible temporal lag.
ANGLE_SMOOTH_WINDOW = 3   # frames: 3 @ 30fps ≈ 100ms lag

# ── Baseline quality check ───────────────────────────────────────
BASELINE_ELBOW_MIN   = 30    # degrees: below this → suspect keypoint failure
BASELINE_ELBOW_MAX   = 170
BASELINE_KNEE_MIN    = 60
BASELINE_KNEE_MAX    = 175
BASELINE_STANCE_MIN_PX = 20  # px: baseline stance narrower than this → ankles likely overlapping
