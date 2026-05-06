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
SIMULTANEOUS_ARM_THRESHOLD  = 6  # arm onset gap ≤ this → simultaneous (arm level)
SIMULTANEOUS_STEP_THRESHOLD = 6  # launch onset gap ≤ this → simultaneous (step level)
ARM_BEFORE_STEP_MIN_GAP     = 2  # arm must precede step by > this to be valid

# ── Attack / arm extension ───────────────────────────────────────
ATTACK_ELBOW_DELTA              = 35  # ° above baseline to trigger ATTACK_START
ATTACK_ELBOW_RATE_DEG_PER_FRAME = 4   # °/frame minimum rate for onset
ATTACK_ELBOW_FULL_DELTA         = 55  # ° above baseline = ATTACK_FULL
ATTACK_MIN_FRAMES               = 2   # consecutive frames to confirm onset

# ── Attack pause (continuity) ────────────────────────────────────
ATTACK_PAUSE_REVERSAL_DEG = 8   # elbow drops this much during attack → pause
PAUSE_MIN_FRAMES          = 3   # consecutive "stopped" frames to confirm pause
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
LUNGE_KNEE_DELTA_DEG    = -20   # ° below baseline front knee → lunge starting
LUNGE_KNEE_FULL_DELTA_DEG = -35 # ° below baseline front knee → full lunge
LUNGE_WIDTH_RATIO       = 1.35  # ankle separation > ratio × baseline → lunge
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

# ── Baseline quality check ───────────────────────────────────────
BASELINE_ELBOW_MIN  = 30    # degrees: below this → suspect keypoint failure
BASELINE_ELBOW_MAX  = 170
BASELINE_KNEE_MIN   = 60
BASELINE_KNEE_MAX   = 175
