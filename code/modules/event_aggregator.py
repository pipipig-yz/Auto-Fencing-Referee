"""Event aggregation and priority judgment for sabre fencing.

Responsibilities:
- Receive per-frame FrameData (from AthleteTracker) and FrameLightResult (from LightModule).
- Manage the analysis window: WINDOW_OPEN → WINDOW_CLOSE.
- Detect action events within the window (arm, step, lunge, pause).
- Apply the priority judgment algorithm (PROJECT.md §8) to produce a Verdict.
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum

import config
from modules.athlete_tracker import AthleteBaseline, AthleteFrame, FrameData
from modules.light_module import FrameLightResult, LightEvent

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
#  Enumerations
# ────────────────────────────────────────────────────────────────────────────

class EventType(str, Enum):
    # System events
    BASELINE_CAPTURED  = "BASELINE_CAPTURED"
    WINDOW_OPEN        = "WINDOW_OPEN"
    WINDOW_CLOSE       = "WINDOW_CLOSE"

    # Arm events — Left
    L_ATTACK_START     = "L_ATTACK_START"
    L_ATTACK_FULL      = "L_ATTACK_FULL"
    L_ATTACK_PAUSE     = "L_ATTACK_PAUSE"

    # Arm events — Right
    R_ATTACK_START     = "R_ATTACK_START"
    R_ATTACK_FULL      = "R_ATTACK_FULL"
    R_ATTACK_PAUSE     = "R_ATTACK_PAUSE"

    # Step / launch events — Left
    L_LAUNCH           = "L_LAUNCH"
    L_STEP_BACK        = "L_STEP_BACK"
    L_LUNGE            = "L_LUNGE"

    # Step / launch events — Right
    R_LAUNCH           = "R_LAUNCH"
    R_STEP_BACK        = "R_STEP_BACK"
    R_LUNGE            = "R_LUNGE"

    # Light events (mirrored from LightModule for unified timeline)
    LIGHT_ON_L         = "LIGHT_ON_L"
    LIGHT_ON_R         = "LIGHT_ON_R"
    LIGHT_ON_BOTH      = "LIGHT_ON_BOTH"
    LIGHT_OFF          = "LIGHT_OFF"


class Verdict(str, Enum):
    PENDING         = "PENDING"          # analysis window not yet closed
    LEFT_SCORES     = "LEFT_SCORES"      # left fencer wins priority
    RIGHT_SCORES    = "RIGHT_SCORES"     # right fencer wins priority
    SIMULTANEOUS    = "SIMULTANEOUS"     # simultaneous action — no point
    UNABLE_TO_JUDGE = "UNABLE_TO_JUDGE"  # both actions faulty, no clear priority


# ────────────────────────────────────────────────────────────────────────────
#  Core data classes
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class EventRecord:
    """A single detected event with its frame timestamp."""
    event_type: EventType
    frame_idx:  int


@dataclass
class FrameAngleRecord:
    """Per-frame angle snapshot recorded during the analysis window."""
    frame_idx:   int
    # Left athlete
    elbow_L:     float   # sword-arm elbow angle (°)
    shoulder_L:  float   # sword-arm shoulder angle (°)
    knee_L:      float   # front knee angle (°)
    thigh_L:     float   # inter-thigh angle (°)
    width_L:     float   # stance width (px)
    com_vel_L:   float   # CoM velocity (px/frame, +ve = forward)
    # Right athlete
    elbow_R:     float
    shoulder_R:  float
    knee_R:      float
    thigh_R:     float
    width_R:     float
    com_vel_R:   float


@dataclass
class AggregatorResult:
    """Complete output produced by EventAggregator after one video."""
    events:             list[EventRecord]       = field(default_factory=list)
    angle_records:      list[FrameAngleRecord]  = field(default_factory=list)
    window_open_frame:  int | None = None
    window_close_frame: int | None = None
    verdict:            Verdict = Verdict.PENDING

    # Key timestamps extracted from events (None if event not detected)
    t_arm_L:    int | None = None   # frame of L_ATTACK_START
    t_arm_R:    int | None = None   # frame of R_ATTACK_START
    t_step_L:   int | None = None   # frame of L_LUNGE
    t_step_R:   int | None = None   # frame of R_LUNGE
    t_launch_L: int | None = None   # frame of L_LAUNCH
    t_launch_R: int | None = None   # frame of R_LAUNCH
    pause_L:    bool = False        # ATTACK_PAUSE detected (≥ PAUSE_MIN_FRAMES after launch)
    pause_R:    bool = False
    step_back_L: bool = False       # L_STEP_BACK detected after L_LAUNCH
    step_back_R: bool = False       # R_STEP_BACK detected after R_LAUNCH


# ────────────────────────────────────────────────────────────────────────────
#  Internal per-athlete state helpers
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class _ArmState:
    onset_frames:   int  = 0
    attack_started: bool = False


@dataclass
class _StepState:
    # LAUNCH
    launch_fwd_frames: int  = 0
    launch_emitted:    bool = False

    # STEP_BACK — emitted once per continuous backward phase when cumulative
    # displacement exceeds 1× the athlete's baseline stance width
    step_back_distance:           float = 0.0
    step_back_emitted_this_phase: bool  = False

    # LUNGE
    lunge_emitted: bool = False

    # ATTACK_PAUSE (§6.7): triggered after LAUNCH when foot + CoM both stop
    pause_frames:         int  = 0
    attack_pause_emitted: bool = False


# ────────────────────────────────────────────────────────────────────────────
#  Main aggregator
# ────────────────────────────────────────────────────────────────────────────

class EventAggregator:
    """Consumes per-frame data and produces a final AggregatorResult.

    Usage (called from main.py or a pipeline runner):
        agg = EventAggregator()
        for frame_idx, frame_bgr in video:
            pose_data  = tracker.update(frame_idx, detections)
            light_data = light_module.update(frame_idx, frame_bgr)
            agg.update(pose_data, light_data, tracker.baseline_L, tracker.baseline_R)
        result = agg.finalize()
    """

    def __init__(self) -> None:
        self._events:             list[EventRecord]      = []
        self._angle_records:      list[FrameAngleRecord] = []
        self._window_open:        bool = False
        self._window_closed:      bool = False
        self._window_open_frame:  int | None = None
        self._window_close_frame: int | None = None
        self._baseline_emitted:   bool = False

        self._arm_L  = _ArmState()
        self._arm_R  = _ArmState()
        self._step_L = _StepState()
        self._step_R = _StepState()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def update(
        self,
        pose:       FrameData,
        light:      FrameLightResult,
        baseline_L: AthleteBaseline,
        baseline_R: AthleteBaseline,
    ) -> list[EventRecord]:
        """Process one frame. Returns events newly emitted this frame."""
        _before = len(self._events)
        frame_idx = pose.frame_idx

        # ── Baseline captured (emit once) ──────────────────────────────
        if pose.baseline_captured and not self._baseline_emitted:
            self._emit(EventType.BASELINE_CAPTURED, frame_idx)
            self._baseline_emitted = True

        # ── Mirror light events into unified timeline ───────────────────
        _LIGHT_MAP = {
            LightEvent.LIGHT_ON_L:    EventType.LIGHT_ON_L,
            LightEvent.LIGHT_ON_R:    EventType.LIGHT_ON_R,
            LightEvent.LIGHT_ON_BOTH: EventType.LIGHT_ON_BOTH,
            LightEvent.LIGHT_OFF:     EventType.LIGHT_OFF,
        }
        for le in light.events:
            if le in _LIGHT_MAP:
                self._emit(_LIGHT_MAP[le], frame_idx)

        # ── Window open detection ───────────────────────────────────────
        if (not self._window_open
                and self._baseline_emitted
                and not self._window_closed):
            vel_L = abs(pose.L.com_velocity) if pose.L.detected else 0.0
            vel_R = abs(pose.R.com_velocity) if pose.R.detected else 0.0
            if max(vel_L, vel_R) >= config.MOVEMENT_START_VEL:
                self._window_open = True
                self._window_open_frame = frame_idx
                self._emit(EventType.WINDOW_OPEN, frame_idx)
                logger.info("frame %d: WINDOW_OPEN (vel_L=%.1f vel_R=%.1f)",
                            frame_idx, vel_L, vel_R)

        # ── Window close detection ──────────────────────────────────────
        if (self._window_open
                and not self._window_closed
                and LightEvent.WINDOW_CLOSE in light.events):
            self._window_closed = True
            self._window_close_frame = frame_idx
            self._emit(EventType.WINDOW_CLOSE, frame_idx)
            logger.info("frame %d: WINDOW_CLOSE", frame_idx)

        # ── Action event detection (only inside the active window) ──────
        if self._window_open and not self._window_closed:
            self._detect_arm_events("L", pose.L, baseline_L, frame_idx)
            self._detect_arm_events("R", pose.R, baseline_R, frame_idx)
            self._detect_step_events("L", pose.L, baseline_L, frame_idx)
            self._detect_step_events("R", pose.R, baseline_R, frame_idx)

            # ── Record per-frame angles for visualization ─────────────────
            self._angle_records.append(FrameAngleRecord(
                frame_idx  = frame_idx,
                elbow_L    = pose.L.angles.get("elbow_angle",        math.nan),
                shoulder_L = pose.L.angles.get("shoulder_angle",     math.nan),
                knee_L     = pose.L.angles.get("front_knee_angle",   math.nan),
                thigh_L    = pose.L.angles.get("inter_thigh_angle",  math.nan),
                width_L    = pose.L.angles.get("stance_width",       math.nan),
                com_vel_L  = pose.L.com_velocity,
                elbow_R    = pose.R.angles.get("elbow_angle",        math.nan),
                shoulder_R = pose.R.angles.get("shoulder_angle",     math.nan),
                knee_R     = pose.R.angles.get("front_knee_angle",   math.nan),
                thigh_R    = pose.R.angles.get("inter_thigh_angle",  math.nan),
                width_R    = pose.R.angles.get("stance_width",       math.nan),
                com_vel_R  = pose.R.com_velocity,
            ))

        return self._events[_before:]

    def finalize(self) -> AggregatorResult:
        """Extract timestamps, run priority algorithm, log result."""
        result = AggregatorResult(
            events=list(self._events),
            angle_records=list(self._angle_records),
            window_open_frame=self._window_open_frame,
            window_close_frame=self._window_close_frame,
        )

        # ── Extract key timestamps from event list ──────────────────────
        for ev in result.events:
            t  = ev.frame_idx
            et = ev.event_type
            if et == EventType.L_ATTACK_START and result.t_arm_L is None:
                result.t_arm_L = t
            elif et == EventType.R_ATTACK_START and result.t_arm_R is None:
                result.t_arm_R = t
            elif et == EventType.L_LUNGE and result.t_step_L is None:
                result.t_step_L = t
            elif et == EventType.R_LUNGE and result.t_step_R is None:
                result.t_step_R = t
            elif et == EventType.L_LAUNCH and result.t_launch_L is None:
                result.t_launch_L = t
            elif et == EventType.R_LAUNCH and result.t_launch_R is None:
                result.t_launch_R = t
            elif et == EventType.L_ATTACK_PAUSE:
                result.pause_L = True
            elif et == EventType.R_ATTACK_PAUSE:
                result.pause_R = True

        # Step_back counts only if it occurs after the fencer's own launch
        for ev in result.events:
            if ev.event_type == EventType.L_STEP_BACK:
                if result.t_launch_L is not None and ev.frame_idx >= result.t_launch_L:
                    result.step_back_L = True
            elif ev.event_type == EventType.R_STEP_BACK:
                if result.t_launch_R is not None and ev.frame_idx >= result.t_launch_R:
                    result.step_back_R = True

        # ── Priority judgment ───────────────────────────────────────────
        result.verdict = _judge_priority(result)

        _log_result(result)
        return result

    def reset(self) -> None:
        self.__init__()

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _emit(self, event_type: EventType, frame_idx: int) -> None:
        self._events.append(EventRecord(event_type=event_type, frame_idx=frame_idx))

    # ── CP3: arm events ────────────────────────────────────────────────

    def _detect_arm_events(
        self,
        side:      str,
        af:        AthleteFrame,
        baseline:  AthleteBaseline,
        frame_idx: int,
    ) -> None:
        state    = self._arm_L  if side == "L" else self._arm_R
        ev_start = EventType.L_ATTACK_START if side == "L" else EventType.R_ATTACK_START

        if not af.detected:
            state.onset_frames = 0
            return

        elbow    = af.angles.get("elbow_angle",    math.nan)
        shoulder = af.angles.get("shoulder_angle", math.nan)

        if math.isnan(elbow) or math.isnan(shoulder):
            state.onset_frames = 0
            return

        if not state.attack_started:
            if (elbow    > config.ATTACK_ELBOW_ABS
                    and shoulder > config.ATTACK_SHOULDER_ABS):
                state.onset_frames += 1
                if state.onset_frames >= config.ATTACK_MIN_FRAMES:
                    state.attack_started = True
                    self._emit(ev_start, frame_idx)
                    logger.info("frame %d: %s (elbow=%.1f° shoulder=%.1f°)",
                                frame_idx, ev_start, elbow, shoulder)
            else:
                state.onset_frames = 0

    # ── CP4: step / launch / lunge / pause events ──────────────────────

    def _detect_step_events(
        self,
        side:      str,
        af:        AthleteFrame,
        baseline:  AthleteBaseline,
        frame_idx: int,
    ) -> None:
        state     = self._step_L  if side == "L" else self._step_R
        arm_state = self._arm_L   if side == "L" else self._arm_R

        ev_launch    = EventType.L_LAUNCH       if side == "L" else EventType.R_LAUNCH
        ev_step_back = EventType.L_STEP_BACK    if side == "L" else EventType.R_STEP_BACK
        ev_lunge     = EventType.L_LUNGE        if side == "L" else EventType.R_LUNGE
        ev_pause     = EventType.L_ATTACK_PAUSE if side == "L" else EventType.R_ATTACK_PAUSE

        if not af.detected or not baseline.captured:
            return

        # Forward velocity: positive = moving toward opponent
        # L faces right (+X), R faces left (-X)
        fwd_vel = af.com_velocity if side == "L" else -af.com_velocity

        # ── LAUNCH ──────────────────────────────────────────────────────
        if not state.launch_emitted:
            if fwd_vel >= config.LAUNCH_COM_VEL_PX:
                state.launch_fwd_frames += 1
                if state.launch_fwd_frames >= config.LAUNCH_MIN_FRAMES:
                    state.launch_emitted = True
                    self._emit(ev_launch, frame_idx)
                    logger.info("frame %d: %s (fwd_vel=%.1f px/f)", frame_idx, ev_launch, fwd_vel)
            else:
                state.launch_fwd_frames = 0

        # ── STEP_BACK ───────────────────────────────────────────────────
        # Emitted once per continuous backward phase when cumulative backward
        # displacement exceeds 1× the athlete's baseline stance width.
        in_back_range = fwd_vel <= -config.STEP_VEL_MIN_PX
        if in_back_range:
            state.step_back_distance += abs(fwd_vel)
            if (not state.step_back_emitted_this_phase
                    and not math.isnan(baseline.stance_width)
                    and baseline.stance_width > 0
                    and state.step_back_distance >= baseline.stance_width):
                state.step_back_emitted_this_phase = True
                self._emit(ev_step_back, frame_idx)
                logger.info("frame %d: %s (back_dist=%.1fpx threshold=%.1fpx)",
                            frame_idx, ev_step_back,
                            state.step_back_distance, baseline.stance_width)
        else:
            state.step_back_distance = 0.0
            state.step_back_emitted_this_phase = False

        # ── LUNGE ───────────────────────────────────────────────────────
        # Three conditions must all pass simultaneously:
        #   1. Front knee angle (absolute) > LUNGE_FRONT_KNEE_ABS  (leg extended)
        #   2. Inter-thigh angle at hip midpoint > LUNGE_THIGH_ANGLE (legs spread)
        #   3. Ankle separation > LUNGE_WIDTH_RATIO × baseline stance width
        knee        = af.angles.get("front_knee_angle",  math.nan)
        inter_thigh = af.angles.get("inter_thigh_angle", math.nan)
        stance      = af.angles.get("stance_width",       math.nan)

        thigh_thresh = (config.LUNGE_THIGH_ANGLE_L if side == "L"
                        else config.LUNGE_THIGH_ANGLE_R)

        if (not math.isnan(knee) and not math.isnan(inter_thigh)
                and not math.isnan(stance) and baseline.stance_width > 0):
            stance_ratio = stance / baseline.stance_width

            if not state.lunge_emitted:
                if (knee > config.LUNGE_FRONT_KNEE_ABS
                        and inter_thigh > thigh_thresh
                        and stance_ratio > config.LUNGE_WIDTH_RATIO):
                    state.lunge_emitted = True
                    self._emit(ev_lunge, frame_idx)
                    logger.info(
                        "frame %d: %s (knee=%.1f° thigh=%.1f° stance_ratio=%.2f)",
                        frame_idx, ev_lunge, knee, inter_thigh, stance_ratio,
                    )

        # ── ATTACK_PAUSE (§6.7): foot + CoM stop after LAUNCH ───────────
        if state.launch_emitted and not state.attack_pause_emitted:
            stopped = (af.front_ankle_vel < config.PAUSE_FOOT_VEL_PX
                       and abs(af.com_velocity) < config.PAUSE_COM_VEL_PX)
            if stopped:
                state.pause_frames += 1
                if state.pause_frames >= config.PAUSE_MIN_FRAMES:
                    state.attack_pause_emitted = True
                    self._emit(ev_pause, frame_idx)
                    logger.info("frame %d: %s (foot_vel=%.1f com_vel=%.1f)",
                                frame_idx, ev_pause, af.front_ankle_vel, af.com_velocity)
            else:
                state.pause_frames = 0


# ────────────────────────────────────────────────────────────────────────────
#  CP5: Priority judgment algorithm (PROJECT.md §8)
# ────────────────────────────────────────────────────────────────────────────

def _judge_priority(r: AggregatorResult) -> Verdict:
    """Three-step cascading priority decision tree.

    Step 1 — arm onset gap > SIMULTANEOUS_ARM_THRESHOLD   → first mover by arm
    Step 2 — lunge onset gap > SIMULTANEOUS_LUNGE_THRESHOLD → first mover by lunge
    Step 3 — launch order: whoever launched first gets priority (any gap)

    Penalty (applied at every step): a fencer forfeits priority if, after their
    own launch, they paused (≥ PAUSE_MIN_FRAMES) OR retreated (STEP_BACK).
    If both forfeit, result is SIMULTANEOUS.
    """

    t_arm_L,    t_arm_R    = r.t_arm_L,    r.t_arm_R
    t_lunge_L,  t_lunge_R  = r.t_step_L,   r.t_step_R
    t_launch_L, t_launch_R = r.t_launch_L, r.t_launch_R

    def lost_priority(side: str) -> bool:
        """True when this fencer paused or retreated after launching."""
        if side == "L":
            return r.pause_L or r.step_back_L
        return r.pause_R or r.step_back_R

    def verdict_for(winner: str) -> Verdict:
        return Verdict.LEFT_SCORES if winner == "L" else Verdict.RIGHT_SCORES

    def resolve(first: str, step: int) -> Verdict:
        """Apply penalty check and return final Verdict for a determined first mover."""
        other = "R" if first == "L" else "L"
        first_loses = lost_priority(first)
        other_loses = lost_priority(other)
        logger.info(
            "Priority step %d: first_mover=%s penalty=%s | other=%s penalty=%s",
            step, first, first_loses, other, other_loses,
        )
        if not first_loses:
            return verdict_for(first)
        if not other_loses:
            return verdict_for(other)
        return Verdict.SIMULTANEOUS  # both forfeited priority

    # ── Step 1: compare arm onset times ────────────────────────────────
    if t_arm_L is None and t_arm_R is None:
        logger.warning("Priority: no arm events detected → UNABLE_TO_JUDGE")
        return Verdict.UNABLE_TO_JUDGE

    if t_arm_L is not None and t_arm_R is not None:
        arm_gap = abs(t_arm_L - t_arm_R)
    else:
        arm_gap = config.SIMULTANEOUS_ARM_THRESHOLD + 1  # only one side extended

    if arm_gap > config.SIMULTANEOUS_ARM_THRESHOLD:
        first = "L" if (t_arm_R is None or
                        (t_arm_L is not None and t_arm_L < t_arm_R)) else "R"
        logger.info("Priority step 1: arm gap=%d > %d → first_mover=%s",
                    arm_gap, config.SIMULTANEOUS_ARM_THRESHOLD, first)
        return resolve(first, step=1)

    logger.info("Priority step 1: arm gap=%d ≤ %d → check lunge",
                arm_gap, config.SIMULTANEOUS_ARM_THRESHOLD)

    # ── Step 2: compare lunge onset times ──────────────────────────────
    if t_lunge_L is not None and t_lunge_R is not None:
        lunge_gap = abs(t_lunge_L - t_lunge_R)
    elif t_lunge_L is not None or t_lunge_R is not None:
        lunge_gap = config.SIMULTANEOUS_LUNGE_THRESHOLD + 1  # only one side lunged
    else:
        lunge_gap = 0  # neither lunged — fall through to launch

    if lunge_gap > config.SIMULTANEOUS_LUNGE_THRESHOLD:
        first = "L" if (t_lunge_R is None or
                        (t_lunge_L is not None and t_lunge_L < t_lunge_R)) else "R"
        logger.info("Priority step 2: lunge gap=%d > %d → first_mover=%s",
                    lunge_gap, config.SIMULTANEOUS_LUNGE_THRESHOLD, first)
        return resolve(first, step=2)

    logger.info("Priority step 2: lunge gap=%d ≤ %d → check launch",
                lunge_gap, config.SIMULTANEOUS_LUNGE_THRESHOLD)

    # ── Step 3: compare launch times ─────────────────────────────────
    if t_launch_L is not None and t_launch_R is not None:
        launch_gap = abs(t_launch_L - t_launch_R)
        if launch_gap > config.SIMULTANEOUS_LAUNCH_THRESHOLD:
            first: str | None = "L" if t_launch_L < t_launch_R else "R"
        else:
            first = None  # within threshold → simultaneous
    elif t_launch_L is not None:
        first = "L"
    elif t_launch_R is not None:
        first = "R"
    else:
        first = None  # neither launched

    if first is not None:
        logger.info("Priority step 3: launch gap=%d > %d → first_mover=%s",
                    launch_gap if t_launch_L is not None and t_launch_R is not None else -1,
                    config.SIMULTANEOUS_LAUNCH_THRESHOLD, first)
        return resolve(first, step=3)

    logger.info("Priority step 3: launches simultaneous (gap ≤ %d) → penalty tiebreak",
                config.SIMULTANEOUS_LAUNCH_THRESHOLD)
    if lost_priority("L") and not lost_priority("R"):
        return Verdict.RIGHT_SCORES
    if lost_priority("R") and not lost_priority("L"):
        return Verdict.LEFT_SCORES
    return Verdict.SIMULTANEOUS


# ────────────────────────────────────────────────────────────────────────────
#  CP6: Result logging
# ────────────────────────────────────────────────────────────────────────────

_SYS_EVENTS = {
    EventType.BASELINE_CAPTURED,
    EventType.WINDOW_OPEN, EventType.WINDOW_CLOSE,
    EventType.LIGHT_ON_L, EventType.LIGHT_ON_R,
    EventType.LIGHT_ON_BOTH, EventType.LIGHT_OFF,
}
_L_EVENTS = {
    EventType.L_ATTACK_START, EventType.L_ATTACK_FULL, EventType.L_ATTACK_PAUSE,
    EventType.L_LAUNCH, EventType.L_STEP_BACK, EventType.L_LUNGE,
}
_R_EVENTS = {
    EventType.R_ATTACK_START, EventType.R_ATTACK_FULL, EventType.R_ATTACK_PAUSE,
    EventType.R_LAUNCH, EventType.R_STEP_BACK, EventType.R_LUNGE,
}


def _log_result(result: AggregatorResult, fps: float = config.TARGET_FPS) -> None:
    """Print the full event timeline (3-column) and verdict to the logger."""
    C0, C1, C2 = 22, 20, 20   # column widths for system / left / right
    sep  = "=" * (14 + C0 + C1 + C2 + 4)
    dash = "-" * (14 + C0 + C1 + C2 + 4)

    header = (f"  {'':11s}  {'SYSTEM':<{C0}}  {'LEFT':<{C1}}  {'RIGHT':<{C2}}")
    logger.info(sep)
    logger.info("EVENT TIMELINE")
    logger.info(header)
    logger.info(sep)

    for ev in result.events:
        t_sec = ev.frame_idx / fps
        ts    = f"f{ev.frame_idx:04d} {t_sec:5.2f}s"
        name  = ev.event_type.value

        if ev.event_type in _SYS_EVENTS:
            col0, col1, col2 = name, "", ""
        elif ev.event_type in _L_EVENTS:
            # strip leading "L_" for brevity
            col0, col1, col2 = "", name[2:], ""
        else:
            col0, col1, col2 = "", "", name[2:]

        logger.info("  %s  %-*s  %-*s  %-*s",
                    ts, C0, col0, C1, col1, C2, col2)

    logger.info(dash)
    logger.info("KEY TIMESTAMPS  (frame | seconds @ %.0f fps)", fps)
    for label, frame in [
        ("T_arm_L",    result.t_arm_L),
        ("T_arm_R",    result.t_arm_R),
        ("T_step_L",   result.t_step_L),
        ("T_step_R",   result.t_step_R),
        ("T_launch_L", result.t_launch_L),
        ("T_launch_R", result.t_launch_R),
    ]:
        if frame is not None:
            logger.info("  %-12s  f%04d  %.2fs", label, frame, frame / fps)
        else:
            logger.info("  %-12s  —", label)
    logger.info("  pause_L = %-5s | pause_R = %s", result.pause_L, result.pause_R)
    logger.info("  step_back_L = %-5s | step_back_R = %s", result.step_back_L, result.step_back_R)
    logger.info("-" * 62)
    logger.info("VERDICT:  %s", result.verdict)
    logger.info(sep)
