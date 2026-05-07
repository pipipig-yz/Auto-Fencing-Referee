"""Plot 1 — Event Timeline.

Horizontal swim-lane plot: LEFT (blue) / RIGHT (orange) / Light (red).
X axis = time in seconds relative to WINDOW_OPEN.

Usage:
    from visualization.timeline_plot import save_timeline
    save_timeline(result, fps=30.0, out_path=Path("output/video/timeline.png"),
                  video_name="互中1左侧", ground_truth="LEFT_SCORES")
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as _fm
from matplotlib.lines import Line2D

# ── Use a CJK-capable font if available (Windows: Microsoft YaHei) ────────────
def _set_cjk_font() -> None:
    available = {f.name for f in _fm.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS"):
        if name in available:
            plt.rcParams["font.family"] = name
            return
_set_cjk_font()

from modules.event_aggregator import AggregatorResult, EventType, Verdict

# ── Colours ───────────────────────────────────────────────────────────────────
_C_L     = "#1565C0"   # dark blue  — left
_C_R     = "#E65100"   # dark orange — right
_C_PAUSE = "#B71C1C"   # dark red   — pause events
_C_LIGHT = "#D50000"   # red        — light
_C_BG    = "#F5F5F5"

# ── Lane assignment: event → (y_lane, colour) ────────────────────────────────
_LANE: dict[EventType, tuple[int, str]] = {
    EventType.L_ATTACK_START: (2, _C_L),
    EventType.L_ATTACK_FULL:  (2, _C_L),
    EventType.L_LAUNCH:       (2, _C_L),
    EventType.L_LUNGE:        (2, _C_L),
    EventType.L_STEP_BACK:    (2, "#90CAF9"),
    EventType.L_ATTACK_PAUSE: (2, _C_PAUSE),
    EventType.R_ATTACK_START: (1, _C_R),
    EventType.R_ATTACK_FULL:  (1, _C_R),
    EventType.R_LAUNCH:       (1, _C_R),
    EventType.R_LUNGE:        (1, _C_R),
    EventType.R_STEP_BACK:    (1, "#FFAB40"),
    EventType.R_ATTACK_PAUSE: (1, _C_PAUSE),
    EventType.LIGHT_ON_L:     (0, _C_LIGHT),
    EventType.LIGHT_ON_R:     (0, _C_LIGHT),
    EventType.LIGHT_ON_BOTH:  (0, _C_LIGHT),
}

_LABEL: dict[EventType, str] = {
    EventType.L_ATTACK_START: "ARM↑",
    EventType.L_ATTACK_FULL:  "ARM✓",
    EventType.L_LAUNCH:       "LAUNCH",
    EventType.L_LUNGE:        "LUNGE",
    EventType.L_STEP_BACK:    "BACK",
    EventType.L_ATTACK_PAUSE: "PAUSE",
    EventType.R_ATTACK_START: "ARM↑",
    EventType.R_ATTACK_FULL:  "ARM✓",
    EventType.R_LAUNCH:       "LAUNCH",
    EventType.R_LUNGE:        "LUNGE",
    EventType.R_STEP_BACK:    "BACK",
    EventType.R_ATTACK_PAUSE: "PAUSE",
    EventType.LIGHT_ON_L:     "L",
    EventType.LIGHT_ON_R:     "R",
    EventType.LIGHT_ON_BOTH:  "BOTH",
}

_VERDICT_COLOUR = {
    Verdict.LEFT_SCORES:     _C_L,
    Verdict.RIGHT_SCORES:    _C_R,
    Verdict.SIMULTANEOUS:    "#6A1B9A",
    Verdict.UNABLE_TO_JUDGE: "#757575",
    Verdict.PENDING:         "#757575",
}


def save_timeline(
    result: AggregatorResult,
    fps: float,
    out_path: Path,
    video_name: str = "",
    ground_truth: str = "",
) -> None:
    """Render and save timeline PNG to *out_path*."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = result.window_open_frame or 0

    def _s(f: int | None) -> float | None:
        return None if f is None else (f - t0) / fps

    # ── Collect plottable events ──────────────────────────────────────────────
    pts: list[tuple[float, int, str, str]] = []   # (t_sec, lane, label, color)
    for ev in result.events:
        if ev.event_type not in _LANE:
            continue
        t = _s(ev.frame_idx)
        if t is None:
            continue
        lane, color = _LANE[ev.event_type]
        pts.append((t, lane, _LABEL.get(ev.event_type, "?"), color))

    t_close = _s(result.window_close_frame)
    all_t   = [p[0] for p in pts]
    x_min   = min(all_t, default=0.0) - 0.08
    x_max   = (t_close + 0.10) if t_close is not None else max(all_t, default=1.0) + 0.1

    # ── Figure / axes ─────────────────────────────────────────────────────────
    width  = max(9.0, (x_max - x_min) * 14)
    fig, ax = plt.subplots(figsize=(width, 3.8))
    ax.set_facecolor(_C_BG)
    fig.patch.set_facecolor("white")

    # Lane labels
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Light", "RIGHT", "LEFT"], fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", colors="#333", length=0)
    for y in [0.5, 1.5]:
        ax.axhline(y, color="#CCCCCC", linewidth=0.8, zorder=0)

    # Pre-window shading
    ax.axvspan(x_min, 0.0, color="#E0E0E0", alpha=0.6, zorder=0)
    ax.axvline(0.0, color="#757575", linewidth=1.2, linestyle="--",
               zorder=2, label="WINDOW_OPEN")

    # Window-close line
    if t_close is not None:
        ax.axvline(t_close, color=_C_LIGHT, linewidth=2.0, linestyle="--",
                   zorder=3, label="LIGHT_ON_BOTH")

    # Dotted key-timestamp verticals
    for frame, lane, color in [
        (result.t_arm_L,    2, _C_L),
        (result.t_launch_L, 2, _C_L),
        (result.t_step_L,   2, _C_L),
        (result.t_arm_R,    1, _C_R),
        (result.t_launch_R, 1, _C_R),
        (result.t_step_R,   1, _C_R),
    ]:
        t = _s(frame)
        if t is not None:
            ax.axvline(t, color=color, linewidth=1.0, linestyle=":",
                       alpha=0.45, zorder=1)

    # Event markers
    for t, lane, label, color in pts:
        ax.plot(t, lane, marker="^", markersize=11, color=color, zorder=5,
                markeredgecolor="white", markeredgewidth=0.7)
        ax.text(t, lane + 0.19, label, ha="center", va="bottom",
                fontsize=7.5, color=color, rotation=40, zorder=6,
                fontweight="bold")

    # ── Verdict box ───────────────────────────────────────────────────────────
    vc    = _VERDICT_COLOUR.get(result.verdict, "#757575")
    vtxt  = result.verdict.value.replace("_", " ")
    ax.text(0.99, 0.97, vtxt, transform=ax.transAxes,
            ha="right", va="top", fontsize=12, fontweight="bold", color=vc,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=vc, linewidth=1.8))

    # Ground-truth label
    if ground_truth:
        match = (ground_truth.upper() in result.verdict.value)
        gt_color = "#2E7D32" if match else "#C62828"
        gt_text  = f"GT: {ground_truth}  {'✓' if match else '✗'}"
        ax.text(0.01, 0.97, gt_text, transform=ax.transAxes,
                ha="left", va="top", fontsize=10, color=gt_color,
                fontweight="bold")

    # ── Title / labels ────────────────────────────────────────────────────────
    title = f"Event Timeline — {video_name}" if video_name else "Event Timeline"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Time (s)  [0 = WINDOW_OPEN]", fontsize=10)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.55, 2.85)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.5, zorder=0)

    handles = [
        mpatches.Patch(color=_C_L,     label="Left athlete"),
        mpatches.Patch(color=_C_R,     label="Right athlete"),
        mpatches.Patch(color=_C_PAUSE, label="Pause / forfeiture"),
        Line2D([0], [0], color=_C_LIGHT, linewidth=2,
               linestyle="--", label="Light ON BOTH"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              framealpha=0.85, edgecolor="#BBBBBB")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
