"""Plot 2 — Joint Angle Curves.

Four vertically stacked subplots sharing the time axis:
    1. LEFT  — Attack  : elbow angle + shoulder angle
    2. LEFT  — Lunge   : front knee + inter-thigh angle  (+width ratio on twin y-axis)
    3. RIGHT — Attack  : elbow angle + shoulder angle
    4. RIGHT — Lunge   : front knee + inter-thigh angle  (+width ratio on twin y-axis)

Right-side panel shows threshold values (name + angle).
All subplots mark: baseline (window-open), ARM↑, LAUNCH, LUNGE, LIGHT-ON.
X-axis uses absolute video time (seconds from frame 0 of the video).

Usage:
    from visualization.angle_plot import save_angle_plot
    save_angle_plot(result, baseline_L, baseline_R, fps=30.0,
                    out_path=Path("output/video/joint_angles.png"),
                    video_name="互中1左侧")
"""

from __future__ import annotations
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
import numpy as np

# ── CJK font ──────────────────────────────────────────────────────────────────
def _set_cjk_font() -> None:
    available = {f.name for f in _fm.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS"):
        if name in available:
            plt.rcParams["font.family"] = name
            return
_set_cjk_font()

import config
from modules.event_aggregator import AggregatorResult, EventType, Verdict
from modules.athlete_tracker import AthleteBaseline

# ── Colours per measurement type (consistent across L/R) ─────────────────────
_C_ELBOW    = "#1565C0"   # blue   — elbow angle
_C_SHOULDER = "#7B1FA2"   # purple — shoulder angle
_C_KNEE     = "#2E7D32"   # green  — front knee
_C_THIGH    = "#E65100"   # orange — inter-thigh
_C_WIDTH    = "#795548"   # brown  — stance width ratio
_C_WINDOW   = "#555555"   # grey   — window open (baseline line)
_C_LIGHT    = "#D50000"   # red    — light on

_VERDICT_COLOUR = {
    Verdict.LEFT_SCORES:     "#1565C0",
    Verdict.RIGHT_SCORES:    "#BF360C",
    Verdict.SIMULTANEOUS:    "#6A1B9A",
    Verdict.UNABLE_TO_JUDGE: "#757575",
    Verdict.PENDING:         "#757575",
}
_VERDICT_SHORT = {
    Verdict.LEFT_SCORES:     "left",
    Verdict.RIGHT_SCORES:    "right",
    Verdict.SIMULTANEOUS:    "simu",
    Verdict.UNABLE_TO_JUDGE: "?",
    Verdict.PENDING:         "?",
}

# ── Event markers per side: (EventType, linestyle, color, label) ──────────────
_L_EVENTS = [
    (EventType.L_ATTACK_START, ":",  "#1565C0", "ARM↑"),
    (EventType.L_LAUNCH,       "--", "#0288D1", "LAUNCH"),
    (EventType.L_LUNGE,        "-.", "#2E7D32", "LUNGE"),
    (EventType.L_ATTACK_PAUSE, ":",  "#C62828", "PAUSE"),
]
_R_EVENTS = [
    (EventType.R_ATTACK_START, ":",  "#BF360C", "ARM↑"),
    (EventType.R_LAUNCH,       "--", "#E64A19", "LAUNCH"),
    (EventType.R_LUNGE,        "-.", "#1B5E20", "LUNGE"),
    (EventType.R_ATTACK_PAUSE, ":",  "#C62828", "PAUSE"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _nan_array(values: list[float]) -> np.ndarray:
    return np.array([v if not math.isnan(v) else np.nan for v in values])


def _annotate_hline_left(ax, y: float, label: str, color: str, alpha: float = 0.9) -> None:
    """Place a small label just outside the y-axis left edge at height y."""
    ax.annotate(
        label,
        xy=(0, y), xycoords=("axes fraction", "data"),
        xytext=(-4, 0), textcoords="offset points",
        ha="right", va="center", fontsize=7, color=color, alpha=alpha,
    )


def _draw_events(
    ax,
    event_times: dict,
    side_markers: list,
    t_window_open: float,
    t_close: float | None,
) -> None:
    """Draw vertical event markers common to both attack and lunge subplots."""
    # Window open → "baseline"
    ax.axvline(t_window_open, color=_C_WINDOW, linewidth=1.5,
               linestyle="-", alpha=0.55, zorder=10)
    ax.text(t_window_open, 0.01, "baseline",
            transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=7.5,
            color=_C_WINDOW, fontweight="bold")

    # Action events (ARM↑, LAUNCH, LUNGE, PAUSE)
    for ev_type, ls, color, short in side_markers:
        t = event_times.get(ev_type)
        if t is None:
            continue
        ax.axvline(t, color=color, linewidth=1.3, linestyle=ls,
                   alpha=0.85, zorder=10)
        ax.text(t + 0.02, 0.98, short,
                transform=ax.get_xaxis_transform(),
                ha="left", va="top", fontsize=7.5,
                color=color, fontweight="bold", rotation=90)

    # Window close → "LIGHT ON"
    if t_close is not None:
        ax.axvline(t_close, color=_C_LIGHT, linewidth=1.8,
                   linestyle="--", alpha=0.7, zorder=10)
        ax.text(t_close + 0.02, 0.98, "LIGHT\nON",
                transform=ax.get_xaxis_transform(),
                ha="left", va="top", fontsize=7.5,
                color=_C_LIGHT, fontweight="bold", rotation=90)


# ── Main entry point ──────────────────────────────────────────────────────────

def save_angle_plot(
    result:     AggregatorResult,
    baseline_L: AthleteBaseline,
    baseline_R: AthleteBaseline,
    fps:        float,
    out_path:   Path,
    video_name: str = "",
) -> None:
    """Render and save joint-angle PNG to *out_path*."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = result.angle_records
    if not records:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No angle data recorded",
                ha="center", va="center", transform=ax.transAxes, fontsize=14)
        plt.savefig(out_path, dpi=120)
        plt.close(fig)
        return

    # ── Time axis (absolute video seconds) ───────────────────────────────────
    ts = np.array([r.frame_idx / fps for r in records])

    # ── Angle arrays ─────────────────────────────────────────────────────────
    elbow_L    = _nan_array([r.elbow_L    for r in records])
    shoulder_L = _nan_array([r.shoulder_L for r in records])
    knee_L     = _nan_array([r.knee_L     for r in records])
    thigh_L    = _nan_array([r.thigh_L    for r in records])
    width_L    = _nan_array([r.width_L    for r in records])

    elbow_R    = _nan_array([r.elbow_R    for r in records])
    shoulder_R = _nan_array([r.shoulder_R for r in records])
    knee_R     = _nan_array([r.knee_R     for r in records])
    thigh_R    = _nan_array([r.thigh_R    for r in records])
    width_R    = _nan_array([r.width_R    for r in records])

    # Normalise stance width to ratio vs baseline (dimensionless, ~1 at rest)
    def _to_ratio(arr: np.ndarray, bl_width: float):
        if not math.isnan(bl_width) and bl_width > 0:
            return arr / bl_width, True
        return arr, False

    wr_L, wr_L_ok = _to_ratio(width_L, baseline_L.stance_width)
    wr_R, wr_R_ok = _to_ratio(width_R, baseline_R.stance_width)

    # ── Event timestamps (absolute video time) ────────────────────────────────
    def _s(f: int | None) -> float | None:
        return None if f is None else f / fps

    event_times: dict[EventType, float] = {}
    for ev in result.events:
        t = _s(ev.frame_idx)
        if t is not None and ev.event_type not in event_times:
            event_times[ev.event_type] = t

    t_window_open = (_s(result.window_open_frame)
                     if result.window_open_frame is not None else float(ts[0]))
    t_close = _s(result.window_close_frame)

    # ── Figure layout ─────────────────────────────────────────────────────────
    data_span = float(ts[-1] - ts[0]) if len(ts) > 1 else 1.0
    fig_w = min(24.0, max(11.0, data_span * 0.9 + 5.0))

    fig, axes = plt.subplots(
        4, 1, figsize=(fig_w, 12),
        sharex=True, gridspec_kw={"hspace": 0.10},
    )
    fig.patch.set_facecolor("white")
    # Leave right margin for threshold text panel
    plt.subplots_adjust(left=0.09, right=0.76, top=0.94, bottom=0.05)

    ax_la, ax_ll, ax_ra, ax_rl = axes

    # ── Header: filename (left) + verdict (right) ─────────────────────────────
    vc     = _VERDICT_COLOUR.get(result.verdict, "#757575")
    vshort = _VERDICT_SHORT.get(result.verdict, "?")
    fig.text(0.01, 0.985, video_name, ha="left", va="top",
             fontsize=11, fontweight="bold", color="#333333")
    fig.text(0.75, 0.985, vshort, ha="right", va="top",
             fontsize=11, fontweight="bold", color=vc,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor=vc, linewidth=1.5))

    # ─────────────────────────────────────────────────────────────────────────
    #  1. LEFT — ATTACK
    # ─────────────────────────────────────────────────────────────────────────
    ax = ax_la
    ax.set_facecolor("#F5F8FF")
    ax.plot(ts, elbow_L,    color=_C_ELBOW,    linewidth=1.8,
            label="Elbow", zorder=3)
    ax.plot(ts, shoulder_L, color=_C_SHOULDER, linewidth=1.5, linestyle="-.",
            label="Shoulder", zorder=3)

    # Baseline horizontal lines
    if not math.isnan(baseline_L.elbow_angle):
        ax.axhline(baseline_L.elbow_angle, color=_C_ELBOW, linewidth=0.8,
                   linestyle=(0, (3, 5)), alpha=0.5, zorder=2)
        _annotate_hline_left(ax, baseline_L.elbow_angle,
                             f"{baseline_L.elbow_angle:.0f}°", _C_ELBOW, 0.75)
    if not math.isnan(baseline_L.shoulder_angle):
        ax.axhline(baseline_L.shoulder_angle, color=_C_SHOULDER, linewidth=0.8,
                   linestyle=(0, (3, 5)), alpha=0.5, zorder=2)
        _annotate_hline_left(ax, baseline_L.shoulder_angle,
                             f"{baseline_L.shoulder_angle:.0f}°", _C_SHOULDER, 0.75)
    # Threshold horizontal lines
    ax.axhline(config.ATTACK_ELBOW_ABS, color=_C_ELBOW, linewidth=1.1,
               linestyle=(0, (1, 3)), alpha=0.75, zorder=2)
    _annotate_hline_left(ax, config.ATTACK_ELBOW_ABS,
                         f"{config.ATTACK_ELBOW_ABS}°", _C_ELBOW)
    ax.axhline(config.ATTACK_SHOULDER_ABS, color=_C_SHOULDER, linewidth=1.1,
               linestyle=(0, (1, 3)), alpha=0.75, zorder=2)
    _annotate_hline_left(ax, config.ATTACK_SHOULDER_ABS,
                         f"{config.ATTACK_SHOULDER_ABS}°", _C_SHOULDER)

    _draw_events(ax, event_times, _L_EVENTS, t_window_open, t_close)
    ax.set_ylabel("L  Attack  (°)", fontsize=9, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7.5, ncol=2, framealpha=0.85,
              edgecolor="#BBBBBB")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.5)
    ax.grid(axis="y", color="#EEEEEE", linewidth=0.4)

    # ─────────────────────────────────────────────────────────────────────────
    #  2. LEFT — LUNGE
    # ─────────────────────────────────────────────────────────────────────────
    ax = ax_ll
    ax.set_facecolor("#F5FFF5")
    ax.plot(ts, knee_L,  color=_C_KNEE,  linewidth=1.8,
            label="Front knee", zorder=3)
    ax.plot(ts, thigh_L, color=_C_THIGH, linewidth=1.5, linestyle="--",
            label="Thigh angle", zorder=3)

    if not math.isnan(baseline_L.front_knee_angle):
        ax.axhline(baseline_L.front_knee_angle, color=_C_KNEE, linewidth=0.8,
                   linestyle=(0, (3, 5)), alpha=0.5, zorder=2)
        _annotate_hline_left(ax, baseline_L.front_knee_angle,
                             f"{baseline_L.front_knee_angle:.0f}°", _C_KNEE, 0.75)
    ax.axhline(config.LUNGE_FRONT_KNEE_ABS, color=_C_KNEE, linewidth=1.1,
               linestyle=(0, (1, 3)), alpha=0.75, zorder=2)
    _annotate_hline_left(ax, config.LUNGE_FRONT_KNEE_ABS,
                         f"{config.LUNGE_FRONT_KNEE_ABS}°", _C_KNEE)
    ax.axhline(config.LUNGE_THIGH_ANGLE_L, color=_C_THIGH, linewidth=1.1,
               linestyle=(0, (1, 3)), alpha=0.75, zorder=2)
    _annotate_hline_left(ax, config.LUNGE_THIGH_ANGLE_L,
                         f"{config.LUNGE_THIGH_ANGLE_L}°", _C_THIGH)

    # Secondary y-axis — width ratio
    ax_wL = ax.twinx()
    wL_label = "Width ratio" if wr_L_ok else "Width (px)"
    ax_wL.plot(ts, wr_L, color=_C_WIDTH, linewidth=1.2, linestyle=":",
               label=wL_label, zorder=3)
    ax_wL.set_ylabel(wL_label, fontsize=8, color=_C_WIDTH)
    ax_wL.tick_params(axis="y", labelcolor=_C_WIDTH, labelsize=7)
    if wr_L_ok:
        ax_wL.axhline(config.LUNGE_WIDTH_RATIO, color=_C_WIDTH, linewidth=1.1,
                      linestyle=(0, (1, 3)), alpha=0.7, zorder=2)
        ax_wL.annotate(
            f"{config.LUNGE_WIDTH_RATIO:.1f}×",
            xy=(1, config.LUNGE_WIDTH_RATIO), xycoords=("axes fraction", "data"),
            xytext=(4, 0), textcoords="offset points",
            ha="left", va="center", fontsize=7, color=_C_WIDTH,
        )

    _draw_events(ax, event_times, _L_EVENTS, t_window_open, t_close)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_wL.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7.5,
              ncol=3, framealpha=0.85, edgecolor="#BBBBBB")
    ax.set_ylabel("L  Lunge  (°)", fontsize=9, fontweight="bold")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.5)
    ax.grid(axis="y", color="#EEEEEE", linewidth=0.4)

    # ─────────────────────────────────────────────────────────────────────────
    #  3. RIGHT — ATTACK
    # ─────────────────────────────────────────────────────────────────────────
    ax = ax_ra
    ax.set_facecolor("#FFF8F5")
    ax.plot(ts, elbow_R,    color=_C_ELBOW,    linewidth=1.8,
            label="Elbow", zorder=3)
    ax.plot(ts, shoulder_R, color=_C_SHOULDER, linewidth=1.5, linestyle="-.",
            label="Shoulder", zorder=3)

    if not math.isnan(baseline_R.elbow_angle):
        ax.axhline(baseline_R.elbow_angle, color=_C_ELBOW, linewidth=0.8,
                   linestyle=(0, (3, 5)), alpha=0.5, zorder=2)
        _annotate_hline_left(ax, baseline_R.elbow_angle,
                             f"{baseline_R.elbow_angle:.0f}°", _C_ELBOW, 0.75)
    if not math.isnan(baseline_R.shoulder_angle):
        ax.axhline(baseline_R.shoulder_angle, color=_C_SHOULDER, linewidth=0.8,
                   linestyle=(0, (3, 5)), alpha=0.5, zorder=2)
        _annotate_hline_left(ax, baseline_R.shoulder_angle,
                             f"{baseline_R.shoulder_angle:.0f}°", _C_SHOULDER, 0.75)
    ax.axhline(config.ATTACK_ELBOW_ABS, color=_C_ELBOW, linewidth=1.1,
               linestyle=(0, (1, 3)), alpha=0.75, zorder=2)
    _annotate_hline_left(ax, config.ATTACK_ELBOW_ABS,
                         f"{config.ATTACK_ELBOW_ABS}°", _C_ELBOW)
    ax.axhline(config.ATTACK_SHOULDER_ABS, color=_C_SHOULDER, linewidth=1.1,
               linestyle=(0, (1, 3)), alpha=0.75, zorder=2)
    _annotate_hline_left(ax, config.ATTACK_SHOULDER_ABS,
                         f"{config.ATTACK_SHOULDER_ABS}°", _C_SHOULDER)

    _draw_events(ax, event_times, _R_EVENTS, t_window_open, t_close)
    ax.set_ylabel("R  Attack  (°)", fontsize=9, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7.5, ncol=2, framealpha=0.85,
              edgecolor="#BBBBBB")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.5)
    ax.grid(axis="y", color="#EEEEEE", linewidth=0.4)

    # ─────────────────────────────────────────────────────────────────────────
    #  4. RIGHT — LUNGE
    # ─────────────────────────────────────────────────────────────────────────
    ax = ax_rl
    ax.set_facecolor("#FFF5F5")
    ax.plot(ts, knee_R,  color=_C_KNEE,  linewidth=1.8,
            label="Front knee", zorder=3)
    ax.plot(ts, thigh_R, color=_C_THIGH, linewidth=1.5, linestyle="--",
            label="Thigh angle", zorder=3)

    if not math.isnan(baseline_R.front_knee_angle):
        ax.axhline(baseline_R.front_knee_angle, color=_C_KNEE, linewidth=0.8,
                   linestyle=(0, (3, 5)), alpha=0.5, zorder=2)
        _annotate_hline_left(ax, baseline_R.front_knee_angle,
                             f"{baseline_R.front_knee_angle:.0f}°", _C_KNEE, 0.75)
    ax.axhline(config.LUNGE_FRONT_KNEE_ABS, color=_C_KNEE, linewidth=1.1,
               linestyle=(0, (1, 3)), alpha=0.75, zorder=2)
    _annotate_hline_left(ax, config.LUNGE_FRONT_KNEE_ABS,
                         f"{config.LUNGE_FRONT_KNEE_ABS}°", _C_KNEE)
    ax.axhline(config.LUNGE_THIGH_ANGLE_R, color=_C_THIGH, linewidth=1.1,
               linestyle=(0, (1, 3)), alpha=0.75, zorder=2)
    _annotate_hline_left(ax, config.LUNGE_THIGH_ANGLE_R,
                         f"{config.LUNGE_THIGH_ANGLE_R}°", _C_THIGH)

    # Secondary y-axis — width ratio
    ax_wR = ax.twinx()
    wR_label = "Width ratio" if wr_R_ok else "Width (px)"
    ax_wR.plot(ts, wr_R, color=_C_WIDTH, linewidth=1.2, linestyle=":",
               label=wR_label, zorder=3)
    ax_wR.set_ylabel(wR_label, fontsize=8, color=_C_WIDTH)
    ax_wR.tick_params(axis="y", labelcolor=_C_WIDTH, labelsize=7)
    if wr_R_ok:
        ax_wR.axhline(config.LUNGE_WIDTH_RATIO, color=_C_WIDTH, linewidth=1.1,
                      linestyle=(0, (1, 3)), alpha=0.7, zorder=2)
        ax_wR.annotate(
            f"{config.LUNGE_WIDTH_RATIO:.1f}×",
            xy=(1, config.LUNGE_WIDTH_RATIO), xycoords=("axes fraction", "data"),
            xytext=(4, 0), textcoords="offset points",
            ha="left", va="center", fontsize=7, color=_C_WIDTH,
        )

    _draw_events(ax, event_times, _R_EVENTS, t_window_open, t_close)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_wR.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7.5,
              ncol=3, framealpha=0.85, edgecolor="#BBBBBB")
    ax.set_ylabel("R  Lunge  (°)", fontsize=9, fontweight="bold")
    ax.set_xlabel("Time (s)  [from video start]", fontsize=10)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.5)
    ax.grid(axis="y", color="#EEEEEE", linewidth=0.4)

    # ── Common x-axis range ───────────────────────────────────────────────────
    t_end = float(ts[-1]) if len(ts) else 1.0
    ax_rl.set_xlim(0, t_end + 0.5)

    # =========================================================================
    #  Right-side text panel — thresholds + baselines
    # =========================================================================
    TX   = 0.78    # left edge in figure fraction
    LH   = 0.033   # line height
    FS   = 8.0     # normal font size
    FS_H = 8.5     # section header font size
    IND  = 0.012   # indent for values

    y = 0.93

    def _row(text, bold=False, color="#333333", indent=0, gap_before=0.0):
        nonlocal y
        y -= gap_before
        fig.text(TX + indent, y, text,
                 ha="left", va="top",
                 fontsize=FS_H if bold else FS,
                 fontweight="bold" if bold else "normal",
                 color=color)
        y -= LH

    def _val(label: str, val: float, color: str, unit: str = "°") -> None:
        v = f"{val:.0f}{unit}" if not math.isnan(val) else "N/A"
        _row(f"{label:<12}{v}", color=color, indent=IND)

    # ── Thresholds ────────────────────────────────────────────────────────────
    _row("THRESHOLDS", bold=True)
    _row("-" * 16, color="#999999")

    _row("Attack:", color="#555555")
    _row(f"  Elbow      {config.ATTACK_ELBOW_ABS}°",    color=_C_ELBOW)
    _row(f"  Shoulder   {config.ATTACK_SHOULDER_ABS}°", color=_C_SHOULDER)

    _row("Lunge:", color="#555555", gap_before=LH * 0.4)
    _row(f"  Knee       {config.LUNGE_FRONT_KNEE_ABS}°",  color=_C_KNEE)
    _row(f"  Thigh L    {config.LUNGE_THIGH_ANGLE_L}°",   color=_C_THIGH)
    _row(f"  Thigh R    {config.LUNGE_THIGH_ANGLE_R}°",   color=_C_THIGH)
    _row(f"  Width      {config.LUNGE_WIDTH_RATIO:.1f}× bl", color=_C_WIDTH)

    # ── L Baseline ────────────────────────────────────────────────────────────
    _row("L BASELINE", bold=True, color="#1565C0", gap_before=LH * 0.8)
    _row("-" * 16, color="#999999")
    _val("  Elbow",    baseline_L.elbow_angle,      _C_ELBOW)
    _val("  Shoulder", baseline_L.shoulder_angle,   _C_SHOULDER)
    _val("  Knee",     baseline_L.front_knee_angle, _C_KNEE)
    _val("  Width",    baseline_L.stance_width,     _C_WIDTH, unit=" px")

    # ── R Baseline ────────────────────────────────────────────────────────────
    _row("R BASELINE", bold=True, color="#BF360C", gap_before=LH * 0.8)
    _row("-" * 16, color="#999999")
    _val("  Elbow",    baseline_R.elbow_angle,      _C_ELBOW)
    _val("  Shoulder", baseline_R.shoulder_angle,   _C_SHOULDER)
    _val("  Knee",     baseline_R.front_knee_angle, _C_KNEE)
    _val("  Width",    baseline_R.stance_width,     _C_WIDTH, unit=" px")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
