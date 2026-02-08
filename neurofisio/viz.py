"""EEG visualization helpers."""

import matplotlib.pyplot as plt
import numpy as np

# Brain-region color palette for the 22-channel TCP montage.
# Grouped by anatomical chain so lateralized activity is instantly visible.
_REGION_COLORS = {
    # Left temporal (channels 0-3): cool teal
    "FP1-F7": "#4dd0e1",
    "F7-T3": "#4dd0e1",
    "T3-T5": "#4dd0e1",
    "T5-O1": "#4dd0e1",
    # Right temporal (channels 4-7): warm coral
    "FP2-F8": "#ff8a65",
    "F8-T4": "#ff8a65",
    "T4-T6": "#ff8a65",
    "T6-O2": "#ff8a65",
    # Central (channels 8-13): amber
    "A1-T3": "#ffd54f",
    "T3-C3": "#ffd54f",
    "C3-CZ": "#ffd54f",
    "CZ-C4": "#ffd54f",
    "C4-T4": "#ffd54f",
    "T4-A2": "#ffd54f",
    # Left parasagittal (channels 14-17): soft green
    "FP1-F3": "#81c784",
    "F3-C3": "#81c784",
    "C3-P3": "#81c784",
    "P3-O1": "#81c784",
    # Right parasagittal (channels 18-21): soft violet
    "FP2-F4": "#ce93d8",
    "F4-C4": "#ce93d8",
    "C4-P4": "#ce93d8",
    "P4-O2": "#ce93d8",
}

_DARK_BG = "#0c0e14"
_DARK_FACE = "#10131a"
_DARK_GRID = "#1e2233"
_DARK_TEXT = "#c8cad0"
_DARK_TITLE = "#e8eaef"


def plot_eeg_segment(
    raw,
    start_sec=0,
    duration_sec=10,
    title="EEG Signal",
    *,
    dark=False,
    colorize=False,
    annotations=None,
):
    """Plot a multi-channel EEG segment as a static matplotlib figure.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        MNE Raw object (original or TCP-montaged).
    start_sec : float
        Start time in seconds.
    duration_sec : float
        Length of the window to display.
    title : str
        Figure title.
    dark : bool
        Use dark clinical theme (dark background, light traces).
    colorize : bool
        Color-code channels by brain region (TCP montage names).
    annotations : list of dict, optional
        Annotation spans to overlay. Each dict must have keys
        ``start``, ``stop``, ``label``.  Optional ``color`` key.

    Returns
    -------
    matplotlib.figure.Figure
    """
    sfreq = raw.info["sfreq"]
    start = int(start_sec * sfreq)
    stop = int((start_sec + duration_sec) * sfreq)
    data = raw.get_data()[:, start:stop]
    times = np.arange(data.shape[1]) / sfreq + start_sec

    n_ch = min(data.shape[0], 22)

    # ── Figure setup ─────────────────────────────────────────────────────
    if dark:
        fig, ax = plt.subplots(figsize=(20, max(8, n_ch * 0.45)), facecolor=_DARK_BG)
        ax.set_facecolor(_DARK_FACE)
        text_color = _DARK_TEXT
        title_color = _DARK_TITLE
        default_trace = "#8892a8"
        grid_color = _DARK_GRID
        spine_color = "#1a1d28"
    else:
        fig, ax = plt.subplots(figsize=(18, max(8, n_ch * 0.5)))
        text_color = "#333333"
        title_color = "#111111"
        default_trace = "k"
        grid_color = "#e0e0e0"
        spine_color = "#cccccc"

    scale = np.percentile(np.abs(data[:n_ch]), 95) * 3
    if scale == 0:
        scale = 1e-4

    # ── Annotation spans (behind traces) ─────────────────────────────────
    _ANN_COLORS = {
        "seiz": "#ef5350",
        "seizure": "#ef5350",
        "slow": "#ffa726",
        "slowing": "#ffa726",
        "bckg": None,  # don't draw background
        "eyem": "#42a5f5",
        "chew": "#ab47bc",
        "musc": "#ab47bc",
        "shiv": "#78909c",
        "elec": "#78909c",
        "spsw": "#ef5350",
        "gped": "#ff7043",
        "pled": "#ffa726",
        "artf": "#78909c",
    }
    if annotations:
        ymax = (n_ch - 1) * scale + scale * 0.5
        for ann in annotations:
            a_start = ann.get("start", 0)
            a_stop = ann.get("stop", 0)
            label = ann.get("label", "")
            color = ann.get("color") or _ANN_COLORS.get(label, "#ffffff22")
            if color is None:
                continue
            if a_stop <= start_sec or a_start >= start_sec + duration_sec:
                continue
            ax.axvspan(
                max(a_start, start_sec),
                min(a_stop, start_sec + duration_sec),
                alpha=0.12 if dark else 0.08,
                color=color,
                zorder=0,
            )
            # Label at top
            mid = (max(a_start, start_sec) + min(a_stop, start_sec + duration_sec)) / 2
            ax.text(
                mid,
                ymax,
                label,
                fontsize=7,
                ha="center",
                va="bottom",
                color=color,
                alpha=0.8,
                fontweight="bold",
            )

    # ── Grid lines ───────────────────────────────────────────────────────
    t_start_int = int(np.ceil(start_sec))
    t_end_int = int(np.floor(start_sec + duration_sec))
    for t in range(t_start_int, t_end_int + 1):
        ax.axvline(t, color=grid_color, linewidth=0.4, zorder=0)

    # Horizontal separators between region groups
    if colorize and n_ch >= 22:
        for boundary in [4, 8, 14, 18]:
            y = (boundary - 0.5) * scale
            ax.axhline(y, color=grid_color, linewidth=0.3, linestyle="--", zorder=0)

    # ── Traces ───────────────────────────────────────────────────────────
    for i in range(n_ch):
        ch_name = raw.ch_names[i]
        if colorize:
            color = _REGION_COLORS.get(ch_name, default_trace)
        else:
            color = default_trace

        offset = i * scale
        ax.plot(times, data[i] + offset, linewidth=0.6, color=color, zorder=2)
        ax.text(
            times[0] - 0.2,
            offset,
            ch_name,
            fontsize=7.5,
            ha="right",
            va="center",
            color=color if colorize else text_color,
            fontfamily="monospace",
            fontweight="medium",
        )

    # ── Axes styling ─────────────────────────────────────────────────────
    ax.set_xlabel("Time (s)", color=text_color, fontsize=9)
    ax.set_title(title, color=title_color, fontsize=11, fontweight="bold", pad=12)
    ax.set_yticks([])
    ax.set_xlim(times[0], times[-1])
    ax.tick_params(colors=text_color, labelsize=8)

    for spine in ax.spines.values():
        spine.set_color(spine_color)

    # Amplitude scale bar (bottom-right)
    if dark or colorize:
        bar_uv = _nice_scale_bar(scale)
        bar_y = -scale * 0.3
        bar_x = times[-1] - duration_sec * 0.02
        ax.plot(
            [bar_x, bar_x],
            [bar_y, bar_y + bar_uv * 1e-6],
            color=text_color,
            linewidth=1.5,
            zorder=5,
        )
        ax.text(
            bar_x + duration_sec * 0.01,
            bar_y + bar_uv * 0.5e-6,
            f"{bar_uv} uV",
            fontsize=7,
            color=text_color,
            va="center",
            fontfamily="monospace",
        )

    plt.tight_layout()
    return fig


def _nice_scale_bar(scale_volts):
    """Pick a round microvolt value for the amplitude scale bar."""
    uv = scale_volts * 1e6
    for candidate in [10, 20, 50, 100, 200, 500, 1000, 2000]:
        if candidate >= uv * 0.2:
            return candidate
    return int(uv)
