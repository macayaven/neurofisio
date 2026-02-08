"""NeuroFisio EEG Explorer — Streamlit app for browsing TUH EEG corpora."""

import matplotlib

matplotlib.use("Agg")  # headless backend — must precede any other matplotlib import

import time as _time
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import pandas as pd
import streamlit as st

from neurofisio import (
    apply_tcp_montage,
    inspect_edf,
    inventory_corpus,
    load_csv_annotations,
    load_lab_annotations,
    load_rec_annotations,
    load_tse_annotations,
    parse_tuh_path,
    plot_eeg_segment,
    safe_read_raw_edf,
)

mne.set_log_level("WARNING")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroFisio EEG Explorer",
    page_icon=":brain:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark clinical theme via CSS injection ────────────────────────────────────
st.markdown(
    """
<style>
/* ── Import a distinctive font pair ─────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=DM+Sans:wght@400;500;600;700&display=swap');

/* ── Root variables ─────────────────────────────────────────────────────── */
:root {
    --bg-primary: #0c0e14;
    --bg-secondary: #10131a;
    --bg-card: #151820;
    --bg-hover: #1a1e2a;
    --border: #1e2233;
    --border-accent: #2a3050;
    --text-primary: #e8eaef;
    --text-secondary: #8892a8;
    --text-muted: #555d72;
    --accent-teal: #4dd0e1;
    --accent-coral: #ff8a65;
    --accent-amber: #ffd54f;
    --accent-green: #81c784;
    --accent-violet: #ce93d8;
}

/* ── Main app background ────────────────────────────────────────────────── */
.stApp, .main .block-container {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
}
section[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* ── Widget labels and text ─────────────────────────────────────────────── */
.stApp label, .stApp .stMarkdown, .stApp p, .stApp span,
.stApp .stCaption, .stApp h1, .stApp h2, .stApp h3,
.stApp [data-testid="stMetricValue"],
.stApp [data-testid="stMetricLabel"] {
    color: var(--text-primary) !important;
}
.stApp [data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.stApp [data-testid="stMetricValue"] {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
}

/* ── Metric cards ───────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 0.6rem 0.8rem !important;
}

/* ── Selectboxes & inputs ───────────────────────────────────────────────── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
[data-testid="stForm"] {
    background-color: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
}
.stSelectbox [data-baseweb="select"] > div {
    background-color: var(--bg-card) !important;
    border-color: var(--border) !important;
}

/* ── Expander ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-primary) !important;
}

/* ── Dataframe ──────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"], .stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}

/* ── Buttons ────────────────────────────────────────────────────────────── */
.stFormSubmitButton > button,
button[kind="primary"] {
    background: linear-gradient(135deg, #1a3a5c, #1a4060) !important;
    border: 1px solid var(--accent-teal) !important;
    color: var(--accent-teal) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em;
    transition: all 0.2s ease !important;
}
.stFormSubmitButton > button:hover,
button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1a4a6c, #1a5070) !important;
    border-color: #66e0ef !important;
    color: #66e0ef !important;
}

/* ── Sidebar styling ────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] .stSelectbox label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted) !important;
}
section[data-testid="stSidebar"] hr {
    border-color: var(--border) !important;
}

/* ── Title area ─────────────────────────────────────────────────────────── */
.app-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 0.5rem;
}
.app-header h1 {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.6rem !important;
    margin: 0 !important;
    color: var(--text-primary) !important;
}
.corpus-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    border: 1px solid;
}
.badge-teal { color: var(--accent-teal); border-color: var(--accent-teal); background: rgba(77,208,225,0.08); }
.badge-coral { color: var(--accent-coral); border-color: var(--accent-coral); background: rgba(255,138,101,0.08); }
.badge-amber { color: var(--accent-amber); border-color: var(--accent-amber); background: rgba(255,213,79,0.08); }
.badge-green { color: var(--accent-green); border-color: var(--accent-green); background: rgba(129,199,132,0.08); }
.badge-violet { color: var(--accent-violet); border-color: var(--accent-violet); background: rgba(206,147,216,0.08); }

/* ── Region legend ──────────────────────────────────────────────────────── */
.region-legend {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    padding: 6px 0;
}
.region-legend span {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.04em;
    display: flex;
    align-items: center;
    gap: 5px;
    color: var(--text-secondary);
}
.region-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
}

/* ── File info strip ────────────────────────────────────────────────────── */
.file-strip {
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
    padding: 8px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-secondary);
}
.file-strip strong {
    color: var(--text-primary);
    font-weight: 500;
}

/* ── Spinner / status ───────────────────────────────────────────────────── */
.stSpinner > div {
    border-color: var(--accent-teal) transparent transparent transparent !important;
}

/* ── Scrollbar ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-accent); border-radius: 3px; }

/* ── Transport bar ─────────────────────────────────────────────────────── */
.transport-bar {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 16px;
    margin: 8px 0 4px 0;
}
.transport-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-secondary);
}
.transport-row .time-display {
    color: var(--accent-teal);
    font-weight: 600;
    font-size: 0.82rem;
    min-width: 80px;
    text-align: center;
}
.transport-row .time-total {
    color: var(--text-muted);
    font-size: 0.7rem;
}
.transport-row .sep {
    color: var(--text-muted);
    margin: 0 2px;
}

/* Transport button styling */
div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
    background: var(--bg-hover) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-secondary) !important;
    font-size: 1.1rem !important;
    padding: 0.25rem 0.6rem !important;
    min-height: 2rem !important;
    border-radius: 6px !important;
    transition: all 0.15s ease !important;
}
div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
    background: var(--bg-card) !important;
    border-color: var(--accent-teal) !important;
    color: var(--accent-teal) !important;
}

/* Play button active state (playing) */
.play-active button[kind="secondary"] {
    border-color: var(--accent-green) !important;
    color: var(--accent-green) !important;
    background: rgba(129,199,132,0.08) !important;
}

/* Scrubber slider — make the track thinner and themed */
.transport-bar .stSlider > div > div > div {
    background: var(--border) !important;
}
.transport-bar .stSlider > div > div > div > div {
    background: var(--accent-teal) !important;
}
.transport-bar .stSlider label { display: none !important; }
.transport-bar .stSlider [data-testid="stTickBarMin"],
.transport-bar .stSlider [data-testid="stTickBarMax"] {
    display: none !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Constants ────────────────────────────────────────────────────────────────
DATA_ROOT = Path("/home/carlos/workspace/neurofisio/data/tuh_eeg")

CORPORA_CONFIG = {
    "TUSL": DATA_ROOT / "tusl",
    "TUAR": DATA_ROOT / "tuar",
    "TUEV": DATA_ROOT / "tuev",
    "TUEP": DATA_ROOT / "tuep",
    "TUAB": DATA_ROOT / "tuab",
    "TUSZ": DATA_ROOT / "tusz",
    "TUEG": DATA_ROOT / "tueg",
}

CORPUS_META = {
    "TUSL": {
        "desc": "Slowing vs Seizure differentiation",
        "size": "1.5 GB",
        "version": "v2.0.1",
        "labels": "seiz, slow, bckg",
        "ann_ext": ".tse",
    },
    "TUAR": {
        "desc": "Per-channel artifact detection",
        "size": "5.4 GB",
        "version": "v3.0.1",
        "labels": "eyem, chew, shiv, musc, elec, ...",
        "ann_ext": ".csv",
    },
    "TUEV": {
        "desc": "6-class EEG event classification",
        "size": "19 GB",
        "version": "v2.0.1",
        "labels": "spsw, gped, pled, eyem, artf, bckg",
        "ann_ext": ".rec",
    },
    "TUEP": {
        "desc": "Epilepsy / no-epilepsy diagnosis",
        "size": "35 GB",
        "version": "v3.0.0",
        "labels": "epilepsy, no_epilepsy",
        "ann_ext": ".csv",
    },
    "TUAB": {
        "desc": "Binary normal / abnormal EEG",
        "size": "58 GB",
        "version": "v3.0.1",
        "labels": "normal, abnormal (by folder)",
        "ann_ext": None,
    },
    "TUSZ": {
        "desc": "Seizure detection benchmark",
        "size": "81 GB",
        "version": "v2.0.3",
        "labels": "seiz, bckg (+ seizure types)",
        "ann_ext": ".csv_bi",
    },
    "TUEG": {
        "desc": "Full TUH EEG Corpus (unlabeled)",
        "size": "1,639 GB",
        "version": "v2.0.1",
        "labels": "N/A",
        "ann_ext": None,
    },
}

ANN_EXTS = [".csv", ".csv_bi", ".tse", ".tse_agg", ".lbl", ".lbl_agg", ".rec", ".lab"]

# ── Cached helpers ───────────────────────────────────────────────────────────


@st.cache_data(ttl=300, show_spinner=False)
def get_available_corpora() -> list[str]:
    """Return corpus names whose directories exist on disk."""
    return [name for name, path in CORPORA_CONFIG.items() if path.exists()]


@st.cache_data(ttl=3600, show_spinner=False)
def scan_subjects(corpus_name: str) -> dict:
    """Scan corpus EDF files grouped by subject. Cached 1 hour."""
    path = CORPORA_CONFIG.get(corpus_name)
    if not path or not path.exists():
        return {}
    subjects: dict[str, list] = {}
    for edf in sorted(path.rglob("*.edf")):
        meta = parse_tuh_path(edf)
        if meta:
            subjects.setdefault(meta["subject_id"], []).append(meta)
    return subjects


@st.cache_data(ttl=3600, show_spinner=False)
def cached_inventory(corpus_name: str) -> dict:
    """Inventory a corpus. Cached 1 hour."""
    path = CORPORA_CONFIG.get(corpus_name)
    if not path or not path.exists():
        return {}
    return inventory_corpus(path, corpus_name)


@st.cache_data(show_spinner=False)
def cached_inspect(edf_path: str) -> dict:
    return inspect_edf(edf_path)


@st.cache_resource(max_entries=5, show_spinner=False)
def cached_load_raw(edf_path: str):
    """Load EDF via MNE. Cached by reference (max 5 files in memory)."""
    return safe_read_raw_edf(edf_path)


@st.cache_resource(max_entries=5, show_spinner=False)
def cached_montage(_raw, edf_path: str):
    """Apply TCP montage. _raw excluded from hash; edf_path is the cache key."""
    return apply_tcp_montage(_raw)


@st.cache_data(show_spinner=False)
def cached_annotations(ann_path: str, ext: str):
    """Load annotation file by extension."""
    if ext in (".csv", ".csv_bi"):
        return load_csv_annotations(ann_path)
    if ext in (".tse", ".tse_agg"):
        return load_tse_annotations(ann_path)
    if ext == ".rec":
        return load_rec_annotations(ann_path)
    if ext in (".lab", ".lbl", ".lbl_agg"):
        return load_lab_annotations(ann_path)
    return pd.DataFrame()


def _load_ann_list(edf_path: str, found_ann: list) -> list[dict]:
    """Load annotation events from colocated files for plot overlay."""
    events = []
    for ext, ann_path in found_ann:
        try:
            result = cached_annotations(ann_path, ext)
            if isinstance(result, tuple):
                df = result[0]
            else:
                df = result
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            # Normalize column names for the plot overlay
            for _, row in df.iterrows():
                ev = {}
                if "start_time" in df.columns:
                    ev["start"] = float(row["start_time"])
                    ev["stop"] = float(row["stop_time"])
                elif "start" in df.columns:
                    ev["start"] = float(row["start"])
                    ev["stop"] = float(row["stop"])
                elif "start_sec" in df.columns:
                    ev["start"] = float(row["start_sec"])
                    ev["stop"] = float(row["stop_sec"])
                else:
                    continue
                ev["label"] = str(row.get("label", ""))
                if ev["label"] and ev["label"] != "bckg":
                    events.append(ev)
        except Exception:
            pass
    return events


# ── Sidebar — navigation ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='padding:4px 0 12px 0;'>"
        "<span style='font-family:DM Sans;font-weight:700;font-size:1.2rem;"
        "color:#e8eaef;'>NeuroFisio</span>"
        "<span style='font-family:JetBrains Mono;font-size:0.65rem;"
        "color:#4dd0e1;margin-left:8px;'>EEG EXPLORER</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    available = get_available_corpora()
    if not available:
        st.error(f"No corpora found under `{DATA_ROOT}`.")
        st.stop()

    corpus_name = st.selectbox("CORPUS", options=available)

    with st.spinner(f"Scanning {corpus_name}..."):
        subjects = scan_subjects(corpus_name)

    if not subjects:
        st.warning(f"No EDF files in {corpus_name}.")
        st.stop()

    sorted_sids = sorted(subjects.keys())

    # Show corpus stats inline
    meta = CORPUS_META.get(corpus_name, {})
    st.caption(
        f"{meta.get('desc', '')}  ·  {meta.get('size', '?')}  ·  {len(sorted_sids)} subjects"
    )

    subject_id = st.selectbox("SUBJECT", options=sorted_sids)

    files = subjects.get(subject_id, [])
    if not files:
        st.warning("No files for this subject.")
        st.stop()

    file_labels = [f"{Path(f['path']).stem}  ({f['montage'] or '?'})" for f in files]
    file_idx = st.selectbox(
        "FILE",
        options=range(len(file_labels)),
        format_func=lambda i: file_labels[i],
    )

    selected = files[file_idx]
    edf_path = selected["path"]

    st.divider()
    st.caption(f"`{Path(edf_path).name}`")

# ── Main area — header ──────────────────────────────────────────────────────
info = cached_inspect(edf_path)
if "error" in info:
    st.error(f"Cannot read file: {info['error']}")
    st.stop()

# Compact header with file identity + metrics in one row
st.markdown(
    f"<div class='app-header'>"
    f"<h1>{corpus_name}</h1>"
    f"<span class='corpus-badge badge-teal'>{meta.get('version', '?')}</span>"
    f"<span class='corpus-badge badge-coral'>{meta.get('size', '?')}</span>"
    f"</div>",
    unsafe_allow_html=True,
)

sfreq_str = f"{info['sample_freqs'][0]:.0f}" if info.get("sample_freqs") else "?"
st.markdown(
    f"<div class='file-strip'>"
    f"<span><strong>{info.get('n_channels', '?')}</strong> channels</span>"
    f"<span><strong>{sfreq_str}</strong> Hz</span>"
    f"<span><strong>{info.get('duration_sec', 0) / 60:.1f}</strong> min</span>"
    f"<span><strong>{selected.get('montage') or 'unknown'}</strong> montage</span>"
    f"<span>subject <strong>{selected['subject_id']}</strong></span>"
    f"<span>s{selected['session_num']:03d} · t{selected['token_num']:03d}</span>"
    f"</div>",
    unsafe_allow_html=True,
)

# ── Region legend ────────────────────────────────────────────────────────────
st.markdown(
    "<div class='region-legend'>"
    "<span><span class='region-dot' style='background:#4dd0e1'></span>L Temporal</span>"
    "<span><span class='region-dot' style='background:#ff8a65'></span>R Temporal</span>"
    "<span><span class='region-dot' style='background:#ffd54f'></span>Central</span>"
    "<span><span class='region-dot' style='background:#81c784'></span>L Parasagittal</span>"
    "<span><span class='region-dot' style='background:#ce93d8'></span>R Parasagittal</span>"
    "</div>",
    unsafe_allow_html=True,
)

# ── EEG signal viewer — transport-bar navigation ─────────────────────────────
max_dur = info.get("duration_sec", 60)

# Detect colocated annotations
edf_p = Path(edf_path)
found_ann = [
    (ext, str(edf_p.with_suffix(ext))) for ext in ANN_EXTS if edf_p.with_suffix(ext).exists()
]

# ── Session-state defaults (reset on file change) ────────────────────────────
if st.session_state.get("_viewer_path") != edf_path:
    st.session_state["_viewer_path"] = edf_path
    st.session_state["t_pos"] = 0.0
    st.session_state["t_dur"] = min(10.0, max_dur)
    st.session_state["playing"] = False
    st.session_state["speed"] = 1.0
    st.session_state["show_ann"] = True

# Convenience aliases
t_pos = st.session_state["t_pos"]
t_dur = st.session_state["t_dur"]
playing = st.session_state["playing"]

# ── Transport controls row ────────────────────────────────────────────────────
st.markdown("<div class='transport-bar'>", unsafe_allow_html=True)

tc1, tc2, tc3, tc4, tc5, tc6, tc7 = st.columns([1, 1, 1, 1, 1, 6, 3])


def _set_pos(pos, stop_play=False):
    """Update time position (scrubber syncs before its next render)."""
    st.session_state["t_pos"] = pos
    if stop_play:
        st.session_state["playing"] = False


with tc1:
    if st.button("⏮", key="t_start", help="Jump to start"):
        _set_pos(0.0, stop_play=True)
        st.rerun()
with tc2:
    if st.button("⏪", key="t_back", help="Step back"):
        _set_pos(max(0.0, t_pos - t_dur))
        st.rerun()
with tc3:
    play_label = "⏸" if playing else "▶"
    play_help = "Pause" if playing else "Play — auto-advance"
    if st.button(play_label, key="t_play", help=play_help):
        st.session_state["playing"] = not playing
        st.rerun()
with tc4:
    if st.button("⏩", key="t_fwd", help="Step forward"):
        _set_pos(min(t_pos + t_dur, max(0.0, max_dur - t_dur)))
        st.rerun()
with tc5:
    if st.button("⏭", key="t_end", help="Jump to end"):
        _set_pos(max(0.0, max_dur - t_dur), stop_play=True)
        st.rerun()

# Time position display + total
with tc6:
    t_end = min(t_pos + t_dur, max_dur)
    st.markdown(
        f"<div class='transport-row'>"
        f"<span class='time-display'>{t_pos:.1f}s</span>"
        f"<span class='sep'>–</span>"
        f"<span class='time-display'>{t_end:.1f}s</span>"
        f"<span class='time-total'>/ {max_dur:.0f}s</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

with tc7:
    speed = st.select_slider(
        "Speed",
        options=[0.5, 1.0, 1.5, 2.0, 3.0],
        value=st.session_state["speed"],
        format_func=lambda x: f"{x:.1f}x",
        key="speed_sl",
        label_visibility="collapsed",
    )
    st.session_state["speed"] = speed

# ── Scrubber — full-width slider ──────────────────────────────────────────────
scrub_max = max(0.0, max_dur - t_dur)
# Sync data-model position into widget state *before* the slider renders
st.session_state["scrubber"] = min(t_pos, scrub_max)
new_pos = st.slider(
    "Position",
    min_value=0.0,
    max_value=max(0.01, scrub_max),
    value=min(t_pos, scrub_max),
    step=max(0.1, t_dur * 0.1),
    format="%.1fs",
    key="scrubber",
    label_visibility="collapsed",
)
if abs(new_pos - t_pos) > 0.05:
    st.session_state["t_pos"] = new_pos
    st.session_state["playing"] = False
    t_pos = new_pos

# ── Settings row (duration + annotations toggle) ─────────────────────────────
sc1, sc2, _pad = st.columns([2, 2, 8])
with sc1:
    new_dur = st.select_slider(
        "Window",
        options=[2, 5, 10, 15, 20, 30, 60],
        value=int(min(t_dur, 60)),
        format_func=lambda x: f"{x}s",
        key="dur_sl",
        label_visibility="collapsed",
    )
    if new_dur != int(t_dur):
        st.session_state["t_dur"] = float(min(new_dur, max_dur))
        t_dur = st.session_state["t_dur"]
with sc2:
    show_annotations = st.checkbox(
        "Annotations",
        value=st.session_state["show_ann"],
        disabled=len(found_ann) == 0,
        key="ann_cb",
    )
    st.session_state["show_ann"] = show_annotations

st.markdown("</div>", unsafe_allow_html=True)

# ── Render EEG plot (always visible once a file is selected) ──────────────────
with st.spinner("Loading EDF & applying TCP montage..."):
    raw = cached_load_raw(edf_path)

if raw is None:
    st.error("Could not read EDF file (may be incomplete or corrupt).")
else:
    tcp = cached_montage(raw, edf_path)
    target = tcp if tcp is not None else raw
    tag = "TCP montage" if tcp is not None else "raw channels"

    # Load annotation events for plot overlay
    ann_events = None
    if show_annotations and found_ann:
        ann_events = _load_ann_list(edf_path, found_ann) or None

    actual_start = min(t_pos, max(0, target.times[-1] - t_dur))
    fig = plot_eeg_segment(
        target,
        start_sec=actual_start,
        duration_sec=t_dur,
        title=f"{Path(edf_path).stem} ({tag})",
        dark=True,
        colorize=True,
        annotations=ann_events,
    )
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        f"{t_dur:.0f}s from {actual_start:.1f}s  ·  "
        f"{target.info['nchan']} ch @ {target.info['sfreq']:.0f} Hz  ·  {tag}"
    )

# ── Auto-play engine ──────────────────────────────────────────────────────────
if st.session_state.get("playing"):
    next_pos = t_pos + t_dur * st.session_state.get("speed", 1.0)
    if next_pos >= max_dur - t_dur:
        _set_pos(max(0.0, max_dur - t_dur), stop_play=True)
    else:
        _set_pos(next_pos)
    _time.sleep(max(0.3, 1.0 / st.session_state.get("speed", 1.0)))
    st.rerun()

# ── Annotations detail (below plot, collapsed by default) ────────────────────
if found_ann:
    with st.expander(
        f"Annotation Data  ({', '.join(ext for ext, _ in found_ann)})", expanded=False
    ):
        for ext, ann_path in found_ann:
            st.markdown(f"**`{ext}`**")
            try:
                result = cached_annotations(ann_path, ext)
                if isinstance(result, tuple):
                    ann_df, ann_meta = result
                    if ann_meta:
                        st.json(ann_meta)
                else:
                    ann_df = result
                if isinstance(ann_df, pd.DataFrame) and not ann_df.empty:
                    st.dataframe(ann_df, height=250)
                else:
                    st.info("No events parsed.")
            except Exception as e:
                st.warning(f"Could not load {ext}: {e}")

# ── Corpus inventory (collapsed, at bottom) ──────────────────────────────────
inv = cached_inventory(corpus_name)
if inv:
    with st.expander("Corpus Inventory", expanded=False):
        ic1, ic2, ic3, ic4 = st.columns(4)
        ic1.metric("EDF Files", inv.get("edf_files", "?"))
        ann_total = sum(
            inv.get(k, 0)
            for k in ("csv_files", "csv_bi_files", "tse_files", "lab_files", "rec_files")
        )
        ic2.metric("Annotations", ann_total)
        ic3.metric("Subjects", inv.get("subjects", "?"))
        if inv.get("montages"):
            montage_str = ", ".join(f"{k} ({v})" for k, v in inv["montages"].items())
            ic4.markdown(f"**Montages**\n\n{montage_str}")
        st.markdown(f"Labels: `{meta.get('labels', 'N/A')}`")
