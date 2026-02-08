# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeuroFisio is a clinical EEG research project built on the Temple University Hospital (TUH) EEG Corpus (~1.8 TB, 27,000+ hours, 15,000 patients). It includes a reusable Python package (`neurofisio/`), a Streamlit web app for interactive EEG browsing (`app/`), and HuggingFace dataset builders for ML training pipelines.

**Product direction:** NeuroTriage — AI-powered clinical EEG interpretation pipeline. See `ONEPAGER_NEUROTRIAGE.md`. Development phases: P0 (TUAB binary classifier) → P1 (foundation encoder) → P2 (seizure detection) → P3 (multi-task) → P4 (report generation).

## Commands

```bash
# ── Package management (uv, not pip) ──────────────────────
uv sync                          # Install all deps from lock file
uv add <package>                 # Add dependency
uv pip install -e .              # Editable install of neurofisio package

# ── Streamlit app ──────────────────────────────────────────
uv run poe app-up                # Start EEG Explorer (background, port 8501)
uv run poe app-up 8502           # Start on custom port
uv run poe app-down              # Stop EEG Explorer
./app/run.sh                     # Foreground launch (alternative)

# ── Notebook ───────────────────────────────────────────────
uv run jupyter lab                # Launch JupyterLab (01_explore_tuh_eeg.ipynb)

# ── HuggingFace dataset building ──────────────────────────
uv run python create_hf_datasets.py --corpus tusl          # Metadata dataset
uv run python create_hf_signal_datasets.py --corpus tusl   # Windowed signal dataset
uv run python create_hf_signal_datasets.py --corpus all    # All corpora

# ── Code quality ───────────────────────────────────────────
uv run poe check                 # Run all checks (lint + format + test)
uv run poe lint                  # Ruff linter only
uv run poe lint-fix              # Ruff linter with auto-fix
uv run poe format                # Check formatting (CI mode, no changes)
uv run poe format-fix            # Auto-format code
uv run poe test                  # Run pytest

# ── Data sync from ISIP ───────────────────────────────────
rsync -auvxL -e "ssh -i ~/.ssh/vanessa-thonon-vb" \
  nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/ data/tuh_eeg/
```



## Architecture

```
neurofisio/                  # Core Python package — all reusable EEG utilities
  __init__.py                # Public API (re-exports from all modules)
  parsers.py                 # parse_tuh_path(), load_{csv,tse,rec,lab}_annotations()
  edf.py                     # inspect_edf(), safe_read_raw_edf(), inventory_corpus()
  montage.py                 # TCP_MONTAGE_PAIRS, TCP_NAMES, apply_tcp_montage()
  viz.py                     # plot_eeg_segment() — matplotlib with dark/colorize/annotations

app/                         # Streamlit web app (EEG Explorer)
  eeg_explorer.py            # Single-file app — sidebar nav, transport bar, plot viewer
  run.sh                     # uv-based launch wrapper
  .streamlit/config.toml     # Server config (headless, 0.0.0.0, no CORS)

create_hf_datasets.py        # Metadata-level HuggingFace dataset builder
create_hf_signal_datasets.py # Signal windowing (22ch × 2500 samples @ 250Hz, 10s windows)
01_explore_tuh_eeg.ipynb     # Exploration notebook (imports from neurofisio package)
```

### Data flow

1. **EDF on disk** → `parsers.parse_tuh_path()` extracts subject/session/token/montage metadata
2. **Load** → `edf.safe_read_raw_edf()` via MNE (returns None on failure, never raises)
3. **Montage** → `montage.apply_tcp_montage()` converts raw channels to 22 bipolar TCP pairs (auto-detects `-REF` vs `-LE` naming)
4. **Visualize** → `viz.plot_eeg_segment()` renders matplotlib figure with optional dark theme, region coloring, annotation overlays
5. **Annotations** → `parsers.load_{csv,tse,rec,lab}_annotations()` parse colocated files into DataFrames

### Streamlit app patterns

- `matplotlib.use("Agg")` must be first import (before any pyplot) to avoid Qt backend conflicts
- `@st.cache_resource(max_entries=5)` for MNE Raw objects (not serializable); `@st.cache_data` for dicts/DataFrames
- Underscore prefix `_raw` in function params excludes them from Streamlit's hash
- Session state keys `t_pos`, `t_dur`, `playing`, `speed` drive the transport-bar navigation
- Auto-play uses `time.sleep()` + `st.rerun()` loop; stops at recording end
- All viewer state resets when `_viewer_path` changes (file switch)
- CSS injection via `st.markdown("<style>...</style>", unsafe_allow_html=True)` for dark clinical theme

### Package conventions

- `setuptools` discovery restricted to `neurofisio*` in `pyproject.toml` to avoid packaging `app/` or `data/`
- All functions that can fail on corrupt EDF files return `None` or empty containers rather than raising
- Annotation loaders handle varying column names (`start_time`/`start`/`start_sec`) via normalization in `_load_ann_list()`

## Infrastructure

- **Compute:** `spark-caeb.local` (Linux aarch64, NVIDIA GB10, 128 GB unified, CUDA 13.0, 3.7TB NVMe)
- **Data root:** `/home/carlos/workspace/neurofisio/data/tuh_eeg/`
- **HuggingFace org:** `macayaven` (dataset prefix: `tuh-eeg-`)
- **Python:** 3.12 (`uv` managed, no virtualenv activation needed)

## EEG Domain Reference

**EDF+ files:** 24-36 channels, predominantly 250 Hz / 16-bit.

**TUH path anatomy:** `edf/{split}/{subject_8char}/{session_date}/{montage}/{subject_session_token.edf}`

**Standard TCP montage (22 bipolar channels):**
```
Left temporal:    FP1-F7, F7-T3, T3-T5, T5-O1
Right temporal:   FP2-F8, F8-T4, T4-T6, T6-O2
Central:          A1-T3, T3-C3, C3-CZ, CZ-C4, C4-T4, T4-A2
Left parasag:     FP1-F3, F3-C3, C3-P3, P3-O1
Right parasag:    FP2-F4, F4-C4, C4-P4, P4-O2
```

**Corpora (all subsets of TUEG):**

| Corpus | Size | Task | Annotation format |
|--------|------|------|-------------------|
| TUSL | 1.5 GB | Seizure vs slowing | `.tse`, `.lbl` |
| TUAR | 5.4 GB | Artifact detection | `.csv` |
| TUEV | 19 GB | 6-class events | `.rec`, `.lab` |
| TUEP | 35 GB | Epilepsy diagnosis | `.csv` |
| TUAB | 58 GB | Normal/abnormal | folder-based |
| TUSZ | 81 GB | Seizure detection | `.csv`, `.csv_bi` |
| TUEG | 1,639 GB | Unlabeled (full) | none |

Start prototyping with TUSL (smallest). TUAB has train/eval splits ready for P0 classifier.

**Key labels:** `seiz`/`bckg` (seizure), `eyem`/`chew`/`shiv`/`musc`/`elec` (artifacts), `spsw`/`gped`/`pled` (periodic discharges), `slow` (slowing).
