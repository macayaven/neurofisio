#!/usr/bin/env python3
"""
Create HuggingFace datasets with actual EEG signal data.

Reads EDF files, applies TCP bipolar montage (22 channels), slices into
10-second non-overlapping windows, and attaches per-window labels from
corpus annotations.

Each row = one 10-second window:
  - signal: [22 channels × 2500 samples] float32 (250 Hz)
  - labels and metadata from annotations

**Memory-safe**: writes parquet shards to disk in chunks, then uploads
the shard folder to HuggingFace. Never holds entire dataset in memory.

Usage:
    python create_hf_signal_datasets.py --corpus tusl
    python create_hf_signal_datasets.py --corpus all
    python create_hf_signal_datasets.py --corpus tusl --dry-run
"""

import argparse
import gc
import json
import re
import shutil
import traceback
from collections import defaultdict
from math import gcd
from pathlib import Path

import numpy as np
import pyedflib
from datasets import Dataset
from huggingface_hub import HfApi
from scipy.signal import resample_poly
from tqdm import tqdm

# ─── Configuration ───────────────────────────────────────────────────────────

DATA_ROOT = Path("/home/carlos/workspace/neurofisio/data/tuh_eeg")
HF_ORG = "macayaven"
HF_PREFIX = "tuh-eeg"
TARGET_SR = 250  # Target sample rate (Hz)
WINDOW_SEC = 10  # Window duration in seconds
N_CHANNELS = 22  # TCP bipolar montage channels
WINDOW_SAMPLES = TARGET_SR * WINDOW_SEC  # 2500

# Chunked upload: write a parquet shard every SHARD_MAX_ROWS rows
SHARD_MAX_ROWS = 5000

# TCP bipolar montage: 22 channel pairs
TCP_PAIRS = [
    ("FP1", "F7"),
    ("F7", "T3"),
    ("T3", "T5"),
    ("T5", "O1"),  # Left temporal
    ("FP2", "F8"),
    ("F8", "T4"),
    ("T4", "T6"),
    ("T6", "O2"),  # Right temporal
    ("A1", "T3"),
    ("T3", "C3"),
    ("C3", "CZ"),
    ("CZ", "C4"),  # Central
    ("C4", "T4"),
    ("T4", "A2"),  # Central cont.
    ("FP1", "F3"),
    ("F3", "C3"),
    ("C3", "P3"),
    ("P3", "O1"),  # Left parasagittal
    ("FP2", "F4"),
    ("F4", "C4"),
    ("C4", "P4"),
    ("P4", "O2"),  # Right parasagittal
]
TCP_CHANNEL_NAMES = [f"{a}-{b}" for a, b in TCP_PAIRS]

# Electrode name aliases (handle naming variations in EDF files)
ELECTRODE_ALIASES = {
    "FP1": ["FP1", "Fp1"],
    "FP2": ["FP2", "Fp2"],
    "F7": ["F7"],
    "F3": ["F3"],
    "FZ": ["FZ", "Fz"],
    "F4": ["F4"],
    "F8": ["F8"],
    "A1": ["A1"],
    "T3": ["T3", "T7"],
    "C3": ["C3"],
    "CZ": ["CZ", "Cz"],
    "C4": ["C4"],
    "T4": ["T4", "T8"],
    "A2": ["A2"],
    "T5": ["T5", "P7"],
    "P3": ["P3"],
    "PZ": ["PZ", "Pz"],
    "P4": ["P4"],
    "T6": ["T6", "P8"],
    "O1": ["O1"],
    "O2": ["O2"],
}


# ─── Signal Processing ──────────────────────────────────────────────────────


def parse_channel_label(label):
    """Extract electrode name from EDF channel label."""
    label = label.strip().upper()
    if label.startswith("EEG "):
        label = label[4:]
    for suffix in ["-REF", "-LE", "-AVG", "-AR"]:
        if label.endswith(suffix):
            label = label[: -len(suffix)]
    return label.strip()


def build_electrode_map(edf_labels):
    """Map electrode names to EDF channel indices."""
    electrode_to_idx = {}
    for idx, raw_label in enumerate(edf_labels):
        name = parse_channel_label(raw_label)
        for canonical, aliases in ELECTRODE_ALIASES.items():
            if name in aliases:
                electrode_to_idx[canonical] = idx
                break
    return electrode_to_idx


def read_edf_signals(edf_path):
    """Read all signals from an EDF file and apply TCP montage.

    Returns:
        signals: np.ndarray [22, n_samples] or None on failure
        duration_sec: float
        original_sr: int
    """
    try:
        f = pyedflib.EdfReader(str(edf_path))
    except Exception:
        return None, 0, 0

    try:
        n_signals = f.signals_in_file
        labels = f.getSignalLabels()
        sample_rates = [f.getSampleFrequency(i) for i in range(n_signals)]
        duration = f.file_duration

        sr_counts = defaultdict(int)
        for sr in sample_rates:
            sr_counts[int(sr)] += 1
        original_sr = max(sr_counts, key=sr_counts.get)

        raw_signals = {}
        for idx, label in enumerate(labels):
            name = parse_channel_label(label)
            for canonical, aliases in ELECTRODE_ALIASES.items():
                if name in aliases and canonical not in raw_signals:
                    sig = f.readSignal(idx)
                    sig_sr = int(sample_rates[idx])
                    if sig_sr != TARGET_SR and sig_sr > 0:
                        g = gcd(TARGET_SR, sig_sr)
                        sig = resample_poly(sig, TARGET_SR // g, sig_sr // g)
                    raw_signals[canonical] = sig
                    break

        f.close()

        n_samples = int(duration * TARGET_SR)
        montage = np.zeros((N_CHANNELS, n_samples), dtype=np.float32)
        valid_channels = 0

        for ch_idx, (anode, cathode) in enumerate(TCP_PAIRS):
            if anode in raw_signals and cathode in raw_signals:
                a_sig = raw_signals[anode]
                c_sig = raw_signals[cathode]
                min_len = min(len(a_sig), len(c_sig), n_samples)
                montage[ch_idx, :min_len] = a_sig[:min_len] - c_sig[:min_len]
                valid_channels += 1

        if valid_channels < 10:
            return None, duration, original_sr

        return montage, duration, original_sr

    except Exception:
        f.close()
        return None, 0, 0


def window_signal(montage, window_sec=WINDOW_SEC, sr=TARGET_SR):
    """Slice montage into non-overlapping windows."""
    window_samples = sr * window_sec
    n_samples = montage.shape[1]
    windows = []
    for start in range(0, n_samples - window_samples + 1, window_samples):
        end = start + window_samples
        chunk = montage[:, start:end]
        start_sec = start / sr
        end_sec = end / sr
        windows.append((chunk, start_sec, end_sec))
    return windows


# ─── Annotation Loaders ─────────────────────────────────────────────────────


def load_csv_annotations(csv_path):
    rows = []
    with open(csv_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("#") or not line or "channel" in line.lower():
                continue
            parts = line.split(",")
            if len(parts) >= 5:
                rows.append(
                    {
                        "channel": parts[0].strip(),
                        "start_time": float(parts[1].strip()),
                        "stop_time": float(parts[2].strip()),
                        "label": parts[3].strip(),
                        "confidence": float(parts[4].strip()),
                    }
                )
    return rows


def load_tse_annotations(tse_path):
    rows = []
    with open(tse_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("version") or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                rows.append(
                    {
                        "start_time": float(parts[0]),
                        "stop_time": float(parts[1]),
                        "label": parts[2],
                        "confidence": float(parts[3]),
                    }
                )
    return rows


def load_rec_annotations(rec_path):
    event_map = {0: "null", 1: "spsw", 2: "gped", 3: "pled", 4: "eyem", 5: "artf", 6: "bckg"}
    rows = []
    with open(rec_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) >= 4:
                rows.append(
                    {
                        "channel_idx": int(parts[0]),
                        "start_time": float(parts[1]),
                        "stop_time": float(parts[2]),
                        "label": event_map.get(int(parts[3]), f"unknown_{parts[3]}"),
                    }
                )
    return rows


def get_window_labels_from_tse(annotations, start_sec, end_sec):
    labels = []
    for ann in annotations:
        overlap_start = max(ann["start_time"], start_sec)
        overlap_end = min(ann["stop_time"], end_sec)
        if overlap_end > overlap_start:
            labels.append((ann["label"], overlap_end - overlap_start))
    if not labels:
        return "bckg"
    label_durations = defaultdict(float)
    for label, dur in labels:
        label_durations[label] += dur
    return max(label_durations, key=label_durations.get)


def get_window_labels_from_csv(annotations, start_sec, end_sec):
    labels = set()
    for ann in annotations:
        overlap_start = max(ann["start_time"], start_sec)
        overlap_end = min(ann["stop_time"], end_sec)
        if overlap_end > overlap_start:
            labels.add(ann["label"])
    return sorted(labels) if labels else ["bckg"]


def get_window_labels_from_rec(annotations, start_sec, end_sec):
    label_counts = defaultdict(int)
    for ann in annotations:
        overlap_start = max(ann["start_time"], start_sec)
        overlap_end = min(ann["stop_time"], end_sec)
        if overlap_end > overlap_start:
            label_counts[ann["label"]] += 1
    if not label_counts:
        return "bckg"
    return max(label_counts, key=label_counts.get)


# ─── Path Parsing ────────────────────────────────────────────────────────────


def parse_subject_id(filepath):
    stem = Path(filepath).stem
    m = re.search(r"([a-z]{8})_s\d{3}", stem)
    if m:
        return m.group(1)
    m = re.search(r"^([a-z]{8})_", stem)
    if m:
        return m.group(1)
    m = re.search(r"/([a-z]{8})/s\d{3}", str(filepath))
    if m:
        return m.group(1)
    return ""


# ─── Generator-Based Corpus Processors ──────────────────────────────────────
# Each generator yields one row dict at a time (never holds full dataset).


def gen_tusl_signals(data_root):
    """TUSL: slowing vs background, per-recording TSE annotations."""
    edf_files = sorted((data_root / "tusl").rglob("*.edf"))
    print(f"  Found {len(edf_files)} EDF files")
    skipped = 0
    for edf_path in tqdm(edf_files, desc="  TUSL"):
        montage, _duration, _orig_sr = read_edf_signals(edf_path)
        if montage is None:
            skipped += 1
            continue
        tse_agg = edf_path.with_suffix(".tse_agg")
        annotations = []
        if tse_agg.exists():
            annotations = load_tse_annotations(tse_agg)
        else:
            tse_files = list(edf_path.parent.glob(f"{edf_path.stem}*.tse"))
            if tse_files:
                annotations = load_tse_annotations(tse_files[0])
        subject_id = parse_subject_id(edf_path)
        for w_idx, (signal, start_sec, end_sec) in enumerate(window_signal(montage)):
            yield {
                "signal": signal.tolist(),
                "label": get_window_labels_from_tse(annotations, start_sec, end_sec),
                "subject_id": subject_id,
                "file_path": str(edf_path),
                "window_idx": w_idx,
                "start_sec": round(start_sec, 2),
                "end_sec": round(end_sec, 2),
            }
        del montage
    print(f"  Skipped {skipped} files (bad channels)")


def gen_tuar_signals(data_root):
    """TUAR: artifact labels per channel per window."""
    edf_files = sorted((data_root / "tuar").rglob("*.edf"))
    print(f"  Found {len(edf_files)} EDF files")
    skipped = 0
    for edf_path in tqdm(edf_files, desc="  TUAR"):
        montage, _duration, _orig_sr = read_edf_signals(edf_path)
        if montage is None:
            skipped += 1
            continue
        csv_path = edf_path.with_suffix(".csv")
        annotations = load_csv_annotations(csv_path) if csv_path.exists() else []
        subject_id = parse_subject_id(edf_path)
        for w_idx, (signal, start_sec, end_sec) in enumerate(window_signal(montage)):
            labels = get_window_labels_from_csv(annotations, start_sec, end_sec)
            yield {
                "signal": signal.tolist(),
                "labels": json.dumps(labels),
                "has_artifact": any(l != "bckg" for l in labels),
                "subject_id": subject_id,
                "file_path": str(edf_path),
                "window_idx": w_idx,
                "start_sec": round(start_sec, 2),
                "end_sec": round(end_sec, 2),
            }
        del montage
    print(f"  Skipped {skipped} files")


def gen_tuev_signals(data_root):
    """TUEV: event classification per window."""
    edf_files = sorted((data_root / "tuev").rglob("*.edf"))
    print(f"  Found {len(edf_files)} EDF files")
    skipped = 0
    for edf_path in tqdm(edf_files, desc="  TUEV"):
        montage, _duration, _orig_sr = read_edf_signals(edf_path)
        if montage is None:
            skipped += 1
            continue
        rec_files = list(edf_path.parent.glob(f"{edf_path.stem}*.rec"))
        annotations = []
        for rec_path in rec_files:
            annotations.extend(load_rec_annotations(rec_path))
        subject_id = parse_subject_id(edf_path)
        split = "train" if "/train/" in str(edf_path) else "eval"
        for w_idx, (signal, start_sec, end_sec) in enumerate(window_signal(montage)):
            yield {
                "signal": signal.tolist(),
                "label": get_window_labels_from_rec(annotations, start_sec, end_sec),
                "split": split,
                "subject_id": subject_id,
                "file_path": str(edf_path),
                "window_idx": w_idx,
                "start_sec": round(start_sec, 2),
                "end_sec": round(end_sec, 2),
            }
        del montage
    print(f"  Skipped {skipped} files")


def gen_tuep_signals(data_root):
    """TUEP: epilepsy/no_epilepsy per recording."""
    edf_files = sorted((data_root / "tuep").rglob("*.edf"))
    print(f"  Found {len(edf_files)} EDF files")
    skipped = 0
    for edf_path in tqdm(edf_files, desc="  TUEP"):
        montage, _duration, _orig_sr = read_edf_signals(edf_path)
        if montage is None:
            skipped += 1
            continue
        path_str = str(edf_path)
        if "/00_epilepsy/" in path_str:
            epilepsy_label = "epilepsy"
        elif "/01_no_epilepsy/" in path_str:
            epilepsy_label = "no_epilepsy"
        else:
            epilepsy_label = "unknown"
        subject_id = parse_subject_id(edf_path)
        for w_idx, (signal, start_sec, end_sec) in enumerate(window_signal(montage)):
            yield {
                "signal": signal.tolist(),
                "epilepsy_label": epilepsy_label,
                "subject_id": subject_id,
                "file_path": str(edf_path),
                "window_idx": w_idx,
                "start_sec": round(start_sec, 2),
                "end_sec": round(end_sec, 2),
            }
        del montage
    print(f"  Skipped {skipped} files")


def gen_tuab_signals(data_root):
    """TUAB: normal/abnormal per recording."""
    edf_files = sorted((data_root / "tuab").rglob("*.edf"))
    print(f"  Found {len(edf_files)} EDF files")
    if not edf_files:
        print("  TUAB not downloaded, skipping")
        return
    skipped = 0
    for edf_path in tqdm(edf_files, desc="  TUAB"):
        montage, _duration, _orig_sr = read_edf_signals(edf_path)
        if montage is None:
            skipped += 1
            continue
        path_str = str(edf_path)
        label = (
            "normal"
            if "/normal/" in path_str
            else "abnormal"
            if "/abnormal/" in path_str
            else "unknown"
        )
        split = "train" if "/train/" in path_str else "eval" if "/eval/" in path_str else "unknown"
        subject_id = parse_subject_id(edf_path)
        for w_idx, (signal, start_sec, end_sec) in enumerate(window_signal(montage)):
            yield {
                "signal": signal.tolist(),
                "label": label,
                "split": split,
                "subject_id": subject_id,
                "file_path": str(edf_path),
                "window_idx": w_idx,
                "start_sec": round(start_sec, 2),
                "end_sec": round(end_sec, 2),
            }
        del montage
    print(f"  Skipped {skipped} files")


def gen_tusz_signals(data_root):
    """TUSZ: seizure detection per window."""
    edf_files = sorted((data_root / "tusz").rglob("*.edf"))
    print(f"  Found {len(edf_files)} EDF files")
    if not edf_files:
        print("  TUSZ not downloaded, skipping")
        return
    skipped = 0
    for edf_path in tqdm(edf_files, desc="  TUSZ"):
        montage, _duration, _orig_sr = read_edf_signals(edf_path)
        if montage is None:
            skipped += 1
            continue
        csv_path = edf_path.with_suffix(".csv")
        annotations = load_csv_annotations(csv_path) if csv_path.exists() else []
        csv_bi = edf_path.with_suffix(".csv_bi")
        bi_annotations = load_csv_annotations(csv_bi) if csv_bi.exists() else []
        path_str = str(edf_path)
        split = "train" if "/train/" in path_str else "eval"
        subject_id = parse_subject_id(edf_path)
        for w_idx, (signal, start_sec, end_sec) in enumerate(window_signal(montage)):
            labels = get_window_labels_from_csv(annotations, start_sec, end_sec)
            bi_labels = (
                get_window_labels_from_csv(bi_annotations, start_sec, end_sec)
                if bi_annotations
                else labels
            )
            has_seizure = any(l not in ("bckg", "background") for l in bi_labels)
            yield {
                "signal": signal.tolist(),
                "labels": json.dumps(labels),
                "has_seizure": has_seizure,
                "split": split,
                "subject_id": subject_id,
                "file_path": str(edf_path),
                "window_idx": w_idx,
                "start_sec": round(start_sec, 2),
                "end_sec": round(end_sec, 2),
            }
        del montage
    print(f"  Skipped {skipped} files")


# ─── Chunked Upload ──────────────────────────────────────────────────────────


def process_and_upload(generator_fn, corpus_name, data_root, dry_run=False):
    """Process EDF files via generator and upload parquet shards incrementally.

    Never holds more than SHARD_MAX_ROWS rows in memory at a time.
    """
    repo_id = f"{HF_ORG}/{HF_PREFIX}-{corpus_name}-signals"
    shard_dir = data_root / f"_shards_{corpus_name}"

    # Clean up any previous partial run
    if shard_dir.exists():
        shutil.rmtree(str(shard_dir))
    shard_dir.mkdir(parents=True)

    if not dry_run:
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)

    buffer = []
    shard_idx = 0
    total_rows = 0
    columns = None

    for row in generator_fn(data_root):
        buffer.append(row)

        if len(buffer) >= SHARD_MAX_ROWS:
            if columns is None:
                columns = list(buffer[0].keys())
            shard_path = shard_dir / f"train-{shard_idx:05d}-of-XXXXX.parquet"
            ds = Dataset.from_dict({k: [r[k] for r in buffer] for k in columns})
            ds.to_parquet(str(shard_path))
            total_rows += len(buffer)
            shard_idx += 1
            print(f"    Shard {shard_idx}: {total_rows:,} rows cumulative")
            buffer.clear()
            del ds
            gc.collect()

    # Write remaining rows
    if buffer:
        if columns is None:
            columns = list(buffer[0].keys())
        shard_path = shard_dir / f"train-{shard_idx:05d}-of-XXXXX.parquet"
        ds = Dataset.from_dict({k: [r[k] for r in buffer] for k in columns})
        ds.to_parquet(str(shard_path))
        total_rows += len(buffer)
        shard_idx += 1
        print(f"    Shard {shard_idx} (final): {total_rows:,} rows total")
        buffer.clear()
        del ds
        gc.collect()

    if total_rows == 0:
        print(f"  No rows generated for {corpus_name}, skipping upload")
        shutil.rmtree(str(shard_dir))
        return

    # Rename shards with correct total count
    for old_path in sorted(shard_dir.glob("*.parquet")):
        new_name = old_path.name.replace("XXXXX", f"{shard_idx:05d}")
        old_path.rename(shard_dir / new_name)

    if dry_run:
        print(f"  [DRY RUN] Would upload {shard_idx} shards ({total_rows:,} rows) to {repo_id}")
        shutil.rmtree(str(shard_dir))
        return

    # Upload all shards
    print(f"  Uploading {shard_idx} shards ({total_rows:,} rows) to {repo_id}...")
    api.upload_folder(
        folder_path=str(shard_dir),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo="data",
    )

    # Upload README
    card = generate_signal_card(corpus_name, total_rows, columns)
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )

    # Cleanup local shards
    shutil.rmtree(str(shard_dir))
    print(f"  Done! {total_rows:,} rows in {shard_idx} shards -> {repo_id}")


def generate_signal_card(corpus_name, n_rows, columns):
    col_table = "\n".join(
        f"| `{c}` | {'22x2500 float32 array (TCP bipolar montage, 250Hz)' if c == 'signal' else '-'} |"
        for c in columns
    )
    size_cat = (
        "n<1K"
        if n_rows < 1000
        else "1K<n<10K"
        if n_rows < 10000
        else "10K<n<100K"
        if n_rows < 100000
        else "100K<n<1M"
    )
    return f"""---
license: other
license_name: tuh-eeg-license
license_link: https://isip.piconepress.com/projects/nedc/html/tuh_eeg/
task_categories:
  - other
tags:
  - eeg
  - clinical
  - neuroscience
  - temple-university-hospital
  - {corpus_name}
  - signal-data
pretty_name: TUH EEG - {corpus_name.upper()} Signal Windows
size_categories:
  - {size_cat}
private: true
---

# TUH EEG {corpus_name.upper()} - Signal Windows

**EEG signal dataset** with 10-second windowed recordings from the TUH {corpus_name.upper()} corpus.

Each row is a 10-second non-overlapping window containing:
- **22-channel TCP bipolar montage** (standard clinical EEG)
- **250 Hz sampling rate** (resampled if needed)
- **{n_rows:,} total windows**

## Signal Format

- Shape: `[22, 2500]` (22 channels x 2500 samples)
- Dtype: float32
- Unit: microvolts (uV)
- Montage: TCP bipolar (FP1-F7, F7-T3, T3-T5, T5-O1, FP2-F8, ...)

## Channel Order (TCP Bipolar)

```
 0: FP1-F7    4: FP2-F8    8: A1-T3    14: FP1-F3   18: FP2-F4
 1: F7-T3     5: F8-T4     9: T3-C3    15: F3-C3    19: F4-C4
 2: T3-T5     6: T4-T6    10: C3-CZ    16: C3-P3    20: C4-P4
 3: T5-O1     7: T6-O2    11: CZ-C4    17: P3-O1    21: P4-O2
                           12: C4-T4
                           13: T4-A2
```

## Schema

| Column | Description |
|--------|-------------|
{col_table}

## Usage

```python
from datasets import load_dataset
import numpy as np

ds = load_dataset("{HF_ORG}/{HF_PREFIX}-{corpus_name}-signals", split="train")

# Get first window
signal = np.array(ds[0]["signal"])  # [22, 2500]
print(signal.shape)
```

## Citation

Obeid, I., & Picone, J. (2016). The Temple University Hospital EEG Data Corpus.
*Frontiers in Neuroscience*, 10, 196.
"""


# ─── Main ────────────────────────────────────────────────────────────────────

GENERATORS = {
    "tusl": gen_tusl_signals,
    "tuar": gen_tuar_signals,
    "tuev": gen_tuev_signals,
    "tuep": gen_tuep_signals,
    "tuab": gen_tuab_signals,
    "tusz": gen_tusz_signals,
}

CORPUS_ORDER = ["tusl", "tuar", "tuev", "tuep", "tuab", "tusz"]


def main():
    parser = argparse.ArgumentParser(description="Create HF signal datasets for TUH EEG")
    parser.add_argument("--corpus", default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    args = parser.parse_args()

    data_root = Path(args.data_root)
    corpora = CORPUS_ORDER if args.corpus == "all" else [args.corpus]

    for corpus in corpora:
        if corpus not in GENERATORS:
            print(f"Unknown corpus: {corpus}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing {corpus.upper()} (signal extraction)")
        print(f"{'=' * 60}")

        try:
            process_and_upload(GENERATORS[corpus], corpus, data_root, dry_run=args.dry_run)
        except Exception as e:
            print(f"  ERROR processing {corpus}: {e}")
            traceback.print_exc()

    print("\nDone!")


if __name__ == "__main__":
    main()
