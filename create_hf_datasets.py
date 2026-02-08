#!/usr/bin/env python3
"""
Create private HuggingFace datasets for each TUH EEG corpus.

Generates metadata-level datasets (file paths, annotations, patient info,
signal stats) that serve as queryable indexes into the local EDF files.
For small corpora (TUSL), also includes windowed signal data.

Usage:
    python create_hf_datasets.py --corpus tusl   # single corpus
    python create_hf_datasets.py --corpus all     # all available
    python create_hf_datasets.py --corpus all --dry-run  # preview only
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import pyedflib
from datasets import Dataset, DatasetDict, Features, Value, Sequence, ClassLabel
from huggingface_hub import HfApi
from tqdm import tqdm

# ─── Configuration ───────────────────────────────────────────────────────────

DATA_ROOT = Path("/home/carlos/workspace/neurofisio/data/tuh_eeg")
HF_ORG = "macayaven"  # HuggingFace username/org
HF_PREFIX = "tuh-eeg"  # dataset name prefix

CORPUS_MAP = {
    "tusl": ("tuh_eeg_slowing", "v2.0.1"),
    "tuar": ("tuh_eeg_artifact", "v3.0.1"),
    "tuev": ("tuh_eeg_events", "v2.0.1"),
    "tuep": ("tuh_eeg_epilepsy", "v3.0.0"),
    "tuab": ("tuh_eeg_abnormal", "v3.0.1"),
    "tusz": ("tuh_eeg_seizure", "v2.0.3"),
    "tueg": ("tuh_eeg", "v2.0.1"),
}

# Standard TCP bipolar montage (22 channels)
TCP_MONTAGE = {
    0: ("FP1", "F7"),  1: ("F7", "T3"),   2: ("T3", "T5"),   3: ("T5", "O1"),
    4: ("FP2", "F8"),  5: ("F8", "T4"),   6: ("T4", "T6"),   7: ("T6", "O2"),
    8: ("A1", "T3"),   9: ("T3", "C3"),  10: ("C3", "CZ"),  11: ("CZ", "C4"),
    12: ("C4", "T4"), 13: ("T4", "A2"),  14: ("FP1", "F3"), 15: ("F3", "C3"),
    16: ("C3", "P3"), 17: ("P3", "O1"),  18: ("FP2", "F4"), 19: ("F4", "C4"),
    20: ("C4", "P4"), 21: ("P4", "O2"),
}
TCP_CHANNEL_NAMES = [f"{a}-{b}" for a, b in TCP_MONTAGE.values()]


# ─── Utility Functions ───────────────────────────────────────────────────────

def parse_tuh_path(filepath):
    """Extract metadata from TUH EEG file path."""
    path_str = str(filepath)
    result = {"file_path": path_str}

    # Extract subject ID from filename first (most reliable: aaaaaaju_s005_t000.edf)
    stem = Path(path_str).stem
    m = re.search(r'([a-z]{8})_s\d{3}', stem)
    if m:
        result["subject_id"] = m.group(1)
    else:
        # Try TUEV-style filename (aaaaaasu_00000001.edf)
        m = re.search(r'^([a-z]{8})_', stem)
        if m:
            result["subject_id"] = m.group(1)
        else:
            # Fallback: from directory path (standard TUEG layout)
            m = re.search(r'/([a-z]{8})/s\d{3}', path_str)
            if m:
                result["subject_id"] = m.group(1)

    # Extract session info
    m = re.search(r'/(s\d{3}_\d{4}(?:_\d{2}(?:_\d{2})?)?)/', path_str)
    if m:
        result["session"] = m.group(1)

    # Extract montage
    m = re.search(r'/(0[1-4]_tcp_(?:ar|le)(?:_a)?)/', path_str)
    if m:
        result["montage"] = m.group(1)

    # Extract token from filename
    m = re.search(r'_t(\d{3})', Path(path_str).stem)
    if m:
        result["token"] = f"t{m.group(1)}"

    return result


def read_edf_metadata(edf_path):
    """Read EDF header metadata without loading signal data."""
    try:
        f = pyedflib.EdfReader(str(edf_path))
        info = {
            "n_channels": f.signals_in_file,
            "duration_sec": round(f.file_duration, 2),
            "channel_labels": f.getSignalLabels(),
            "sample_rates": [f.getSampleFrequency(i) for i in range(f.signals_in_file)],
            "patient_info": f.getPatientName() or "",
            "start_date": str(f.getStartdatetime()),
        }
        f.close()
        return info
    except Exception as e:
        return {"error": str(e)}


def load_csv_annotations(csv_path):
    """Load TUH EEG CSV annotations (TUAR, TUEP, TUSZ format)."""
    metadata = {}
    rows = []
    with open(csv_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("#"):
                # Parse metadata from comment header
                if "=" in line:
                    key, val = line.lstrip("# ").split("=", 1)
                    metadata[key.strip()] = val.strip()
            elif line and "channel" not in line.lower():
                parts = line.split(",")
                if len(parts) >= 5:
                    rows.append({
                        "channel": parts[0].strip(),
                        "start_time": float(parts[1].strip()),
                        "stop_time": float(parts[2].strip()),
                        "label": parts[3].strip(),
                        "confidence": float(parts[4].strip()),
                    })
    return metadata, rows


def load_tse_annotations(tse_path):
    """Load TSE annotations (TUSL format)."""
    rows = []
    with open(tse_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("version") or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                rows.append({
                    "start_time": float(parts[0]),
                    "stop_time": float(parts[1]),
                    "label": parts[2],
                    "confidence": float(parts[3]),
                })
    return rows


def load_rec_annotations(rec_path):
    """Load REC annotations (TUEV format)."""
    rows = []
    event_map = {0: "null", 1: "spsw", 2: "gped", 3: "pled", 4: "eyem", 5: "artf", 6: "bckg"}
    with open(rec_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) >= 4:
                rows.append({
                    "channel_idx": int(parts[0]),
                    "start_time": float(parts[1]),
                    "stop_time": float(parts[2]),
                    "label_code": int(parts[3]),
                    "label": event_map.get(int(parts[3]), f"unknown_{parts[3]}"),
                })
    return rows


# ─── Corpus-Specific Processors ─────────────────────────────────────────────

def process_tusl(data_root, include_signals=False):
    """Process TUH Slowing Corpus (TUSL)."""
    tusl_dir = data_root / "tusl"
    edf_files = sorted(tusl_dir.rglob("*.edf"))
    print(f"  Found {len(edf_files)} EDF files in TUSL")

    records = []
    for edf_path in tqdm(edf_files, desc="  TUSL"):
        meta = parse_tuh_path(edf_path)
        edf_info = read_edf_metadata(edf_path)
        if "error" in edf_info:
            meta["error"] = edf_info["error"]
            records.append(meta)
            continue

        meta.update({
            "n_channels": edf_info["n_channels"],
            "duration_sec": edf_info["duration_sec"],
            "sample_rate": int(edf_info["sample_rates"][0]) if edf_info["sample_rates"] else 0,
            "start_date": edf_info["start_date"],
        })

        # Load TSE aggregate annotations
        tse_agg = edf_path.with_suffix(".tse_agg")
        if tse_agg.exists():
            ann = load_tse_annotations(tse_agg)
            labels = list(set(r["label"] for r in ann))
            meta["labels"] = labels
            meta["n_annotations"] = len(ann)
            meta["annotations_json"] = json.dumps(ann)
        else:
            # Try per-annotator TSE files
            tse_files = list(edf_path.parent.glob(f"{edf_path.stem}*.tse"))
            if tse_files:
                ann = load_tse_annotations(tse_files[0])
                labels = list(set(r["label"] for r in ann))
                meta["labels"] = labels
                meta["n_annotations"] = len(ann)
                meta["annotations_json"] = json.dumps(ann)
            else:
                meta["labels"] = []
                meta["n_annotations"] = 0
                meta["annotations_json"] = "[]"

        records.append(meta)

    return records


def process_tuar(data_root):
    """Process TUH Artifact Corpus (TUAR)."""
    tuar_dir = data_root / "tuar"
    edf_files = sorted(tuar_dir.rglob("*.edf"))
    print(f"  Found {len(edf_files)} EDF files in TUAR")

    records = []
    for edf_path in tqdm(edf_files, desc="  TUAR"):
        meta = parse_tuh_path(edf_path)
        edf_info = read_edf_metadata(edf_path)
        if "error" in edf_info:
            meta["error"] = edf_info["error"]
            records.append(meta)
            continue

        meta.update({
            "n_channels": edf_info["n_channels"],
            "duration_sec": edf_info["duration_sec"],
            "sample_rate": int(edf_info["sample_rates"][0]) if edf_info["sample_rates"] else 0,
            "start_date": edf_info["start_date"],
        })

        # Load CSV annotations
        csv_path = edf_path.with_suffix(".csv")
        if csv_path.exists():
            csv_meta, ann = load_csv_annotations(csv_path)
            labels = list(set(r["label"] for r in ann))
            channels_with_artifacts = list(set(r["channel"] for r in ann))
            meta["labels"] = labels
            meta["n_annotations"] = len(ann)
            meta["channels_with_artifacts"] = channels_with_artifacts
            meta["annotations_json"] = json.dumps(ann)
        else:
            meta["labels"] = []
            meta["n_annotations"] = 0
            meta["channels_with_artifacts"] = []
            meta["annotations_json"] = "[]"

        # Check for seizure overlay
        seiz_csv = edf_path.with_name(edf_path.stem + "_seiz.csv")
        meta["has_seizure_overlay"] = seiz_csv.exists()

        records.append(meta)

    return records


def process_tuev(data_root):
    """Process TUH Events Corpus (TUEV)."""
    tuev_dir = data_root / "tuev"
    edf_files = sorted(tuev_dir.rglob("*.edf"))
    print(f"  Found {len(edf_files)} EDF files in TUEV")

    records = []
    for edf_path in tqdm(edf_files, desc="  TUEV"):
        meta = parse_tuh_path(edf_path)
        edf_info = read_edf_metadata(edf_path)
        if "error" in edf_info:
            meta["error"] = edf_info["error"]
            records.append(meta)
            continue

        meta.update({
            "n_channels": edf_info["n_channels"],
            "duration_sec": edf_info["duration_sec"],
            "sample_rate": int(edf_info["sample_rates"][0]) if edf_info["sample_rates"] else 0,
            "start_date": edf_info["start_date"],
        })

        # Determine split from path
        if "/train/" in str(edf_path):
            meta["split"] = "train"
        elif "/eval/" in str(edf_path):
            meta["split"] = "eval"
        else:
            meta["split"] = "unknown"

        # Load REC annotations (per-file aggregate)
        rec_files = list(edf_path.parent.glob(f"{edf_path.stem}*.rec"))
        all_annotations = []
        event_counts = defaultdict(int)
        for rec_path in rec_files:
            ann = load_rec_annotations(rec_path)
            all_annotations.extend(ann)
            for r in ann:
                event_counts[r["label"]] += 1

        meta["labels"] = list(event_counts.keys())
        meta["n_annotations"] = len(all_annotations)
        meta["event_counts"] = json.dumps(dict(event_counts))
        meta["annotations_json"] = json.dumps(all_annotations[:500])  # Cap to avoid huge rows

        records.append(meta)

    return records


def process_tuep(data_root):
    """Process TUH Epilepsy Corpus (TUEP)."""
    tuep_dir = data_root / "tuep"
    edf_files = sorted(tuep_dir.rglob("*.edf"))
    print(f"  Found {len(edf_files)} EDF files in TUEP")

    records = []
    for edf_path in tqdm(edf_files, desc="  TUEP"):
        meta = parse_tuh_path(edf_path)
        edf_info = read_edf_metadata(edf_path)
        if "error" in edf_info:
            meta["error"] = edf_info["error"]
            records.append(meta)
            continue

        meta.update({
            "n_channels": edf_info["n_channels"],
            "duration_sec": edf_info["duration_sec"],
            "sample_rate": int(edf_info["sample_rates"][0]) if edf_info["sample_rates"] else 0,
            "start_date": edf_info["start_date"],
        })

        # Determine epilepsy label from directory
        if "/00_epilepsy/" in str(edf_path):
            meta["epilepsy_label"] = "epilepsy"
        elif "/01_no_epilepsy/" in str(edf_path):
            meta["epilepsy_label"] = "no_epilepsy"
        else:
            meta["epilepsy_label"] = "unknown"

        # Load CSV annotations if present
        csv_path = edf_path.with_suffix(".csv")
        if csv_path.exists():
            csv_meta, ann = load_csv_annotations(csv_path)
            labels = list(set(r["label"] for r in ann))
            meta["labels"] = labels
            meta["n_annotations"] = len(ann)
            meta["annotations_json"] = json.dumps(ann[:500])
        else:
            meta["labels"] = []
            meta["n_annotations"] = 0
            meta["annotations_json"] = "[]"

        records.append(meta)

    return records


def process_tuab(data_root):
    """Process TUH Abnormal EEG Corpus (TUAB)."""
    tuab_dir = data_root / "tuab"
    edf_files = sorted(tuab_dir.rglob("*.edf"))
    print(f"  Found {len(edf_files)} EDF files in TUAB")

    if len(edf_files) == 0:
        print("  TUAB not yet downloaded, skipping")
        return []

    records = []
    for edf_path in tqdm(edf_files, desc="  TUAB"):
        meta = parse_tuh_path(edf_path)
        edf_info = read_edf_metadata(edf_path)
        if "error" in edf_info:
            meta["error"] = edf_info["error"]
            records.append(meta)
            continue

        meta.update({
            "n_channels": edf_info["n_channels"],
            "duration_sec": edf_info["duration_sec"],
            "sample_rate": int(edf_info["sample_rates"][0]) if edf_info["sample_rates"] else 0,
            "start_date": edf_info["start_date"],
        })

        # Determine label and split from path
        path_str = str(edf_path)
        if "/normal/" in path_str:
            meta["abnormal_label"] = "normal"
        elif "/abnormal/" in path_str:
            meta["abnormal_label"] = "abnormal"
        else:
            meta["abnormal_label"] = "unknown"

        if "/train/" in path_str:
            meta["split"] = "train"
        elif "/eval/" in path_str:
            meta["split"] = "eval"
        else:
            meta["split"] = "unknown"

        records.append(meta)

    return records


def process_tusz(data_root):
    """Process TUH Seizure Corpus (TUSZ)."""
    tusz_dir = data_root / "tusz"
    edf_files = sorted(tusz_dir.rglob("*.edf"))
    print(f"  Found {len(edf_files)} EDF files in TUSZ")

    if len(edf_files) == 0:
        print("  TUSZ not yet downloaded, skipping")
        return []

    records = []
    for edf_path in tqdm(edf_files, desc="  TUSZ"):
        meta = parse_tuh_path(edf_path)
        edf_info = read_edf_metadata(edf_path)
        if "error" in edf_info:
            meta["error"] = edf_info["error"]
            records.append(meta)
            continue

        meta.update({
            "n_channels": edf_info["n_channels"],
            "duration_sec": edf_info["duration_sec"],
            "sample_rate": int(edf_info["sample_rates"][0]) if edf_info["sample_rates"] else 0,
            "start_date": edf_info["start_date"],
        })

        # Determine split from path
        path_str = str(edf_path)
        if "/train/" in path_str:
            meta["split"] = "train"
        elif "/eval/" in path_str or "/dev/" in path_str:
            meta["split"] = "eval"
        else:
            meta["split"] = "unknown"

        # Load CSV annotations (multi-class)
        csv_path = edf_path.with_suffix(".csv")
        if csv_path.exists():
            csv_meta, ann = load_csv_annotations(csv_path)
            labels = list(set(r["label"] for r in ann))
            meta["labels"] = labels
            meta["n_annotations"] = len(ann)
            meta["has_seizure"] = any(l != "bckg" for l in labels)
            meta["annotations_json"] = json.dumps(ann[:500])
        else:
            meta["labels"] = []
            meta["n_annotations"] = 0
            meta["has_seizure"] = False
            meta["annotations_json"] = "[]"

        # Also check binary annotations
        csv_bi = edf_path.with_suffix(".csv_bi")
        meta["has_binary_annotations"] = csv_bi.exists()

        records.append(meta)

    return records


def process_tueg(data_root):
    """Process TUH EEG Corpus (TUEG) - metadata only, no annotations."""
    tueg_dir = data_root / "tueg"
    edf_files = sorted(tueg_dir.rglob("*.edf"))
    print(f"  Found {len(edf_files)} EDF files in TUEG")

    if len(edf_files) < 100:
        print("  TUEG not yet fully downloaded, skipping")
        return []

    records = []
    for edf_path in tqdm(edf_files, desc="  TUEG"):
        meta = parse_tuh_path(edf_path)
        edf_info = read_edf_metadata(edf_path)
        if "error" in edf_info:
            meta["error"] = edf_info["error"]
            records.append(meta)
            continue

        meta.update({
            "n_channels": edf_info["n_channels"],
            "duration_sec": edf_info["duration_sec"],
            "sample_rate": int(edf_info["sample_rates"][0]) if edf_info["sample_rates"] else 0,
            "start_date": edf_info["start_date"],
        })

        records.append(meta)

    return records


# ─── Dataset Upload ──────────────────────────────────────────────────────────

def records_to_dataset(records, corpus_name):
    """Convert records to a HuggingFace Dataset, normalizing columns."""
    if not records:
        return None

    # Normalize: ensure all records have the same keys
    all_keys = set()
    for r in records:
        all_keys.update(r.keys())

    for r in records:
        for k in all_keys:
            if k not in r:
                if k in ("labels", "channels_with_artifacts"):
                    r[k] = []
                elif k in ("n_annotations", "n_channels", "sample_rate"):
                    r[k] = 0
                elif k in ("duration_sec",):
                    r[k] = 0.0
                elif k in ("has_seizure", "has_seizure_overlay", "has_binary_annotations"):
                    r[k] = False
                else:
                    r[k] = ""

    df = pd.DataFrame(records)

    # Convert list columns to strings for parquet compatibility
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)

    return Dataset.from_pandas(df, preserve_index=False)


def create_dataset_card(corpus_name, version, n_records, columns):
    """Generate a dataset card (README.md) for the HF repo."""
    column_table = "\n".join(f"| `{c}` | - |" for c in columns)

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
pretty_name: TUH EEG - {corpus_name.upper()} Metadata Index
size_categories:
  - 1K<n<10K
private: true
---

# TUH EEG {corpus_name.upper()} - Metadata Index

**Metadata-level dataset** for the Temple University Hospital {corpus_name.upper()} corpus ({version}).

This dataset contains file-level metadata and annotation summaries for {n_records} EDF recordings.
It serves as a queryable index into the raw EDF files stored locally.

## Source

- **Corpus:** {CORPUS_MAP.get(corpus_name, (corpus_name, ""))[0]}
- **Version:** {version}
- **Citation:** Obeid & Picone (2016), *Frontiers in Neuroscience* 10:196

## Schema

| Column | Description |
|--------|-------------|
{column_table}

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{HF_ORG}/{HF_PREFIX}-{corpus_name}")
print(ds)
# Filter for files with specific labels, durations, etc.
```

## License

This dataset is derived from the TUH EEG Corpus and subject to its
[usage agreement](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/).
"""


def upload_to_hf(dataset, corpus_name, version, dry_run=False):
    """Upload dataset to HuggingFace Hub as a private repo."""
    repo_id = f"{HF_ORG}/{HF_PREFIX}-{corpus_name}"

    if dry_run:
        print(f"  [DRY RUN] Would upload {len(dataset)} rows to {repo_id}")
        print(f"  Columns: {dataset.column_names}")
        print(f"  Sample row: {dataset[0]}")
        return

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
        print(f"  Created/verified repo: {repo_id}")
    except Exception as e:
        print(f"  Warning creating repo: {e}")

    # Push dataset
    dataset.push_to_hub(repo_id, private=True)
    print(f"  Uploaded {len(dataset)} rows to https://huggingface.co/datasets/{repo_id}")

    # Upload dataset card
    card = create_dataset_card(corpus_name, version, len(dataset), dataset.column_names)
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"  Uploaded dataset card")


# ─── Main ────────────────────────────────────────────────────────────────────

PROCESSORS = {
    "tusl": process_tusl,
    "tuar": process_tuar,
    "tuev": process_tuev,
    "tuep": process_tuep,
    "tuab": process_tuab,
    "tusz": process_tusz,
    "tueg": process_tueg,
}

# Processing order: smallest first
CORPUS_ORDER = ["tusl", "tuar", "tuev", "tuep", "tuab", "tusz", "tueg"]


def main():
    parser = argparse.ArgumentParser(description="Create HF datasets for TUH EEG corpora")
    parser.add_argument("--corpus", default="all", help="Corpus to process (tusl/tuar/tuev/tuep/tuab/tusz/tueg/all)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without uploading")
    parser.add_argument("--data-root", default=str(DATA_ROOT), help="Path to tuh_eeg data directory")
    args = parser.parse_args()

    data_root = Path(args.data_root)

    if args.corpus == "all":
        corpora = CORPUS_ORDER
    else:
        corpora = [args.corpus]

    for corpus in corpora:
        if corpus not in PROCESSORS:
            print(f"Unknown corpus: {corpus}")
            continue

        full_name, version = CORPUS_MAP[corpus]
        print(f"\n{'='*60}")
        print(f"Processing {corpus.upper()} ({full_name} {version})")
        print(f"{'='*60}")

        processor = PROCESSORS[corpus]
        records = processor(data_root)

        if not records:
            print(f"  No records found for {corpus}, skipping upload")
            continue

        print(f"  Processed {len(records)} records")

        dataset = records_to_dataset(records, corpus)
        if dataset is None:
            continue

        print(f"  Dataset columns: {dataset.column_names}")
        print(f"  Sample: {dataset[0]}")

        upload_to_hf(dataset, corpus, version, dry_run=args.dry_run)

    print(f"\nDone! Processed {len(corpora)} corpora.")


if __name__ == "__main__":
    main()
