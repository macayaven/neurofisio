"""EDF file inspection, loading, and corpus inventory."""

import glob
from collections import Counter
from pathlib import Path

import mne
import pyedflib

from .parsers import parse_tuh_path


def inspect_edf(edf_path):
    """Read EDF header without loading signal data.

    Returns a dict with n_channels, channel_labels, duration_sec,
    sample_freqs, patient_info.  On failure returns ``{'error': msg}``.
    """
    try:
        f = pyedflib.EdfReader(str(edf_path))
        info = {
            "n_channels": f.signals_in_file,
            "channel_labels": f.getSignalLabels(),
            "duration_sec": f.file_duration,
            "sample_freqs": [f.getSampleFrequency(i) for i in range(min(f.signals_in_file, 5))],
            "patient_info": f.getPatientName(),
        }
        f.close()
        return info
    except Exception as e:
        return {"error": str(e)}


def safe_read_raw_edf(edf_path, **kwargs):
    """Load an EDF via MNE, returning ``None`` for partial/corrupt files."""
    try:
        return mne.io.read_raw_edf(edf_path, preload=True, verbose=False, **kwargs)
    except Exception:
        return None


def inventory_corpus(corpus_path, name):
    """Build inventory of a TUH EEG corpus directory.

    Returns a dict with file counts, subject count, montage distribution,
    and sample file paths.
    """
    corpus_path = Path(corpus_path)
    edf_files = sorted(glob.glob(str(corpus_path / "**/*.edf"), recursive=True))
    csv_files = sorted(glob.glob(str(corpus_path / "**/*.csv"), recursive=True))
    csv_bi_files = sorted(glob.glob(str(corpus_path / "**/*.csv_bi"), recursive=True))
    tse_files = sorted(glob.glob(str(corpus_path / "**/*.tse"), recursive=True))
    lab_files = sorted(glob.glob(str(corpus_path / "**/*.lab"), recursive=True))
    rec_files = sorted(glob.glob(str(corpus_path / "**/*.rec"), recursive=True))

    subjects = set()
    montages = Counter()
    for f in edf_files:
        meta = parse_tuh_path(f)
        if meta:
            subjects.add(meta["subject_id"])
            if meta["montage"]:
                montages[meta["montage"]] += 1

    return {
        "corpus": name,
        "edf_files": len(edf_files),
        "csv_files": len(csv_files),
        "csv_bi_files": len(csv_bi_files),
        "tse_files": len(tse_files),
        "lab_files": len(lab_files),
        "rec_files": len(rec_files),
        "subjects": len(subjects),
        "montages": dict(montages),
        "sample_edf": edf_files[0] if edf_files else None,
        "sample_csv": csv_files[0] if csv_files else None,
    }
