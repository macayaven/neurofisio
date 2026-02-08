"""TUH EEG annotation and path parsers."""

import re
from pathlib import Path

import pandas as pd


def parse_tuh_path(filepath):
    """Extract metadata from a TUH EEG file path.

    Returns dict with subject_id, session_num, token_num, montage, path
    or None if the filename doesn't match TUH naming convention.
    """
    p = Path(filepath)
    match = re.match(r'([a-z]{8})_s(\d+)_t(\d+)', p.stem)
    if not match:
        return None
    montage = None
    for part in p.parts:
        if part.startswith(('01_', '02_', '03_', '04_')):
            montage = part
            break
    return {
        'subject_id': match.group(1),
        'session_num': int(match.group(2)),
        'token_num': int(match.group(3)),
        'montage': montage,
        'path': str(filepath),
    }


def load_csv_annotations(csv_path):
    """Load TUH CSV annotations (TUSZ, TUAR, TUEP).

    Skips ``#`` comment header lines and extracts ``key = value`` metadata.
    Returns (DataFrame, metadata_dict).
    """
    try:
        metadata = {}
        header_lines = 0
        with open(csv_path) as f:
            for line in f:
                if line.startswith('#'):
                    header_lines += 1
                    if '=' in line:
                        key, val = line.strip('# \n').split('=', 1)
                        metadata[key.strip()] = val.strip()
                else:
                    break
        df = pd.read_csv(csv_path, skiprows=header_lines)
        return df, metadata
    except Exception:
        return pd.DataFrame(), {}


def load_tse_annotations(tse_path):
    """Load TSE annotations (TUSL).

    Format per line: ``start_sec stop_sec label [prob]``
    """
    events = []
    try:
        with open(tse_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(('#', 'version')):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    events.append({
                        'start': float(parts[0]),
                        'stop': float(parts[1]),
                        'label': parts[2],
                        'prob': float(parts[3]) if len(parts) > 3 else 1.0,
                    })
    except Exception:
        pass
    return pd.DataFrame(events)


def load_rec_annotations(rec_path):
    """Load REC annotations (TUEV).

    Format per line: ``channel,start,stop,label_code``
    """
    LABEL_MAP = {1: 'spsw', 2: 'gped', 3: 'pled', 4: 'eyem', 5: 'artf', 6: 'bckg'}
    events = []
    try:
        with open(rec_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                if len(parts) >= 4:
                    events.append({
                        'channel': int(parts[0]),
                        'start': float(parts[1]),
                        'stop': float(parts[2]),
                        'label_code': int(parts[3]),
                        'label': LABEL_MAP.get(int(parts[3]), f'unknown_{parts[3]}'),
                    })
    except Exception:
        pass
    return pd.DataFrame(events)


def load_lab_annotations(lab_path):
    """Load LAB annotations (TUEV/TUSL).

    Format per line: ``start_10us stop_10us label``
    Timestamps are in units of 10 microseconds.
    """
    events = []
    try:
        with open(lab_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    events.append({
                        'start_sec': int(parts[0]) / 1e5,
                        'stop_sec': int(parts[1]) / 1e5,
                        'label': parts[2],
                    })
    except Exception:
        pass
    return pd.DataFrame(events)
