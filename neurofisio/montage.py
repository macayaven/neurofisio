"""Standard TCP bipolar montage used across all TUH EEG corpora."""

import mne
import numpy as np

# 22 bipolar channel pairs for the TCP montage (anode, cathode).
# AR channels use ``EEG XX-REF``, LE channels ``EEG XX-LE``.
TCP_MONTAGE_PAIRS = [
    ("EEG FP1-REF", "EEG F7-REF"),
    ("EEG F7-REF", "EEG T3-REF"),
    ("EEG T3-REF", "EEG T5-REF"),
    ("EEG T5-REF", "EEG O1-REF"),
    ("EEG FP2-REF", "EEG F8-REF"),
    ("EEG F8-REF", "EEG T4-REF"),
    ("EEG T4-REF", "EEG T6-REF"),
    ("EEG T6-REF", "EEG O2-REF"),
    ("EEG A1-REF", "EEG T3-REF"),
    ("EEG T3-REF", "EEG C3-REF"),
    ("EEG C3-REF", "EEG CZ-REF"),
    ("EEG CZ-REF", "EEG C4-REF"),
    ("EEG C4-REF", "EEG T4-REF"),
    ("EEG T4-REF", "EEG A2-REF"),
    ("EEG FP1-REF", "EEG F3-REF"),
    ("EEG F3-REF", "EEG C3-REF"),
    ("EEG C3-REF", "EEG P3-REF"),
    ("EEG P3-REF", "EEG O1-REF"),
    ("EEG FP2-REF", "EEG F4-REF"),
    ("EEG F4-REF", "EEG C4-REF"),
    ("EEG C4-REF", "EEG P4-REF"),
    ("EEG P4-REF", "EEG O2-REF"),
]

TCP_NAMES = [
    "FP1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "FP2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
    "A1-T3",
    "T3-C3",
    "C3-CZ",
    "CZ-C4",
    "C4-T4",
    "T4-A2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
]


def apply_tcp_montage(raw):
    """Convert a raw MNE object to the 22-channel TCP bipolar montage.

    Automatically detects ``-REF`` vs ``-LE`` channel naming.
    Returns a new ``RawArray`` or ``None`` if no matching pairs are found.
    """
    available = raw.ch_names
    pairs = list(TCP_MONTAGE_PAIRS)

    if any("LE" in ch for ch in available):
        pairs = [(a.replace("-REF", "-LE"), b.replace("-REF", "-LE")) for a, b in pairs]

    bipolar_data, ch_names = [], []
    for i, (anode, cathode) in enumerate(pairs):
        if anode in available and cathode in available:
            sig = raw.get_data(picks=anode)[0] - raw.get_data(picks=cathode)[0]
            bipolar_data.append(sig)
            ch_names.append(TCP_NAMES[i])

    if not bipolar_data:
        return None
    info = mne.create_info(ch_names, raw.info["sfreq"], ch_types="eeg")
    return mne.io.RawArray(np.array(bipolar_data), info)
