"""neurofisio — TUH EEG corpus utilities for the NeuroTriage project."""

from .edf import (
    inspect_edf,
    inventory_corpus,
    safe_read_raw_edf,
)
from .montage import (
    TCP_MONTAGE_PAIRS,
    TCP_NAMES,
    apply_tcp_montage,
)
from .parsers import (
    load_csv_annotations,
    load_lab_annotations,
    load_rec_annotations,
    load_tse_annotations,
    parse_tuh_path,
)
from .viz import (
    plot_eeg_segment,
)

__all__ = [
    # montage
    "TCP_MONTAGE_PAIRS",
    "TCP_NAMES",
    "apply_tcp_montage",
    # edf
    "inspect_edf",
    "inventory_corpus",
    "load_csv_annotations",
    "load_lab_annotations",
    "load_rec_annotations",
    "load_tse_annotations",
    # parsers
    "parse_tuh_path",
    # viz
    "plot_eeg_segment",
    "safe_read_raw_edf",
]
