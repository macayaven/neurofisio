"""neurofisio — TUH EEG corpus utilities for the NeuroTriage project."""

from .parsers import (
    parse_tuh_path,
    load_csv_annotations,
    load_tse_annotations,
    load_rec_annotations,
    load_lab_annotations,
)
from .edf import (
    inspect_edf,
    safe_read_raw_edf,
    inventory_corpus,
)
from .montage import (
    TCP_MONTAGE_PAIRS,
    TCP_NAMES,
    apply_tcp_montage,
)
from .viz import (
    plot_eeg_segment,
)

__all__ = [
    # parsers
    'parse_tuh_path',
    'load_csv_annotations',
    'load_tse_annotations',
    'load_rec_annotations',
    'load_lab_annotations',
    # edf
    'inspect_edf',
    'safe_read_raw_edf',
    'inventory_corpus',
    # montage
    'TCP_MONTAGE_PAIRS',
    'TCP_NAMES',
    'apply_tcp_montage',
    # viz
    'plot_eeg_segment',
]
