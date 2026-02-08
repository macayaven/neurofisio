"""Smoke tests — verify the package imports and public API is intact."""

import neurofisio


def test_package_imports():
    """All public symbols from neurofisio.__init__ should be importable."""
    expected = [
        "parse_tuh_path",
        "load_csv_annotations",
        "load_tse_annotations",
        "load_rec_annotations",
        "load_lab_annotations",
        "inspect_edf",
        "safe_read_raw_edf",
        "inventory_corpus",
        "TCP_MONTAGE_PAIRS",
        "TCP_NAMES",
        "apply_tcp_montage",
        "plot_eeg_segment",
    ]
    for name in expected:
        assert hasattr(neurofisio, name), f"neurofisio.{name} not found in public API"


def test_tcp_montage_constants():
    """TCP montage constants should have expected dimensions."""
    assert len(neurofisio.TCP_MONTAGE_PAIRS) == 22
    assert len(neurofisio.TCP_NAMES) == 22
    assert all(len(pair) == 2 for pair in neurofisio.TCP_MONTAGE_PAIRS)


def test_parse_tuh_path_valid():
    """parse_tuh_path should extract metadata from a valid TUH filename."""
    result = neurofisio.parse_tuh_path(
        "edf/train/aaaaaaaa/s001_2020_01_01/02_tcp_le/aaaaaaaa_s001_t001.edf"
    )
    assert result is not None
    assert result["subject_id"] == "aaaaaaaa"
    assert result["session_num"] == 1
    assert result["token_num"] == 1
    assert result["montage"] == "02_tcp_le"


def test_parse_tuh_path_invalid():
    """parse_tuh_path should return None for non-TUH filenames."""
    assert neurofisio.parse_tuh_path("/some/random/file.edf") is None
