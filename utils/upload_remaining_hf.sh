#!/bin/bash
# Run this manually after TUSZ/TUEG downloads complete to create their HF datasets.
# Also re-runs TUAB to capture any files added since the partial upload.
#
# Usage: ./upload_remaining_hf.sh

VENV="/home/carlos/workspace/neurofisio/.venv/bin/python"
SCRIPT="/home/carlos/workspace/neurofisio/create_hf_datasets.py"

echo "=== Checking download status ==="
for corpus in tuab tusz tueg; do
    count=$(find /home/carlos/workspace/neurofisio/data/tuh_eeg/$corpus -name '*.edf' 2>/dev/null | wc -l)
    size=$(du -sh /home/carlos/workspace/neurofisio/data/tuh_eeg/$corpus 2>/dev/null | cut -f1)
    echo "  $corpus: $count EDF files, $size"
done

echo ""
echo "=== Processing corpora ==="

# Re-run TUAB (may have more files now)
echo "--- TUAB ---"
$VENV $SCRIPT --corpus tuab

# TUSZ
TUSZ_COUNT=$(find /home/carlos/workspace/neurofisio/data/tuh_eeg/tusz -name '*.edf' 2>/dev/null | wc -l)
if [ "$TUSZ_COUNT" -gt 0 ]; then
    echo "--- TUSZ ($TUSZ_COUNT files) ---"
    $VENV $SCRIPT --corpus tusz
else
    echo "--- TUSZ: not yet downloaded, skipping ---"
fi

# TUEG
TUEG_COUNT=$(find /home/carlos/workspace/neurofisio/data/tuh_eeg/tueg -name '*.edf' 2>/dev/null | wc -l)
if [ "$TUEG_COUNT" -gt 100 ]; then
    echo "--- TUEG ($TUEG_COUNT files) ---"
    $VENV $SCRIPT --corpus tueg
else
    echo "--- TUEG: not yet downloaded ($TUEG_COUNT files), skipping ---"
fi

echo ""
echo "=== Done ==="
