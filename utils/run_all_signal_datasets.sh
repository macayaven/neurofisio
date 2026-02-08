#!/bin/bash
# Process and upload signal datasets for all available corpora sequentially.
# V2: Uses chunked parquet shard uploads (memory-safe).
# Run from spark-caeb as: nohup ./run_all_signal_datasets.sh > logs/signal_datasets_master_v2.log 2>&1 &

VENV="/home/carlos/workspace/neurofisio/.venv/bin/python"
SCRIPT="/home/carlos/workspace/neurofisio/create_hf_signal_datasets.py"
LOGDIR="/home/carlos/workspace/neurofisio/data/tuh_eeg/logs"

mkdir -p "$LOGDIR"

for corpus in tuar tuev tuep tuab tusz; do
    echo "[$(date)] Starting signal extraction: $corpus"

    # Check if corpus has EDF files
    count=$(find /home/carlos/workspace/neurofisio/data/tuh_eeg/$corpus -name '*.edf' 2>/dev/null | wc -l)
    if [ "$count" -eq 0 ]; then
        echo "[$(date)] $corpus: no EDF files found, skipping"
        continue
    fi
    echo "[$(date)] $corpus: $count EDF files found"

    $VENV $SCRIPT --corpus $corpus 2>&1 | tee "$LOGDIR/${corpus}_signals_v2.log"
    exit_code=$?
    echo "[$(date)] Finished $corpus with exit code $exit_code"
    echo ""
done

echo "[$(date)] All signal datasets complete!"
