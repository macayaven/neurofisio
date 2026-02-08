#!/usr/bin/env bash
# Launch the NeuroFisio EEG Explorer web app.
# Usage: ./app/run.sh [--port 8501]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Streamlit reads .streamlit/config.toml from the working directory
cd "$SCRIPT_DIR"
exec uv run --project "$PROJECT_DIR" streamlit run eeg_explorer.py "$@"
