#!/bin/bash
# -----------------------------------------------------------------------------
# replace_files.sh - Script to replace specific Python modules with custom versions
# -----------------------------------------------------------------------------
#
# This script replaces three Python modules with custom versions:
# 1. PyTorch Lightning's progress.py module with custom_progress.py
# 2. lean_dojo's traced_data.py module with custom_traced_data.py
# 3. lean_dojo's utils.py module with custom_utils.py
#
# Custom files are taken from this repository (the directory containing this
# script), not from RAID_DIR.
#
# Usage:
#   ./replace_files.sh
#
# Requirements:
#   - Python with pytorch_lightning and lean_dojo installed
#   - Custom module files in this repository
#
# Note: This script modifies installed Python packages. Use with caution.
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Replace PyTorch Lightning progress.py
PL_PATH="$(python -c "import pytorch_lightning as pl; print(pl.__file__)")"
PROGRESS_PY_PATH="$(dirname "$PL_PATH")/loops/progress.py"
echo "Replacing $PROGRESS_PY_PATH"
cp "${SCRIPT_DIR}/custom_progress.py" "$PROGRESS_PY_PATH"

# Replace lean_dojo traced_data.py
LD_PATH="$(python -c "import lean_dojo; print(lean_dojo.__file__)")"
TRACED_DATA_PY_PATH="$(dirname "$LD_PATH")/data_extraction/traced_data.py"
echo "Replacing $TRACED_DATA_PY_PATH"
cp "${SCRIPT_DIR}/custom_traced_data.py" "$TRACED_DATA_PY_PATH"

# Replace lean_dojo utils.py
UTILS_PY_PATH="$(dirname "$LD_PATH")/utils.py"
echo "Replacing $UTILS_PY_PATH"
cp "${SCRIPT_DIR}/custom_utils.py" "$UTILS_PY_PATH"

echo "Done replacing LeanAgent custom modules."
