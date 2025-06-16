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
# The script first locates the installation paths of the modules using Python's
# import mechanism, then replaces them with custom versions from the RAID_DIR
# directory. Before and after replacement, it displays the contents of each file
# for verification.
#
# Usage:
#   1. Set the RAID_DIR environment variable to point to the directory containing
#      the custom module files
#   2. Run the script: ./replace_files.sh
#
# Requirements:
#   - Python with pytorch_lightning and lean_dojo installed
#   - Custom module files (custom_progress.py, custom_traced_data.py, custom_utils.py)
#     located in $RAID_DIR/LeanAgent/
#
# Note: This script modifies installed Python packages. Use with caution as it may
# affect the behavior of any code using these packages.
# -----------------------------------------------------------------------------
export RAID_DIR="<RAID_DIR>"

# Replace PyTorch Lightning progress.py
python -c "import pytorch_lightning as pl; print(pl.__file__)" > pl_path.txt
PL_PATH=$(cat pl_path.txt)
PROGRESS_PY_PATH=$(dirname $PL_PATH)/loops/progress.py
echo "Contents before replacement:"
cat $PROGRESS_PY_PATH
echo "Replacing $PROGRESS_PY_PATH"
cp ${RAID_DIR}/LeanAgent/custom_progress.py $PROGRESS_PY_PATH
echo "Contents after replacement:"
cat $PROGRESS_PY_PATH

# Replace lean_dojo traced_data.py
python -c "import lean_dojo; print(lean_dojo.__file__)" > ld_path.txt
LD_PATH=$(cat ld_path.txt)
TRACED_DATA_PY_PATH=$(dirname $LD_PATH)/data_extraction/traced_data.py
echo "Contents before replacement:"
cat $TRACED_DATA_PY_PATH
echo "Replacing $TRACED_DATA_PY_PATH"
cp ${RAID_DIR}/LeanAgent/custom_traced_data.py $TRACED_DATA_PY_PATH
echo "Contents after replacement:"
cat $TRACED_DATA_PY_PATH

# Replace lean_dojo utils.py
python -c "import lean_dojo; print(lean_dojo.__file__)" > ld_path.txt
LD_PATH=$(cat ld_path.txt)
UTILS_PY_PATH=$(dirname $LD_PATH)/utils.py
echo "Contents before replacement:"
cat $UTILS_PY_PATH
echo "Replacing $UTILS_PY_PATH"
cp ${RAID_DIR}/LeanAgent/custom_utils.py $UTILS_PY_PATH
echo "Contents after replacement:"
cat $UTILS_PY_PATH

rm -f pl_path.txt ld_path.txt
