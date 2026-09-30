# LeanAgent Runner Script
#
# This script sets up and executes the LeanAgent environment by:
# 1. Setting the RAID directory path
# 2. Changing to the LeanAgent directory
# 3. Activating the conda environment for LeanAgent
# 4. Setting environment variables:
#    - Adding LeanAgent to PYTHONPATH
#    - Setting GitHub access token
#    - Setting cache directory for lean_dojo
# 5. Cleaning up:
#    - Removing temporary Ray files
#    - Creating a new Ray temp directory
#    - Stopping any running Ray instances
#    - Killing existing Python processes to free memory
# 6. Displaying GPU information via nvidia-smi
# 7. Running the main leanagent.py script
#
# Prerequisites:
# - Conda installed with LeanAgent environment
# - RAID_DIR: Replace <RAID_DIR> with path to RAID storage
# - PATH_TO_CONDA_ENV: Replace with path to conda installation
# - GITHUB_ACCESS_TOKEN: Replace <GITHUB_ACCESS_TOKEN> with valid token
#
# Usage: bash run_leanagent.sh
#!/bin/bash
export RAID_DIR="<RAID_DIR>"
cd ${RAID_DIR}/LeanAgent
echo "Script executed from: ${PWD}"
source <PATH_TO_CONDA_ENV>/etc/profile.d/conda.sh
conda activate LeanAgent
export PYTHONPATH="${PYTHONPATH}:${RAID_DIR}/LeanAgent"
export GITHUB_ACCESS_TOKEN="<GITHUB_ACCESS_TOKEN>"
export CACHE_DIR="${RAID_DIR}/.cache/lean_dojo"
echo "Removing old cache files"
rm -rf /tmp/ray
export RAY_TMPDIR="${RAID_DIR}/tmp"
rm -rf ${RAY_TMPDIR}
mkdir "${RAY_TMPDIR}"
echo "Stopping ray"
ray stop --force
echo "Skipping blanket kill of all Python processes (was: kill -9 every python PID)."
echo "If you need to free GPU memory, stop LeanAgent/Ray workers explicitly."
sleep 2
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi
echo "Running leanagent.py"
python leanagent.py
