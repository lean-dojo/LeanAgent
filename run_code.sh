#!/bin/bash
cd /home/adarsh/ReProver
echo "Script executed from: ${PWD}"
source /home/adarsh/miniconda3/etc/profile.d/conda.sh
conda activate ReProver
export PYTHONPATH="${PYTHONPATH}:/home/adarsh/ReProver"
export GITHUB_ACCESS_TOKEN="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"
export CACHE_DIR="/raid/adarsh/.cache/lean_dojo"
# TODO: change for GPU
export CUDA_VISIBLE_DEVICES=1 # TODO: remove from release
python /home/adarsh/ReProver/main.py
