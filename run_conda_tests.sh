#!/bin/bash
# Script to run tests in the conda environment for LeanAgent

# Configure conda environment name
ENV_NAME=${1:-leanagent}

# Activate conda
eval "$(conda shell.bash hook)"

# Check if the environment exists
if ! conda env list | grep -q "$ENV_NAME"; then
    echo "❌ Conda environment '$ENV_NAME' not found!"
    echo "Please run ./setup_conda_env.sh first."
    exit 1
fi

echo "Activating conda environment '$ENV_NAME'..."
conda activate $ENV_NAME

echo "Running tests with coverage report..."
python -m pytest --cov=leanagent tests/ -v 