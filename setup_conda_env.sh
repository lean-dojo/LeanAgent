#!/bin/bash
# Script to set up a conda environment for LeanAgent development

# Exit on error
set -e

# Configure conda environment name
ENV_NAME=${1:-leanagent}
PYTHON_VERSION="3.10"  # Using a specific Python version for compatibility

echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."

# Create conda environment
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

# Activate environment
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Install core dependencies using pip
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Install the package in development mode
echo "Installing LeanAgent in development mode..."
pip install -e .

echo "✅ Conda environment '$ENV_NAME' set up successfully!"
echo "To activate: conda activate $ENV_NAME"
echo "To run tests: bash run_tests.sh"
echo "To run the CLI: ./run_leanagent.sh" 