# LeanAgent: Lifelong Learning for Formal Theorem Proving

LeanAgent is a novel lifelong learning framework for formal theorem proving that continuously generalizes to and improves on ever-expanding mathematical knowledge without forgetting previously learned knowledge.

## Features

- Lifelong learning for formal theorem proving
- Curriculum learning based on theorem complexity
- Progressive training of retrieval models
- Best-first tree search for theorem proving
- Configuration management with support for files, environment variables, and CLI
- Modular design with clear separation of concerns
- Integrated test suite with high code coverage

## Documentation

Generated docs by Devin are here: [https://deepwiki.com/lean-dojo/LeanAgent](https://deepwiki.com/lean-dojo/LeanAgent). It includes a detailed explanation + diagrams.

## Requirements

* Supported platforms: Linux and macOS
* Git >= 2.25
* 3.9 <= Python < 3.12
* wget
* elan
* Sufficient disk space for model checkpoints and data

## Installation

### Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/lean-dojo/LeanAgent.git
cd LeanAgent

# Create and set up conda environment
./setup_conda_env.sh

# Activate the environment
conda activate leanagent
```

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/lean-dojo/LeanAgent.git
cd LeanAgent

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Usage

### Command Line Interface

```bash
# Show configuration
leanagent config --show

# Set configuration value
leanagent config --set data root_dir /path/to/data

# Run a component
leanagent run --component retrieval
```

### Configuration

Configuration can be loaded from:
1. YAML configuration files
2. Environment variables prefixed with `LEANAGENT_`
3. Command-line arguments

Example configuration file:
```yaml
data:
  root_dir: "./data"
  repo_dir: "./data/repos"
retrieval:
  k: 10
  similarity_threshold: 0.7
```

Environment variables use double underscore to indicate nesting:
```bash
export LEANAGENT_DATA__ROOT_DIR="/path/to/data"
export LEANAGENT_RETRIEVAL__K=20
```

## Development

### Running Tests

```bash
# Run all tests with the current Python environment
./run_tests.sh

# Run tests in the conda environment
./run_conda_tests.sh
```

### Project Structure

```
leanagent/
├── __init__.py           # Version and package info
├── config.py             # Configuration management
├── cli.py                # Command-line interface
├── retrieval/            # Retrieval functionality
├── prover/               # Prover functionality 
├── db/                   # Database functionality
└── utils/                # Utility functions
```

## Original LeanAgent Setup

The following sections describe the original setup process for LeanAgent if you want to use it directly.

### Step 1: Configure Environment

1. Set the `RAID_DIR` variable to your desired directory path
2. Install Conda if not already installed
3. Create and activate a dedicated environment:

```
conda create -n "LeanAgent" python=3.10
conda activate LeanAgent
pip install -r requirements.txt
```

4. Create a GitHub personal access token and set the environment variable `GITHUB_ACCESS_TOKEN`

### Step 2: Install Models

1. For ReProver's Tactic Generator:

```
pip install gdown
gdown https://drive.google.com/uc?id=11DXxixg6S4-hUA-u-78geOxEB7J7rCoX
```

Move the downloaded file to `{RAID_DIR}/`

2. For ReProver's Starting Retriever:

```
gdown https://drive.google.com/uc?id=1aRd1jQPu_TX15Ib5htzn3wZqHYowl3ax
```

Move the downloaded file to:
* `{RAID_DIR}/checkpoints/`
* `{RAID_DIR}/{CHECKPOINT_DIR}/`

3. For the latest LeanAgent checkpoint from the paper:

```
gdown https://drive.google.com/uc?id=1plkC7Y5n0OVCJ0Ad6pH8_mwbALKYY_FY
```

Move the downloaded file to `{RAID_DIR}/checkpoints/`

## Architecture Overview

LeanAgent is a lifelong learning framework for formal theorem proving that continuously generalizes to and improves on expanding mathematical knowledge without forgetting previously learned information. The codebase consists of several key components:

1. Repository management and data extraction
2. Dynamic database management
3. Curriculum learning strategy
4. Progressive training of the retriever
5. Theorem proving with best-first tree search

### Repository Management and Data Extraction

LeanAgent begins by searching for and processing Lean repositories from GitHub. The system maintains a list of known repositories to avoid duplicate processing and uses the GitHub API to identify repositories with Lean as their primary language.

For each identified repository, LeanAgent checks compatibility by determining if the repository uses a supported Lean version. This is done by examining the repository's `lean-toolchain` configuration file. Currently, supported versions range from Lean `4.3.0-rc2` to Lean `4.8.0-rc1`. If compatible, the system clones the repository and extracts its commit SHA for version tracking.

### Curriculum Learning Strategy

LeanAgent implements a sophisticated curriculum learning strategy based on theorem complexity. The system calculates the complexity of each theorem using an exponential function of the number of proof steps: $e^S$ where $S$ is the number of proof steps. This exponential scaling accounts for the combinatorial explosion of possible proof paths as proofs get longer.

### Progressive Training of the Retriever

LeanAgent employs a simple but effective progressive training approach to avoid catastrophic forgetting. Starting with a pre-trained retriever (based on ByT5), the system trains on each new repository dataset for one additional epoch.

### `sorry` Theorem Proving

LeanAgent identifies theorems marked with `sorry` in the Lean files and attempts to generate formal proofs for them using a best-first tree search approach.

## Contributing

1. Make sure all tests pass before submitting a PR
2. Follow test-driven development practices
3. Maintain high code coverage
4. Keep PRs focused on a single conceptual unit

## Citation

If you find our work useful, please consider citing our paper:

```
@inproceedings{kumarappan2025leanagent,
  title={{LeanAgent}: Lifelong Learning for Formal Theorem Proving},
  author={Kumarappan, Adarsh and Tiwari, Mo and Song, Peiyang and George, Robert Joseph and Xiao, Chaowei and Anandkumar, Anima},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

Adarsh Kumarappan and Mo Tiwari contributed equally to this work.
