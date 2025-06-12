# Command Mapping: Original LeanAgent vs. Refactored Version

This document maps the commands used in the original LeanAgent repository to the corresponding commands in our refactored version.

## Original Commands

In the original LeanAgent repository, operations were performed using shell scripts:

```bash
# Run LeanAgent with default options
bash run_leanagent.sh

# Compute Fisher Information Matrix
bash run_compute_fisher.sh
```

These scripts had hardcoded paths and configuration, requiring manual edits to the scripts to change parameters or directories.

## New Commands

The refactored version provides a modern command-line interface with the following improvements:

1. Structured command hierarchy
2. Configuration management via files, environment variables, and CLI
3. Modular components that can be run individually
4. Better help and documentation

### Basic Commands

| Original Command | New Command | Description |
|------------------|-------------|-------------|
| `bash run_leanagent.sh` | `leanagent run --component all` | Run the full LeanAgent system |
| Edit `run_leanagent.sh` to change parameters | `leanagent config --set section option value` | Change configuration values |
| N/A | `leanagent config --show` | Display current configuration |
| N/A | `leanagent run --component retrieval` | Run only the retrieval component |
| N/A | `leanagent run --component prover` | Run only the theorem prover component |

### Configuration Management

Instead of editing shell scripts or Python files, configuration can now be managed in several ways:

1. **Configuration files**: Create a YAML config file and load it with `leanagent -c config.yaml`
2. **Environment variables**: Set variables like `LEANAGENT_DATA__ROOT_DIR="/path/to/data"`
3. **Command line**: Use `leanagent config --set data root_dir /path/to/data`

### Example Workflows

#### Original Workflow:

```bash
# 1. Edit run_leanagent.sh to set RAID_DIR and other parameters
vim run_leanagent.sh

# 2. Run LeanAgent
bash run_leanagent.sh

# 3. To compute Fisher matrices (for EWC)
bash run_compute_fisher.sh
```

#### New Workflow:

```bash
# 1. Set configuration
leanagent config --set data root_dir /path/to/data
leanagent config --set data repo_dir /path/to/repos
leanagent config --set retrieval k 10

# 2. Run the full system
leanagent run

# 3. For Fisher matrix computation (EWC)
leanagent run --component fisher
```

## Environment Variables

| Original Variable | New Variable | Description |
|-------------------|--------------|-------------|
| `RAID_DIR` | `LEANAGENT_DATA__ROOT_DIR` | Base directory for all data |
| `GITHUB_ACCESS_TOKEN` | `LEANAGENT_GITHUB__ACCESS_TOKEN` | GitHub personal access token |
| N/A | `LEANAGENT_RETRIEVAL__K` | Number of premises to retrieve |
| N/A | `LEANAGENT_CONFIG_FILE` | Path to configuration file 