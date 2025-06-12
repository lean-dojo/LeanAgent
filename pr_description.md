# Refactoring LeanAgent: Modern CLI and Configuration

This PR refactors LeanAgent to provide a modern CLI and configuration system while preserving all the original functionality. The goal is to make the codebase more maintainable, testable, and user-friendly without changing the core theorem proving capabilities.

## Changes

1. **Modern Python Package Structure**
   - Restructured code into a proper Python package with a clear module hierarchy
   - Added proper `setup.py` and `pyproject.toml` for installation
   - Added type hints throughout the codebase

2. **Command Line Interface**
   - Created a structured CLI with subcommands using `argparse`
   - Commands now follow modern patterns: `leanagent run --component retrieval`
   - Added help text and documentation for all commands

3. **Configuration Management**
   - Added support for YAML configuration files
   - Added support for environment variables with proper nesting (e.g., `LEANAGENT_DATA__ROOT_DIR`)
   - Added CLI-based configuration management: `leanagent config --set data root_dir /path/to/data`
   - Configuration values are properly typed and validated

4. **Testing Infrastructure**
   - Added pytest infrastructure
   - Added unit tests for core components
   - Added test fixtures and helpers

5. **Documentation**
   - Updated README.md to explain both the original and new interfaces
   - Added command_mapping.md to help users transition from the old to the new interface
   - Added inline docstrings throughout the codebase

6. **Migration Tools**
   - Added `migrate_config.py` script to help users convert from editing shell scripts to using configuration files
   - The script automatically extracts variables from `run_leanagent.sh` and creates a YAML config file
   - Optionally generates environment variable export commands

## Command Mapping

The `command_mapping.md` file provides a detailed mapping from old commands to new commands. Here's a brief summary:

| Original Command | New Command | Description |
|------------------|-------------|-------------|
| `bash run_leanagent.sh` | `leanagent run --component all` | Run the full LeanAgent system |
| Edit `run_leanagent.sh` | `leanagent config --set section option value` | Change configuration values |
| N/A | `leanagent config --show` | Display current configuration |
| N/A | `leanagent run --component retrieval` | Run only the retrieval component |

## Installation

The package can now be installed with pip:

```bash
# Install from the repository
pip install -e .

# Install from PyPI (future)
pip install leanagent
```

## Migration Guide

For users of the original LeanAgent, we've provided tools to ease the transition:

1. **Automatic configuration migration**:
   ```bash
   python migrate_config.py --input run_leanagent.sh --output config.yaml
   ```
   
2. **Environment variable generation**:
   ```bash
   python migrate_config.py --env > leanagent-env.sh
   source leanagent-env.sh
   ```

3. **Command mapping document** (`command_mapping.md`) with detailed examples

## Backward Compatibility

While the interface has changed, we've preserved all the original functionality:

1. The original scripts still work as they did before
2. The new CLI provides all the same capabilities but with improved usability
3. The core theorem proving engine remains unchanged

## Future Work

This refactoring lays the groundwork for:

1. Better testing and continuous integration
2. More modular components that can be reused in other projects
3. Easier collaboration through clearer code organization
4. Better documentation and examples

## Credits

This PR preserves all the original work and credit goes to the original authors of LeanAgent. The refactoring aims only to make their great work more accessible and maintainable.

Adarsh Kumarappan and Mo Tiwari contributed equally to the original LeanAgent work. 