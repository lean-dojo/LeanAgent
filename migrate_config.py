#!/usr/bin/env python3
"""
Migration script to help users transition from the old configuration approach 
(editing run_leanagent.sh) to the new configuration system (YAML files).

This script:
1. Parses the existing run_leanagent.sh file
2. Extracts configuration variables
3. Creates a YAML configuration file
4. Optionally generates export commands for environment variables
"""

import argparse
import os
import re
import sys
import yaml


def parse_run_leanagent_sh(file_path):
    """Parse the run_leanagent.sh file and extract configuration variables."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None
    
    config = {
        "data": {},
        "retrieval": {},
        "prover": {},
        "github": {}
    }
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract variables using regex
    raid_dir_match = re.search(r'RAID_DIR\s*=\s*["\']?(.*?)["\']?\s*(?:#.*)?$', content, re.MULTILINE)
    if raid_dir_match:
        config["data"]["root_dir"] = raid_dir_match.group(1)
    
    # Extract GitHub access token
    github_token_match = re.search(r'GITHUB_ACCESS_TOKEN\s*=\s*["\']?(.*?)["\']?\s*(?:#.*)?$', content, re.MULTILINE)
    if github_token_match and github_token_match.group(1) != '""':
        config["github"]["access_token"] = github_token_match.group(1)
    
    # Extract other variables - add more as needed
    checkpoint_dir_match = re.search(r'CHECKPOINT_DIR\s*=\s*["\']?(.*?)["\']?\s*(?:#.*)?$', content, re.MULTILINE)
    if checkpoint_dir_match:
        config["data"]["checkpoint_dir"] = checkpoint_dir_match.group(1)
    
    data_dir_match = re.search(r'DATA_DIR\s*=\s*["\']?(.*?)["\']?\s*(?:#.*)?$', content, re.MULTILINE)
    if data_dir_match:
        config["data"]["data_dir"] = data_dir_match.group(1)
    
    # Extract options from the main function
    # This is more complex since these might be buried in Python code
    # For a full solution, you would need more sophisticated parsing
    
    return config


def create_yaml_config(config, output_file):
    """Create a YAML configuration file from the extracted config."""
    # Remove empty sections
    for section in list(config.keys()):
        if not config[section]:
            del config[section]
    
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Configuration saved to {output_file}")


def generate_env_vars(config):
    """Generate export commands for environment variables."""
    env_vars = []
    
    for section, options in config.items():
        for option, value in options.items():
            env_name = f"LEANAGENT_{section.upper()}__{option.upper()}"
            env_vars.append(f"export {env_name}=\"{value}\"")
    
    return env_vars


def main():
    parser = argparse.ArgumentParser(
        description="Migrate from run_leanagent.sh to YAML configuration"
    )
    parser.add_argument(
        "--input", "-i",
        default="run_leanagent.sh",
        help="Input run_leanagent.sh file path"
    )
    parser.add_argument(
        "--output", "-o",
        default="config.yaml",
        help="Output YAML configuration file path"
    )
    parser.add_argument(
        "--env", "-e",
        action="store_true",
        help="Generate environment variable export commands"
    )
    
    args = parser.parse_args()
    
    config = parse_run_leanagent_sh(args.input)
    if not config:
        return 1
    
    create_yaml_config(config, args.output)
    
    if args.env:
        env_vars = generate_env_vars(config)
        print("\nEnvironment variable commands:")
        for var in env_vars:
            print(var)
        
        print("\nSave these to a file (e.g., leanagent-env.sh) and source them with:")
        print(f"source leanagent-env.sh")
    
    print("\nNow you can run LeanAgent with:")
    print(f"leanagent -c {args.output} run")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 