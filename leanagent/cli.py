"""
Command-line interface for LeanAgent.

This module provides a command-line interface for interacting with LeanAgent
functionality, including configuration management and running components.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from leanagent.config import get_config


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.
    
    Args:
        args: List of command-line arguments (defaults to sys.argv[1:])
        
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="LeanAgent - Lifelong Learning for Formal Theorem Proving",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Global options
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase verbosity (can be used multiple times)"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Command: config
    config_parser = subparsers.add_parser(
        "config",
        help="View or modify configuration"
    )
    config_parser.add_argument(
        "--show",
        action="store_true",
        help="Show current configuration"
    )
    config_parser.add_argument(
        "--set",
        nargs=3,
        metavar=("SECTION", "OPTION", "VALUE"),
        action="append",
        help="Set configuration value (can be used multiple times)"
    )
    
    # Command: run
    run_parser = subparsers.add_parser(
        "run",
        help="Run a component or the full system"
    )
    run_parser.add_argument(
        "--component",
        choices=["retrieval", "prover", "all"],
        default="all",
        help="Component to run"
    )
    
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the command-line interface.
    
    Args:
        args: Command-line arguments (defaults to sys.argv[1:])
        
    Returns:
        Exit code
    """
    parsed_args = parse_args(args)
    
    # Load configuration
    config = get_config(parsed_args.config)
    
    if parsed_args.command == "config":
        if parsed_args.show:
            # Show current configuration
            print("Current configuration:")
            for section, options in config.as_dict().items():
                print(f"[{section}]")
                for option, value in options.items():
                    print(f"  {option} = {value}")
                print()
        
        if parsed_args.set:
            # Set configuration values
            for section, option, value in parsed_args.set:
                # Try to convert value to appropriate type
                config.set(section, option, config._convert_value(value))
            
            # Show updated configuration if verbose
            if parsed_args.verbose > 0:
                print("Updated configuration:")
                for section, options in config.as_dict().items():
                    print(f"[{section}]")
                    for option, value in options.items():
                        print(f"  {option} = {value}")
                    print()
    
    elif parsed_args.command == "run":
        component = parsed_args.component
        print(f"Running component: {component}")
        # This would call actual component functionality
        # For now, just a placeholder
        print("This is a placeholder for running components.")
        
    else:
        # No command specified, show help
        parse_args(["--help"])
    
    return 0


def run_main() -> None:
    """Run the main function and exit with its return code."""
    sys.exit(main())


if __name__ == "__main__":
    run_main() 