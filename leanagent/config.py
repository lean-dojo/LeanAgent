"""
Configuration management for LeanAgent.

This module provides functionality to load, validate, and access configuration settings
from multiple sources with the following precedence (highest to lowest):
1. Command-line arguments
2. Environment variables
3. Configuration file
4. Default values
"""

import os
import yaml
import copy
from typing import Any, Dict, Optional, Union
from pathlib import Path


class Config:
    """Configuration manager for LeanAgent.
    
    Loads and provides access to configuration values from multiple sources.
    """
    
    DEFAULT_CONFIG = {
        "data": {
            "root_dir": "./data",
            "repo_dir": "./data/repos",
            "cache_dir": "./data/cache",
        },
        "retrieval": {
            "embedding_model": "text-embedding-ada-002",
            "k": 10,
            "similarity_threshold": 0.7,
        },
        "prover": {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 8192,
        },
        "db": {
            "path": "./data/db",
            "chunk_size": 1000,
        },
    }
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration with optional config file path.
        
        Args:
            config_path: Path to YAML configuration file (optional)
        """
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        
        # Load config from file if provided
        if config_path:
            self._load_from_file(config_path)
            
        # Override with environment variables
        self._load_from_env()
    
    def _load_from_file(self, config_path: Union[str, Path]) -> None:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    self._update_nested_dict(self.config, file_config)
        except (yaml.YAMLError, FileNotFoundError) as e:
            print(f"Error loading config file: {e}")
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables.
        
        Environment variables should be prefixed with LEANAGENT_ and use double underscores
        to represent nested keys, e.g., LEANAGENT_DATA__ROOT_DIR for config['data']['root_dir']
        """
        prefix = "LEANAGENT_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                env_key = key[len(prefix):]
                if "__" in env_key:
                    section, option = env_key.split("__", 1)
                    # Convert to lowercase to match config keys
                    section = section.lower()
                    option = option.lower()
                    if section in self.config:
                        # Convert value to appropriate type
                        converted_value = self._convert_value(value)
                        # Ensure section is a dictionary
                        if not isinstance(self.config[section], dict):
                            self.config[section] = {}
                        # Set the value
                        self.config[section][option] = converted_value
    
    def _update_nested_dict(self, d: Dict[str, Any], u: Dict[str, Any]) -> None:
        """Recursively update a nested dictionary with another nested dictionary.
        
        Args:
            d: Target dictionary to update
            u: Source dictionary with new values
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
    
    def _convert_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert string values to appropriate Python types.
        
        Args:
            value: String value to convert
            
        Returns:
            Converted value as appropriate type
        """
        # Try to convert to bool
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
            
        # Try to convert to number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            # Keep as string if not a number
            return value
    
    def get(self, section: str, option: str, default: Any = None) -> Any:
        """Get configuration value from specified section and option.
        
        Args:
            section: Configuration section
            option: Option name within section
            default: Default value if option doesn't exist
            
        Returns:
            Configuration value or default
        """
        return self.config.get(section, {}).get(option, default)
    
    def set(self, section: str, option: str, value: Any) -> None:
        """Set configuration value for specified section and option.
        
        Args:
            section: Configuration section
            option: Option name within section
            value: Value to set
        """
        if section not in self.config:
            self.config[section] = {}
        self.config[section][option] = value
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from a dictionary (e.g., from command line args).
        
        Args:
            config_dict: Dictionary with configuration values
        """
        self._update_nested_dict(self.config, config_dict)
    
    def as_dict(self) -> Dict[str, Dict[str, Any]]:
        """Return complete configuration as dictionary.
        
        Returns:
            Complete configuration dictionary (deep copy)
        """
        return copy.deepcopy(self.config)


# Global configuration instance
_config = None


def get_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Get or create the global configuration instance.
    
    Args:
        config_path: Path to YAML configuration file (optional)
        
    Returns:
        Global Config instance
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config 