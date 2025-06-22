"""Configuration validation for LeanAgent."""

import os
from pathlib import Path
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from .config import LeanAgentConfig


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ConfigValidator:
    """Validates LeanAgent configuration."""
    
    @staticmethod
    def validate(config: 'LeanAgentConfig') -> None:
        """
        Validate the entire configuration.
        
        Args:
            config: LeanAgentConfig instance to validate
            
        Raises:
            ConfigValidationError: If validation fails
        """
        ConfigValidator._validate_paths(config)
        ConfigValidator._validate_training_config(config)
        ConfigValidator._validate_model_config(config)
        ConfigValidator._validate_github_config(config)
        ConfigValidator._validate_hardware_config(config)
    
    @staticmethod
    def _validate_paths(config: 'LeanAgentConfig') -> None:
        """Validate path configurations."""
        if not config.raid_dir:
            raise ConfigValidationError("RAID_DIR must be specified")
        
        if not config.raid_dir.exists():
            try:
                config.raid_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                raise ConfigValidationError(f"Cannot create RAID_DIR: {config.raid_dir}")
        
        # Check write permissions
        test_file = config.raid_dir / "test_write_permissions"
        try:
            test_file.write_text("test")
            test_file.unlink()
        except PermissionError:
            raise ConfigValidationError(f"No write permissions for RAID_DIR: {config.raid_dir}")
    
    @staticmethod
    def _validate_training_config(config: 'LeanAgentConfig') -> None:
        """Validate training configuration."""
        training = config.training
        
        if training.batch_size <= 0:
            raise ConfigValidationError("Batch size must be positive")
        
        if training.learning_rate <= 0:
            raise ConfigValidationError("Learning rate must be positive")
        
        if training.warmup_steps < 0:
            raise ConfigValidationError("Warmup steps cannot be negative")
        
        if training.max_seq_len <= 0:
            raise ConfigValidationError("Max sequence length must be positive")
        
        if training.precision not in ["32", "16", "bf16", "bf16-mixed"]:
            raise ConfigValidationError(f"Invalid precision: {training.precision}")
    
    @staticmethod 
    def _validate_model_config(config: 'LeanAgentConfig') -> None:
        """Validate model configuration."""
        model = config.model
        
        if model.num_retrieved <= 0:
            raise ConfigValidationError("Number of retrieved premises must be positive")
        
        if model.beam_size <= 0:
            raise ConfigValidationError("Beam size must be positive")
        
        if model.search_timeout <= 0:
            raise ConfigValidationError("Search timeout must be positive")
    
    @staticmethod
    def _validate_github_config(config: 'LeanAgentConfig') -> None:
        """Validate GitHub configuration."""
        if not config.github_token:
            raise ConfigValidationError(
                "GitHub token is required. Set GITHUB_ACCESS_TOKEN environment variable."
            )
    
    @staticmethod
    def _validate_hardware_config(config: 'LeanAgentConfig') -> None:
        """Validate hardware configuration."""
        if config.distributed_training and config.num_gpus > 1:
            if not torch.cuda.is_available():
                raise ConfigValidationError("CUDA not available for distributed training")
            
            if torch.cuda.device_count() < config.num_gpus:
                raise ConfigValidationError(
                    f"Requested {config.num_gpus} GPUs but only "
                    f"{torch.cuda.device_count()} available"
                )
        
        # Validate memory requirements
        if config.distributed_training:
            try:
                # Check if we have enough memory for the model
                torch.cuda.empty_cache()
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                if total_memory < 10:  # Minimum 10GB for the retriever model
                    raise ConfigValidationError(
                        f"Insufficient GPU memory: {total_memory:.1f}GB. "
                        "Minimum 10GB required for distributed training."
                    )
            except Exception:
                # If we can't check memory, issue a warning but don't fail
                pass 