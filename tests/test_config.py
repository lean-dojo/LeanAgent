"""
Tests for LeanAgent configuration system.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

from src.config import LeanAgentConfig, ConfigValidator, PathManager
from src.config.validation import ConfigValidationError


class TestLeanAgentConfig:
    """Tests for the main configuration class."""

    def test_config_creation_with_defaults(self):
        """Test configuration creation with default values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["RAID_DIR"] = temp_dir
            os.environ["GITHUB_ACCESS_TOKEN"] = "test_token"
            
            config = LeanAgentConfig()
            
            assert config.raid_dir == Path(temp_dir)
            assert config.training.batch_size == 4
            assert config.model.num_retrieved == 100
            assert config.curriculum.enabled is True
            assert config.progressive_training is True

    def test_config_from_env(self):
        """Test configuration creation from environment variables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set environment variables
            env_vars = {
                "RAID_DIR": temp_dir,
                "GITHUB_ACCESS_TOKEN": "test_token",
                "LEANAGENT_BATCH_SIZE": "8",
                "LEANAGENT_LEARNING_RATE": "0.001",
                "LEANAGENT_NUM_REPOS": "20"
            }
            
            with patch.dict(os.environ, env_vars):
                config = LeanAgentConfig.from_env()
                
                assert config.training.batch_size == 8
                assert config.training.learning_rate == 0.001
                assert config.curriculum.num_repos == 20

    def test_config_to_dict(self):
        """Test configuration serialization to dictionary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["RAID_DIR"] = temp_dir
            os.environ["GITHUB_ACCESS_TOKEN"] = "test_token"
            
            config = LeanAgentConfig()
            config_dict = config.to_dict()
            
            assert "training" in config_dict
            assert "model" in config_dict
            assert "curriculum" in config_dict
            assert "database" in config_dict
            assert config_dict["progressive_training"] is True

    def test_database_path_property(self):
        """Test database path property."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["RAID_DIR"] = temp_dir
            os.environ["GITHUB_ACCESS_TOKEN"] = "test_token"
            
            config = LeanAgentConfig()
            expected_path = Path(temp_dir) / "data" / "leanagent_database.json"
            
            assert config.database_path == expected_path

    def test_directory_creation(self):
        """Test that directories are created during initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["RAID_DIR"] = temp_dir
            os.environ["GITHUB_ACCESS_TOKEN"] = "test_token"
            
            config = LeanAgentConfig()
            
            # Check that directories were created
            assert config.path_manager.data_dir.exists()
            assert config.path_manager.checkpoint_dir.exists()
            assert config.path_manager.repos_dir.exists()
            assert config.path_manager.logs_dir.exists()


class TestConfigValidator:
    """Tests for configuration validation."""

    def test_validate_successful(self):
        """Test successful validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["RAID_DIR"] = temp_dir
            os.environ["GITHUB_ACCESS_TOKEN"] = "test_token"
            
            config = LeanAgentConfig()
            # Should not raise any exception
            ConfigValidator.validate(config)

    def test_validate_missing_github_token(self):
        """Test validation failure with missing GitHub token."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["RAID_DIR"] = temp_dir
            if "GITHUB_ACCESS_TOKEN" in os.environ:
                del os.environ["GITHUB_ACCESS_TOKEN"]
            
            with pytest.raises(ConfigValidationError, match="GitHub token is required"):
                config = LeanAgentConfig()

    def test_validate_invalid_batch_size(self):
        """Test validation failure with invalid batch size."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["RAID_DIR"] = temp_dir
            os.environ["GITHUB_ACCESS_TOKEN"] = "test_token"
            
            config = LeanAgentConfig()
            config.training.batch_size = -1
            
            with pytest.raises(ConfigValidationError, match="Batch size must be positive"):
                ConfigValidator.validate(config)

    def test_validate_invalid_learning_rate(self):
        """Test validation failure with invalid learning rate."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["RAID_DIR"] = temp_dir
            os.environ["GITHUB_ACCESS_TOKEN"] = "test_token"
            
            config = LeanAgentConfig()
            config.training.learning_rate = 0
            
            with pytest.raises(ConfigValidationError, match="Learning rate must be positive"):
                ConfigValidator.validate(config)

    def test_validate_invalid_precision(self):
        """Test validation failure with invalid precision."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["RAID_DIR"] = temp_dir
            os.environ["GITHUB_ACCESS_TOKEN"] = "test_token"
            
            config = LeanAgentConfig()
            config.training.precision = "invalid"
            
            with pytest.raises(ConfigValidationError, match="Invalid precision"):
                ConfigValidator.validate(config)

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_validate_hardware_insufficient_gpus(self, mock_device_count, mock_cuda_available):
        """Test validation failure with insufficient GPUs."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 2
        
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["RAID_DIR"] = temp_dir
            os.environ["GITHUB_ACCESS_TOKEN"] = "test_token"
            
            config = LeanAgentConfig()
            config.distributed_training = True
            config.num_gpus = 4  # More than available
            
            with pytest.raises(ConfigValidationError, match="Requested 4 GPUs but only 2 available"):
                ConfigValidator.validate(config)


class TestPathManager:
    """Tests for path management."""

    def test_path_manager_initialization(self):
        """Test PathManager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path_manager = PathManager(Path(temp_dir))
            
            assert path_manager.base_dir == Path(temp_dir)
            assert path_manager.data_dir == Path(temp_dir) / "data"
            assert path_manager.checkpoint_dir == Path(temp_dir) / "checkpoints"

    def test_get_repo_dir(self):
        """Test repository directory path generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path_manager = PathManager(Path(temp_dir))
            
            repo_dir = path_manager.get_repo_dir("user/repo", "abc123def456")
            expected = Path(temp_dir) / "repos_new" / "user_repo_abc123de"
            
            assert repo_dir == expected

    def test_get_checkpoint_path(self):
        """Test checkpoint path generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path_manager = PathManager(Path(temp_dir))
            
            checkpoint_path = path_manager.get_checkpoint_path("model", 10, 0.95)
            expected = Path(temp_dir) / "checkpoints" / "model_epoch_10_metric_0.950.ckpt"
            
            assert checkpoint_path == expected

    def test_ensure_directories_exist(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path_manager = PathManager(Path(temp_dir))
            path_manager.ensure_directories_exist()
            
            # Check that all directories were created
            assert path_manager.data_dir.exists()
            assert path_manager.checkpoint_dir.exists()
            assert path_manager.repos_dir.exists()
            assert path_manager.logs_dir.exists()
            assert path_manager.tmp_dir.exists()
            assert path_manager.fisher_dir.exists() 