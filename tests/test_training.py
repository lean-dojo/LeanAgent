"""
Tests for Training Module

Comprehensive tests for the progressive training pipeline including:
- Progressive training manager
- Fisher Information computation
- Checkpoint management
- Learning rate scheduling
- Training callbacks
"""

import pytest
import tempfile
import torch
import pytorch_lightning as pl
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.config import LeanAgentConfig, TrainingConfig
from src.training import (
    ProgressiveTrainer,
    FisherComputationManager,
    CheckpointManager,
    LearningRateScheduler,
    LeanAgentCallbacks
)
from src.training.progressive import TrainingMetrics, ProgressiveTrainingResult
from src.training.checkpoints import CheckpointMetadata
from src.training.scheduler import WarmupCosineScheduler, AdaptiveWarmupScheduler


class MockPremiseRetriever(pl.LightningModule):
    """Mock premise retriever for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
        self.trainer = None
        self.fisher_info = {}
        self.lamda = 0.0
        
    def forward(self, *args, **kwargs):
        return torch.tensor(0.5)
    
    def set_fisher_info(self, fisher_info):
        self.fisher_info = fisher_info
        
    def set_lambda(self, lambda_value):
        self.lamda = lambda_value
        
    def ewc_loss(self):
        return torch.tensor(0.1 * self.lamda)
    
    @classmethod
    def load(cls, checkpoint_path, device, freeze=False, config=None):
        return cls()


class MockRetrievalDataModule(pl.LightningDataModule):
    """Mock data module for testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.ds_train = Mock()
        self.ds_val = Mock()
        self.ds_pred = Mock()
        
        # Mock dataset sizes
        self.ds_train.__len__ = Mock(return_value=100)
        self.ds_val.__len__ = Mock(return_value=20)
        self.ds_pred.__len__ = Mock(return_value=20)
    
    def setup(self, stage=None):
        pass


@pytest.fixture
def test_config():
    """Create test configuration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = LeanAgentConfig()
        config.raid_dir = Path(temp_dir)
        config.training = TrainingConfig(
            batch_size=2,
            learning_rate=1e-3,
            warmup_steps=10,
            epochs_per_repo=1,
            ewc_lambda=0.1
        )
        config.num_gpus = 0  # CPU only for tests
        config.use_fisher_information = True
        config.__post_init__()  # Setup paths
        yield config


@pytest.fixture
def mock_database():
    """Create mock database."""
    database = Mock()
    database.repositories = []
    database.generate_merged_dataset = Mock()
    return database


class TestProgressiveTrainer:
    """Test the ProgressiveTrainer class."""
    
    @patch('src.training.progressive.PremiseRetriever', MockPremiseRetriever)
    @patch('src.training.progressive.RetrievalDataModule', MockRetrievalDataModule)
    def test_initialization(self, test_config):
        """Test progressive trainer initialization."""
        trainer = ProgressiveTrainer(test_config)
        
        assert trainer.config == test_config
        assert trainer.training_config == test_config.training
        assert trainer.current_epoch == 0
        assert trainer.training_history == []
        assert trainer.device.type == "cpu"  # No GPU in test
        assert trainer.checkpoint_manager is not None
        assert trainer.fisher_manager is not None  # Fisher enabled in config
    
    @patch('src.training.progressive.PremiseRetriever', MockPremiseRetriever)
    def test_initialize_model(self, test_config):
        """Test model initialization."""
        trainer = ProgressiveTrainer(test_config)
        
        # Mock checkpoint manager
        trainer.checkpoint_manager.find_latest_checkpoint = Mock(
            side_effect=FileNotFoundError("No checkpoint found")
        )
        
        model = trainer.initialize_model()
        
        assert isinstance(model, MockPremiseRetriever)
        assert trainer.model == model
    
    @patch('src.training.progressive.PremiseRetriever', MockPremiseRetriever)
    @patch('src.training.progressive.RetrievalDataModule', MockRetrievalDataModule)
    def test_prepare_data_module(self, test_config):
        """Test data module preparation."""
        trainer = ProgressiveTrainer(test_config)
        
        # Create mock dataset directory
        dataset_path = str(test_config.path_manager.data_dir / "test_dataset")
        Path(dataset_path).mkdir(parents=True, exist_ok=True)
        
        # Create mock files
        (Path(dataset_path) / "corpus.jsonl").touch()
        (Path(dataset_path) / "random").mkdir(exist_ok=True)
        
        data_module = trainer.prepare_data_module(dataset_path)
        
        assert isinstance(data_module, MockRetrievalDataModule)
    
    @patch('src.training.progressive.PremiseRetriever', MockPremiseRetriever)
    @patch('src.training.progressive.RetrievalDataModule', MockRetrievalDataModule)
    def test_train_on_repository(self, test_config, mock_database):
        """Test training on a single repository."""
        trainer = ProgressiveTrainer(test_config)
        
        # Create mock dataset directory
        dataset_path = str(test_config.path_manager.data_dir / "test_dataset")
        Path(dataset_path).mkdir(parents=True, exist_ok=True)
        (Path(dataset_path) / "corpus.jsonl").touch()
        (Path(dataset_path) / "random").mkdir(exist_ok=True)
        
        # Mock trainer components
        with patch('pytorch_lightning.Trainer') as mock_pl_trainer:
            mock_trainer_instance = Mock()
            mock_trainer_instance.fit = Mock()
            mock_trainer_instance.current_epoch = 1
            mock_trainer_instance.callback_metrics = {
                'loss_train': torch.tensor(0.5),
                'Recall@10_val': torch.tensor(0.75)
            }
            mock_pl_trainer.return_value = mock_trainer_instance
            
            # Mock checkpoint saving
            trainer.checkpoint_manager.save_checkpoint = Mock(
                return_value=Path("/mock/checkpoint.ckpt")
            )
            
            result = trainer.train_on_repository(dataset_path, "test_repo")
            
            assert isinstance(result, ProgressiveTrainingResult)
            assert result.success == True
            assert result.repository_name == "test_repo"
            assert result.metrics is not None
            assert result.checkpoint_path is not None
    
    def test_calculate_stability_score(self, test_config):
        """Test stability score calculation."""
        trainer = ProgressiveTrainer(test_config)
        
        # No history - should return 1.0
        stability = trainer._calculate_stability_score()
        assert stability == 1.0
        
        # Add some training history
        trainer.training_history = [
            TrainingMetrics(0, "repo1", 0.5, 0.8, 100.0, 1.0, 0.8),
            TrainingMetrics(1, "repo2", 0.4, 0.7, 120.0, 1.0, 0.7),
            TrainingMetrics(2, "repo3", 0.3, 0.9, 110.0, 1.0, 0.9)  # Current
        ]
        
        stability = trainer._calculate_stability_score()
        assert 0.0 <= stability <= 1.0


class TestFisherComputationManager:
    """Test the FisherComputationManager class."""
    
    def test_initialization(self, test_config):
        """Test Fisher computation manager initialization."""
        manager = FisherComputationManager(test_config)
        
        assert manager.config == test_config
        assert manager.fisher_dir.exists()
        assert manager.device.type == "cpu"
    
    @patch('src.training.fisher.FisherComputationModule')
    @patch('pytorch_lightning.Trainer')
    def test_compute_fisher_information(self, mock_trainer, mock_fisher_module, test_config):
        """Test Fisher Information computation."""
        manager = FisherComputationManager(test_config)
        
        # Mock components
        mock_model = Mock()
        mock_data_module = Mock()
        mock_trainer_instance = Mock()
        mock_trainer.return_value = mock_trainer_instance
        
        # Mock Fisher module
        mock_fisher_instance = Mock()
        mock_fisher_module.return_value = mock_fisher_instance
        
        result_path = manager.compute_fisher_information(
            mock_model, mock_data_module, "test_repo"
        )
        
        assert isinstance(result_path, str)
        assert "test_repo" in result_path
        mock_trainer_instance.fit.assert_called_once()
    
    def test_find_latest_fisher_info(self, test_config):
        """Test finding latest Fisher Information file."""
        manager = FisherComputationManager(test_config)
        
        # No files - should return None
        result = manager.find_latest_fisher_info()
        assert result is None
        
        # Create mock Fisher files
        fisher_file = manager.fisher_dir / "fisher_info_test.pkl"
        fisher_file.touch()
        
        result = manager.find_latest_fisher_info()
        assert result == str(fisher_file)


class TestCheckpointManager:
    """Test the CheckpointManager class."""
    
    def test_initialization(self, test_config):
        """Test checkpoint manager initialization."""
        manager = CheckpointManager(test_config)
        
        assert manager.config == test_config
        assert manager.checkpoint_dir.exists()
        assert manager.checkpoint_history == []
    
    def test_save_checkpoint(self, test_config):
        """Test checkpoint saving."""
        manager = CheckpointManager(test_config)
        
        # Create mock model
        mock_model = MockPremiseRetriever()
        mock_model.trainer = Mock()
        mock_model.trainer.current_epoch = 1
        mock_model.trainer.global_step = 100
        mock_model.trainer.logged_metrics = {'loss_train': torch.tensor(0.5)}
        mock_model.trainer.save_checkpoint = Mock()
        
        checkpoint_path = manager.save_checkpoint(
            mock_model, "test_repo", 0.75
        )
        
        assert checkpoint_path.exists()
        assert len(manager.checkpoint_history) == 1
        
        metadata = manager.checkpoint_history[0]
        assert metadata.repository_name == "test_repo"
        assert metadata.recall_at_10 == 0.75
    
    def test_find_latest_checkpoint(self, test_config):
        """Test finding latest checkpoint."""
        manager = CheckpointManager(test_config)
        
        # No checkpoints - should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            manager.find_latest_checkpoint()
        
        # Create mock checkpoint
        checkpoint_file = manager.checkpoint_dir / "test.ckpt"
        checkpoint_file.touch()
        
        result = manager.find_latest_checkpoint()
        assert result == str(checkpoint_file)
    
    def test_cleanup_old_checkpoints(self, test_config):
        """Test checkpoint cleanup."""
        manager = CheckpointManager(test_config)
        
        # Create multiple mock checkpoints
        for i in range(15):
            checkpoint_file = manager.checkpoint_dir / f"checkpoint_{i}.ckpt"
            checkpoint_file.touch()
            
            metadata = CheckpointMetadata(
                repository_name=f"repo_{i}",
                epoch=i,
                global_step=i * 100,
                recall_at_10=0.5 + i * 0.01,
                loss_train=0.5 - i * 0.01,
                timestamp=datetime.now().isoformat(),
                model_config={},
                training_time=100.0,
                checkpoint_path=str(checkpoint_file),
                file_size_mb=10.0
            )
            manager.checkpoint_history.append(metadata)
        
        # Cleanup keeping last 5 and best 3
        manager.cleanup_old_checkpoints(keep_last_n=5, keep_best_n=3)
        
        # Should keep at most 8 checkpoints (5 latest + 3 best, with possible overlap)
        assert len(manager.checkpoint_history) <= 8


class TestLearningRateScheduler:
    """Test the LearningRateScheduler class."""
    
    def test_warmup_cosine_scheduler(self, test_config):
        """Test warmup cosine scheduler."""
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(10))], lr=1e-3)
        scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_steps=10,
            total_steps=100,
            max_lr=1e-3,
            min_lr=0.0
        )
        
        # Test warmup phase
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        assert 0 < lr < 1e-3
        
        # Test after warmup
        for _ in range(15):  # Past warmup
            scheduler.step()
        
        lr = scheduler.get_last_lr()[0]
        assert lr <= 1e-3
    
    def test_adaptive_scheduler(self, test_config):
        """Test adaptive warmup scheduler."""
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(10))], lr=1e-3)
        scheduler = AdaptiveWarmupScheduler(
            optimizer=optimizer,
            base_lr=1e-3,
            warmup_steps=10
        )
        
        # Test first repository
        initial_lr = scheduler.get_lr()[0]
        
        # Start new repository
        scheduler.new_repository()
        assert scheduler.repository_count == 1
        
        # Start another repository
        scheduler.new_repository()
        assert scheduler.repository_count == 2
        
        # Learning rate should be adapted
        new_lr = scheduler.get_lr()[0]
        assert new_lr <= initial_lr  # Should be reduced for later repositories
    
    def test_scheduler_manager(self, test_config):
        """Test learning rate scheduler manager."""
        manager = LearningRateScheduler(test_config)
        
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(10))], lr=1e-3)
        scheduler = manager.create_scheduler(optimizer, 100, "test_repo")
        
        assert scheduler is not None
        assert manager.current_scheduler == scheduler
        
        # Test PyTorch Lightning configuration
        config = manager.get_pytorch_lightning_config(optimizer, 100, "test_repo")
        assert "scheduler" in config
        assert "interval" in config
        assert config["interval"] == "step"


class TestLeanAgentCallbacks:
    """Test the LeanAgentCallbacks class."""
    
    def test_initialization(self, test_config):
        """Test callbacks manager initialization."""
        callbacks = LeanAgentCallbacks(test_config)
        
        assert callbacks.config == test_config
        assert callbacks.lifelong_monitor is not None
        assert callbacks.stability_monitor is not None
        assert callbacks.ewc_monitor is not None  # Fisher enabled in config
    
    def test_get_callbacks(self, test_config):
        """Test callback creation for repository."""
        callbacks = LeanAgentCallbacks(test_config)
        
        callback_list = callbacks.get_callbacks("test_repo")
        
        assert len(callback_list) >= 5  # Should have multiple callbacks
        
        # Check for required callback types
        callback_types = [type(cb).__name__ for cb in callback_list]
        assert "ModelCheckpoint" in callback_types
        assert "EarlyStopping" in callback_types
        assert "LearningRateMonitor" in callback_types
    
    def test_training_summary(self, test_config):
        """Test training summary generation."""
        callbacks = LeanAgentCallbacks(test_config)
        
        summary = callbacks.get_training_summary()
        
        assert "lifelong_learning_metrics" in summary
        assert "stability_plasticity_metrics" in summary
        assert "ewc_metrics" in summary  # Fisher enabled
    
    def test_reset_for_new_curriculum(self, test_config):
        """Test resetting callback state."""
        callbacks = LeanAgentCallbacks(test_config)
        
        # Add some mock data
        callbacks.lifelong_monitor.training_metrics.append({"test": "data"})
        callbacks.stability_monitor.repository_performance["test"] = [0.5, 0.6]
        
        # Reset
        callbacks.reset_for_new_curriculum()
        
        assert len(callbacks.lifelong_monitor.training_metrics) == 0
        assert len(callbacks.stability_monitor.repository_performance) == 0


class TestTrainingIntegration:
    """Integration tests for training components."""
    
    @patch('src.training.progressive.PremiseRetriever', MockPremiseRetriever)
    @patch('src.training.progressive.RetrievalDataModule', MockRetrievalDataModule)
    def test_end_to_end_training_workflow(self, test_config, mock_database):
        """Test complete training workflow."""
        # Initialize trainer
        trainer = ProgressiveTrainer(test_config)
        
        # Create mock curriculum
        curriculum = [
            ("repo1", "https://github.com/test/repo1"),
            ("repo2", "https://github.com/test/repo2")
        ]
        
        # Create mock dataset directories
        for repo_name, _ in curriculum:
            dataset_path = test_config.path_manager.data_dir / f"merged_with_new_{repo_name}"
            dataset_path.mkdir(parents=True, exist_ok=True)
            (dataset_path / "corpus.jsonl").touch()
            (dataset_path / "random").mkdir(exist_ok=True)
        
        # Mock PyTorch Lightning trainer
        with patch('pytorch_lightning.Trainer') as mock_pl_trainer:
            mock_trainer_instance = Mock()
            mock_trainer_instance.fit = Mock()
            mock_trainer_instance.current_epoch = 1
            mock_trainer_instance.callback_metrics = {
                'loss_train': torch.tensor(0.5),
                'Recall@10_val': torch.tensor(0.75)
            }
            mock_pl_trainer.return_value = mock_trainer_instance
            
            # Mock checkpoint saving
            trainer.checkpoint_manager.save_checkpoint = Mock(
                return_value=Path("/mock/checkpoint.ckpt")
            )
            
            # Run training
            results = trainer.train_curriculum(curriculum, mock_database)
            
            assert results["total_repositories"] == 2
            assert results["successful_trainings"] >= 0
            assert "final_metrics" in results
    
    def test_training_metrics_collection(self, test_config):
        """Test training metrics collection and analysis."""
        trainer = ProgressiveTrainer(test_config)
        
        # Add mock training history
        trainer.training_history = [
            TrainingMetrics(0, "repo1", 0.8, 0.75, 100.0, 1.0, 0.75),
            TrainingMetrics(1, "repo2", 0.6, 0.80, 120.0, 0.9, 0.80),
            TrainingMetrics(2, "repo3", 0.4, 0.85, 110.0, 0.95, 0.85)
        ]
        
        # Get training summary
        summary = trainer.get_training_summary()
        
        assert summary["total_repositories_trained"] == 3
        assert summary["current_epoch"] == 2
        assert "final_metrics" in summary
        assert "training_history" in summary
        
        # Check final metrics calculation
        final_metrics = summary["final_metrics"]
        assert "average_recall_at_10" in final_metrics
        assert "final_recall_at_10" in final_metrics
        assert final_metrics["final_recall_at_10"] == 0.85  # Last repository


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__]) 