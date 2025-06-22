"""
Progressive Training Manager for LeanAgent

This module implements the core progressive training algorithm that enables
lifelong learning without catastrophic forgetting. Key features:

1. **Progressive Training**: Trains on one epoch per repository to balance stability/plasticity
2. **EWC Integration**: Optional Elastic Weight Consolidation using Fisher Information Matrix
3. **Curriculum Support**: Integrates with curriculum learning for optimal repository ordering
4. **Checkpoint Management**: Automatic model saving and recovery
5. **Distributed Training**: Multi-GPU support with proper synchronization

Research Implementation:
- Follows the paper's progressive training methodology
- One epoch per repository to prevent overfitting
- Balances stability (retaining old knowledge) with plasticity (learning new tasks)
- Integrates with retrieval-augmented generation for theorem proving
"""

import os
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import timedelta

import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning import seed_everything

from ..config import LeanAgentConfig, TrainingConfig
from ..database import DynamicDatabase
from .fisher import FisherComputationManager
from .callbacks import LeanAgentCallbacks
from .checkpoints import CheckpointManager
from .scheduler import LearningRateScheduler

# Import retrieval components
try:
    from retrieval.model import PremiseRetriever
    from retrieval.datamodule import RetrievalDataModule
except ImportError:
    # Fallback for when retrieval module is not available
    PremiseRetriever = None
    RetrievalDataModule = None


@dataclass
class TrainingMetrics:
    """Training metrics for a single epoch/repository."""
    epoch: int
    repository_name: str
    loss_train: float
    recall_at_10: float
    training_time: float
    stability_score: float
    plasticity_score: float
    ewc_loss: float = 0.0


@dataclass
class ProgressiveTrainingResult:
    """Result of progressive training on a repository."""
    success: bool
    repository_name: str
    metrics: Optional[TrainingMetrics] = None
    checkpoint_path: Optional[Path] = None
    error_message: Optional[str] = None


class ProgressiveTrainer:
    """
    Progressive Training Manager for LeanAgent's lifelong learning system.
    
    This class implements the core training algorithm from the research paper:
    - Progressive training on one epoch per repository
    - Optional EWC (Elastic Weight Consolidation) for catastrophic forgetting prevention
    - Integration with curriculum learning and dynamic database
    - Distributed training support
    
    The trainer maintains stability (preserving old knowledge) while enabling
    plasticity (learning new knowledge) through careful training regime.
    """
    
    def __init__(self, config: LeanAgentConfig):
        """
        Initialize the Progressive Trainer.
        
        Args:
            config: LeanAgent configuration containing training parameters
        """
        self.config = config
        self.training_config = config.training
        self.logger = self._setup_logging()
        
        # Training components
        self.checkpoint_manager = CheckpointManager(config)
        self.fisher_manager = FisherComputationManager(config) if config.use_fisher_information else None
        self.callbacks = LeanAgentCallbacks(config)
        self.scheduler = LearningRateScheduler(config)
        
        # Training state
        self.current_epoch = 0
        self.training_history: List[TrainingMetrics] = []
        self.model = None
        self.device = self._setup_device()
        
        # Check dependencies for production mode
        self.production_mode = PremiseRetriever is None or RetrievalDataModule is None
        if self.production_mode:
            self.logger.warning("⚠️  Running in PRODUCTION MODE - Training components not available")
            self.logger.warning("⚠️  Only inference and database operations will be available")
        
        self.logger.info("Progressive Trainer initialized")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Fisher Information: {'Enabled' if self.fisher_manager else 'Disabled'}")
        self.logger.info(f"Production Mode: {'Enabled' if self.production_mode else 'Disabled'}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the trainer."""
        logger = logging.getLogger(f"{__name__}.ProgressiveTrainer")
        logger.setLevel(logging.INFO)
        return logger
    
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if torch.cuda.is_available() and self.config.num_gpus > 0:
            return torch.device("cuda")
        else:
            self.logger.warning("CUDA not available or num_gpus=0, using CPU (will be slow)")
            return torch.device("cpu")
    
    def initialize_model(self, checkpoint_path: Optional[str] = None) -> PremiseRetriever:
        """
        Initialize or load the premise retriever model.
        
        Args:
            checkpoint_path: Path to existing checkpoint, if None uses latest
            
        Returns:
            Initialized PremiseRetriever model
        """
        if checkpoint_path is None:
            try:
                checkpoint_path = self.checkpoint_manager.find_latest_checkpoint()
                self.logger.info(f"Found latest checkpoint: {checkpoint_path}")
            except FileNotFoundError as e:
                self.logger.error(str(e))
                # Use default mathlib4 checkpoint as fallback
                checkpoint_path = str(self.config.path_manager.checkpoint_dir / "mathlib4_29dcec074de168ac2bf835a77ef68bbe069194c5.ckpt")
        
        # Model configuration from research paper
        model_config = {
            "model_name": "kaiyuy/leandojo-lean4-retriever-byt5-small",
            "lr": self.training_config.learning_rate,
            "warmup_steps": self.training_config.warmup_steps,
            "max_seq_len": self.training_config.max_seq_len,
            "num_retrieved": 100,
        }
        
        self.model = PremiseRetriever.load(
            checkpoint_path, self.device, freeze=False, config=model_config
        )
        self.model.train()
        
        self.logger.info(f"Loaded premise retriever from {checkpoint_path}")
        return self.model
    
    def setup_fisher_information(self, repository_name: str) -> None:
        """
        Setup Fisher Information Matrix for EWC if enabled.
        
        Args:
            repository_name: Name of the repository for Fisher computation
        """
        if not self.fisher_manager or not self.model:
            return
        
        try:
            # Load previous Fisher Information Matrix
            fisher_info = self.fisher_manager.load_latest_fisher_info()
            if fisher_info:
                self.model.set_fisher_info(fisher_info)
                self.logger.info("Fisher Information Matrix loaded for EWC")
            
            # Set EWC lambda value
            lambda_value = self.training_config.ewc_lambda
            self.model.set_lambda(lambda_value)
            self.logger.info(f"EWC lambda set to {lambda_value}")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup Fisher Information: {e}")
    
    def create_trainer(self, repository_name: str) -> pl.Trainer:
        """
        Create PyTorch Lightning trainer for progressive training.
        
        Args:
            repository_name: Name of current repository for logging
            
        Returns:
            Configured PyTorch Lightning trainer
        """
        # Setup distributed training if multiple GPUs
        strategy = "auto"
        if self.config.num_gpus > 1:
            # Very long timeout for distributed training stability
            timeout = timedelta(seconds=7 * 24 * 60 * 60)  # 1 week
            strategy = DDPStrategy(timeout=timeout)
            
            # NCCL environment setup
            os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
            os.environ['NCCL_TIMEOUT'] = str(timeout.total_seconds() * 1000)
        
        # Custom log directory
        log_dir = self.config.path_manager.logs_dir / f"progressive_training_{repository_name}"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create trainer with research paper configuration
        trainer = pl.Trainer(
            accelerator="gpu" if self.device.type == "cuda" else "cpu",
            devices=self.config.num_gpus if self.device.type == "cuda" else 1,
            strategy=strategy,
            precision=self.training_config.precision,
            gradient_clip_val=self.training_config.gradient_clip_val,
            accumulate_grad_batches=self.training_config.accumulate_grad_batches,
            max_epochs=self.current_epoch + self.training_config.epochs_per_repo,
            callbacks=self.callbacks.get_callbacks(repository_name),
            log_every_n_steps=1,
            num_sanity_val_steps=0,
            default_root_dir=str(log_dir),
            enable_progress_bar=True,
            enable_model_summary=True,
        )
        
        return trainer
    
    def prepare_data_module(self, dataset_path: str) -> RetrievalDataModule:
        """
        Prepare data module for training on a repository dataset.
        
        Args:
            dataset_path: Path to the repository dataset
            
        Returns:
            Configured RetrievalDataModule
        """
        # Paths following LeanDojo format
        corpus_path = os.path.join(dataset_path, "corpus.jsonl")
        data_path = os.path.join(dataset_path, "random")
        
        self.logger.info(f"Preparing data module - Data: {data_path}, Corpus: {corpus_path}")
        
        # Create data module with research paper configuration
        data_module = RetrievalDataModule(
            data_path=data_path,
            corpus_path=corpus_path,
            num_negatives=3,  # Paper configuration
            num_in_file_negatives=1,  # Paper configuration
            model_name="google/byt5-small",  # Tokenizer model
            batch_size=self.training_config.batch_size,
            eval_batch_size=64,
            max_seq_len=1024,  # Extended for better context
            num_workers=4
        )
        
        data_module.setup(stage='fit')
        
        # Log dataset sizes
        self.logger.info(f"Training dataset size: {len(data_module.ds_train)}")
        self.logger.info(f"Validation dataset size: {len(data_module.ds_val)}")
        self.logger.info(f"Testing dataset size: {len(data_module.ds_pred)}")
        
        return data_module
    
    def train_on_repository(self, dataset_path: str, repository_name: str) -> ProgressiveTrainingResult:
        """
        Train the model on a single repository dataset.
        
        This implements the core progressive training algorithm:
        1. Load/initialize model from latest checkpoint
        2. Setup Fisher Information for EWC (if enabled)
        3. Train for exactly one epoch (paper specification)
        4. Save checkpoint and compute metrics
        5. Optionally compute Fisher Information for next iteration
        
        Args:
            dataset_path: Path to the repository dataset
            repository_name: Name of the repository being trained on
            
        Returns:
            ProgressiveTrainingResult with training outcomes
        """
        self.logger.info(f"Starting progressive training on repository: {repository_name}")
        self.logger.info(f"Dataset path: {dataset_path}")
        
        try:
            # 1. Initialize model
            if self.model is None:
                self.initialize_model()
            
            # 2. Setup Fisher Information for EWC
            self.setup_fisher_information(repository_name)
            
            # 3. Prepare data
            data_module = self.prepare_data_module(dataset_path)
            
            # 4. Create trainer
            trainer = self.create_trainer(repository_name)
            
            # 5. Set random seed for reproducibility
            seed_everything(3407)  # Paper's seed
            
            # 6. Progressive training (one epoch)
            self.logger.info(f"Starting training from epoch {self.current_epoch} to {self.current_epoch + self.training_config.epochs_per_repo}")
            
            start_time = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
            end_time = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
            
            if start_time:
                start_time.record()
            
            # Synchronization barriers for distributed training
            if hasattr(trainer.strategy, 'barrier'):
                self.logger.info("Synchronizing before training")
                trainer.strategy.barrier()
            
            # Main training step
            trainer.fit(self.model, datamodule=data_module)
            
            if hasattr(trainer.strategy, 'barrier'):
                self.logger.info("Synchronizing after training")
                trainer.strategy.barrier()
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                training_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
            else:
                training_time = 0.0
            
            # 7. Extract training metrics
            metrics = self._extract_training_metrics(
                trainer, repository_name, training_time
            )
            
            # 8. Update training state
            self.current_epoch = trainer.current_epoch
            self.training_history.append(metrics)
            
            # 9. Save checkpoint
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                self.model, repository_name, metrics.recall_at_10
            )
            
            # 10. Compute Fisher Information for next iteration (if enabled)
            if self.fisher_manager:
                try:
                    self.fisher_manager.compute_fisher_information(
                        self.model, data_module, repository_name
                    )
                except Exception as e:
                    self.logger.warning(f"Fisher computation failed: {e}")
            
            self.logger.info(f"Progressive training completed successfully on {repository_name}")
            self.logger.info(f"Final metrics - Loss: {metrics.loss_train:.4f}, Recall@10: {metrics.recall_at_10:.4f}")
            
            return ProgressiveTrainingResult(
                success=True,
                repository_name=repository_name,
                metrics=metrics,
                checkpoint_path=checkpoint_path
            )
            
        except Exception as e:
            error_msg = f"Progressive training failed on {repository_name}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
            return ProgressiveTrainingResult(
                success=False,
                repository_name=repository_name,
                error_message=error_msg
            )
    
    def _extract_training_metrics(self, trainer: pl.Trainer, repository_name: str, training_time: float) -> TrainingMetrics:
        """
        Extract training metrics from completed training.
        
        Args:
            trainer: Completed PyTorch Lightning trainer
            repository_name: Name of the repository
            training_time: Training time in seconds
            
        Returns:
            TrainingMetrics with performance data
        """
        # Get metrics from trainer
        callback_metrics = trainer.callback_metrics
        
        # Extract key metrics with fallbacks
        loss_train = float(callback_metrics.get('loss_train', 0.0))
        recall_at_10 = float(callback_metrics.get('Recall@10_val', 0.0))
        
        # Calculate stability and plasticity scores
        stability_score = self._calculate_stability_score()
        plasticity_score = recall_at_10  # Current performance as plasticity measure
        
        # EWC loss if available
        ewc_loss = 0.0
        if self.model and hasattr(self.model, 'ewc_loss'):
            try:
                ewc_loss = float(self.model.ewc_loss())
            except:
                ewc_loss = 0.0
        
        return TrainingMetrics(
            epoch=trainer.current_epoch,
            repository_name=repository_name,
            loss_train=loss_train,
            recall_at_10=recall_at_10,
            training_time=training_time,
            stability_score=stability_score,
            plasticity_score=plasticity_score,
            ewc_loss=ewc_loss
        )
    
    def _calculate_stability_score(self) -> float:
        """
        Calculate stability score based on training history.
        
        Stability measures how well the model retains performance on previous tasks.
        
        Returns:
            Stability score (higher is better)
        """
        if len(self.training_history) < 2:
            return 1.0  # Perfect stability for first repository
        
        # Calculate average performance drop from peak
        stability_scores = []
        for i, metrics in enumerate(self.training_history[:-1]):  # Exclude current
            peak_recall = metrics.recall_at_10
            current_recall = self.training_history[-1].recall_at_10
            
            # Stability is maintained performance relative to peak
            if peak_recall > 0:
                stability = min(1.0, current_recall / peak_recall)
                stability_scores.append(stability)
        
        return sum(stability_scores) / len(stability_scores) if stability_scores else 1.0
    
    def train_curriculum(self, curriculum: List[Tuple[str, str]], database: DynamicDatabase) -> Dict[str, Any]:
        """
        Train on a complete curriculum of repositories.
        
        This implements the full lifelong learning pipeline:
        1. Process repositories in curriculum order
        2. Generate datasets from dynamic database
        3. Progressive training on each repository
        4. Accumulate knowledge without forgetting
        
        Args:
            curriculum: List of (repository_name, repository_url) tuples in curriculum order
            database: Dynamic database containing repository data
            
        Returns:
            Dictionary with comprehensive training results
        """
        self.logger.info(f"Starting curriculum training on {len(curriculum)} repositories")
        
        results = {
            "total_repositories": len(curriculum),
            "successful_trainings": 0,
            "failed_trainings": 0,
            "training_results": [],
            "final_metrics": {},
            "total_time": 0.0
        }
        
        start_time = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
        end_time = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
        
        if start_time:
            start_time.record()
        
        for i, (repo_name, repo_url) in enumerate(curriculum):
            self.logger.info(f"Processing repository {i+1}/{len(curriculum)}: {repo_name}")
            
            try:
                # Generate dataset for this repository
                dataset_path = self._generate_repository_dataset(repo_name, repo_url, database)
                
                # Progressive training on this repository
                training_result = self.train_on_repository(dataset_path, repo_name)
                results["training_results"].append(training_result)
                
                if training_result.success:
                    results["successful_trainings"] += 1
                    self.logger.info(f"Successfully trained on {repo_name}")
                else:
                    results["failed_trainings"] += 1
                    self.logger.error(f"Failed to train on {repo_name}: {training_result.error_message}")
                
            except Exception as e:
                self.logger.error(f"Error processing repository {repo_name}: {e}")
                results["failed_trainings"] += 1
                results["training_results"].append(
                    ProgressiveTrainingResult(
                        success=False,
                        repository_name=repo_name,
                        error_message=str(e)
                    )
                )
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            results["total_time"] = start_time.elapsed_time(end_time) / 1000.0
        
        # Calculate final metrics
        results["final_metrics"] = self._calculate_final_metrics()
        
        self.logger.info(f"Curriculum training completed")
        self.logger.info(f"Success rate: {results['successful_trainings']}/{results['total_repositories']}")
        
        return results
    
    def _generate_repository_dataset(self, repo_name: str, repo_url: str, database: DynamicDatabase) -> str:
        """
        Generate dataset for a repository using the dynamic database.
        
        Args:
            repo_name: Name of the repository
            repo_url: URL of the repository  
            database: Dynamic database instance
            
        Returns:
            Path to generated dataset
        """
        # Create dataset directory
        dataset_dir = self.config.path_manager.data_dir / f"merged_with_new_{repo_name}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Use database to generate merged dataset
        database.generate_merged_dataset(str(dataset_dir), [repo_name])
        
        self.logger.info(f"Generated dataset for {repo_name} at {dataset_dir}")
        return str(dataset_dir)
    
    def _calculate_final_metrics(self) -> Dict[str, float]:
        """
        Calculate final performance metrics across all training.
        
        Returns:
            Dictionary of final metrics
        """
        if not self.training_history:
            return {}
        
        # Average metrics across all repositories
        avg_recall = sum(m.recall_at_10 for m in self.training_history) / len(self.training_history)
        avg_loss = sum(m.loss_train for m in self.training_history) / len(self.training_history)
        avg_stability = sum(m.stability_score for m in self.training_history) / len(self.training_history)
        avg_plasticity = sum(m.plasticity_score for m in self.training_history) / len(self.training_history)
        total_training_time = sum(m.training_time for m in self.training_history)
        
        # Final performance on latest repository
        final_recall = self.training_history[-1].recall_at_10
        final_loss = self.training_history[-1].loss_train
        
        return {
            "average_recall_at_10": avg_recall,
            "average_loss": avg_loss,
            "average_stability": avg_stability,
            "average_plasticity": avg_plasticity,
            "final_recall_at_10": final_recall,
            "final_loss": final_loss,
            "total_training_time": total_training_time,
            "total_epochs": self.current_epoch
        }
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive training summary.
        
        Returns:
            Dictionary with training history and statistics
        """
        return {
            "current_epoch": self.current_epoch,
            "total_repositories_trained": len(self.training_history),
            "training_history": [
                {
                    "repository": m.repository_name,
                    "epoch": m.epoch,
                    "recall_at_10": m.recall_at_10,
                    "loss": m.loss_train,
                    "stability": m.stability_score,
                    "plasticity": m.plasticity_score,
                    "training_time": m.training_time
                }
                for m in self.training_history
            ],
            "final_metrics": self._calculate_final_metrics(),
            "configuration": {
                "epochs_per_repo": self.training_config.epochs_per_repo,
                "learning_rate": self.training_config.learning_rate,
                "batch_size": self.training_config.batch_size,
                "use_fisher": self.fisher_manager is not None,
                "device": str(self.device),
                "num_gpus": self.config.num_gpus
            }
        } 