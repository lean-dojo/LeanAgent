"""
Training Callbacks for LeanAgent

This module provides specialized PyTorch Lightning callbacks for the
progressive training pipeline, including:

1. **Model Checkpointing**: Save best models based on validation metrics
2. **Early Stopping**: Prevent overfitting during training
3. **Learning Rate Monitoring**: Track learning rate changes
4. **Custom Progress Tracking**: Monitor lifelong learning metrics
5. **Stability/Plasticity Monitoring**: Track the stability-plasticity tradeoff

Research Implementation:
- Monitors Recall@10 as primary metric (paper specification)
- Implements custom callbacks for lifelong learning metrics
- Provides comprehensive logging for analysis
- Integrates with distributed training
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    LearningRateMonitor,
    Callback
)

from ..config import LeanAgentConfig


class LifelongLearningMonitor(Callback):
    """
    Custom callback to monitor lifelong learning metrics.
    
    Tracks stability, plasticity, and other metrics specific to
    lifelong learning in theorem proving.
    """
    
    def __init__(self, config: LeanAgentConfig):
        """
        Initialize lifelong learning monitor.
        
        Args:
            config: LeanAgent configuration
        """
        super().__init__()
        self.config = config
        self.logger = self._setup_logging()
        
        # Metrics storage
        self.training_metrics: List[Dict[str, Any]] = []
        self.validation_metrics: List[Dict[str, Any]] = []
        
        # Current repository tracking
        self.current_repository = "unknown"
        self.training_start_time = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the monitor."""
        logger = logging.getLogger(f"{__name__}.LifelongLearningMonitor")
        logger.setLevel(logging.INFO)
        return logger
    
    def set_current_repository(self, repository_name: str):
        """
        Set the current repository being trained on.
        
        Args:
            repository_name: Name of the current repository
        """
        self.current_repository = repository_name
        self.logger.info(f"Monitoring lifelong learning for repository: {repository_name}")
    
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called when training starts."""
        self.training_start_time = datetime.now()
        self.logger.info(f"Training started for {self.current_repository}")
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called at the end of each training epoch."""
        # Collect training metrics
        metrics = {
            'repository': self.current_repository,
            'epoch': trainer.current_epoch,
            'global_step': trainer.global_step,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Add logged metrics
        for key, value in trainer.logged_metrics.items():
            if 'train' in key.lower():
                metrics[key] = float(value)
        
        self.training_metrics.append(metrics)
        
        # Log key metrics
        if 'loss_train' in metrics:
            self.logger.info(f"Epoch {trainer.current_epoch} - Training Loss: {metrics['loss_train']:.4f}")
    
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called at the end of validation."""
        # Collect validation metrics
        metrics = {
            'repository': self.current_repository,
            'epoch': trainer.current_epoch,
            'global_step': trainer.global_step,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Add logged metrics
        for key, value in trainer.logged_metrics.items():
            if 'val' in key.lower() or key in ['Recall@10_val', 'MRR', 'Recall@1']:
                metrics[key] = float(value)
        
        self.validation_metrics.append(metrics)
        
        # Log key metrics
        if 'Recall@10_val' in metrics:
            self.logger.info(f"Epoch {trainer.current_epoch} - Recall@10: {metrics['Recall@10_val']:.4f}")
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called when training ends."""
        if self.training_start_time:
            training_time = (datetime.now() - self.training_start_time).total_seconds()
            self.logger.info(f"Training completed for {self.current_repository} in {training_time:.2f} seconds")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of collected metrics.
        
        Returns:
            Dictionary with metrics summary
        """
        return {
            'current_repository': self.current_repository,
            'total_training_steps': len(self.training_metrics),
            'total_validation_steps': len(self.validation_metrics),
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics
        }


class StabilityPlasticityMonitor(Callback):
    """
    Monitor the stability-plasticity tradeoff during training.
    
    This callback tracks how well the model retains old knowledge (stability)
    while learning new knowledge (plasticity) - key for lifelong learning.
    """
    
    def __init__(self, config: LeanAgentConfig):
        """
        Initialize stability-plasticity monitor.
        
        Args:
            config: LeanAgent configuration
        """
        super().__init__()
        self.config = config
        self.logger = self._setup_logging()
        
        # Historical performance tracking
        self.repository_performance: Dict[str, List[float]] = {}
        self.current_repository = "unknown"
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the monitor."""
        logger = logging.getLogger(f"{__name__}.StabilityPlasticityMonitor")
        logger.setLevel(logging.INFO)
        return logger
    
    def set_current_repository(self, repository_name: str):
        """
        Set the current repository being trained on.
        
        Args:
            repository_name: Name of the current repository
        """
        self.current_repository = repository_name
        if repository_name not in self.repository_performance:
            self.repository_performance[repository_name] = []
    
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Track performance for stability-plasticity analysis."""
        # Get current Recall@10 performance
        metrics = trainer.logged_metrics
        recall_at_10 = float(metrics.get('Recall@10_val', 0.0))
        
        # Store performance for current repository
        self.repository_performance[self.current_repository].append(recall_at_10)
        
        # Calculate stability and plasticity scores
        stability_score = self._calculate_stability()
        plasticity_score = recall_at_10  # Current performance as plasticity measure
        
        # Log metrics
        trainer.logger.log_metrics({
            'stability_score': stability_score,
            'plasticity_score': plasticity_score,
            'stability_plasticity_ratio': stability_score / max(plasticity_score, 1e-8)
        }, step=trainer.global_step)
        
        self.logger.info(f"Stability: {stability_score:.4f}, Plasticity: {plasticity_score:.4f}")
    
    def _calculate_stability(self) -> float:
        """
        Calculate stability score based on performance retention.
        
        Stability measures how well performance is maintained on previous repositories.
        
        Returns:
            Stability score (0.0 to 1.0, higher is better)
        """
        if len(self.repository_performance) <= 1:
            return 1.0  # Perfect stability for first repository
        
        stability_scores = []
        
        # For each previous repository, check if performance is maintained
        for repo_name, performances in self.repository_performance.items():
            if repo_name == self.current_repository or not performances:
                continue
            
            # Peak performance on this repository
            peak_performance = max(performances)
            
            # Current performance (latest)
            current_performance = performances[-1]
            
            # Stability is maintained performance relative to peak
            if peak_performance > 0:
                stability = min(1.0, current_performance / peak_performance)
                stability_scores.append(stability)
        
        return sum(stability_scores) / len(stability_scores) if stability_scores else 1.0
    
    def get_stability_plasticity_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive stability-plasticity analysis.
        
        Returns:
            Dictionary with stability-plasticity metrics
        """
        summary = {
            'current_repository': self.current_repository,
            'total_repositories': len(self.repository_performance),
            'repository_performance': self.repository_performance,
            'current_stability': self._calculate_stability()
        }
        
        # Calculate average performance per repository
        avg_performance = {}
        for repo, performances in self.repository_performance.items():
            if performances:
                avg_performance[repo] = sum(performances) / len(performances)
        
        summary['average_performance_per_repository'] = avg_performance
        
        return summary


class EWCLossMonitor(Callback):
    """
    Monitor EWC (Elastic Weight Consolidation) loss during training.
    
    Tracks the EWC penalty term to understand how the model balances
    new learning with preservation of important parameters.
    """
    
    def __init__(self, config: LeanAgentConfig):
        """
        Initialize EWC loss monitor.
        
        Args:
            config: LeanAgent configuration
        """
        super().__init__()
        self.config = config
        self.logger = self._setup_logging()
        self.ewc_losses: List[float] = []
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the monitor."""
        logger = logging.getLogger(f"{__name__}.EWCLossMonitor")
        logger.setLevel(logging.INFO)
        return logger
    
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx):
        """Monitor EWC loss after each training batch."""
        # Check if model has EWC capability
        if hasattr(pl_module, 'ewc_loss'):
            try:
                ewc_loss = float(pl_module.ewc_loss())
                self.ewc_losses.append(ewc_loss)
                
                # Log EWC loss periodically
                if batch_idx % 100 == 0:  # Every 100 batches
                    trainer.logger.log_metrics({
                        'ewc_loss': ewc_loss
                    }, step=trainer.global_step)
                    
                    if ewc_loss > 0:
                        self.logger.info(f"Batch {batch_idx} - EWC Loss: {ewc_loss:.6f}")
                        
            except Exception as e:
                self.logger.warning(f"Failed to compute EWC loss: {e}")
    
    def get_ewc_summary(self) -> Dict[str, Any]:
        """
        Get EWC loss summary statistics.
        
        Returns:
            Dictionary with EWC loss statistics
        """
        if not self.ewc_losses:
            return {'ewc_enabled': False}
        
        return {
            'ewc_enabled': True,
            'total_ewc_measurements': len(self.ewc_losses),
            'average_ewc_loss': sum(self.ewc_losses) / len(self.ewc_losses),
            'max_ewc_loss': max(self.ewc_losses),
            'min_ewc_loss': min(self.ewc_losses),
            'final_ewc_loss': self.ewc_losses[-1] if self.ewc_losses else 0.0
        }


class LeanAgentCallbacks:
    """
    Callback manager for LeanAgent training.
    
    Provides a centralized way to create and manage all training callbacks
    for the progressive training pipeline.
    """
    
    def __init__(self, config: LeanAgentConfig):
        """
        Initialize callback manager.
        
        Args:
            config: LeanAgent configuration
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize custom monitors
        self.lifelong_monitor = LifelongLearningMonitor(config)
        self.stability_monitor = StabilityPlasticityMonitor(config)
        self.ewc_monitor = EWCLossMonitor(config) if config.use_fisher_information else None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for callback manager."""
        logger = logging.getLogger(f"{__name__}.LeanAgentCallbacks")
        logger.setLevel(logging.INFO)
        return logger
    
    def get_callbacks(self, repository_name: str) -> List[Callback]:
        """
        Get all callbacks for training on a specific repository.
        
        Args:
            repository_name: Name of the repository being trained on
            
        Returns:
            List of PyTorch Lightning callbacks
        """
        # Update monitors with current repository
        self.lifelong_monitor.set_current_repository(repository_name)
        self.stability_monitor.set_current_repository(repository_name)
        
        callbacks = []
        
        # 1. Model Checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(self.config.path_manager.checkpoint_dir),
            filename=f"{repository_name}_{{epoch}}-{{Recall@10_val:.2f}}",
            monitor="Recall@10_val",
            mode="max",
            save_top_k=3,  # Keep top 3 checkpoints
            save_last=True,  # Always save the last checkpoint
            verbose=True,
            every_n_epochs=1,
        )
        callbacks.append(checkpoint_callback)
        
        # 2. Early Stopping (conservative to prevent premature stopping)
        early_stop_callback = EarlyStopping(
            monitor="Recall@10_val",
            patience=10,  # More patience for progressive training
            mode="max",
            verbose=True,
            strict=False,  # Don't fail if metric not available
        )
        callbacks.append(early_stop_callback)
        
        # 3. Learning Rate Monitoring
        lr_monitor = LearningRateMonitor(
            logging_interval='step',
            log_momentum=True
        )
        callbacks.append(lr_monitor)
        
        # 4. Custom Lifelong Learning Monitors
        callbacks.append(self.lifelong_monitor)
        callbacks.append(self.stability_monitor)
        
        # 5. EWC Monitor (if enabled)
        if self.ewc_monitor:
            callbacks.append(self.ewc_monitor)
        
        self.logger.info(f"Created {len(callbacks)} callbacks for repository: {repository_name}")
        
        return callbacks
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive training summary from all callbacks.
        
        Returns:
            Dictionary with training summary from all monitors
        """
        summary = {
            'lifelong_learning_metrics': self.lifelong_monitor.get_metrics_summary(),
            'stability_plasticity_metrics': self.stability_monitor.get_stability_plasticity_summary(),
        }
        
        if self.ewc_monitor:
            summary['ewc_metrics'] = self.ewc_monitor.get_ewc_summary()
        
        return summary
    
    def reset_for_new_curriculum(self):
        """
        Reset callback state for a new curriculum.
        
        Call this when starting training on a completely new set of repositories.
        """
        self.lifelong_monitor.training_metrics.clear()
        self.lifelong_monitor.validation_metrics.clear()
        self.stability_monitor.repository_performance.clear()
        
        if self.ewc_monitor:
            self.ewc_monitor.ewc_losses.clear()
        
        self.logger.info("Callback state reset for new curriculum") 