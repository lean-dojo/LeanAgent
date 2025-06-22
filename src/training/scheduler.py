"""
Learning Rate Scheduler for LeanAgent

This module implements the learning rate scheduling strategy used in the
research paper for progressive training:

1. **Linear Warmup**: Gradual learning rate increase from 0 to max_lr
2. **Cosine Decay**: Smooth decay to 0 following cosine schedule
3. **Progressive Training Adaptation**: Scheduling across multiple repositories
4. **Distributed Training Support**: Proper scaling for multi-GPU training

Research Implementation:
- Follows the paper's learning rate schedule (1e-3 max, 1000 warmup steps)
- Integrates with PyTorch Lightning optimizers
- Supports curriculum learning across repositories
- Provides learning rate monitoring and logging
"""

import math
import logging
from typing import Dict, Any, Optional, List, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import pytorch_lightning as pl

from ..config import LeanAgentConfig, TrainingConfig


class WarmupCosineScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by cosine decay.
    
    This implements the learning rate schedule from the research paper:
    1. Linear warmup from 0 to max_lr over warmup_steps
    2. Cosine decay from max_lr to 0 over remaining steps
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        max_lr: float,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        """
        Initialize warmup cosine scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            max_lr: Maximum learning rate after warmup
            min_lr: Minimum learning rate at end of decay
            last_epoch: Last epoch number for resuming
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """
        Calculate learning rate for current step.
        
        Returns:
            List of learning rates for each parameter group
        """
        current_step = self.last_epoch + 1
        
        if current_step < self.warmup_steps:
            # Linear warmup phase
            lr_scale = current_step / self.warmup_steps
            current_lr = self.max_lr * lr_scale
        else:
            # Cosine decay phase
            decay_steps = current_step - self.warmup_steps
            total_decay_steps = self.total_steps - self.warmup_steps
            
            if total_decay_steps > 0:
                # Cosine decay from max_lr to min_lr
                cosine_factor = 0.5 * (1 + math.cos(math.pi * decay_steps / total_decay_steps))
                current_lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_factor
            else:
                current_lr = self.max_lr
        
        return [current_lr for _ in self.optimizer.param_groups]


class AdaptiveWarmupScheduler(_LRScheduler):
    """
    Adaptive learning rate scheduler for progressive training.
    
    This scheduler adapts the learning rate based on:
    1. Current repository in the curriculum
    2. Performance on previous repositories
    3. Stability-plasticity tradeoff requirements
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float,
        warmup_steps: int,
        adaptation_factor: float = 0.9,
        min_lr_factor: float = 0.1,
        last_epoch: int = -1
    ):
        """
        Initialize adaptive warmup scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            base_lr: Base learning rate
            warmup_steps: Number of warmup steps per repository
            adaptation_factor: Factor to reduce learning rate for later repositories
            min_lr_factor: Minimum learning rate as fraction of base_lr
            last_epoch: Last epoch number for resuming
        """
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.adaptation_factor = adaptation_factor
        self.min_lr_factor = min_lr_factor
        
        # Progressive training state
        self.current_repository_step = 0
        self.repository_count = 0
        
        super().__init__(optimizer, last_epoch)
    
    def new_repository(self):
        """
        Signal the start of training on a new repository.
        
        Resets step counter and adapts learning rate for the new repository.
        """
        self.repository_count += 1
        self.current_repository_step = 0
        
        # Adapt base learning rate for later repositories
        adapted_lr = self.base_lr * (self.adaptation_factor ** (self.repository_count - 1))
        adapted_lr = max(adapted_lr, self.base_lr * self.min_lr_factor)
        
        # Update optimizer base learning rates
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = adapted_lr
    
    def get_lr(self) -> List[float]:
        """
        Calculate learning rate for current step within repository.
        
        Returns:
            List of learning rates for each parameter group
        """
        if self.current_repository_step < self.warmup_steps:
            # Linear warmup within current repository
            warmup_factor = self.current_repository_step / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Use base learning rate after warmup
            return list(self.base_lrs)
    
    def step(self, epoch: Optional[int] = None):
        """
        Update learning rate for next step.
        
        Args:
            epoch: Current epoch (optional)
        """
        self.current_repository_step += 1
        super().step(epoch)


class LearningRateScheduler:
    """
    Learning rate scheduler manager for LeanAgent progressive training.
    
    Provides centralized management of learning rate scheduling across
    the entire lifelong learning pipeline.
    """
    
    def __init__(self, config: LeanAgentConfig):
        """
        Initialize learning rate scheduler manager.
        
        Args:
            config: LeanAgent configuration
        """
        self.config = config
        self.training_config = config.training
        self.logger = self._setup_logging()
        
        # Scheduler configuration
        self.scheduler_type = getattr(self.training_config, 'scheduler_type', 'warmup_cosine')
        self.current_scheduler = None
        
        self.logger.info("Learning Rate Scheduler Manager initialized")
        self.logger.info(f"Scheduler type: {self.scheduler_type}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for scheduler manager."""
        logger = logging.getLogger(f"{__name__}.LearningRateScheduler")
        logger.setLevel(logging.INFO)
        return logger
    
    def create_scheduler(
        self, 
        optimizer: Optimizer, 
        total_steps: int,
        repository_name: Optional[str] = None
    ) -> _LRScheduler:
        """
        Create learning rate scheduler for training.
        
        Args:
            optimizer: PyTorch optimizer
            total_steps: Total number of training steps
            repository_name: Name of current repository (for logging)
            
        Returns:
            Learning rate scheduler instance
        """
        if self.scheduler_type == 'warmup_cosine':
            scheduler = WarmupCosineScheduler(
                optimizer=optimizer,
                warmup_steps=self.training_config.warmup_steps,
                total_steps=total_steps,
                max_lr=self.training_config.learning_rate,
                min_lr=0.0
            )
        elif self.scheduler_type == 'adaptive':
            scheduler = AdaptiveWarmupScheduler(
                optimizer=optimizer,
                base_lr=self.training_config.learning_rate,
                warmup_steps=self.training_config.warmup_steps,
                adaptation_factor=0.9,
                min_lr_factor=0.1
            )
        else:
            # Fallback: constant learning rate
            from torch.optim.lr_scheduler import ConstantLR
            scheduler = ConstantLR(optimizer, factor=1.0)
        
        self.current_scheduler = scheduler
        
        repository_info = f" for {repository_name}" if repository_name else ""
        self.logger.info(f"Created {self.scheduler_type} scheduler{repository_info}")
        self.logger.info(f"Total steps: {total_steps}, Warmup steps: {self.training_config.warmup_steps}")
        
        return scheduler
    
    def get_pytorch_lightning_config(
        self, 
        optimizer: Optimizer,
        total_steps: int,
        repository_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get PyTorch Lightning scheduler configuration.
        
        Args:
            optimizer: PyTorch optimizer
            total_steps: Total number of training steps
            repository_name: Name of current repository
            
        Returns:
            Dictionary with scheduler configuration for PyTorch Lightning
        """
        scheduler = self.create_scheduler(optimizer, total_steps, repository_name)
        
        # PyTorch Lightning scheduler configuration
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",  # Update every step
            "frequency": 1,      # Update frequency
            "monitor": None,     # No metric monitoring for LR scheduling
            "strict": False,     # Don't fail if metric not available
            "name": f"lr_scheduler_{repository_name}" if repository_name else "lr_scheduler"
        }
        
        return scheduler_config
    
    def get_current_lr(self) -> Optional[float]:
        """
        Get current learning rate from active scheduler.
        
        Returns:
            Current learning rate, or None if no active scheduler
        """
        if self.current_scheduler is None:
            return None
        
        # Get learning rate from first parameter group
        if hasattr(self.current_scheduler, 'get_last_lr'):
            lrs = self.current_scheduler.get_last_lr()
            return lrs[0] if lrs else None
        
        return None
    
    def step_scheduler(self):
        """
        Step the current scheduler (for manual stepping).
        """
        if self.current_scheduler:
            self.current_scheduler.step()
    
    def new_repository_training(self):
        """
        Signal the start of training on a new repository.
        
        For adaptive schedulers, this triggers learning rate adaptation.
        """
        if isinstance(self.current_scheduler, AdaptiveWarmupScheduler):
            self.current_scheduler.new_repository()
            self.logger.info("Adapted learning rate for new repository")
    
    def get_scheduler_summary(self) -> Dict[str, Any]:
        """
        Get summary of scheduler configuration and state.
        
        Returns:
            Dictionary with scheduler information
        """
        summary = {
            "scheduler_type": self.scheduler_type,
            "max_learning_rate": self.training_config.learning_rate,
            "warmup_steps": self.training_config.warmup_steps,
            "current_lr": self.get_current_lr(),
            "has_active_scheduler": self.current_scheduler is not None
        }
        
        # Add scheduler-specific information
        if isinstance(self.current_scheduler, WarmupCosineScheduler):
            summary.update({
                "total_steps": self.current_scheduler.total_steps,
                "min_lr": self.current_scheduler.min_lr,
                "current_step": self.current_scheduler.last_epoch + 1
            })
        elif isinstance(self.current_scheduler, AdaptiveWarmupScheduler):
            summary.update({
                "repository_count": self.current_scheduler.repository_count,
                "current_repository_step": self.current_scheduler.current_repository_step,
                "adaptation_factor": self.current_scheduler.adaptation_factor
            })
        
        return summary


def get_optimizers(
    parameters,
    trainer: pl.Trainer,
    lr: float,
    warmup_steps: int,
    weight_decay: float = 0.01
) -> Dict[str, Any]:
    """
    Create optimizer and scheduler configuration for PyTorch Lightning.
    
    This function replicates the optimizer setup from the original code
    with proper learning rate scheduling.
    
    Args:
        parameters: Model parameters to optimize
        trainer: PyTorch Lightning trainer
        lr: Learning rate
        warmup_steps: Number of warmup steps
        weight_decay: Weight decay factor
        
    Returns:
        Dictionary with optimizer and scheduler configuration
    """
    # Create AdamW optimizer (paper specification)
    optimizer = torch.optim.AdamW(
        parameters, 
        lr=lr, 
        weight_decay=weight_decay
    )
    
    # Calculate total training steps
    total_steps = trainer.estimated_stepping_batches
    
    # Create learning rate scheduler
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        max_lr=lr,
        min_lr=0.0
    )
    
    # Return PyTorch Lightning configuration
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "monitor": None,
            "strict": False,
            "name": "warmup_cosine_scheduler"
        }
    } 