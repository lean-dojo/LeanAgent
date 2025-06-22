"""
Training Module for LeanAgent

This module provides the progressive training pipeline for lifelong learning
in formal theorem proving, including:
- Progressive training management
- Fisher Information Matrix computation for EWC
- Training callbacks and monitoring
- Model checkpointing and recovery
"""

from .progressive import ProgressiveTrainer
from .fisher import FisherComputationManager
from .callbacks import LeanAgentCallbacks
from .checkpoints import CheckpointManager
from .scheduler import LearningRateScheduler

__all__ = [
    'ProgressiveTrainer',
    'FisherComputationManager', 
    'LeanAgentCallbacks',
    'CheckpointManager',
    'LearningRateScheduler'
] 