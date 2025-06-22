"""
Fisher Information Matrix Computation for EWC

This module implements Fisher Information Matrix (FIM) computation for
Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting.

Key Features:
1. **Distributed FIM Computation**: Multi-GPU Fisher matrix calculation
2. **Parameter Importance Estimation**: Identifies critical parameters for previous tasks
3. **EWC Loss Integration**: Provides Fisher matrix for EWC penalty terms
4. **Automatic Persistence**: Saves/loads Fisher matrices across training sessions

Research Implementation:
- Computes Fisher Information as in Kirkpatrick et al. (2017)
- Integrates with progressive training pipeline
- Handles distributed computation across multiple GPUs
- Provides adaptive parameter importance for complex theorem proving tasks
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Union
from datetime import timedelta

import torch
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

from ..config import LeanAgentConfig


class FisherComputationModule(pl.LightningModule):
    """
    PyTorch Lightning module for computing Fisher Information Matrix.
    
    This module wraps a trained model to compute the Fisher Information Matrix
    by accumulating squared gradients during a forward pass over the training data.
    The Fisher matrix is used to identify important parameters for EWC.
    """
    
    def __init__(self, model: pl.LightningModule):
        """
        Initialize Fisher computation module.
        
        Args:
            model: The trained model to compute Fisher Information for
        """
        super().__init__()
        self.model = model
        self.fisher_info: Dict[str, torch.Tensor] = {}
        self.automatic_optimization = False  # Manual optimization for Fisher computation
        
        # Initialize Fisher information storage
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher_info[name] = torch.zeros_like(param.data)
    
    def training_step(self, batch, batch_idx):
        """
        Compute Fisher Information for a single batch.
        
        This method:
        1. Computes the loss using the model
        2. Performs backward pass to get gradients
        3. Accumulates squared gradients in Fisher Information matrix
        
        Args:
            batch: Training batch data
            batch_idx: Batch index
            
        Returns:
            Loss tensor for logging
        """
        # Reset gradients
        self.model.zero_grad()
        
        # Forward pass through the model
        loss = self.model(
            batch["context_ids"],
            batch["context_mask"],
            batch["pos_premise_ids"],
            batch["pos_premise_mask"],
            batch["neg_premises_ids"],
            batch["neg_premises_mask"],
            batch["label"],
        )
        
        # Manual backward pass
        self.manual_backward(loss)
        
        # Accumulate squared gradients in Fisher Information matrix
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Fisher Information is the expectation of squared gradients
                self.fisher_info[name] += param.grad.data ** 2
        
        # Log the loss
        self.log('fisher_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        
        return loss
    
    def on_train_epoch_end(self):
        """
        Finalize Fisher Information computation at epoch end.
        
        Normalizes the accumulated Fisher Information by the number of batches
        and synchronizes across distributed processes if applicable.
        """
        # Get the number of training batches
        num_batches = len(self.trainer.train_dataloader)
        
        # Normalize Fisher Information by number of batches
        for name in self.fisher_info:
            self.fisher_info[name] /= num_batches
        
        # Synchronize across distributed processes
        if dist.is_initialized():
            for name in self.fisher_info:
                dist.all_reduce(self.fisher_info[name], op=dist.ReduceOp.SUM)
                self.fisher_info[name] /= dist.get_world_size()
    
    def save_fisher_info(self, file_path: str):
        """
        Save Fisher Information Matrix to file.
        
        Args:
            file_path: Path to save the Fisher Information Matrix
        """
        # Convert to CPU tensors for saving
        cpu_fisher_info = {
            name: tensor.cpu() for name, tensor in self.fisher_info.items()
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save to pickle file
        with open(file_path, 'wb') as f:
            pickle.dump(cpu_fisher_info, f)
        
        logging.info(f"Fisher Information Matrix saved to {file_path}")


class FisherComputationManager:
    """
    Manager for Fisher Information Matrix computation and storage.
    
    This class handles the complete lifecycle of Fisher Information computation:
    1. Setting up distributed training for FIM computation
    2. Running Fisher computation on training data
    3. Saving and loading Fisher matrices
    4. Integrating with the progressive training pipeline
    """
    
    def __init__(self, config: LeanAgentConfig):
        """
        Initialize Fisher computation manager.
        
        Args:
            config: LeanAgent configuration
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Fisher storage directory
        self.fisher_dir = config.path_manager.checkpoint_dir / "fisher"
        self.fisher_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.logger.info("Fisher Computation Manager initialized")
        self.logger.info(f"Fisher storage directory: {self.fisher_dir}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for Fisher computation."""
        logger = logging.getLogger(f"{__name__}.FisherComputationManager")
        logger.setLevel(logging.INFO)
        return logger
    
    def compute_fisher_information(
        self, 
        model: pl.LightningModule, 
        data_module: pl.LightningDataModule,
        repository_name: str
    ) -> str:
        """
        Compute Fisher Information Matrix for a trained model.
        
        Args:
            model: Trained model to compute Fisher Information for
            data_module: Data module containing training data
            repository_name: Name of repository for file naming
            
        Returns:
            Path to saved Fisher Information Matrix file
        """
        self.logger.info(f"Computing Fisher Information Matrix for {repository_name}")
        
        try:
            # Create Fisher computation module
            fisher_module = FisherComputationModule(model)
            
            # Setup distributed training if multiple GPUs
            strategy = "auto"
            if self.config.num_gpus > 1:
                timeout = timedelta(seconds=7 * 24 * 60 * 60)  # 1 week timeout
                strategy = DDPStrategy(timeout=timeout)
                
                # NCCL environment setup
                os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
                os.environ['NCCL_TIMEOUT'] = str(timeout.total_seconds() * 1000)
            
            # Create trainer for Fisher computation
            trainer = pl.Trainer(
                accelerator="gpu" if self.device.type == "cuda" else "cpu",
                devices=self.config.num_gpus if self.device.type == "cuda" else 1,
                strategy=strategy,
                precision=self.config.training.precision,
                max_epochs=1,  # Single epoch for Fisher computation
                log_every_n_steps=1,
                num_sanity_val_steps=0,
                enable_progress_bar=True,
                logger=False,  # Disable logging for Fisher computation
            )
            
            # Synchronization barrier before training
            if hasattr(trainer.strategy, 'barrier'):
                self.logger.info("Synchronizing before Fisher computation")
                trainer.strategy.barrier()
            
            # Run Fisher computation
            trainer.fit(fisher_module, datamodule=data_module)
            
            # Synchronization barrier after training
            if hasattr(trainer.strategy, 'barrier'):
                self.logger.info("Synchronizing after Fisher computation")
                trainer.strategy.barrier()
            
            # Save Fisher Information Matrix
            fisher_file_path = self.fisher_dir / f"fisher_info_{repository_name}.pkl"
            
            # Only save from the main process in distributed training
            if not hasattr(trainer, 'is_global_zero') or trainer.is_global_zero:
                fisher_module.save_fisher_info(str(fisher_file_path))
                self.logger.info(f"Fisher Information Matrix saved to {fisher_file_path}")
            
            return str(fisher_file_path)
            
        except Exception as e:
            self.logger.error(f"Fisher computation failed: {e}")
            raise
    
    def load_fisher_information(self, file_path: str) -> Dict[str, torch.Tensor]:
        """
        Load Fisher Information Matrix from file.
        
        Args:
            file_path: Path to Fisher Information Matrix file
            
        Returns:
            Dictionary mapping parameter names to Fisher Information tensors
        """
        try:
            with open(file_path, 'rb') as f:
                fisher_info = pickle.load(f)
            
            # Move tensors to appropriate device
            device_fisher_info = {
                name: tensor.to(self.device) for name, tensor in fisher_info.items()
            }
            
            self.logger.info(f"Fisher Information Matrix loaded from {file_path}")
            return device_fisher_info
            
        except Exception as e:
            self.logger.error(f"Failed to load Fisher Information from {file_path}: {e}")
            raise
    
    def find_latest_fisher_info(self) -> Optional[str]:
        """
        Find the most recent Fisher Information Matrix file.
        
        Returns:
            Path to latest Fisher Information file, or None if not found
        """
        fisher_files = list(self.fisher_dir.glob("fisher_info_*.pkl"))
        
        if not fisher_files:
            self.logger.warning("No Fisher Information files found")
            return None
        
        # Sort by modification time and return the latest
        latest_file = max(fisher_files, key=lambda f: f.stat().st_mtime)
        self.logger.info(f"Found latest Fisher Information file: {latest_file}")
        return str(latest_file)
    
    def load_latest_fisher_info(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Load the most recent Fisher Information Matrix.
        
        Returns:
            Fisher Information dictionary, or None if not found
        """
        latest_file = self.find_latest_fisher_info()
        if latest_file:
            return self.load_fisher_information(latest_file)
        return None
    
    def get_fisher_summary(self) -> Dict[str, Any]:
        """
        Get summary information about available Fisher Information matrices.
        
        Returns:
            Dictionary with Fisher Information statistics
        """
        fisher_files = list(self.fisher_dir.glob("fisher_info_*.pkl"))
        
        summary = {
            "total_fisher_files": len(fisher_files),
            "fisher_files": [f.name for f in fisher_files],
            "latest_fisher_file": None,
            "fisher_dir": str(self.fisher_dir)
        }
        
        if fisher_files:
            latest_file = max(fisher_files, key=lambda f: f.stat().st_mtime)
            summary["latest_fisher_file"] = latest_file.name
        
        return summary
    
    def cleanup_old_fisher_files(self, keep_last_n: int = 5):
        """
        Clean up old Fisher Information files, keeping only the most recent ones.
        
        Args:
            keep_last_n: Number of recent Fisher files to keep
        """
        fisher_files = list(self.fisher_dir.glob("fisher_info_*.pkl"))
        
        if len(fisher_files) <= keep_last_n:
            return
        
        # Sort by modification time (newest first)
        sorted_files = sorted(fisher_files, key=lambda f: f.stat().st_mtime, reverse=True)
        
        # Remove old files
        files_to_remove = sorted_files[keep_last_n:]
        for file_path in files_to_remove:
            try:
                file_path.unlink()
                self.logger.info(f"Removed old Fisher file: {file_path.name}")
            except Exception as e:
                self.logger.warning(f"Failed to remove {file_path.name}: {e}")
        
        self.logger.info(f"Cleaned up {len(files_to_remove)} old Fisher Information files") 