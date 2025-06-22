"""
Checkpoint Management for LeanAgent

This module provides comprehensive checkpoint management for the progressive
training pipeline, including:

1. **Model Checkpoint Saving**: Save models at optimal points during training
2. **Checkpoint Recovery**: Load and resume from saved checkpoints
3. **Version Management**: Track different model versions across repositories
4. **Automatic Cleanup**: Manage disk space by removing old checkpoints
5. **Metadata Tracking**: Store training metadata with checkpoints

Research Implementation:
- Saves checkpoints after each repository training (progressive training)
- Maintains checkpoint history for analysis
- Supports distributed training checkpoint synchronization
- Integrates with Fisher Information Matrix storage
"""

import os
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

import torch
import pytorch_lightning as pl

from ..config import LeanAgentConfig


@dataclass
class CheckpointMetadata:
    """Metadata associated with a model checkpoint."""
    repository_name: str
    epoch: int
    global_step: int
    recall_at_10: float
    loss_train: float
    timestamp: str
    model_config: Dict[str, Any]
    training_time: float
    checkpoint_path: str
    file_size_mb: float


class CheckpointManager:
    """
    Comprehensive checkpoint management for LeanAgent progressive training.
    
    Handles saving, loading, and managing model checkpoints throughout
    the lifelong learning process. Provides versioning, metadata tracking,
    and automatic cleanup capabilities.
    """
    
    def __init__(self, config: LeanAgentConfig):
        """
        Initialize checkpoint manager.
        
        Args:
            config: LeanAgent configuration
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Checkpoint storage directories
        self.checkpoint_dir = config.path_manager.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata storage
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self.checkpoint_history: List[CheckpointMetadata] = []
        
        # Load existing metadata
        self._load_checkpoint_metadata()
        
        self.logger.info("Checkpoint Manager initialized")
        self.logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for checkpoint manager."""
        logger = logging.getLogger(f"{__name__}.CheckpointManager")
        logger.setLevel(logging.INFO)
        return logger
    
    def _load_checkpoint_metadata(self):
        """Load existing checkpoint metadata from file."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata_list = json.load(f)
                
                # Convert dictionaries back to CheckpointMetadata objects
                self.checkpoint_history = [
                    CheckpointMetadata(**metadata) for metadata in metadata_list
                ]
                
                self.logger.info(f"Loaded {len(self.checkpoint_history)} checkpoint metadata records")
            else:
                self.checkpoint_history = []
                
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint metadata: {e}")
            self.checkpoint_history = []
    
    def _save_checkpoint_metadata(self):
        """Save checkpoint metadata to file."""
        try:
            # Convert CheckpointMetadata objects to dictionaries
            metadata_list = [asdict(metadata) for metadata in self.checkpoint_history]
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata_list, f, indent=2)
                
            self.logger.debug("Checkpoint metadata saved")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint metadata: {e}")
    
    def save_checkpoint(
        self, 
        model: pl.LightningModule, 
        repository_name: str, 
        recall_at_10: float,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save a model checkpoint with comprehensive metadata.
        
        Args:
            model: PyTorch Lightning model to save
            repository_name: Name of the repository the model was trained on
            recall_at_10: Validation Recall@10 score
            additional_metadata: Optional additional metadata to store
            
        Returns:
            Path to the saved checkpoint file
        """
        try:
            # Generate checkpoint filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{repository_name}_{timestamp}_recall_{recall_at_10:.3f}.ckpt"
            checkpoint_path = self.checkpoint_dir / filename
            
            # Save the model checkpoint
            trainer = model.trainer if hasattr(model, 'trainer') else None
            if trainer:
                trainer.save_checkpoint(str(checkpoint_path))
            else:
                # Fallback: save model state dict
                torch.save({
                    'state_dict': model.state_dict(),
                    'hyper_parameters': getattr(model, 'hparams', {}),
                }, str(checkpoint_path))
            
            # Get file size
            file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            
            # Create metadata
            metadata = CheckpointMetadata(
                repository_name=repository_name,
                epoch=trainer.current_epoch if trainer else 0,
                global_step=trainer.global_step if trainer else 0,
                recall_at_10=recall_at_10,
                loss_train=float(trainer.logged_metrics.get('loss_train', 0.0)) if trainer else 0.0,
                timestamp=datetime.now().isoformat(),
                model_config=getattr(model, 'hparams', {}),
                training_time=0.0,  # Will be updated by training manager
                checkpoint_path=str(checkpoint_path),
                file_size_mb=file_size_mb
            )
            
            # Add additional metadata if provided
            if additional_metadata:
                for key, value in additional_metadata.items():
                    if hasattr(metadata, key):
                        setattr(metadata, key, value)
            
            # Store metadata
            self.checkpoint_history.append(metadata)
            self._save_checkpoint_metadata()
            
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            self.logger.info(f"Recall@10: {recall_at_10:.4f}, File size: {file_size_mb:.2f} MB")
            
            return checkpoint_path
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def find_latest_checkpoint(self) -> str:
        """
        Find the most recent checkpoint file.
        
        Returns:
            Path to the latest checkpoint file
            
        Raises:
            FileNotFoundError: If no checkpoints are found
        """
        # Check metadata first
        if self.checkpoint_history:
            latest_metadata = max(self.checkpoint_history, key=lambda m: m.timestamp)
            if Path(latest_metadata.checkpoint_path).exists():
                self.logger.info(f"Found latest checkpoint from metadata: {latest_metadata.checkpoint_path}")
                return latest_metadata.checkpoint_path
        
        # Fallback: scan directory for .ckpt files
        checkpoint_files = list(self.checkpoint_dir.glob("*.ckpt"))
        
        if not checkpoint_files:
            raise FileNotFoundError("No checkpoint files found")
        
        # Sort by modification time and return the latest
        latest_file = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
        self.logger.info(f"Found latest checkpoint from directory scan: {latest_file}")
        return str(latest_file)
    
    def find_best_checkpoint(self, metric: str = "recall_at_10") -> Optional[str]:
        """
        Find the checkpoint with the best performance on a specific metric.
        
        Args:
            metric: Metric to optimize for (default: "recall_at_10")
            
        Returns:
            Path to the best checkpoint, or None if not found
        """
        if not self.checkpoint_history:
            return None
        
        try:
            # Find checkpoint with highest metric value
            best_metadata = max(self.checkpoint_history, key=lambda m: getattr(m, metric))
            
            if Path(best_metadata.checkpoint_path).exists():
                self.logger.info(f"Found best checkpoint by {metric}: {best_metadata.checkpoint_path}")
                self.logger.info(f"Best {metric}: {getattr(best_metadata, metric):.4f}")
                return best_metadata.checkpoint_path
            else:
                self.logger.warning(f"Best checkpoint file not found: {best_metadata.checkpoint_path}")
                return None
                
        except AttributeError:
            self.logger.error(f"Invalid metric: {metric}")
            return None
    
    def find_checkpoint_for_repository(self, repository_name: str) -> Optional[str]:
        """
        Find the latest checkpoint for a specific repository.
        
        Args:
            repository_name: Name of the repository
            
        Returns:
            Path to the repository's checkpoint, or None if not found
        """
        repo_checkpoints = [
            m for m in self.checkpoint_history 
            if m.repository_name == repository_name
        ]
        
        if not repo_checkpoints:
            return None
        
        # Get the latest checkpoint for this repository
        latest_repo_checkpoint = max(repo_checkpoints, key=lambda m: m.timestamp)
        
        if Path(latest_repo_checkpoint.checkpoint_path).exists():
            self.logger.info(f"Found checkpoint for {repository_name}: {latest_repo_checkpoint.checkpoint_path}")
            return latest_repo_checkpoint.checkpoint_path
        else:
            self.logger.warning(f"Checkpoint file not found: {latest_repo_checkpoint.checkpoint_path}")
            return None
    
    def load_checkpoint(
        self, 
        checkpoint_path: str, 
        model_class: type,
        device: torch.device,
        **kwargs
    ) -> pl.LightningModule:
        """
        Load a model from checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            model_class: Model class to instantiate
            device: Device to load the model on
            **kwargs: Additional arguments for model loading
            
        Returns:
            Loaded model instance
        """
        try:
            self.logger.info(f"Loading checkpoint from: {checkpoint_path}")
            
            # Load using the model class's load method if available
            if hasattr(model_class, 'load'):
                model = model_class.load(checkpoint_path, device, **kwargs)
            else:
                # Fallback: load checkpoint and create model
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model = model_class(**kwargs)
                model.load_state_dict(checkpoint['state_dict'])
                model.to(device)
            
            self.logger.info(f"Successfully loaded model from {checkpoint_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            raise
    
    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of all checkpoints.
        
        Returns:
            Dictionary with checkpoint statistics and information
        """
        if not self.checkpoint_history:
            return {
                "total_checkpoints": 0,
                "checkpoint_dir": str(self.checkpoint_dir)
            }
        
        # Calculate statistics
        total_size_mb = sum(m.file_size_mb for m in self.checkpoint_history)
        avg_recall = sum(m.recall_at_10 for m in self.checkpoint_history) / len(self.checkpoint_history)
        best_recall = max(m.recall_at_10 for m in self.checkpoint_history)
        
        # Repository statistics
        repositories = list(set(m.repository_name for m in self.checkpoint_history))
        
        # Latest checkpoint info
        latest_checkpoint = max(self.checkpoint_history, key=lambda m: m.timestamp)
        
        return {
            "total_checkpoints": len(self.checkpoint_history),
            "total_size_mb": total_size_mb,
            "average_recall_at_10": avg_recall,
            "best_recall_at_10": best_recall,
            "repositories_trained": len(repositories),
            "repository_names": repositories,
            "latest_checkpoint": {
                "repository": latest_checkpoint.repository_name,
                "recall_at_10": latest_checkpoint.recall_at_10,
                "timestamp": latest_checkpoint.timestamp,
                "path": latest_checkpoint.checkpoint_path
            },
            "checkpoint_dir": str(self.checkpoint_dir)
        }
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 10, keep_best_n: int = 5):
        """
        Clean up old checkpoints to save disk space.
        
        Args:
            keep_last_n: Number of most recent checkpoints to keep
            keep_best_n: Number of best-performing checkpoints to keep
        """
        if len(self.checkpoint_history) <= keep_last_n:
            return
        
        # Sort checkpoints by timestamp (newest first)
        sorted_by_time = sorted(self.checkpoint_history, key=lambda m: m.timestamp, reverse=True)
        
        # Sort checkpoints by recall@10 (best first)
        sorted_by_performance = sorted(self.checkpoint_history, key=lambda m: m.recall_at_10, reverse=True)
        
        # Determine checkpoints to keep
        keep_checkpoints = set()
        
        # Keep last N checkpoints
        for checkpoint in sorted_by_time[:keep_last_n]:
            keep_checkpoints.add(checkpoint.checkpoint_path)
        
        # Keep best N checkpoints
        for checkpoint in sorted_by_performance[:keep_best_n]:
            keep_checkpoints.add(checkpoint.checkpoint_path)
        
        # Remove checkpoints not in keep set
        removed_count = 0
        remaining_checkpoints = []
        
        for checkpoint in self.checkpoint_history:
            if checkpoint.checkpoint_path not in keep_checkpoints:
                try:
                    Path(checkpoint.checkpoint_path).unlink()
                    removed_count += 1
                    self.logger.info(f"Removed old checkpoint: {Path(checkpoint.checkpoint_path).name}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove checkpoint {checkpoint.checkpoint_path}: {e}")
                    remaining_checkpoints.append(checkpoint)  # Keep in metadata if file removal failed
            else:
                remaining_checkpoints.append(checkpoint)
        
        # Update checkpoint history
        self.checkpoint_history = remaining_checkpoints
        self._save_checkpoint_metadata()
        
        self.logger.info(f"Cleanup completed: removed {removed_count} old checkpoints")
    
    def export_checkpoint_report(self, output_path: Optional[str] = None) -> str:
        """
        Export a detailed report of all checkpoints.
        
        Args:
            output_path: Path to save the report (default: checkpoint_dir/report.json)
            
        Returns:
            Path to the exported report file
        """
        if output_path is None:
            output_path = str(self.checkpoint_dir / "checkpoint_report.json")
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": self.get_checkpoint_summary(),
            "detailed_history": [asdict(checkpoint) for checkpoint in self.checkpoint_history]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Checkpoint report exported to: {output_path}")
        return output_path 