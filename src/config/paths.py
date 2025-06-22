"""Path management for LeanAgent."""

from pathlib import Path
from typing import Optional


class PathManager:
    """Manages all paths used by LeanAgent."""
    
    def __init__(self, base_dir: Path):
        """
        Initialize path manager.
        
        Args:
            base_dir: Base directory for all LeanAgent data
        """
        self.base_dir = Path(base_dir)
        
        # Core directories
        self.data_dir = self.base_dir / "data"
        self.checkpoint_dir = self.base_dir / "checkpoints"
        self.repos_dir = self.base_dir / "repos_new"
        self.logs_dir = self.base_dir / "logs"
        self.tmp_dir = self.base_dir / "tmp"
        self.fisher_dir = self.base_dir / "fisher"
        
        # Data subdirectories
        self.datasets_dir = self.data_dir / "datasets"
        self.proof_logs_dir = self.data_dir / "proof_logs"
        self.eval_results_dir = self.data_dir / "eval_results"
        
        # Log subdirectories
        self.training_logs_dir = self.logs_dir / "training"
        self.proving_logs_dir = self.logs_dir / "proving"
        self.lightning_logs_dir = self.logs_dir / "lightning_logs"
    
    def get_repo_dir(self, repo_name: str, commit_sha: str) -> Path:
        """
        Get directory for a specific repository.
        
        Args:
            repo_name: Name of the repository
            commit_sha: Commit SHA
            
        Returns:
            Path to repository directory
        """
        safe_name = repo_name.replace("/", "_").replace(".", "_")
        return self.repos_dir / f"{safe_name}_{commit_sha[:8]}"
    
    def get_dataset_dir(self, name: str) -> Path:
        """
        Get directory for a specific dataset.
        
        Args:
            name: Dataset name
            
        Returns:
            Path to dataset directory
        """
        return self.datasets_dir / name
    
    def get_checkpoint_path(self, model_name: str, epoch: int, metric_value: float) -> Path:
        """
        Get path for a model checkpoint.
        
        Args:
            model_name: Name of the model
            epoch: Training epoch
            metric_value: Value of the monitored metric
            
        Returns:
            Path to checkpoint file
        """
        filename = f"{model_name}_epoch_{epoch}_metric_{metric_value:.3f}.ckpt"
        return self.checkpoint_dir / filename
    
    def get_fisher_path(self, repo_name: str, commit_sha: str) -> Path:
        """
        Get path for Fisher Information Matrix file.
        
        Args:
            repo_name: Name of the repository
            commit_sha: Commit SHA
            
        Returns:
            Path to Fisher matrix file
        """
        safe_name = repo_name.replace("/", "_").replace(".", "_")
        filename = f"fisher_{safe_name}_{commit_sha[:8]}.pkl"
        return self.fisher_dir / filename
    
    def get_proof_log_path(self, repo_name: str, timestamp: Optional[str] = None) -> Path:
        """
        Get path for proof logs.
        
        Args:
            repo_name: Name of the repository
            timestamp: Optional timestamp string
            
        Returns:
            Path to proof log file
        """
        safe_name = repo_name.replace("/", "_").replace(".", "_")
        if timestamp:
            filename = f"proofs_{safe_name}_{timestamp}.json"
        else:
            filename = f"proofs_{safe_name}.json"
        return self.proof_logs_dir / filename
    
    def get_training_log_dir(self, experiment_name: str) -> Path:
        """
        Get directory for training logs.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Path to training log directory
        """
        return self.training_logs_dir / experiment_name
    
    def ensure_directories_exist(self) -> None:
        """Create all necessary directories."""
        directories = [
            self.data_dir,
            self.checkpoint_dir,
            self.repos_dir,
            self.logs_dir,
            self.tmp_dir,
            self.fisher_dir,
            self.datasets_dir,
            self.proof_logs_dir,
            self.eval_results_dir,
            self.training_logs_dir,
            self.proving_logs_dir,
            self.lightning_logs_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True) 