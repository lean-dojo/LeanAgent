"""
LeanAgent Configuration Management

Handles all configuration parameters with proper validation and environment variable support.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
from .validation import ConfigValidator
from .paths import PathManager


@dataclass
class TrainingConfig:
    """Training-specific configuration."""
    batch_size: int = 4
    learning_rate: float = 1e-3
    warmup_steps: int = 1000
    max_seq_len: int = 512
    epochs_per_repo: int = 1
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 4
    precision: str = "bf16-mixed"
    ewc_lambda: float = 0.1  # EWC regularization strength
    scheduler_type: str = "warmup_cosine"  # Learning rate scheduler type
    

@dataclass 
class ModelConfig:
    """Model-specific configuration."""
    model_name: str = "kaiyuy/leandojo-lean4-retriever-byt5-small"
    num_retrieved: int = 100
    max_expansions: Optional[int] = None
    beam_size: int = 5
    search_timeout: int = 600  # 10 minutes


@dataclass
class CurriculumConfig:
    """Curriculum learning configuration."""
    enabled: bool = True
    complexity_metric: str = "exp_proof_steps"  # e^S where S = proof steps
    easy_percentile: float = 33.0
    hard_percentile: float = 67.0
    num_repos: int = 15


@dataclass
class DatabaseConfig:
    """Database configuration."""
    file_name: str = "leanagent_database.json"
    backup_enabled: bool = True
    compression_enabled: bool = False


@dataclass  
class ProvingConfig:
    """Theorem proving configuration."""
    # Tactic generation
    generator_type: str = "rag"  # "fixed", "rag", "gpt4", "vllm"
    tactic: Optional[str] = None  # For fixed tactic generator
    module: Optional[str] = None  # For fixed tactic generator
    
    # Search parameters
    timeout: int = 600  # 10 minutes per theorem
    max_expansions: Optional[int] = None
    num_sampled_tactics: int = 64
    
    # Distributed proving
    num_workers: int = 1
    num_gpus: int = 0
    use_vllm: bool = False
    
    # Batch processing
    batch_size: int = 12
    save_interval_minutes: int = 30
    encountered_theorems_file: str = "encountered_theorems.pkl"
    
    # Paths
    indexed_corpus_path: Optional[str] = None
    
    # Debugging
    debug: bool = False
    

@dataclass
class LeanAgentConfig:
    """Main configuration class for LeanAgent."""
    
    # Core paths
    raid_dir: Path = field(default_factory=lambda: Path(os.environ.get('RAID_DIR', '/tmp/leanagent')))
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints" 
    repos_dir: str = "repos_new"
    logs_dir: str = "logs"
    
    # Sub-configurations
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    proving: ProvingConfig = field(default_factory=ProvingConfig)
    
    # Runtime options
    progressive_training: bool = True
    use_fisher_information: bool = False
    single_repo_mode: bool = True
    distributed_training: bool = False
    num_gpus: int = 1
    
    # GitHub integration
    github_token: Optional[str] = field(default_factory=lambda: os.environ.get('GITHUB_ACCESS_TOKEN'))
    
    # Lean toolchain
    lean_versions: List[str] = field(default_factory=lambda: [
        "4.3.0-rc2", "4.4.0", "4.5.0", "4.6.0", "4.7.0", "4.8.0-rc1"
    ])
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.path_manager = PathManager(self.raid_dir)
        ConfigValidator.validate(self)
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories."""
        directories = [
            self.path_manager.data_dir,
            self.path_manager.checkpoint_dir,
            self.path_manager.repos_dir,
            self.path_manager.logs_dir,
            self.path_manager.tmp_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def database_path(self) -> Path:
        """Get full path to database file."""
        return self.path_manager.data_dir / self.database.file_name
    
    @classmethod
    def from_env(cls) -> 'LeanAgentConfig':
        """Create configuration from environment variables."""
        config = cls()
        
        # Override with environment variables if present
        if 'LEANAGENT_BATCH_SIZE' in os.environ:
            config.training.batch_size = int(os.environ['LEANAGENT_BATCH_SIZE'])
        if 'LEANAGENT_LEARNING_RATE' in os.environ:
            config.training.learning_rate = float(os.environ['LEANAGENT_LEARNING_RATE'])
        if 'LEANAGENT_NUM_REPOS' in os.environ:
            config.curriculum.num_repos = int(os.environ['LEANAGENT_NUM_REPOS'])
            
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'raid_dir': str(self.raid_dir),
            'data_dir': self.data_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'repos_dir': self.repos_dir,
            'logs_dir': self.logs_dir,
            'training': self.training.__dict__,
            'model': self.model.__dict__,
            'curriculum': self.curriculum.__dict__,
            'database': self.database.__dict__,
            'proving': self.proving.__dict__,
            'progressive_training': self.progressive_training,
            'use_fisher_information': self.use_fisher_information,
            'single_repo_mode': self.single_repo_mode,
            'distributed_training': self.distributed_training,
            'num_gpus': self.num_gpus,
            'lean_versions': self.lean_versions
        } 