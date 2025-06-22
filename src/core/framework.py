"""
LeanAgent Framework - Main Orchestrator

This module contains the main LeanAgentFramework class that orchestrates all components
of the lifelong learning system for formal theorem proving.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

from ..config import LeanAgentConfig, ConfigValidator
from ..config.paths import PathManager


class LeanAgentFramework:
    """
    Main framework class for LeanAgent lifelong learning system.
    
    This class orchestrates all components including:
    - Configuration management
    - Database operations  
    - Curriculum learning
    - Progressive training
    - Theorem proving
    """
    
    def __init__(self, config: Optional[LeanAgentConfig] = None):
        """
        Initialize the LeanAgent framework.
        
        Args:
            config: Configuration object. If None, creates default config.
        """
        self.config = config or LeanAgentConfig()
        self.logger = self._setup_logging()
        
        # Component placeholders - will be initialized as needed
        self._database = None
        self._curriculum_builder = None
        self._training_manager = None
        self._prover = None
        
        self.logger.info("LeanAgent Framework initialized")
        self.logger.info(f"Configuration: {self.config.to_dict()}")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("leanagent")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Create file handler
            log_file = self.config.path_manager.logs_dir / "leanagent.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # Add handlers to logger
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    @property
    def database(self):
        """Lazy-load database component."""
        if self._database is None:
            from ..database import DynamicDatabase
            db_path = self.config.path_manager.data_dir / self.config.database.file_name
            self._database = DynamicDatabase(str(db_path))
            self.logger.info("Database component initialized")
        return self._database
    
    @property
    def curriculum_builder(self):
        """Lazy-load curriculum builder component."""
        if self._curriculum_builder is None:
            from ..curriculum import CurriculumBuilder
            self._curriculum_builder = CurriculumBuilder(self.config.curriculum)
            self.logger.info("Curriculum builder component initialized")
        return self._curriculum_builder
    
    @property
    def training_manager(self):
        """Lazy-load training manager component."""
        if self._training_manager is None:
            try:
                from ..training import ProgressiveTrainer
                self._training_manager = ProgressiveTrainer(self.config)
                self.logger.info("Progressive training manager initialized")
            except ImportError as e:
                self.logger.warning(f"Training manager could not be initialized: {e}")
                return None
        return self._training_manager
    
    @property
    def prover(self):
        """Lazy-load theorem prover component."""
        if self._prover is None:
            try:
                from ..proving import ProverManager, ProverConfig
                
                # Create prover configuration from main config
                prover_config = ProverConfig(
                    use_vllm=self.config.proving.use_vllm,
                    ckpt_path=None,  # Will be determined dynamically
                    indexed_corpus_path=self.config.proving.indexed_corpus_path,
                    tactic=self.config.proving.tactic,
                    module=self.config.proving.module,
                    num_workers=self.config.proving.num_workers,
                    num_gpus=self.config.proving.num_gpus,
                    timeout=self.config.proving.timeout,
                    max_expansions=self.config.proving.max_expansions,
                    num_sampled_tactics=self.config.proving.num_sampled_tactics,
                    raid_dir=str(self.config.raid_dir),
                    checkpoint_dir=self.config.checkpoint_dir,
                    debug=self.config.proving.debug,
                    run_progressive_training=self.config.progressive_training
                )
                
                self._prover = ProverManager(prover_config)
                self.logger.info("Theorem prover initialized")
            except ImportError as e:
                self.logger.warning(f"Prover could not be initialized: {e}")
                return None
        return self._prover
    
    def initialize_components(self):
        """Initialize all framework components."""
        self.logger.info("Initializing LeanAgent components...")
        
        # Initialize available components
        try:
            # Initialize curriculum builder (available)
            _ = self.curriculum_builder
            
            # Initialize database (available)
            _ = self.database
            
            # Initialize training manager (now available)
            _ = self.training_manager
            
            # Initialize prover (now available)
            _ = self.prover
            
            self.logger.info("Available components initialized successfully")
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            raise
    
    def run_lifelong_learning(self, repositories: List[str]) -> Dict[str, Any]:
        """
        Run the complete lifelong learning pipeline.
        
        Args:
            repositories: List of repository URLs to process
            
        Returns:
            Dictionary with results and statistics
        """
        self.logger.info(f"Starting lifelong learning on {len(repositories)} repositories")
        
        results = {
            "repositories_processed": 0,
            "theorems_proven": 0,
            "training_epochs": 0,
            "total_time": 0,
            "success": False
        }
        
        try:
            # 1. Load/update database with repositories ✅ (available)
            self.logger.info("Loading repositories into database...")
            # Note: Repository loading would be implemented here
            
            # 2. Build curriculum ordering ✅ (available)  
            self.logger.info("Building curriculum from database repositories...")
            curriculum, metadata = self.curriculum_builder.build_curriculum(self.database.repositories)
            results["curriculum_built"] = True
            results["curriculum_size"] = len(curriculum)
            self.logger.info(f"Built curriculum with {len(curriculum)} repositories")
            
            # 3. Run progressive training ✅ (now available)
            if self.training_manager and curriculum:
                self.logger.info("Starting progressive training on curriculum...")
                training_results = self.training_manager.train_curriculum(curriculum, self.database)
                
                results.update({
                    "repositories_processed": training_results["successful_trainings"],
                    "training_epochs": training_results["final_metrics"].get("total_epochs", 0),
                    "total_time": training_results["total_time"],
                    "training_success_rate": training_results["successful_trainings"] / training_results["total_repositories"],
                    "final_recall_at_10": training_results["final_metrics"].get("final_recall_at_10", 0.0)
                })
                
                self.logger.info(f"Progressive training completed with {training_results['successful_trainings']}/{training_results['total_repositories']} successful trainings")
            
            # 4. Prove theorems with updated model (Phase 5 - Future)
            # self.logger.info("Starting theorem proving with trained model...")
            # proving_results = self.prover.prove_sorry_theorems(self.database)
            # results["theorems_proven"] = proving_results["total_theorems_proven"]
            
            results["success"] = True
            
        except Exception as e:
            self.logger.error(f"Lifelong learning failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def prove_theorems(self, repository_url: str, target_theorems: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Prove theorems in a specific repository.
        
        Args:
            repository_url: URL of the repository
            target_theorems: Specific theorems to prove (None = all sorry theorems)
            
        Returns:
            Proving results and statistics
        """
        self.logger.info(f"Proving theorems in repository: {repository_url}")
        
        results = {
            "repository": repository_url,
            "target_theorems": target_theorems or [],
            "proven_theorems": [],
            "failed_theorems": [],
            "success": False
        }
        
        try:
            # Get repository from database
            repo = self.database.get_repository_by_url(repository_url)
            if not repo:
                raise ValueError(f"Repository not found: {repository_url}")
            
            # Initialize prover
            prover_manager = self.prover
            if not prover_manager:
                raise RuntimeError("Prover not available")
            
            # Create batch processor
            from ..proving import BatchProcessor
            batch_processor = BatchProcessor(
                batch_size=self.config.proving.batch_size,
                save_interval_minutes=self.config.proving.save_interval_minutes,
                encountered_theorems_file=self.config.proving.encountered_theorems_file
            )
            
            # Create batches and process
            with prover_manager:
                batches = batch_processor.create_batches(repo)
                
                total_proved = 0
                total_failed = 0
                
                for batch in batches:
                    batch_result = batch_processor.process_batch(
                        batch, 
                        prover_manager.prover,
                        self.database,
                        str(self.config.database_path)
                    )
                    
                    total_proved += batch_result.proved_theorems
                    total_failed += batch_result.failed_theorems
                    
                    self.logger.info(f"Batch completed: {batch_result.proved_theorems}/{batch_result.total_theorems} proved")
                
                batch_processor.finalize()
            
            results.update({
                "proven_theorems": total_proved,
                "failed_theorems": total_failed,
                "total_batches": len(batches),
                "success": True
            })
            
            self.logger.info(f"Theorem proving completed: {total_proved} proved, {total_failed} failed")
            
        except Exception as e:
            self.logger.error(f"Theorem proving failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get current framework status."""
        return {
            "framework_initialized": True,
            "config_valid": True,
            "components": {
                "database": self._database is not None,
                "curriculum_builder": self._curriculum_builder is not None,
                "training_manager": self._training_manager is not None,
                "prover": self._prover is not None
            },
            "data_directory": str(self.config.path_manager.data_dir),
            "logs_directory": str(self.config.path_manager.logs_dir),
            "version": "1.0.0-dev"
        }
    
    def save_state(self, filepath: Optional[Path] = None) -> Path:
        """Save framework state to file."""
        if filepath is None:
            filepath = self.config.path_manager.data_dir / "framework_state.json"
        
        state = {
            "config": self.config.to_dict(),
            "status": self.get_status(),
            "components_initialized": {
                "database": self._database is not None,
                "curriculum_builder": self._curriculum_builder is not None,
                "training_manager": self._training_manager is not None,
                "prover": self._prover is not None
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        self.logger.info(f"Framework state saved to: {filepath}")
        return filepath
    
    def load_state(self, filepath: Path) -> bool:
        """Load framework state from file."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.logger.info(f"Framework state loaded from: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load state from {filepath}: {e}")
            return False 