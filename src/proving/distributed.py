"""
Distributed Theorem Proving

This module provides distributed theorem proving capabilities using Ray
for parallelization across multiple workers and GPUs. It includes the
main DistributedProver class and supporting infrastructure.

The distributed system coordinates multiple proof search processes,
manages resources efficiently, and provides fault tolerance for
large-scale theorem proving tasks.
"""

import os
import sys
import time
import torch
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from loguru import logger

from .search import BestFirstSearchProver, SearchResult
from .generators import TacticGenerator, TacticGeneratorFactory

try:
    import ray
    from ray.util.actor_pool import ActorPool
    RAY_AVAILABLE = True
except ImportError:
    logger.warning("Ray not available - distributed proving unavailable")
    RAY_AVAILABLE = False

try:
    from lean_dojo import LeanGitRepo, Theorem, Pos
    LEAN_DOJO_AVAILABLE = True
except (ImportError, Exception) as e:
    logger.warning(f"LeanDojo not available - using mock types: {e}")
    LEAN_DOJO_AVAILABLE = False
    LeanGitRepo = Any
    Theorem = Any
    Pos = Any


@dataclass
class ProverConfig:
    """Configuration for distributed prover setup."""
    
    # Model configuration
    use_vllm: bool = False
    ckpt_path: Optional[str] = None
    indexed_corpus_path: Optional[str] = None
    tactic: Optional[str] = None
    module: Optional[str] = None
    
    # Resource configuration
    num_workers: int = 1
    num_gpus: int = 0
    
    # Search configuration
    timeout: int = 600
    max_expansions: Optional[int] = None
    num_sampled_tactics: int = 64
    
    # Paths
    raid_dir: str = "/tmp"
    checkpoint_dir: str = "checkpoints"
    
    # Debugging
    debug: bool = False
    run_progressive_training: bool = True


if RAY_AVAILABLE:
    @ray.remote
    class ProverActor:
        """Ray actor for running proof search instances."""
        
        def __init__(self, tac_gen: TacticGenerator, timeout: int,
                     max_expansions: Optional[int], num_sampled_tactics: int,
                     debug: bool):
            """Initialize prover actor with search configuration."""
            self.prover = BestFirstSearchProver(
                tac_gen=tac_gen,
                timeout=timeout,
                max_expansions=max_expansions,
                num_sampled_tactics=num_sampled_tactics,
                debug=debug
            )
        
        def search(self, repo: LeanGitRepo, thm: Theorem, pos: Pos) -> Optional[SearchResult]:
            """Execute proof search for a single theorem."""
            return self.prover.search(repo, thm, pos)
        
        def get_statistics(self) -> Dict[str, Any]:
            """Get prover statistics."""
            return self.prover.get_statistics()


class DistributedProver:
    """
    Distributed theorem prover using Ray for parallelization.
    
    This prover manages multiple worker processes across potentially multiple
    machines and GPUs to efficiently search for proofs in parallel.
    """
    
    def __init__(self, config: ProverConfig):
        """
        Initialize the distributed prover.
        
        Args:
            config: Configuration for the distributed prover
        """
        self.config = config
        self.logger = logger
        
        # Validate configuration
        self._validate_config()
        
        # Initialize components
        self.tac_gen = None
        self.distributed = config.num_workers > 1
        self.prover = None
        self.prover_pool = None
        
        # Set up the prover
        self._setup_prover()
    
    def _validate_config(self):
        """Validate prover configuration."""
        if self.config.ckpt_path is None:
            if not (self.config.tactic and not self.config.indexed_corpus_path):
                raise ValueError("Must provide either ckpt_path or (tactic without indexed_corpus_path)")
        else:
            if self.config.tactic or self.config.module:
                raise ValueError("Cannot specify both ckpt_path and tactic/module")
        
        if self.config.num_workers > 1 and not RAY_AVAILABLE:
            raise ImportError("Ray required for distributed proving")
    
    def _setup_prover(self):
        """Set up the tactic generator and prover infrastructure."""
        self.logger.info("Setting up distributed prover")
        
        # Create tactic generator
        self.tac_gen = self._create_tactic_generator()
        
        if not self.distributed:
            # Single-process setup
            self.logger.info("Setting up single-process prover")
            self.prover = BestFirstSearchProver(
                tac_gen=self.tac_gen,
                timeout=self.config.timeout,
                max_expansions=self.config.max_expansions,
                num_sampled_tactics=self.config.num_sampled_tactics,
                debug=self.config.debug
            )
        else:
            # Distributed setup
            self.logger.info(f"Setting up distributed prover with {self.config.num_workers} workers")
            self._setup_distributed_prover()
    
    def _create_tactic_generator(self) -> TacticGenerator:
        """Create the appropriate tactic generator."""
        if self.config.ckpt_path is None:
            # Fixed tactic generator
            self.logger.info("Using FixedTacticGenerator")
            return TacticGeneratorFactory.create_generator(
                "fixed",
                tactic=self.config.tactic,
                module=self.config.module
            )
        
        elif self.config.use_vllm:
            # vLLM-based generator
            self.logger.info("Using vLLM tactic generator")
            return TacticGeneratorFactory.create_generator(
                "vllm",
                model_path=self.config.ckpt_path,
                num_gpus=self.config.num_gpus
            )
        
        else:
            # Retrieval-augmented generator
            self.logger.info("Using retrieval-augmented generator")
            device = torch.device("cuda") if self.config.num_gpus > 0 else torch.device("cpu")
            
            # Determine model checkpoint path
            model_checkpoint_path = self._get_model_checkpoint_path()
            
            config = {
                "model_name": "kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small",
                "lr": 1e-3,
                "warmup_steps": 1000,
                "num_beams": 5,
                "eval_num_retrieved": 10,
                "eval_num_workers": 1,
                "eval_num_gpus": 4,
                "eval_num_theorems": 100,
                "max_inp_seq_len": 512,
                "max_oup_seq_len": 128,
                "ret_ckpt_path": model_checkpoint_path,
            }
            
            tac_gen = TacticGeneratorFactory.create_generator(
                "rag",
                model_path=self.config.ckpt_path,
                device=device,
                config=config
            )
            
            # Load and reindex corpus if provided
            if self.config.indexed_corpus_path:
                tac_gen.load_corpus(self.config.indexed_corpus_path)
                tac_gen.reindex_corpus(batch_size=32)
            
            return tac_gen
    
    def _get_model_checkpoint_path(self) -> str:
        """Get the model checkpoint path for retrieval."""
        if self.config.run_progressive_training:
            return self._find_latest_checkpoint()
        else:
            # Use default mathlib checkpoint
            return f"{self.config.raid_dir}/checkpoints/mathlib4_29dcec074de168ac2bf835a77ef68bbe069194c5.ckpt"
    
    def _find_latest_checkpoint(self) -> str:
        """Find the most recent checkpoint."""
        checkpoint_dir = os.path.join(self.config.raid_dir, self.config.checkpoint_dir)
        
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        checkpoints = [
            os.path.join(checkpoint_dir, f) 
            for f in os.listdir(checkpoint_dir) 
            if f.endswith(".ckpt")
        ]
        
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        
        latest = max(checkpoints, key=os.path.getmtime)
        self.logger.info(f"Using latest checkpoint: {latest}")
        return latest
    
    def _setup_distributed_prover(self):
        """Set up distributed prover with Ray actors."""
        if not RAY_AVAILABLE:
            raise ImportError("Ray required for distributed proving")
        
        # Calculate GPU allocation
        if self.config.num_gpus >= 1:
            self.logger.info(f"Launching {self.config.num_workers} workers with {self.config.num_gpus} GPUs")
            
            if self.config.use_vllm:
                # vLLM manages GPUs internally
                num_gpus_per_worker = 0
            else:
                num_gpus_per_worker = self.config.num_gpus / self.config.num_workers
            
            # Create GPU-enabled actors
            provers = [
                ProverActor.options(num_gpus=num_gpus_per_worker).remote(
                    tac_gen=self.tac_gen,
                    timeout=self.config.timeout,
                    max_expansions=self.config.max_expansions,
                    num_sampled_tactics=self.config.num_sampled_tactics,
                    debug=self.config.debug,
                )
                for _ in range(self.config.num_workers)
            ]
        else:
            # CPU-only workers
            self.logger.info(f"Launching {self.config.num_workers} CPU workers")
            provers = [
                ProverActor.remote(
                    tac_gen=self.tac_gen,
                    timeout=self.config.timeout,
                    max_expansions=self.config.max_expansions,
                    num_sampled_tactics=self.config.num_sampled_tactics,
                    debug=self.config.debug,
                )
                for _ in range(self.config.num_workers)
            ]
        
        self.prover_pool = ActorPool(provers)
    
    def search_unordered(self, repo: LeanGitRepo, theorems: List[Theorem], 
                        positions: List[Pos]) -> List[Optional[SearchResult]]:
        """
        Perform parallel proof search for multiple theorems.
        
        Args:
            repo: Lean Git repository
            theorems: List of theorems to prove
            positions: List of theorem positions
            
        Returns:
            List of search results (order not guaranteed to match input)
        """
        if not theorems:
            return []
        
        self.logger.info(f"Starting proof search for {len(theorems)} theorems")
        
        if not self.distributed:
            # Single-process execution
            self.logger.info("Using single-process execution")
            results = []
            for thm, pos in zip(theorems, positions):
                result = self.prover.search(repo, thm, pos)
                results.append(result)
            return results
        
        # Distributed execution
        self.logger.info("Using distributed execution")
        try:
            results = list(
                self.prover_pool.map_unordered(
                    lambda p, x: p.search.remote(repo, x[0], x[1]),
                    zip(theorems, positions),
                )
            )
            
            self.logger.info(f"Completed proof search for {len(results)} theorems")
            return results
            
        except ray.exceptions.RayActorError as ex:
            self.logger.error(f"Ray actor error during proof search: {ex}")
            sys.exit(1)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get distributed prover statistics."""
        stats = {
            "config": {
                "num_workers": self.config.num_workers,
                "num_gpus": self.config.num_gpus,
                "timeout": self.config.timeout,
                "max_expansions": self.config.max_expansions,
                "num_sampled_tactics": self.config.num_sampled_tactics,
                "use_vllm": self.config.use_vllm,
                "distributed": self.distributed,
            },
            "tactic_generator": self.tac_gen.get_name() if self.tac_gen else None,
        }
        
        if not self.distributed and self.prover:
            stats.update(self.prover.get_statistics())
        
        return stats
    
    def shutdown(self):
        """Shutdown the distributed prover and clean up resources."""
        if self.prover_pool:
            self.logger.info("Shutting down distributed prover")
            # Ray actors are automatically cleaned up
            self.prover_pool = None
        
        if RAY_AVAILABLE and ray.is_initialized():
            self.logger.info("Shutting down Ray")
            ray.shutdown()


class ProverManager:
    """
    High-level manager for theorem proving operations.
    
    This class provides a simplified interface for setting up and running
    distributed theorem proving with proper resource management.
    """
    
    def __init__(self, config: ProverConfig):
        """Initialize the prover manager."""
        self.config = config
        self.logger = logger
        self.prover: Optional[DistributedProver] = None
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
    
    def start(self):
        """Start the prover manager and initialize resources."""
        self.logger.info("Starting prover manager")
        
        # Initialize Ray if needed
        if self.config.num_workers > 1 and RAY_AVAILABLE:
            if not ray.is_initialized():
                ray.init()
                self.logger.info("Ray initialized")
        
        # Create distributed prover
        self.prover = DistributedProver(self.config)
        self.logger.info("Distributed prover initialized")
    
    def prove_theorems(self, repo: LeanGitRepo, theorems: List[Theorem], 
                      positions: List[Pos]) -> List[Optional[SearchResult]]:
        """Prove a list of theorems."""
        if not self.prover:
            raise RuntimeError("Prover manager not started")
        
        return self.prover.search_unordered(repo, theorems, positions)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get prover statistics."""
        if not self.prover:
            return {"status": "not_started"}
        
        return self.prover.get_statistics()
    
    def shutdown(self):
        """Shutdown the prover manager and clean up resources."""
        if self.prover:
            self.prover.shutdown()
            self.prover = None
        
        self.logger.info("Prover manager shutdown complete") 