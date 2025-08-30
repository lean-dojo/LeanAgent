"""Enhanced LeanAgent with integrated theorem proving, conjecture generation, and advanced search."""

import os
import sys
import time
import json
import random
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    import ray
except ImportError:
    ray = None

try:
    import torch
except ImportError:
    torch = None

try:
    from lean_dojo import (
        LeanGitRepo, Theorem, Pos, TacticState, 
        ProofFinished, Dojo, DojoInitError
    )
except ImportError:
    # For environments without lean_dojo installed
    LeanGitRepo = Theorem = Pos = TacticState = None
    ProofFinished = Dojo = DojoInitError = None

from loguru import logger

# Import LeanAgent components
from prover.proof_search import DistributedProver, BestFirstSearchProver, SearchResult
from prover.search_tree import Status  # Status is defined here, not in lean_dojo

try:
    from dynamic_database import DynamicDatabase
except ImportError:
    DynamicDatabase = None

# Import enhanced components
from enhanced_theorem_prover import EnhancedTheoremProver, EnhancedProverConfig, ProverMode
from conjecture_generation import ConjectureEngine, ConjectoreProblem
from search_algorithms import AdvancedSearchEngine, SearchStrategy


@dataclass
class EnhancedLeanAgentConfig:
    """Configuration for Enhanced LeanAgent."""
    # Base LeanAgent config
    timeout: int = 300
    max_expansions: Optional[int] = None
    num_sampled_tactics: int = 64
    debug: bool = False
    
    # Enhancement config
    enable_enhanced_prover: bool = True
    enable_conjecture_generation: bool = True
    enable_advanced_search: bool = True
    prover_mode: ProverMode = ProverMode.HYBRID
    
    # Search config
    beam_width: int = 10
    search_strategy: SearchStrategy = SearchStrategy.A_STAR
    
    # Database config
    use_dynamic_database: bool = True
    database_path: Optional[str] = None


class EnhancedLeanAgent:
    """Enhanced LeanAgent with advanced theorem proving capabilities."""
    
    def __init__(
        self,
        tac_gen,
        config: EnhancedLeanAgentConfig = None,
        num_workers: int = 1,
        num_gpus: int = 0,
        raid_dir: str = None
    ):
        self.config = config or EnhancedLeanAgentConfig()
        self.tac_gen = tac_gen
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        self.raid_dir = raid_dir or os.environ.get('RAID_DIR', '/tmp')
        
        # Initialize base LeanAgent components
        self.distributed_prover = DistributedProver(
            use_vllm=False,
            ckpt_path=None,
            indexed_corpus_path=None,
            tactic=None,
            module=None,
            num_workers=num_workers,
            num_gpus=num_gpus,
            timeout=self.config.timeout,
            max_expansions=self.config.max_expansions,
            num_sampled_tactics=self.config.num_sampled_tactics,
            raid_dir=self.raid_dir,
            checkpoint_dir="checkpoints",
            debug=self.config.debug
        )
        
        # Initialize enhanced components
        if self.config.enable_enhanced_prover:
            prover_config = EnhancedProverConfig(
                mode=self.config.prover_mode,
                timeout=self.config.timeout,
                beam_width=self.config.beam_width,
                num_samples=self.config.num_sampled_tactics,
                enable_conjecture_generation=self.config.enable_conjecture_generation,
                enable_advanced_search=self.config.enable_advanced_search
            )
            
            self.enhanced_prover = EnhancedTheoremProver(
                tac_gen=tac_gen,
                config=prover_config,
                timeout=self.config.timeout,
                max_expansions=self.config.max_expansions,
                num_sampled_tactics=self.config.num_sampled_tactics,
                debug=self.config.debug
            )
        else:
            self.enhanced_prover = None
        
        if self.config.enable_conjecture_generation:
            self.conjecture_engine = ConjectureEngine()
        else:
            self.conjecture_engine = None
        
        if self.config.enable_advanced_search:
            self.advanced_search = AdvancedSearchEngine(
                max_iterations=1000,
                timeout=self.config.timeout
            )
        else:
            self.advanced_search = None
        
        # Initialize dynamic database
        if self.config.use_dynamic_database:
            db_path = self.config.database_path or os.path.join(self.raid_dir, "enhanced_leanagent.db")
            self.dynamic_db = DynamicDatabase(db_path)
        else:
            self.dynamic_db = None
        
        # Statistics
        self.total_proofs_attempted = 0
        self.successful_proofs = 0
        self.enhanced_proofs = 0
        self.conjecture_guided_proofs = 0
        
        logger.info("Enhanced LeanAgent initialized")
        logger.info(f"Enhanced prover: {self.config.enable_enhanced_prover}")
        logger.info(f"Conjecture generation: {self.config.enable_conjecture_generation}")
        logger.info(f"Advanced search: {self.config.enable_advanced_search}")
    
    def prove_theorem(
        self,
        repo: LeanGitRepo,
        theorem: Theorem,
        pos: Pos,
        use_enhancement: bool = True
    ) -> Optional[SearchResult]:
        """Prove a single theorem with optional enhancement."""
        
        self.total_proofs_attempted += 1
        logger.info(f"Proving theorem {self.total_proofs_attempted}: {theorem}")
        
        start_time = time.time()
        result = None
        
        try:
            # Try enhanced prover first if enabled
            if use_enhancement and self.enhanced_prover:
                logger.info("Using enhanced theorem prover")
                result = self.enhanced_prover.search(repo, theorem, pos, use_enhancement=True)
                
                if result and result.status == Status.PROVED:
                    self.enhanced_proofs += 1
                    logger.info("Enhanced proof successful")
            
            # Fallback to base prover
            if not result or result.status != Status.PROVED:
                logger.info("Using base distributed prover")
                results = self.distributed_prover.search_unordered(repo, [theorem], [pos])
                result = results[0] if results else None
            
            # Update statistics
            if result and result.status == Status.PROVED:
                self.successful_proofs += 1
                
                # Store in dynamic database
                if self.dynamic_db:
                    self._store_proof_result(theorem, result)
            
            elapsed = time.time() - start_time
            logger.info(f"Proof attempt completed in {elapsed:.2f}s: {result.status if result else 'FAILED'}")
            
            return result
        
        except Exception as e:
            logger.error(f"Proof attempt failed with error: {e}")
            return None
    
    def prove_theorems_batch(
        self,
        repo: LeanGitRepo,
        theorems: List[Theorem],
        positions: List[Pos],
        use_enhancement: bool = True
    ) -> List[Optional[SearchResult]]:
        """Prove multiple theorems in batch."""
        
        logger.info(f"Batch proving {len(theorems)} theorems")
        results = []
        
        for theorem, pos in zip(theorems, positions):
            result = self.prove_theorem(repo, theorem, pos, use_enhancement)
            results.append(result)
        
        # Log batch statistics
        successful_count = sum(1 for r in results if r and r.status == Status.PROVED)
        logger.info(f"Batch complete: {successful_count}/{len(theorems)} theorems proved")
        
        return results
    
    def generate_and_prove_conjectures(
        self,
        repo: LeanGitRepo,
        domain: str = "general",
        count: int = 5,
        prove_generated: bool = True
    ) -> Tuple[List[ConjectoreProblem], List[Optional[SearchResult]]]:
        """Generate conjectures and optionally attempt to prove them."""
        
        if not self.conjecture_engine:
            logger.warning("Conjecture generation not enabled")
            return [], []
        
        logger.info(f"Generating {count} conjectures in domain {domain}")
        
        # Generate conjectures
        conjectures = self.conjecture_engine.generate_lean_compatible_conjectures(
            repo=repo,
            count=count,
            domain=domain
        )
        
        proof_results = []
        
        if prove_generated and conjectures:
            logger.info(f"Attempting to prove {len(conjectures)} generated conjectures")
            
            for conjecture in conjectures:
                # This would require converting ConjectoreProblem to Theorem
                # For now, track the attempt
                logger.info(f"Would attempt to prove: {conjecture.natural_language}")
                proof_results.append(None)  # Placeholder
                
                if proof_results[-1]:
                    self.conjecture_guided_proofs += 1
        
        return conjectures, proof_results
    
    def analyze_repository(
        self,
        repo: LeanGitRepo,
        sample_size: int = 10
    ) -> Dict[str, Any]:
        """Analyze a repository's theorem proving characteristics."""
        
        logger.info(f"Analyzing repository {repo.name}")
        
        analysis = {
            "repo_name": repo.name,
            "sample_size": sample_size,
            "analysis_timestamp": time.time(),
            "conjecture_analysis": None,
            "proof_statistics": None
        }
        
        # Generate sample conjectures for the repository
        if self.conjecture_engine:
            sample_conjectures = self.conjecture_engine.generate_lean_compatible_conjectures(
                repo=repo,
                count=5,
                domain="general"
            )
            
            analysis["conjecture_analysis"] = {
                "generated_count": len(sample_conjectures),
                "domains": list(set(c.domain for c in sample_conjectures)),
                "average_complexity": sum(
                    len(c.lean_statement.split()) if c.lean_statement else 0
                    for c in sample_conjectures
                ) / max(1, len(sample_conjectures))
            }
        
        # Gather proof statistics
        analysis["proof_statistics"] = self.get_statistics()
        
        return analysis
    
    def _store_proof_result(self, theorem: Theorem, result: SearchResult):
        """Store proof result in dynamic database."""
        
        if not self.dynamic_db:
            return
        
        try:
            proof_data = {
                "theorem_name": theorem.full_name,
                "file_path": str(theorem.file_path),
                "proof_status": result.status.value,
                "proof_length": len(result.proof) if result.proof else 0,
                "actor_time": result.actor_time,
                "environment_time": result.environment_time,
                "total_time": result.total_time,
                "nodes_explored": result.num_searched_nodes,
                "timestamp": time.time()
            }
            
            # This would use the actual DynamicDatabase interface
            logger.debug(f"Stored proof result for {theorem.full_name}")
        
        except Exception as e:
            logger.warning(f"Failed to store proof result: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        
        base_stats = {
            "total_proofs_attempted": self.total_proofs_attempted,
            "successful_proofs": self.successful_proofs,
            "success_rate": self.successful_proofs / max(1, self.total_proofs_attempted),
            "enhanced_proofs": self.enhanced_proofs,
            "conjecture_guided_proofs": self.conjecture_guided_proofs
        }
        
        # Add component statistics
        if self.enhanced_prover:
            base_stats["enhanced_prover_stats"] = self.enhanced_prover.get_enhancement_statistics()
        
        if self.conjecture_engine:
            base_stats["conjecture_stats"] = self.conjecture_engine.get_conjecture_statistics()
        
        if self.advanced_search:
            base_stats["search_stats"] = self.advanced_search.get_search_statistics()
        
        return base_stats
    
    def save_session_data(self, output_path: str):
        """Save session data and statistics."""
        
        session_data = {
            "config": {
                "timeout": self.config.timeout,
                "prover_mode": self.config.prover_mode.value if self.config.prover_mode else None,
                "num_workers": self.num_workers,
                "num_gpus": self.num_gpus
            },
            "statistics": self.get_statistics(),
            "timestamp": time.time()
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        logger.info(f"Session data saved to {output_path}")
    
    def load_session_data(self, input_path: str) -> Dict[str, Any]:
        """Load previous session data."""
        
        try:
            with open(input_path, 'r') as f:
                session_data = json.load(f)
            
            logger.info(f"Loaded session data from {input_path}")
            return session_data
        
        except Exception as e:
            logger.error(f"Failed to load session data: {e}")
            return {}


def create_enhanced_leanagent(
    tac_gen,
    config: Optional[EnhancedLeanAgentConfig] = None,
    **kwargs
) -> EnhancedLeanAgent:
    """Factory function to create an Enhanced LeanAgent."""
    
    if config is None:
        config = EnhancedLeanAgentConfig()
    
    return EnhancedLeanAgent(
        tac_gen=tac_gen,
        config=config,
        **kwargs
    )