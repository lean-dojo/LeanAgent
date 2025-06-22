"""
LeanAgent Theorem Proving Module

This module provides a comprehensive theorem proving pipeline for formal mathematical
theorem proving in Lean. It implements the core proving algorithms and distributed
computation capabilities used in the LeanAgent research framework.

Key Components:
- Search algorithms (best-first search with tactic generation)
- Tactic generators (retrieval-augmented, GPT-4, fixed tactics)
- Distributed proving with Ray-based parallelization
- Proof result processing and batch management
- Search tree structures and node management

The module integrates with LeanDojo for Lean environment interaction and supports
both single-node and distributed proving workflows.
"""

from .search import BestFirstSearchProver, SearchResult, SearchStatus
from .distributed import DistributedProver, ProverManager, ProverConfig
from .generators import TacticGenerator, RetrievalAugmentedGenerator, FixedTacticGenerator
from .batch import BatchProcessor, TheoremBatch
from .tree import SearchTree, SearchNode, ProofNode, ErrorNode

__all__ = [
    # Core search components
    "BestFirstSearchProver",
    "SearchResult", 
    "SearchStatus",
    
    # Distributed proving
    "DistributedProver",
    "ProverManager",
    "ProverConfig",
    
    # Tactic generators
    "TacticGenerator",
    "RetrievalAugmentedGenerator", 
    "FixedTacticGenerator",
    
    # Batch processing
    "BatchProcessor",
    "TheoremBatch",
    
    # Search tree
    "SearchTree",
    "SearchNode",
    "ProofNode",
    "ErrorNode",
]

# Version info
__version__ = "1.0.0"
__author__ = "LeanAgent Research Team" 