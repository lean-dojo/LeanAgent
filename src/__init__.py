"""
LeanAgent: Lifelong Learning for Formal Theorem Proving

A framework for continuous learning across mathematical domains without catastrophic forgetting.
"""

__version__ = "1.0.0"
__author__ = "LeanAgent Team"

# Import working modules
from .config import LeanAgentConfig, ConfigValidator, PathManager
from .core import LeanAgentFramework
from .database import (
    DynamicDatabase, 
    Repository, 
    Theorem, 
    Premise, 
    PremiseFile,
    Annotation,
    AnnotatedTactic,
    DatasetExporter,
    DataSplitter
)
from .curriculum import (
    CurriculumBuilder,
    DifficultyCalculator,
    DifficultyCategory,
    RepositorySorter,
    SortingStrategy,
    ComplexityMetric,
    ExponentialProofStepsMetric
)

# Training Components
try:
    from .training import (
        ProgressiveTrainer,
        FisherComputationManager,
        LeanAgentCallbacks,
        CheckpointManager,
        LearningRateScheduler
    )
    _training_available = True
except ImportError:
    _training_available = False

# Proving Components
try:
    from .proving import (
        BestFirstSearchProver,
        SearchResult,
        SearchStatus,
        DistributedProver,
        ProverManager,
        TacticGenerator,
        FixedTacticGenerator,
        BatchProcessor,
        TheoremBatch
    )
    _proving_available = True
except ImportError:
    _proving_available = False

__all__ = [
    "LeanAgentConfig",
    "ConfigValidator", 
    "PathManager",
    "LeanAgentFramework",
    "DynamicDatabase",
    "Repository",
    "Theorem", 
    "Premise",
    "PremiseFile",
    "Annotation",
    "AnnotatedTactic",
    "DatasetExporter",
    "DataSplitter",
    "CurriculumBuilder",
    "DifficultyCalculator",
    "DifficultyCategory",
    "RepositorySorter",
    "SortingStrategy",
    "ComplexityMetric",
    "ExponentialProofStepsMetric"
]

# Add training components to __all__ if available
if _training_available:
    __all__.extend([
        "ProgressiveTrainer",
        "FisherComputationManager",
        "LeanAgentCallbacks",
        "CheckpointManager",
        "LearningRateScheduler"
    ])

# Add proving components to __all__ if available  
if _proving_available:
    __all__.extend([
        "BestFirstSearchProver",
        "SearchResult", 
        "SearchStatus",
        "DistributedProver",
        "ProverManager",
        "TacticGenerator",
        "FixedTacticGenerator",
        "BatchProcessor",
        "TheoremBatch"
    ]) 