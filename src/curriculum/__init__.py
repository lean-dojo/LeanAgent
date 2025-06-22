"""
Curriculum Learning Module for LeanAgent

This module provides curriculum learning capabilities for theorem proving,
including difficulty calculation, repository sorting, and curriculum construction.
"""

from .difficulty import DifficultyCalculator, DifficultyCategory
from .builder import CurriculumBuilder
from .metrics import ComplexityMetric, ExponentialProofStepsMetric, LinearProofStepsMetric, TacticComplexityMetric, get_complexity_metric
from .sorter import RepositorySorter, SortingStrategy

__all__ = [
    "DifficultyCalculator",
    "DifficultyCategory", 
    "CurriculumBuilder",
    "ComplexityMetric",
    "ExponentialProofStepsMetric",
    "LinearProofStepsMetric",
    "TacticComplexityMetric",
    "get_complexity_metric",
    "RepositorySorter",
    "SortingStrategy"
] 