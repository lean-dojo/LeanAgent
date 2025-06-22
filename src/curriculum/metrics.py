"""
Complexity Metrics for Curriculum Learning

This module provides different complexity metrics for calculating theorem difficulty
in curriculum learning scenarios.
"""

import math
from abc import ABC, abstractmethod
from typing import Union, List, Optional
from loguru import logger

from ..database.models import Theorem, AnnotatedTactic


class ComplexityMetric(ABC):
    """
    Abstract base class for complexity metrics.
    
    Complexity metrics calculate the difficulty of theorems based on various factors
    such as proof length, tactics used, or mathematical concepts involved.
    """
    
    @abstractmethod
    def calculate(self, theorem: Theorem) -> Union[float, None]:
        """
        Calculate the complexity/difficulty of a theorem.
        
        Args:
            theorem: The theorem to evaluate
            
        Returns:
            Float representing difficulty (higher = more difficult),
            or None if difficulty cannot be determined
        """
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return the name of this complexity metric."""
        pass


class ExponentialProofStepsMetric(ComplexityMetric):
    """
    Complexity metric based on exponential function of proof steps: e^S where S = proof steps.
    
    This metric accounts for the combinatorial explosion of possible proof paths
    as proofs get longer. Theorems with 'sorry' tactics are assigned infinite complexity.
    """
    
    def __init__(self):
        """Initialize the exponential proof steps metric."""
        self.logger = logger
    
    def calculate(self, theorem: Theorem) -> Union[float, None]:
        """
        Calculate difficulty using e^S where S is the number of proof steps.
        
        Args:
            theorem: The theorem to evaluate
            
        Returns:
            - float('inf') for theorems with 'sorry' tactics (no proof)
            - None for theorems with no tactics (to be distributed later)
            - exp(num_steps) for theorems with complete proofs
        """
        if not theorem.traced_tactics:
            return None  # No tactics to analyze
        
        proof_steps = theorem.traced_tactics
        
        # Check for 'sorry' tactics indicating incomplete proofs
        if any('sorry' in step.tactic for step in proof_steps):
            self.logger.debug(f"Theorem {theorem.full_name} contains 'sorry' - assigning infinite difficulty")
            return float('inf')
        
        # Empty proof steps list
        if len(proof_steps) == 0:
            return None  # To be distributed later
        
        # Calculate exponential difficulty
        num_steps = len(proof_steps)
        difficulty = math.exp(num_steps)
        
        self.logger.debug(f"Theorem {theorem.full_name}: {num_steps} steps -> difficulty {difficulty:.2f}")
        return difficulty
    
    def name(self) -> str:
        """Return the name of this metric."""
        return "exp_proof_steps"


class LinearProofStepsMetric(ComplexityMetric):
    """
    Simple linear complexity metric based on number of proof steps.
    
    This is a simpler alternative to the exponential metric for comparison purposes.
    """
    
    def calculate(self, theorem: Theorem) -> Union[float, None]:
        """
        Calculate difficulty as the linear number of proof steps.
        
        Args:
            theorem: The theorem to evaluate
            
        Returns:
            - float('inf') for theorems with 'sorry' tactics
            - None for theorems with no tactics
            - float(num_steps) for theorems with complete proofs
        """
        if not theorem.traced_tactics:
            return None
        
        proof_steps = theorem.traced_tactics
        
        if any('sorry' in step.tactic for step in proof_steps):
            return float('inf')
        
        if len(proof_steps) == 0:
            return None
        
        return float(len(proof_steps))
    
    def name(self) -> str:
        """Return the name of this metric."""
        return "linear_proof_steps"


class TacticComplexityMetric(ComplexityMetric):
    """
    Complexity metric based on the types of tactics used in proofs.
    
    Different tactics have different complexity weights based on their
    mathematical sophistication.
    """
    
    def __init__(self, tactic_weights: Optional[dict] = None):
        """
        Initialize with optional custom tactic weights.
        
        Args:
            tactic_weights: Dictionary mapping tactic names to complexity weights
        """
        # Default weights for common Lean tactics
        self.tactic_weights = tactic_weights or {
            'rfl': 1.0,         # Reflexivity
            'simp': 1.5,        # Simplification
            'ring': 2.0,        # Ring solver
            'omega': 2.5,       # Arithmetic solver
            'tactic': 1.0,      # Generic tactic
            'apply': 1.5,       # Apply lemma
            'exact': 1.0,       # Exact proof
            'rewrite': 2.0,     # Rewriting
            'induction': 3.0,   # Mathematical induction
            'cases': 2.5,       # Case analysis
        }
    
    def calculate(self, theorem: Theorem) -> Union[float, None]:
        """
        Calculate difficulty based on weighted sum of tactic complexities.
        
        Args:
            theorem: The theorem to evaluate
            
        Returns:
            Weighted complexity score based on tactics used
        """
        if not theorem.traced_tactics:
            return None
        
        proof_steps = theorem.traced_tactics
        
        if any('sorry' in step.tactic for step in proof_steps):
            return float('inf')
        
        if len(proof_steps) == 0:
            return None
        
        total_complexity = 0.0
        for step in proof_steps:
            tactic_name = step.tactic.split()[0] if step.tactic else 'tactic'
            weight = self.tactic_weights.get(tactic_name, 1.0)  # Default weight 1.0
            total_complexity += weight
        
        return total_complexity
    
    def name(self) -> str:
        """Return the name of this metric."""
        return "tactic_complexity"


def get_complexity_metric(metric_name: str) -> ComplexityMetric:
    """
    Factory function to get complexity metric by name.
    
    Args:
        metric_name: Name of the metric to create
        
    Returns:
        Instance of the requested complexity metric
        
    Raises:
        ValueError: If metric name is not recognized
    """
    metrics = {
        "exp_proof_steps": ExponentialProofStepsMetric,
        "linear_proof_steps": LinearProofStepsMetric,
        "tactic_complexity": TacticComplexityMetric,
    }
    
    if metric_name not in metrics:
        available = ", ".join(metrics.keys())
        raise ValueError(f"Unknown complexity metric '{metric_name}'. Available: {available}")
    
    return metrics[metric_name]() 