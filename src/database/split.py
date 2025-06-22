"""
Data splitting strategies for LeanAgent database.

This module provides different strategies for splitting theorem datasets into
train/validation/test sets for machine learning experiments.
"""

import random
from typing import List, Dict
from collections import defaultdict
from loguru import logger

from .models import Theorem


# Type alias for split data structure
Split = Dict[str, List[Theorem]]


class DataSplitter:
    """
    Class for splitting theorem datasets using different strategies.
    
    This class provides various methods for splitting datasets, including
    random splitting and novel premise-based splitting.
    """
    
    def __init__(self, random_seed: int = 3407):
        """
        Initialize the DataSplitter.
        
        Args:
            random_seed (int): Random seed for reproducible splits.
        """
        self.random_seed = random_seed
        random.seed(random_seed)
    
    def split_data(self, theorems: List[Theorem], 
                   num_val_pct: float = 0.02, 
                   num_test_pct: float = 0.02) -> Dict[str, Split]:
        """
        Split data using multiple strategies.
        
        Args:
            theorems (List[Theorem]): List of theorems to split.
            num_val_pct (float): Percentage of data for validation set.
            num_test_pct (float): Percentage of data for test set.
            
        Returns:
            Dict[str, Split]: Dictionary mapping strategy names to split data.
        """
        num_theorems = len(theorems)
        num_val = int(num_theorems * num_val_pct)
        num_test = int(num_theorems * num_test_pct)
        
        logger.info(f"Splitting {num_theorems} theorems into train/val/test")
        logger.info(f"Val: {num_val} ({num_val_pct:.1%}), Test: {num_test} ({num_test_pct:.1%})")

        return {
            "random": self._split_randomly(theorems, num_val, num_test),
            "novel_premises": self._split_by_premise(theorems, num_val, num_test),
        }

    def _split_randomly(self, theorems: List[Theorem], num_val: int, num_test: int) -> Split:
        """
        Split theorems randomly into train/validation/test sets.
        
        Args:
            theorems (List[Theorem]): List of theorems to split.
            num_val (int): Number of theorems for validation set.
            num_test (int): Number of theorems for test set.
            
        Returns:
            Split: Dictionary with 'train', 'val', and 'test' keys.
        """
        theorems_copy = theorems.copy()
        random.shuffle(theorems_copy)
        
        num_train = len(theorems_copy) - num_val - num_test
        
        split = {
            "train": theorems_copy[:num_train],
            "val": theorems_copy[num_train:num_train + num_val],
            "test": theorems_copy[num_train + num_val:],
        }
        
        logger.info(f"Random split - Train: {len(split['train'])}, "
                   f"Val: {len(split['val'])}, Test: {len(split['test'])}")
        
        return split

    def _split_by_premise(self, theorems: List[Theorem], num_val: int, num_test: int) -> Split:
        """
        Split theorems by ensuring novel premises in validation/test sets.
        
        This method ensures that validation and test sets contain theorems that use
        premises not seen in the training set, making the evaluation more challenging.
        
        Args:
            theorems (List[Theorem]): List of theorems to split.
            num_val (int): Number of theorems for validation set.
            num_test (int): Number of theorems for test set.
            
        Returns:
            Split: Dictionary with 'train', 'val', and 'test' keys.
        """
        num_val_test = num_val + num_test
        theorems_val_test = []

        # Group theorems by the premises they use
        theorems_by_premises = defaultdict(list)
        
        for theorem in theorems:
            if theorem.traced_tactics:
                for tactic in theorem.traced_tactics:
                    if tactic.annotated_tactic and len(tactic.annotated_tactic) > 1:
                        for annotation in tactic.annotated_tactic[1]:
                            theorems_by_premises[annotation.full_name].append(theorem)

        # Sort by number of theorems using each premise (ascending)
        # This prioritizes rare premises for val/test sets
        theorems_by_premises = sorted(theorems_by_premises.items(), key=lambda x: len(x[1]))
        
        logger.info(f"Found {len(theorems_by_premises)} unique premises")

        # Select theorems for val/test that use rare premises
        for premise_name, premise_theorems in theorems_by_premises:
            if len(theorems_val_test) >= num_val_test:
                break
                
            # Add theorems that aren't already in val/test set
            for theorem in premise_theorems:
                if theorem not in theorems_val_test and len(theorems_val_test) < num_val_test:
                    theorems_val_test.append(theorem)

        # Remaining theorems go to training set
        theorems_train = [t for t in theorems if t not in theorems_val_test]
        
        # Shuffle val/test theorems and split them
        random.shuffle(theorems_val_test)

        split = {
            "train": theorems_train,
            "val": theorems_val_test[:num_val],
            "test": theorems_val_test[num_val:],
        }
        
        logger.info(f"Novel premise split - Train: {len(split['train'])}, "
                   f"Val: {len(split['val'])}, Test: {len(split['test'])}")
        
        return split 