"""
Repository Sorting for Curriculum Learning

This module provides functionality for sorting repositories based on difficulty
characteristics to create optimal curriculum orderings.
"""

from typing import List, Dict, Tuple, Optional, Callable
from abc import ABC, abstractmethod
from enum import Enum
from loguru import logger

from ..database.models import Repository
from .difficulty import DifficultyCategory


class SortingStrategy(Enum):
    """Enumeration of repository sorting strategies."""
    EASY_FIRST = "easy_first"           # Start with repositories having most easy theorems
    HARD_FIRST = "hard_first"           # Start with repositories having most hard theorems  
    BALANCED = "balanced"               # Balance easy and hard theorems
    RANDOM = "random"                   # Random ordering
    MIXED_DIFFICULTY = "mixed_difficulty"  # Alternate between easy and hard repositories


class RepositorySorter:
    """
    Sorter for repositories based on curriculum learning strategies.
    
    This class provides various strategies for ordering repositories to create
    effective learning curricula for theorem proving.
    """
    
    def __init__(self, strategy: SortingStrategy = SortingStrategy.EASY_FIRST):
        """
        Initialize the repository sorter.
        
        Args:
            strategy: The sorting strategy to use
        """
        self.strategy = strategy
        self.logger = logger
        
        self.logger.info(f"RepositorySorter initialized with {strategy.value} strategy")
    
    def sort_by_easy_theorems(
        self, 
        categorized_theorems: Dict[Repository, Dict[DifficultyCategory, List[Tuple]]]
    ) -> List[Repository]:
        """
        Sort repositories by number of easy theorems (descending).
        
        This is the primary curriculum learning strategy: start with repositories
        that have the most easy theorems to build foundational knowledge.
        
        Args:
            categorized_theorems: Difficulty-categorized theorems by repository
            
        Returns:
            List of repositories sorted by easy theorem count (most easy first)
        """
        self.logger.info("Sorting repositories by easy theorem count...")
        
        def easy_count(repo: Repository) -> int:
            return len(categorized_theorems.get(repo, {}).get(DifficultyCategory.EASY, []))
        
        sorted_repos = sorted(categorized_theorems.keys(), key=easy_count, reverse=True)
        
        # Log sorting results
        for i, repo in enumerate(sorted_repos[:5]):  # Show top 5
            count = easy_count(repo)
            self.logger.info(f"  {i+1}. {repo.name}: {count} easy theorems")
        
        self.logger.info(f"Sorted {len(sorted_repos)} repositories by easy theorem count")
        return sorted_repos
    
    def sort_by_hard_theorems(
        self, 
        categorized_theorems: Dict[Repository, Dict[DifficultyCategory, List[Tuple]]]
    ) -> List[Repository]:
        """
        Sort repositories by number of hard theorems (descending).
        
        This strategy starts with the most challenging repositories first.
        
        Args:
            categorized_theorems: Difficulty-categorized theorems by repository
            
        Returns:
            List of repositories sorted by hard theorem count (most hard first)
        """
        self.logger.info("Sorting repositories by hard theorem count...")
        
        def hard_count(repo: Repository) -> int:
            hard_regular = len(categorized_theorems.get(repo, {}).get(DifficultyCategory.HARD, []))
            hard_no_proof = len(categorized_theorems.get(repo, {}).get(DifficultyCategory.NO_PROOF, []))
            return hard_regular + hard_no_proof
        
        sorted_repos = sorted(categorized_theorems.keys(), key=hard_count, reverse=True)
        
        # Log sorting results
        for i, repo in enumerate(sorted_repos[:5]):  # Show top 5
            count = hard_count(repo)
            self.logger.info(f"  {i+1}. {repo.name}: {count} hard theorems")
        
        self.logger.info(f"Sorted {len(sorted_repos)} repositories by hard theorem count")
        return sorted_repos
    
    def sort_by_difficulty_balance(
        self, 
        categorized_theorems: Dict[Repository, Dict[DifficultyCategory, List[Tuple]]]
    ) -> List[Repository]:
        """
        Sort repositories by difficulty balance score.
        
        Repositories with a good mix of easy, medium, and hard theorems are preferred.
        
        Args:
            categorized_theorems: Difficulty-categorized theorems by repository
            
        Returns:
            List of repositories sorted by balance score
        """
        self.logger.info("Sorting repositories by difficulty balance...")
        
        def balance_score(repo: Repository) -> float:
            categories = categorized_theorems.get(repo, {})
            easy_count = len(categories.get(DifficultyCategory.EASY, []))
            medium_count = len(categories.get(DifficultyCategory.MEDIUM, []))
            hard_count = len(categories.get(DifficultyCategory.HARD, []))
            
            total = easy_count + medium_count + hard_count
            if total == 0:
                return 0.0
            
            # Calculate balance as inverse of variance from equal distribution
            target_ratio = 1.0 / 3.0  # Equal distribution
            easy_ratio = easy_count / total
            medium_ratio = medium_count / total
            hard_ratio = hard_count / total
            
            variance = ((easy_ratio - target_ratio) ** 2 + 
                       (medium_ratio - target_ratio) ** 2 + 
                       (hard_ratio - target_ratio) ** 2) / 3
            
            return 1.0 / (1.0 + variance)  # Higher score = better balance
        
        sorted_repos = sorted(categorized_theorems.keys(), key=balance_score, reverse=True)
        
        # Log sorting results
        for i, repo in enumerate(sorted_repos[:5]):  # Show top 5
            score = balance_score(repo)
            self.logger.info(f"  {i+1}. {repo.name}: balance score {score:.3f}")
        
        self.logger.info(f"Sorted {len(sorted_repos)} repositories by difficulty balance")
        return sorted_repos
    
    def sort_mixed_difficulty(
        self, 
        categorized_theorems: Dict[Repository, Dict[DifficultyCategory, List[Tuple]]]
    ) -> List[Repository]:
        """
        Create mixed difficulty curriculum by alternating easy and hard repositories.
        
        Args:
            categorized_theorems: Difficulty-categorized theorems by repository
            
        Returns:
            List of repositories with alternating difficulty levels
        """
        self.logger.info("Creating mixed difficulty curriculum...")
        
        # Get easy-first and hard-first orderings
        easy_ordered = self.sort_by_easy_theorems(categorized_theorems)
        hard_ordered = self.sort_by_hard_theorems(categorized_theorems)
        
        # Alternate between them
        mixed_order = []
        easy_idx = 0
        hard_idx = 0
        prefer_easy = True
        
        while easy_idx < len(easy_ordered) or hard_idx < len(hard_ordered):
            if prefer_easy and easy_idx < len(easy_ordered):
                repo = easy_ordered[easy_idx]
                if repo not in mixed_order:
                    mixed_order.append(repo)
                easy_idx += 1
            elif hard_idx < len(hard_ordered):
                repo = hard_ordered[hard_idx]
                if repo not in mixed_order:
                    mixed_order.append(repo)
                hard_idx += 1
            else:
                # Fallback: add remaining easy repos
                repo = easy_ordered[easy_idx]
                if repo not in mixed_order:
                    mixed_order.append(repo)
                easy_idx += 1
            
            prefer_easy = not prefer_easy  # Alternate preference
        
        self.logger.info(f"Created mixed difficulty curriculum with {len(mixed_order)} repositories")
        return mixed_order
    
    def sort_random(
        self, 
        categorized_theorems: Dict[Repository, Dict[DifficultyCategory, List[Tuple]]]
    ) -> List[Repository]:
        """
        Sort repositories randomly (for baseline comparisons).
        
        Args:
            categorized_theorems: Difficulty-categorized theorems by repository
            
        Returns:
            Randomly ordered list of repositories
        """
        import random
        
        self.logger.info("Creating random repository ordering...")
        
        repos = list(categorized_theorems.keys())
        random.shuffle(repos)
        
        self.logger.info(f"Created random ordering of {len(repos)} repositories")
        return repos
    
    def sort_repositories(
        self, 
        categorized_theorems: Dict[Repository, Dict[DifficultyCategory, List[Tuple]]]
    ) -> List[Repository]:
        """
        Sort repositories using the configured strategy.
        
        Args:
            categorized_theorems: Difficulty-categorized theorems by repository
            
        Returns:
            List of repositories sorted according to the strategy
        """
        if self.strategy == SortingStrategy.EASY_FIRST:
            return self.sort_by_easy_theorems(categorized_theorems)
        elif self.strategy == SortingStrategy.HARD_FIRST:
            return self.sort_by_hard_theorems(categorized_theorems)
        elif self.strategy == SortingStrategy.BALANCED:
            return self.sort_by_difficulty_balance(categorized_theorems)
        elif self.strategy == SortingStrategy.MIXED_DIFFICULTY:
            return self.sort_mixed_difficulty(categorized_theorems)
        elif self.strategy == SortingStrategy.RANDOM:
            return self.sort_random(categorized_theorems)
        else:
            raise ValueError(f"Unknown sorting strategy: {self.strategy}")
    
    def get_curriculum_statistics(
        self, 
        sorted_repositories: List[Repository],
        categorized_theorems: Dict[Repository, Dict[DifficultyCategory, List[Tuple]]]
    ) -> Dict[str, any]:
        """
        Generate statistics about the curriculum ordering.
        
        Args:
            sorted_repositories: The ordered curriculum
            categorized_theorems: Difficulty-categorized theorems
            
        Returns:
            Dictionary with curriculum statistics
        """
        stats = {
            "total_repositories": len(sorted_repositories),
            "ordering_strategy": self.strategy.value,
            "repository_details": [],
            "cumulative_easy": [],
            "cumulative_total": []
        }
        
        cumulative_easy = 0
        cumulative_total = 0
        
        for i, repo in enumerate(sorted_repositories):
            categories = categorized_theorems.get(repo, {})
            easy_count = len(categories.get(DifficultyCategory.EASY, []))
            medium_count = len(categories.get(DifficultyCategory.MEDIUM, []))
            hard_count = len(categories.get(DifficultyCategory.HARD, []))
            no_proof_count = len(categories.get(DifficultyCategory.NO_PROOF, []))
            
            total_count = easy_count + medium_count + hard_count + no_proof_count
            
            cumulative_easy += easy_count
            cumulative_total += total_count
            
            repo_stats = {
                "position": i + 1,
                "name": repo.name,
                "url": repo.url,
                "easy_theorems": easy_count,
                "medium_theorems": medium_count,
                "hard_theorems": hard_count,
                "no_proof_theorems": no_proof_count,
                "total_theorems": total_count,
                "cumulative_easy": cumulative_easy,
                "cumulative_total": cumulative_total
            }
            
            stats["repository_details"].append(repo_stats)
            stats["cumulative_easy"].append(cumulative_easy)
            stats["cumulative_total"].append(cumulative_total)
        
        return stats 