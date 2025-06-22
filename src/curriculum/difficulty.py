"""
Difficulty Calculation and Categorization for Curriculum Learning

This module provides functionality for calculating theorem difficulties
and categorizing them into curriculum-appropriate groups.
"""

import numpy as np
from enum import Enum
from typing import List, Dict, Union, Tuple, Optional
from collections import defaultdict
from loguru import logger

from ..database.models import Theorem, Repository
from .metrics import ComplexityMetric, get_complexity_metric


class DifficultyCategory(Enum):
    """Enumeration of difficulty categories for curriculum learning."""
    EASY = "Easy"
    MEDIUM = "Medium" 
    HARD = "Hard"
    NO_PROOF = "Hard (No proof)"
    TO_DISTRIBUTE = "To_Distribute"


class DifficultyCalculator:
    """
    Calculator for theorem difficulties using configurable complexity metrics.
    
    This class handles the calculation of theorem difficulties and their categorization
    into curriculum-appropriate groups based on percentile thresholds.
    """
    
    def __init__(
        self, 
        complexity_metric: Union[str, ComplexityMetric] = "exp_proof_steps",
        easy_percentile: float = 33.0,
        hard_percentile: float = 67.0
    ):
        """
        Initialize the difficulty calculator.
        
        Args:
            complexity_metric: Either a string name or ComplexityMetric instance
            easy_percentile: Percentile threshold for easy theorems (default: 33%)
            hard_percentile: Percentile threshold for hard theorems (default: 67%)
        """
        self.logger = logger
        
        # Initialize complexity metric
        if isinstance(complexity_metric, str):
            self.complexity_metric = get_complexity_metric(complexity_metric)
        else:
            self.complexity_metric = complexity_metric
        
        # Validate percentiles
        if not (0 <= easy_percentile <= hard_percentile <= 100):
            raise ValueError("Percentiles must satisfy: 0 <= easy_percentile <= hard_percentile <= 100")
        
        self.easy_percentile = easy_percentile
        self.hard_percentile = hard_percentile
        
        self.logger.info(f"DifficultyCalculator initialized with {self.complexity_metric.name()} metric")
        self.logger.info(f"Percentile thresholds: Easy <= {easy_percentile}%, Hard > {hard_percentile}%")
    
    def calculate_difficulty(self, theorem: Theorem) -> Union[float, None]:
        """
        Calculate the difficulty of a single theorem.
        
        Args:
            theorem: The theorem to evaluate
            
        Returns:
            Difficulty score or None if cannot be determined
        """
        return self.complexity_metric.calculate(theorem)
    
    def calculate_all_difficulties(self, repositories: List[Repository]) -> Tuple[Dict[Repository, List[Tuple]], List[float]]:
        """
        Calculate difficulties for all theorems across repositories.
        
        Args:
            repositories: List of repositories to process
            
        Returns:
            Tuple of (difficulties_by_repo, all_finite_difficulties)
            where difficulties_by_repo maps Repository -> List[(theorem_name, file_path, start, end, difficulty)]
            and all_finite_difficulties contains all non-None, non-infinite difficulty values
        """
        difficulties_by_repo = defaultdict(list)
        all_difficulties = []

        self.logger.info("Calculating difficulties for all theorems...")
        
        for repo in repositories:
            self.logger.info(f"Processing repository: {repo.name}")
            repo_theorems = 0
            
            for theorem in repo.get_all_theorems:
                difficulty = self.calculate_difficulty(theorem)
                theorem.difficulty_rating = difficulty  # Store in theorem
                
                # Store difficulty information
                difficulties_by_repo[repo].append((
                    theorem.full_name,
                    str(theorem.file_path),
                    tuple(theorem.start),
                    tuple(theorem.end),
                    difficulty
                ))
                
                # Collect finite difficulties for percentile calculation
                if difficulty is not None and difficulty != float('inf'):
                    all_difficulties.append(difficulty)
                
                repo_theorems += 1
            
            self.logger.info(f"Processed {repo_theorems} theorems in {repo.name}")

        self.logger.info(f"Total finite difficulties calculated: {len(all_difficulties)}")
        return dict(difficulties_by_repo), all_difficulties
    
    def calculate_percentiles(self, difficulties: List[float]) -> Tuple[float, float]:
        """
        Calculate percentile thresholds from difficulty values.
        
        Args:
            difficulties: List of difficulty values
            
        Returns:
            Tuple of (easy_threshold, hard_threshold)
        """
        if not difficulties:
            self.logger.warning("No finite difficulties found, using default thresholds")
            return 1.0, 10.0  # Default thresholds
        
        percentiles = np.percentile(difficulties, [self.easy_percentile, self.hard_percentile])
        easy_threshold, hard_threshold = percentiles[0], percentiles[1]
        
        self.logger.info(f"Calculated thresholds: Easy <= {easy_threshold:.2f}, Hard > {hard_threshold:.2f}")
        return easy_threshold, hard_threshold
    
    def categorize_difficulty(self, difficulty: Union[float, None], percentiles: Tuple[float, float]) -> DifficultyCategory:
        """
        Categorize a difficulty value into a curriculum category.
        
        Args:
            difficulty: The difficulty value to categorize
            percentiles: Tuple of (easy_threshold, hard_threshold)
            
        Returns:
            DifficultyCategory enum value
        """
        if difficulty is None:
            return DifficultyCategory.TO_DISTRIBUTE
        if difficulty == float('inf'):
            return DifficultyCategory.NO_PROOF
        elif difficulty <= percentiles[0]:
            return DifficultyCategory.EASY
        elif difficulty <= percentiles[1]:
            return DifficultyCategory.MEDIUM
        else:
            return DifficultyCategory.HARD
    
    def categorize_all_theorems(
        self, 
        difficulties_by_repo: Dict[Repository, List[Tuple]], 
        percentiles: Tuple[float, float]
    ) -> Dict[Repository, Dict[DifficultyCategory, List[Tuple]]]:
        """
        Categorize all theorems by difficulty across repositories.
        
        Args:
            difficulties_by_repo: Repository -> theorem difficulty tuples mapping
            percentiles: Difficulty thresholds for categorization
            
        Returns:
            Nested dictionary: Repository -> DifficultyCategory -> List[theorem_tuples]
        """
        categorized_theorems = defaultdict(lambda: defaultdict(list))

        self.logger.info("Categorizing theorems by difficulty...")
        
        for repo, theorems in difficulties_by_repo.items():
            self.logger.debug(f"Categorizing {len(theorems)} theorems in {repo.name}")
            
            for theorem_name, file_path, start, end, difficulty in theorems:
                category = self.categorize_difficulty(difficulty, percentiles)
                categorized_theorems[repo][category].append((theorem_name, file_path, start, end, difficulty))

        return dict(categorized_theorems)
    
    def distribute_undetermined_theorems(
        self, 
        categorized_theorems: Dict[Repository, Dict[DifficultyCategory, List[Tuple]]]
    ) -> Dict[Repository, Dict[DifficultyCategory, List[Tuple]]]:
        """
        Distribute theorems that couldn't be categorized (TO_DISTRIBUTE) evenly across categories.
        
        Args:
            categorized_theorems: Current categorization
            
        Returns:
            Updated categorization with TO_DISTRIBUTE theorems distributed
        """
        self.logger.info("Distributing undetermined theorems across difficulty categories...")
        
        for repo in categorized_theorems:
            to_distribute = categorized_theorems[repo].get(DifficultyCategory.TO_DISTRIBUTE, [])
            
            if to_distribute:
                self.logger.info(f"Distributing {len(to_distribute)} theorems in {repo.name}")
                
                # Distribute evenly across Easy, Medium, Hard
                chunk_size = len(to_distribute) // 3
                categories = [DifficultyCategory.EASY, DifficultyCategory.MEDIUM, DifficultyCategory.HARD]
                
                for i, category in enumerate(categories):
                    start_idx = i * chunk_size
                    end_idx = start_idx + chunk_size if i < 2 else None  # Last chunk gets remainder
                    chunk = to_distribute[start_idx:end_idx]
                    categorized_theorems[repo][category].extend(chunk)
                
                # Remove the TO_DISTRIBUTE category
                del categorized_theorems[repo][DifficultyCategory.TO_DISTRIBUTE]
                
                self.logger.debug(f"Distribution complete for {repo.name}")
        
        return categorized_theorems
    
    def get_difficulty_summary(
        self, 
        categorized_theorems: Dict[Repository, Dict[DifficultyCategory, List[Tuple]]]
    ) -> Dict[str, Dict[str, int]]:
        """
        Generate a summary of difficulty distributions across repositories.
        
        Args:
            categorized_theorems: Categorized theorems by repository
            
        Returns:
            Summary dictionary with counts by repository and overall totals
        """
        summary = {"repositories": {}, "overall": {}}
        overall_counts = defaultdict(int)
        
        for repo, categories in categorized_theorems.items():
            repo_counts = {}
            for category, theorems in categories.items():
                count = len(theorems)
                repo_counts[category.value] = count
                overall_counts[category.value] += count
            summary["repositories"][repo.name] = repo_counts
        
        summary["overall"] = dict(overall_counts)
        return summary
    
    def process_repositories(
        self, 
        repositories: List[Repository]
    ) -> Tuple[Dict[Repository, Dict[DifficultyCategory, List[Tuple]]], Tuple[float, float]]:
        """
        Complete difficulty processing pipeline for repositories.
        
        Args:
            repositories: List of repositories to process
            
        Returns:
            Tuple of (categorized_theorems, percentile_thresholds)
        """
        # Calculate all difficulties
        difficulties_by_repo, all_difficulties = self.calculate_all_difficulties(repositories)
        
        # Calculate percentile thresholds
        percentiles = self.calculate_percentiles(all_difficulties)
        
        # Categorize theorems
        categorized_theorems = self.categorize_all_theorems(difficulties_by_repo, percentiles)
        
        # Distribute undetermined theorems
        categorized_theorems = self.distribute_undetermined_theorems(categorized_theorems)
        
        # Log summary
        summary = self.get_difficulty_summary(categorized_theorems)
        total_theorems = sum(summary["overall"].values())
        
        self.logger.info("Difficulty processing complete!")
        self.logger.info(f"Total theorems processed: {total_theorems}")
        for category, count in summary["overall"].items():
            percentage = (count / total_theorems) * 100 if total_theorems > 0 else 0
            self.logger.info(f"  {category}: {count} theorems ({percentage:.2f}%)")
        
        return categorized_theorems, percentiles 