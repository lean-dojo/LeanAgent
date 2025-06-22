"""
Curriculum Builder for LeanAgent

This module provides the main curriculum construction pipeline,
orchestrating difficulty calculation, categorization, and repository sorting.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from loguru import logger

from ..config.config import CurriculumConfig
from ..database.models import Repository
from .difficulty import DifficultyCalculator, DifficultyCategory
from .sorter import RepositorySorter, SortingStrategy
from .metrics import get_complexity_metric


class CurriculumBuilder:
    """
    Main curriculum builder for LeanAgent.
    
    This class orchestrates the complete curriculum learning pipeline:
    1. Difficulty calculation for all theorems
    2. Categorization into Easy/Medium/Hard groups
    3. Repository sorting based on curriculum strategy
    4. Curriculum statistics and persistence
    """
    
    def __init__(self, config: CurriculumConfig):
        """
        Initialize the curriculum builder.
        
        Args:
            config: Curriculum configuration settings
        """
        self.config = config
        self.logger = logger
        
        # Initialize components
        self.difficulty_calculator = DifficultyCalculator(
            complexity_metric=config.complexity_metric,
            easy_percentile=config.easy_percentile,
            hard_percentile=config.hard_percentile
        )
        
        # Map string to enum for sorting strategy
        strategy_mapping = {
            "easy_first": SortingStrategy.EASY_FIRST,
            "hard_first": SortingStrategy.HARD_FIRST,
            "balanced": SortingStrategy.BALANCED,
            "random": SortingStrategy.RANDOM,
            "mixed_difficulty": SortingStrategy.MIXED_DIFFICULTY
        }
        
        # Default to easy_first if not specified
        strategy = strategy_mapping.get("easy_first", SortingStrategy.EASY_FIRST)
        self.repository_sorter = RepositorySorter(strategy)
        
        self.logger.info(f"CurriculumBuilder initialized with {config.complexity_metric} metric")
        self.logger.info(f"Curriculum configuration: {config.num_repos} repos, "
                        f"percentiles: {config.easy_percentile}%/{config.hard_percentile}%")
    
    def build_curriculum(self, repositories: List[Repository]) -> Tuple[List[Repository], Dict[str, Any]]:
        """
        Build a complete curriculum from repositories.
        
        Args:
            repositories: List of repositories to include in curriculum
            
        Returns:
            Tuple of (ordered_repositories, curriculum_metadata)
        """
        self.logger.info(f"Building curriculum from {len(repositories)} repositories...")
        
        # Step 1: Calculate difficulties and categorize theorems
        categorized_theorems, percentiles = self.difficulty_calculator.process_repositories(repositories)
        
        # Step 2: Sort repositories according to strategy
        sorted_repositories = self.repository_sorter.sort_repositories(categorized_theorems)
        
        # Step 3: Generate curriculum statistics
        statistics = self.repository_sorter.get_curriculum_statistics(
            sorted_repositories, categorized_theorems
        )
        
        # Step 4: Create curriculum metadata
        metadata = {
            "curriculum_config": self.config.__dict__,
            "complexity_metric": self.difficulty_calculator.complexity_metric.name(),
            "percentile_thresholds": {
                "easy_threshold": percentiles[0],
                "hard_threshold": percentiles[1]
            },
            "sorting_strategy": self.repository_sorter.strategy.value,
            "statistics": statistics,
            "categorized_theorems_summary": self._summarize_categorized_theorems(categorized_theorems)
        }
        
        self.logger.info("Curriculum construction completed successfully!")
        self.logger.info(f"Final curriculum order: {len(sorted_repositories)} repositories")
        
        # Log top repositories
        for i, repo in enumerate(sorted_repositories[:5]):
            easy_count = len(categorized_theorems.get(repo, {}).get(DifficultyCategory.EASY, []))
            total_count = sum(len(theorems) for theorems in categorized_theorems.get(repo, {}).values())
            self.logger.info(f"  {i+1}. {repo.name}: {easy_count} easy / {total_count} total theorems")
        
        return sorted_repositories, metadata
    
    def _summarize_categorized_theorems(
        self, 
        categorized_theorems: Dict[Repository, Dict[DifficultyCategory, List[Tuple]]]
    ) -> Dict[str, Any]:
        """
        Create a summary of categorized theorems for metadata.
        
        Args:
            categorized_theorems: Categorized theorems by repository
            
        Returns:
            Summary dictionary with counts and percentages
        """
        summary = {"by_repository": {}, "overall_totals": {}}
        overall_counts = {category.value: 0 for category in DifficultyCategory}
        
        for repo, categories in categorized_theorems.items():
            repo_summary = {}
            repo_total = 0
            
            for category, theorems in categories.items():
                count = len(theorems)
                repo_summary[category.value] = count
                overall_counts[category.value] += count
                repo_total += count
            
            repo_summary["total"] = repo_total
            summary["by_repository"][repo.name] = repo_summary
        
        # Calculate percentages
        total_theorems = sum(overall_counts.values())
        overall_percentages = {}
        for category, count in overall_counts.items():
            percentage = (count / total_theorems) * 100 if total_theorems > 0 else 0
            overall_percentages[category] = {
                "count": count,
                "percentage": round(percentage, 2)
            }
        
        summary["overall_totals"] = overall_percentages
        summary["total_theorems"] = total_theorems
        
        return summary
    
    def save_curriculum(
        self, 
        sorted_repositories: List[Repository], 
        metadata: Dict[str, Any],
        output_path: Union[str, Path]
    ) -> None:
        """
        Save curriculum to a JSON file.
        
        Args:
            sorted_repositories: Ordered curriculum repositories
            metadata: Curriculum metadata
            output_path: Path to save the curriculum file
        """
        output_path = Path(output_path)
        
        # Prepare curriculum data for serialization
        curriculum_data = {
            "metadata": metadata,
            "repositories": [
                {
                    "position": i + 1,
                    "name": repo.name,
                    "url": repo.url,
                    "commit": repo.commit,
                    "lean_version": repo.lean_version,
                    "lean_dojo_version": repo.lean_dojo_version
                }
                for i, repo in enumerate(sorted_repositories)
            ]
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(curriculum_data, f, indent=2)
        
        self.logger.info(f"Curriculum saved to {output_path}")
        self.logger.info(f"Saved {len(sorted_repositories)} repositories with complete metadata")
    
    @classmethod
    def load_curriculum(cls, curriculum_path: Union[str, Path]) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Load curriculum from a JSON file.
        
        Args:
            curriculum_path: Path to the curriculum file
            
        Returns:
            Tuple of (repository_list, metadata)
        """
        curriculum_path = Path(curriculum_path)
        
        if not curriculum_path.exists():
            raise FileNotFoundError(f"Curriculum file not found: {curriculum_path}")
        
        with open(curriculum_path, 'r') as f:
            curriculum_data = json.load(f)
        
        repositories = curriculum_data.get("repositories", [])
        metadata = curriculum_data.get("metadata", {})
        
        logger.info(f"Loaded curriculum from {curriculum_path}")
        logger.info(f"Contains {len(repositories)} repositories")
        
        return repositories, metadata
    
    def update_repository_difficulties(self, repositories: List[Repository]) -> None:
        """
        Update difficulty ratings for theorems in repositories.
        
        This method calculates and stores difficulty ratings in theorem objects
        without full curriculum construction.
        
        Args:
            repositories: Repositories to update
        """
        self.logger.info(f"Updating difficulty ratings for {len(repositories)} repositories...")
        
        for repo in repositories:
            self.logger.debug(f"Processing theorems in {repo.name}")
            updated_count = 0
            
            for theorem in repo.get_all_theorems:
                difficulty = self.difficulty_calculator.calculate_difficulty(theorem)
                theorem.difficulty_rating = difficulty
                updated_count += 1
            
            self.logger.debug(f"Updated {updated_count} theorems in {repo.name}")
        
        self.logger.info("Difficulty rating update completed")
    
    def validate_curriculum(
        self, 
        sorted_repositories: List[Repository], 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate a curriculum for consistency and completeness.
        
        Args:
            sorted_repositories: The curriculum to validate
            metadata: Associated metadata
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "statistics": {}
        }
        
        # Check repository count
        expected_count = self.config.num_repos
        actual_count = len(sorted_repositories)
        
        if actual_count != expected_count:
            validation_results["warnings"].append(
                f"Repository count mismatch: expected {expected_count}, got {actual_count}"
            )
        
        # Check for duplicate repositories
        urls = [repo.url for repo in sorted_repositories]
        if len(urls) != len(set(urls)):
            validation_results["errors"].append("Duplicate repositories found in curriculum")
            validation_results["valid"] = False
        
        # Check theorem counts
        total_theorems = 0
        repos_with_no_theorems = []
        
        for repo in sorted_repositories:
            repo_theorem_count = repo.total_theorems
            total_theorems += repo_theorem_count
            
            if repo_theorem_count == 0:
                repos_with_no_theorems.append(repo.name)
        
        if repos_with_no_theorems:
            validation_results["warnings"].append(
                f"Repositories with no theorems: {repos_with_no_theorems}"
            )
        
        validation_results["statistics"] = {
            "total_repositories": actual_count,
            "total_theorems": total_theorems,
            "average_theorems_per_repo": total_theorems / actual_count if actual_count > 0 else 0,
            "repos_with_no_theorems": len(repos_with_no_theorems)
        }
        
        if validation_results["errors"]:
            validation_results["valid"] = False
        
        # Log validation results
        if validation_results["valid"]:
            self.logger.info("Curriculum validation passed")
        else:
            self.logger.error("Curriculum validation failed")
            for error in validation_results["errors"]:
                self.logger.error(f"  Error: {error}")
        
        for warning in validation_results["warnings"]:
            self.logger.warning(f"  Warning: {warning}")
        
        return validation_results 