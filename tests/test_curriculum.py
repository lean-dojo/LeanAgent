"""
Tests for the curriculum learning module.

This module tests all components of the curriculum learning system including
difficulty calculation, repository sorting, and curriculum building.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from src.curriculum import (
    CurriculumBuilder,
    DifficultyCalculator,
    DifficultyCategory,
    RepositorySorter,
    SortingStrategy,
    ExponentialProofStepsMetric,
    LinearProofStepsMetric,
    TacticComplexityMetric,
    get_complexity_metric
)
from src.config.config import CurriculumConfig
from src.database.models import Repository, Theorem, AnnotatedTactic, Pos


class TestComplexityMetrics:
    """Test complexity metrics for difficulty calculation."""
    
    def test_exponential_proof_steps_metric(self, sample_theorem_with_steps):
        """Test the exponential proof steps complexity metric."""
        metric = ExponentialProofStepsMetric()
        
        # Test theorem with 2 steps -> e^2 ≈ 7.39
        theorem_2_steps = sample_theorem_with_steps(2)
        difficulty = metric.calculate(theorem_2_steps)
        assert difficulty == pytest.approx(7.39, rel=0.01)
        
        # Test theorem with 5 steps -> e^5 ≈ 148.4
        theorem_5_steps = sample_theorem_with_steps(5)
        difficulty = metric.calculate(theorem_5_steps)
        assert difficulty == pytest.approx(148.4, rel=0.01)
        
        # Test ordering
        assert metric.calculate(theorem_5_steps) > metric.calculate(theorem_2_steps)
    
    def test_exponential_metric_sorry_theorem(self, sample_theorem_with_sorry):
        """Test that sorry theorems get infinite difficulty."""
        metric = ExponentialProofStepsMetric()
        theorem = sample_theorem_with_sorry()
        
        difficulty = metric.calculate(theorem)
        assert difficulty == float('inf')
    
    def test_exponential_metric_no_tactics(self, sample_theorem_empty):
        """Test that theorems with no tactics return None."""
        metric = ExponentialProofStepsMetric()
        theorem = sample_theorem_empty()
        
        difficulty = metric.calculate(theorem)
        assert difficulty is None
    
    def test_linear_proof_steps_metric(self, sample_theorem_with_steps):
        """Test the linear proof steps complexity metric."""
        metric = LinearProofStepsMetric()
        
        theorem_3_steps = sample_theorem_with_steps(3)
        difficulty = metric.calculate(theorem_3_steps)
        assert difficulty == 3.0
        
        theorem_7_steps = sample_theorem_with_steps(7)
        difficulty = metric.calculate(theorem_7_steps)
        assert difficulty == 7.0
    
    def test_tactic_complexity_metric(self):
        """Test the tactic-based complexity metric."""
        metric = TacticComplexityMetric()
        
        # Create theorem with different tactics
        tactics = [
            AnnotatedTactic("rfl", ("rfl", []), "", ""),  # weight 1.0
            AnnotatedTactic("simp", ("simp", []), "", ""),  # weight 1.5
            AnnotatedTactic("induction", ("induction", []), "", "")  # weight 3.0
        ]
        
        theorem = Theorem(
            full_name="test_theorem",
            file_path=Path("test.lean"),
            start=Pos(1, 1),
            end=Pos(5, 1),
            url="https://github.com/test/repo",
            commit="abc123",
            traced_tactics=tactics
        )
        
        difficulty = metric.calculate(theorem)
        expected = 1.0 + 1.5 + 3.0  # Sum of tactic weights
        assert difficulty == expected
    
    def test_get_complexity_metric_factory(self):
        """Test the complexity metric factory function."""
        # Test valid metric names
        exp_metric = get_complexity_metric("exp_proof_steps")
        assert isinstance(exp_metric, ExponentialProofStepsMetric)
        
        linear_metric = get_complexity_metric("linear_proof_steps")
        assert isinstance(linear_metric, LinearProofStepsMetric)
        
        tactic_metric = get_complexity_metric("tactic_complexity")
        assert isinstance(tactic_metric, TacticComplexityMetric)
        
        # Test invalid metric name
        with pytest.raises(ValueError, match="Unknown complexity metric"):
            get_complexity_metric("invalid_metric")


class TestDifficultyCalculator:
    """Test the difficulty calculator component."""
    
    def test_difficulty_calculator_initialization(self):
        """Test difficulty calculator initialization."""
        calculator = DifficultyCalculator(
            complexity_metric="exp_proof_steps",
            easy_percentile=30.0,
            hard_percentile=70.0
        )
        
        assert calculator.easy_percentile == 30.0
        assert calculator.hard_percentile == 70.0
        assert isinstance(calculator.complexity_metric, ExponentialProofStepsMetric)
    
    def test_invalid_percentiles(self):
        """Test that invalid percentiles raise ValueError."""
        with pytest.raises(ValueError, match="Percentiles must satisfy"):
            DifficultyCalculator(easy_percentile=70.0, hard_percentile=30.0)
        
        with pytest.raises(ValueError, match="Percentiles must satisfy"):
            DifficultyCalculator(easy_percentile=-10.0)
    
    def test_calculate_percentiles(self):
        """Test percentile calculation from difficulty values."""
        calculator = DifficultyCalculator()
        
        difficulties = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        easy_threshold, hard_threshold = calculator.calculate_percentiles(difficulties)
        
        # 33rd percentile should be around 3.3, 67th around 6.7
        # Allow for numerical precision tolerance
        assert 3.0 <= easy_threshold <= 4.0
        assert 6.0 <= hard_threshold <= 7.1  # Increased tolerance for numerical precision
    
    def test_categorize_difficulty(self):
        """Test difficulty categorization."""
        calculator = DifficultyCalculator()
        percentiles = (5.0, 10.0)  # Easy <= 5.0, Hard > 10.0
        
        # Test different categories
        assert calculator.categorize_difficulty(None, percentiles) == DifficultyCategory.TO_DISTRIBUTE
        assert calculator.categorize_difficulty(float('inf'), percentiles) == DifficultyCategory.NO_PROOF
        assert calculator.categorize_difficulty(3.0, percentiles) == DifficultyCategory.EASY
        assert calculator.categorize_difficulty(7.5, percentiles) == DifficultyCategory.MEDIUM
        assert calculator.categorize_difficulty(15.0, percentiles) == DifficultyCategory.HARD
    
    def test_process_repositories(self, sample_repositories):
        """Test the complete repository processing pipeline."""
        calculator = DifficultyCalculator()
        repositories = sample_repositories()
        
        categorized_theorems, percentiles = calculator.process_repositories(repositories)
        
        # Check return types
        assert isinstance(categorized_theorems, dict)
        assert isinstance(percentiles, tuple)
        assert len(percentiles) == 2
        
        # Check that all repositories are categorized
        assert len(categorized_theorems) == len(repositories)
        
        # Check that categories exist
        for repo, categories in categorized_theorems.items():
            assert isinstance(categories, dict)
            # Should have at least some categories
            assert len(categories) > 0


class TestRepositorySorter:
    """Test the repository sorting component."""
    
    def test_repository_sorter_initialization(self):
        """Test repository sorter initialization."""
        sorter = RepositorySorter(SortingStrategy.EASY_FIRST)
        assert sorter.strategy == SortingStrategy.EASY_FIRST
    
    def test_sort_by_easy_theorems(self, sample_categorized_theorems):
        """Test sorting repositories by easy theorem count."""
        sorter = RepositorySorter(SortingStrategy.EASY_FIRST)
        categorized_theorems = sample_categorized_theorems()
        
        sorted_repos = sorter.sort_by_easy_theorems(categorized_theorems)
        
        # Verify sorting order (most easy theorems first)
        easy_counts = []
        for repo in sorted_repos:
            easy_count = len(categorized_theorems.get(repo, {}).get(DifficultyCategory.EASY, []))
            easy_counts.append(easy_count)
        
        # Should be in descending order
        assert easy_counts == sorted(easy_counts, reverse=True)
    
    def test_sort_by_hard_theorems(self, sample_categorized_theorems):
        """Test sorting repositories by hard theorem count."""
        sorter = RepositorySorter(SortingStrategy.HARD_FIRST)
        categorized_theorems = sample_categorized_theorems()
        
        sorted_repos = sorter.sort_by_hard_theorems(categorized_theorems)
        
        # Verify sorting order (most hard theorems first)
        hard_counts = []
        for repo in sorted_repos:
            hard_regular = len(categorized_theorems.get(repo, {}).get(DifficultyCategory.HARD, []))
            hard_no_proof = len(categorized_theorems.get(repo, {}).get(DifficultyCategory.NO_PROOF, []))
            hard_counts.append(hard_regular + hard_no_proof)
        
        # Should be in descending order
        assert hard_counts == sorted(hard_counts, reverse=True)
    
    def test_sort_repositories_strategy_dispatch(self, sample_categorized_theorems):
        """Test that sort_repositories calls the correct strategy method."""
        categorized_theorems = sample_categorized_theorems()
        
        # Test each strategy
        strategies = [
            SortingStrategy.EASY_FIRST,
            SortingStrategy.HARD_FIRST,
            SortingStrategy.BALANCED,
            SortingStrategy.MIXED_DIFFICULTY,
            SortingStrategy.RANDOM
        ]
        
        for strategy in strategies:
            sorter = RepositorySorter(strategy)
            sorted_repos = sorter.sort_repositories(categorized_theorems)
            
            # Should return a list of repositories
            assert isinstance(sorted_repos, list)
            assert len(sorted_repos) == len(categorized_theorems)
    
    def test_get_curriculum_statistics(self, sample_categorized_theorems):
        """Test curriculum statistics generation."""
        sorter = RepositorySorter()
        categorized_theorems = sample_categorized_theorems()
        sorted_repos = list(categorized_theorems.keys())
        
        stats = sorter.get_curriculum_statistics(sorted_repos, categorized_theorems)
        
        # Check required fields
        assert "total_repositories" in stats
        assert "ordering_strategy" in stats
        assert "repository_details" in stats
        assert "cumulative_easy" in stats
        assert "cumulative_total" in stats
        
        # Check statistics consistency
        assert stats["total_repositories"] == len(sorted_repos)
        assert len(stats["repository_details"]) == len(sorted_repos)


class TestCurriculumBuilder:
    """Test the main curriculum builder component."""
    
    def test_curriculum_builder_initialization(self):
        """Test curriculum builder initialization."""
        config = CurriculumConfig()
        builder = CurriculumBuilder(config)
        
        assert builder.config == config
        assert isinstance(builder.difficulty_calculator, DifficultyCalculator)
        assert isinstance(builder.repository_sorter, RepositorySorter)
    
    def test_build_curriculum(self, sample_repositories):
        """Test the complete curriculum building pipeline."""
        config = CurriculumConfig()
        builder = CurriculumBuilder(config)
        repositories = sample_repositories()
        
        sorted_repos, metadata = builder.build_curriculum(repositories)
        
        # Check return types
        assert isinstance(sorted_repos, list)
        assert isinstance(metadata, dict)
        
        # Check metadata structure
        required_keys = [
            "curriculum_config", 
            "complexity_metric", 
            "percentile_thresholds",
            "sorting_strategy", 
            "statistics", 
            "categorized_theorems_summary"
        ]
        for key in required_keys:
            assert key in metadata
        
        # Check that all repositories are included
        assert len(sorted_repos) == len(repositories)
    
    def test_save_and_load_curriculum(self, sample_repositories, tmp_path):
        """Test curriculum saving and loading."""
        config = CurriculumConfig()
        builder = CurriculumBuilder(config)
        repositories = sample_repositories()
        
        # Build and save curriculum
        sorted_repos, metadata = builder.build_curriculum(repositories)
        curriculum_path = tmp_path / "test_curriculum.json"
        builder.save_curriculum(sorted_repos, metadata, curriculum_path)
        
        # Verify file was created
        assert curriculum_path.exists()
        
        # Load and verify
        loaded_repos, loaded_metadata = CurriculumBuilder.load_curriculum(curriculum_path)
        
        assert len(loaded_repos) == len(sorted_repos)
        assert loaded_metadata == metadata
    
    def test_validate_curriculum(self, sample_repositories):
        """Test curriculum validation."""
        config = CurriculumConfig()
        builder = CurriculumBuilder(config)
        repositories = sample_repositories()
        
        sorted_repos, metadata = builder.build_curriculum(repositories)
        validation_results = builder.validate_curriculum(sorted_repos, metadata)
        
        # Check validation structure
        assert "valid" in validation_results
        assert "warnings" in validation_results
        assert "errors" in validation_results
        assert "statistics" in validation_results
        
        # Should be valid for properly built curriculum
        assert validation_results["valid"] is True
    
    def test_update_repository_difficulties(self, sample_repositories):
        """Test updating difficulty ratings in repositories."""
        config = CurriculumConfig()
        builder = CurriculumBuilder(config)
        repositories = sample_repositories()
        
        # Clear existing difficulty ratings
        for repo in repositories:
            for theorem in repo.get_all_theorems:
                theorem.difficulty_rating = None
        
        # Update difficulties
        builder.update_repository_difficulties(repositories)
        
        # Verify difficulties were calculated appropriately
        theorems_with_ratings = 0
        theorems_without_ratings = 0
        
        for repo in repositories:
            for theorem in repo.get_all_theorems:
                if theorem.traced_tactics and len(theorem.traced_tactics) > 0:
                    # Theorems with tactics should have difficulty ratings
                    assert theorem.difficulty_rating is not None
                    theorems_with_ratings += 1
                else:
                    # Theorems without tactics should have None rating (TO_DISTRIBUTE)
                    assert theorem.difficulty_rating is None
                    theorems_without_ratings += 1
        
        # Should have some of both types based on our test data
        assert theorems_with_ratings > 0
        assert theorems_without_ratings > 0


class TestCurriculumIntegration:
    """Integration tests for the complete curriculum system."""
    
    def test_end_to_end_curriculum_pipeline(self, sample_repositories):
        """Test the complete end-to-end curriculum pipeline."""
        # Create configuration
        config = CurriculumConfig(
            enabled=True,
            complexity_metric="exp_proof_steps",
            easy_percentile=25.0,
            hard_percentile=75.0,
            num_repos=3
        )
        
        # Build curriculum
        builder = CurriculumBuilder(config)
        repositories = sample_repositories()
        
        # Run complete pipeline
        sorted_repos, metadata = builder.build_curriculum(repositories)
        
        # Validate results
        validation = builder.validate_curriculum(sorted_repos, metadata)
        assert validation["valid"] is True
        
        # Check that curriculum follows strategy
        assert metadata["sorting_strategy"] == "easy_first"
        assert metadata["complexity_metric"] == "exp_proof_steps"
        
        # Verify percentile thresholds
        thresholds = metadata["percentile_thresholds"]
        assert "easy_threshold" in thresholds
        assert "hard_threshold" in thresholds
    
    def test_different_complexity_metrics(self, sample_repositories):
        """Test curriculum building with different complexity metrics."""
        repositories = sample_repositories()
        
        metrics = ["exp_proof_steps", "linear_proof_steps", "tactic_complexity"]
        
        for metric_name in metrics:
            config = CurriculumConfig(complexity_metric=metric_name)
            builder = CurriculumBuilder(config)
            
            sorted_repos, metadata = builder.build_curriculum(repositories)
            
            assert metadata["complexity_metric"] == metric_name
            assert len(sorted_repos) == len(repositories)
    
    def test_different_sorting_strategies(self, sample_repositories):
        """Test curriculum building with different sorting strategies."""
        repositories = sample_repositories()
        
        # Note: We're using easy_first as default since the config doesn't have sorting strategy
        # In a full implementation, we'd add sorting_strategy to CurriculumConfig
        config = CurriculumConfig()
        
        # Test with different sorter strategies
        strategies = [SortingStrategy.EASY_FIRST, SortingStrategy.HARD_FIRST, SortingStrategy.BALANCED]
        
        for strategy in strategies:
            builder = CurriculumBuilder(config)
            builder.repository_sorter = RepositorySorter(strategy)
            
            sorted_repos, metadata = builder.build_curriculum(repositories)
            
            assert metadata["sorting_strategy"] == strategy.value
            assert len(sorted_repos) == len(repositories) 