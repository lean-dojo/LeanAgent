"""
Pytest configuration and fixtures for LeanAgent tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
import os
import json

# Set up test environment variables
os.environ["GITHUB_ACCESS_TOKEN"] = "test_token"
os.environ["RAID_DIR"] = "/tmp/test_leanagent"

from src.config import LeanAgentConfig, ConfigValidator, PathManager


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp(prefix="leanagent_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_config(temp_dir):
    """Create a test configuration."""
    os.environ["RAID_DIR"] = str(temp_dir)
    config = LeanAgentConfig.from_env()
    return config


@pytest.fixture
def mock_lean_repo():
    """Create a mock Lean repository."""
    repo = Mock()
    repo.url = "https://github.com/test/repo"
    repo.commit = "abc123def456"
    repo.name = "test_repo"
    return repo


@pytest.fixture
def sample_training_data():
    """Create sample training data."""
    return {
        "train": [
            {
                "theorem": "theorem test1 : True := trivial",
                "proof_steps": ["trivial"],
                "premises": ["True.intro"]
            }
        ],
        "val": [
            {
                "theorem": "theorem test2 : False ∨ True := Or.inr trivial", 
                "proof_steps": ["right", "trivial"],
                "premises": ["Or.inr", "True.intro"]
            }
        ],
        "test": [
            {
                "theorem": "theorem test3 : 1 + 1 = 2 := rfl",
                "proof_steps": ["rfl"],
                "premises": ["Eq.refl"]
            }
        ]
    }


@pytest.fixture
def mock_github_api():
    """Mock GitHub API responses."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "items": [
            {
                "full_name": "test/repo1",
                "html_url": "https://github.com/test/repo1",
                "language": "Lean",
                "stargazers_count": 100
            },
            {
                "full_name": "test/repo2", 
                "html_url": "https://github.com/test/repo2",
                "language": "Lean",
                "stargazers_count": 50
            }
        ]
    }
    mock_response.status_code = 200
    return mock_response


@pytest.fixture
def mock_pytorch_lightning():
    """Mock PyTorch Lightning components."""
    mock_trainer = Mock()
    mock_model = Mock()
    mock_datamodule = Mock()
    
    return {
        "trainer": mock_trainer,
        "model": mock_model,
        "datamodule": mock_datamodule
    }


@pytest.fixture
def mock_distributed_prover():
    """Mock the distributed prover."""
    prover = Mock()
    prover.search.return_value = Mock(
        success=True,
        proof="trivial",
        search_time=1.5,
        num_expansions=10
    )
    return prover


@pytest.fixture(autouse=True)
def cleanup_environment():
    """Clean up environment variables after each test."""
    yield
    # Restore original environment or clean up test-specific variables
    test_vars = [k for k in os.environ if k.startswith("LEANAGENT_")]
    for var in test_vars:
        del os.environ[var]


class TestDataFactory:
    """Factory for creating test data objects."""
    
    @staticmethod
    def create_config(temp_dir=None):
        """Create a test configuration."""
        if temp_dir:
            os.environ["RAID_DIR"] = str(temp_dir)
        os.environ["GITHUB_ACCESS_TOKEN"] = "test_token"
        return LeanAgentConfig()
    
    @staticmethod
    def create_mock_theorem(name="test_theorem", difficulty=None):
        """Create a mock theorem object."""
        theorem = Mock()
        theorem.full_name = name
        theorem.file_path = "Test.lean"
        theorem.difficulty_rating = difficulty or 2.5
        theorem.url = "https://github.com/test/repo"
        theorem.commit = "abc123"
        return theorem
    
    @staticmethod
    def create_mock_repository(name="test_repo", num_theorems=5):
        """Create a mock repository object."""
        repo = Mock()
        repo.name = name
        repo.url = f"https://github.com/test/{name}"
        repo.commit = "abc123def456"
        repo.lean_version = "4.6.0"
        repo.total_theorems = num_theorems
        repo.proven_theorems = [
            TestDataFactory.create_mock_theorem(f"theorem_{i}")
            for i in range(num_theorems)
        ]
        return repo


@pytest.fixture
def test_data_factory():
    """Provide access to test data factory."""
    return TestDataFactory


# Curriculum learning specific fixtures

@pytest.fixture
def sample_theorem_with_steps():
    """Create a theorem with specified number of proof steps."""
    def _create_theorem(num_steps):
        from src.database.models import Theorem, AnnotatedTactic, Pos
        from pathlib import Path
        
        tactics = [
            AnnotatedTactic(f"tactic_{i}", (f"tactic_{i}", []), "", "")
            for i in range(num_steps)
        ]
        
        return Theorem(
            full_name=f"test_theorem_{num_steps}_steps",
            file_path=Path("test.lean"),
            start=Pos(1, 1),
            end=Pos(5, 1),
            url="https://github.com/test/repo",
            commit="abc123",
            traced_tactics=tactics
        )
    return _create_theorem


@pytest.fixture
def sample_theorem_with_sorry():
    """Create a theorem with sorry tactics."""
    def _create_theorem():
        from src.database.models import Theorem, AnnotatedTactic, Pos
        from pathlib import Path
        
        tactics = [
            AnnotatedTactic("sorry", ("sorry", []), "", "")
        ]
        
        return Theorem(
            full_name="test_sorry_theorem",
            file_path=Path("test.lean"),
            start=Pos(1, 1),
            end=Pos(5, 1),
            url="https://github.com/test/repo", 
            commit="abc123",
            traced_tactics=tactics
        )
    return _create_theorem


@pytest.fixture 
def sample_theorem_empty():
    """Create a theorem with no tactics."""
    def _create_theorem():
        from src.database.models import Theorem, Pos
        from pathlib import Path
        
        return Theorem(
            full_name="test_empty_theorem",
            file_path=Path("test.lean"),
            start=Pos(1, 1),
            end=Pos(5, 1),
            url="https://github.com/test/repo",
            commit="abc123",
            traced_tactics=[]
        )
    return _create_theorem


@pytest.fixture
def sample_repositories():
    """Create sample repositories with varying theorem difficulties."""
    def _create_repositories():
        from src.database.models import Repository, Theorem, AnnotatedTactic, Pos
        from pathlib import Path
        
        repositories = []
        
        for i in range(3):
            repo = Repository(
                url=f"https://github.com/test/repo_{i}",
                name=f"test_repo_{i}",
                commit=f"commit_{i}",
                lean_version="4.8.0",
                lean_dojo_version="2.2.0",
                metadata={"test": f"repo_{i}"}
            )
            
            # Add theorems with different difficulties
            for j in range(5):
                # Vary number of tactics to create different difficulties
                num_tactics = (i * 2 + j) % 6  # 0-5 tactics
                tactics = [
                    AnnotatedTactic(f"tactic_{k}", (f"tactic_{k}", []), "", "")
                    for k in range(num_tactics)
                ]
                
                theorem = Theorem(
                    full_name=f"theorem_{i}_{j}",
                    file_path=Path(f"Test_{i}.lean"),
                    start=Pos(j * 5 + 1, 1),
                    end=Pos(j * 5 + 5, 1),
                    url=repo.url,
                    commit=repo.commit,
                    traced_tactics=tactics
                )
                
                repo.proven_theorems.append(theorem)
            
            repositories.append(repo)
        
        return repositories
    return _create_repositories


@pytest.fixture
def sample_categorized_theorems():
    """Create sample categorized theorems for testing."""
    def _create_categorized():
        from src.database.models import Repository
        from src.curriculum.difficulty import DifficultyCategory
        
        repositories = []
        for i in range(2):
            repo = Repository(
                url=f"https://github.com/test/repo_{i}",
                name=f"test_repo_{i}",
                commit=f"commit_{i}",
                lean_version="4.8.0",
                lean_dojo_version="2.2.0",
                metadata={}
            )
            repositories.append(repo)
        
        # Create mock categorized theorems
        categorized = {}
        for i, repo in enumerate(repositories):
            categorized[repo] = {
                DifficultyCategory.EASY: [
                    (f"easy_theorem_{j}", "test.lean", (1, 1), (5, 1), 2.0)
                    for j in range(3 + i)  # repo_0: 3 easy, repo_1: 4 easy
                ],
                DifficultyCategory.MEDIUM: [
                    (f"medium_theorem_{j}", "test.lean", (6, 1), (10, 1), 7.0)
                    for j in range(2)
                ],
                DifficultyCategory.HARD: [
                    (f"hard_theorem_{j}", "test.lean", (11, 1), (15, 1), 15.0)
                    for j in range(1 + i)  # repo_0: 1 hard, repo_1: 2 hard
                ]
            }
        
        return categorized
    return _create_categorized 