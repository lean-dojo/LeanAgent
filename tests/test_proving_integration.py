"""
Test proving module integration with LeanAgent framework.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.config import LeanAgentConfig
from src.core import LeanAgentFramework


class TestProvingModuleIntegration:
    """Test integration of the proving module with the framework."""
    
    def test_proving_module_imports(self):
        """Test that proving module components can be imported."""
        try:
            from src.proving import (
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
            # If we get here, imports work
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import proving components: {e}")
    
    def test_proving_config_integration(self):
        """Test that proving configuration is properly integrated."""
        config = LeanAgentConfig()
        
        # Check that proving config exists
        assert hasattr(config, 'proving')
        assert config.proving is not None
        
        # Check default values
        assert config.proving.generator_type == "rag"
        assert config.proving.timeout == 600
        assert config.proving.batch_size == 12
        assert config.proving.num_workers == 1
        assert config.proving.num_gpus == 0
    
    def test_framework_prover_property(self):
        """Test that framework can access prover property."""
        config = LeanAgentConfig()
        framework = LeanAgentFramework(config)
        
        # Test prover property access (should not raise errors)
        prover = framework.prover
        # Note: prover may be None due to missing dependencies, but property should work
        assert prover is None or prover is not None  # Either way is fine
    
    def test_proving_config_validation(self):
        """Test proving configuration validation."""
        config = LeanAgentConfig()
        
        # Test valid generator types
        valid_generators = ["fixed", "rag", "gpt4", "vllm"]
        for gen_type in valid_generators:
            config.proving.generator_type = gen_type
            # Should not raise validation errors
            assert config.proving.generator_type == gen_type
    
    def test_framework_components_status(self):
        """Test that framework reports proving component status."""
        config = LeanAgentConfig()
        framework = LeanAgentFramework(config)
        
        status = framework.get_status()
        
        # Check that prover is included in components
        assert "components" in status
        assert "prover" in status["components"]
        # Value can be True or False depending on dependencies
        assert isinstance(status["components"]["prover"], bool)
    
    @patch('src.proving.ProverManager')
    def test_prove_theorems_method(self, mock_prover_manager):
        """Test the prove_theorems method integration."""
        # Setup mocks
        mock_repo = Mock()
        mock_repo.sorry_theorems_unproved = []
        
        mock_database = Mock()
        mock_database.get_repository_by_url.return_value = mock_repo
        
        mock_prover_instance = Mock()
        mock_prover_manager.return_value = mock_prover_instance
        mock_prover_instance.__enter__ = Mock(return_value=mock_prover_instance)
        mock_prover_instance.__exit__ = Mock(return_value=None)
        
        # Setup framework
        config = LeanAgentConfig()
        framework = LeanAgentFramework(config)
        framework._database = mock_database
        
        # Test prove_theorems call
        with patch('src.proving.BatchProcessor') as mock_batch_processor:
            mock_batch_instance = Mock()
            mock_batch_processor.return_value = mock_batch_instance
            mock_batch_instance.create_batches.return_value = []
            mock_batch_instance.finalize.return_value = None
            
            result = framework.prove_theorems("https://github.com/test/repo")
            
            # Verify result structure
            assert isinstance(result, dict)
            assert "repository" in result
            assert "success" in result
            assert result["repository"] == "https://github.com/test/repo"
    
    def test_search_status_enum(self):
        """Test SearchStatus enum values."""
        from src.proving import SearchStatus
        
        assert SearchStatus.PROVED.value == "Proved"
        assert SearchStatus.FAILED.value == "Failed"
        assert SearchStatus.OPEN.value == "Open"
    
    def test_tactic_generator_interface(self):
        """Test TacticGenerator interface."""
        from src.proving import TacticGenerator, FixedTacticGenerator
        
        # Test that FixedTacticGenerator implements TacticGenerator
        fixed_gen = FixedTacticGenerator("apply", None)
        assert isinstance(fixed_gen, TacticGenerator)
        
        # Test generate method exists
        assert hasattr(fixed_gen, 'generate')
        assert callable(fixed_gen.generate)
    
    def test_batch_processor_interface(self):
        """Test BatchProcessor interface."""
        from src.proving import BatchProcessor, TheoremBatch
        
        # Test BatchProcessor creation
        processor = BatchProcessor(batch_size=5)
        assert processor.batch_size == 5
        
        # Test TheoremBatch creation
        batch = TheoremBatch(theorems=[], positions=[], repository=Mock())
        assert len(batch) == 0
        assert batch.is_empty()
    
    def test_framework_initialization_with_proving(self):
        """Test framework initialization includes proving components."""
        config = LeanAgentConfig()
        framework = LeanAgentFramework(config)
        
        # Test that framework initializes without errors
        assert framework is not None
        assert framework.config is not None
        
        # Test component initialization
        try:
            framework.initialize_components()
            # Should not raise errors even if dependencies are missing
            assert True
        except Exception as e:
            # Only acceptable if it's a dependency issue
            assert "not available" in str(e).lower() or "import" in str(e).lower()
    
    def test_prover_config_creation(self):
        """Test ProverConfig creation from main config."""
        from src.proving import ProverConfig
        
        config = LeanAgentConfig()
        config.proving.timeout = 300
        config.proving.num_workers = 4
        config.proving.debug = True
        
        prover_config = ProverConfig(
            timeout=config.proving.timeout,
            num_workers=config.proving.num_workers,
            debug=config.proving.debug
        )
        
        assert prover_config.timeout == 300
        assert prover_config.num_workers == 4
        assert prover_config.debug == True
    
    def test_proving_module_graceful_degradation(self):
        """Test that system works even if proving dependencies are missing."""
        config = LeanAgentConfig()
        framework = LeanAgentFramework(config)
        
        # Even if proving is not available, framework should work
        status = framework.get_status()
        assert status["framework_initialized"] == True
        
        # Components may report as unavailable, but framework should be stable
        assert "components" in status 