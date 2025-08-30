"""Example usage of Enhanced LeanAgent with integrated theorem proving capabilities."""

import os
import sys
from pathlib import Path

# Add LeanAgent to path
sys.path.append(str(Path(__file__).parent))

from lean_dojo import LeanGitRepo, Theorem, Pos
from loguru import logger

# Import enhanced components
from leanagent_enhanced import EnhancedLeanAgent, EnhancedLeanAgentConfig, ProverMode
from generator.model import FixedTacticGenerator
from search_algorithms import SearchStrategy


def main():
    """Demonstrate Enhanced LeanAgent capabilities."""
    
    logger.info("Enhanced LeanAgent Example")
    
    # Configure enhanced agent
    config = EnhancedLeanAgentConfig(
        timeout=60,
        num_sampled_tactics=32,
        enable_enhanced_prover=True,
        enable_conjecture_generation=True,
        enable_advanced_search=True,
        prover_mode=ProverMode.HYBRID,
        search_strategy=SearchStrategy.A_STAR,
        beam_width=5
    )
    
    # Create a simple tactic generator (for demonstration)
    tac_gen = FixedTacticGenerator("simp", None)
    
    # Initialize Enhanced LeanAgent
    enhanced_agent = EnhancedLeanAgent(
        tac_gen=tac_gen,
        config=config,
        num_workers=1,
        num_gpus=0
    )
    
    logger.info("Enhanced LeanAgent initialized successfully")
    
    # Example: Generate conjectures
    if enhanced_agent.conjecture_engine:
        logger.info("Generating sample conjectures...")
        
        # Generate conjectures in different domains
        graph_conjectures = enhanced_agent.conjecture_engine.generate_conjectures(
            domain="graphs", 
            count=3,
            max_bounds={"n": 8, "k": 4}
        )
        
        arithmetic_conjectures = enhanced_agent.conjecture_engine.generate_conjectures(
            domain="arithmetic",
            count=3,
            max_bounds={"n": 15}
        )
        
        logger.info(f"Generated {len(graph_conjectures)} graph conjectures")
        logger.info(f"Generated {len(arithmetic_conjectures)} arithmetic conjectures")
        
        # Display sample conjectures
        for i, conjecture in enumerate(graph_conjectures[:2]):
            logger.info(f"Graph conjecture {i+1}: {conjecture.natural_language}")
            if conjecture.lean_statement:
                logger.info(f"Lean statement: {conjecture.lean_statement}")
        
        for i, conjecture in enumerate(arithmetic_conjectures[:2]):
            logger.info(f"Arithmetic conjecture {i+1}: {conjecture.natural_language}")
            if conjecture.lean_statement:
                logger.info(f"Lean statement: {conjecture.lean_statement}")
    
    # Display statistics
    stats = enhanced_agent.get_statistics()
    logger.info("Enhanced LeanAgent Statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            logger.info(f"{key}:")
            for subkey, subvalue in value.items():
                logger.info(f"  {subkey}: {subvalue}")
        else:
            logger.info(f"{key}: {value}")
    
    logger.info("Example completed successfully")


if __name__ == "__main__":
    main()