# Enhanced LeanAgent: Advanced Theorem Proving with Integrated Reasoning

Enhanced LeanAgent extends the original [LeanAgent](https://arxiv.org/abs/2410.06209) with sophisticated theorem proving capabilities, conjecture generation, and advanced search algorithms integrated from crosstaalk.

## New Features

### 🧠 Enhanced Theorem Prover
- **Multiple Prover Modes**: Basic, Enhanced, Conjecture-guided, and Hybrid approaches
- **Advanced Tactics Integration**: Sophisticated reasoning capabilities
- **Adaptive Strategy Selection**: Automatically chooses the best approach based on problem complexity
- **Statistics Tracking**: Comprehensive performance monitoring

### 🎯 Conjecture Generation Engine
- **Domain-Specific Generation**: Graph theory, arithmetic, set theory, and more
- **Lean-Compatible Output**: Generates valid Lean 4 theorem statements
- **Template-Based System**: Extensible conjecture templates
- **Validation Pipeline**: Ensures generated conjectures are syntactically correct

### 🔍 Advanced Search Algorithms
- **Multiple Search Strategies**: A*, Beam Search, MCTS, Bidirectional, IDA*
- **Intelligent Heuristics**: Lean-specific evaluation functions
- **Adaptive Search**: Switches strategies based on problem characteristics
- **Performance Optimization**: Efficient search space exploration

## Quick Start

### Basic Usage

```python
from leanagent_enhanced import EnhancedLeanAgent, EnhancedLeanAgentConfig, ProverMode
from generator.model import FixedTacticGenerator
from search_algorithms import SearchStrategy

# Configure enhanced agent
config = EnhancedLeanAgentConfig(
    timeout=120,
    enable_enhanced_prover=True,
    enable_conjecture_generation=True,
    prover_mode=ProverMode.HYBRID,
    search_strategy=SearchStrategy.A_STAR
)

# Create tactic generator (use your preferred generator)
tac_gen = FixedTacticGenerator("simp", None)

# Initialize Enhanced LeanAgent
agent = EnhancedLeanAgent(tac_gen=tac_gen, config=config)

# Prove a theorem with enhancements
result = agent.prove_theorem(repo, theorem, pos, use_enhancement=True)

# Generate and analyze conjectures
conjectures, proofs = agent.generate_and_prove_conjectures(
    repo=repo, 
    domain="arithmetic", 
    count=5
)
```

### Running the Example

```bash
python enhanced_example.py
```

## Architecture

### Core Components

1. **EnhancedTheoremProver**: Main proving engine with multiple strategies
2. **ConjectureEngine**: Generates mathematical conjectures in Lean format  
3. **AdvancedSearchEngine**: Implements sophisticated search algorithms
4. **EnhancedLeanAgent**: Integrates all components with the original LeanAgent

### Integration with LeanAgent

The enhanced components are designed to seamlessly integrate with LeanAgent's existing architecture:

- **Preserves Original API**: All existing LeanAgent functionality remains unchanged
- **Optional Enhancements**: Enhanced features can be enabled/disabled via configuration
- **Backward Compatibility**: Falls back to original LeanAgent behavior when enhancements fail
- **Unified Statistics**: Comprehensive tracking across all components

## Configuration Options

### EnhancedLeanAgentConfig

```python
@dataclass
class EnhancedLeanAgentConfig:
    # Base configuration
    timeout: int = 300
    num_sampled_tactics: int = 64
    debug: bool = False
    
    # Enhancement toggles
    enable_enhanced_prover: bool = True
    enable_conjecture_generation: bool = True  
    enable_advanced_search: bool = True
    
    # Strategy selection
    prover_mode: ProverMode = ProverMode.HYBRID
    search_strategy: SearchStrategy = SearchStrategy.A_STAR
    
    # Performance tuning
    beam_width: int = 10
    use_dynamic_database: bool = True
```

### Prover Modes

- **BASIC**: Uses original LeanAgent prover only
- **ENHANCED**: Enhanced prover with advanced parameters
- **CONJECTURE**: Conjecture-guided proof search
- **HYBRID**: Tries multiple approaches automatically

### Search Strategies

- **A_STAR**: Optimal search with heuristics
- **BEAM_SEARCH**: Limited breadth-first exploration  
- **MCTS**: Monte Carlo Tree Search
- **BIDIRECTIONAL**: Forward and backward search
- **IDA_STAR**: Memory-efficient iterative deepening

## Conjecture Generation

### Supported Domains

- **Graph Theory**: Chromatic numbers, planarity, connectivity
- **Arithmetic**: Sum identities, divisibility, prime properties
- **Set Theory**: Inclusion, union/intersection properties
- **Lists**: Length, concatenation, membership

### Example Conjectures

**Graph Theory:**
```lean
theorem graph_chromatic_5_3 : 
  ∃ (G : SimpleGraph (Fin 5)), G.chromaticNumber ≥ 3
```

**Arithmetic:**
```lean
theorem sum_odds_eq_square (n : ℕ) (h : n ≤ 10) : 
  (Finset.range n).sum (fun i => 2 * i + 1) = n ^ 2
```

## Performance Features

### Statistics and Monitoring

```python
stats = agent.get_statistics()
print(f"Success rate: {stats['success_rate']:.2%}")
print(f"Enhanced proofs: {stats['enhanced_proofs']}")
print(f"Conjecture-guided: {stats['conjecture_guided_proofs']}")
```

### Session Management

```python
# Save session data
agent.save_session_data("session_results.json")

# Load previous session
previous_data = agent.load_session_data("session_results.json")
```

### Repository Analysis

```python
# Analyze theorem proving characteristics
analysis = agent.analyze_repository(repo, sample_size=20)
print(f"Generated {analysis['conjecture_analysis']['generated_count']} conjectures")
```

## Implementation Notes

### Lean 4 Integration

- Uses `lean_dojo` for Lean 4 interaction
- Compatible with existing LeanAgent tactic generators
- Proper handling of `TacticState` objects
- Integrates with LeanAgent's distributed proving

### Error Handling

- Graceful fallback to original LeanAgent on enhancement failures
- Comprehensive error logging and debugging support
- Timeout handling for long-running enhancements
- Validation of generated conjectures

### Performance Considerations

- Enhanced components add computational overhead
- Configurable timeouts and resource limits
- Optional components can be disabled for performance
- Efficient caching and memoization where applicable

## Future Enhancements

- [ ] Machine learning guided conjecture generation
- [ ] Integration with external theorem databases
- [ ] Advanced proof optimization techniques
- [ ] Multi-modal reasoning support
- [ ] Distributed conjecture generation

## Contributing

Contributions to Enhanced LeanAgent are welcome! Please ensure:

1. New features integrate cleanly with existing LeanAgent architecture
2. Comprehensive testing with various Lean repositories  
3. Documentation updates for new functionality
4. Backward compatibility preservation

## License

Enhanced LeanAgent maintains the same license as the original LeanAgent project.