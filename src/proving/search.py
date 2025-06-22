"""
Best-First Search for Theorem Proving

This module implements the core search algorithm for theorem proving using
best-first search with tactic generation. It provides the main proving
logic that explores the proof search space systematically.

The search algorithm maintains a priority queue of proof states and uses
tactic generators to propose next steps, building a search tree until
a proof is found or resources are exhausted.
"""

import time
import asyncio
import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any, Dict
from loguru import logger

from .tree import SearchTree, SearchStatus, InternalNode, ProofNode, ErrorNode
from .generators import TacticGenerator

try:
    from lean_dojo import (
        Pos,
        Dojo,
        Theorem,
        LeanGitRepo,
        TacticState,
        LeanError,
        TimeoutError,
        ProofFinished,
        ProofGivenUp,
        DojoInitError,
        DojoCrashError,
        DojoHardTimeoutError,
    )
    LEAN_DOJO_AVAILABLE = True
except (ImportError, Exception) as e:
    logger.warning(f"LeanDojo not available - search will use mock implementations: {e}")
    LEAN_DOJO_AVAILABLE = False
    # Mock types for testing
    Pos = Any
    Dojo = Any
    Theorem = Any
    LeanGitRepo = Any
    TacticState = Any
    LeanError = Exception
    TimeoutError = Exception
    ProofFinished = Any
    ProofGivenUp = Exception
    DojoInitError = Exception
    DojoCrashError = Exception
    DojoHardTimeoutError = Exception


@dataclass(frozen=True)
class SearchResult:
    """The result of attempting to prove a theorem."""
    
    theorem: Theorem
    status: SearchStatus
    proof: Optional[List[str]]
    
    # Search statistics
    actor_time: float
    environment_time: float
    total_time: float
    num_total_nodes: int
    num_searched_nodes: int
    
    def __str__(self) -> str:
        return (f"SearchResult(theorem={self.theorem}, status={self.status}, "
                f"proof_length={len(self.proof) if self.proof else 0}, "
                f"total_time={self.total_time:.2f}s, nodes={self.num_total_nodes})")


class BestFirstSearchProver:
    """
    A theorem prover using best-first search with tactic generation.
    
    This prover explores the proof search space by maintaining a priority queue
    of proof states, using tactic generators to propose tactics, and building
    a search tree until a proof is found or resources are exhausted.
    """
    
    def __init__(self,
                 tac_gen: TacticGenerator,
                 timeout: int,
                 max_expansions: Optional[int] = None,
                 num_sampled_tactics: int = 64,
                 debug: bool = False):
        """
        Initialize the best-first search prover.
        
        Args:
            tac_gen: Tactic generator for suggesting proof steps
            timeout: Maximum time in seconds for proof search
            max_expansions: Maximum number of node expansions (None = unlimited)
            num_sampled_tactics: Number of tactics to sample per state
            debug: Enable debug logging and invariant checking
        """
        self.tac_gen = tac_gen
        self.timeout = timeout
        self.max_expansions = max_expansions
        self.num_sampled_tactics = num_sampled_tactics
        self.debug = debug
        self.logger = logger
        
        # Search state
        self.search_tree: Optional[SearchTree] = None
        self.dojo = None
        self.theorem = None
        self.position = None
        
        # Timing statistics
        self.actor_time = 0.0
        self.environment_time = 0.0
        self.total_time = None
        
    def search(self, repo: LeanGitRepo, thm: Theorem, pos: Pos) -> Optional[SearchResult]:
        """
        Perform best-first search to find a proof for the given theorem.
        
        Args:
            repo: Lean Git repository containing the theorem
            thm: Theorem to be proved
            pos: Position information for the theorem
            
        Returns:
            SearchResult with proof information, or None if initialization failed
        """
        self.logger.info(f"Proving {thm}")
        
        # Initialize search state
        self.theorem = thm
        self.position = pos
        self.actor_time = 0.0
        self.environment_time = 0.0
        
        if not LEAN_DOJO_AVAILABLE:
            self.logger.warning("LeanDojo not available - returning mock result")
            return self._create_mock_result(thm)
        
        # Determine additional imports
        additional_imports = []
        if hasattr(self.tac_gen, 'module') and self.tac_gen.module:
            additional_imports = [self.tac_gen.module]
        
        try:
            with Dojo(thm, self.timeout, additional_imports=additional_imports) as (dojo, init_state):
                self.dojo = dojo
                self.search_tree = SearchTree(init_state)
                
                try:
                    asyncio.run(self._best_first_search())
                except DojoCrashError as ex:
                    self.logger.warning(f"Dojo crashed with {ex} when proving {thm}")
                
            # Extract results
            if self.search_tree.root.status == SearchStatus.PROVED:
                proof = self.search_tree.extract_proof()
            else:
                proof = None
            
            result = SearchResult(
                theorem=thm,
                status=self.search_tree.root.status,
                proof=proof,
                actor_time=self.actor_time,
                environment_time=self.environment_time,
                total_time=self.total_time,
                num_total_nodes=len(self.search_tree.nodes),
                num_searched_nodes=self.search_tree.num_expansions,
            )
            
            self.logger.info(f"Search completed: {result}")
            return result
            
        except DojoInitError as ex:
            self.logger.warning(f"Dojo initialization failed: {ex}")
            return None
    
    async def _best_first_search(self) -> None:
        """Main best-first search loop."""
        time_start = time.monotonic()
        tolerance = 1.0  # second
        
        # Priority queue for open nodes
        priority_queue = asyncio.PriorityQueue()
        priority_queue.put_nowait((-self.search_tree.root.priority, self.search_tree.root))
        
        while True:
            if priority_queue.empty():
                self.logger.info("Ran out of nodes to search")
                break
            
            try:
                await self._search_step(priority_queue)
            except DojoHardTimeoutError:
                current_time = time.monotonic()
                elapsed = current_time - time_start
                self.logger.info(f"Hard timeout: {elapsed:.2f}s >= {self.timeout}s")
                assert elapsed + tolerance >= self.timeout
            
            # Update total time
            self.total_time = time.monotonic() - time_start + tolerance
            
            # Check termination conditions
            if self.total_time > self.timeout:
                if self.search_tree.root.status == SearchStatus.PROVED:
                    self.logger.info("Found a proof!")
                else:
                    self.search_tree.root.status = SearchStatus.OPEN
                    self.logger.info("Hit timeout limit")
                break
            
            if (self.max_expansions is not None and 
                self.search_tree.num_expansions > self.max_expansions):
                if self.search_tree.root.status == SearchStatus.PROVED:
                    self.logger.info("Found a proof!")
                else:
                    self.search_tree.root.status = SearchStatus.OPEN
                    self.logger.info("Hit expansion limit")
                break
            
            if self.search_tree.root.status == SearchStatus.FAILED:
                self.logger.info("Search failed - no more options")
                break
            
            if self.search_tree.root.status == SearchStatus.PROVED:
                self.logger.info("Found a proof!")
                break
    
    async def _search_step(self, priority_queue: asyncio.PriorityQueue) -> None:
        """
        Perform a single search step.
        
        Selects the highest priority node, generates tactics, and expands
        the search tree with new nodes for each valid tactic result.
        """
        # Get next node to expand
        _, node = await priority_queue.get()
        
        if not isinstance(node, InternalNode) or node.status != SearchStatus.OPEN:
            return  # Skip non-expandable nodes
        
        self.search_tree.num_expansions += 1
        self.logger.debug(f"Expanding node {self.search_tree.num_expansions}: {node.state}")
        
        # Generate tactics for this state
        state_str = str(node.state)
        tactics = await self._generate_tactics(state_str)
        
        if not tactics:
            self.logger.debug("No tactics generated")
            node.status = SearchStatus.FAILED
            node.update_status()
            return
        
        # Try each suggested tactic
        found_proof = False
        for tactic, logprob in tactics:
            edge, child_node = self._run_tactic(node, tactic, logprob, priority_queue)
            
            if isinstance(child_node, ProofNode):
                found_proof = True
                self.logger.info(f"Proof step found: {tactic}")
                break
        
        # Update node status based on children
        node.update_status()
        
        if self.debug:
            self.search_tree.check_invariants()
    
    @torch.no_grad()
    async def _generate_tactics(self, state: str) -> List[Tuple[str, float]]:
        """Generate tactics for the given state."""
        t0 = time.monotonic()
        
        # Get file path for the theorem
        file_path = str(self.theorem.file_path)
        if hasattr(self.theorem, 'repo') and self.theorem.repo != self._get_repo():
            # Handle cross-repository references
            file_path = (self.theorem.repo.get_packages_dir() / 
                        self.theorem.repo.name / file_path)
        
        # Generate tactics using the tactic generator
        suggestions = self.tac_gen.generate(
            state=state,
            file_path=file_path,
            theorem_full_name=self.theorem.full_name,
            theorem_pos=self.position,
            num_samples=self.num_sampled_tactics,
        )
        
        self.actor_time += time.monotonic() - t0
        
        self.logger.debug(f"Generated {len(suggestions)} tactics: {[t[0] for t in suggestions[:5]]}")
        return suggestions
    
    def _run_tactic(self, node: InternalNode, tactic: str, logprob: float,
                   priority_queue: asyncio.PriorityQueue) -> Tuple[Any, Any]:
        """
        Execute a tactic and create the resulting node.
        
        Args:
            node: Parent node to execute tactic from
            tactic: Tactic string to execute
            logprob: Log probability of the tactic
            priority_queue: Queue to add new open nodes to
            
        Returns:
            Tuple of (edge, child_node)
        """
        t0 = time.monotonic()
        
        # Execute tactic in Lean environment
        response = self.dojo.run_tac(node.state, tactic)
        
        elapsed = time.monotonic() - t0
        self.environment_time += elapsed
        
        # Create edge and child node
        edge, child_node = self.search_tree.add_edge(node, tactic, response, logprob)
        
        # Add open nodes to priority queue
        if (isinstance(child_node, InternalNode) and 
            child_node.status == SearchStatus.OPEN):
            priority_queue.put_nowait((-child_node.priority, child_node))
        
        return edge, child_node
    
    def _get_repo(self) -> LeanGitRepo:
        """Get the repository for the current theorem."""
        # This would be set during search initialization
        return getattr(self, '_repo', None)
    
    def _create_mock_result(self, theorem: Theorem) -> SearchResult:
        """Create a mock search result for testing without LeanDojo."""
        return SearchResult(
            theorem=theorem,
            status=SearchStatus.FAILED,
            proof=None,
            actor_time=0.0,
            environment_time=0.0,
            total_time=0.0,
            num_total_nodes=1,
            num_searched_nodes=0,
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed search statistics."""
        stats = {
            "actor_time": self.actor_time,
            "environment_time": self.environment_time,
            "total_time": self.total_time,
            "timeout": self.timeout,
            "max_expansions": self.max_expansions,
            "num_sampled_tactics": self.num_sampled_tactics,
            "tactic_generator": self.tac_gen.get_name(),
        }
        
        if self.search_tree:
            stats.update(self.search_tree.get_statistics())
        
        return stats 