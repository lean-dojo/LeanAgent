"""
Search Tree Management for Theorem Proving

This module implements the search tree structures used in best-first search
for theorem proving. It provides node types for different search states
and tree management utilities.

The search tree represents the proof search space where:
- Internal nodes represent intermediate proof states
- Proof nodes represent successfully completed proofs  
- Error nodes represent failed proof attempts
"""

import math
from enum import Enum
from abc import ABC, abstractmethod
from functools import total_ordering
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Iterable, Union, Any
from loguru import logger

try:
    from lean_dojo import (
        TacticState,
        LeanError,
        TimeoutError,
        ProofGivenUp,
        ProofFinished,
    )
    LEAN_DOJO_AVAILABLE = True
except (ImportError, Exception) as e:
    logger.warning(f"LeanDojo not available - search tree will use mock types: {e}")
    LEAN_DOJO_AVAILABLE = False
    # Mock types for testing without LeanDojo
    TacticState = Any
    LeanError = Exception
    TimeoutError = Exception
    ProofGivenUp = Exception
    ProofFinished = Any


class SearchStatus(Enum):
    """Status of a search node or proof search."""
    PROVED = "Proved"    # Node has at least one known proof
    FAILED = "Failed"    # Node has exhausted options and cannot be proved
    OPEN = "Open"        # Node has not been proven or given up on yet


@dataclass
class Edge:
    """An edge in the search tree representing a tactic application."""
    tactic: str
    src: 'SearchNode'
    dst: 'SearchNode'


@total_ordering
class SearchNode(ABC):
    """Abstract base class for all search tree nodes."""
    
    def __init__(self, cumulative_logprob: float = 0.0):
        self.cumulative_logprob = cumulative_logprob
        self.status = SearchStatus.OPEN
        
    @property
    @abstractmethod
    def priority(self) -> float:
        """Priority for search ordering (higher = better)."""
        pass
    
    def __lt__(self, other):
        return self.priority < other.priority
    
    def __eq__(self, other):
        return self.priority == other.priority


class InternalNode(SearchNode):
    """A search node representing an intermediate proof state."""
    
    def __init__(self, state: TacticState, cumulative_logprob: float = 0.0):
        super().__init__(cumulative_logprob)
        self.state = state
        self.out_edges: List[Edge] = []
        self.in_edges: List[Edge] = []
        
    @property
    def priority(self) -> float:
        """Priority based on cumulative log probability."""
        return self.cumulative_logprob
    
    def add_child(self, tactic: str, child_node: 'SearchNode') -> Edge:
        """Add a child node connected by a tactic."""
        edge = Edge(tactic=tactic, src=self, dst=child_node)
        self.out_edges.append(edge)
        
        if isinstance(child_node, InternalNode):
            child_node.in_edges.append(edge)
            
        return edge
    
    def extract_proof(self) -> List[Edge]:
        """Extract proof path from root to this node."""
        if self.status != SearchStatus.PROVED:
            return []
        
        # Find path to a proved leaf
        for edge in self.out_edges:
            if edge.dst.status == SearchStatus.PROVED:
                if isinstance(edge.dst, ProofNode):
                    return [edge]
                else:
                    # Recursively get proof from child
                    child_proof = edge.dst.extract_proof()
                    if child_proof:
                        return [edge] + child_proof
        
        return []
    
    def update_status(self):
        """Update node status based on children."""
        if not self.out_edges:
            return  # No children yet
        
        # Check if any child is proved
        for edge in self.out_edges:
            if edge.dst.status == SearchStatus.PROVED:
                self.status = SearchStatus.PROVED
                return
        
        # Check if all children are failed
        all_failed = all(edge.dst.status == SearchStatus.FAILED 
                        for edge in self.out_edges)
        if all_failed:
            self.status = SearchStatus.FAILED
    
    def check_invariants(self):
        """Verify node invariants for debugging."""
        # Check that all outgoing edges have this node as source
        for edge in self.out_edges:
            assert edge.src == self, "Edge source mismatch"
        
        # Check that all incoming edges have this node as destination
        for edge in self.in_edges:
            assert edge.dst == self, "Edge destination mismatch"


class ProofNode(SearchNode):
    """A search node representing a completed proof."""
    
    def __init__(self, proof_finished: ProofFinished):
        super().__init__(0.0)  # Completed proofs have max priority
        self.proof_finished = proof_finished
        self.status = SearchStatus.PROVED
        
    @property
    def priority(self) -> float:
        """Completed proofs have highest priority."""
        return float('inf')


class ErrorNode(SearchNode):
    """A search node representing a failed proof attempt."""
    
    def __init__(self, error: Union[LeanError, TimeoutError, ProofGivenUp]):
        super().__init__(-float('inf'))  # Failed nodes have lowest priority
        self.error = error
        self.status = SearchStatus.FAILED
        
    @property
    def priority(self) -> float:
        """Failed nodes have lowest priority."""
        return -float('inf')


class SearchTree:
    """
    Manages the search tree for theorem proving.
    
    The search tree tracks all explored proof states and their relationships,
    enabling efficient best-first search and proof extraction.
    """
    
    def __init__(self, root_state: TacticState):
        """
        Initialize search tree with root state.
        
        Args:
            root_state: Initial tactic state for the theorem
        """
        self.root = InternalNode(state=root_state, cumulative_logprob=0.0)
        self.nodes = {root_state: self.root}
        self.num_expansions = 0
        
    def get_or_create_node(self, response: Union[TacticState, ProofFinished, LeanError, TimeoutError, ProofGivenUp],
                          cumulative_logprob: float = 0.0) -> SearchNode:
        """
        Get existing node or create new node for the response.
        
        Args:
            response: LeanDojo response from tactic execution
            cumulative_logprob: Cumulative log probability for the path
            
        Returns:
            SearchNode corresponding to the response
        """
        try:
            # Return existing node if we've seen this response
            return self.nodes[response]
        except KeyError:
            # Create new node based on response type
            if isinstance(response, ProofFinished):
                node = ProofNode(response)
            elif type(response) in (LeanError, TimeoutError, ProofGivenUp):
                node = ErrorNode(response)
            else:
                # Assume it's a TacticState
                node = InternalNode(state=response, cumulative_logprob=cumulative_logprob)
            
            self.nodes[response] = node
            return node
    
    def add_edge(self, parent: InternalNode, tactic: str, response: Any, 
                logprob: float) -> Tuple[Edge, SearchNode]:
        """
        Add an edge to the search tree.
        
        Args:
            parent: Parent node
            tactic: Tactic that was applied
            response: LeanDojo response
            logprob: Log probability of the tactic
            
        Returns:
            Tuple of (edge, child_node)
        """
        cumulative_logprob = parent.cumulative_logprob + logprob
        child_node = self.get_or_create_node(response, cumulative_logprob)
        edge = parent.add_child(tactic, child_node)
        
        # Update parent status
        parent.update_status()
        
        return edge, child_node
    
    def get_open_nodes(self) -> List[InternalNode]:
        """Get all open internal nodes for expansion."""
        return [node for node in self.nodes.values() 
                if isinstance(node, InternalNode) and node.status == SearchStatus.OPEN]
    
    def extract_proof(self) -> Optional[List[str]]:
        """
        Extract proof tactics from root to a proved leaf.
        
        Returns:
            List of tactic strings if proof found, None otherwise
        """
        if self.root.status != SearchStatus.PROVED:
            return None
        
        proof_edges = self.root.extract_proof()
        return [edge.tactic for edge in proof_edges]
    
    def get_statistics(self) -> dict:
        """Get search tree statistics."""
        total_nodes = len(self.nodes)
        proved_nodes = sum(1 for node in self.nodes.values() 
                          if node.status == SearchStatus.PROVED)
        failed_nodes = sum(1 for node in self.nodes.values() 
                          if node.status == SearchStatus.FAILED)
        open_nodes = total_nodes - proved_nodes - failed_nodes
        
        return {
            "total_nodes": total_nodes,
            "proved_nodes": proved_nodes,
            "failed_nodes": failed_nodes,
            "open_nodes": open_nodes,
            "expansions": self.num_expansions,
            "root_status": self.root.status.value
        }
    
    def check_invariants(self):
        """Verify tree invariants for debugging."""
        # Check root node
        assert isinstance(self.root, InternalNode), "Root must be InternalNode"
        assert len(self.root.in_edges) == 0, "Root should have no incoming edges"
        
        # Check all nodes
        for response, node in self.nodes.items():
            if isinstance(node, InternalNode):
                node.check_invariants()
            
            # Verify node is in correct state
            if isinstance(response, ProofFinished):
                assert isinstance(node, ProofNode)
                assert node.status == SearchStatus.PROVED
            elif type(response) in (LeanError, TimeoutError, ProofGivenUp):
                assert isinstance(node, ErrorNode)
                assert node.status == SearchStatus.FAILED 