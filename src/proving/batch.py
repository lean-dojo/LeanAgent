"""
Batch Processing for Theorem Proving

This module provides batch processing capabilities for theorem proving,
allowing efficient processing of multiple theorems simultaneously with
proper result handling and progress tracking.

The batch processor manages theorem queues, coordinates with distributed
provers, and handles result aggregation and database updates.
"""

import time
import pickle
import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional, Any
from loguru import logger

from .search import SearchResult, SearchStatus

try:
    from lean_dojo import Theorem as LeanDojoTheorem, Pos, LeanGitRepo
    LEAN_DOJO_AVAILABLE = True
except (ImportError, Exception) as e:
    logger.warning(f"LeanDojo not available - using mock types: {e}")
    LEAN_DOJO_AVAILABLE = False
    LeanDojoTheorem = Any
    Pos = Any
    LeanGitRepo = Any

# Import from database module
try:
    from ..database.models import Theorem, AnnotatedTactic, Repository
    DATABASE_AVAILABLE = True
except ImportError:
    logger.warning("Database models not available - using mock types")
    DATABASE_AVAILABLE = False
    Theorem = Any
    AnnotatedTactic = Any
    Repository = Any


@dataclass
class TheoremBatch:
    """A batch of theorems to be processed together."""
    
    theorems: List[Tuple[Theorem, LeanDojoTheorem]]
    positions: List[Pos] 
    repository: Repository
    batch_id: str = field(default_factory=lambda: str(datetime.datetime.now().timestamp()))
    
    def __len__(self) -> int:
        return len(self.theorems)
    
    def is_empty(self) -> bool:
        return len(self.theorems) == 0
    
    def get_lean_dojo_theorems(self) -> List[LeanDojoTheorem]:
        """Extract LeanDojo theorems from the batch."""
        return [ldj_thm for _, ldj_thm in self.theorems]
    
    def get_theorem_map(self) -> Dict[LeanDojoTheorem, Theorem]:
        """Create mapping from LeanDojo theorems to our theorems."""
        return {ldj_thm: thm for thm, ldj_thm in self.theorems}


@dataclass
class BatchResult:
    """Results from processing a theorem batch."""
    
    batch_id: str
    total_theorems: int
    proved_theorems: int
    failed_theorems: int
    processing_time: float
    results: List[Optional[SearchResult]]
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate for this batch."""
        if self.total_theorems == 0:
            return 0.0
        return self.proved_theorems / self.total_theorems


class BatchProcessor:
    """
    Processes batches of theorems for proof search.
    
    The batch processor coordinates between the database, theorem selection,
    proof search, and result handling to efficiently process multiple theorems.
    """
    
    def __init__(self, 
                 batch_size: int = 12,
                 save_interval_minutes: int = 30,
                 encountered_theorems_file: str = "encountered_theorems.pkl"):
        """
        Initialize the batch processor.
        
        Args:
            batch_size: Number of theorems to process in each batch
            save_interval_minutes: How often to save progress
            encountered_theorems_file: File to track processed theorems
        """
        self.batch_size = batch_size
        self.save_interval = datetime.timedelta(minutes=save_interval_minutes)
        self.encountered_theorems_file = encountered_theorems_file
        self.logger = logger
        
        # State tracking
        self.processed_theorems: Set[Tuple[str, str, Tuple[int, int], Tuple[int, int]]] = set()
        self.all_encountered_theorems: Set[Tuple[str, str, Tuple[int, int], Tuple[int, int]]] = set()
        self.last_save_time = datetime.datetime.now()
        
        # Load previous state
        self._load_encountered_theorems()
    
    def create_batches(self, repository: Repository) -> List[TheoremBatch]:
        """
        Create theorem batches from a repository.
        
        Args:
            repository: Repository containing theorems to process
            
        Returns:
            List of theorem batches ready for processing
        """
        sorry_theorems = repository.sorry_theorems_unproved
        repo_url = repository.url
        repo_commit = repository.commit
        
        self.logger.info(f"Creating batches from {len(sorry_theorems)} sorry theorems")
        
        batches = []
        current_batch_theorems = []
        current_batch_positions = []
        
        for theorem in sorry_theorems:
            # Skip theorems from dependencies
            if theorem.url != repo_url or theorem.commit != repo_commit:
                continue
            
            theorem_id = self._theorem_identifier(theorem)
            
            # Skip already encountered theorems
            if theorem_id in self.all_encountered_theorems:
                self.logger.debug(f"Skipping already encountered theorem: {theorem.full_name}")
                continue
            
            # Skip already processed theorems in this session
            if theorem_id in self.processed_theorems:
                self.logger.debug(f"Skipping already processed theorem: {theorem.full_name}")
                continue
            
            # Mark as encountered and processed
            self.all_encountered_theorems.add(theorem_id)
            self.processed_theorems.add(theorem_id)
            
            # Convert to LeanDojo theorem
            if LEAN_DOJO_AVAILABLE:
                lean_dojo_theorem = LeanDojoTheorem(
                    repo=LeanGitRepo(repo_url, repo_commit),
                    file_path=theorem.file_path,
                    full_name=theorem.full_name
                )
                position = Pos(*theorem.start)
            else:
                lean_dojo_theorem = None
                position = None
            
            current_batch_theorems.append((theorem, lean_dojo_theorem))
            current_batch_positions.append(position)
            
            # Create batch when full
            if len(current_batch_theorems) >= self.batch_size:
                batch = TheoremBatch(
                    theorems=current_batch_theorems,
                    positions=current_batch_positions,
                    repository=repository
                )
                batches.append(batch)
                
                current_batch_theorems = []
                current_batch_positions = []
        
        # Add remaining theorems as final batch
        if current_batch_theorems:
            batch = TheoremBatch(
                theorems=current_batch_theorems,
                positions=current_batch_positions,
                repository=repository
            )
            batches.append(batch)
        
        self.logger.info(f"Created {len(batches)} batches for processing")
        return batches
    
    def process_batch(self, batch: TheoremBatch, prover, database, 
                     database_path: str) -> BatchResult:
        """
        Process a single theorem batch.
        
        Args:
            batch: Batch of theorems to process
            prover: Distributed prover instance
            database: Dynamic database instance
            database_path: Path to save database updates
            
        Returns:
            BatchResult with processing statistics
        """
        start_time = time.time()
        self.logger.info(f"Processing batch {batch.batch_id} with {len(batch)} theorems")
        
        if batch.is_empty():
            return BatchResult(
                batch_id=batch.batch_id,
                total_theorems=0,
                proved_theorems=0,
                failed_theorems=0,
                processing_time=0.0,
                results=[]
            )
        
        # Execute proof search
        lean_dojo_theorems = batch.get_lean_dojo_theorems()
        repo = LeanGitRepo(batch.repository.url, batch.repository.commit)
        
        try:
            results = prover.search_unordered(repo, lean_dojo_theorems, batch.positions)
        except Exception as e:
            self.logger.error(f"Error during proof search: {e}")
            results = [None] * len(batch)
        
        # Process results
        theorem_map = batch.get_theorem_map()
        proved_count = 0
        failed_count = 0
        
        for result in results:
            if isinstance(result, SearchResult):
                if result.theorem in theorem_map:
                    theorem = theorem_map[result.theorem]
                    
                    if result.status == SearchStatus.PROVED:
                        self._handle_proved_theorem(theorem, result, batch.repository, database)
                        proved_count += 1
                        self.logger.info(f"Proof found for {theorem.full_name}")
                    else:
                        failed_count += 1
                        self.logger.debug(f"No proof found for {theorem.full_name}")
                else:
                    self.logger.warning(f"Theorem not found in batch mapping: {result.theorem}")
                    failed_count += 1
            else:
                failed_count += 1
                self.logger.warning("Unexpected result type from prover")
        
        # Save database updates
        database.to_json(database_path)
        
        # Check if we need to save progress
        current_time = datetime.datetime.now()
        if current_time - self.last_save_time >= self.save_interval:
            self._save_progress()
            self.last_save_time = current_time
        
        processing_time = time.time() - start_time
        batch_result = BatchResult(
            batch_id=batch.batch_id,
            total_theorems=len(batch),
            proved_theorems=proved_count,
            failed_theorems=failed_count,
            processing_time=processing_time,
            results=results
        )
        
        self.logger.info(f"Batch {batch.batch_id} completed: "
                        f"{proved_count}/{len(batch)} proved in {processing_time:.2f}s")
        
        return batch_result
    
    def _handle_proved_theorem(self, theorem: Theorem, result: SearchResult, 
                              repository: Repository, database):
        """Handle a successfully proved theorem."""
        if not result.proof:
            self.logger.warning(f"Proved theorem {theorem.full_name} has no proof tactics")
            return
        
        # Convert proof to traced tactics
        traced_tactics = []
        for tactic in result.proof:
            traced_tactic = AnnotatedTactic(
                tactic=tactic,
                annotated_tactic=(tactic, []),
                state_before="",
                state_after=""
            )
            traced_tactics.append(traced_tactic)
        
        # Update theorem with proof
        theorem.traced_tactics = traced_tactics
        
        # Update repository status
        repository.change_sorry_to_proven(theorem, "proof_log.txt")
        
        # Update database
        database.update_repository(repository)
        
        self.logger.debug(f"Updated theorem {theorem.full_name} with {len(traced_tactics)} tactics")
    
    def _theorem_identifier(self, theorem: Theorem) -> Tuple[str, str, Tuple[int, int], Tuple[int, int]]:
        """Create unique identifier for a theorem."""
        return (
            theorem.full_name,
            str(theorem.file_path),
            tuple(theorem.start),
            tuple(theorem.end)
        )
    
    def _load_encountered_theorems(self):
        """Load previously encountered theorems from disk."""
        if not Path(self.encountered_theorems_file).exists():
            self.logger.info(f"No previous progress file found: {self.encountered_theorems_file}")
            return
        
        try:
            with open(self.encountered_theorems_file, 'rb') as f:
                content = f.read()
                if content:
                    self.all_encountered_theorems = pickle.loads(content)
                    self.logger.info(f"Loaded {len(self.all_encountered_theorems)} encountered theorems")
                else:
                    self.logger.warning(f"Progress file {self.encountered_theorems_file} is empty")
        except (EOFError, pickle.UnpicklingError) as e:
            self.logger.warning(f"Error reading progress file: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error loading progress: {e}")
    
    def _save_progress(self):
        """Save current progress to disk."""
        try:
            self.logger.info(f"Saving progress: {len(self.all_encountered_theorems)} encountered theorems")
            with open(self.encountered_theorems_file, 'wb') as f:
                pickle.dump(self.all_encountered_theorems, f)
        except Exception as e:
            self.logger.error(f"Error saving progress: {e}")
    
    def finalize(self):
        """Finalize batch processing and save final progress."""
        self._save_progress()
        self.logger.info("Batch processing finalized")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        return {
            "batch_size": self.batch_size,
            "total_encountered": len(self.all_encountered_theorems),
            "session_processed": len(self.processed_theorems),
            "save_interval_minutes": self.save_interval.total_seconds() / 60,
            "encountered_theorems_file": self.encountered_theorems_file,
        } 