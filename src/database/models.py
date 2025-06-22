"""
Data models for LeanAgent database system.

This module contains all the core data classes for representing theorems, premises,
repositories, and other mathematical entities in the LeanAgent system.
"""

from __future__ import annotations
import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple, Set
from pathlib import Path
from loguru import logger

# Optional import of lean_dojo with fallback
try:
    from lean_dojo.data_extraction.lean import Pos
    logger.info("✅ Successfully imported Pos from lean_dojo")
except (ImportError, Exception) as e:
    # Fallback implementation when lean_dojo is not available or fails to initialize
    logger.warning(f"⚠️  lean_dojo not available ({e}), using fallback Pos implementation")
    
    @dataclass
    class Pos:
        line: int
        column: int
        
        def __init__(self, line: int, column: int):
            self.line = line
            self.column = column
        
        def __iter__(self):
            return iter([self.line, self.column])
        
        def __repr__(self):
            return f"Pos({self.line}, {self.column})"
        
        def __eq__(self, other):
            if isinstance(other, Pos):
                return self.line == other.line and self.column == other.column
            return False

from .utils import parse_pos


@dataclass
class Annotation:
    """
    Annotation class represents a code annotation with its full name, definition path, 
    and position details.
    
    Attributes:
        full_name (str): The full name of the annotation.
        def_path (str): The file path where the annotation is defined.
        def_pos (Pos): The starting position of the annotation definition.
        def_end_pos (Pos): The ending position of the annotation definition.
    """
    full_name: str
    def_path: str
    def_pos: Pos
    def_end_pos: Pos

    @classmethod
    def from_dict(cls, data: Dict) -> Annotation:
        """Creates an Annotation instance from a dictionary."""
        if not all(key in data for key in ["full_name", "def_path", "def_pos", "def_end_pos"]):
            raise ValueError("Invalid Annotation data format")
        return cls(
            full_name=data["full_name"],
            def_path=data["def_path"],
            def_pos=parse_pos(data["def_pos"]),
            def_end_pos=parse_pos(data["def_end_pos"])
        )
    
    def to_dict(self) -> Dict:
        """Converts the Annotation instance to a dictionary."""
        return {
            "full_name": self.full_name,
            "def_path": self.def_path,
            "def_pos": repr(self.def_pos),
            "def_end_pos": repr(self.def_end_pos)
        }


@dataclass
class AnnotatedTactic:
    """
    AnnotatedTactic represents a tactic with its annotations and states before and after application.
    
    Attributes:
        tactic (str): The tactic applied.
        annotated_tactic (Tuple[str, List[Annotation]]): A tuple containing the tactic and annotations.
        state_before (str): The state before the tactic is applied.
        state_after (str): The state after the tactic is applied.
    """
    tactic: str
    annotated_tactic: Tuple[str, List[Annotation]]
    state_before: str
    state_after: str

    @classmethod
    def from_dict(cls, data: Dict) -> AnnotatedTactic:
        """Creates an AnnotatedTactic instance from a dictionary."""
        if not all(key in data for key in ["tactic", "annotated_tactic", "state_before", "state_after"]):
            raise ValueError("Invalid AnnotatedTactic data format")
        return cls(
            tactic=data["tactic"],
            annotated_tactic=(
                data["annotated_tactic"][0],
                [Annotation.from_dict(a) for a in data["annotated_tactic"][1]]
            ),
            state_before=data["state_before"],
            state_after=data["state_after"]
        )
    
    def to_dict(self) -> Dict:
        """Converts the AnnotatedTactic instance to a dictionary."""
        return {
            "tactic": self.tactic,
            "annotated_tactic": [
                self.annotated_tactic[0],
                [a.to_dict() for a in self.annotated_tactic[1]]
            ],
            "state_before": self.state_before,
            "state_after": self.state_after
        }


@dataclass
class Theorem:
    """
    A class to represent a theorem with its associated metadata.
    
    Attributes:
        full_name (str): The full name of the theorem.
        file_path (Path): The file path where the theorem is located.
        start (Pos): The starting position of the theorem in the file.
        end (Pos): The ending position of the theorem in the file.
        url (str): The URL associated with the theorem.
        commit (str): The commit hash associated with the theorem.
        theorem_statement (str, optional): The statement of the theorem.
        traced_tactics (List[AnnotatedTactic], optional): A list of traced tactics.
        difficulty_rating (float, optional): The difficulty rating of the theorem.
    """
    full_name: str
    file_path: Path
    start: Pos
    end: Pos
    url: str
    commit: str
    theorem_statement: str = None
    traced_tactics: Optional[List[AnnotatedTactic]] = field(default_factory=list)
    difficulty_rating: Optional[float] = None

    def __eq__(self, other):
        if not isinstance(other, Theorem):
            return NotImplemented
        return self.is_same_theorem(other)

    def is_same_theorem(self, other: Theorem) -> bool:
        """Checks if two Theorem instances represent the same theorem."""
        return (self.full_name == other.full_name and
                self.file_path == other.file_path and
                self.start == other.start and
                self.end == other.end)

    @classmethod
    def from_dict(cls, data: Dict, url: str, commit: str) -> Theorem:
        """Creates a Theorem instance from a dictionary."""
        if not all(key in data for key in ["full_name", "file_path", "start", "end"]):
            raise ValueError("Invalid Theorem data format")
        return cls(
            full_name=data["full_name"],
            theorem_statement=data.get("theorem_statement"),
            file_path=Path(data["file_path"]),
            start=parse_pos(data["start"]),
            end=parse_pos(data["end"]),
            url=url,
            commit=commit,
            traced_tactics=[
                AnnotatedTactic.from_dict(t) for t in data.get("traced_tactics", [])
            ],
            difficulty_rating=data.get("difficulty_rating")
        )
    
    def to_dict(self) -> Dict:
        """Converts the Theorem instance to a dictionary."""
        return {
            "full_name": self.full_name,
            "theorem_statement": self.theorem_statement,
            "file_path": str(self.file_path),
            "start": repr(self.start),
            "end": repr(self.end),
            "url": self.url,
            "commit": self.commit,
            "traced_tactics": [t.to_dict() for t in (self.traced_tactics or [])],
            "difficulty_rating": self.difficulty_rating
        }


@dataclass
class Premise:
    """
    A class to represent a mathematical premise.
    
    Attributes:
        full_name (str): The full name of the premise.
        code (str): The code content of the premise.
        start (Pos): The starting position in the file.
        end (Pos): The ending position in the file.
        kind (str): The kind/type of the premise.
    """
    full_name: str
    code: str
    start: Pos
    end: Pos
    kind: str

    @classmethod
    def from_dict(cls, data: Dict) -> Premise:
        """Creates a Premise instance from a dictionary."""
        if not all(key in data for key in ["full_name", "code", "start", "end", "kind"]):
            raise ValueError("Invalid Premise data format")
        return cls(
            full_name=data["full_name"],
            code=data["code"],
            start=parse_pos(data["start"]),
            end=parse_pos(data["end"]),
            kind=data["kind"]
        )
    
    def to_dict(self) -> Dict:
        """Converts the Premise instance to a dictionary."""
        return {
            "full_name": self.full_name,
            "code": self.code,
            "start": repr(self.start),
            "end": repr(self.end),
            "kind": self.kind
        }


@dataclass
class PremiseFile:
    """
    A class to represent a file containing mathematical premises.
    
    Attributes:
        path (Path): The file path.
        imports (List[str]): List of imported modules.
        premises (List[Premise]): List of premises in the file.
    """
    path: Path
    imports: List[str]
    premises: List[Premise]

    @classmethod
    def from_dict(cls, data: Dict) -> PremiseFile:
        """Creates a PremiseFile instance from a dictionary."""
        if not all(key in data for key in ["path", "imports", "premises"]):
            raise ValueError("Invalid PremiseFile data format")
        return cls(
            path=Path(data["path"]),
            imports=data["imports"],
            premises=[Premise.from_dict(p) for p in data["premises"]]
        )
    
    def to_dict(self) -> Dict:
        """Converts the PremiseFile instance to a dictionary."""
        return {
            "path": str(self.path),
            "imports": self.imports,
            "premises": [p.to_dict() for p in self.premises]
        }


@dataclass
class Repository:
    """
    A class to represent a mathematical repository with all its associated data.
    
    This is the main container for all mathematical knowledge from a single repository.
    """
    url: str
    name: str
    commit: str
    lean_version: str
    lean_dojo_version: str
    metadata: Dict[str, str]
    proven_theorems: List[Theorem] = field(default_factory=list)
    sorry_theorems_proved: List[Theorem] = field(default_factory=list)
    sorry_theorems_unproved: List[Theorem] = field(default_factory=list)
    premise_files: List[PremiseFile] = field(default_factory=list)
    files_traced: List[Path] = field(default_factory=list)
    pr_url: Optional[str] = None

    def __eq__(self, other):
        if not isinstance(other, Repository):
            return NotImplemented
        return self.url == other.url and self.commit == other.commit

    def __hash__(self):
        return hash((self.url, self.commit))

    @property
    def total_theorems(self) -> int:
        """Total number of theorems in the repository."""
        return len(self.proven_theorems) + len(self.sorry_theorems_proved) + len(self.sorry_theorems_unproved)

    @property
    def num_proven_theorems(self) -> int:
        return len(self.proven_theorems)

    @property
    def num_sorry_theorems_proved(self) -> int:
        return len(self.sorry_theorems_proved)

    @property
    def num_sorry_theorems_unproved(self) -> int:
        return len(self.sorry_theorems_unproved)

    @property
    def num_sorry_theorems(self) -> int:
        return len(self.sorry_theorems_proved) + len(self.sorry_theorems_unproved)

    @property
    def num_premise_files(self) -> int:
        return len(self.premise_files)

    @property
    def num_premises(self) -> int:
        return sum(len(pf.premises) for pf in self.premise_files)

    @property
    def num_files_traced(self) -> int:
        return len(self.files_traced)

    @property
    def get_all_theorems(self) -> List[Theorem]:
        """Get all theorems from all categories."""
        return self.proven_theorems + self.sorry_theorems_proved + self.sorry_theorems_unproved

    def get_theorem(self, full_name: str, file_path: str) -> Optional[Theorem]:
        """Find a theorem by name and file path."""
        for theorem in self.get_all_theorems:
            if theorem.full_name == full_name and str(theorem.file_path) == file_path:
                return theorem
        return None

    def update_theorem(self, theorem: Theorem) -> None:
        """Update an existing theorem or add it if not found."""
        for i, existing_theorem in enumerate(self.proven_theorems):
            if existing_theorem.is_same_theorem(theorem):
                self.proven_theorems[i] = theorem
                return
        # If not found, add to appropriate list based on context
        self.proven_theorems.append(theorem)

    def get_premise_file(self, path: str) -> Optional[PremiseFile]:
        """Find a premise file by path."""
        return next((pf for pf in self.premise_files if str(pf.path) == path), None)

    def get_file_traced(self, path: str) -> Optional[Path]:
        """Check if a file has been traced."""
        return next((f for f in self.files_traced if str(f) == path), None)

    @classmethod
    def from_dict(cls, data: Dict) -> Repository:
        """Creates a Repository instance from a dictionary."""
        required_keys = ["url", "name", "commit", "lean_version", "lean_dojo_version", "metadata"]
        if not all(key in data for key in required_keys):
            raise ValueError("Invalid Repository data format")
        
        return cls(
            url=data["url"],
            name=data["name"],
            commit=data["commit"],
            lean_version=data["lean_version"],
            lean_dojo_version=data["lean_dojo_version"],
            metadata=data["metadata"],
            proven_theorems=[
                Theorem.from_dict(t, data["url"], data["commit"]) 
                for t in data.get("proven_theorems", [])
            ],
            sorry_theorems_proved=[
                Theorem.from_dict(t, data["url"], data["commit"]) 
                for t in data.get("sorry_theorems_proved", [])
            ],
            sorry_theorems_unproved=[
                Theorem.from_dict(t, data["url"], data["commit"]) 
                for t in data.get("sorry_theorems_unproved", [])
            ],
            premise_files=[
                PremiseFile.from_dict(pf) for pf in data.get("premise_files", [])
            ],
            files_traced=[Path(f) for f in data.get("files_traced", [])],
            pr_url=data.get("pr_url")
        )
    
    def to_dict(self) -> Dict:
        """Converts the Repository instance to a dictionary."""
        return {
            "url": self.url,
            "name": self.name,
            "commit": self.commit,
            "lean_version": self.lean_version,
            "lean_dojo_version": self.lean_dojo_version,
            "metadata": self.metadata,
            "proven_theorems": [t.to_dict() for t in self.proven_theorems],
            "sorry_theorems_proved": [t.to_dict() for t in self.sorry_theorems_proved],
            "sorry_theorems_unproved": [t.to_dict() for t in self.sorry_theorems_unproved],
            "premise_files": [pf.to_dict() for pf in self.premise_files],
            "files_traced": [str(f) for f in self.files_traced],
            "pr_url": self.pr_url
        }

    def change_sorry_to_proven(self, theorem: Theorem, log_file: str) -> None:
        """Move a theorem from sorry_unproved to sorry_proved."""
        logger.info(f"Changing theorem '{theorem.full_name}' from sorry to proven. Log: {log_file}")
        
        # Find and remove from unproved
        for i, t in enumerate(self.sorry_theorems_unproved):
            if t.is_same_theorem(theorem):
                self.sorry_theorems_unproved.pop(i)
                break
        else:
            logger.warning(f"Theorem '{theorem.full_name}' not found in sorry_theorems_unproved")
        
        # Add to proved if not already there
        if not any(t.is_same_theorem(theorem) for t in self.sorry_theorems_proved):
            self.sorry_theorems_proved.append(theorem)
            logger.info(f"Added theorem '{theorem.full_name}' to sorry_theorems_proved")
        else:
            logger.info(f"Theorem '{theorem.full_name}' already in sorry_theorems_proved") 