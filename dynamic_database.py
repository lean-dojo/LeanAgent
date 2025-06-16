from __future__ import annotations
import datetime
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Union, Tuple, Set
from pathlib import Path
from lean_dojo.data_extraction.lean import Pos
from tqdm import tqdm
import random
from collections import defaultdict
from loguru import logger
import shutil

def parse_pos(pos_str):
    """
    Parses a position string or list into a Pos object.

    Args:
        pos_str (str or list): The position data, either as a string in the format 'Pos(x, y)' 
                               or as a list [x, y].

    Returns:
        Pos: A Pos object initialized with the parsed coordinates.

    Raises:
        ValueError: If the input format is neither a string nor a list.
    """
    if isinstance(pos_str, str):
        # pos_str came from a JSON file
        pos_parts = pos_str.replace('Pos', '').replace('(', '').replace(')', '').split(',')
        return Pos(int(pos_parts[0]), int(pos_parts[1]))
    elif isinstance(pos_str, list):
        # pos_str came from a dictionary initialization
        return Pos(*pos_str)
    else:
        raise ValueError(f"Unexpected format for Pos: {pos_str}")


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
    Methods:
        from_dict(data: Dict) -> Annotation:
            Creates an Annotation instance from a dictionary.
        to_dict() -> Dict:
            Converts the Annotation instance to a dictionary.
    """
    full_name: str
    def_path: str
    def_pos: Pos
    def_end_pos: Pos

    @classmethod
    def from_dict(cls, data: Dict) -> Annotation:
        if not all(key in data for key in ["full_name", "def_path", "def_pos", "def_end_pos"]):
            raise ValueError("Invalid Annotation data format")
        return cls(
            full_name=data["full_name"],
            def_path=data["def_path"],
            def_pos=parse_pos(data["def_pos"]),
            def_end_pos=parse_pos(data["def_end_pos"])
        )
    
    def to_dict(self) -> Dict:
        return {
            "full_name": self.full_name,
            "def_path": self.def_path,
            "def_pos": repr(self.def_pos),
            "def_end_pos": repr(self.def_end_pos)
        }

@dataclass
class AnnotatedTactic:
    """
    AnnotatedTactic is a data class that represents a tactic with its annotations and states before and after its application.
    Attributes:
        tactic (str): The tactic applied.
        annotated_tactic (Tuple[str, List[Annotation]]): A tuple containing the tactic and a list of annotations.
        state_before (str): The state before the tactic is applied.
        state_after (str): The state after the tactic is applied.
    Methods:
        from_dict(cls, data: Dict) -> AnnotatedTactic:
            Creates an AnnotatedTactic instance from a dictionary.
            Args:
                data (Dict): A dictionary containing the keys "tactic", "annotated_tactic", "state_before", and "state_after".
            Returns:
                AnnotatedTactic: An instance of AnnotatedTactic.
            Raises:
                ValueError: If the dictionary does not contain the required keys.
        to_dict(self) -> Dict:
            Converts the AnnotatedTactic instance to a dictionary.
            Returns:
                Dict: A dictionary representation of the AnnotatedTactic instance.
    """
    tactic: str
    annotated_tactic: Tuple[str, List[Annotation]]
    state_before: str
    state_after: str

    @classmethod
    def from_dict(cls, data: Dict) -> AnnotatedTactic:
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
        traced_tactics (Optional[List[AnnotatedTactic]], optional): A list of traced tactics.
        difficulty_rating (Optional[float], optional): The difficulty rating of the theorem.
    Methods:
        __eq__(self, other): Checks if two Theorem instances are equal.
        is_same_theorem(self, other: Theorem) -> bool: Checks if two Theorem instances represent the same theorem.
        from_dict(cls, data: Dict, url: str, commit: str) -> Theorem: Creates a Theorem instance from a dictionary.
        to_dict(self) -> Dict: Converts the Theorem instance to a dictionary.
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
        return (self.full_name == other.full_name and
                self.file_path == other.file_path and
                self.start == other.start and
                self.end == other.end)

    @classmethod
    def from_dict(cls, data: Dict, url: str, commit: str) -> Theorem:
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
    A class representing a Premise with various attributes.
    Attributes:
        full_name (str): The full name of the premise.
        code (str): The code associated with the premise.
        start (Pos): The starting position of the premise.
        end (Pos): The ending position of the premise.
        kind (str): The kind or type of the premise.
    Methods:
        from_dict(cls, data: Dict) -> Premise:
            Creates an instance of Premise from a dictionary.
            Args:
                data (Dict): A dictionary containing the premise data.
            Returns:
                Premise: An instance of the Premise class.
            Raises:
                ValueError: If the dictionary does not contain the required keys.
        to_dict(self) -> Dict:
            Converts the Premise instance to a dictionary.
            Returns:
                Dict: A dictionary representation of the Premise instance.
    """
    full_name: str
    code: str
    start: Pos
    end: Pos
    kind: str

    @classmethod
    def from_dict(cls, data: Dict) -> Premise:
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
    Represents a file containing premises and their associated imports.
    Attributes:
        path (Path): The file path.
        imports (List[str]): A list of import statements.
        premises (List[Premise]): A list of premises.
    Methods:
        from_dict(cls, data: Dict) -> PremiseFile:
            Creates an instance of PremiseFile from a dictionary.
            Args:
                data (Dict): A dictionary containing the keys "path", "imports", and "premises".
            Returns:
                PremiseFile: An instance of PremiseFile.
            Raises:
                ValueError: If the dictionary does not contain the required keys.
        to_dict(self) -> Dict:
            Converts the PremiseFile instance to a dictionary.
            Returns:
                Dict: A dictionary representation of the PremiseFile instance.
    """
    path: Path
    imports: List[str]
    premises: List[Premise]

    @classmethod
    def from_dict(cls, data: Dict) -> PremiseFile:
        if not all(key in data for key in ["path", "imports", "premises"]):
            raise ValueError("Invalid PremiseFile data format")
        return cls(
            path=Path(data["path"]),
            imports=data["imports"],
            premises=[Premise.from_dict(p) for p in data["premises"]]
        )
    
    def to_dict(self) -> Dict:
        return {
            "path": str(self.path),
            "imports": self.imports,
            "premises": [p.to_dict() for p in self.premises]
        }

@dataclass
class Repository:
    """
    Repository class represents a repository with various attributes and methods to manage theorems and premise files.
    Attributes:
        url (str): URL of the repository.
        name (str): Name of the repository.
        commit (str): Commit hash of the repository.
        lean_version (str): Version of Lean used in the repository.
        lean_dojo_version (str): Version of Lean Dojo used in the repository.
        metadata (Dict[str, str]): Metadata associated with the repository.
        proven_theorems (List[Theorem]): List of proven theorems.
        sorry_theorems_proved (List[Theorem]): List of sorry theorems that have been proved.
        sorry_theorems_unproved (List[Theorem]): List of sorry theorems that are unproved.
        premise_files (List[PremiseFile]): List of premise files.
        files_traced (List[Path]): List of traced files.
        pr_url (Optional[str]): URL of the pull request.
    Methods:
        __eq__(self, other): Checks equality between two Repository instances.
        __hash__(self): Returns the hash value of the Repository instance.
        total_theorems(self) -> int: Returns the total number of theorems.
        num_proven_theorems(self) -> int: Returns the number of proven theorems.
        num_sorry_theorems_proved(self) -> int: Returns the number of sorry theorems that have been proved.
        num_sorry_theorems_unproved(self) -> int: Returns the number of sorry theorems that are unproved.
        num_sorry_theorems(self) -> int: Returns the total number of sorry theorems.
        num_premise_files(self) -> int: Returns the number of premise files.
        num_premises(self) -> int: Returns the total number of premises.
        num_files_traced(self) -> int: Returns the number of traced files.
        get_all_theorems(self) -> List[Theorem]: Returns a list of all theorems.
        get_theorem(self, full_name: str, file_path: str) -> Optional[Theorem]: Retrieves a theorem by its full name and file path.
        update_theorem(self, theorem: Theorem) -> None: Updates an existing theorem.
        get_premise_file(self, path: str) -> Optional[PremiseFile]: Retrieves a premise file by its path.
        get_file_traced(self, path: str) -> Optional[Path]: Retrieves a traced file by its path.
        from_dict(cls, data: Dict) -> Repository: Creates a Repository instance from a dictionary.
        to_dict(self) -> Dict: Converts the Repository instance to a dictionary.
        change_sorry_to_proven(self, theorem: Theorem, log_file: str) -> None: Changes a sorry theorem to a proven theorem and logs the change.
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
        return (self.url == other.url and 
                self.name == other.name and
                self.commit == other.commit and
                self.lean_version == other.lean_version and
                self.lean_dojo_version == other.lean_dojo_version)

    def __hash__(self):
        return hash((self.url, self.name, self.commit, self.lean_version, self.lean_dojo_version))

    @property
    def total_theorems(self) -> int:
        return self.num_proven_theorems + self.num_sorry_theorems

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
        return self.num_sorry_theorems_proved + self.num_sorry_theorems_unproved
    
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
        return self.proven_theorems + self.sorry_theorems_proved + self.sorry_theorems_unproved
    
    def get_theorem(self, full_name: str, file_path: str) -> Optional[Theorem]:
        for thm_list in [self.proven_theorems, self.sorry_theorems_proved, self.sorry_theorems_unproved]:
            for thm in thm_list:
                if thm.full_name == full_name and (str(thm.file_path) == file_path or (file_path == "" and str(thm.file_path) == ".")):
                    return thm
        return None
    
    def update_theorem(self, theorem: Theorem) -> None:
        for thm_list in [self.proven_theorems, self.sorry_theorems_proved, self.sorry_theorems_unproved]:
            for i, thm in enumerate(thm_list):
                if thm.is_same_theorem(theorem):
                    thm_list[i] = theorem
                    return
        raise ValueError(f"Theorem '{theorem.full_name}' not found.")
    
    def get_premise_file(self, path: str) -> Optional[PremiseFile]:
        return next((pf for pf in self.premise_files if str(pf.path) == path), None)

    def get_file_traced(self, path: str) -> Optional[Path]:
        return next((f for f in self.files_traced if str(f) == path), None)

    @classmethod
    def from_dict(cls, data: Dict) -> Repository:
        if not all(key in data for key in ["url", "name", "commit", "lean_version", "lean_dojo_version", "metadata"]):
            raise ValueError("Invalid Repository data format")
        if "date_processed" not in data["metadata"]:
            raise ValueError("Metadata must contain the 'date_processed' key")

        metadata = data["metadata"].copy()
        if isinstance(metadata["date_processed"], str):
            metadata["date_processed"] = datetime.datetime.fromisoformat(metadata["date_processed"])
        
        repo = cls(
            url=data["url"],
            name=data["name"],
            commit=data["commit"],
            lean_version=data["lean_version"],
            lean_dojo_version=data["lean_dojo_version"],
            metadata=metadata,
            files_traced=[],
            pr_url=data.get("pr_url")
        )

        if all(key in data for key in ["theorems_folder", "premise_files_corpus", "files_traced"]):
            if not all(os.path.exists(data[key]) for key in ["theorems_folder", "premise_files_corpus", "files_traced"]):
                raise ValueError("Paths to data cannot be empty when creating repo from dataset")

            theorems_folder = Path(data["theorems_folder"])
            for file in theorems_folder.glob("*.json"):
                with open(file, 'r') as f:
                    theorem_data = json.load(f)
                for t_data in tqdm(theorem_data):
                    theorem = Theorem.from_dict(t_data, repo.url, repo.commit)
                    if any('sorry' in step.tactic for step in (theorem.traced_tactics or [])):
                        repo.sorry_theorems_unproved.append(theorem)
                    else:
                        repo.proven_theorems.append(theorem)

            with open(data["premise_files_corpus"], 'r') as f:
                for line in f:
                    premise_file_data = json.loads(line)
                    premise_file = PremiseFile.from_dict(premise_file_data)
                    repo.premise_files.append(premise_file)

            with open(data["files_traced"], 'r') as f:
                for line in f:
                    traced_file_data = json.loads(line)
                    repo.files_traced.append(Path(traced_file_data["traced_file_path"]))
        else:
            # Process theorems and premises from the existing data structure
            repo.proven_theorems = [Theorem.from_dict(t, repo.url, repo.commit) for t in data.get("proven_theorems", [])]
            repo.sorry_theorems_proved = [Theorem.from_dict(t, repo.url, repo.commit) for t in data.get("sorry_theorems_proved", [])]
            repo.sorry_theorems_unproved = [Theorem.from_dict(t, repo.url, repo.commit) for t in data.get("sorry_theorems_unproved", [])]
            repo.premise_files = [PremiseFile.from_dict(pf) for pf in data.get("premise_files", [])]
            repo.files_traced = [Path(file) for file in data.get("files_traced", [])]

        return repo
    
    def to_dict(self) -> Dict:
        metadata_copy = self.metadata.copy()
        if isinstance(metadata_copy["date_processed"], datetime.datetime):
            metadata_copy["date_processed"] = metadata_copy["date_processed"].isoformat()
        return {
            "url": self.url,
            "name": self.name,
            "commit": self.commit,
            "lean_version": self.lean_version,
            "lean_dojo_version": self.lean_dojo_version,
            "metadata": metadata_copy,
            "total_theorems": self.total_theorems,
            "num_proven_theorems": self.num_proven_theorems,
            "num_sorry_theorems": self.num_sorry_theorems,
            "num_sorry_theorems_proved": self.num_sorry_theorems_proved,
            "num_sorry_theorems_unproved": self.num_sorry_theorems_unproved,
            "num_premise_files": self.num_premise_files,
            "num_premises": self.num_premises,
            "num_files_traced": self.num_files_traced,
            "proven_theorems": [t.to_dict() for t in self.proven_theorems],
            "sorry_theorems_proved": [t.to_dict() for t in self.sorry_theorems_proved],
            "sorry_theorems_unproved": [t.to_dict() for t in self.sorry_theorems_unproved],
            "premise_files": [pf.to_dict() for pf in self.premise_files],
            "files_traced": [str(file) for file in self.files_traced],
            "pr_url": self.pr_url
        }

    def change_sorry_to_proven(self, theorem: Theorem, log_file: str) -> None:
        if theorem in self.sorry_theorems_unproved:
            self.sorry_theorems_unproved.remove(theorem)
            self.sorry_theorems_proved.append(theorem)

            message = f"Theorem proved: {theorem.full_name} in {theorem.file_path} for repo {self.name} (commit: {self.commit})"
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_entry = f"{timestamp} - {message}\n"
            
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            with open(log_file, 'a') as f:
                f.write(log_entry)
        else:
            raise ValueError("The theorem is not in the list of unproved sorry theorems.")

def safe_remove_dir_path(dir_path):
    """
    Safely removes a directory if it exists.
    
    Attempts to remove the directory multiple times in case of permission errors.
    
    Args:
        dir_path (Path): Path object representing the directory to remove
        
    Raises:
        PermissionError: If the directory cannot be removed after multiple attempts
        
    Returns:
        None
    """
    if dir_path.exists():
        logger.warning(f"{dir_path} already exists. Removing it now.")
        max_retries = 5
        for attempt in range(max_retries):
            try:
                shutil.rmtree(dir_path)
                break
            except PermissionError as e:
                if attempt < max_retries - 1:
                    time.sleep(0.1)  # Wait a bit before retrying
                else:
                    logger.error(f"Failed to remove {dir_path} after {max_retries} attempts: {e}")
                    raise

@dataclass
class DynamicDatabase:
    """
    A class that manages a collection of repositories containing Lean theorem proofs.
    The DynamicDatabase class provides functionality for:
    1. Managing repositories (adding, retrieving, updating, deleting)
    2. Generating merged datasets from multiple repositories
    3. Splitting theorem data for training/validation/testing
    4. Exporting proofs, corpus data, and metadata
    Attributes:
        repositories: List of Repository objects managed by the database
    Methods:
        generate_merged_dataset: Creates a merged dataset from multiple repositories
        _merge_corpus: Merges premise files from multiple repositories
        _split_data: Splits theorem data using different strategies
        _split_randomly: Splits theorems randomly into train/val/test sets
        _split_by_premise: Splits theorems based on premises to ensure premise novelty
        _export_proofs: Exports theorem proofs in JSON format
        _export_traced_files: Exports information about traced files
        _export_metadata: Exports metadata about repositories and statistics
        add_repository: Adds a new repository to the database
        get_repository: Retrieves a repository by URL and commit
        update_repository: Updates an existing repository
        print_database_contents: Logs the current database contents
        delete_repository: Removes a repository from the database
        to_dict: Converts the database to a dictionary representation
        from_dict: Creates a database instance from a dictionary
        to_json: Serializes the database to a JSON file
        from_json: Deserializes a database from a JSON file
        update_json: Updates an existing JSON file with current database state
    """
    repositories: List[Repository] = field(default_factory=list)

    SPLIT = Dict[str, List[Theorem]]

    def generate_merged_dataset(self, output_path: Path, repos_to_include: Optional[List[Tuple[str, str]]] = None) -> None:
        """
        Generate a merged dataset from multiple repositories in the database.
        
        :param output_path: Path where the merged dataset will be saved
        :param repos_to_include: List of tuples (url, commit) of repositories to include in the dataset. 
                                 If None, all repos are included.
        """
        random.seed(3407)
        
        output_path.mkdir(parents=True, exist_ok=True)

        repos_to_process = self.repositories if repos_to_include is None else [
            repo for repo in self.repositories if (repo.url, repo.commit) in repos_to_include
        ]

        if repos_to_include is None:
            logger.info("Merging all repositories in the database.")
        else:
            logger.info("Merging selected repositories in the database:")
            for url, commit in repos_to_include:
                logger.info(f"  - {url} (commit: {commit})")

        all_theorems = {}
        all_traced_files = set()

        for repo in repos_to_process:
            for theorem in repo.get_all_theorems:
                key = (theorem.file_path, theorem.full_name, list(theorem.start)[0], list(theorem.start)[1], list(theorem.end)[0], list(theorem.end)[1])
                date_processed = repo.metadata["date_processed"]
                if isinstance(date_processed, str):
                    date_processed = datetime.datetime.fromisoformat(date_processed)
                if key not in all_theorems or date_processed > all_theorems[key][1]:
                    all_theorems[key] = (theorem, date_processed)

            all_traced_files.update(repo.files_traced)

        theorems = [t for t, _ in all_theorems.values()]
        splits = self._split_data(theorems)

        safe_remove_dir_path(output_path)

        self._export_proofs(splits, output_path)
        logger.info(f"Exported proofs to {output_path}")

        self._merge_corpus(repos_to_process, output_path)
        logger.info(f"Merged and exported corpus to {output_path}")

        self._export_traced_files(all_traced_files, output_path)
        logger.info(f"Exported traced files to {output_path}")

        self._export_metadata(repos_to_process, output_path)
        logger.info(f"Exported metadata to {output_path}")

    def _merge_corpus(self, repos: List[Repository], output_path: Path) -> None:
        merged_corpus = {}
        for repo in repos:
            for premise_file in repo.premise_files:
                file_data = {
                    "path": str(premise_file.path),
                    "imports": premise_file.imports,
                    "premises": [
                        {
                            "full_name": premise.full_name,
                            "code": premise.code,
                            "start": list(premise.start),
                            "end": list(premise.end),
                            "kind": premise.kind
                        } for premise in premise_file.premises
                    ]
                }
                path = file_data['path']
                if path not in merged_corpus:
                    merged_corpus[path] = json.dumps(file_data)

        with open(output_path / "corpus.jsonl", 'w') as f:
            for line in merged_corpus.values():
                f.write(line + "\n")

    def _split_data(self, theorems: List[Theorem], num_val_pct: float = 0.02, num_test_pct: float = 0.02) -> Dict[str, SPLIT]:
        num_theorems = len(theorems)
        num_val = int(num_theorems * num_val_pct)
        num_test = int(num_theorems * num_test_pct)

        return {
            "random": self._split_randomly(theorems, num_val, num_test),
            "novel_premises": self._split_by_premise(theorems, num_val, num_test),
        }

    def _split_randomly(self, theorems: List[Theorem], num_val: int, num_test: int) -> SPLIT:
        random.shuffle(theorems)
        num_train = len(theorems) - num_val - num_test
        return {
            "train": theorems[:num_train],
            "val": theorems[num_train : num_train + num_val],
            "test": theorems[num_train + num_val :],
        }

    def _split_by_premise(self, theorems: List[Theorem], num_val: int, num_test: int) -> SPLIT:
        num_val_test = num_val + num_test
        theorems_val_test = []

        theorems_by_premises = defaultdict(list)
        for t in theorems:
            if t.traced_tactics:
                for tactic in t.traced_tactics:
                    for annotation in tactic.annotated_tactic[1]:
                        theorems_by_premises[annotation.full_name].append(t)

        theorems_by_premises = sorted(theorems_by_premises.items(), key=lambda x: len(x[1]))

        for _, thms in theorems_by_premises:
            if len(theorems_val_test) < num_val_test:
                theorems_val_test.extend([t for t in thms if t not in theorems_val_test])
            else:
                break

        theorems_train = [t for t in theorems if t not in theorems_val_test]
        random.shuffle(theorems_val_test)

        return {
            "train": theorems_train,
            "val": theorems_val_test[:num_val],
            "test": theorems_val_test[num_val:],
        }

    def _export_proofs(self, splits: Dict[str, SPLIT], output_path: Path) -> None:
        for strategy, split in splits.items():
            strategy_dir = output_path / strategy
            strategy_dir.mkdir(parents=True, exist_ok=True)

            for name, theorems in split.items():
                data = []
                for thm in theorems:
                    tactics = [
                        {
                            "tactic": t.tactic,
                            "annotated_tactic": [
                                t.annotated_tactic[0],
                                [
                                    {
                                        "full_name": a.full_name,
                                        "def_path": str(a.def_path),
                                        "def_pos": list(a.def_pos),
                                        "def_end_pos": list(a.def_end_pos)
                                    } for a in t.annotated_tactic[1]
                                ]
                            ],
                            "state_before": t.state_before,
                            "state_after": t.state_after,
                        }
                        for t in thm.traced_tactics
                        if t.state_before != "no goals" and "·" not in t.tactic
                    ]
                    data.append({
                        "url": thm.url,
                        "commit": thm.commit,
                        "file_path": str(thm.file_path),
                        "full_name": thm.full_name,
                        "theorem_statement": thm.theorem_statement,
                        "start": list(thm.start),
                        "end": list(thm.end),
                        "traced_tactics": tactics,
                    })

                output_file = strategy_dir / f"{name}.json"
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)

    def _export_traced_files(self, all_traced_files: Set[Path], output_path: Path) -> None:
        with open(output_path / "traced_files.jsonl", 'w') as f:
            for file in all_traced_files:
                f.write(json.dumps({
                    "traced_file_path": str(file)
                }) + "\n")

    def _export_metadata(self, repos: List[Repository], output_path: Path) -> None:
        metadata = {
            "repositories": [
                {
                    "url": repo.url,
                    "name": repo.name,
                    "commit": repo.commit,
                    "lean_version": repo.lean_version,
                    "lean_dojo_version": repo.lean_dojo_version,
                    "metadata": repo.metadata,
                } for repo in repos
            ],
            "total_theorems": sum(repo.total_theorems for repo in repos),
            "num_proven_theorems": sum(repo.num_proven_theorems for repo in repos),
            "num_sorry_theorems": sum(repo.num_sorry_theorems for repo in repos),
            "num_premise_files": sum(repo.num_premise_files for repo in repos),
            "num_premises": sum(repo.num_premises for repo in repos),
            "num_files_traced": sum(repo.num_files_traced for repo in repos),
        }

        for repo_data in metadata["repositories"]:
            if isinstance(repo_data["metadata"]["date_processed"], datetime.datetime):
                repo_data["metadata"]["date_processed"] = repo_data["metadata"]["date_processed"].isoformat()
        
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    def add_repository(self, repo: Repository) -> None:
        logger.info(f"Attempting to add repository: {repo.url} (commit: {repo.commit})")
        if repo not in self.repositories:
            self.repositories.append(repo)
            logger.info(f"Added new repository: {repo.url} (commit: {repo.commit})")
        else:
            logger.info(f"Repository '{repo.url}' with commit '{repo.commit}' already exists in the database.")

    def get_repository(self, url: str, commit: str) -> Optional[Repository]:
        for repo in self.repositories:
            if repo.url == url and repo.commit == commit:
                return repo
        return None

    def update_repository(self, updated_repo: Repository) -> None:
        logger.info(f"Attempting to update repository: {updated_repo.url} (commit: {updated_repo.commit})")
        for i, repo in enumerate(self.repositories):
            if repo == updated_repo:
                self.repositories[i] = updated_repo
                logger.info(f"Updated repository: {updated_repo.url} (commit: {updated_repo.commit})")
                return
        logger.error(f"Repository '{updated_repo.url}' with commit '{updated_repo.commit}' not found for update.")
        raise ValueError(f"Repository '{updated_repo.url}' with commit '{updated_repo.commit}' not found.")

    def print_database_contents(self):
        logger.info("Current database contents:")
        for repo in self.repositories:
            logger.info(f"  - {repo.url} (commit: {repo.commit})")

    def delete_repository(self, url: str, commit: str) -> None:
        for i, repo in enumerate(self.repositories):
            if repo.url == url and repo.commit == commit:
                del self.repositories[i]
                return
        raise ValueError(f"Repository '{url}' with commit '{commit}' not found.")

    def to_dict(self) -> Dict:
        return {
            "repositories": [repo.to_dict() for repo in self.repositories]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> DynamicDatabase:
        if "repositories" not in data:
            raise ValueError("Invalid DynamicDatabase data format")
        db = cls()
        for repo_data in data["repositories"]:
            repo = Repository.from_dict(repo_data)
            db.add_repository(repo)
        return db

    def to_json(self, file_path: str) -> None:
        """Serialize the database to a JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, file_path: str) -> DynamicDatabase:
        """Deserialize the database from a JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def update_json(self, file_path: str) -> None:
        """Update an existing JSON file with the current database state."""
        try:
            existing_db = self.from_json(file_path)
        except FileNotFoundError:
            existing_db = DynamicDatabase()

        for repo in self.repositories:
            existing_db.update_repository(repo)

        existing_db.to_json(file_path)