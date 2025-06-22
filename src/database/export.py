"""
Dataset export functionality for LeanAgent database.

This module handles exporting datasets in various formats for machine learning
and theorem proving experiments.
"""

import json
import datetime
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from loguru import logger

from .models import Repository, Theorem
from .utils import safe_remove_dir_path
from .split import DataSplitter, Split


class DatasetExporter:
    """
    Handles exporting datasets from repositories to various formats.
    
    This class manages the complete export pipeline including data merging,
    splitting, and format conversion for machine learning experiments.
    """
    
    def __init__(self, data_splitter: Optional[DataSplitter] = None):
        """
        Initialize the DatasetExporter.
        
        Args:
            data_splitter (DataSplitter, optional): Custom data splitter. 
                                                   If None, creates default splitter.
        """
        self.data_splitter = data_splitter or DataSplitter()
    
    def generate_merged_dataset(self, 
                               repositories: List[Repository],
                               output_path: Path, 
                               repos_to_include: Optional[List[Tuple[str, str]]] = None) -> None:
        """
        Generate a merged dataset from multiple repositories.
        
        Args:
            repositories (List[Repository]): List of all repositories.
            output_path (Path): Path where the merged dataset will be saved.
            repos_to_include (List[Tuple[str, str]], optional): List of (url, commit) tuples 
                                                               of repositories to include.
        """
        output_path.mkdir(parents=True, exist_ok=True)

        repos_to_process = repositories
        if repos_to_include is not None:
            repos_to_process = [
                repo for repo in repositories 
                if (repo.url, repo.commit) in repos_to_include
            ]

        if repos_to_include is None:
            logger.info("Merging all repositories in the database.")
        else:
            logger.info("Merging selected repositories in the database:")
            for url, commit in repos_to_include:
                logger.info(f"  - {url} (commit: {commit})")

        # Merge theorems with deduplication
        all_theorems = {}
        all_traced_files = set()

        for repo in repos_to_process:
            for theorem in repo.get_all_theorems:
                # Create unique key for deduplication
                key = (theorem.file_path, theorem.full_name, 
                      list(theorem.start)[0], list(theorem.start)[1], 
                      list(theorem.end)[0], list(theorem.end)[1])
                
                date_processed = repo.metadata["date_processed"]
                if isinstance(date_processed, str):
                    date_processed = datetime.datetime.fromisoformat(date_processed)
                
                # Keep the most recent version of each theorem
                if key not in all_theorems or date_processed > all_theorems[key][1]:
                    all_theorems[key] = (theorem, date_processed)

            all_traced_files.update(repo.files_traced)

        theorems = [theorem for theorem, _ in all_theorems.values()]
        logger.info(f"Merged {len(theorems)} unique theorems from {len(repos_to_process)} repositories")
        
        # Split the data
        splits = self.data_splitter.split_data(theorems)

        # Clean output directory
        safe_remove_dir_path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export all components
        self._export_proofs(splits, output_path)
        logger.info(f"Exported proofs to {output_path}")

        self._merge_corpus(repos_to_process, output_path)
        logger.info(f"Merged and exported corpus to {output_path}")

        self._export_traced_files(all_traced_files, output_path)
        logger.info(f"Exported traced files to {output_path}")

        self._export_metadata(repos_to_process, output_path)
        logger.info(f"Exported metadata to {output_path}")

    def _merge_corpus(self, repos: List[Repository], output_path: Path) -> None:
        """
        Merge corpus files from multiple repositories.
        
        Args:
            repos (List[Repository]): List of repositories to merge.
            output_path (Path): Output directory path.
        """
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

        corpus_file = output_path / "corpus.jsonl"
        with open(corpus_file, 'w') as f:
            for line in merged_corpus.values():
                f.write(line + "\n")
        
        logger.info(f"Merged {len(merged_corpus)} premise files into corpus.jsonl")

    def _export_proofs(self, splits: Dict[str, Split], output_path: Path) -> None:
        """
        Export proof data in multiple split strategies.
        
        Args:
            splits (Dict[str, Split]): Dictionary of split strategies and their data.
            output_path (Path): Output directory path.
        """
        for strategy, split in splits.items():
            strategy_dir = output_path / strategy
            strategy_dir.mkdir(parents=True, exist_ok=True)

            for split_name, theorems in split.items():
                data = []
                for theorem in theorems:
                    # Filter tactics - exclude "no goals" and bullet tactics
                    tactics = [
                        {
                            "tactic": tactic.tactic,
                            "annotated_tactic": [
                                tactic.annotated_tactic[0],
                                [
                                    {
                                        "full_name": annotation.full_name,
                                        "def_path": str(annotation.def_path),
                                        "def_pos": list(annotation.def_pos),
                                        "def_end_pos": list(annotation.def_end_pos)
                                    } for annotation in tactic.annotated_tactic[1]
                                ]
                            ],
                            "state_before": tactic.state_before,
                            "state_after": tactic.state_after,
                        }
                        for tactic in theorem.traced_tactics
                        if (tactic.state_before != "no goals" and 
                            "·" not in tactic.tactic)
                    ]
                    
                    theorem_data = {
                        "url": theorem.url,
                        "commit": theorem.commit,
                        "file_path": str(theorem.file_path),
                        "full_name": theorem.full_name,
                        "theorem_statement": theorem.theorem_statement,
                        "start": list(theorem.start),
                        "end": list(theorem.end),
                        "traced_tactics": tactics,
                    }
                    data.append(theorem_data)

                output_file = strategy_dir / f"{split_name}.json"
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                logger.debug(f"Exported {len(data)} theorems to {output_file}")

    def _export_traced_files(self, all_traced_files: Set[Path], output_path: Path) -> None:
        """
        Export list of traced files.
        
        Args:
            all_traced_files (Set[Path]): Set of all traced file paths.
            output_path (Path): Output directory path.
        """
        traced_files_path = output_path / "traced_files.jsonl"
        with open(traced_files_path, 'w') as f:
            for file_path in all_traced_files:
                f.write(json.dumps({
                    "traced_file_path": str(file_path)
                }) + "\n")
        
        logger.info(f"Exported {len(all_traced_files)} traced files to traced_files.jsonl")

    def _export_metadata(self, repos: List[Repository], output_path: Path) -> None:
        """
        Export metadata about the repositories and datasets.
        
        Args:
            repos (List[Repository]): List of repositories.
            output_path (Path): Output directory path.
        """
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

        # Convert datetime objects to ISO format strings
        for repo_data in metadata["repositories"]:
            if isinstance(repo_data["metadata"].get("date_processed"), datetime.datetime):
                repo_data["metadata"]["date_processed"] = \
                    repo_data["metadata"]["date_processed"].isoformat()
        
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Exported metadata for {len(repos)} repositories to metadata.json") 