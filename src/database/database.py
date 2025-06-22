"""
Main database class for LeanAgent.

This module contains the DynamicDatabase class that manages repositories and
orchestrates all database operations for the LeanAgent system.
"""

import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from loguru import logger

from .models import Repository
from .export import DatasetExporter
from .split import DataSplitter


class DynamicDatabase:
    """
    Dynamic database for managing mathematical repositories and theorems.
    
    This class provides the main interface for managing repositories,
    theorems, and exporting datasets for machine learning experiments.
    """
    
    def __init__(self, data_splitter: Optional[DataSplitter] = None):
        """
        Initialize the DynamicDatabase.
        
        Args:
            data_splitter (DataSplitter, optional): Custom data splitter for exports.
        """
        self.repositories: List[Repository] = []
        self.exporter = DatasetExporter(data_splitter)
    
    def generate_merged_dataset(self, 
                               output_path: Path, 
                               repos_to_include: Optional[List[Tuple[str, str]]] = None) -> None:
        """
        Generate a merged dataset from multiple repositories in the database.
        
        Args:
            output_path (Path): Path where the merged dataset will be saved.
            repos_to_include (List[Tuple[str, str]], optional): List of (url, commit) tuples 
                                                               of repositories to include.
        """
        self.exporter.generate_merged_dataset(
            repositories=self.repositories,
            output_path=output_path,
            repos_to_include=repos_to_include
        )

    def add_repository(self, repo: Repository) -> None:
        """
        Add a repository to the database.
        
        Args:
            repo (Repository): Repository to add.
        """
        logger.info(f"Attempting to add repository: {repo.url} (commit: {repo.commit})")
        
        if repo not in self.repositories:
            self.repositories.append(repo)
            logger.info(f"Added new repository: {repo.url} (commit: {repo.commit})")
        else:
            logger.info(f"Repository '{repo.url}' with commit '{repo.commit}' already exists in the database.")

    def get_repository(self, url: str, commit: str) -> Optional[Repository]:
        """
        Get a repository by URL and commit.
        
        Args:
            url (str): Repository URL.
            commit (str): Commit hash.
            
        Returns:
            Repository or None: The repository if found, None otherwise.
        """
        for repo in self.repositories:
            if repo.url == url and repo.commit == commit:
                return repo
        return None

    def update_repository(self, updated_repo: Repository) -> None:
        """
        Update an existing repository in the database.
        
        Args:
            updated_repo (Repository): Updated repository data.
            
        Raises:
            ValueError: If repository is not found in the database.
        """
        logger.info(f"Attempting to update repository: {updated_repo.url} (commit: {updated_repo.commit})")
        
        for i, repo in enumerate(self.repositories):
            if repo == updated_repo:
                self.repositories[i] = updated_repo
                logger.info(f"Updated repository: {updated_repo.url} (commit: {updated_repo.commit})")
                return
        
        logger.error(f"Repository '{updated_repo.url}' with commit '{updated_repo.commit}' not found for update.")
        raise ValueError(f"Repository '{updated_repo.url}' with commit '{updated_repo.commit}' not found.")

    def delete_repository(self, url: str, commit: str) -> None:
        """
        Delete a repository from the database.
        
        Args:
            url (str): Repository URL.
            commit (str): Commit hash.
            
        Raises:
            ValueError: If repository is not found in the database.
        """
        for i, repo in enumerate(self.repositories):
            if repo.url == url and repo.commit == commit:
                del self.repositories[i]
                logger.info(f"Deleted repository: {url} (commit: {commit})")
                return
        
        raise ValueError(f"Repository '{url}' with commit '{commit}' not found.")

    def print_database_contents(self) -> None:
        """Print a summary of database contents."""
        logger.info("Current database contents:")
        logger.info(f"Total repositories: {len(self.repositories)}")
        
        for repo in self.repositories:
            logger.info(f"  - {repo.url} (commit: {repo.commit})")
            logger.info(f"    Theorems: {repo.total_theorems} "
                       f"(proven: {repo.num_proven_theorems}, "
                       f"sorry: {repo.num_sorry_theorems})")
            logger.info(f"    Premises: {repo.num_premises} files, "
                       f"{repo.num_premise_files} premise files")

    def get_statistics(self) -> Dict[str, int]:
        """
        Get database statistics.
        
        Returns:
            Dict[str, int]: Dictionary with database statistics.
        """
        return {
            "total_repositories": len(self.repositories),
            "total_theorems": sum(repo.total_theorems for repo in self.repositories),
            "total_proven_theorems": sum(repo.num_proven_theorems for repo in self.repositories),
            "total_sorry_theorems": sum(repo.num_sorry_theorems for repo in self.repositories),
            "total_premise_files": sum(repo.num_premise_files for repo in self.repositories),
            "total_premises": sum(repo.num_premises for repo in self.repositories),
            "total_files_traced": sum(repo.num_files_traced for repo in self.repositories),
        }

    def to_dict(self) -> Dict:
        """
        Convert the database to a dictionary.
        
        Returns:
            Dict: Dictionary representation of the database.
        """
        return {
            "repositories": [repo.to_dict() for repo in self.repositories]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DynamicDatabase":
        """
        Create a database from a dictionary.
        
        Args:
            data (Dict): Dictionary containing database data.
            
        Returns:
            DynamicDatabase: New database instance.
            
        Raises:
            ValueError: If data format is invalid.
        """
        if "repositories" not in data:
            raise ValueError("Invalid DynamicDatabase data format")
        
        db = cls()
        for repo_data in data["repositories"]:
            repo = Repository.from_dict(repo_data)
            db.add_repository(repo)
        
        return db

    def to_json(self, file_path: str) -> None:
        """
        Serialize the database to a JSON file.
        
        Args:
            file_path (str): Path to the output JSON file.
        """
        logger.info(f"Saving database to {file_path}")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Database saved successfully to {file_path}")

    @classmethod
    def from_json(cls, file_path: str) -> "DynamicDatabase":
        """
        Load the database from a JSON file.
        
        Args:
            file_path (str): Path to the JSON file.
            
        Returns:
            DynamicDatabase: Database loaded from file.
        """
        logger.info(f"Loading database from {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        db = cls.from_dict(data)
        logger.info(f"Database loaded successfully from {file_path}")
        
        return db

    def update_json(self, file_path: str) -> None:
        """
        Update an existing JSON file with the current database state.
        
        This method loads an existing database file, merges it with the current
        database state, and saves the updated version.
        
        Args:
            file_path (str): Path to the JSON file to update.
        """
        try:
            existing_db = self.from_json(file_path)
            logger.info(f"Loaded existing database with {len(existing_db.repositories)} repositories")
        except FileNotFoundError:
            logger.info("No existing database file found, creating new one")
            existing_db = DynamicDatabase()

        # Update existing database with current repositories
        for repo in self.repositories:
            try:
                existing_db.update_repository(repo)
            except ValueError:
                # Repository doesn't exist, add it
                existing_db.add_repository(repo)

        # Save the updated database
        existing_db.to_json(file_path)
        logger.info(f"Updated database file {file_path}") 