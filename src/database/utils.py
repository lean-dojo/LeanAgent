"""
Utility functions for the LeanAgent database system.

This module contains helper functions for parsing, file operations, and other utilities
used across the database system.
"""

import shutil
from pathlib import Path
from typing import Union, List
from loguru import logger

# Optional import of lean_dojo with fallback
try:
    from lean_dojo.data_extraction.lean import Pos
    logger.info("✅ Successfully imported Pos from lean_dojo in utils")
except (ImportError, Exception) as e:
    # Fallback implementation when lean_dojo is not available or fails to initialize
    logger.warning(f"⚠️  lean_dojo not available in utils ({e}), using fallback Pos implementation")
    from dataclasses import dataclass
    
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


def parse_pos(pos_str: Union[str, List[int]]) -> Pos:
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


def safe_remove_dir_path(dir_path: Union[str, Path]) -> None:
    """
    Safely remove a directory and all its contents.
    
    Args:
        dir_path (str or Path): Path to the directory to remove.
    """
    dir_path = Path(dir_path)
    
    if not dir_path.exists():
        logger.debug(f"Directory {dir_path} does not exist, nothing to remove")
        return
        
    if not dir_path.is_dir():
        logger.warning(f"Path {dir_path} is not a directory")
        return
    
    try:
        shutil.rmtree(dir_path)
        logger.info(f"Successfully removed directory: {dir_path}")
    except PermissionError as e:
        logger.error(f"Permission denied when removing {dir_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error removing directory {dir_path}: {e}")
        raise 