import json
import shutil
import random
import networkx as nx
from copy import copy
from pathlib import Path
from loguru import logger
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Union
import time
import lean_dojo
from lean_dojo import *
from lean_dojo.constants import LEAN4_PACKAGES_DIR
import re
import subprocess
import sys

random.seed(3407)  # https://arxiv.org/abs/2109.08203

RAID_DIR = os.environ.get('RAID_DIR')
SPLIT_NAME = str  # train/val/test
SPLIT = Dict[SPLIT_NAME, List[TracedTheorem]]
SPLIT_STRATEGY = str
_LEAN4_VERSION_REGEX = re.compile(r"leanprover/lean4:(?P<version>.+?)")

def get_lean4_version_from_config(toolchain: str) -> str:
    """Return the required Lean version given a ``lean-toolchain`` config."""
    m = _LEAN4_VERSION_REGEX.fullmatch(toolchain.strip())
    assert m is not None, "Invalid config."
    return m["version"]

def is_supported_version(v) -> bool:
    """
    Check if ``v`` is at least `v4.3.0-rc2` and at most `v4.8.0-rc1`.
    Note: Lean versions are generally not backwards-compatible. Also, the Lean FRO
    keeps bumping the default versions of repos to the latest version, which is
    not necessarily the latest stable version. So, we need to be careful about
    what we choose to support.
    """
    if not v.startswith("v"):
        return False
    v = v[1:]
    major, minor, patch = [int(_) for _ in v.split("-")[0].split(".")]
    if major < 4 or (major == 4 and minor < 3) or (major == 4 and minor > 8) or (major == 4 and minor == 8 and patch > 1):
        return False
    if (
        major > 4
        or (major == 4 and minor > 3)
        or (major == 4 and minor == 3 and patch > 0)
    ):
        return True
    assert major == 4 and minor == 3 and patch == 0
    if "4.3.0-rc" in v:
        rc = int(v.split("-")[1][2:])
        return rc >= 2
    else:
        return True

def _split_sequentially(
    traced_theorems: List[TracedTheorem],
    num_val: int,
    num_test: int
) -> SPLIT:
    """Split ``traced_theorems`` sequentially into train/val/test."""
    num_theorems = len(traced_theorems)
    num_train = num_theorems - num_val - num_test
    return {
        "train": traced_theorems[:num_train],
        "val": traced_theorems[num_train : num_train + num_val],
        "test": traced_theorems[num_train + num_val :],
    }


def split_randomly(
    traced_theorems: List[TracedTheorem],
    num_val: int,
    num_test: int
) -> SPLIT:
    """Split ``traced_theorems`` randomly into train/val/test."""
    logger.info("Splitting the theorems randomly")
    traced_theorems = copy(traced_theorems)
    random.shuffle(traced_theorems)
    return _split_sequentially(traced_theorems, num_val, num_test)

def split_by_premise(
    traced_theorems: List[TracedTheorem],
    num_val: int,
    num_test: int
) -> SPLIT:
    """
    Split theorems into train/val/test so that proofs in val/test rely on at
    least one novel premise that does not appear in train.
    """
    logger.info("Splitting the theorems by premises")

    # Figure out the number of theorems in train/val/test.
    num_theorems = len(traced_theorems)
    num_val_test = num_val + num_test
    num_train = num_theorems - num_val_test
    theorems_val_test = set()

    # Map each premise to a list of theorems using it.
    theorems_by_premises = defaultdict(list)
    for t in traced_theorems:
        for p in t.get_premise_full_names():
            theorems_by_premises[p].append(t)

    # Sort the premises by the number of theorems using them (in ascending order).
    theorems_by_premises = sorted(theorems_by_premises.items(), key=lambda x: len(x[1]))

    # For each premise, put all theorems using it into val_test so that it does not appear in train.
    for _, thms in theorems_by_premises:
        if len(theorems_val_test) < num_val_test:
            theorems_val_test.update(thms)

    # All other theorems go to train.
    theorems_train = [t for t in traced_theorems if t not in theorems_val_test]
    theorems_val_test = list(theorems_val_test)
    random.shuffle(theorems_val_test)

    return {
        "train": theorems_train,
        "val": theorems_val_test[:num_val],
        "test": theorems_val_test[num_val:],
    }

def split_data(traced_repo: TracedRepo, num_val_pct: float = 0.02, num_test_pct: float = 0.02) -> Dict[SPLIT_STRATEGY, SPLIT]:
    """
    Split the traced theorems into training, validation, and test sets.

    This function extracts theorems from the provided TracedRepo object, excluding
    theorems from the Lean 4 repository itself. The theorems are then split using 
    multiple strategies, including random splitting and splitting by novel premises.

    Args:
        traced_repo (TracedRepo): A TracedRepo object containing the theorems to split.
        num_val_pct (float, optional): The percentage of theorems to use for validation.
            Defaults to 0.02 (2%).
        num_test_pct (float, optional): The percentage of theorems to use for testing.
            Defaults to 0.02 (2%).

    Returns:
        Dict[SPLIT_STRATEGY, SPLIT]: A dictionary where keys are split strategy names
            ("random", "novel_premises") and values are split dictionaries containing
            "train", "val", and "test" theorem lists.

    Example:
        >>> repo = TracedRepo(...)
        >>> splits = split_data(repo)
        >>> len(splits["random"]["train"])  # Number of training theorems in random split
    """
    # Skip theorems in the Lean 4 repo itself.
    traced_theorems = [
        thm for thm in traced_repo.get_traced_theorems() if not thm.repo.is_lean4
    ]
    logger.info(f"{len(traced_theorems)} theorems in total")
    num_theorems = len(traced_theorems)
    num_val = int(num_theorems * num_val_pct)
    num_test = int(num_theorems * num_test_pct)

    logger.info(f"{num_theorems} theorems in total, with {num_val} for validation and {num_test} for testing")

    return {
        "random": split_randomly(traced_theorems, num_val, num_test),
        "novel_premises": split_by_premise(traced_theorems, num_val, num_test),
    }

def _get_file_path(traced_repo: TracedRepo, thm: TracedTheorem) -> str:
    """
    Get the file path for a given theorem in a traced repository.

    This function computes the appropriate file path for a theorem, handling two cases:
    1. When the theorem belongs to the traced repository itself
    2. When the theorem belongs to one of the dependencies of the repository

    Args:
        traced_repo (TracedRepo): The traced repository context
        thm (TracedTheorem): The theorem whose file path is requested

    Returns:
        str: The file path to the theorem

    Raises:
        ValueError: If the theorem belongs to a repository that is not the traced repository
                    or one of its dependencies
    """
    if thm.repo == traced_repo.repo:
        # The theorem belongs to the traced repo itself.
        return str(thm.theorem.file_path)
    else:
        # The theorem belongs to one of the dependencies.
        for name, dep in traced_repo.dependencies.items():
            if dep == thm.repo:
                return f"{LEAN4_PACKAGES_DIR}/{name}/{thm.theorem.file_path}"
        raise ValueError(f"Unable to find the dependency {thm.repo}")


def export_proofs(
    splits: Dict[SPLIT_STRATEGY, SPLIT], dst_path: Path, traced_repo: TracedRepo
) -> None:
    """
    Export proofs from a traced repository to the specified destination path.
    This function processes the given splits (organized by strategy) and writes the theorem proofs 
    to JSON files in the destination directory. Each theorem is exported with its metadata, 
    including URL, commit, file path, theorem statement, and traced tactics.
    Args:
        splits: Dictionary mapping split strategies to actual splits. Each split maps dataset
               names (like 'train', 'val', 'test') to lists of theorems.
        dst_path: Destination path where proof files will be written.
        traced_repo: The traced repository containing the theorem proofs.
    Returns:
        int: The total number of theorems exported.
    Each theorem is exported with the following information:
    - URL and commit of the source repository
    - File path within the repository
    - Full theorem name
    - Theorem statement (if available)
    - Start and end positions
    - Traced tactics with their states before and after application
    Tactics with "no goals" state or containing "·" are excluded.
    """
    total_theorems = 0
    for strategy, split in splits.items():
        split_dir = dst_path / strategy
        split_dir.mkdir(parents=True)

        for name, theorems in split.items():
            data = []
            num_tactics = 0

            for thm in theorems:
                tactics = [
                    {
                        "tactic": t.tactic,
                        "annotated_tactic": t.get_annotated_tactic(),
                        "state_before": t.state_before,
                        "state_after": t.state_after,
                    }
                    for t in thm.get_traced_tactics()
                    if t.state_before != "no goals"
                    and "·" not in t.tactic  # Ignore "·".
                ]
                num_tactics += len(tactics)

                theorem_statement = None
                if thm.has_tactic_proof() and thm.get_tactic_proof() is not None:
                    theorem_statement = thm.get_theorem_statement()
                
                data.append(
                    {
                        "url": traced_repo.repo.url,
                        "commit": traced_repo.repo.commit,
                        "file_path": _get_file_path(traced_repo, thm),
                        "full_name": thm.theorem.full_name,
                        "theorem_statement": theorem_statement,
                        "start": list(thm.start),
                        "end": list(thm.end),
                        "traced_tactics": tactics,
                    }
                )
            oup_path = split_dir / f"{name}.json"
            json.dump(data, oup_path.open("wt"))
            logger.info(
                f"{len(theorems)} theorems and {num_tactics} tactics saved to {oup_path}"
            )
            total_theorems += len(theorems)
    logger.info(f"{total_theorems} theorems saved in total")
    return total_theorems


def export_premises(traced_repo: TracedRepo, dst_path: Path) -> None:
    """
    Export all premise definitions in a traced repository to the destination path.
    This function performs two main operations:
    1. Exports all premises (theorems/definitions) to a corpus.jsonl file, maintaining topological order
    2. Exports information about all traced files to traced_files.jsonl
    Args:
        traced_repo: A TracedRepo object containing the traced files and their dependencies
        dst_path: Path to the directory where the output files will be saved
    Returns:
        A tuple containing:
        - The number of premises exported
        - The number of traced files processed
    """
    oup_path = dst_path / "corpus.jsonl"
    num_premises = 0

    with oup_path.open("wt") as oup:
        G = traced_repo.traced_files_graph

        for tf_node in reversed(list(nx.topological_sort(G))):
            tf = G.nodes[tf_node]["traced_file"]
            imports = [str(_) for _ in G.successors(tf_node)]
            premises = tf.get_premise_definitions()
            num_premises += len(premises)
            oup.write(
                json.dumps(
                    {"path": str(tf.path), "imports": imports, "premises": premises}
                )
                + "\n"
            )
    logger.info(
        f"{num_premises} theorems/definitions from {len(traced_repo.traced_files)} files saved to {oup_path}"
    )
    
    oup_path = dst_path / "traced_files.jsonl"
    with oup_path.open("wt") as oup:
        for traced_file in traced_repo.traced_files:
            source_file = traced_file.lean_file
            source_file_path = source_file.path
            oup.write(
                json.dumps(
                    {"traced_file_path": str(source_file_path)}
                )
                + "\n"
            )

    return num_premises, len(traced_repo.traced_files)


def export_metadata(traced_repo: TracedRepo, dst_path: Path, **kwargs) -> None:
    """
    Export the metadata of a traced repository to a specified destination path.

    This function creates a JSON file containing metadata information about the repository,
    including user-provided keyword arguments, creation timestamp, repository URL and commit,
    and LeanDojo version.

    Parameters
    ----------
    traced_repo : TracedRepo
        The traced repository object containing repository information.
    dst_path : Path
        Destination directory path where the metadata file will be saved.
    **kwargs : dict
        Additional metadata key-value pairs to include in the output.

    Returns
    -------
    None
        The function writes metadata to a JSON file but doesn't return any value.

    Notes
    -----
    The metadata is saved as 'metadata.json' in the specified destination directory.
    """
    metadata = dict(kwargs)
    metadata["creation_time"] = str(datetime.now())
    metadata["from_repo"] = {
        "url": traced_repo.repo.url,
        "commit": traced_repo.repo.commit,
    }
    metadata["leandojo_version"] = lean_dojo.__version__
    json.dump(metadata, (dst_path / "metadata.json").open("wt"))


def safe_remove_dir(dir_path):
    """
    Safely removes a directory if it exists.
    
    This function attempts to remove the specified directory, with multiple retries
    in case of permission errors. A warning is logged if the directory already exists.
    
    Args:
        dir_path (str): Path to the directory to be removed.
        
    Raises:
        PermissionError: If the directory cannot be removed after multiple attempts
                         due to permission issues.
    
    Note:
        The function will retry up to 5 times with a 0.1 second delay between attempts
        if a PermissionError occurs.
    """
    if os.path.exists(dir_path):
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


def safe_remove_dir_path(dir_path):
    """
    Safely removes a directory and all its contents if it exists.
    
    Uses multiple attempts with a small delay between them to handle potential
    permission errors that might occur on some systems when removing directories.
    
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

def export_data(
    traced_repo: TracedRepo,
    splits: Dict[SPLIT_STRATEGY, SPLIT],
    dst_path: Union[str, Path],
    **kwargs,
) -> None:
    """Export a traced repository's content to a specified destination path.
    
    This function exports proofs, premises, licenses, and metadata from a traced 
    repository to a specified destination path. The repository's theorems should have 
    been split using a strategy defined in `splits`.
    
    Args:
        traced_repo: The traced repository containing the data to export.
        splits: Dictionary mapping split strategies to their corresponding splits.
        dst_path: Destination path where the data will be exported. Can be a string or Path object.
        **kwargs: Additional keyword arguments to pass to export_metadata.
    
    Returns:
        tuple: A tuple containing (number of premises, number of files traced, total theorems exported).
        
    Note:
        Any existing content at the destination path will be removed.
    """
    if isinstance(dst_path, str):
        dst_path = Path(dst_path)
    safe_remove_dir_path(dst_path)

    # Export the proofs.
    total_theorems = export_proofs(splits, dst_path, traced_repo)
    logger.info("Successfully exported the proofs")

    # Export the premises (theorems, definitions, etc.).
    num_premises, num_files_traced = export_premises(traced_repo, dst_path)
    logger.info("Successfully exported the premises")

    # Export metadata.
    export_metadata(traced_repo, dst_path, **kwargs)
    logger.info("Successfully exported the metadata")

    return num_premises, num_files_traced, total_theorems

def configure_leandojo():
    """
    Configure the LeanDojo environment for benchmarking.
    
    This function sets up the logger configuration for LeanDojo and displays
    important environment variables including the current working directory
    and various constants related to process management.
    
    It removes any existing logger handlers and adds a new handler for stderr
    with DEBUG level logging.
    
    No parameters are required, and the function does not return any values.
    """
    constants.logger.remove()
    constants.logger.add(sys.stderr, level="DEBUG")

    # constants.NUM_WORKERS = 1
    # constants.MAX_NUM_PROCS = 2 # 1 worker + 1 main process
    # constants.NUM_PROCS = 2

    logger.info(f"Current working directory: {os.getcwd()}")

def main(url, commit, dst_dir):
    """
    Generates a benchmark dataset for Lean 4 proofs from a specified repository.
    This function clones a Lean 4 repository, configures the appropriate Lean toolchain
    version, traces the repository using LeanDojo, and exports the trace data to a 
    designated directory.
    Args:
        url (str): The URL of the Lean 4 Git repository to clone
        commit (str): The specific commit hash to check out
        dst_dir (str): The destination directory where the benchmark dataset will be saved
    Returns:
        tuple: A tuple containing:
            - traced_repo: The traced repository object or None if tracing failed
            - num_premises (int): Number of premises found
            - num_files_traced (int): Number of files successfully traced
            - total_theorems (int): Total number of theorems, or 10 if tracing failed
    Raises:
        Various exceptions may be caught and logged during repository tracing
    """
    logger.info(f"Generating dataset to go into {dst_dir}")
    repo = LeanGitRepo(url, commit)

    # we need to change the toolchain version that LeanAgent uses
    # to match the repo we are currently tracing
    config = repo.get_config("lean-toolchain")
    logger.info(f"lean toolchain version: {config}")
    v = get_lean4_version_from_config(config["content"])
    logger.info(f"lean version v: {v}")
    logger.info(f"is supported: {is_supported_version(v)}")
    if not is_supported_version(v):  # Won't get here since we checked for a compatible commit, but sanity check in case
        logger.info("Unsupported version")
    v = v[1:] # ignore "v" at beginning
    
    # TODO: So that user doesn't have to set the path manually when needed, we can move this to a config file.
    lean_dir2 = f"/.elan/toolchains/leanprover--lean4---{v}"
    lean_dir3 = f"~/.elan/toolchains/leanprover--lean4---{v}"
    logger.info(f"lean path2 {lean_dir2}")
    logger.info(f"lean path3 {lean_dir3}")
    if not os.path.exists(lean_dir2):
        logger.info(f"Lean toolchain path 2 does not exist: {lean_dir2}")
    if not os.path.exists(lean_dir3):
        logger.info(f"Lean toolchain path 3 does not exist: {lean_dir3}")
    os.environ['LEAN4_PATH'] = lean_dir2
    os.environ['PATH'] = f"{lean_dir2}/bin:{os.environ.get('PATH', '')}"
    logger.info(f"Switched to Lean toolchain at: {lean_dir2}")

    logger.info(f"lean --version: {subprocess.run(['lean', '--version'], capture_output=True).stdout.decode('utf-8')}")
    logger.info(f"repo: {repo}")

    logger.info("Configuring LeanDojo again...")
    configure_leandojo()
    logger.info("LeanDojo configured")

    try:
        logger.info("Tracing the repo...")
        traced_repo = trace(repo)
        logger.info("Successfully traced the repo")
    except Exception as e:
        logger.info(f"Failed to trace repo {repo} because of {e}")
        return None, 0, 0, 10
    safe_remove_dir(dst_dir)
    splits = split_data(traced_repo)
    logger.info("Successfully split the data")
    num_premises, num_files_traced, total_theorems = export_data(traced_repo, splits, dst_dir)
    logger.info("Successfully exported the data")
    return traced_repo, num_premises, num_files_traced, total_theorems
