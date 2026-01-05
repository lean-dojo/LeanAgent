import fcntl
import json
import os
import pickle
import random

import sys
import time
import traceback

from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Set, Tuple


import numpy as np
import pytorch_lightning as pl
import ray

import torch
from lean_dojo import *
from lean_dojo import LeanGitRepo, Pos
from lean_dojo import Theorem
from lean_dojo import Theorem as LeanDojoTheorem

from loguru import logger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from tqdm import tqdm

import generate_benchmark_lean4
from dynamic_database import AnnotatedTactic, Theorem, DynamicDatabase
from prover.proof_search import DistributedProver, SearchResult, Status
from retrieval.datamodule import RetrievalDataModule
from retrieval.main import run_cli
from retrieval.model import PremiseRetriever

from git_utils import find_and_save_compatible_commits, search_github_repositories, should_skip_repo, add_repo_to_database, sort_repositories_by_difficulty, save_sorted_repos
from filenames import DATA_DIR, RAID_DIR, CHECKPOINT_DIR, EVAL_RESULTS_FILE_PATH, DB_FILE_NAME, PROOF_LOG_FILE_NAME, ENCOUNTERED_THEOREMS_FILE, FISHER_DIR

# Set the seed for reproducibility
personal_access_token = os.environ.get("GITHUB_ACCESS_TOKEN")

random.seed(3407)  # https://arxiv.org/abs/2109.08203
BATCH_SIZE = 4


repos_for_merged_dataset = []
repos_for_proving = []
lean_git_repos = []
repos = []

SEED_REPOS = [
    LeanGitRepo(
        "https://github.com/ImperialCollegeLondon/FLT",
        "b208a302cdcbfadce33d8165f0b054bfa17e2147",
    ),
    LeanGitRepo(
        "https://github.com/HEPLean/PhysLean",
        "60f1ebc3eb015f78a3719ee4085344a600d0af50",
    ),
    LeanGitRepo(
        "https://github.com/verse-lab/veil",
        "a9fe7205c57f7b6ee8b350bfc87b9b4b28c57781",
    ),
]


@contextmanager
def _locked(path: str, mode: str):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, mode) as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield handle
        finally:
            if any(flag in mode for flag in ("w", "a", "+")):
                handle.flush()
                os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_json_locked(path: str):
    with _locked(path, "r") as handle:
        return json.load(handle)


def write_json_locked(
    path: str,
    obj,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
) -> None:
    with _locked(path, "w") as handle:
        json.dump(
            obj,
            handle,
            indent=indent,
            ensure_ascii=ensure_ascii,
            sort_keys=sort_keys,
        )


def save_database_locked(db: DynamicDatabase, path: str) -> None:
    """Persist the dynamic database safely across processes."""
    write_json_locked(path, db.to_dict(), ensure_ascii=False)


def append_text_locked(path: str, *chunks: str) -> None:
    """Atomically append textual data while coordinating concurrent writers."""
    with _locked(path, "a") as handle:
        handle.writelines(chunks)
def _eval(data, preds_map) -> Tuple[float, float, float]:
    """Evaluates the retrieval model."""
    R1 = []
    R10 = []
    MRR = []

    for thm in tqdm(data):
        for i, _ in enumerate(thm["traced_tactics"]):
            pred = None
            key = (thm["file_path"], thm["full_name"], tuple(thm["start"]), i)
            if key in preds_map:
                pred = preds_map[key]
            else:
                continue
            
            all_pos_premises = set(pred["all_pos_premises"])
            if len(all_pos_premises) == 0:
                continue

            retrieved_premises = pred["retrieved_premises"]
            TP1 = retrieved_premises[0] in all_pos_premises
            R1.append(float(TP1) / len(all_pos_premises))
            TP10 = len(all_pos_premises.intersection(retrieved_premises[:10]))
            R10.append(float(TP10) / len(all_pos_premises))

            for j, p in enumerate(retrieved_premises):
                if p in all_pos_premises:
                    MRR.append(1.0 / (j + 1))
                    break
            else:
                MRR.append(0.0)

    R1 = 100 * np.mean(R1)
    R10 = 100 * np.mean(R10)
    MRR = np.mean(MRR)
    return R1, R10, MRR


def load_fisher_information(file_path):
    """Loads the Fisher Information Matrix."""
    try:
        with open(file_path, "rb") as f:
            fisher_info = pickle.load(f)
        logger.info("Fisher Information successfully loaded.")
        return fisher_info
    except FileNotFoundError:
        logger.error(f"No Fisher Information file found at {file_path}.")
        return None


def find_latest_checkpoint():
    """Finds the most recent checkpoint."""
    all_checkpoints = [
        os.path.join(CHECKPOINT_DIR, f)
        for f in os.listdir(CHECKPOINT_DIR)
        if f.endswith(".ckpt")
    ]
    if len(all_checkpoints) == 0:
        raise FileNotFoundError("No checkpoints found.")
    
    latest_checkpoint = max(all_checkpoints, key=os.path.getmtime)
    logger.info(f"Using the latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint


def find_latest_fisher():
    """Finds the most recent Fisher Information Matrix."""
    all_fisher = [
        os.path.join(FISHER_DIR, f)
        for f in os.listdir(FISHER_DIR)
        if f.endswith(".pkl")
    ]
    if len(all_fisher) == 0:
        raise FileNotFoundError("No Fisher Information Matrices found.")
    
    latest_fisher = max(all_fisher, key=os.path.getmtime)
    logger.info(f"Using the latest Fisher Information Matrix: {latest_fisher}")
    return latest_fisher


def theorem_identifier(
    theorem: Theorem,
) -> Tuple[str, str, Tuple[int, int], Tuple[int, int]]:
    """Returns a unique identifier for a theorem."""
    return (
        theorem.full_name,
        str(theorem.file_path),
        tuple(theorem.start),
        tuple(theorem.end),
    )


def process_theorem_batch(
    theorem_batch, positions_batch, repo, db, prover, dynamic_database_json_path
):
    """Processes a batch of theorems."""
    lean_dojo_theorems = [t[1] for t in theorem_batch]
    results = prover.search_unordered(
        LeanGitRepo(repo.url, repo.commit), lean_dojo_theorems, positions_batch
    )

    # Create a mapping from LeanDojoTheorem to our Theorem
    theorem_map = {ldj_thm: thm for thm, ldj_thm in theorem_batch}

    for result in results:
        if isinstance(result, SearchResult):
            if result.theorem in theorem_map:
                theorem = theorem_map[result.theorem]
                if result.status == Status.PROVED:
                    logger.info(f"Proof found for {theorem.full_name}")
                    traced_tactics = [
                        AnnotatedTactic(
                            tactic=tactic,
                            annotated_tactic=(tactic, []),
                            state_before="",
                            state_after="",
                        )
                        for tactic in result.proof
                    ]
                    theorem.traced_tactics = traced_tactics
                    repo.change_sorry_to_proven(theorem, PROOF_LOG_FILE_NAME)
                    db.update_repository(repo)
                    logger.info(f"Updated theorem {theorem.full_name} in the database")
                else:
                    logger.info(f"No proof found for {theorem.full_name}")
            else:
                logger.warning(f"Theorem not found in theorem_map: {result.theorem}")
        else:
            logger.warning(f"Unexpected result type")

    save_database_locked(db, dynamic_database_json_path)


def save_progress(all_encountered_theorems):
    """Saves the set of encountered theorems."""
    logger.info("Saving encountered theorems...")
    with open(ENCOUNTERED_THEOREMS_FILE, "wb") as f:
        pickle.dump(all_encountered_theorems, f)


def load_encountered_theorems(file_path):
    """Loads the theorems that have been encountered."""
    all_encountered_theorems = set()
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                file_content = f.read()
                if file_content:  # Check if the file is not empty
                    all_encountered_theorems = pickle.loads(file_content)
                else:
                    logger.warning(
                        f"The file {file_path} is empty. Starting with an empty set."
                    )
        except (EOFError, pickle.UnpicklingError) as e:
            logger.warning(
                f"Error reading {file_path}: {e}. Starting with an empty set."
            )
        except Exception as e:
            logger.error(
                f"Unexpected error when reading {file_path}: {e}. Starting with an empty set."
            )
    else:
        logger.info(f"The file {file_path} does not exist. Starting with an empty set.")

    return all_encountered_theorems


def prove_sorry_theorems(
    db: DynamicDatabase,
    prover: DistributedProver,
    dynamic_database_json_path,
    repos_to_include: Optional[List[Tuple[str, str]]] = None,
    batch_size: int = 12,
):
    """Proves sorry theorems."""
    repos_to_process = (
        db.repositories
        if repos_to_include is None
        else [
            repo
            for repo in db.repositories
            if (repo.url, repo.commit) in repos_to_include
        ]
    )

    # To avoid proving the same theorem multiple times, potentially from different versions of the
    # same repo, we sort the repositories
    repos_to_process.sort(key=lambda r: r.metadata["date_processed"], reverse=True)

    processed_theorems: Set[Tuple[str, str, Tuple[int, int], Tuple[int, int]]] = set()
    all_encountered_theorems: Set[Tuple[str, str, Tuple[int, int], Tuple[int, int]]] = (
        set()
    )
    last_save_time = datetime.now()
    save_interval = timedelta(minutes=30)

    # Load previously encountered theorems
    all_encountered_theorems = load_encountered_theorems(ENCOUNTERED_THEOREMS_FILE)

    for repo in repos_to_process:
        sorry_theorems = repo.sorry_theorems_unproved
        repo_url = repo.url
        repo_commit = repo.commit

        logger.info(f"Found {len(sorry_theorems)} sorry theorems to prove")

        theorem_batch = []
        positions_batch = []

        for theorem in tqdm(
            sorry_theorems, desc=f"Processing theorems from {repo.name}", unit="theorem"
        ):
            # Ignore sorry theorems from the repo's dependencies
            if theorem.url != repo_url or theorem.commit != repo_commit:
                continue

            theorem_id = theorem_identifier(theorem)

            if theorem_id in all_encountered_theorems:
                logger.info(
                    f"Skipping already encountered theorem: {theorem.full_name}"
                )
                continue

            all_encountered_theorems.add(theorem_id)
            if theorem_id in processed_theorems:
                logger.info(f"Skipping already processed theorem: {theorem.full_name}")
                continue

            processed_theorems.add(theorem_id)

            logger.info(f"Searching for proof for {theorem.full_name}")
            logger.info(f"Position: {theorem.start}")

            # Convert our Theorem to LeanDojo Theorem
            lean_dojo_theorem = LeanDojoTheorem(
                repo=LeanGitRepo(repo_url, repo_commit),
                file_path=theorem.file_path,
                full_name=theorem.full_name,
            )

            theorem_batch.append((theorem, lean_dojo_theorem))
            positions_batch.append(Pos(*theorem.start))

            if len(theorem_batch) == batch_size:
                process_theorem_batch(
                    theorem_batch,
                    positions_batch,
                    repo,
                    db,
                    prover,
                    dynamic_database_json_path,
                )
                theorem_batch = []
                positions_batch = []

            current_time = datetime.now()
            if current_time - last_save_time >= save_interval:
                save_progress(all_encountered_theorems)
                last_save_time = current_time

        # Process any remaining theorems in the last batch
        if theorem_batch:
            process_theorem_batch(
                theorem_batch,
                positions_batch,
                repo,
                db,
                prover,
                dynamic_database_json_path,
            )

    save_progress(all_encountered_theorems)
    logger.info("Finished attempting to prove sorry theorems")
    
def replace_sorry_with_proof(proofs):
    """Replace the `sorry` with the proof text in the Lean files."""
    logger.info(f"Replacing sorries with {len(proofs)} proofs!")
    # Group proofs by file paths
    proofs_by_file = {}
    for proof in proofs:
        file_path, start, end, proof_text, theorem_name = proof
        if file_path not in proofs_by_file:
            proofs_by_file[file_path] = []
        proofs_by_file[file_path].append((start, end, proof_text))

    for file_path, proofs in proofs_by_file.items():
        with open(file_path, "r") as file:
            lines = file.readlines()

        # sort proof by starting line and column number (working bottom up retains positions)
        proofs.sort(key=lambda x: (x[0].line_nb, x[0].column_nb), reverse=True)

        for start, end, proof_text in proofs:
            start_line, start_col = start.line_nb - 1, start.column_nb - 1
            end_line, end_col = end.line_nb - 1, end.column_nb - 1
            original_text = "".join(lines[start_line : end_line + 1])
            new_text = original_text.replace("sorry", proof_text, 1)
            lines[start_line : end_line + 1] = new_text

            with open(file_path, "w") as file:
                file.writelines(lines)

    logger.info("Finished replacing sorries with proofs!")

def initialize_database(dynamic_database_json_path: str) -> DynamicDatabase:
    """Initializes or loads the dynamic database."""
    # Check if the current process is the main one
    is_main_process = int(os.environ.get("LOCAL_RANK", "0")) == 0

    # Initialize the database if it doesn't exist or is empty
    if is_main_process:
        logger.info("Starting the main process")
        if (
            not os.path.exists(dynamic_database_json_path)
            or os.path.getsize(dynamic_database_json_path) == 0
        ):
            # File doesn't exist or is empty, initialize it
            logger.info(
                f"\nInitializing new database at {dynamic_database_json_path}\n"
            )
            db = DynamicDatabase()
            save_database_locked(db, dynamic_database_json_path)
        else:
            try:
                logger.info(f"Loading database from {dynamic_database_json_path}")
                db = DynamicDatabase.from_json(dynamic_database_json_path)
                logger.info(f"Loaded database from {dynamic_database_json_path}")
            except json.JSONDecodeError:
                # If there's an error decoding the JSON, initialize a new database
                logger.warning(
                    f"Error decoding JSON from {dynamic_database_json_path}. Initializing new database."
                )
                db = DynamicDatabase()
                save_database_locked(db, dynamic_database_json_path)
    
    return db

def get_repos(curriculum_learning: bool, num_repos: int, dynamic_database_json_path: str, db: DynamicDatabase):
    global lean_git_repos
    global repos
    # If curriculum learning is enabled, initialize repositories and sort them by difficulty
    repo_info_file = os.path.join(DATA_DIR, "repo_info_compatible.json")
    # Check if the current process is the main one
    is_main_process = int(os.environ.get("LOCAL_RANK", "0")) == 0
    if curriculum_learning:
        logger.info("Starting curriculum learning")
        if is_main_process:
            if num_repos < 3:
                logger.warning("num_repos should be at least 3 for curriculum learning")

            failure_records: List[Tuple[str, str]] = []

            for seed_repo in SEED_REPOS:
                if db.get_repository(seed_repo.url, seed_repo.commit) is None:
                    logger.info(
                        f"Seeding database with {seed_repo.url}@{seed_repo.commit}"
                    )
                    result = add_repo_to_database(
                        dynamic_database_json_path, seed_repo, db
                    )
                    if result in ("success", "already_present"):
                        logger.info(f"Seeded repo {seed_repo.url}")
                    else:
                        failure_records.append((seed_repo.url, result))
                else:
                    logger.info(
                        f"Seed repository {seed_repo.url}@{seed_repo.commit} already present"
                    )

            existing_repo_count = len(db.repositories)
            target_repo_count = max(3, num_repos)

            lean_git_repos, repos = search_github_repositories(
                lean_git_repos, repos, "Lean", target_repo_count
            )

            processed_idx = 0
            extra_searches = 0
            max_extra_searches = 10

            while len(db.repositories) < target_repo_count:
                while (
                    processed_idx < len(lean_git_repos)
                    and len(db.repositories) < target_repo_count
                ):
                    repo = lean_git_repos[processed_idx]
                    processed_idx += 1
                    print("\n\n")
                    logger.info(f"Processing new repo: {repo.url}")
                    result = add_repo_to_database(
                        dynamic_database_json_path, repo, db
                    )
                    if result in ("success", "already_present"):
                        logger.info(f"Successfully added repo {repo.url}")
                    else:
                        failure_records.append((repo.url, result))

                if len(db.repositories) >= target_repo_count:
                    break

                if extra_searches >= max_extra_searches:
                    raise ValueError(
                        "Unable to find enough compatible repositories for curriculum learning"
                    )

                extra_searches += 1
                needed = max(1, target_repo_count - len(db.repositories))
                logger.info(
                    f"Searching for {needed} additional repositories to meet the curriculum requirement"
                )
                lean_git_repos, repos = search_github_repositories(
                    lean_git_repos, repos, "Lean", needed
                )

            if failure_records:
                logger.warning("Tracing failures/skip summary:")
                for repo_url, reason in failure_records:
                    logger.warning(f"  {repo_url} -> {reason}")

            newly_added = len(db.repositories) - existing_repo_count
            logger.info(
                f"Successfully added {newly_added} repositories to the database (total: {len(db.repositories)})"
            )

            sorted_repos, categorized_theorems, percentiles = (
                sort_repositories_by_difficulty(db)
            )
            
            print("Sorted repositories. Saving now...")
            save_database_locked(db, dynamic_database_json_path)
            save_sorted_repos(sorted_repos, "sorted_repos.json")
            
            print("Summary of theorem difficulties by URL:")
            for repo in sorted_repos:
                print(f"\nURL: {repo.url}")
                for category in ["Easy", "Medium", "Hard", "Hard (No proof)"]:
                    theorems = categorized_theorems[repo][category]
                    print(f"  {category}: {len(theorems)} theorems")
                    if theorems:
                        sorted_theorems = sorted(
                            theorems,
                            key=lambda x: (
                                x[2] if x[2] is not None else -float("inf")
                            ),
                            reverse=True,
                        )[:3]
                        for name, path, _start, _end, diff in sorted_theorems:
                            diff_str = f"{diff:.2f}" if diff is not None else "N/A"
                            print(
                                f"    - {name} (File: {path}, Difficulty: {diff_str})"
                            )

            print("\nOverall Statistics:")
            total_theorems = sum(
                len(theorems)
                for categories in categorized_theorems.values()
                for theorems in categories.values()
            )
            for category in ["Easy", "Medium", "Hard", "Hard (No proof)"]:
                count = sum(
                    len(categories[category])
                    for categories in categorized_theorems.values()
                )
                percentage = (count / total_theorems) * 100
                print(f"{category}: {count} theorems ({percentage:.2f}%)")

            print(
                f"\nPercentile thresholds: Easy <= {percentiles[0]:.2f}, Medium <= {percentiles[1]:.2f}, Hard > {percentiles[1]:.2f}"
            )

            logger.info("Finding compatible repositories...")
            updated_repos = find_and_save_compatible_commits(repo_info_file, sorted_repos)
            lean_git_repos = [LeanGitRepo(repo["url"], repo["commit"]) for repo in updated_repos]
            logger.info("Finished finding compatible repositories")
    else:
        logger.info("Starting without curriculum learning")
        if is_main_process:
            lean_git_repos, repos = search_github_repositories(lean_git_repos, repos, "lean", num_repos)

            for i in range(len(lean_git_repos)):
                repo = lean_git_repos[i]
                logger.info(f"Processing {repo.url}")
                result = add_repo_to_database(dynamic_database_json_path, repo, db)
                if result is not None:
                    logger.info(f"Successfully added repo {repo.url}")
            
            logger.info(f"Successfully added {num_repos} repositories to the database")

            logger.info("Finding compatible repositories...")
            updated_repos = find_and_save_compatible_commits(repo_info_file, lean_git_repos)
            lean_git_repos = [LeanGitRepo(repo["url"], repo["commit"]) for repo in updated_repos]
            logger.info("Finished finding compatible repositories")
    
    return lean_git_repos, repos, updated_repos

def main():
    """
    Main function to run LeanAgent.
    """
    global repos_for_merged_dataset
    global repos_for_proving
    global lean_git_repos
    global repos
    try:
        current_epoch = 0
        epochs_per_repo = 1
        run_progressive_training = True
        use_fisher = True  # FIXED: Enable EWC for lifelong learning
        single_repo = False  # FIXED: Enable cumulative learning across repos
        curriculum_learning = True
        num_repos = 23  # FIXED: Full paper reproduction (was 3)
        dynamic_database_json_path = os.path.join(RAID_DIR, DB_FILE_NAME)

        lambdas = None
        if run_progressive_training:
            logger.info("Running progressive training")
            lambdas = [0.1]
        else:
            logger.info("Running retrieval baseline")
            lambdas = [0.0]

        # Add debug information
        logger.info("Configuring LeanDojo...")
        generate_benchmark_lean4.configure_leandojo()
        logger.info("LeanDojo configured")

        db = initialize_database(dynamic_database_json_path)
        # FIXED: Create required directories
        os.makedirs(FISHER_DIR, exist_ok=True)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        logger.info(f"Found {num_repos} repositories")

        lean_git_repos, repos, updated_repos = get_repos(curriculum_learning, num_repos, dynamic_database_json_path, db)

        repo_info_file = os.path.join(DATA_DIR, "repo_info_compatible.json")
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                repo_info = read_json_locked(repo_info_file)
                break
            except (json.JSONDecodeError, FileNotFoundError):
                if attempt == max_attempts - 1:
                    raise Exception(
                        "Failed to read repository information after multiple attempts"
                    )
                time.sleep(1)

        # Load compatible repositories
        lean_git_repos = [
            LeanGitRepo(info["url"].replace(".git", ""), info["commit"])
            for info in repo_info
        ]

        is_main_process = int(os.environ.get("LOCAL_RANK", "0")) == 0
        
        # Iterate over each repository and lambda value
        for i in range(num_repos):
            for lambda_value in lambdas:
                logger.info(f"length of lean_git_repos: {len(lean_git_repos)}")
                logger.info(f"i: {i}")
                repo = lean_git_repos[i]
                sha = repo.commit
                dir_name = repo.url.split("/")[-1] + "_" + sha
                result = True
                if is_main_process:
                    logger.info("Main process")
                    logger.info(f"Using lambda = {lambda_value}")
                    logger.info(f"Processing {repo.url}")

                    if single_repo:
                        repos_for_merged_dataset = []
                    repos_for_proving = []

                    # Create a directory for the merged dataset if it doesn't exist
                    # FIXED: Use DATA_DIR directly (it's already full path)
                    dst_dir = Path(DATA_DIR) / f"merged_with_new_{dir_name}"
                    if (repo.url, repo.commit) not in repos_for_merged_dataset:
                        logger.info("Adding repo to repos_for_merged_dataset")
                        repos_for_merged_dataset.append((repo.url, repo.commit))
                        if not single_repo:
                            repos_for_proving.append((repo.url, repo.commit))
                    else:
                        logger.info("Repo already in repos_for_merged_dataset")

                    db.generate_merged_dataset(dst_dir, repos_for_merged_dataset)

                dst_dir = os.path.join(DATA_DIR, f"merged_with_new_{dir_name}")
                new_data_path = dst_dir

                logger.info("All GPUs")
                model_checkpoint_path = None
                best_model = None
                data_module = None
                if run_progressive_training:
                    try:
                        model_checkpoint_path = find_latest_checkpoint()
                        logger.info(f"Found latest checkpoint: {model_checkpoint_path}")
                    except FileNotFoundError as e:
                        logger.error(str(e))
                        model_checkpoint_path = f"{RAID_DIR}/checkpoints/mathlib4_29dcec074de168ac2bf835a77ef68bbe069194c5.ckpt"

                    # Train the model on the new dataset that we generated from the dynamic database.
                    logger.info("Inside train_test_fisher")
                    logger.info(f"Starting training at epoch {current_epoch}")
                    seed_everything(3407)

                    # Progessive Training

                    if not torch.cuda.is_available():
                        logger.warning(
                            "Indexing the corpus using CPU can be very slow."
                        )
                        device = torch.device("cpu")
                    else:
                        device = torch.device("cuda")

                    config = {
                        "model_name": "kaiyuy/leandojo-lean4-retriever-byt5-small",
                        "lr": 1e-3,
                        "warmup_steps": 1000,
                        "max_seq_len": 512,
                        "num_retrieved": 100,
                    }

                    model = PremiseRetriever.load(
                        model_checkpoint_path, device, freeze=False, config=config
                    )
                    model.train()
                    logger.info(f"Loaded premise retriever at {model_checkpoint_path}")

                    # Load previous Fisher Information Matrix for current EWC
                    if use_fisher:
                        latest_fisher = find_latest_fisher()
                        fisher_info = load_fisher_information(latest_fisher)
                        model.set_fisher_info(fisher_info)
                        logger.info("Fisher Information Matrix loaded.")

                    # Initialize ModelCheckpoint and EarlyStopping
                    dir_name = new_data_path.split("/")[-1]
                    filename_suffix = f"_lambda_{lambda_value}"
                    checkpoint_callback = ModelCheckpoint(
                        dirpath=CHECKPOINT_DIR,
                        filename=dir_name
                        + filename_suffix
                        + "_{epoch}-{Recall@10_val:.2f}",
                        verbose=True,
                        save_top_k=-1,  # Save all checkpoints
                        every_n_epochs=1,  # Save every epoch (which is just once in this case)
                        monitor="Recall@10_val",
                        mode="max",
                    )

                    early_stop_callback = EarlyStopping(
                        monitor="Recall@10_val", patience=5, mode="max", verbose=True
                    )

                    lr_monitor = LearningRateMonitor(logging_interval="step")

                    # Set up environment variables for NCCL
                    VERY_LONG_TIMEOUT = 7 * 24 * 60 * 60 * 52  # 1 year
                    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
                    os.environ["NCCL_TIMEOUT"] = str(VERY_LONG_TIMEOUT * 1000)

                    # Create a custom log directory for Lightning
                    custom_log_dir = os.path.join(
                        RAID_DIR,
                        "lightning_logs",
                        f"{dir_name}_{use_fisher}_lambda_{lambda_value}",
                    )
                    os.makedirs(custom_log_dir, exist_ok=True)

                    # Initialize DDP strategy
                    ddp_strategy = DDPStrategy(
                        timeout=timedelta(seconds=VERY_LONG_TIMEOUT)
                    )
                    trainer = pl.Trainer(
                        accelerator="gpu",
                        gradient_clip_val=1.0,
                        precision="bf16-mixed",
                        strategy=ddp_strategy,
                        devices=4,
                        accumulate_grad_batches=4,
                        callbacks=[
                            lr_monitor,
                            checkpoint_callback,
                            early_stop_callback,
                        ],
                        max_epochs=current_epoch + epochs_per_repo,
                        log_every_n_steps=1,
                        num_sanity_val_steps=0,
                        default_root_dir=custom_log_dir,
                    )

                    # Barrier before data module
                    logger.info("right before barrier for data module")
                    trainer.strategy.barrier()
                    should_skip, skip_repo_url = should_skip_repo()
                    if should_skip:
                        logger.info(
                            f"Skipping repository {skip_repo_url} due to preprocessing issues"
                        )
                        trainer.strategy.barrier()
                        if is_main_process:
                            logger.info("Removing skip file")
                            skip_file_path = os.path.join(
                                DATA_DIR, "skip_repo.txt"
                            )
                            os.remove(skip_file_path)
                        continue

                    # Set lambda value for the model
                    model.set_lambda(lambda_value)
                    corpus_path = new_data_path + "/corpus.jsonl"
                    data_path = new_data_path + "/random"
                    logger.info(f"Data path: {data_path}")
                    data_module = RetrievalDataModule(
                        data_path=data_path,
                        corpus_path=corpus_path,
                        num_negatives=3,
                        num_in_file_negatives=1,
                        model_name="google/byt5-small",
                        batch_size=BATCH_SIZE,
                        eval_batch_size=64,
                        max_seq_len=1024,
                        num_workers=4,
                    )
                    data_module.setup(stage="fit")

                    logger.info(
                        f"Training dataset size after load: {len(data_module.ds_train)}"
                    )
                    logger.info(
                        f"Validation dataset size after load: {len(data_module.ds_val)}"
                    )
                    logger.info(
                        f"Testing dataset size after load: {len(data_module.ds_pred)}"
                    )

                    logger.info(
                        f"Starting progressive training from epoch {current_epoch} to {current_epoch + epochs_per_repo}"
                    )

                    # Train the model
                    try:
                        logger.info("hit the barrier before training")
                        trainer.strategy.barrier()
                        trainer.fit(
                            model,
                            datamodule=data_module,
                            ckpt_path=model_checkpoint_path,
                        )
                        logger.info("hit the barrier after training")
                        trainer.strategy.barrier()
                    except Exception as e:
                        print(f"An error occurred during training: {str(e)}")
                        print(traceback.format_exc())

                    logger.info(
                        f"Finished progressive training at epoch {trainer.current_epoch}"
                    )

                    # Testing for Average Recall

                    try:
                        best_model_path = find_latest_checkpoint()
                        logger.info(f"Found latest checkpoint: {best_model_path}")
                        best_model = PremiseRetriever.load(
                            best_model_path, device, freeze=False, config=config
                        )
                    except FileNotFoundError as e:
                        logger.error(f"No checkpoint found: {str(e)}")
                        logger.warning("Using the current model state.")
                        best_model = model

                    best_model.eval()

                    logger.info("Testing...")
                    total_R1, total_R10, total_MRR = [], [], []
                    dataset_path = DATA_DIR
                    testing_paths = [
                        os.path.join(dataset_path, d) for d in os.listdir(dataset_path)
                    ]
                    if is_main_process:
                        append_text_locked(
                            EVAL_RESULTS_FILE_PATH,
                            "\n\n\n",
                            f"Results for {dir_name} with lambda = {lambda_value}",
                        )
                    for data_path in testing_paths:
                        if "merged" not in data_path:
                            continue

                        run_cli(best_model_path, data_path)
                        if is_main_process:
                            num_gpus = 4
                            preds_map = {}
                            for gpu_id in range(num_gpus):
                                with open(f"test_pickle_{gpu_id}.pkl", "rb") as f:
                                    preds = pickle.load(f)
                                    preds_map.update(preds)

                            logger.info("Loaded the predictions pickle files")
                            data_path = os.path.join(data_path, "random", "test.json")
                            data = json.load(open(data_path))
                            logger.info(f"Evaluating on {data_path}")
                            R1, R10, MRR = _eval(data, preds_map)
                            logger.info(f"R@1 = {R1} %, R@10 = {R10} %, MRR = {MRR}")
                            total_R1.append(R1)
                            total_R10.append(R10)
                            total_MRR.append(MRR)
                            append_text_locked(
                                EVAL_RESULTS_FILE_PATH,
                                "\n\n\n",
                                f"Intermediate results for {data_path}",
                                "\n\n\n",
                                f"R@1 = {R1} %, R@10 = {R10} %, MRR = {MRR}",
                            )

                    if is_main_process:
                        avg_R1 = np.mean(total_R1)
                        avg_R10 = np.mean(total_R10)
                        avg_MRR = np.mean(total_MRR)

                        logger.info(
                            f"Average R@1 = {avg_R1} %, R@10 = {avg_R10} %, MRR = {avg_MRR}"
                        )

                        if not os.path.exists(EVAL_RESULTS_FILE_PATH):
                            append_text_locked(EVAL_RESULTS_FILE_PATH, "")

                        append_text_locked(
                            EVAL_RESULTS_FILE_PATH,
                            "\n\n\n",
                            f"Average R@1 = {avg_R1} %, R@10 = {avg_R10} %, MRR = {avg_MRR}",
                        )
                else:
                    model_checkpoint_path = f"{RAID_DIR}/checkpoints/mathlib4_29dcec074de168ac2bf835a77ef68bbe069194c5.ckpt"
                    if result is None:
                        logger.info(
                            f"Skipping repository {repo.url} due to preprocessing issues"
                        )
                        continue

                if is_main_process:
                    logger.info("Starting the prover")

                    if ray.is_initialized():
                        logger.info("Shutting down Ray before proving")
                        ray.shutdown()
                    
                    # ADDED: Compute Fisher Information Matrix after training (before proving)
                    if use_fisher and i < num_repos - 1:  # Don't compute Fisher after last repo
                        logger.info("="*80)
                        logger.info("COMPUTING FISHER INFORMATION MATRIX")
                        logger.info("="*80)
                        
                        from retrieval.fisher_computation_module import FisherComputationModule
                        
                        # Create Fisher computation module with current best model
                        fisher_module = FisherComputationModule(best_model)
                        
                        # Setup trainer for Fisher computation
                        fisher_trainer = pl.Trainer(
                            accelerator="gpu",
                            precision="bf16-mixed",
                            strategy=ddp_strategy,
                            devices=4,
                            max_epochs=1,
                            log_every_n_steps=1,
                            num_sanity_val_steps=0,
                        )
                        
                        try:
                            logger.info("Computing Fisher matrix...")
                            fisher_trainer.strategy.barrier()
                            fisher_trainer.fit(fisher_module, datamodule=data_module)
                            fisher_trainer.strategy.barrier()
                            
                            # Save the Fisher Information Matrix
                            if fisher_trainer.is_global_zero:
                                fisher_file_path = os.path.join(
                                    FISHER_DIR,
                                    f"fisher_info_{dir_name}_distributed.pkl",
                                )
                                fisher_module.save_fisher_info(fisher_file_path)
                                logger.info(f"Fisher Information Matrix saved at {fisher_file_path}")
                        except Exception as e:
                            logger.error(f"Error during Fisher computation: {str(e)}")
                            print(traceback.format_exc())
                        
                        logger.info("Finished computing Fisher matrix")

                    # Set up the prover
                    use_vllm = False
                    # FIXED: Use os.path.join instead of string concatenation
                    corpus_path = os.path.join(str(dst_dir), "corpus.jsonl")
                    tactic = (
                        None  # `None` since we are not using a fixed tactic generator
                    )
                    module = (
                        None  # `None` since we are not using a fixed tactic generator
                    )
                    num_workers = 4
                    num_gpus = 4
                    timeout = 600
                    max_expansions = None
                    num_sampled_tactics = 64
                    debug = False
                    ckpt_path = f"{RAID_DIR}/model_lightning.ckpt"
                    prover = DistributedProver(
                        use_vllm,
                        ckpt_path,
                        corpus_path,
                        tactic,
                        module,
                        num_workers,
                        num_gpus=num_gpus,
                        timeout=timeout,
                        max_expansions=max_expansions,
                        num_sampled_tactics=num_sampled_tactics,
                        raid_dir=RAID_DIR,
                        checkpoint_dir=CHECKPOINT_DIR,
                        debug=debug,
                        run_progressive_training=run_progressive_training,
                    )

                    # Prove sorry theorems
                    if single_repo:
                        prove_sorry_theorems(
                            db,
                            prover,
                            dynamic_database_json_path,
                            repos_for_merged_dataset,
                        )
                    else:
                        prove_sorry_theorems(
                            db, prover, dynamic_database_json_path, repos_for_proving
                        )
                    save_database_locked(db, dynamic_database_json_path)

                    logger.info("Finished searching for proofs of sorry theorems")

                    if ray.is_initialized():
                        logger.info("Shutting down Ray after proving")
                        ray.shutdown()

                    # proofs = []
                    # Uncomment if you would like to contribute back to the repos!
                    # else:
                    #     base_branch = get_default_branch(repo_no_dir)
                    #     subprocess.run(["git", "-C", repo, "fetch", "origin", base_branch], check=True)
                    #     subprocess.run(["git", "-C", repo, "checkout", base_branch], check=True)
                    #     subprocess.run(["git", "-C", repo, "pull", "origin", base_branch], check=True)
                    #     create_or_switch_branch(repo, TMP_BRANCH, base_branch)
                    #     replace_sorry_with_proof(proofs)
                    #     committed = commit_changes(repo, COMMIT_MESSAGE)
                    #     if committed:
                    #         push_changes(repo, TMP_BRANCH)
                    #         url = str(create_pull_request(repo_no_dir, PR_TITLE, PR_BODY, TMP_BRANCH))
                    #     shutil.rmtree(repo)

                logger.info("Finished processing the repository")
                current_epoch += epochs_per_repo
                logger.info(f"current epoch: {current_epoch}")
                # FIXED: Removed early return to allow all 23 repos to be processed
                # Fisher computation will happen between repos via external script
                # if use_fisher:
                #     # Need to return to compute the FIM\n                #     return
                
                # ADDED: Second sorry proving pass after all repos ("Add. After")
                if is_main_process and i == num_repos - 1:
                    logger.info("=" * 80)
                    logger.info("STARTING SECOND PASS: 'Add. After' with final model")
                    logger.info("=" * 80)
                    
                    if ray.is_initialized():
                        logger.info("Shutting down Ray before second pass")
                        ray.shutdown()
                    
                    # Use the final checkpoint  
                    try:
                        final_checkpoint = find_latest_checkpoint()
                        logger.info(f"Using final checkpoint for second pass: {final_checkpoint}")
                    except FileNotFoundError:
                        logger.error("No checkpoint found for second pass")
                        
                    # Create new prover with final model
                    final_prover = DistributedProver(
                        use_vllm,
                        ckpt_path,
                        corpus_path,
                        tactic,
                        module,
                        num_workers,
                        num_gpus=num_gpus,
                        timeout=timeout,
                        max_expansions=max_expansions,
                        num_sampled_tactics=num_sampled_tactics,
                        raid_dir=RAID_DIR,
                        checkpoint_dir=CHECKPOINT_DIR,
                        debug=debug,
                        run_progressive_training=run_progressive_training,
                    )
                    
                    # Reprove ALL repos with final model
                    logger.info("Reproving ALL repositories with final model for 'Add. After' pass")
                    prove_sorry_theorems(
                        db,
                        final_prover,
                        dynamic_database_json_path,
                        repos_to_include=None,  # All repos
                    )
                    
                    save_database_locked(db, dynamic_database_json_path)
                    logger.info("Completed 'Add. After' pass")
                    
                    if ray.is_initialized():
                        logger.info("Shutting down Ray after second pass")
                        ray.shutdown()

    except Exception as e:
        logger.info(f"An error occurred: {e}", file=sys.stderr)
        traceback.print_exc()


if __name__ == "__main__":
    main()
