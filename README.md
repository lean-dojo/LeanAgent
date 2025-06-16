# LeanAgent: Lifelong Learning for Formal Theorem Proving

[LeanAgent](https://arxiv.org/abs/2410.06209) is a novel lifelong learning framework for formal theorem proving that continuously generalizes to and improves on ever-expanding mathematical knowledge without forgetting previously learned knowledge.

## Documentation
Generated docs by Devin are here: [https://deepwiki.com/lean-dojo/LeanAgent]. It includes a detailed explanation + diagrams.

## Setup

## Quick Start with Taskfile

### Prerequisites
1. Install Homebrew (https://brew.sh) if you don't have it.
2. Install Taskfile ( https://taskfile.dev/#/installation)
3. Modify the `.env.template` secret file in the repository root to provide GitHub token and rename `.env.template` to `.env`.

To see all the tasks available to run, run 
```bash
task
```
### Environment and Package Setup

Run the following command to install all necessary packages.

```bash
task setup
```

### Run LeanAgent

Runs the main LeanAgent script, which orchestrates an end-to-end loop that downloads compatible Lean repositories, searches for unsolved theorems, trains and updates neural components, and attempts automated proof discovery to commit back synthesized proofs.
```bash
task run
```

### Run EWC (Elastic Weight Consolidation) for Lifelong Learning
Runs an alternating loop of one training epoch (`task run`) followed by Fisher-matrix computation (`task run_fisher`). This reproduces the Elastic Weight Consolidation (EWC) implementation reported in the LeanAgent paper \[[1](https://arxiv.org/abs/2410.06209)\].  

The authors found that EWC (i.e. using the Fisher Information Matrix) resulted in **sub-optimal lifelong-learning performance** compared with the default curriculum/progressive strategy. Therefore **you only need this step if you want to replicate those results**. For normal training, simply run `task run` each epoch and skip the Fisher stage.

```bash
# one training → Fisher cycle
task run_ewc

# five train→Fisher cycles 
task run_ewc -- 5
```

### Running Tests

To run the unit test suite, use the `task` command after setup is complete:

```bash
task test
```

## Architecture Overview

LeanAgent is a lifelong learning framework for formal theorem proving that continuously generalizes to and improves on expanding mathematical knowledge without forgetting previously learned information. The codebase consists of several key components:

1. Repository management and data extraction
2. Dynamic database management
3. Curriculum learning strategy
4. Progressive training of the retriever
5. Theorem proving with best-first tree search

## Repository Management and Data Extraction

LeanAgent begins by searching for and processing Lean repositories from GitHub. The system maintains a list of known repositories to avoid duplicate processing and uses the GitHub API to identify repositories with Lean as their primary language.

For each identified repository, LeanAgent checks compatibility by determining if the repository uses a supported Lean version. This is done by examining the repository's `lean-toolchain` configuration file. Currently, supported versions range from Lean `4.3.0-rc2` to Lean `4.8.0-rc1`. If compatible, the system clones the repository and extracts its commit SHA for version tracking.

To ensure compatibility with older repositories, LeanAgent can search through a repository's commit history to find the most recent commit that uses a supported Lean version. This allows the system to work with repositories that might have updated to newer, unsupported Lean versions.

The system maintains a list of known repositories that are either already processed or unsuitable for other reasons. This list includes repositories focused on functional programming rather than mathematics, repositories with trace problems, or those with too few theorems to be useful for training.

Once a compatible repository is identified, LeanAgent begins the data extraction process:

1. Repository Tracing: The system uses LeanDojo's `trace` function to extract theorem definitions, proofs, and premises from the repository. This involves:
    - Parsing Lean source files to identify theorems and their positions
    - Extracting theorem statements and proof tactics
    - Building a dependency graph of imported modules and files
    - Identifying premise definitions that theorems rely on

2. Theorem Processing: Each theorem is stored with its full name, file path, source positions, URL, commit reference, and theorem statement

3. Premise Collection: The system extracts premises (theorems, lemmas, definitions) from the repository, recording their:

    - Full name and kind
    - Source code and position information
    - File path and module dependencies

4. Tracing File Dependencies: LeanAgent builds a topological graph of file dependencies to ensure premises are properly ordered during dataset generation. This ordering is crucial for theorem proving, as premises can only be used after they've been defined.

Then, LeanAgent organizes the traced data into a structured dataset:

1. Proof Export: For each theorem in the dataset, LeanAgent exports:

    - The theorem statement and metadata
    - The sequence of tactics comprising the proof
    - Annotated tactics that show which premises were used
    - Proof states before and after each tactic application

2. Premise Export: LeanAgent exports premise definitions in a topologically sorted order to maintain dependency relationships. Each premise file contains:

    - File path and imported modules
    - List of premises with their full names, code, source positions, and kinds

3. Metadata Generation: Additional metadata is recorded for each dataset:
    - Creation timestamp
    - Source repository URL and commit
    - LeanDojo version used for tracing
    - Statistics about theorems, premises, and files

Once generated, the dataset is integrated into LeanAgent's dynamic database (described in more detail in the next section). To do this, a Repository object is created with the extracted data and the repository is added to the dynamic database.

## Dynamic Database Management

At the core of LeanAgent is a custom dynamic database that tracks and manages all mathematical knowledge across repositories. This database stores:

- Repository metadata (URL, name, commit SHA, Lean version)
- Theorems categorized by status (proven, sorry-but-now-proven, unproven)
- Premise files with their imports and individual premises
- Traced files for tracking processing progress
- Detailed theorem information including file paths, positions, and statements

The database provides functionality to add new repositories, update existing ones, and generate merged datasets from multiple repositories. It uses a JSON-based storage format that can be persisted between runs. Specifically, each class implements `to_dict` and `from_dict` methods for conversion. Special handling is provided for non-JSON-serializable types like `datetime` objects, and the serialization format preserves the full structure of theorems, tactics, and premises. This enables the database to be persisted between runs and distributed across computing resources when necessary.

When generating a dataset from the database, the database can deduplicate theorems and premises when merging repositories, prioritizing the most recently added versions.

The database implementation consists of several interrelated data structures:

1. Annotation: References to mathematical entities within tactics. This includes the fully qualified name of the referenced entity, path to the file defining the entity, and source code positions marking the entity's definition boundaries.
2. Annotated Tactic: Represents a Lean tactic with its annotations and proof states. This includes the raw tactic string, a tuple containing the tactic and its annotations, and the proof states before and after tactic application.
3. Theorem: Represents a mathematical theorem in Lean. This includes storing essential metadata (full name, file path, source positions, url, and commit), containing the theorem statement and its traced tactics (sequence of tactics proving the theorem), and including a difficulty rating used for curriculum learning.
4. Premise: Represents a mathematical fact that can be used in proofs. This includes the fully qualified name of the premise, the actual Lean code defining the premise, the start and end source positions, and the type of premise (e.g., theorem, definition, axiom).
5. Premise File: Represents a Lean source file containing premises. This includes the file path, list of imported modules, and list of Premise objects defined in the file.
6. Repository: Represents a Lean GitHub repository with all its mathematical content. This includes containing repository metadata (url, name, commit, lean version, etc.), categorizing theorems as theorems with complete proofs, previously unproven theorems now proved by LeanAgent, or theorems marked with `sorry` that haven't been proven yet, maintaining lists of Premise Files, and providing helper methods for accessing and manipulating theorems and premises.

## Curriculum Learning Strategy

LeanAgent implements a sophisticated curriculum learning strategy based on theorem complexity. The system calculates the complexity of each theorem using an exponential function of the number of proof steps: $e^S$ where $S$ is the number of proof steps. This exponential scaling accounts for the combinatorial explosion of possible proof paths as proofs get longer.

Theorems with no proofs (sorry theorems) are assigned infinite complexity. The system computes the 33rd and 67th percentiles of complexity across all theorems to establish thresholds for categorizing theorems as Easy, Medium, or Hard.

Repositories are then sorted based on the number of easy theorems they contain. This sorting forms the basis of the curriculum, with LeanAgent starting on repositories with the highest number of easy theorems and gradually progressing to more challenging ones.

For theorems without existing proofs, the system distributes them evenly across the three difficulty categories to maintain balance in the curriculum. This approach ensures that LeanAgent builds foundational knowledge before attempting more complex mathematical domains.

## Progressive Training of the Retriever

LeanAgent employs a simple but effective progressive training approach to avoid catastrophic forgetting. Starting with a pre-trained retriever (based on ByT5), the system trains on each new repository dataset for one additional epoch.

This limited exposure to new data helps prevent overfitting while allowing the model to incorporate essential new information. The retriever is continuously updated during training, and embeddings for all premises are precomputed after each training session to ensure proper evaluation.

The system saves checkpoints based on validation performance, specifically the recall at 10 (R@10) metric, which measures how often the correct premise is among the top 10 retrieved premises. LeanAgent evaluates both plasticity (ability to learn new information) and stability (retention of previous knowledge) by measuring:

1. Performance on the current repository (plasticity)
2. Average performance across all previously seen repositories (stability)

This progressive training process is repeated for each repository in the curriculum, allowing LeanAgent to incrementally build its mathematical knowledge while preserving what it has already learned.

## `sorry` Theorem Proving
LeanAgent identifies theorems marked with `sorry` in the Lean files and attempts to generate formal proofs for them using a best-first tree search approach. For each sorry theorem:

1. The system processes the premise corpus to build a knowledge base of available facts and definitions
2. It constructs a directed dependency graph representing file imports to determine which premises are accessible
3. For each proof state, LeanAgent:

    - Retrieves relevant premises based on similarity to the current state
    - Filters to the top 25% most relevant accessible premises
    - Generates tactic candidates using beam search
    - Applies each tactic through Lean to obtain potential next states
    - Adds successful tactic applications as new edges in the proof search tree
    - Selects the tactic with the highest cumulative log probability

The search continues until a proof is found, all possibilities are exhausted, or a time limit (10 minutes) is reached. Successful proofs are added to the dynamic database, allowing LeanAgent to learn from them in future progressive training.

The system processes theorems in batches (typically 12 theorems per batch) for efficiency and maintains a record of encountered theorems to avoid redundant proving attempts. It also saves progress periodically to ensure resilience against interruptions.

## Repository Integration and Contributions

When LeanAgent successfully proves a sorry theorem, it can integrate the generated proof into the original Lean files and create pull requests to propose these changes to repository maintainers. This process involves:

1. Creating a temporary branch in the repository
2. Locating and replacing `sorry` keywords with the generated proof text
3. Committing and pushing the changes
4. Creating a pull request with an explanatory title and body

When submitting proven theorems back to repositories, LeanAgent uses a consistent formatting approach with standardized branch names, commit messages, and PR templates that identify the contributions as coming from LeanAgent.

This contribution mechanism helps advance the development of mathematical repositories while also generating additional training data for future research.

## Further Implementation Details
LeanAgent is implemented with several key configurations that guide its operation:
1. Environment Configuration: The system relies on a designated directory for storage, with specialized paths for repositories, datasets, checkpoints, and evaluation results. This centralized structure ensures all components can access necessary resources.
2. Repository Management: LeanAgent maintains a comprehensive list of known repositories (approximately 250+) to avoid duplicate processing and focus on mathematical content. It categorizes repositories by usability, flagging those with trace problems, too few theorems, or irrelevant content.
3. Repository Compatibility: For consistent behavior, LeanAgent has special handling for common repositories, using predefined compatible commits for critical sources like mathlib4, SciLean, and PFR.
4. Distributed Processing: The core architecture uses Ray for distributed computation and PyTorch Lightning for distributed training across multiple GPUs (typically 4 NVIDIA A100s). This enables efficient processing of repository batches and parallel theorem proving. Specifically, LeanAgent employs PyTorch Lightning's Distributed Data Parallel (DDP) strategy for efficient multi-GPU training, with custom timeout settings to accommodate lengthy operations during progressive learning. The DDP implementation ensures synchronized gradient updates across devices while maintaining model consistency.
5. Repository Identification: A unique identifier is created for each theorem using its full name, file path, and position coordinates, ensuring theorems can be accurately tracked across different versions of repositories.
6. Resource Management: LeanAgent includes mechanisms to shut down Ray instances between different proof search phases to prevent resource leaks and ensure consistent performance across long runs.

## Fisher Information Matrix Creation for EWC in Ablation Studies

The Fisher Information Matrix (FIM) is a key component of Elastic Weight Consolidation (EWC), a technique used in our ablation studies. Although this resulted in suboptimal lifelong learning performance, we describe how to generate and use the Fisher Information Matrix to help replicate our results.

1. The FIM computation is performed using PyTorch Lightning's distributed training capabilities. This distributed approach allows LeanAgent to efficiently compute the FIM across multiple GPUs, significantly speeding up the process for large models.
2. To compute the FIM, we use forward passes through the model using training data, compute the gradients of the model's log-likelihood with respect to its parameters, and square these gradients as well as average them across training examples.
2. After computation, the FIM is saved for future use. The FIM is saved only by the global zero process to avoid redundant writes.
3. In the ablation studies, the FIM is used during progressive training by loading the matrix and setting the lambda parameter to control the strength of EWC regularization. During training, the model uses the FIM to compute the EWC regularization term, which is added to the standard loss function.
4. After training on a new repository, a new FIM is computed to incorporate the importance of parameters for the newly learned tasks.

Note: Due to the inherent stochasticity of LLMs and the best-first tree search algorithm used in LeanAgent, your results may vary from those reported in the paper. Differences in hardware, random seeds, and minor implementation details can all affect theorem proving performance.

## Citation

If you find our work useful, please consider citing [our paper](https://arxiv.org/abs/2410.06209):

```bibtex
@inproceedings{kumarappan2025leanagent,
  title={{LeanAgent}: Lifelong Learning for Formal Theorem Proving},
  author={Kumarappan, Adarsh and Tiwari, Mo and Song, Peiyang and George, Robert Joseph and Xiao, Chaowei and Anandkumar, Anima},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

Adarsh Kumarappan and Mo Tiwari contributed equally to this work.
