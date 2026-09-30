# Issues found in lean-dojo/LeanAgent

Audit of [lean-dojo/LeanAgent](https://github.com/lean-dojo/LeanAgent) `main`
(as of this contribution). Status reflects what this branch fixes.

## Upstream GitHub

| ID | Title | Status |
|----|-------|--------|
| [#7](https://github.com/lean-dojo/LeanAgent/issues/7) | Lean version support incompatible with LeanDojo tracer | Open — fixed in this branch (`lean_toolchain.py`) |
| [#10](https://github.com/lean-dojo/LeanAgent/pull/10) | PR: enforce matching Lean and LeanDojo versions | Open (related to #7) |
| [#9](https://github.com/lean-dojo/LeanAgent/pull/9) | PR: tests for `escape_regex_special_chars` | Open — also covered here |
| [#3](https://github.com/lean-dojo/LeanAgent/pull/3)/[#4](https://github.com/lean-dojo/LeanAgent/pull/4) | Taskfile workflow | Open — not addressed here |

## Bugs fixed in this branch

### Critical

1. **`dynamic_database.py` SyntaxError** — `@dataclass` was followed by a
   module-level docstring *before* `class …`, so the file could not be imported
   and the large `unittest_dynamic_database.py` suite was dead.
2. **Missing `import time`** in `dynamic_database.py` (used by
   `safe_remove_dir_path` retries).

### High (Lean / LeanDojo — issue #7)

3. **Broken Lean toolchain switching** in `generate_benchmark_lean4.py`:
   hardcoded `/.elan/...`, unexpanded `~/...`, continued on missing paths, and
   missing `import os`. Replaced with `lean_toolchain.activate_lean_toolchain`
   plus LeanDojo version validation.
4. **`is_supported_version` accepted Lean 5+** (copied minimum-only check) and
   disagreed with docs. Tightened to Lean 4.3.0-rc2 … 4.8.0.
5. **`configure_leandojo()` used undefined `constants`** — now
   `from lean_dojo import constants`.

### High (runtime crashes)

6. **`prove_sorry_theorems`**: `Set` was used without importing it;
   `datetime.datetime.now()` after `from datetime import datetime`.
7. **GitHub search `Authorization` header** omitted the `token ` prefix
   (unlike the PR helpers in the same file).
8. **`get_compatible_commit` ran `git init` / `git fetch` in CWD** — now uses
   a temporary directory and skips per-commit failures.
9. **`common._is_deepspeed_checkpoint` raised `FileExistsError`** for a missing
   path — now `FileNotFoundError`.
10. **`compute_fisher.py` referenced undefined `model`** when no checkpoint
    existed — now exits with a clear error.

### Medium

11. **`replace_sorry_with_proof` assigned a string into a list slice**,
    exploding lines into characters — assigns `[new_text]` instead.
12. **`run_leanagent.sh` / `run_compute_fisher.sh` killed every Python PID** —
    removed the blanket `kill -9`.
13. **`replace_files.sh` required files under `$RAID_DIR/LeanAgent/`** — now
    copies from the repository directory.
14. **`RAID_DIR` unset produced `"None/tmp"` paths** — `leanagent.py` now
    fails fast.

## Still open / good follow-ups

- Double-counting theorems across split strategies in `export_proofs` /
  `total_theorems` size gate.
- `write_skip_file` is never called; skip-repo logic is incomplete.
- Move `unittest_dynamic_database.py` under `tests/` once LeanDojo is available
  in CI.
- README claims macOS support while `vllm` is Linux/CUDA-oriented.
- Duplicate imports / cleanup in `leanagent.py`.
