# Contributing to LeanAgent (upstream: lean-dojo/LeanAgent)

This workspace tracks fixes intended for upstream contribution to
[lean-dojo/LeanAgent](https://github.com/lean-dojo/LeanAgent).

## Open upstream issue

- **[#7 Lean version support incompatible with LeanDojo tracer](https://github.com/lean-dojo/LeanAgent/issues/7)**  
  LeanAgent supports Lean ≤ ~4.8, but tracing fails if a newer Lean is on
  `PATH`. Fixed here by `lean_toolchain.py` + validation in
  `generate_benchmark_lean4.py`.

There is also an open upstream PR ([#10](https://github.com/lean-dojo/LeanAgent/pull/10))
with a related approach. This branch independently implements the same idea
plus additional crash fixes listed in `ISSUES_FOUND.md`.

## Local development (lightweight tests)

These tests do **not** require GPU, Lean mathlib, or the full ML stack:

```bash
python -m pip install pytest
python -m pytest tests/test_smoke.py tests/test_lean_toolchain.py tests/test_escape_regex.py tests/test_common.py -q
```

Full LeanAgent training/proving still follows the upstream README setup
(`RAID_DIR`, conda env, `lean_dojo==1.9.0`, model downloads).

## Suggested upstream PR split

1. **Critical**: `dynamic_database.py` SyntaxError (dataclass docstrings)
2. **Issue #7**: `lean_toolchain.py` + fail-fast tracing validation
3. **Crash fixes**: missing imports, GitHub auth header, `FileNotFoundError`,
   safer `get_compatible_commit`, `replace_sorry_with_proof`, runner scripts
