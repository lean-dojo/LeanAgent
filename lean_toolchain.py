"""Resolve and activate the exact Lean toolchain required for tracing.

LeanDojo compiles its tracer with whichever ``lean`` is first on ``PATH``.
If a newer default toolchain is active, tracing LeanAgent-supported repos
(≤ v4.8.x) fails with cryptic API errors such as missing ``HashMap.get?``
or ``TSyntax``/``Syntax`` mismatches (see upstream issue #7).
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

_LEAN_VERSION_OUTPUT_REGEX = re.compile(r"Lean \(version (?P<version>[^,\s)]+)")


def get_lean_toolchain_dir(version: str) -> Path:
    """Return elan's installation directory for a Lean toolchain version."""
    if not version.startswith("v"):
        raise ValueError(f"Lean version must start with 'v', got {version!r}")

    elan_home = Path(os.environ.get("ELAN_HOME", "~/.elan")).expanduser()
    return elan_home / "toolchains" / f"leanprover--lean4---{version[1:]}"


def activate_lean_toolchain(version: str) -> Path:
    """Put the requested installed Lean toolchain first on ``PATH``.

    Returns the absolute path to the activated ``lean`` binary.
    Raises ``RuntimeError`` if the toolchain is missing or reports the wrong version.
    """
    toolchain_dir = get_lean_toolchain_dir(version)
    lean_binary = toolchain_dir / "bin" / "lean"
    if not lean_binary.is_file():
        raise RuntimeError(
            f"Lean toolchain {version} is not installed at {toolchain_dir}. "
            f"Install it with: elan toolchain install leanprover/lean4:{version}"
        )

    result = subprocess.run(
        [str(lean_binary), "--version"],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not run Lean toolchain {version} at {lean_binary}: "
            f"{result.stderr.strip()}"
        )

    match = _LEAN_VERSION_OUTPUT_REGEX.search(result.stdout)
    actual_version = match["version"] if match else "unknown"
    expected_version = version[1:]
    if actual_version != expected_version:
        raise RuntimeError(
            f"Lean toolchain at {toolchain_dir} reports {actual_version}, "
            f"but the repository requires {expected_version}."
        )

    os.environ["LEAN4_PATH"] = str(toolchain_dir)
    os.environ["PATH"] = f"{lean_binary.parent}{os.pathsep}{os.environ.get('PATH', '')}"
    return lean_binary
