"""Tests for Lean version support and Lean toolchain activation.

These tests avoid importing generate_benchmark_lean4.py (which pulls lean_dojo)
by extracting is_supported_version via AST.
"""

from __future__ import annotations

import ast
import os
import types
from pathlib import Path

import pytest

import lean_toolchain
from lean_toolchain import activate_lean_toolchain, get_lean_toolchain_dir


def _load_is_supported_version():
    src = Path(__file__).resolve().parents[1].joinpath("generate_benchmark_lean4.py").read_text()
    tree = ast.parse(src)
    func = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "is_supported_version"
    )
    module = types.ModuleType("is_supported_version_mod")
    exec(
        compile(ast.Module(body=[func], type_ignores=[]), "<is_supported_version>", "exec"),
        module.__dict__,
    )
    return module.is_supported_version


is_supported_version = _load_is_supported_version()


@pytest.mark.parametrize(
    "version, expected",
    [
        ("v4.2.0", False),
        ("v4.3.0-rc1", False),
        ("v4.3.0-rc2", True),
        ("v4.3.0", True),
        ("v4.7.0", True),
        ("v4.8.0-rc1", True),
        ("v4.8.0", True),
        ("v4.8.1", False),
        ("v4.9.0", False),
        ("v5.0.0", False),
        ("4.8.0", False),
        ("", False),
        (None, False),
    ],
)
def test_is_supported_version(version, expected):
    assert is_supported_version(version) is expected


def test_get_lean_toolchain_dir_uses_elan_home(monkeypatch, tmp_path):
    monkeypatch.setenv("ELAN_HOME", str(tmp_path / "elan"))

    toolchain_dir = get_lean_toolchain_dir("v4.8.0")

    assert toolchain_dir == (
        tmp_path / "elan" / "toolchains" / "leanprover--lean4---4.8.0"
    )


def test_get_lean_toolchain_dir_rejects_unprefixed_version():
    with pytest.raises(ValueError, match="must start with 'v'"):
        get_lean_toolchain_dir("4.8.0")


def test_activate_lean_toolchain_uses_the_requested_binary(monkeypatch, tmp_path):
    toolchain_dir = tmp_path / "elan" / "toolchains" / "leanprover--lean4---4.8.0"
    lean_binary = toolchain_dir / "bin" / "lean"
    lean_binary.parent.mkdir(parents=True)
    lean_binary.touch()
    monkeypatch.setenv("ELAN_HOME", str(tmp_path / "elan"))
    monkeypatch.setenv("PATH", "existing-path")

    class Result:
        returncode = 0
        stdout = "Lean (version 4.8.0, test)"
        stderr = ""

    monkeypatch.setattr(lean_toolchain.subprocess, "run", lambda *args, **kwargs: Result())

    assert activate_lean_toolchain("v4.8.0") == lean_binary
    assert os.environ["LEAN4_PATH"] == str(toolchain_dir)
    assert os.environ["PATH"].split(os.pathsep) == [str(lean_binary.parent), "existing-path"]


def test_activate_lean_toolchain_rejects_a_wrong_lean_version(monkeypatch, tmp_path):
    toolchain_dir = tmp_path / "elan" / "toolchains" / "leanprover--lean4---4.8.0"
    lean_binary = toolchain_dir / "bin" / "lean"
    lean_binary.parent.mkdir(parents=True)
    lean_binary.touch()
    monkeypatch.setenv("ELAN_HOME", str(tmp_path / "elan"))

    class Result:
        returncode = 0
        stdout = "Lean (version 4.9.0, test)"
        stderr = ""

    monkeypatch.setattr(lean_toolchain.subprocess, "run", lambda *args, **kwargs: Result())

    with pytest.raises(RuntimeError, match="reports 4.9.0"):
        activate_lean_toolchain("v4.8.0")


def test_activate_lean_toolchain_missing_binary(monkeypatch, tmp_path):
    monkeypatch.setenv("ELAN_HOME", str(tmp_path / "elan"))
    with pytest.raises(RuntimeError, match="is not installed"):
        activate_lean_toolchain("v4.8.0")
