import os

import pytest

import lean_toolchain
from lean_toolchain import activate_lean_toolchain, get_lean_toolchain_dir


def test_get_lean_toolchain_dir_uses_elan_home(monkeypatch, tmp_path):
    monkeypatch.setenv("ELAN_HOME", str(tmp_path / "elan"))

    toolchain_dir = get_lean_toolchain_dir("v4.8.0")

    assert toolchain_dir == (
        tmp_path
        / "elan"
        / "toolchains"
        / "leanprover--lean4---4.8.0"
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
