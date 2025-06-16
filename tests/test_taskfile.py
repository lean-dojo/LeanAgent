import re
from pathlib import Path

TASKFILE = Path(__file__).resolve().parent.parent / "Taskfile.yaml"


def _read_taskfile() -> str:
    assert TASKFILE.is_file(), "Taskfile.yaml missing"
    return TASKFILE.read_text(encoding="utf-8")


def test_taskfile_contains_core_tasks():
    """Verify that essential tasks are present in Taskfile.yaml."""
    text = _read_taskfile()
    for task in [
        "setup:",
        "run:",
        "run_fisher:",
        "run_ewc:",
        "create_conda_env:",
        "install_requirements:",
        "replace_files:",
        "install_elan:",
        "download_checkpoint_data:",
        "test:",
        "check_github_token:",
    ]:
        assert task in text, f"Task '{task.rstrip(':')}' not found in Taskfile.yaml"


def test_no_unresolved_python_placeholders():
    """Ensure we don't leave {raid_dir} style placeholders inside Taskfile (Go templates use {{.}})."""
    text = _read_taskfile()
    unresolved = re.findall(r"\{[a-zA-Z_]+\}", text)
    # Allow curly braces used in bash parameter expansion e.g., ${RAY_TMPDIR}
    unresolved = [u for u in unresolved if not u.startswith("${")]
    assert not unresolved, f"Found unresolved placeholders: {unresolved}"
