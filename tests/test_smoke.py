"""Smoke tests that critical modules are syntactically valid and importable."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative_path",
    [
        "dynamic_database.py",
        "lean_toolchain.py",
        "leanagent_utils.py",
        "common.py",
        "generate_benchmark_lean4.py",
        "compute_fisher.py",
    ],
)
def test_module_parses(relative_path):
    source = (ROOT / relative_path).read_text()
    ast.parse(source, filename=relative_path)


def test_dynamic_database_dataclass_docstrings_are_inside_classes():
    """Regression for the SyntaxError where docstrings sat between @dataclass and class."""
    source = (ROOT / "dynamic_database.py").read_text()
    assert '@dataclass\n"""' not in source
    assert "@dataclass\nclass " in source


def test_leanagent_utils_remove_marks():
    from leanagent_utils import remove_marks

    assert remove_marks("<a>hello</a> world") == "hello world"
