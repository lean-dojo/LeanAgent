"""Unit tests for escape_regex_special_chars and related common helpers.

These tests intentionally avoid importing heavy deps (torch / lean_dojo) by
exercising the pure helper directly.
"""

import re
from pathlib import Path

import pytest


def _load_escape_regex_special_chars():
    """Extract escape_regex_special_chars without importing common.py (torch)."""
    import ast
    import types

    src = Path(__file__).resolve().parents[1].joinpath("common.py").read_text()
    tree = ast.parse(src)
    func = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "escape_regex_special_chars"
    )
    module = types.ModuleType("escape_mod")
    module.re = re
    exec(
        compile(ast.Module(body=[func], type_ignores=[]), "<escape>", "exec"),
        module.__dict__,
    )
    return module.escape_regex_special_chars


escape_regex_special_chars = _load_escape_regex_special_chars()


@pytest.mark.parametrize(
    "text, expected",
    [
        ("plain", "plain"),
        ("a.b", r"a\.b"),
        ("a*b+c?", r"a\*b\+c\?"),
        ("(x|y)", r"\(x\|y\)"),
        ("[abc]", r"\[abc\]"),
        ("^start$", r"\^start\$"),
        ("", ""),
        (r"already\escaped", r"already\\escaped"),
    ],
)
def test_escape_regex_special_chars(text, expected):
    assert escape_regex_special_chars(text) == expected
    # Escaped text should match the original literally when used as a pattern.
    assert re.fullmatch(escape_regex_special_chars(text), text) is not None
