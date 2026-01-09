"""Tests for the escape_regex_special_chars function in common.py"""

import pytest
from common import escape_regex_special_chars


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        # Special regex characters that should be escaped
        ("file.name", "file\\.name"),
        ("file*name", "file\\*name"),
        ("file+name", "file\\+name"),
        ("file?name", "file\\?name"),
        ("file(name)", "file\\(name\\)"),
        ("file[name]", "file\\[name\\]"),
        ("file{name}", "file\\{name\\}"),
        ("file^name", "file\\^name"),
        ("file$name", "file\\$name"),
        ("file|name", "file\\|name"),
        
        # Normal text (no special characters)
        ("normal_text", "normal_text"),
        ("hello world", "hello\\ world"),
        
        # Edge cases
        ("", ""),
        ("a", "a"),
        ("123", "123"),
    ],
)
def test_escape_regex_special_chars(input_text, expected_output):
    """Test that escape_regex_special_chars correctly escapes regex special characters."""
    result = escape_regex_special_chars(input_text)
    assert result == expected_output

