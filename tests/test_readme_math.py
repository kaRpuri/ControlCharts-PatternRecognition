"""Guards on the README math blocks.

GitHub sanitizes math content as HTML before MathJax runs, so a literal
angle bracket inside an expression is consumed as the start of a tag. That
silently eats whatever follows and leaves the formula unbalanced, which
renders as an error box instead of the equation. A line starting with a
Markdown list marker inside a block ends the paragraph early and breaks the
delimiter pair the same way. Both have happened here, and neither shows up
in the test suite or in ruff, so they are checked directly.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"

DISPLAY_BLOCK = re.compile(r"^\$\$\n(.*?)^\$\$", re.S | re.M)
INLINE_MATH = re.compile(r"(?<!\$)\$([^$\n]+)\$(?!\$)")
LIST_MARKER = re.compile(r"^ {0,3}(?:[-+*]|\d+\.)\s")


def math_expressions() -> list[tuple[str, str]]:
    """Every display and inline math expression in the README."""
    text = README.read_text()
    found = [("display", m.group(1)) for m in DISPLAY_BLOCK.finditer(text)]
    found += [("inline", m.group(1)) for m in INLINE_MATH.finditer(text)]
    return found


def test_readme_has_math() -> None:
    """Guard the guard: the patterns above must actually match something."""
    assert len(math_expressions()) >= 5


@pytest.mark.parametrize("kind,expression", math_expressions())
def test_no_angle_brackets_in_math(kind: str, expression: str) -> None:
    """Use \\lt and \\gt; a bare < or > is eaten by the HTML sanitizer."""
    assert "<" not in expression and ">" not in expression, (
        f"{kind} math contains an angle bracket, use \\lt or \\gt: {expression.strip()[:70]}"
    )


@pytest.mark.parametrize("kind,expression", math_expressions())
def test_braces_balance(kind: str, expression: str) -> None:
    """Unbalanced braces render as an error box rather than an equation."""
    depth = 0
    for char in re.sub(r"\\[{}]", "", expression):
        depth += char == "{"
        depth -= char == "}"
        assert depth >= 0, f"{kind} math closes a brace it never opened"
    assert depth == 0, f"{kind} math leaves {depth} brace(s) open: {expression.strip()[:70]}"


def test_no_list_markers_inside_display_blocks() -> None:
    """A line starting with + or - ends the paragraph and breaks the $$ pair."""
    for block in DISPLAY_BLOCK.findall(README.read_text()):
        offenders = [line for line in block.splitlines() if LIST_MARKER.match(line)]
        assert not offenders, f"line reads as a Markdown list item: {offenders}"


def test_display_delimiters_are_paired() -> None:
    """Every $$ opens or closes a block, so the count is even."""
    assert README.read_text().count("$$") % 2 == 0
