"""Backslash hygiene of the gallery scripts' reStructuredText.

Sphinx-Gallery takes the module docstring through Python (a ``\\`` there
is one backslash in the RST) but copies ``# %%`` comment cells verbatim
(a ``\\`` there stays doubled and breaks every ``\\sqrt``).  Both
directions have shipped broken formulas; this pins the rule.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = sorted((ROOT / "examples").rglob("*.py"))

_DOCSTRING = re.compile(r'\A(?P<prefix>[rRuU]*)"""(?P<body>.*?)"""', re.DOTALL)
_SINGLE_BACKSLASH = re.compile(r"(?<!\\)\\(?!\\)")


def _split(text: str) -> tuple[str, bool, str, int]:
    """Return (docstring, is_raw, rest, first line number of rest)."""
    m = _DOCSTRING.match(text)
    if m is None:
        return "", False, text, 1
    raw = "r" in m.group("prefix").lower()
    return m.group("body"), raw, text[m.end() :], text[: m.end()].count("\n") + 1


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_backslashes_match_their_context(script: Path) -> None:
    doc, raw, rest, start = _split(script.read_text(encoding="utf-8"))
    bad: list[str] = []
    if not raw:
        for i, line in enumerate(doc.split("\n"), start=1):
            if _SINGLE_BACKSLASH.search(line):
                bad.append(
                    f"docstring line {i}: single backslash in a non-raw docstring: {line.strip()}"
                )
    for i, line in enumerate(rest.split("\n"), start=start):
        if line.lstrip().startswith("#") and "\\\\" in line:
            bad.append(
                f"line {i}: doubled backslash in a comment cell (copied verbatim): {line.strip()}"
            )
    assert not bad, "\n".join(bad)
