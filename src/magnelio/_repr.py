"""How Magnelio's objects introduce themselves.

A repr says what an object *is*, how big it is and what state it is in
— never what it holds.  A scattering result carries a 201 × 4 × 4
S-matrix and the port signals of every channel; printed as data that
is a screenful of numbers nobody reads, and the one thing a user wants
to know — did it stop on the energy criterion, after how many steps,
how long ago — is nowhere on it.  The helpers here render key/value
blocks and small tables, as text for a terminal and as HTML for a
notebook cell, so every result, run and project answers in the same
voice.

The HTML is deliberately plain: one ``<table>`` with a rule under the
header and no colours of its own, so it reads on a light and on a dark
notebook theme alike.
"""

from __future__ import annotations

import datetime
import html
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "fmt_array",
    "fmt_db",
    "fmt_value",
    "html_kv",
    "html_table",
    "kv_block",
    "text_table",
]

_TABLE_STYLE = "border-collapse:collapse;font-family:monospace;font-size:90%"
_HEAD_STYLE = "text-align:left;padding:2px 10px 2px 0;border-bottom:1px solid currentColor"
_CELL_STYLE = "text-align:left;padding:2px 10px 2px 0;white-space:nowrap"
_RIGHT = "text-align:right;"


def fmt_array(a) -> str:
    """``float64[3000]``, ``complex128[201×2×2]`` — dtype and shape, never content."""
    a = np.asarray(a)
    shape = "×".join(str(n) for n in a.shape) if a.ndim else "scalar"
    return f"{a.dtype}[{shape}]"


def fmt_db(value: float | None) -> str:
    """A level in dB with one decimal; ``None`` prints as ``—``."""
    if value is None:
        return "—"
    return f"{float(value):.1f} dB"


def fmt_value(value) -> str:
    """One value the way a summary shows it.

    ``None`` is ``—``; a float keeps four significant digits; an array
    shows dtype and shape; a datetime shows to the second; a tuple of
    strings and ints (a channel key) reads as ``port1:0``; everything
    else is ``str()``.
    """
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, np.ndarray):
        return fmt_array(value)
    if isinstance(value, datetime.datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str):
        return f"{value[0]}:{value[1]}"
    if isinstance(value, (list, tuple)):
        return ", ".join(fmt_value(v) for v in value)
    return str(value)


def kv_block(title: str, rows: Iterable[tuple[str, object]]) -> str:
    """A title line and aligned ``key  value`` lines under it."""
    rows = list(rows)
    if not rows:
        return title
    width = max(len(k) for k, _ in rows)
    lines = [title]
    for key, value in rows:
        text = fmt_value(value)
        first, *rest = text.split("\n")
        lines.append(f"  {key:<{width}}  {first}")
        lines.extend(f"  {'':<{width}}  {line}" for line in rest)
    return "\n".join(lines)


def text_table(
    columns: Sequence[str],
    rows: Iterable[Sequence[object]],
    *,
    align: str | None = None,
) -> str:
    """Aligned columns with a header row and a rule under it.

    ``align`` is one letter per column, ``l`` or ``r``; numbers read
    better right-aligned.  Default: every column left.
    """
    cells = [[fmt_value(v) for v in row] for row in rows]
    align = align or "l" * len(columns)
    widths = [
        max(len(name), *(len(row[i]) for row in cells)) if cells else len(name)
        for i, name in enumerate(columns)
    ]

    def _line(values: Sequence[str]) -> str:
        parts = []
        for value, width, a in zip(values, widths, align):
            parts.append(value.rjust(width) if a == "r" else value.ljust(width))
        return "  ".join(parts).rstrip()

    out = [_line(list(columns)), "  ".join("─" * w for w in widths)]
    out.extend(_line(row) for row in cells)
    return "\n".join(out)


def html_kv(title: str, rows: Iterable[tuple[str, object]]) -> str:
    """The HTML twin of :func:`kv_block`: a caption and a two-column table."""
    body = "".join(
        f"<tr><th style='{_HEAD_STYLE.replace('border-bottom:1px solid currentColor', '')}'>"
        f"{html.escape(str(key))}</th>"
        f"<td style='{_CELL_STYLE}'>{html.escape(fmt_value(value))}</td></tr>"
        for key, value in rows
    )
    return (
        f"<table style='{_TABLE_STYLE}'>"
        f"<caption style='text-align:left;font-weight:bold;padding:2px 0'>"
        f"{html.escape(title)}</caption>{body}</table>"
    )


def html_table(
    columns: Sequence[str],
    rows: Iterable[Sequence[object]],
    *,
    caption: str | None = None,
    align: str | None = None,
) -> str:
    """The HTML twin of :func:`text_table`."""
    align = align or "l" * len(columns)
    head = "".join(
        f"<th style='{_HEAD_STYLE}{_RIGHT if a == 'r' else ''}'>{html.escape(str(c))}</th>"
        for c, a in zip(columns, align)
    )
    body = "".join(
        "<tr>"
        + "".join(
            f"<td style='{_CELL_STYLE}{_RIGHT if a == 'r' else ''}'>"
            f"{html.escape(fmt_value(v))}</td>"
            for v, a in zip(row, align)
        )
        + "</tr>"
        for row in rows
    )
    cap = (
        f"<caption style='text-align:left;font-weight:bold;padding:2px 0'>"
        f"{html.escape(caption)}</caption>"
        if caption
        else ""
    )
    return f"<table style='{_TABLE_STYLE}'>{cap}<tr>{head}</tr>{body}</table>"
