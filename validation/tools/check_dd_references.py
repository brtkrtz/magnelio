"""Design-decision reference audit: every DD-NNN citation must resolve.

``design-decisions.md`` is the anchor system of the project: source
comments, tests, validation scripts and the reference documents cite
decisions by number.  This script cross-checks the two directions:

* every ``DD-NNN`` referenced anywhere in the audited tree must exist
  as a heading in ``design-decisions.md`` (a compaction or renumbering
  must never orphan a citation);
* every DD entry with **zero** external references is reported
  informationally — an unreferenced decision is either historical
  (candidate for the superseded tombstone treatment) or its consumers
  forgot to cite it.

Audited tree: ``src/``, ``tests/``, ``validation/``, ``examples/``,
``benchmarks/``, ``docs/`` (sources only, ``docs/_build`` excluded),
``STATUS.md``, ``spec.md``, ``known-bugs.md``, ``README.md``.

Usage::

    ~/.local/share/mamba/envs/mio/bin/python validation/tools/check_dd_references.py

Exit status 0 when every citation resolves, 1 otherwise (unreferenced
entries alone do not fail the audit).
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

AUDIT_DIRS = ["src", "tests", "validation", "examples", "benchmarks", "docs"]
AUDIT_FILES = ["STATUS.md", "spec.md", "known-bugs.md", "README.md"]
AUDIT_SUFFIXES = {".py", ".md", ".rst", ".ipynb", ".txt"}
EXCLUDE_PARTS = {"_build", "__pycache__", ".ipynb_checkpoints"}

DD_REF = re.compile(r"DD-(\d{3}[ab]?)")
DD_HEADING = re.compile(r"^#{2,3} DD-(\d{3}[ab]?)\b", re.MULTILINE)


def audited_files() -> list[Path]:
    files: list[Path] = []
    for d in AUDIT_DIRS:
        for p in (REPO / d).rglob("*"):
            if not p.is_file() or p.suffix not in AUDIT_SUFFIXES:
                continue
            if EXCLUDE_PARTS.intersection(p.parts):
                continue
            files.append(p)
    files.extend(REPO / f for f in AUDIT_FILES if (REPO / f).is_file())
    return files


def main() -> int:
    dd_file = REPO / "design-decisions.md"
    headings = set(DD_HEADING.findall(dd_file.read_text(encoding="utf-8")))

    referenced: dict[str, set[str]] = defaultdict(set)
    for path in audited_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = str(path.relative_to(REPO))
        for num in DD_REF.findall(text):
            referenced[num].add(rel)

    print(f"design-decisions.md: {len(headings)} DD headings")
    print(f"audited tree: {len(referenced)} distinct DD numbers cited")

    # DD-026 is cited bare while the entries are split into 026a/026b.
    def resolves(num: str) -> bool:
        return num in headings or (num + "a") in headings

    dangling = {n: files for n, files in referenced.items() if not resolves(n)}
    unreferenced = sorted(
        h for h in headings if h not in referenced and h.rstrip("ab") not in referenced
    )

    if unreferenced:
        print(f"\n{len(unreferenced)} DD entr(y/ies) without any external reference:")
        for num in unreferenced:
            print(f"  DD-{num}")

    if dangling:
        print(f"\n{len(dangling)} DANGLING citation number(s):")
        for num in sorted(dangling):
            files = sorted(dangling[num])
            shown = ", ".join(files[:4]) + (" …" if len(files) > 4 else "")
            print(f"  DD-{num}: cited in {shown}")
        return 1

    print("\nAll DD citations resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
