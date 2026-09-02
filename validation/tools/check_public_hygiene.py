"""Public-hygiene audit: what reaches the public repository stays tool-neutral and private-free.

The repository is public and every pushed commit stays reachable by its
hash, so the audit must run *before* a push and over *every* commit of
the pushed range, not only over the tip — a line that entered in one
commit and left in a later one is published all the same.  The
``pre-push`` hook of the maintainers' workspace does exactly that; CI
and pre-commit run the same script on the tree they see.

What is refused, per line of every tracked text file:

* the name of a commercial electromagnetic solver or suite — the
  library documents methods and conventions in generic terms and never
  positions itself against a product (design-decisions.md DD-241; the
  same reasoning retired one vendor-coined term, which is refused too);
* a reference to the maintainers' private workspace (``reference_docs/``,
  ``investigations/``, ``userscripts/``) that is not labelled as an
  internal record or dossier within two lines — the marking convention
  of the ``design-decisions.md`` preamble, which declares it once for
  every citation that file holds;
* an absolute home-directory path;
* an e-mail address other than a GitHub noreply one; ``SECURITY.md``
  carries the project contact and is exempt from that rule.

The denylist itself lives here, so this file is excluded from its own
audit; nothing else is.  Binary files are skipped by ``git grep -I``.

Usage::

    python validation/tools/check_public_hygiene.py                  # working tree
    python validation/tools/check_public_hygiene.py --rev HEAD        # one commit
    python validation/tools/check_public_hygiene.py --range origin/main..main

``--range BASE..TIP`` audits every commit of the range for leaks — the
names, home paths and addresses — and the tip for everything including
the labelling convention, reporting only what ``BASE`` does not already
carry, so the pre-push gate cannot be tripped by history that is public
already.  Exit status 0 when nothing is found, 1 otherwise, 2 on a
usage or git error.  Standard library only, so any ``python3`` runs it.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SELF = "validation/tools/check_public_hygiene.py"

# --- the denylist -----------------------------------------------------------

# Vendor names are matched case-sensitively as the acronyms are written
# (``\b`` keeps ``cst`` as a variable name legal); product names that are
# ordinary words are matched as the two-word product only.
_VENDOR_ACRONYMS = r"\b(?:CST|HFSS|COMSOL|FEKO|XFdtd|ANSYS|Ansys)\b"
_VENDOR_PRODUCTS = r"(?i)\bMicrowave\s+Studio\b|\bLumerical\b|\bEmpire\s+XPU\b"
_RETIRED_TERM = r"\bPBA\b"
_PRIVATE_PATH = r"\b(?:reference_docs|investigations|userscripts)/"
_HOME_PATH = r"/home/[A-Za-z0-9_.-]+"
_EMAIL = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"

# A private path is legal when the two lines above, the line itself or
# the line below label it — the vocabulary of the design-decisions.md
# preamble ("internal record", "measurement dossier", "kept outside the
# public repository"); the window is joined first because the label
# wraps across lines.  design-decisions.md declares the convention once
# in its preamble for every citation it holds.
_MARKER = re.compile(r"\b(?:internal|dossiers?|records?)\b|kept outside", re.I)
_MARKER_EXEMPT_FILES = {"design-decisions.md"}
_EMAIL_ALLOW = re.compile(r"@users\.noreply\.github\.com$")
_EMAIL_EXEMPT_FILES = {"SECURITY.md"}

# (rule, pattern, every_commit).  A leak — a name, a path into a home
# directory, an address — is refused in every commit of a pushed range,
# because every commit is published.  The labelling convention for
# private-workspace citations is a property of the text people read,
# so it is held on the tip only; the paths themselves are provenance
# anchors the design-decisions.md preamble admits.
RULES: list[tuple[str, re.Pattern[str], bool]] = [
    ("commercial solver name", re.compile(_VENDOR_ACRONYMS), True),
    ("commercial solver name", re.compile(_VENDOR_PRODUCTS), True),
    ("retired vendor term", re.compile(_RETIRED_TERM), True),
    ("private workspace path without internal-record marking", re.compile(_PRIVATE_PATH), False),
    ("absolute home-directory path", re.compile(_HOME_PATH), True),
    ("e-mail address", re.compile(_EMAIL), True),
]

# One ERE for ``git grep`` that over-approximates every rule; Python
# decides.  Keeping the candidate pass in git makes a 45-commit range a
# few seconds instead of a full tree read per commit.
_CANDIDATE_ERE = (
    r"CST|HFSS|COMSOL|FEKO|XFdtd|ANSYS|Ansys|Microwave[[:space:]]+Studio|Lumerical|Empire"
    r"|PBA|reference_docs/|investigations/|userscripts/|/home/"
    r"|[[:alnum:]._%+-]+@[[:alnum:]-]+\.[[:alnum:].-]+"
)

PATHSPEC = [".", f":!{SELF}", ":!*.gbr", ":!*.gbrjob", ":!*.step", ":!*.stp", ":!*.brep"]


# --- git plumbing -----------------------------------------------------------


def _git(*args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode not in (0, 1):  # git grep exits 1 on "no match"
        sys.stderr.write(proc.stderr)
        raise SystemExit(2)
    return proc.stdout


def _candidates(rev: str | None) -> list[tuple[str, int, str]]:
    """``(path, line_no, text)`` of every line ``git grep`` flags in ``rev`` or the tree."""
    args = ["grep", "-n", "-I", "-E", "-e", _CANDIDATE_ERE]
    if rev is not None:
        args.append(rev)
    args += ["--", *PATHSPEC]
    out: list[tuple[str, int, str]] = []
    for line in _git(*args).splitlines():
        if rev is not None:
            line = line.split(":", 1)[1]  # strip the ``rev:`` prefix
        path, no, text = line.split(":", 2)
        out.append((path, int(no), text))
    return out


_file_cache: dict[tuple[str | None, str], list[str]] = {}


def _lines(rev: str | None, path: str) -> list[str]:
    key = (rev, path)
    if key not in _file_cache:
        if rev is None:
            text = (REPO / path).read_text(encoding="utf-8", errors="replace")
        else:
            text = _git("show", f"{rev}:{path}")
        _file_cache[key] = text.splitlines()
    return _file_cache[key]


# --- the audit --------------------------------------------------------------


def _violations(rev: str | None, *, leaks_only: bool = False) -> list[tuple[str, int, str, str]]:
    """``(path, line_no, rule, text)`` for one tree."""
    found: list[tuple[str, int, str, str]] = []
    for path, no, text in _candidates(rev):
        for rule, pattern, every_commit in RULES:
            if leaks_only and not every_commit:
                continue
            m = pattern.search(text)
            if not m:
                continue
            if rule == "e-mail address":
                if Path(path).name in _EMAIL_EXEMPT_FILES or _EMAIL_ALLOW.search(m.group(0)):
                    continue
            elif rule.startswith("private workspace path"):
                if path in _MARKER_EXEMPT_FILES:
                    continue
                lines = _lines(rev, path)
                window = " ".join(lines[max(0, no - 3) : no + 1])  # two above, the line, one below
                if _MARKER.search(window):
                    continue
            found.append((path, no, rule, text.strip()))
    return found


def _key(hit: tuple[str, int, str, str]) -> tuple[str, str, str]:
    """A violation's identity across commits: the line number may move, the text does not."""
    path, _no, rule, text = hit
    return (path, rule, " ".join(text.split()))


def _report(label: str, hits: list[tuple[str, int, str, str]]) -> None:
    for path, no, rule, text in hits:
        prefix = f"{label}:" if label else ""
        print(f"{prefix}{path}:{no}: [{rule}] {text[:160]}")


def main(argv: list[str]) -> int:
    if not argv:
        hits = _violations(None)
        _report("", hits)
        print(f"working tree: {len(hits)} violation(s)")
        return 1 if hits else 0
    if argv[0] == "--rev" and len(argv) == 2:
        rev = _git("rev-parse", "--short", argv[1]).strip()
        hits = _violations(argv[1])
        _report(rev, hits)
        print(f"{rev}: {len(hits)} violation(s)")
        return 1 if hits else 0
    if argv[0] == "--range" and len(argv) == 2 and ".." in argv[1]:
        # Every commit of BASE..TIP, reporting only what BASE does not
        # already carry: a line that is public on BASE is not a new leak,
        # and a gate that shouts about history it cannot change is
        # switched off.  Line numbers move between commits; identity is
        # (path, rule, text).
        base, tip = argv[1].split("..", 1)
        baseline = {_key(h) for h in _violations(base)}
        revs = _git("rev-list", argv[1]).split()
        seen: set[tuple[str, str, str]] = set()
        total = 0
        for full in revs:
            hits = _violations(full, leaks_only=True)
            fresh = [h for h in hits if _key(h) not in baseline | seen]
            seen.update(_key(h) for h in fresh)
            _report(full[:7], fresh)
            total += len(fresh)
        # The tip carries every rule: it is the text the public reads.
        tip_hits = [h for h in _violations(tip) if _key(h) not in baseline | seen]
        _report(f"{_git('rev-parse', '--short', tip).strip()} (tip)", tip_hits)
        total += len(tip_hits)
        print(f"{len(revs)} commit(s) audited against {base}: {total} new violation(s)")
        return 1 if total else 0
    sys.stderr.write(__doc__.split("Usage::", 1)[1])
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
