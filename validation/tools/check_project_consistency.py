"""Audit current-state length, release metadata and changelog coverage.

Run from any directory with Python 3.11 or newer. Only local files and
available Git release tags are inspected; no package imports or network
access are needed. Source archives without Git metadata still check the
current version and changelog. The development gates are recorded in DD-241.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tomllib
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STATUS_LINE_LIMIT = 400
VERSION = r"\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?(?:\+[A-Za-z0-9.-]+)?"
RELEASE_HEADING = re.compile(rf"^## \[({VERSION})\] - (\d{{4}}-\d{{2}}-\d{{2}})$", re.M)


def _python_version(text: str) -> str:
    values = []
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets
        ):
            values.append(ast.literal_eval(node.value))
    if len(values) != 1 or not isinstance(values[0], str):
        raise ValueError("expected one literal __version__ string")
    return values[0]


def _cff_version(text: str) -> str:
    values = re.findall(r"^version:\s*(.*?)\s*$", text, re.M)
    if len(values) != 1:
        raise ValueError("expected one version field")
    value = values[0].split("#", 1)[0].strip()
    if value.startswith(('"', "'")):
        value = ast.literal_eval(value)
    if not re.fullmatch(VERSION, value):
        raise ValueError("version must be a semantic-version string")
    return value


def _release_tags(root: Path) -> list[str]:
    if not (root / ".git").exists():
        return []
    result = subprocess.run(
        ["git", "tag", "--list", "v*"], cwd=root, capture_output=True, text=True, check=True
    )
    return [tag[1:] for tag in result.stdout.splitlines() if re.fullmatch(rf"v{VERSION}", tag)]


def check_repository(root: Path, *, release_tags: list[str] | None = None) -> list[str]:
    """Return violations without importing or changing the checked project."""
    errors = []
    versions = {}
    readers = {
        "pyproject.toml": lambda text: tomllib.loads(text)["project"]["version"],
        "src/magnelio/_version.py": _python_version,
        "CITATION.cff": _cff_version,
    }
    for name, reader in readers.items():
        try:
            value = reader((root / name).read_text(encoding="utf-8"))
            if not isinstance(value, str) or not re.fullmatch(VERSION, value):
                raise ValueError("version must be a semantic-version string")
            versions[name] = value
        except (OSError, SyntaxError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{name}: {exc}")
    if len(set(versions.values())) > 1:
        errors.append("version mismatch: " + ", ".join(f"{k}={v}" for k, v in versions.items()))

    try:
        count = len((root / "STATUS.md").read_text(encoding="utf-8").splitlines())
        if count > STATUS_LINE_LIMIT:
            errors.append(f"STATUS.md: {count} lines exceeds {STATUS_LINE_LIMIT}")
    except OSError as exc:
        errors.append(f"STATUS.md: {exc}")

    try:
        changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
        headings = RELEASE_HEADING.findall(changelog)
        for version, stamp in headings:
            try:
                date.fromisoformat(stamp)
            except ValueError:
                errors.append(f"CHANGELOG.md: invalid date {stamp} for {version}")
        releases = [version for version, _ in headings]
        duplicates = sorted({version for version in releases if releases.count(version) > 1})
        if duplicates:
            errors.append("CHANGELOG.md: duplicate releases: " + ", ".join(duplicates))
        if len(re.findall(r"^## \[Unreleased\]$", changelog, re.M)) != 1:
            errors.append("CHANGELOG.md: expected one Unreleased section")
        current = versions.get("pyproject.toml")
        if current is not None and current not in releases:
            errors.append(f"CHANGELOG.md: missing dated entry for current version {current}")
        tags = _release_tags(root) if release_tags is None else release_tags
        missing = sorted(set(tags) - set(releases))
        if missing:
            errors.append("CHANGELOG.md: missing release-tag entries: " + ", ".join(missing))
    except (OSError, subprocess.CalledProcessError) as exc:
        errors.append(f"release/changelog audit: {exc}")
    return errors


def main() -> int:
    errors = check_repository(REPO)
    if errors:
        print("\n".join(errors))
        return 1
    print("Project consistency is clean: STATUS, versions and available local release tags")
    return 0


if __name__ == "__main__":
    sys.exit(main())
