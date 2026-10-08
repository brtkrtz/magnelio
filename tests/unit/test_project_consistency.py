"""Maintenance gates reject metadata drift independently of the solver."""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "validation/tools/check_project_consistency.py"
_SPEC = importlib.util.spec_from_file_location("project_consistency_gate", _PATH)
gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate)


@pytest.fixture
def repository(tmp_path):
    (tmp_path / "src/magnelio").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.9.0"\n')
    (tmp_path / "src/magnelio/_version.py").write_text('__version__ = "0.9.0"\n')
    (tmp_path / "CITATION.cff").write_text('cff-version: 1.2.0\nversion: "0.9.0"\n')
    (tmp_path / "STATUS.md").write_text("current state\n" * 400)
    (tmp_path / "CHANGELOG.md").write_text("## [Unreleased]\n\n## [0.9.0] - 2026-10-08\n")
    return tmp_path


def test_source_archive_and_exact_length_limit_pass(repository):
    assert gate.check_repository(repository) == []


def test_status_limit_rejects_one_excess_line(repository):
    (repository / "STATUS.md").write_text("current state\n" * 401)
    assert any("401 lines" in error for error in gate.check_repository(repository))


@pytest.mark.parametrize("name", ["pyproject.toml", "src/magnelio/_version.py", "CITATION.cff"])
def test_each_version_source_can_detect_drift(repository, name):
    path = repository / name
    path.write_text(path.read_text().replace("0.9.0", "0.8.2"))
    assert any("version mismatch" in error for error in gate.check_repository(repository))


def test_version_source_is_parsed_without_execution(repository):
    path = repository / "src/magnelio/_version.py"
    path.write_text('raise RuntimeError("must not execute")\n__version__ = "0.9.0"\n')
    assert gate.check_repository(repository) == []


def test_missing_current_release_and_available_tag_are_rejected(repository):
    (repository / "CHANGELOG.md").write_text("## [Unreleased]\n")
    errors = gate.check_repository(repository, release_tags=["0.8.2", "0.9.0"])
    assert any("current version 0.9.0" in error for error in errors)
    assert any("release-tag entries: 0.8.2, 0.9.0" in error for error in errors)


def test_duplicate_release_is_rejected(repository):
    path = repository / "CHANGELOG.md"
    path.write_text(path.read_text() + "\n## [0.9.0] - 2026-10-08\n")
    assert any("duplicate releases" in error for error in gate.check_repository(repository))


def test_unclosed_citation_quote_is_rejected(repository):
    (repository / "CITATION.cff").write_text('version: "0.9.0\n')
    assert any("CITATION.cff" in error for error in gate.check_repository(repository))


def test_nonexistent_release_date_is_rejected(repository):
    path = repository / "CHANGELOG.md"
    path.write_text(path.read_text().replace("2026-10-08", "2026-02-30"))
    assert any("invalid date" in error for error in gate.check_repository(repository))


@pytest.mark.parametrize("name", ["CITATION.cff", "src/magnelio/_version.py"])
def test_missing_version_field_is_rejected(repository, name):
    (repository / name).write_text("")
    assert any(name in error for error in gate.check_repository(repository))
