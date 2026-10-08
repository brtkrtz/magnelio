# Contributing to Magnelio

## Asking, suggesting, showing

[GitHub Discussions](https://github.com/brtkrtz/magnelio/discussions) is the place for questions (*Q&A*),
feature ideas (*Ideas*), models and results you want to share (*Show
and tell*) and anything else that is not a defect.  If you are unsure
whether something is a bug, ask there first — a discussion can be
turned into an issue, and the other way round.  Security-relevant
findings go through the private route in `SECURITY.md`, not through a
public thread.

## Reporting a bug

File it in the [issue tracker](https://github.com/brtkrtz/magnelio/issues);
the bug form asks for version, backend and a minimal script.  Issues are
the inbound channel — where a report arrives, gets its reproduction
questions answered, and closes when the fix lands.

`known-bugs.md` is the other half and serves a different purpose: the
numbered `KB-` record of investigated defects, with the measurements
that characterise them and the reason an open one stays open.  Its
numbers are stable anchors that code comments and design decisions point
at, and resolved entries stay as struck-through tombstones rather than
disappearing.  A report that turns out to be a structural defect earns a
`KB-` entry, and the issue links to it; routine reports do not need one.

## Development setup

The geometry stack needs pythonocc-core, which exists only on
conda-forge, so the development environment is conda-based:

```bash
mamba env create -f environment.yml
mamba activate mio
pip install -e .[dev]
```

## Checks

Every change must pass the same gates CI runs:

```bash
ruff check .
ruff format --check .
python -m pytest tests/unit -q
```

`tests/integration` runs full solver problems and takes considerably
longer; run it when your change touches the numerics.  GPU-gated tests
skip on their own without a CUDA device.  `pre-commit install` sets up
the ruff hooks locally (same rules, pinned in
`.pre-commit-config.yaml`).

Repository-specific gates:

```bash
python validation/tools/check_dd_references.py   # DD anchors resolve
python validation/tools/check_api_surface.py     # public surface unchanged
python validation/tools/check_public_hygiene.py  # public content is safe to ship
python validation/tools/check_imports.py         # script imports resolve
```

## Conventions

- **Open development work.** [TODO.md](TODO.md) tracks robustness and
  validation campaigns, investigations, maintenance and deferred improvements.
  An entry is a work item, not an accepted design or a promised feature.
  Investigated defects belong in `known-bugs.md`; decisions belong in
  `design-decisions.md` once a strategy is accepted. `STATUS.md` records
  the current state rather than duplicating the task list.
- **Design decisions.**  Architectural and numerical choices are
  recorded in `design-decisions.md` as numbered `DD-` entries — read
  the relevant entries before changing an area, and record new
  decisions there.  DD numbers are stable anchors; never renumber.
  Current state lives in `STATUS.md`, investigated defects in
  `known-bugs.md` (`KB-` numbers, equally stable).
- **Commits.**  Conventional Commits (`feat:`, `fix:`, `refactor:`,
  `docs:`, `test:`, …).
- **Docstrings.**  NumPy style for the public API; public docstrings
  and error messages carry no `DD-` references (those belong in code
  comments).
- **Documentation and releases.** User-visible changes need concept and
  limitation coverage in `docs/methods/` and, for standard workflows, a
  tutorial or how-to; docstrings alone are insufficient. Record notable
  changes under `CHANGELOG.md`'s Unreleased section, including documentation.
  Keep `STATUS.md` current and at most 400 lines; replace obsolete state
  rather than appending session reports. Every release, including patches,
  needs a changelog entry and matching versions in `pyproject.toml`,
  `src/magnelio/_version.py` and `CITATION.cff`.
- **Script directories.**  `examples/` uses only the public high-level
  API; scripts that need internals go to `validation/` (anchored by a
  DD entry that names them) or `benchmarks/`.
- **Optional tooling.**  `validation/tools/draw_structure.py --render`
  needs Graphviz (`dot`) on the PATH.

## Scope

Magnelio targets general 3D electromagnetic field simulation for
production use.  Design goals in priority order: accuracy →
generality → efficiency → convenience.  New features must cover the
general case (arbitrary 3D geometry, arbitrary material
distributions); simplified special cases are not accepted as defaults.
