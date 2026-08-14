# Contributing to Magnelio

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

Two repository-specific gates:

```bash
python validation/tools/check_dd_references.py   # DD anchors resolve
python validation/tools/check_api_surface.py     # public surface unchanged
```

## Conventions

- **Design decisions.**  Architectural and numerical choices are
  recorded in `design-decisions.md` as numbered `DD-` entries — read
  the relevant entries before changing an area, and record new
  decisions there.  DD numbers are stable anchors; never renumber.
  Current state lives in `STATUS.md`, open bugs in `known-bugs.md`.
- **Commits.**  Conventional Commits (`feat:`, `fix:`, `refactor:`,
  `docs:`, `test:`, …).
- **Docstrings.**  NumPy style for the public API; public docstrings
  and error messages carry no `DD-` references (those belong in code
  comments).
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
