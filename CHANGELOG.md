# Changelog

All notable changes to Magnelio are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).  While the
major version is 0, minor releases may change the public API.

## [0.1.0] - 2026-08-14

First public release.

### Added

- FIT time-domain leapfrog solver on a structured non-uniform hexahedral
  grid, with conformal (sub-cell) material matrices and selectable
  float32/float64 precision
- NumPy (CPU) and CuPy (CUDA GPU) backends; `backend="auto"` picks the
  GPU when available, with CUDA-graph stepping
- Waveguide ports with exact discrete transparent boundaries:
  TEM / QTEM / TE / TM / hybrid modes, multi-mode, declared on the model
  before meshing; lumped (RLC-backed) ports
- Boundary conditions: PEC, PMC, CPML, periodic, and symmetry planes —
  declared once on the model, carried by the mesh
- Materials: isotropic and diagonal-anisotropic, pole-residue dispersion
  for ε(ω) and μ(ω) with built-in vector fitting, conductor losses
  (perturbative or SIBC wall model), surface roughness models
  (Hammerstad, Huray)
- Geometry: CSG primitives and Boolean operators (`a - b`, `a + b`,
  `a & b`), chainable transforms, profile-based construction (loft,
  sweep, revolve, shell), STEP import via pythonocc-core; thin wires
- Field monitors (time and frequency domain, flux, wall loss),
  plane-wave source (TF/SF), 3D cavity eigenmode solver
- Project store on disk: streamed results, bit-exact resume,
  post-processing on stored data (HDF5); every run generates a
  ready-to-open ParaView session
- Touchstone (`.sNp`) export and `scikit-rf` adapter
- Sphinx documentation: tutorial series (14 executable tutorials),
  API reference, and method chapters with literature sources
