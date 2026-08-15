# Changelog

All notable changes to Magnelio are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).  While the
major version is 0, minor releases may change the public API.

## [Unreleased]

### Changed

- Vector field plots (monitor slices and port-mode profiles) draw their
  arrows on an isotropic raster interpolated from the field instead of
  sampling the computational grid, so a locally refined mesh no longer
  shows up as clustered arrows.  `density` now counts arrows along the
  longer axis of the slice; the shorter one follows at equal spacing
- Port-mode plots of a port cut by a symmetry plane show the full
  cross-section — field and geometry overlay mirrored — instead of the
  simulated half, matching what the impedance and power figures already
  reported

### Fixed

- Port-mode plots now cover the whole port window.  They used to stop
  half a cell short on each side, which on a graded mesh could hide a
  tenth of the cross-section and put a seam through the middle of a
  mirrored full-model plot
- Port-mode plots draw the field itself instead of the solver's
  internal edge quantities.  On a graded mesh the two differ by the
  local cell size, which tilted the arrows and biased their length —
  most visibly at a curved conductor, where the profile read up to
  17 % low.  Both the E and the H picture are affected
- Arrows are no longer drawn inside conductors, and the interpolation
  no longer averages field values across a conductor boundary
- Saving a mirrored plot with `bbox_inches="tight"` produced a hugely
  oversized image that included geometry outside the plotted region

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
