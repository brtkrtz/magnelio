# Changelog

All notable changes to Magnelio are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).  While the
major version is 0, minor releases may change the public API.

## [Unreleased]

### Changed

- The online documentation is published in two versions.  Opening the
  documentation now lands on the docs for the newest release; the
  development docs, built from the current state of the repository, are
  one entry away in the version menu in the navigation bar and mark
  themselves with a banner on every page.  Previously the site always
  showed the development state, so it could describe features that were
  not yet in any released version
- **Breaking:** `MonitorFieldFrequency.data` is now always in fields per
  1 W of incident CW power — E in V/m, H in A/m, per square root of a
  watt.  A run divides its own excitation spectrum out of the recorded
  transform, and so does a monitor read back from a project store, so
  the step no longer has to be remembered.  Where no excitation
  reference is available, `data` raises instead of quietly handing back
  the raw transform; `data_raw` returns that transform unchanged, in
  field units times seconds.  Scripts that read `data` without calling
  `renormalize` were off by the excitation spectrum — often by many
  orders of magnitude — and will now either be correct or say so.
  Calling `renormalize` yourself remains supported and cannot
  double-apply
- Frequency-monitor fields exported to ParaView carry the same units as
  the monitor in Python (fields per 1 W CW).  They were previously
  written as the raw transform, so a value read off the renderer and
  the same value read in a script disagreed
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

- ParaView sessions show the full model across declared symmetry planes
  again.  Recent ParaView releases renamed the properties of their
  reflection filter, and the export fell back to the unmirrored data
  without saying so — leaving an unused mirror entry in the pipeline
  browser and a picture of half the model.  If a build exposes no
  usable reflection filter at all, the session now says so in the
  render view and the export raises a warning instead of quietly
  showing half
- Mirrored field data carries the correct sign.  Reflection alone
  continues a field the way it behaves across a magnetic wall, which is
  the opposite of what an electric wall requires, so on a model with
  planes of both kinds one mirrored half had every arrow reversed —
  visible as an even mode where the mode is odd.  Single components
  (`Ex`, `Hz_im`, …) were left uncorrected on every plane, and could
  disagree with the vector they belong to
- A solid could lose half its cross-section on a mesh plane without any
  warning at all, so parts of a conductor were meshed as if they were
  air.  Contour assembly followed the geometry kernel's own wire
  builder, which happily joins section curves into a branched shape
  that is no longer a single loop; everything past the branch was then
  silently skipped.  Cross-sections are now assembled directly from the
  section curves, and a junction is resolved by following the curve
  that continues smoothly.  This mostly shows up on bodies made by
  lofting, revolving or fusing, near the outer edge of a curved face —
  and it also fixes cutting planes that lie exactly in a flat face of a
  fused body, which previously came back short
- A curved solid could lose its entire cross-section on isolated mesh
  planes, warning about "open section chains" without saying which body
  or what it cost.  The plane in question grazes the solid's own curved
  face, and the mesher's recovery step — which re-takes such a plane
  slightly to one side — was too short to get clear of it on a fine
  mesh, so the material bookkeeping and the cell classification could
  end up disagreeing about where the conductor is.  The recovery step no
  longer shrinks with the mesh, and the two now always agree.  Affected
  regions lost their sub-cell resolution and fell back to a staircase
  approximation; conductor edges near the boundary of a lofted or
  revolved body regain it.  The warning, if it still appears, now names
  the body by its material, says how much was dropped and how far the
  recovery searched, and states that the bulk material distribution is
  unaffected — plus the remedy, which is to change the cell size near
  that plane
- Port-mode plots draw the outermost ring of arrows again.  It was
  blanked wherever the field component running *along* the window
  boundary vanished — which an electric wall forces it to do, on every
  port, everywhere along the frame — even though the component standing
  *perpendicular* to that wall is at its maximum there.  Boundary arrows
  now appear, standing perpendicular on the wall as they should.  Note
  that the arrow raster is independent of the mesh: to read the field in
  a thin gap, raise `density` rather than refining the mesh
- A waveguide port could quietly lose its exact, reflection-free
  termination and fall back to the first-order absorbing one when its
  cross-section sat on a mirrored copy of a curved solid — leaving a
  −30 dB-class reflection floor on that channel while a geometrically
  identical, unmirrored port on the same model had none.  Impedances and
  field results are unaffected; the reflection floor of the affected
  channel improves
- Material boundaries that run through a domain boundary face — a
  symmetry plane, a PEC or PMC wall, an absorbing face — are now
  resolved below the cell there, as they already were everywhere else.
  The outermost layer used to round every partially filled edge to
  fully dielectric or fully metal, which cost a half model up to a few
  parts per thousand against the full model it is meant to reproduce
  exactly.  A symmetric structure solved on half the mesh now returns
  the full model's answer to machine precision
- Waveguide ports on curved conductors no longer warn about a residual
  bias in their power scale — the bias is corrected instead.  Where a
  sub-cell conductor edge hands its share to a neighbour, the port's
  power bookkeeping now accounts for it; on a mixed conformal/staircase
  port pair the reciprocity residual drops from 0.013 dB to below
  0.00001 dB.  Absolute S-parameter magnitudes at ports cut by curved
  conductors change by a few parts in a thousand
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
