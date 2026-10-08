# Magnelio — Project Status

*Last updated: 2026-10-08.* **Release v0.9.0**, commit `5e26a849`.
Geometry construction and owned topology, explicit API names, viewer/result
controls, memory planning, selective stored-result access, finite-drive
stopping and CPML/CAD/integral fixes are released. Upgrade instructions:
`docs/migration-api-naming.md` and `docs/migration-geometry.md`.

This file states what *is*. Chronology: `git log --first-parent main`;
reasoning and measurement certificates: `design-decisions.md`;
investigated defects: `known-bugs.md`; pending engineering work: `TODO.md`.

## Release and validation

- Versions agree at **0.9.0** in `pyproject.toml`, `_version.py` and
  `CITATION.cff`. All 23 local release tags have changelog entries.
- Latest full local release acceptance: **4436 passed / 0 failed /
  39 skipped**, 4475 collected, 1087.00 s, 130 warnings. Minimum available
  RAM was 11.22 GiB. Tests default to NumPy/double; explicit single/GPU
  settings override them. Skipped paths are not verified.
- Accepted GitHub CI under pythonocc/OCCT 8.0.1: **3943 passed / 5 skipped**.
  Local CAD acceptance uses 7.9.0. Hosted documentation at the accepted
  source commit passed with the full gallery executed; offline release HTML
  also passed. Exact-tag deployment and conda-forge publication have separate
  delivery status, recorded in the release checklist.
- GitHub release and PyPI publication are recorded as complete. The
  conda-forge 0.9.0 update is submitted as feedstock PR #20; its last recorded
  state is open with successful build/lint. Do not infer package availability
  from the submitted recipe. Docs channels are `/stable/` (tag) and `/dev/`
  (main), with deployment gated by `DEPLOY_DOCS` or manual dispatch.
- Evidence: `investigations/release-0.9.0/RELEASE-CHECKLIST.md` and
  `cad-fixed-full-pytest.log` (internal records).
- Current maintenance checks: Ruff, format, public hygiene, DD-reference and
  API-surface gates pass; 782 imports in 101 repository scripts resolve.
  The README's complete WR-90 workflow is checked separately from the
  release suite. Evidence: `investigations/project-health-2026-10-08/`
  (internal record). The release suite is not rerun for prose-only changes.

## Current architecture

- **General 3D library.** FIT-TD is the current workhorse; eigenmode analyses
  share its material matrices. Simple lines/cavities are validation cases.
  One edge/face material pipeline drives the constitutive matrices; local
  PEC re-masking preserves air gaps (DD-276), and intrinsic dielectric
  contrast is separated from curved-wall free-area loss (DD-277).
- **Backends and precision.** Production defaults are `backend="auto"`
  and single time-loop precision, honouring `MAGNELIO_BACKEND` and
  `MAGNELIO_PRECISION`. Explicit arguments win. CUDA fused/graph kernels,
  Numba CPU kernels and array-stencil fallbacks are available. Geometry,
  mode solves and DFT accumulation remain double. NumPy/double defaults
  above describe the tests, not the library (DD-032/DD-094).
- **Geometry.** Dimensional Curve/Profile/Sheet/Solid values, affine
  transforms, owner-bound topology, named histories and relative Path poses
  are implemented (DD-275). Partition, section, imprint, priority insertion,
  projection, offsets, bends, blends, components, face consumers and
  selective Solid/Sheet STEP/BREP exchange are shipped. WP6.2 diagnostics
  were cancelled. Construction tolerances and kernel approximation limits
  remain explicit in `docs/methods/geometry.md`.
- **Meshing.** Structured non-uniform grids and conformal material filling
  support arbitrary material assemblies. `MeshControl(robust_sections=True)`
  selects bounded-surface sections; the default is False (DD-287).
  Independent material/section certificates pass. The route preserves trims,
  holes, shared edges, embedded coedges and endpoint-tag aliases; ambiguous
  sensitive cuts fail explicitly. Valid-CAD robustness beyond the tested
  corpus remains an open campaign, not a consequence of the test count.
- **Ports.** Declarative ports/elements/sources travel with the mesh.
  Certified numerical TEM/TE/TM chains use exact discrete transparent
  boundaries; analytical-path ports use modal Mur. Inhomogeneous CW ports
  use frequency-local true modes; broadband band ports use the Galerkin
  subspace DTBC (DD-054–057). `port_model="modal"` remains the default.
  `solve_ports()` reports termination and chain certificates. Multimode
  observations must include the propagating channels (DD-229/DD-235).
- **Sources and stopping.** Shared waveforms default to 25 dB Gaussian
  upper-edge attenuation and a 4.5-tau peak (DD-281). Decay stopping waits
  for all finite drives, including delays and plane-wave retardation.
  Current energy/signal peaks survive continuation (DD-282). Decay stopping
  does not certify spectral accuracy; narrow resonances can still be
  falsely accepted by growing-window comparisons.
- **Symmetry.** Model declarations control clipping and physical parity.
  Ports, source amplitudes and flux/power report full-model quantities;
  field views and ParaView mirror on access. CPML samples E at normal-axis
  nodes and H at cell centres; legacy checkpoints retain their former
  profiles (DD-154/DD-155/DD-172/DD-262/DD-286). The resonant full/half
  dipole comparison has max complex dS11 = 3.04e-5.
- **Results and storage.** AnalysisTD runs simultaneous excitations;
  AnalysisScatteringTD runs one incident channel per record. Stored signals
  and selected S channels are read/derived selectively, with cached spectra
  and bounded DFT temporaries (DD-283). Projects stream HDF5, resume
  bit-exactly, pre-register pending runs and expose watch/energy plots.
  Electric-field path integrals retain complex phase (DD-284).
- **Public API.** Thin 14-name core plus 16 curated domain namespaces;
  names have one documented home. DD-279 naming changes and the dimensional
  geometry migration are reflected in methods, examples and upgrade guides.
  Underscore modules and unexported plumbing have no stability guarantee.
- **Viewer.** Inline notebook widgets, browser views from scripts and native
  fallback; grouped visibility, cuts, boundary overlays, field arrows,
  numeric colour limits and whole-volume views are implemented (DD-280).
  Viewer state does not change solver physics.

## Accuracy and remaining limits

The DD-named `validation/` certificates retain the full conditions and
measured floors. A port reflection floor is not a global 3D accuracy claim.

- Double-precision modal TEM certificates: about −131 to −164 dB on
  parallel-plate and coax fixtures (DD-054). TE/TM certificates: about
  −124 to −166 dB on rectangular/round waveguides (DD-055).
- Inhomogeneous CW/band certificates use their certified physical mode
  families and finite-record conditions (DD-056/DD-057). Single precision
  can raise very low floors substantially. **KB-038 is open:** field/port
  round-trip quantisation produces record-length erosion; modal ports show
  the same mechanism. User guidance: `docs/methods/precision.md`.
- Curved-PEC benchmarks: round-waveguide TE11 cutoff −0.29…−0.14 %;
  rotated cavity 0.11–0.63 %, observed order about 1.66 (DD-053).
- **KB-043 is open on the default fast mesher route:** thin material
  regions can be lost near tangency. Opt-in bounded sections mitigate it.
  KB-053/054 rejection defects are fixed; direct 1/4/16/64 conditioning
  checks and the 320-rectangle material certificate pass (DD-287).
- Five controlled 51-cell route comparisons: 638.6 → 1700.6 s (+166.3 %),
  identical 8,582,496 cells; main-process RSS 6.25 → 6.76 GiB, process-tree
  RSS 10.24 → 10.68 GiB. Finite-record S/Ez changes are small; global
  area-error improvement and actual wall-loss effects are unmeasured.
  Evidence: `investigations/near-tangent-hesr-impact/` (internal dossier).
- **Large-grid TD setup memory remains limiting.** At 168.7 million cells,
  curl-list payload alone is approximately 112 GiB, alongside about 48 GiB
  of mesh arrays and conversion/Lanczos workspaces; two earlyoom kills are
  recorded. `estimate_memory()` reports phase/monitor budgets and unknown
  auxiliary costs (DD-278). A budgeted 8.58-million-cell, 144,901-step run
  completed at 13.91 GiB sampled peak RSS. Evidence:
  `investigations/hesr-memory-termination/` and
  `investigations/hesr-memory-budget/` (internal dossiers).
- **GPU port coverage is narrow.** `TestBandDTBCOnGPU` covers a device port;
  broad CPU/GPU placement and many-port workload validation are open in
  `TODO.md`. Passing CPU-default tests does not establish GPU equivalence.
- **Band runtime:** convolution/axis ranking are improved (DD-245/DD-247),
  but frequency-local mode/LU post-processing and a like-for-like default
  axis comparison remain open. Selective stored-result access (DD-283)
  improves latency; it does not remove that physical mode-solve cost.
- **Modal transmission overshoot:** |S21| 1.0030 frozen / 1.0078 dispersive,
  growing with frequency and independent of band rank, remains unresolved
  in the DD-244 investigation. This is separate from a termination floor.
- **Oblique wire/path correction (DD-269):** shipped staircase paths retain
  orientation/contact errors. Investigated energy-consistent alternatives
  improve several reference cases, but general junctions, tilted metal
  contact and finite-pad bond accuracy are not accepted. No experimental
  correction replaces production ThinWire. Evidence:
  `investigations/oblique-lumped-path/` (internal dossier).
- **Initial/replayed fields:** auxiliary states start quiescent; recorded
  field sources remain in RAM until completion, do not complete symmetry
  planes, and only warn on conductor crossings of recording-box faces.
  General incident fields must satisfy Maxwell's equations (DD-224/DD-225).
- **Conductor references:** multi-conductor `dispersion()` reports the
  channel's reference; TDResult has no reference impedances (DD-244).
  Further backend, loss, truncation and energy-growth work lives in `TODO.md`.

## Documentation and maintenance

- `examples/tutorials/`: **17 executable gallery tutorials** plus the
  rendered Cassegrain example; Tutorial 14 consolidates geometry and CAD/PCB
  import recipes. `examples/howto/` supplies task-oriented recipes.
- Sphinx sources cover tutorials, how-tos, API, methods, migration and
  bibliography. Generated gallery pages are build outputs; edit `examples/`.
  Methods no longer show internal DD numbers; old section links are retained.
  Bibliographic verification belongs in the private provenance ledger; the
  public source-acquisition note is removed. Missing ARPACK/Panofsky–Wenzel
  archive records are now in the private procurement checklist.
- Fresh offline HTML passes with warnings treated as errors and gallery
  execution disabled (prose-only edits). The README runs on CPU in single
  and double precision; its Touchstone export warns about varying modal
  references as documented. All 287 DD headings and their numbers survive.
- README installation recommends JupyterLab and the viewer extra; it uses
  agentic-coding terminology and describes current geometry/viewer workflows.
- Documentation historically needed follow-ups (including symmetry and two
  naming-migration tutorial consumers). Passing builds cannot establish
  that every feature and limitation has always been documented.
- The specification's obvious backend/CI/example drift is corrected; a
  broader architecture/numerics reconciliation remains tracked in `TODO.md`.
  A first editorial pass condenses 13 completed/refined DD entries, preserving
  all 287 DD headings/numbers and their file/evidence references. Detailed
  staging narratives remain in release history and the internal record
  `investigations/dd-log-compaction-2026-10-08/`.
  DD/KB anchor numbers must survive any further compaction. The only explicit
  document-length limit in the workspace instructions is this file's
  **400 lines**; long DD history is permitted but merits selective compaction.
