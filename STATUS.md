# Magnelio — Project Status

*Last updated: 2026-08-16.*  Latest work: a frequency monitor's `data`
states a unit or refuses to answer (DD-170).  Its DFT bins are the
transient folded with the excitation spectrum, and dividing that out —
the step that makes them fields per 1 W CW — was a call the caller had
to remember, mentioned nowhere in `docs/` or `examples/`.  Forgotten,
`data` returned the raw bins under the same name, so one property
carried two different units depending on a state nothing exposed; and
the pair `data`/`data_raw` promised a distinction that, in the store,
did not exist at all (`data_raw = data`, the same object).  Measured on
a kicker worksheet: transverse shunt impedance off by 5e18, with the
excitation's own spectral shape read as the device's frequency
response.  Every run path now divides by its own excitation — modal,
band, streamed, resume — as does the store reader and the ParaView
export, which reads the stored bins directly and would otherwise have
shipped a second channel disagreeing with the first by nine decades.
Without a reference, `data` raises and names both ways out; `data_raw`
keeps returning the undivided bins.  The division itself lives once, in
`monitors/_dft.py`, so no two callers can cancel the excitation in
subtly different conventions.  Before that, a ParaView session mirrors
its declared symmetry planes again, and mirrors the field rather than
something that looks like it (DD-169).  The renderer's reflection
filter had every one of its properties renamed under it, so placing a
plane raised, a broad `except` returned the unmirrored source, and the
session showed half a model as if it were whole — with the half-built
filter still visible in the pipeline browser.  Beyond the renaming, the
filter's polar transformation is not the physical continuation: against
`mirror_sign` it is right for E across a magnetic wall and H across an
electric one and off by a global minus for the other two, and it never
touches the single components at all.  The signs now come from
`mirror_sign` itself, resolved per plane at export time, so a model
carrying one plane of each type gets both halves right.  Two flattening
steps go with it, each for a measured reason: reflecting a composite
dataset put the continued vector on different cells than the untouched
components (agreeing to the last bit at the reader, 2.2e3 V/m apart
after), and joining two halves hid the seam from cell-to-point
averaging, leaving 2.0 % of the peak as tangential field on an electric
wall.  Both are zero now.  Before that, cross-section contours are
assembled by chaining the kernel's section edges on a graph of their
endpoints instead of feeding them to a wire builder (DD-168).  That
builder accepted an edge at a vertex already joining two, making a
branched pseudo-wire, and the explorer then walked one arm of the
branch and stopped — the edges past the branch were never tessellated
and left no open chain to warn about.  Measured on a stripline
coupler: fourteen section edges, seven reaching the polygons, half the
electrode's cross-section absent over a 70 µm band.  Chained instead,
the returned area falls smoothly across that band where it used to
halve, the mesh keeps its grid but books 11737 → 11923 conformal Hx
faces and 64904 → 64934 PEC cells, and the two-brick seam plane —
lossy on the plain path for the same reason, which is what
`exact_at_faces` was introduced to work around — returns its
40.00000 mm² whole.  Before that, the section retry that steps
off a degenerate cutting plane got a length scale of its own (DD-167).
It had been taking its step from the tessellation deflection, so the
conformal-area pass — which tessellates ten times finer than the cell
classification on purpose — inherited a ten times shorter reach and
could no longer leave near-tangency bands the classification pass
cleared easily.  The two then disagreed about where the material is:
cells classified conductor whose material matrices saw nothing there.
The mesher makes such planes itself, by anchoring a grid line on a
feature's lateral extreme so that the neighbouring cell-centre plane
grazes it — and refining moves that plane *closer* to the tangency.
On a stripline coupler the conductor's whole cross-section was dropped
on two planes; it now sections cleanly, the conformal H-face count on
the worst plane goes 378 → 942, and the warning, which named neither
body nor amount nor consequence, now names all three.  Before that,
port-mode plots draw their outermost ring of arrows again (DD-166).
DD-162 had grown the picture
out to the window and then decided the added lines' validity from the
component running *along* them — which an electric wall forces to zero,
on every port, all the way round the frame — so the ring was dropped
even where the perpendicular component carried the maximum (measured on
a stripline port: 0.0 tangential against 6.05e7 normal).  Validity is
now inherited from the interior line: *on* a conductor is not *in* one.
Before that, where two candidate ladders
of the conformal M_μ pairing both certify, the better-conditioned one
now supplies the target instead of whichever axis was listed first
(DD-165).  Agreement is tested at 1e-6 and the DTBC gate demands 1e-8,
so a jitter in between produced a mass the pairing certified and the
port rejected — quietly: on the stripline coupler the port sitting on
a mirrored coax stub fell to Mur at a spread of 1.7e-8 while its
unmirrored twin certified at 7e-15.  It reads 6.3e-14 now.  The root
cause this had been recorded under was refuted by re-measuring it: the
classifier's two dual-face integrals share one area budget and agree to
3.9e-15 on that model.  Before that, the conformal classifier learned to
reach the domain boundary faces (DD-164), so a material contour
running through a symmetry, PEC, PMC or absorbing face is resolved
below the cell there as it already was everywhere else.  What had kept
this open for a session was the absence of a certificate that could
judge it; the one built for it is a magnetic half model against its
full model on a matched grid, an exact identity with a known target,
and it reads −2.3e-03 before and 4.7e-15 after.  The single gate that
moved the other way, a band-DTBC floor, turned out to be a kernel-fit
residual whose own spread under the fit's resolution is 30–52 dB.
Before that, the port-power conformality
patch learned to see enlarged-cell donations parked on staircase edges
(DD-163) — DD-095's last documented blind spot, which had started
warning on ordinary coaxial ports instead of correcting itself;
mixed-pair reciprocity goes from 0.013 dB to 0.000005 dB.  Before
that, field plots learned to draw on their own isotropic raster
instead of the computational grid, port modes are shown full-model,
and cells buried in a conductor stay blank (DD-160); mode-profile
plots then learned that the solvers hand them FIT grid
quantities, not field samples, and divide by the edge metric before
drawing (DD-161) — which is where most of KB-018's "solver error"
actually lived — and now reach the port window instead of stopping at
the last cell centre (DD-162).  Current state in brief: symmetry planes
are complete — boundary declaration + mesh-time domain clip (DD-154)
and full-model power semantics on ports, recorders and flux monitors
(DD-155), certified against the natively built half model.  The last
pre-v0.1.0 API break is the string/tuple symmetry vocabulary DD-159
(after the DD-153 unification: `corners=`, `normal=`/`position=`,
`name`, written-out radii).  The
time step comes from the measured spectral radius (DD-150, 17–34× on
conformal meshes), degenerate conformal edges are frozen instead of
pinning `dt` (DD-147/DD-149), and OCCT Booleans no longer edit their
operand shapes (DD-146).  Post-release-prep hardening (2026-08-14):
section contours are closed by contract (DD-157) and conductor
grouping fuses labels through PEC cell bodies (DD-156) — both found
on the stripline-coupler worksheet — as was the escape-reach defect
DD-167 amends DD-157 with.  **known-bugs.md has no open
entry:** KB-018 was closed by DD-161, KB-019 by DD-164, KB-017 by
DD-165, KB-020 by DD-167 and KB-021 by DD-168.  One thing remains
documented rather than
fixed.  From KB-017, a tolerance gap: the pairing calls two ladder
targets equal at 1e-6 while the DTBC gate demands 1e-8, and DD-165
makes the choice inside that band optimal without closing it.  The
grazing-incidence residual DD-167 recorded — one of two symmetric
slivers returned instead of both — turned out not to be the section
operator at all and is closed by DD-168.

**Suite: 2171 passed / 6 skipped / 0 failed** (2026-08-16, GPU box —
unit 1825 (+2 scikit-rf skips), integration 346 (+4 skips); GPU tests
need `CUPY_ACCELERATORS=""` when the interpreter binary is called
directly).  The DD-150 step change re-measured three fixture windows
(interval stride, lumped-port guard, SIBC band edge) — the reasoning
is in the DD entry, not a physics regression.  The long-standing
caveat is gone: `test_coax_tem_vs_te_tm` reproduced on 2026-08-12,
was measured rather than guessed — the degenerate basis does *not*
rotate; the convergence residual does — and is closed by DD-142
(30/30 green after).  Per-DD gate accounting lives in the
`design-decisions.md` entries.

**v0.1.0 published (2026-08-14):** tag `v0.1.0` on
github.com/brtkrtz/magnelio (public, CI + docs green), released on
PyPI via the tag-triggered trusted-publishing workflow (sdist +
noarch wheel, verified installable in a clean venv without
pythonocc-core).  **conda-forge as the primary distribution
channel** (the CAD geometry stack needs pythonocc-core, which exists
only on conda-forge) is in flight: staged-recipes PR prepared on
branch `brtkrtz/staged-recipes:magnelio` (v1 recipe, pythonocc-core
>=7.9), awaiting submission/review.  PyPI stays the secondary,
geometry-less install path.  CI (`.github/workflows/ci.yml`, lint +
unit suite) is live.  The repo is ruff-clean: rule set `E`/`F`/`I` pinned in
`pyproject.toml`, enforced by workflow rule, pre-commit and CI.
License LGPL-3.0-or-later in place (`COPYING`, `pyproject.toml`,
README); all runtime dependencies carry permissive licenses.  The
IP / license / provenance review is closed (internal record:
`reference_docs/ip-provenance-record.md`, kept outside the public
repository).  Two standing rules from it: pythonocc-core (LGPL-3.0)
is used strictly via dynamic import — never vendor or statically
embed it; the Vector Fitting code is a from-paper reimplementation —
never copy from the authors' reference implementation.

This file is durable reference state: current pipeline, ports, public
API, measured floors, open and closed construction sites, deferred
items.  It is **not** a session log — the session-by-session narrative
is in the git history, and the reasoning behind every decision is in
`design-decisions.md`.  Keep it that way when updating: replace what
changed, do not append.

## Recent decisions

Newest first, one line each; the full record is the DD entry.

* **DD-159** (2026-08-14) — string/tuple symmetry vocabulary: bare `"SymmetryPEC"`/`"SymmetryPMC"` clip at plane 0.0, `("SymmetryPEC", position)` clips elsewhere, `"ForceSymmetry*"` declares an as-built half model; the `Symmetry` class is removed.
* **DD-158** (2026-08-14) — unregistered-wall warning only for scenes with lossy conductors (declared at mesh time, or via the σ-fallback at `resolve_wall_conductors` time); registration stays unconditional.
* **DD-157** (2026-08-14) — section contours are closed or the plane is re-taken (nudge retry, loud drop); open chains on tangent-band planes of Boolean solids booked fantasy coverage and sent coax ports to Mur; closes KB-015.
* **DD-156** (2026-08-14) — conductor grouping: PEC-cell links fuse component labels, never add nodes; kills phantom conductors from isolated staircase fragments; closes KB-014.
* **DD-155** (2026-08-14) — full-model power semantics under symmetry: injection ×1/√(2^k), recorder ×√(2^k), flux ×2 per cutting plane.
* **DD-154** (2026-08-13) — symmetry planes as boundary declarations; a declared `position=` clips the domain at mesh time, bit-exact vs the half model; full-model port impedances, mirrored plots, ParaView Reflect.
* **DD-153** (2026-08-13) — one API vocabulary: `corners=`, `normal=`/`position=`, `name`, written-out radii, `PortAnalytical(family=)`, `from_ranges`; breaking, no shims.
* **DD-152** (2026-08-13) — geometric queries never read triangulation; closes KB-012 (`plot()` changed meshes).
* **DD-151** (2026-08-13) — face planes outrank bbox extents in plane clustering; closes KB-013 (50 nm domain offset sent ports to Mur).
* **DD-150** (2026-08-13) — `dt` from the measured spectral radius (Lanczos, Gershgorin fallback, cached); 17–34× over the eps/mu-min heuristic.
* **DD-149** (2026-08-13) — free-area floor is a 1 % threshold, not `== 0`; floored E edges are frozen, not bulk.
* **DD-148** (2026-08-13) — a sub-face port is drawn as its window, not the whole wall.
* **DD-147** (2026-08-13) — `M_eps = 0` edges frozen out of solver and CFL (they pinned `dt` ×10³ and emitted NaN).
* **DD-146** (2026-08-13) — Booleans keep their operand shapes intact (OCCT edits reached back into cached user bodies).
* **DD-145** (2026-08-13) — structure diagrams, four views (`validation/tools/draw_structure.py`).
* **DD-144** (2026-08-12) — `blend="tangent"` loft (Bezier spine); `ruled=` → three-valued `blend=`.
* **DD-143** (2026-08-12) — cross-sections draw wires, discrete ports and lumped elements from their definitions.
* **DD-142** (2026-08-12) — deterministic ARPACK start vector in the 2D mode solver; closes KB-010.
* **DD-141** (2026-08-12) — section pool admission by measured sample, not face-count estimate.
* **DD-140** (2026-08-12) — `MonitorFieldFrequency(interval=)` with Nyquist guards.
* **DD-139** (2026-08-12) — eigenmodes export to ParaView on the monitor path.
* **DD-138** (2026-08-12) — eigenmode auto-shift escalation ladder + under-delivery warning; closes KB-011.
* **DD-137** (2026-08-11) — cross-sections on a plane lying in a face: opt-in `exact_at_faces=` (plot only).
* **DD-131…DD-136** (2026-08-11) — profile geometry: `Curve.joined`/`covered`, `geo.Path`, `Cylinder(inner_radius=, angle_deg=)`, `shelled`/`thickened`, n-ary `Loft`, `Curve.traced`, `Shape.volume()`.
* **DD-130** (2026-08-11) — `Brick.from_ranges` per-axis authoring.
* **DD-129** (2026-08-10) — scattering-result documentation contract; docstring gate sweeps every namespace `__all__`.
* **DD-128** (2026-08-10) — public `geo.Shape` base class; verbs carry their own documentation.
* **DD-127** (2026-08-10) — optional `material`: construction bodies; `add()` rejects material-less solids.
* **DD-126** (2026-08-10) — `mirrored()` plane-mirror verb.
* **DD-125** (2026-08-10) — meshes are immutable inputs (port-plane flatten is builder-local).
* **DD-124** (2026-08-10) — footprint-exact thin-sheet rasterisation; keep `min_cell_size` ≥ ~3·t.
* **DD-123** (2026-08-10) — declarative passive lumped elements (`circuit.LumpedElement`).
* **DD-122** (2026-08-10) — port-signal stall watchdog + `max_time_steps="auto"`; closes KB-008.
* **DD-121** (2026-08-09) — 3D field slice plots (`normal=`/`position=`, ⊙/⊗ vectors, geometry overlays).
* **DD-120** (2026-08-08) — scale-robust geometry pipeline (|ΔS| ≤ 1.9e-7 over six decades).
* **DD-119** (2026-08-08) — `geo`/`plots` renames; `import magnelio as mio` example style.
* **DD-118** (2026-08-07) — PMC window edges own the full boundary cell; `z_line` machine-exact.
* **DD-117** (2026-08-07) — thin core (10 pinned names) + domain namespaces; noun-first monitors.
* **DD-116** (2026-08-07) — documentation portal, four pillars.
* **DD-115** (2026-08-04) — ready-to-open ParaView sessions from the store.
* **DD-114** (2026-08-03) — `port_signal_stop_db="auto"` run default, armed past the step estimate.
* **DD-108…DD-113** (2026-08-02) — pre-release API rework: namespaces, model-declared ports, store schema v1.0, result contract, CSG operators + verbs, `magnelio.constants`.
* **DD-107** (2026-07-30) — three equidistant cells at every domain face (modal-port requirement).
* **DD-106** (2026-07-30) — deterministic conventions on tangent section planes (slab defect 4.55e-01 → ~1e-12).
* **DD-105** (2026-07-30) — mesh warnings report what costs something; Γ ≈ 2.5·(h/λ)²·(1−1/g²).
* **DD-104** (2026-07-30) — monitor regions via `corners=`; interval recording until the stop criterion (hard break).
* **DD-103** (2026-07-30) — boundary closure declared once on the model (hard break; undeclared faces close PEC).

Older decisions: `design-decisions.md`.

## Working practices earned the hard way

* **Verify a numerics fix across the mesh-control range, not on the
  mesh that exposed it.**  DD-147 was checked at the setting that
  triggered it and at one neighbour; the same collapse was waiting two
  cell sizes away, because the classifier reaches zero by accumulation
  and lands on a rounding remainder as often as on a clean zero.  A
  guard written as `== 0` against a computed quantity is a guard that
  fires half the time — use a threshold (DD-149).
* **A run that never advances looks exactly like a run that needs more
  steps.**  Before reading a stalled time-domain run as slow
  convergence, compare `dt` against `courant_dt(mesh.grid)` (no
  material factor) and look at `result.reference_signal`: a monotone
  1e-18 ramp is a Gaussian tail that has not arrived, not a broken
  excitation.  The energy line's `0.0 dB` means "current = running
  maximum", which a barely-started run reports just like a resonant one
  (DD-147).
* **A cached OCC solid is shared mutable state.**  Kernel calls that
  look read-only are not: OCCT Booleans default to editing their
  arguments, and a Boolean *result* shares sub-shapes with its
  operands, so damage propagates backwards into the user's bodies
  (DD-146).  When geometry misbehaves only after something else ran,
  measure `BRep_Tool::Tolerance` over edges and vertices —
  `bounding_box()` stays exactly right while the solid becomes
  unusable, because tolerance moves the fuzzy zone, not the geometry.
* **Stage hunks, never whole files.**  An uncommitted working-tree
  experiment (`energy_stop_db` 70 → 40) was once swept into a commit
  by a whole-file `git add` and broke 21 physics integration tests.
  The working tree is shared.
* **Worktree A/B runs need `PYTHONPATH=<worktree>/src`** — the editable
  install pins the main checkout's `src`, so without it a worktree run
  imports the main tree's code and the attribution is meaningless.
* **Re-check the script directories after every API break.**  Nothing
  under `examples/`, `validation/`, `benchmarks/` or the internal
  `investigations/` dossiers (internal records, kept outside the
  public repository) has test coverage, so a rename breaks them
  silently.  Ten scripts once failed at import for months (they used a
  long-deleted recorder class) while every API break still dutifully
  migrated them.  An AST walk that imports each `from magnelio…`
  module and checks `hasattr` per name finds this in seconds
  (`validation/tools/check_imports.py`).

## Script directories

`examples/` is the public-API surface — `examples/tutorials/` holds
the 14 gallery tutorials, no internal imports.  All run to completion
on the GPU box on pure defaults: the DD-096 port-signal criterion is
on by default (DD-114) because the energy criterion alone never fires
on the TM-cut-off plateau of a shielded lossless structure.
`validation/` holds the 29 scripts that legitimately use internals:
the certificates that regenerate the measured floors quoted below, and
the spikes whose conclusions became DD entries.  `benchmarks/` is
runtime and memory profiling.  The `investigations/<topic>/` dossiers
cited across the tree (each pairing a measurement record with its
probes) are the maintainers' internal records, kept outside the public
repository — the citations stay as provenance anchors.  A
`validation/` script earns its keep by being named in a DD —
unreferenced ones rot unnoticed.

## Current architecture state

One unified mass-matrix pipeline.  ``mesh.edge_material``
(``EdgeMaterialData``, four categories + per-edge free-area fraction
``f_A``) drives ``build_M_eps`` / ``build_M_sigma``;
``mesh.face_material`` (``FaceMaterialData``, three categories)
drives ``build_M_mu``.  On H faces with a unique locally
translation-invariant ladder direction the meshing-time coupling pass
(``couple_face_material_pairs``, DD-053) replaces the Krietenstein
value by the LC-consistent pair value ``ε0μ0·ε_pair·μ̄·d·d̃ / M_ε``;
Krietenstein remains the correction on genuinely 3D contours.  The
classifier re-masks tangential-surface E edges (both endpoints on the
same conductor component), so 2D mode solvers and the 3D update see
the same conductor.  FIT-TD, ``EigenmodeSolver3D`` and
``Numerical2DModeSolver`` all consume the same matrices — no
``apply_dm`` switch.

Time-loop precision (DD-094): ``precision="single"|"double"`` on
``AnalysisScatteringTD`` / ``FITTimeDomainSolver``, default ``None``
→ ``MAGNELIO_PRECISION`` else **single** (float32) — the production
default.  Fields, α/β, curl and the ADE/SIBC aux-states carry the
dtype; DFT, ports, eigenmodes and geometry stay double, and ``/denom``
is never single (the CFL is not a float32 quantity).

CFL: ``courant_dt`` reads ``compute_min_effective_eps`` *and*
``compute_min_effective_mu`` (both with a 1 % A_face_free floor for
the ε / μ side respectively).  ``AnalysisScatteringTD`` threads both
through ``courant_dt`` automatically.

Port terminations (DD-054 TEM, DD-055 TE/TM): numerical-path modes
whose co-located pair product certifies a uniform feed chain
(``r = dt/√(M_ε·M_μ)``, weighted RMS spread < 1e-8) run the **exact
discrete transparent boundary condition** (``ports/modal/dtbc.py``;
Klein-Gordon mass ``q = ω̂_c·dt`` from the 2D eigenvalue of the
3D-restricted transversal operator, ``q = 0`` for TEM; ghost-relation
convolution, kernel auto-extended past the run length → exact,
excitation prescribed at the ghost plane).  The a/b decomposition
de-staggers with the exact discrete factor ``λ^{1/2}`` and — for
dispersive modes — uses the exact discrete wave impedance
``dtbc_wave_impedance`` (``port_line_params = (r, q, z0)``).  The TM
2D eigenproblem is the exact restriction ``build_2d_tm_curl_curl``
(DD-055; the former lumped node-Laplace was metric-inconsistent).
``PortSpecNumerical(mode_type=None)`` = unified multi-mode port (TE +
TM merged by cut-off in one operator).  Inhomogeneous QTEM/hybrid
lines measured **CW** use the per-frequency true-mode port (DD-056,
``build_cw_true_mode_port``): channels = eigenpairs of the quadratic
ζ-pencil built from the production matrices at the port
(``ports/modal/zeta_pencil.py``; sparse shift-invert with
unit-circle-arc targets, uniformity certificates as the pair-gate
analogue), each terminated by the frequency-local exact DTBC — the
closed-form ``(r_eff, q_eff)`` fit is exact at ``f_cw`` by
construction and reuses ``DTBCTermination`` unchanged.  The CW a/b
decomposition (``cw_lockin_phasors`` + ``cw_decompose``) solves the
exact 2×2 phasor system per port (de-stagger and discrete impedance
contained; ``i_out = −conj(i_in)``, the reversed-wave current sign).
Multi-channel ports project dual-basis (Gram inverse).
*Pulsed broadband* runs on inhomogeneous lines use the **Galerkin
band-subspace DTBC** (DD-057, ``build_band_dtbc_port``): the tracked
mode-family traces over the band span a real W-orthonormal rank-p
subspace, the exterior is Galerkin-projected onto it (palindromic
W-symmetry inherited → passive by construction) and closed by the
exact small-system DTBC kernels at size 2p (ghost = swapped
projected pencil, excitation = unswapped; auto-extension past the
run).  The ghost source tracks the family direction per frequency
(``set_excitation_band``: erfc-product spectral window, compactness
gate), and ``compute_band_s_parameters`` decomposes ONE pulsed
record per frequency with the DD-056 true-mode machinery.  Only
analytical-path modes remain on modal Mur-1st (DD-047).

Symmetry planes (DD-154/DD-155, vocabulary DD-159): a symmetry plane
is a boundary declaration — ``"SymmetryPEC"``/``"SymmetryPMC"`` clip
the domain at plane 0.0, ``("SymmetryPEC", position)`` at the given
coordinate, ``"ForceSymmetryPEC"``/``"ForceSymmetryPMC"`` declare an
as-built half model without clipping; the ``BoundaryConditions`` face
field keeps the physical wall type, semantics live in the canonical
``symmetry`` map (``symmetry_entries()``).  The clip happens at mesh
time — the mirror half is never meshed, pinned bit-exact against the
natively built half model.  Port reports publish
full-model impedances (PMC cut ÷2, PEC cut ×2); declared source
amplitudes are full-model quantities (injection ×1/√(2^k), recorder
×√(2^k) on ``record_scale``, ``reference_signal`` stays unscaled;
``MonitorFluxTime`` books ×2 per cutting plane); field plots,
overlays and ParaView mirror the recorded half on read.  Certificate:
``validation/symmetry_full_vs_half_certificate.py`` (|Δ|S|| ≤ 1.5e-3,
a-peak Δ 0.064 %, flux Δ 0.12 %).

Public API (thin core + domain namespaces; DD-117, refines DD-108):
* **Core** — the top-level ``magnelio`` namespace (10 names, pinned
  in ``check_api_surface.py``): ``GeometryModel``, ``Material``,
  ``Mesh``/``MeshControl``, ``BoundaryConditions``, the problem
  classes ``AnalysisScatteringTD``/``AnalysisEigenmode``, and
  ``resume``/``open_project``.  Ports are declared on the model
  before meshing (DD-109).  ``port_model`` (DD-063/DD-064) selects
  the port pipeline: ``"modal"`` (default), ``"band"`` (DD-057), or
  ``"auto"``.  Both result implementations (in-RAM and ``Project``
  reader) satisfy the shared scattering-result contract
  (``magnelio.analysis.result_interface``):
  ``S``/``db``/``phase``/``a``/``b``/``plot_s``/``to_touchstone``/
  ``to_skrf``/``settings``.
* **Domain namespaces** — one per subject area, curated ``__all__``,
  one documented home per name: ``magnelio.geo`` (``Shape`` — the base
  class documenting the operators and verbs — primitives, CSG,
  ``Curve``, ``ThinWire``), ``magnelio.materials``,
  ``magnelio.mesh`` (``GridLines``, ``BoxFace``),
  ``magnelio.boundaries``, ``magnelio.ports`` (declarative ``Port*``
  trio, ``PortSpec*`` family, conductor specs, ``Mode``/``ModeType``,
  reports), ``magnelio.sources``, ``magnelio.monitors``
  (``MonitorFieldTime``, ``MonitorFieldFrequency``,
  ``MonitorFluxTime``, ``MonitorWallLoss``), ``magnelio.circuit``
  (``SeriesRLC``/``ParallelRLC``, ``EdgePath``, curve rasteriser),
  ``magnelio.signals``, ``magnelio.solver``, ``magnelio.analysis``
  (result types), ``magnelio.post``, ``magnelio.plots``,
  ``magnelio.io``, ``magnelio.constants``.
* **Internals** — underscore modules (``_operators``, ``_fields``,
  ``_backend``, ``ports._modal``, ``ports._lumped``, …) plus
  soft-private plumbing outside the curated ``__all__`` (port
  builders/operators, ``PortSignalRecorder``, ``MonitorRegion``,
  ``destaggered_power_waves``, ``ScatteringResultMixin``), no
  stability guarantee.

Curved-PEC accuracy (re-measured under DD-053): round-WG TE11 cut-off
−0.29…−0.14 % at n_t ∈ {17, 25, 33, 49}; rotated rectangular cavity
0.11–0.63 % at h ∈ {1.25 … 4} mm with ``p_obs ≈ 1.66`` (both equal or
better than the DD-051 record).

TEM port floors (DD-054, band 0.25–10 GHz, max/median): parallel
plate uniform −138.7/−164.0 dB, graded −136.1/−158.1 dB, PTFE rect
coax −159.3/−159.4 dB, conformal round coax −131.0/−131.3 dB at
unchanged conformal z_line 48.12 Ω — all 30+ dB below the −100 dB
reflection-free acceptance line
(``validation/dtbc_tem_port_floors.py``).

TE/TM port floors (DD-055, CW lock-in through the production solver,
``kg_dtbc_wg_port_floors.py``): WR-90 TE10 −150.4 dB at 1.01·f̂_c,
−153…−166 dB across the band; WR-90 TM11 −137.3 dB at 1.01·f̂_c,
−154…−165 dB; conformal round WG TE11 −124…−132 dB and TM01
−124/−129 dB (1.05–1.5·f̂_c) — all 24+ dB below the acceptance line.
Pulsed band-edge S-parameters on dispersive lines are
finite-record-truncation limited (see "Deferred").

QTEM/hybrid CW port floors (DD-056,
``qtem_cw_dtbc_port_floors.py``, production chain end-to-end,
|S21| = 0.00 dB throughout): layered half-filled plate fundamental
−244.6…−196.5 dB (1–7.8 GHz) and second hybrid mode −176.3/−194.6/
−200.6 dB at 1.01/1.05/1.2·f̂_c (f̂_c = 8.4465 GHz, two-channel
dual-basis port); dielectric-block line −250.2…−225.2 dB; shielded
microstrip −250.8…−206.5 dB — 76–150 dB below the −100 dB line.
Cost-watch resolved: mode solve 31–433 ms per port vs 3D runs of
1.9–196 s per point (share ≤ ~3 % toy lines, 0.9 % production-sized
microstrip); scaling N = 2 710/11 132/45 112 → 86 ms/0.49 s/3.6 s per
point.

QTEM/hybrid pulsed broadband port floors (DD-057,
``qtem_band_dtbc_port_floors.py``, ONE pulsed run per case through
the production chain, |S21| = 0.00 dB, record end/peak
1e-11…1e-13): layered fundamental −159.6…−231.3 dB (1.0–7.8 GHz,
18 points), layered second family −166.7…−189.8 dB at
1.05–1.28·f̂_c (the 1.01·f̂_c point stays on the DD-056 CW anchor,
−176.3 dB — finite-record limitation), dielectric block
−186.7…−202.8 dB, shielded microstrip −171.1…−211.0 dB — all
59+ dB below the −100 dB line.  A-priori boundary ceilings on the
family points −114…−125 dB (ρ-offset evaluation floor).

**Documentation portal (DD-116):** Sphinx/MyST site under `docs/`
(myst-parser + sphinxcontrib-bibtex + pydata-sphinx-theme +
sphinx-gallery, `pip install -e .[docs]`, build `sphinx-build -b html
docs docs/_build/html`; the build is warning-free — verified with
`sphinx -E`, since a cached rebuild proves nothing).  User-facing
convention: public docstrings, API pages and error messages carry no
DD references (developer breadcrumbs live in code comments and
design-decisions.md), and Magnelio is described as a *library* for
full-wave 3D EM simulation — never as a "suite", never identified
with FIT.  A user-visible feature counts as finished only once the
prose documents it: docstrings reach the reader who already knows the
feature exists, not the one who would have to discover it.  Symmetry
planes were the case that established the rule — four DDs of
implementation with zero occurrences in `docs/` or `examples/`, now a
section in the boundary-conditions chapter and the standing practice
in tutorial 09.  Pillars: Tutorials (generated from
`examples/tutorials/*.py`, tutorials 01–14 shipped and given a
reader-perspective polish pass — full gallery build ~8:40, clean;
tutorial 13, the DR-filter capstone, is deliberately the most
expensive page at ~5.5 min since the design path is the content),
API reference (high-level page + one page per component namespace),
Numerical methods (ten chapters, every method with citations,
in-house derivations marked in prose), Bibliography.
`docs/references.bib` holds 60 entries with bibliographic data only —
the citation-confidence bookkeeping lives exclusively in the
maintainers' internal records (internal record:
`reference_docs/provenance-ledger.md`); public docs and BibTeX carry
no verification labels or notes.

**Planned-run pre-registration (DD-070 follow-up):** multi-excitation
analyses pre-register every planned run as ``pending`` in the run
index (`ProjectStore.register_planned_runs`), so the project status
no longer flickers to ``"done"`` between sequential runs.  Reader
skips ``pending`` in aggregates, raises a clear error on per-run
access; watcher idiom: poll ``status``, skip ``state == "pending"``.

## Open construction sites

* **Symmetry planes — known limitations (DD-154/DD-155 complete).**
  Lumped ports at a symmetry plane stay uncorrected; ParaView's
  FlipAllInputArrays mirrors H like a polar vector (magnitude right,
  mirrored-half arrow sign inverted).
* **Mesh-build speed.**  DD-101 (`compute_edge_pec_fractions`:
  face-bbox slab prefilter + cached intersectors) and DD-102
  (`compute_face_material_areas`: exact planar section engine, OCC
  delegation only for curved/tangent/coplanar planes) took the
  slotline coupler mesh 142 → 14 s with bit-identical mesh gates;
  DD-141 fixed pool *admission* by measuring a sample (60-post
  fixture 12.1 → 5.3 s; the 1002-primitive stress case keeps its pool
  and its 58.0 → 33.3 s).  Remaining, in value order: (1) a process
  pool over edge chunks for `compute_edge_pec_fractions` — still the
  candidate for coupler-class geometries (22.8 s of 33.6 s at
  n = 100; pythonocc SWIG never releases the GIL, so processes only),
  with two measured caveats: it inherits the same ~5 s spawn floor,
  and on a 60-post row the function is not dominant, so the case has
  to be made per geometry class; (2) quadric extension of the planar
  engine (plane × cylinder sections are exact circles or line pairs)
  if curved-geometry meshing ever dominates; (3) a bin/BVH over face
  boxes if geometries grow another order of magnitude.

**Closed construction sites** (full record in the DD entry; kept here
so the question "was this ever a problem?" has a cheap answer):

* ~~DD-107 undershoot reporting gap~~ — CLOSED 2026-08-12: buffered
  boundary intervals are exempt from the wavelength-driven skip only
  when the buffer forced extra cells; the warning names the remedy.
* ~~KB-011: `AnalysisEigenmode` returns nothing on sparsely filled
  high-contrast cavities~~ — CLOSED 2026-08-12, **DD-138**.
* ~~KB-010: `test_coax_tem_vs_te_tm` fails intermittently~~ — CLOSED
  2026-08-12, **DD-142** (fixed ARPACK `v0`; 30/30 green).
* ~~KB-009: QTEM hybrid modes fail on x-normal port faces~~ — CLOSED
  2026-08-12: per-edge normal strides in `PeriodChain.et_step`; the
  workaround (feed lines along z) is no longer needed.
* ~~KB-008: `port_signal_stop_db="auto"` stalls on band-edge
  plateaus~~ — CLOSED 2026-08-10, **DD-122**.
* ~~Section-prefill pool re-ran unguarded user scripts~~ — CLOSED:
  `_hidden_main_module()` clears `__main__.__spec__` for the pool's
  lifetime; user scripts must not need a `__main__` guard.
* ~~PEC-fill runtime cost of dead conductor volume~~ — CLOSED,
  **DD-100**: dead-tile skipping, default-on, bit-identical
  (whole-step coax2rect −33 % single / −57 % double).
* ~~Mixed-port-type reciprocity violation ~0.6 dB~~ — CLOSED,
  **DD-095**: per-edge conformality patch in `_calibrate_v_i`;
  post-fix −0.0001 dB.
* ~~Mur late-time instability near hybrid cut-off~~ — CLOSED,
  **DD-096**: complement absorber + `port_signal_stop_db`.
* ~~Curved shapes tangent to the domain boundary give cube modes~~ —
  CLOSED (the former KB-003, **DD-049**/**DD-103**); re-measured:
  sphere TM₁₀₁ within −0.27 % at 14³.  See `known-bugs.md`.

## Deferred / nice-to-have

* **Contributor documentation for the `validation/tools/` gates** —
  `check_imports.py`, `check_dd_references.py`, `check_api_surface.py`
  and `draw_structure.py` (DD-145) are discoverable only through
  their own docstrings.  A short CONTRIBUTING page is being added for
  the v0.1.0 release to introduce the four together — including the
  one external requirement no dependency declares:
  `draw_structure.py --render` needs a Graphviz `dot` binary (the DOT
  source on stdout needs nothing).
* **TE/TM Mur limit near cutoff** — retired: the DD-055 Klein-Gordon
  DTBC removes the DD-047 −19 dB peak structurally (measured −150 dB
  at 1.01·f̂_c).  Mur remains only as the fallback for
  analytical-path modes and — with explicit specs at
  ``port_model="modal"`` — inhomogeneous QTEM; the analysis routes
  the latter through the band pipeline by default (DD-063).
* **Time-domain power waves on band results** — the band port's
  recorded channels are fixed subspace projections; a calibrated
  a/b time series needs per-frequency phasor synthesis (e.g.
  interpolated over the tracking grid).  ``result.a()/b()`` raise
  with guidance on band results today (DD-063).
* **Cheap single-profile QTEM port upgrade** — measured and
  refuted at the DD-064 state: the mid-band true-mode profile with
  the fitted ``(r_eff, q_eff)`` scalar DTBC gains 5–20 dB in the
  band interior but loses at the lower band edge (−19 dB < Mur's
  −26 dB) and pollutes |S21| by up to 1.2 dB — the frequency-local
  KG fit implies an artificial cut-off near the lower edge.
  Candidate future WP: cut-off-free symbol fit (q ≡ 0), edge-aware
  fit, or 2–3 profiles; derive-then-measure required.
* **Pulsed band-edge S-parameters on dispersive lines** are
  finite-record-truncation limited (the cut-off resonance decays
  algebraically; ~+10 dB per 10× run length — a measurement-
  methodology bound, not a port defect).  Candidate future feature:
  late-time autoregressive signal estimation for pulsed runs; the
  certified measurement today is CW lock-in
  (``validation/kg_dtbc_wg_port_floors.py``).
* **Residual GPU small-grid floor** — ~0.41 ms/step at 10k cells is
  per-port feedback round trips plus the Python loop rest, not kernel
  time.  Needs port-hook restructuring (DD-092).
* **`wall_model="sibc"` as the default** — its own decision now that
  the DD-091 validation record exists; today the perturbative
  DD-082/DD-087/DD-098 chain is the default and SIBC is opt-in.
* **Tensor (gyrotropic) μ** — the DD-089 H-side ADE is scalar per
  axis; a full tensor needs a different update topology.
* **Off-Yee field-monitor interpolation** — monitors sample on the Yee
  positions; arbitrary observation points would need interpolation
  that preserves the DD-085 physical units.
* **No far-field / antenna-pattern post-processing** — tutorial 08
  reads S11, input impedance and near fields only; a far-field
  transform is a missing feature, not a tutorial gap.
