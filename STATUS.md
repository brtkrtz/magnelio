# Magnelio — Project Status

*Last updated: 2026-09-04.*  **Released v0.5.2** (2026-09-04; a patch
under the Cargo reading — nothing in it breaks a 0.5.0 script).  In it:
**DD-245** (partitioned boundary convolution), **DD-246** (progress
output), the *Numerical precision* chapter `docs/methods/precision.md`
(its single-precision length law recorded in KB-038), **DD-247** (the
201-point axis re-measured: no dominant item left, the arc-fan lead
void) and **DD-248** (`port_source="dispersive"`).

Before it, v0.5.1 (2026-09-03) carried DD-242/DD-243 (the section-engine
reach repair; KB-041/042/044 closed), DD-244 (the reference impedance a
published quantity, `renormalize`, `report.dispersion`,
`refine_port_modes`; KB-027 closed) and the band-DTBC port on the GPU
(KB-045); v0.5.0 (2026-09-02) carried DD-224…DD-241 — the API grammar
and its Phases A–D, the pair-product gate as a reflection budget
(DD-228/229, closes KB-022), the content gate on the public remote
(DD-241), ten breaking changes and `docs/migration-0.5.md`.  The
band/QTEM track **DD-230…DD-239 is closed** and opened KB-038.

Unreleased on feature branches: **DD-249/DD-250** — `lofted(blend=
"tangent")` between facing faces builds the eased taper (a Hermite loft
on the plain loft's poles, exact zero wall slope at both joints);
**DD-251** — in-place progress in notebooks, the viewer survives *Run
All*, tile-skip line gone; **DD-252** — `refine_port_modes` converges
the cut-off of a TE/TM mode.

Open: KB-023, KB-038, KB-043 and KB-046.  Unit and integration: 3339 passed /
10 skipped (2026-09-04, `CUPY_ACCELERATORS=""`).  Channels: GitHub, PyPI,
conda-forge and the two docs channels below.

This file states what *is*.  Chronology: `git log --first-parent main`;
reasoning: `design-decisions.md`; open bugs: `known-bugs.md`.  Measured
floors regenerate from the `validation/` certificates their DDs name.

## Recent decisions

Newest first, one line each; the full record is the DD entry.

* **DD-252** (2026-09-04) — `refine_port_modes` converges what the mode family defines.  The DD-244 ladder defaulted to `z_line`, and the round port of a rectangular-to-circular taper — whose cut-off the grid reads **0.33 % low**, which β amplifies to 2 % at 11.9 GHz — answered `has no line impedance`.  `target="auto"` (default) resolves from the level-0 report: line impedance for TEM/quasi-TEM, cut-off for TE/TM, written into the report; an explicit `z_line` on a TE mode names the way out.  Gates in `test_modal_refinement.py`.
* **DD-251** (2026-09-04) — a notebook is an in-place stream, the viewer starts its server on the running loop (transport named; a warm start at import measured and rejected), and the tile-skip line is gone.  Three findings from one notebook session.  *Run All* aborted at the first `plot()` with `RuntimeError: cannot enter context ... is already entered`: PyVista's `elegantly_launch` nests a second loop into the kernel's via `nest_asyncio2`, and under ipykernel 7 that nested loop picks up the *next queued cell* — reproduced without a browser by queueing four requests through `jupyter_client` (cells 3–4 never ran).  The viewer now starts the trame server as a task on the running loop, shows an empty `Output` at once and fills it on `server.ready`; all four queued cells run.  DD-246 had filed the notebook's stream with the logs (`isatty()` is false), so a one-minute GPU run printed its first line at **86 %** of the march; an `ipykernel` stream is now an in-place stream at **0.5 s** cadence, between terminal (0.1 s) and log (30 s).  The bare `Tile skip` print is removed; the statistic stays on the solver.  Gates `TestNotebookStream`; record `investigations/viewer3d/runall_repro.py`.
* **DD-250** (2026-09-04) — `blend="tangent"` between facing profiles is a loft with Hermite end rows, not a sweep.  DD-249 had made the coaxial taper *build* (the straight spine's ulp-noise snapped away) but left it the creased plain loft with a warning pointing at hand-made intermediate sections.  The eased taper is not new geometry: it is the plain loft **re-parametrised** — the same family of cross-sections redistributed along the axis under a law whose derivative vanishes at both ends — so OCC's own outline matching is reused: `ThruSections(ruled)` hands back lateral B-spline faces of v-degree 1 with one pole row on each wire (the circle arrives as a **degree-7 polynomial**, a cone as a rational periodic surface via `NurbsConvert`), and each face is rebuilt as a v-cubic with rows `A, A + τd·n_a, B + τd·n_b, B`, weights riding along.  End tangent exact (**0.0e+00** lateral component, all faces), 4 ms build, the notebook's mesh **2.2 s against 6.7 s** for 13 sampled sections; between two circles the volume is the closed-form smoothstep cone to 1e-8.  The regime is the **normals** (`|n_a + n_b| ≤ 1e-6`), not the spine's straightness: a laterally offset pair has a bent spine and still wants parallel sections — a smooth dog-leg — and the sweep survives tilts down to 1e-7 rad, so DD-249's snap and warning are gone; faces that look *away* from each other are refused.  Bent pairs keep DD-144's `MakePipeShell` sweep untouched.  Side finding, **KB-046**: OCC's fixed Gauss volume rule reads the rational Hermite face 0.9 % too large, and the adaptive rule is no fix (it drifts with its tolerance on the polar-parametrised dish), so `volume()` stays quadrature-limited there and the gate integrates adaptively itself.  Measured on the WR-75 → Ø 15.9 mm taper (GPU): worst in-band |S11| **−18.47 dB** eased against **−13.92 dB** straight.  Gates `TestTangentBlend` (facing pair, zero-slope ratio 4, smoothstep cone volume, dog-leg, looking-away); chapter `docs/methods/geometry.md` section *Lofts*; record `investigations/taper-tangency/MEASUREMENTS.md`.

## Working practices earned the hard way

* **Verify a numerics fix across the mesh-control range, not on the
  mesh that exposed it** (DD-147: the same collapse waited two cell
  sizes away); a guard `== 0` on a computed quantity fires half the
  time — use a threshold (DD-149).
* **Derive a refinement law from the maximum over the parameter, not
  from a spot check.**  The section arc's sagitta at a *fixed* u is
  `r·du²/(8|cos u|)` and carries no tilt dependence at all, so a probe
  at one point argues for exponent 0; only the maximum over u — at
  u = ±π/2 — yields the exponent the law needs, 1 in place of the
  shipped 3, which had been over-spending points (DD-240).
* **A run that never advances looks exactly like a run that needs more
  steps.**  Compare `dt` against `courant_dt(mesh.grid)` and read
  `result.reference_signal` (a monotone 1e-18 ramp is a Gaussian tail
  not yet arrived); the energy line's `0.0 dB` means "current = running
  maximum", the same for a barely-started and a resonant run (DD-147).
* **A cached OCC solid is shared mutable state.**  OCCT Booleans edit
  their arguments and a result shares sub-shapes with its operands, so
  damage propagates backwards into the user's bodies (DD-146); when
  geometry misbehaves only after something else ran, measure
  `BRep_Tool::Tolerance` — `bounding_box()` stays right meanwhile.
* **Stage hunks, never whole files.**  An uncommitted experiment
  (`energy_stop_db` 70 → 40) once rode a whole-file `git add` into a
  commit and broke 21 physics tests.
* **Worktree A/B runs need `PYTHONPATH=<worktree>/src`** — the editable
  install pins the main checkout's `src`; without it the A/B is void.
* **A cost pinned on a loaded box is not a cost.**  The same port build
  measured 28.6 ms at load 0.15 and 1801 ms at load 68 on 16 cores —
  63x of spread at constant CPU work, which is how KB-040 was opened.
  Pin thread-limited CPU time, or run alone (DD-239).
* **Re-check the script directories after every API break.**  Nothing
  under `examples/`, `validation/`, `benchmarks/` or the internal `investigations/` dossiers
  has test coverage (ten scripts once failed at import for months);
  `validation/tools/check_imports.py` finds it in seconds.

## Script directories

`examples/` is the public-API surface — `examples/tutorials/` holds the
20 gallery tutorials, no internal imports, all running to completion on
the GPU box on pure defaults (the DD-096 port-signal criterion is on by
default, DD-114: the energy criterion alone never fires on a shielded
lossless structure's TM-cut-off plateau).  `validation/` holds the
scripts that legitimately use internals — the certificates regenerating
the floors quoted below, now printing the time-loop precision beside
every number, and the spikes whose conclusions became DD entries; one
earns its keep by being named in a DD.  `benchmarks/` is runtime and
memory profiling.  The `investigations/<topic>/` dossiers cited across
the tree are the maintainers' internal records, kept outside the
public repository — citations as provenance anchors.

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
``compute_min_effective_mu`` (each with a 1 % A_face_free floor), and
``AnalysisScatteringTD`` threads both through it automatically.

Port terminations (DD-054 TEM, DD-055 TE/TM): numerical-path modes
whose co-located pair product certifies a uniform feed chain
(``r = dt/√(M_ε·M_μ)``, weighted RMS spread inside the DD-229 budget)
run the **exact discrete transparent boundary condition**
(``ports/modal/dtbc.py``; Klein-Gordon mass ``q = ω̂_c·dt`` from the 2D
eigenvalue of the 3D-restricted transversal operator, ``q = 0`` for
TEM; ghost-relation convolution, kernel auto-extended past the run
length → exact, excitation prescribed at the ghost plane).  The a/b
decomposition de-staggers with the exact discrete factor ``λ^{1/2}``
and — for dispersive modes — uses the exact discrete wave impedance
``dtbc_wave_impedance`` (``port_line_params = (r, q, z0)``).  The TM 2D
eigenproblem is the exact restriction ``build_2d_tm_curl_curl``
(DD-055).  ``PortSpecNumerical(mode_type=None)`` = unified multi-mode
port (TE + TM merged by cut-off in one operator); K > 2 line modes are
the modal basis (Gram eigenbasis for TEM, capacitance pencil for QTEM,
DD-196).  Inhomogeneous QTEM/hybrid lines measured **CW** use the
per-frequency true-mode port (DD-056, ``build_cw_true_mode_port``):
channels = eigenpairs of the quadratic ζ-pencil built from the
production matrices at the port (``ports/modal/zeta_pencil.py``; sparse
shift-invert with unit-circle-arc targets, uniformity certificates as
the pair-gate analogue), each terminated by the frequency-local exact
DTBC — the closed-form ``(r_eff, q_eff)`` fit is exact at ``f_cw`` and
reuses ``DTBCTermination`` unchanged.  The CW a/b decomposition
(``cw_lockin_phasors`` + ``cw_decompose``) solves the exact 2×2 phasor
system per port (de-stagger and discrete impedance contained;
``i_out = −conj(i_in)``); multi-channel ports project dual-basis.
*Pulsed broadband* runs on inhomogeneous lines use the **Galerkin
band-subspace DTBC** (DD-057, ``build_band_dtbc_port``): the tracked
mode-family traces over the band span a real W-orthonormal rank-p
subspace, the exterior is Galerkin-projected onto it (palindromic
W-symmetry → passive by construction) and closed by exact small-system
DTBC kernels at size 2p (ghost = swapped pencil, excitation =
unswapped), auto-extending past the run.  The ghost source tracks the
family direction per frequency (``set_excitation_band``); for a static
fundamental — not a waveguide one — that table is continued to f = 0
and closed with the Laplace trace (DD-232), so a default axis runs with
no lower roll-off.  ``compute_band_s_parameters`` decomposes ONE pulsed
record per frequency (DD-056) in one joint least squares over *all*
channels: a mode no channel claims is absorbed into the matched ones,
not flagged, so a port needs a channel per propagating mode and the
loop warns when it is short (DD-235); a run resumes bit-exactly
(DD-233).  Only analytical-path modes stay on modal Mur-1st (DD-047),
whose floor is the near-cancellation of the boundary and profile errors
of one quasi-static Laplace mode (DD-238).  The pair-product gate is a
reflection budget (DD-229): spread ≤ 2e-6, a chain contribution of at
most −114 dB.  Both certificate stages are
loud when they withhold the exact termination (DD-067, DD-228), and
the choice is published per channel off ``solve_ports()``:
``termination``, the measurement ``chain_spread``, and
``chain_floor_db`` — a bound belonging to the exact termination, hence
``None`` on a Mur channel (DD-239).

Symmetry planes (DD-154/DD-155, vocabulary DD-159): a symmetry plane is
a boundary declaration (``"SymmetryPEC"``/``"SymmetryPMC"``, optionally
at a position; the ``Force*`` spellings declare an as-built half model
without clipping), the ``BoundaryConditions`` face field keeps the
physical wall type and the semantics live in the canonical ``symmetry``
map.  The clip happens at mesh time — the mirror half is never meshed,
pinned bit-exact against the natively built half model.  Port reports
publish full-model impedances (PMC cut ÷2, PEC cut ×2); declared source
amplitudes are full-model quantities (injection ×1/√(2^k), recorder
×√(2^k), ``reference_signal`` unscaled; ``MonitorFluxTime`` books ×2
per cutting plane); field plots, overlays and ParaView mirror the
recorded half on read.  Certificate
``validation/symmetry_full_vs_half_certificate.py``: |Δ|S|| ≤ 1.5e-3,
a-peak Δ 0.064 %, flux Δ 0.12 %.

Public API (thin core + domain namespaces; DD-117, refines DD-108;
grammar DD-224):
* **Core** — the top-level ``magnelio`` namespace (12 names, pinned
  in ``check_api_surface.py``): ``GeometryModel``, ``Material``,
  ``Mesh``/``MeshControl``, ``BoundaryConditions``, ``Excitation``,
  the problem classes ``AnalysisTD``/``AnalysisScatteringTD``/
  ``AnalysisEigenmode``, and ``resume``/``open_project``.  Ports,
  elements and sources are declared on the model before meshing
  (DD-109, DD-123, DD-224) and travel with the mesh.  ``AnalysisTD`` is
  one leapfrog march under any list of simultaneous ``Excitation``s
  (``run(excitations=…, t_end=, name=)`` → ``TDResult``: port signals,
  sampled drives, ``a``/``b``, energy trace, monitors);
  ``AnalysisScatteringTD`` derives from it and drives one channel per
  run (``run(excited=…)``).  ``port_model`` (DD-063/DD-064) selects the
  port pipeline: ``"modal"`` (default), ``"band"`` (DD-057) or
  ``"auto"``.  Both scattering result implementations (in-RAM and
  ``Project`` reader) satisfy ``magnelio.analysis.result_interface``,
  and ``Project.result(name)`` rebuilds the ``TDResult`` of any run.
* **Domain namespaces** — one per subject area, curated ``__all__``,
  one documented home per name: ``geo`` (``Shape`` — the base class
  documenting the operators and verbs — primitives, CSG, ``Curve``,
  ``ThinWire``), ``materials``, ``mesh`` (``GridLines``, ``BoxFace``),
  ``boundaries``, ``ports`` (declarative ``Port*`` trio, ``PortSpec*``
  family, conductor specs, ``Mode``/``ModeType``, reports),
  ``sources``, ``monitors``, ``circuit``, ``signals``, ``solver``,
  ``analysis`` (result types), ``post``, ``plots``, ``io``,
  ``constants``.
* **Internals** — underscore modules (``_operators``, ``_fields``,
  ``_backend``, ``ports._modal``, ``ports._lumped``, …) plus
  soft-private plumbing outside the curated ``__all__`` (port
  builders/operators, recorders, monitor regions, result mixins), no
  stability guarantee.

Curved-PEC accuracy (re-measured under DD-053): round-WG TE11 cut-off
−0.29…−0.14 % at n_t ∈ {17, 25, 33, 49}; rotated rectangular cavity
0.11–0.63 % at h ∈ {1.25 … 4} mm with ``p_obs ≈ 1.66`` (both equal or
better than the DD-051 record).

Port floors, every one of them pinned with the time loop in
**double** (see below).  TEM (DD-054, 0.25–10 GHz, max/median,
``validation/dtbc_tem_port_floors.py``): parallel plate uniform
−138.7/−164.0 dB, graded −136.1/−158.1 dB, PTFE rect coax
−159.3/−159.4 dB, conformal round coax −131.0/−131.3 dB at unchanged
conformal z_line 48.12 Ω.  TE/TM (DD-055, CW lock-in through the
production solver, ``kg_dtbc_wg_port_floors.py``): WR-90 TE10
−150.4 dB and TM11 −137.3 dB at 1.01·f̂_c, −153…−166 dB across the
band; conformal round WG TE11 −124…−132 dB, TM01 −124/−129 dB.
QTEM/hybrid CW (DD-056, ``qtem_cw_dtbc_port_floors.py``, production
chain end to end, |S21| = 0.00 dB): layered half-filled plate
fundamental −244.6…−196.5 dB and second hybrid mode −176.3…−200.6 dB
(f̂_c = 8.4465 GHz, two-channel dual-basis port), dielectric-block line
−250.2…−225.2 dB, shielded microstrip −250.8…−206.5 dB; the
per-frequency mode solve costs ≤ 3 % of a run, 0.9 % production-sized.
QTEM/hybrid pulsed broadband (DD-057, ``qtem_band_dtbc_port_floors.py``,
ONE pulsed run per case): layered fundamental −159.6…−231.3 dB, layered
second family −166.7…−189.8 dB (the 1.01·f̂_c point stays on the DD-056
CW anchor), dielectric block
−186.7…−202.8 dB, shielded microstrip −171.1…−211.0 dB, a-priori
boundary ceilings on the family points −114…−125 dB.  All sit 24+ dB
below the −100 dB reflection-free acceptance line; pulsed band-edge
S-parameters on dispersive lines are record-truncation limited (see
"Deferred").

**At the shipped ``precision="single"`` (DD-094) those are not the
floors a run reads.**  The same fixtures at the same HEAD sit up to
90 dB higher — WR-90 TE10 −124.5 dB, the QTEM CW cases −155…−173 dB,
the pulsed band certificates −114.1…−129.9 dB and −146.6…−168.1 dB
(KB-038) — while the two conformal round-WG legs, cross-section- and
not wordlength-limited, do not move; re-run in double every leg returns
to its pinned class.  The band-DTBC length law is the same defect: in
single the floor loses 4.75 dB (worst) / 6.35 dB (median) per doubling
of the run, in double it is flat (−149.12 → −149.13 dB).

**Documentation portal (DD-116):** Sphinx/MyST site under `docs/`
(`pip install -e .[docs]`, `sphinx-build -b html docs
docs/_build/html`; warning-free — verified with `sphinx -E`, a cached
rebuild proves nothing).  Pillars: Tutorials (from
`examples/tutorials/*.py`, 01–20 shipped; full gallery ~8:40, of which
tutorial 13, the DR-filter capstone, is ~5.5 min), API reference,
Numerical methods (thirteen chapters, every method cited, in-house
derivations marked in prose), Bibliography.  `docs/references.bib`
holds 63 entries, bibliographic data only; the citation-confidence
bookkeeping lives exclusively in the maintainers' internal record
`reference_docs/provenance-ledger.md`.  Conventions: no DD references
in docstrings, API pages or error messages; Magnelio is a *library*
for full-wave 3D EM simulation, never a "suite" and never identified
with FIT; a feature is finished only once the prose documents it (the
rule symmetry planes established); and **a tutorial derives plot
scales from the data, never from an absolute constant** — only CI
catches a stale `vmax`, since sphinx-gallery re-executes a tutorial
when *its script* changes, not when the library under it does
(`build_docs.sh --clean` locally).

**Two published documentation channels (DD-171):** `/stable/` (from a
`v*` tag) and `/dev/` (from main), root redirecting to stable, one
shared `switcher.json`, a banner on every dev page.  Pages is served
from `gh-pages` — the Docs workflow clones it shallow and writes only
its own channel, so a main push leaves the release docs untouched; the
`.nojekyll` marker is load-bearing (Jekyll would hide `_static/`), and
a build reads `MAGNELIO_DOCS_CHANNEL` (unset — every local build — is
dev).

**Planned-run pre-registration (DD-070 follow-up):** multi-excitation
analyses pre-register every planned run as ``pending``
(`ProjectStore.register_planned_runs`), so the project status no longer
flickers to ``"done"`` between sequential runs; the reader skips
``pending`` (watcher idiom: poll ``status``, skip it).

## Open construction sites

* **Band-pipeline runtime** — the convolution lead is **done**
  (DD-245): partitioned history fold, O(N²p²) → O(N log²N p²), 51.7x on
  the fold, 4.5x on the production 3D run, band-against-modal
  **1453x → 45.6x** on DD-231's fixture, every certificate |S11|
  unchanged to the digit, floor 9 dB *better* than DD-231 recorded.
  DD-231's default stands (its blockers were the axis refusal and the
  missing `a()`/`b()`, not the cost).  **On a 201-point axis no item
  dominates any more** (DD-247): postprocessing 32.8 %, kernels 30.1 %,
  build mode tracking 24.9 %, field 5.9 %, convolution 4.4 % —
  314.9 s → 81.2 s, and the two behind the postprocessing are *port
  build*, together 55 %.  The **arc-fan lead is void**: DD-244's
  continuation runs the full fan at 4 of 402 axis points, so the cut is
  ~5 % of the axis, not 3.12x of the largest item, and DD-235's `k`
  gate need not be earned.  Left: postprocessing is `eigs` + `splu` at
  96.6 % over a per-frequency LU that cannot be amortised (shift *and*
  matrix move with frequency); unpriced beside it, one factorisation
  **per channel** per frequency and `k = 4` where one mode is consumed
  — both bite on multi-conductor cross-sections, not on this
  one-channel fixture.  The like-for-like default-axis run (22610 steps
  under the DC anchor) remains unmeasured.
* **Band port floor (KB-038)** — wordlength question answered, defect
  not fixed.  The convolution state was **already double**, so the probe
  the register proposed was a no-op; the single-precision contact is the
  per-step round trip through the field array in `update_e`, which alone
  reproduces the length law and covers 84-92 % of the double-to-single
  gap (the two sides partly cancel — quantising one is worse than both).
  What is left is a *solver* decision (port plane and first interior
  period in double, bulk single), not a port-side one; not priced.  The
  law is **not band-specific**: the ordinary modal port erodes at the
  same rate but saturates at the float32 floor (−112.6 dB) where the
  band port runs past it, and the band-median does not move at all — now
  documented for users in `docs/methods/precision.md`.  Full register
  entry and dossier `investigations/kb038-wordlength/`.
* **Ports on the GPU** — only `TestBandDTBCOnGPU` (KB-045) exercises a
  port on a device; `tests/conftest.py` pins the suite to NumPy.
* **The launch pair (DD-239 → DD-244 → DD-248) — closed.**
  `port_source="dispersive"` launches a rank-2/3 family and carries the
  per-frequency split with it; tutorial 09 reads **−32.88 → −38.90 dB**,
  the endpoint being DD-239's far-port floor to 0.07 dB.  Default
  unchanged.  Left behind it: the decomposition **overshoots unity
  transmission** (|S21| 1.0030 frozen, 1.0078 dispersive, rank-
  independent, growing with f) — DD-244's to own; and the far-port
  floor is now the *only* term left, so the band DTBC's decibels reach
  the user again.
* **Facet section engine (KB-043)** — the reach campaign is closed
  (DD-240/242/243 close KB-039, KB-041, KB-042 and KB-044).  Open:
  **KB-043**, pre-existing and two-sided — within ~1e-7 m of a
  generatrix the kernel section
  collapses to 0.0 while the facet path books 44–64 % of truth
  (r = 2.30 mm, d = 1e-7: 2.7619e-07 facet / 0.0 kernel / 4.2895e-07
  true), so neither is trustworthy there and widening the tangency band
  would hand those planes to the worse one — which is why it stays a
  rounding guard.  Undecided beside it: `radians(5)·|c_n|` binds in
  every fixture tested and its origin is undocumented, so the corrected
  sagitta exponent is largely latent and the measured facet/exact
  bit-identity is a consequence of that cap, not a structural
  guarantee; and `_FACET_REFINE_FRACTION = 0.1` leaves the facet path a
  3.16x finer sagitta budget than the exact one.  One guard test
  (`TestConicRunSurvivesAnUnbuildableArc`) flaked 4/4 in a tree-copy
  window and has passed 13/13 since — unreproduced.
* **API blueprint (DD-224) — Phases A–D complete** (listed above);
  Phase E ff. is a reserved-name roadmap, not scheduled work, each
  entry earning its own DD.  Field-source limits: the recording lives
  in memory until the run ends; a conductor crossing a box face is
  warned about, not handled; a box open at a PEC/PMC wall records fewer
  than six faces; replay completes no symmetry planes.  Every auxiliary
  state (absorber, ADE, SIBC, port) starts an initial-field run
  quiescent rather than in the steady state a mode would have built —
  stable, worth −0.31 % on a SIBC wall Q — and a general incident field
  must solve Maxwell itself, or it leaks.  Blueprint: internal record
  `investigations/api-blueprint/`.
* **Symmetry planes — known limitations (DD-154/DD-155/DD-172).**
  Lumped ports/elements on a symmetry plane are corrected since DD-172;
  ParaView's FlipAllInputArrays mirrors H like a polar vector.  CPML
  min/max faces are not mirror images (KB-023) — full-vs-half parity of
  resonant open structures floors at ~1e-2.
* **Ports with several signal conductors** report the channel's own
  reference in `dispersion()` rather than a modal power–current
  impedance (no single Ampère loop; DD-244); `TDResult` carries neither
  reference impedances nor dispersion records.
* **Mesh build** — speed campaign closed 2026-08-29 (DD-201…DD-223):
  `benchmarks/bench_mesh_build.py` reads 16 Lange couplers 9.6 s at
  3.7 M cells, 240 posts 1.8 s, 16 × 16 patch array 6.4 s at 1.8 M.
  Deferred work, A/B switches, traps: DD-223.  Open against it: KB-043.

Closed construction sites are tombstoned where they were decided and
are not repeated here.

## Deferred / nice-to-have

* **Pulsed band-edge S-parameters on dispersive lines** are record-
  truncation limited; candidate: late-time AR estimation.
* **A third compute backend** — assessed, nothing built (DD-180);
  blocker is `xp is not np` as the capability test.  Metal rejected (no
  FP64), CuPy on ROCm is the candidate.
* **Residual GPU small-grid floor** (~0.41 ms/step at 10k cells, port
  round trips — DD-092); **tensor (gyrotropic) μ** (DD-089's ADE is
  scalar per axis); **off-Yee field-monitor interpolation** (must
  preserve the DD-085 units); **far-field accepted power on the streamed
  path** (``gain`` raises until it wires ``1 − Σ|S|²``, DD-070).
