# Magnelio — Project Status

*Last updated: 2026-09-03.*  **Released v0.5.1** (2026-09-03; a patch
under the Cargo reading — nothing in it breaks a 0.5.0 script).  In it:
**DD-242/DD-243 finish the section-engine reach repair** (quadric faces
projected onto their implicit surface, then one chord budget for all
three section paths — curved-face results move slightly toward the
converged value and cost up to a fifth more mesh CPU; KB-041/042/044
closed), **DD-244 makes the reference impedance a published quantity**
(`reference_impedance`, `renormalize`, a per-frequency
`report.dispersion` sweep, and the port plane's own convergence ladder
`refine_port_modes` on a new `MeshControl(subdivide=...)`; KB-027
closed), and **the band-DTBC port runs on the GPU** (KB-045, found and
closed the same day: it had crashed under `backend="auto"` on any CUDA
machine).

Before it, v0.5.0 (2026-09-02) carried DD-224…DD-241 — the API grammar
and its Phases A–D, the pair-product gate as a reflection budget
(DD-228/229, closes KB-022), the content gate on the public remote
(DD-241) — with ten breaking changes and `docs/migration-0.5.md`.  The
band/QTEM track **DD-230…DD-239 is closed, with every shipped default
unchanged**; its closing result is that the port floor is not what caps
the number a user reads (DD-239: an exactly transparent port is worth
at most 1.93 dB of tutorial 09's |S11|), and it opened KB-038.

On `main` since 2026-09-03, unreleased: **DD-245** (the partitioned
boundary convolution, below) and the *Numerical precision* methods
chapter `docs/methods/precision.md`, whose measurement of the
single-precision length law is recorded in KB-038.

Open: KB-023, KB-038 and KB-043.  Unit and integration together: 3289
passed / 10 skipped (2026-09-03, with `CUPY_ACCELERATORS=""`).
Channels: GitHub, PyPI, conda-forge and the two docs channels below.

This file states what *is*.  Chronology: `git log --first-parent main`
(one feature per merge); reasoning: `design-decisions.md`; open bugs:
`known-bugs.md`.  Measured floors regenerate from the `validation/`
certificates named in their DD entries.

## Recent decisions

Newest first, one line each; the full record is the DD entry.

* **DD-245** (2026-09-03) — the boundary convolution is partitioned, and the band pipeline's largest item goes with it.  `advance` folded the whole history every step, O(N²p²), which DD-236 had priced at 59.3 % of an 18-point production run; it is now a directly folded head `[1, 256)` plus levels `[H·2^k, H·2^(k+1))` carried by overlap-save FFTs — each level reaches no closer to the present than its own block length, so its whole next window is computable from history already written and one FFT per `B_k` steps amortises to O(p²) per step, O(N log²N p²) for the run.  The same sum in a different accumulation order: no fit, no approximation, no passivity question.  Measured — the fold alone 6.6x at 4096 steps, 23.1x at 16384, **51.7x** at 36864 and p = 9 (28.16 → 0.54 s), 56.1x at p = 17; against the direct sum, max relative error 3.5e-15…1.0e-14 held against the *norm* of the sum, which is the yardstick a reduction needs; end to end the production microstrip 3D run 109 → 24 s (**4.5x**), the layered case 31 → 11 s, and **every |S11| on both certificate axes is identical to the printed digit**, so no floor is re-pinned.  Two things worth carrying: the accumulator's **build order is part of the resumed state** — replaying it in one pass instead of in window-opening order misses the bit-exact resume gate by 3e-14 — and the block spectra cost **exactly twice the kernel array** (84.6 MB at p = 9 / n_kernel = 65536, 301.9 MB at p = 17, per kernel per port), because a real rfft of length 2B has B+1 complex bins.  Gate `tests/unit/test_band_dtbc.py::TestPartitionedConvolution`; record `investigations/band-convolution/MEASUREMENTS.md`.
* **DD-244** (2026-09-02) — the reference impedance is a published quantity, the port's dispersion is solved on demand, and the port plane converges on its own.  Three user questions (Z(f) of a port mode, what the S-parameters are referenced to and how to move them to 50 Ω, how far the numerical impedance is from converged) had no answer; behind them stands the port operating with one frequency-flat mode while the grid carries a dispersive one.  **Measured first: the DD-239 successor as named does not deliver** — the exact per-frequency split of the production record reads |S11| −44.0 / −30.8 / −28.4 dB at 5 / 10 / 15 GHz against the shipped −38.1 / −46.3 / −32.9, because it removes the split term alone and exposes the drive-port launch residue (DD-239's source term to within 1–2 dB); the default split stays.  What ships instead: (1) `ports/_modal/dispersion.py`, the ζ-pencil continued along an axis (4.0 s against 29.6 s per 201 points, ζ to 4e-13), returning per mode the channel reference `V_in√ζ/I_in` (real to roundoff, DD-239's "106.40 Ω") *and* the power–current impedance of the true mode from its own fields, `Z_PI = 2P/|I|²` (discrete Poynting flux, Ampère loop over the signal conductor's nodes) — the two differ by 2.4 % on the microstrip even at 0.2 GHz, and Z_PI is the one that meets the quasi-static value in the static limit (to 3e-3; to 1e-8 on a homogeneous line), so `PortReport.dispersion(f)` publishes it (51.5 → 53.4 Ω, ε_eff 2.99 → 3.28 over the band, the curve the S21 phase traces); the band decomposition runs on the same module with every mode scaled to unit power (`S21` across dissimilar ports is now a power ratio); (2) `SParameterResult.reference_impedances`, `result.reference_impedance(port)`, `result.renormalize(z_ref)` (exact real-reference power-wave re-referencing, round trip 4e-15, textbook checks), Touchstone stating the real common reference in `R` with `z_ref=` to renormalise and a warning where Touchstone 1.x cannot express the references, `to_skrf` with `z0` per port and frequency; (3) **KB-027 closed** — modal runs keep a self-contained dispersion record per quasi-TEM port (stored under the band schema plus plane masses, curl slice, 2D gradient, conductor nodes) and `deembed` shifts by ζ(f)^(−d/dz): residual S21 phase +0.3 / +1.5 / +1.8° over the whole 20 mm line against −1.5 / −10.6 / −29.5° before; (4) `refine_port_modes` — `forced_planes` cannot reproduce a grid (anchors trigger feature refinement, 35 × 74 for a 14 × 23 grid), so the ladder is a port slab cut by the mesher's own symmetry clip (reproduces the user's port grid and report to 1e-12) plus the new nested `MeshControl.subdivide` of the finished grid; tutorial 09 now designs its line on paper (Hammerstad–Jensen with thickness correction, 1.473 mm for 50 Ω) and its 25-node grid reads it **at 46.0 Ω** (46.0 → 48.8 → 50.2 → 50.9 → 51.2 Ω, order 1.02, Richardson ≈ 51.5 Ω; the shield costs 1.2 Ω, a four-times larger box converges at 52.7 Ω, the rest is the formula's thickness correction at t/h = 0.25), shown next to its dispersion curve and its renormalisation to 50 Ω (−26 dB one-port, a grid artefact).  Certificates: `tests/integration/test_qtem_dispersive_reference.py`, `tests/unit/test_dispersion.py`, `tests/unit/test_modal_refinement.py`, `tests/unit/test_sparameter_renormalize.py`; record `investigations/qtem-dispersive-reference/MEASUREMENTS.md`.
* **DD-243** (2026-09-02) — one chord budget for all three section paths, closing KB-044.  The facet path refined section chords to a tenth of the deflection, the exact engine's conic arcs and every kernel-delegated plane tessellated at the full deflection: a polygon of chords books (2/3)·sagitta too little area per unit boundary length, a radius short by h_min/150 on every cylinder, cone, sphere or torus cut across its curvature (7.0…9.3e-3 of a cell against a δ/1000 reference), **first order in the cell size** because the deflection is tied to h_min — so it halves where the scheme's own error quarters and leads on fine meshes (Δf/f ≈ 3.3e-4 on a cavity of R = 20 h, 8e-5 at R = 80 h) — and a **step** of that size whenever a sweep carried a body from the kernel's path to the facet path (the first non-zero fillet radius), read by a user as the fillet's effect.  Now `SECTION_CHORD_FRACTION = 0.1` of the deflection everywhere: `GCPnts_TangentialDeflection` is fed the budget (open-chain test and nudge ladder keep the deflection), the exact engine's `du_max` and its compiled twin scale by it.  Measured: kernel vs converged 7.0…9.3e-3 → 0.79…1.15e-3 of a cell; a cylinder across its axis is one polygon on all three paths (kernel vs facet 8e-14, exact vs facet 2e-16); the two DD-217 gates that arm A alone broke pass, engine and kernel having moved together.  Price **+19 % mesh-build CPU** on the fillet-heavy probe (5.41 → 6.43 s), above KB-044's +11…12 % because the exact engine's fillet arcs took the finer budget too.  No pinned number moved; the one fallout was a *fixture*: the round coax forced through the QTEM path lost its exactly degenerate TE11 pair (the inner conductor's 72-point circle was four-fold symmetric under the 5° cap, the 130-point one is not; the split, 2.9e-9 in ζ, clears the pencil's 1e-9 dedup and both polarisations certify as real channels), so the refusal gate is now pinned on a square coax, whose symmetry no tessellation touches.

Older decisions: `design-decisions.md`.

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
  the fold, 4.5x on the production 3D run (109 → 24 s), every
  certificate |S11| unchanged to the digit, at twice the kernel array in
  block spectra (84.6 MB at p = 9 / n_kernel = 65536).  End to end on
  DD-231's fixture the band-against-modal ratio falls **1453x → 45.6x**
  (2688.9 → 93.3 s; this entry 3.38x of it, the rest of the stack 8.5x),
  floor 9 dB *better* than DD-231 recorded.  DD-231's default stands —
  its deciding blockers were the axis refusal and the missing `a()`/
  `b()`, not the cost.  Left: the **kernel build** is now the largest
  item of an 18-point run; on a 201-point axis the **postprocessing**
  (51 % at DD-236) is the whole question, where the arc-fan cut is 3.12x
  and bit-identical but wants a gate expressed as a `k` argument, not a
  point count; and the like-for-like default-axis run is possible for
  the first time (22610 steps under the DC anchor, fewer than the 39831
  at f_min = 3 GHz) and unmeasured.
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
* **Ports on the GPU** — nothing outside `TestBandDTBCOnGPU` (KB-045,
  closed) exercises a port on a device: `tests/conftest.py` pins the
  suite to NumPy.  Worth knowing for the next port family.
* **The quasi-static power-wave split (DD-239 → DD-244)** — the exact
  per-frequency split alone is *measured not to help*: it exposes the
  drive-port launch residue and reads −27.9 dB worst against the shipped
  −32.9.  The one route left to the user-visible |S11| without the band
  boundary is a dispersive *source* on the modal port (a low-rank family
  of profiles × waveforms at the plane overwrite — the band port's
  source without its boundary); not started, not priced.
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
  impedance (no single Ampère loop; DD-244), and `TDResult` carries
  neither reference impedances nor dispersion records.
* **Mesh build** — the speed campaign closed 2026-08-29 (DD-201…
  DD-223): `benchmarks/bench_mesh_build.py` reads 16 Lange couplers
  9.6 s at 3.7 M cells, 240 posts 1.8 s, 16 × 16 patch array 6.4 s at
  1.8 M, every ladder row on its reference (`pool/hash_refs/`).
  Deferred work, A/B switches, traps: DD-223.  Open against it: KB-043.

Closed construction sites are tombstoned where they were decided (the
DD entry and `known-bugs.md`) and are not repeated here.

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
