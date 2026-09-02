# Magnelio — Project Status

*Last updated: 2026-09-02.*  **Released v0.5.0** (2026-09-02;
DD-224…DD-241 — the first minor bump under the Cargo reading, ten
breaking changes, upgrade guide `docs/migration-0.5.md`; the public
remote has a content gate since DD-241).  In it: **DD-224 Phases A–D complete** (the API
grammar, `AnalysisTD` + `TDResult`, project-store schema 2.0,
`magnelio.fields.FieldState`, field and current-path sources,
tutorial 20), **the pair-product gate as a reflection budget**
(DD-228/229, closes KB-022) and **the section-engine reach repair**
(DD-240, closes KB-041, opens KB-042/KB-043).  The band/QTEM track
**DD-230…DD-239 is closed, with every shipped default unchanged**:
project-store runs, `port_model="modal"`, a DC-reaching axis,
bit-exact resume (closes KB-037), the −70 dB pricing, the short-basis
warning, the production cost ranking (opens KB-038), the dead middle
path between Mur-1st and the band DTBC, the two-term cancellation
behind the modal-Mur port floor (DD-238), and the pricing of the number
a user actually reads (DD-239: an exactly transparent port is worth at
most 1.93 dB of tutorial 09's |S11|).  Open: KB-023,
KB-027, KB-038 (diagnosed — the single-precision production default),
KB-042 and KB-043.  Unit 2820 / 4 skipped and integration 452 / 5 skipped
(2026-09-02, with `CUPY_ACCELERATORS=""` — without it four GPU tests
fail on nvrtc in the sandbox).  Channels: GitHub, PyPI, conda-forge and
the two docs channels below.

This file states what *is*.  Chronology: `git log --first-parent main`
(one feature per merge); reasoning: `design-decisions.md`; open bugs:
`known-bugs.md`.  Measured floors regenerate from the `validation/`
certificates named in their DD entries.

## Recent decisions

Newest first, one line each; the full record is the DD entry.

* **DD-241** (2026-09-02) — the public repository has a content gate, not only a branch gate.  The workspace `pre-push` hook had guarded *which* commits reach the public remote and never *what is in them*; the rules about content — no commercial solver named anywhere in the tree, private-workspace citations labelled as internal records, no home paths or addresses — lived in the assistant's own notes, and the review before the first push after a model change found them broken twice (one vendor name public since v0.4.4, in every PyPI and conda-forge artefact because the sdist packs the developer records — reworded forward, it cannot be taken back; a second one in all 45 unpublished commits — removed by rewriting them).  `validation/tools/check_public_hygiene.py` is the gate: pre-commit and CI audit the tree, the `pre-push` hook audits **every commit of every range** pushed to the public remote against what the remote already carries — the form that matters, since a line that entered in one commit and left in a later one is published all the same, and exactly such a line is what the tip-only audit missed and the range audit found.
* **DD-240** (2026-09-01) — the free-form section gate was per *shape*, not per *face*, and that cost ports their exact termination two bodies away.  One B-spline face set a shape-wide flag, `_PlanarSectionEngine._build` handed the whole solid to the facet path and returned before the exact tables were ever assembled, so the analytic cylinders of that solid — and, through a kernel-fused Boolean, of the *air* body that owns the port cross-section — were sectioned against a triangulation.  A triangulated cylinder is a prism, so the chord depends on where the plane falls between node rows: the coupler bore `M_μ` Hx read 8.911369e-10 / 8.920923e-10 / 8.920996e-10 / 8.920997e-10 over four y-layers against the kernel's uniform 1.550977e-09 — 42.5 % low *and* drifting, and it is uniformity the exact termination consumes, so `_port_chain_slab_defect` read 8.4165e-02 against its 1e-8 gate and demoted both ports to Mur (`MAGNELIO_FACET_SECTIONS=0` on byte-identical geometry certifies it: both dtbc, slab defect 2.2192e-10; distance from the loft was irrelevant because the damage is lost invariance, not lost accuracy).  **The repair answers analytic faces of a facetted shape from their own geometry** — `_project_cylinder` solves the in-plane projection exactly, so a plane parallel to the axis lands on the generatrix and parallel planes book identical areas (cheaper than the one-step Newton lift it replaces), genuine conic runs are laid down as exact conics while the generatrix case is deliberately left uncompressed (chording it makes the fixture worse, 8.4e-02 → 1.4e-02), and a shape with no free-form face stays bit-identical.  Two mitigations are measured dead: a per-plane fallback is neutral (8.4165e-02 unchanged — the deviating masses all cross the loft's x-extent) and not lifting analytic faces at all reaches only 1.4957e-02 (a raw prism is invariant along its axis only while untrimmed, and the bore is Boolean-trimmed).  Two defects in the repair itself were found adversarially and fixed before landing: the open-run branch dropped a conic run before knowing its replacement arc could be built, collapsing it to a straight chord (89.8 % and 32.7 % / 100 % of a section area on high-aspect cylinders, reproduced independently on two fixtures), and a plane *exactly* tangent to a cylinder manufactured a full circle out of a degenerate trace (1.5386530746873848e-06 m² where kernel and pre-repair tree both book 0.0; radius sweep 4/16 → 0/16, screened with a relative `_TANGENCY_ROUNDING = 1e-12` because one case missed at 4.337e-19).  On the way the arc refinement step's `|c_n|³` was **re-derived as `|c_n|¹`** — the section is an ellipse with semi-axes r and r/|c_n|, the sagitta maximal at u = ±π/2 (max-over-u against prediction 0.999988…1.000000 at c_n = 3e-3…0.9) — and fixed at all three sites; it was the sole cause of the declined-arc regime (8/8 → 0/8), the achieved sagitta drops from 152.6x the requested deflection to 0.0003x, the closed-form area error from −7.5e-03 to −1.1e-08, and a 0.4 mm wire at 0.057° from 74204 points / 9517.2 ms to 11996 / 160.4 ms.  Blast radius, measured by isolated A/B: unit 2820 and integration 452 green, ten pinned artefacts pass and nine are bit-identical (the chamfer certificate digit for digit, its 2.66 GHz re-pin intact; 30/30 mesher stress including five tangent-cylinder fixtures); the one drift is a printed diagnostic, PEC area on Hx +7.22e-9 relative, adjudicated **toward** truth (3.4 % closer to the converged value, the two arms agreeing to 3e-10 at a 1e-4 deflection fraction).  Cost neutral where the angular cap already bound, 4.0x faster on the tilted-cylinder regime the exponent targets.  Open behind it: KB-042 and KB-043 (below).  Certificates: `validation/section_chain_completeness_certificate.py`, `validation/pair_ladder_choice_certificate.py`.
* **DD-239** (2026-09-01) — the campaign closes with every default unchanged, because the port floor is not what caps the number a user reads.  The tutorial 09 microstrip reproduces through the public path (|S11| worst −32.88 dB against the printed −32.9 dB, |S21| −0.000 dB) and its three error terms are priced one at a time against exact counterparts: at 15 GHz the a/b split is −28.98 dB, the drive-port source −30.17 dB and the far-port floor −43.73 dB, and the three recombine as phasors onto the run within 0.3 dB at 5 / 10 / 15 GHz (instrument floor −172…−183 dB, 118 dB of headroom).  **An exactly transparent port is worth at most +1.93 dB** of the reported |S11| (−0.45 / +0.72 / +1.93 dB, +1.81 dB on a 2.3x finer mesh), while removing split and source together buys +3.68…+24.03 dB — so DD-238's low-rank band port would aim at the smallest of the three terms.  The recorded cap formula for the split is **refuted**: read as the reflection it leaks, `(βΔz/2)³(1−r²)/6` prices 50.2–73.0 dB below the measurement and moves 37.68 dB over an Nz = 25…104 ladder on which the measurement moves 0.34 dB.  The split is 99.5 % **wave impedance**: `compute_s_parameters` values a QTEM channel at the frequency-flat quasi-static 103.03 Ω while the discrete wave carries 106.40 / 108.45 / 110.55 Ω (−3.17…−6.80 %, unmoved across the whole mesh ladder), so |b/a| = |ΔZ/Z|/2 = −29 dB at 15 GHz; a 2×2 control correcting impedance and de-stagger phase separately lands at −316…−323 dB with both corrected, so the split has exactly those two parameters and no third.  Split and source are the **same defect twice** — the frequency-flat quasi-static port mode — 170.9° apart at 10 GHz, which *is* the −46.6 dB mid-band dip tutorial 09 prints.  The named successor is arithmetic, not a new boundary: the ζ-pencil already computes Z(ω) and γ(ω) per port per frequency, and feeding those to the power-wave split instead of the quasi-static value is the candidate (it needs the whole rfft axis, not one bin; cost not measured).  The same close-out resolved KB-040 (a contention artefact — CPU-pinned, the 2D port mode build is at most 1.05x / 1.42x / 2.56x its pin, while wall clock on the same call spread 63x between load 0.15 and load 68), diagnosed KB-038 and bisected KB-039 to DD-199, opening KB-041 (closed one day later by DD-240).  Certificate: `validation/qtem_modal_mur_sparam_floor.py`.
* **DD-238** (2026-09-01) — the modal-Mur port floor is a two-term cancellation, and the one gain DD-237 left as free is refuted by the measured run it demanded.  The instrument is `|a2/b2|` at the *passive* port under an exact-DTBC true-mode generator (own floor −145…−176 dB, 90+ dB of headroom, CW lock-in, never `compute_s_parameters` — its de-staggered split caps at −40…−60 dB, above the effect; line length irrelevant, 0.00 dB between Nz = 24 and Nz = 120).  The shipped floor is **better than the a priori said**: worst in band −52.3 dB microstrip, −36.5 dB layered, −55.6 dB block against DD-237's −38.9 / −27.2 / −47.7 — the scalar Γ is an *exact* instrument for what it prices (0.01 dB against measurement on a true discrete profile) but prices only one of the port's two error terms, so it is pessimistic by 7.5–14.9 dB and is **not a bound**.  The mechanism: `build_modal_port` takes the Mur coefficient *and* the mode profile from the same frequency-flat quasi-static Laplace mode, the two errors are comparable and measured 161–180° out of phase, and the floor is their residual — block at 6.2 GHz closes to three digits, boundary 4.103e−3, profile 2.443e−3, residual 1.660e−3 = the measured −55.60 dB.  The 2×2 with the exact-DTBC control shows correcting *either* term alone is worse and correcting **both** lands on the instrument floor (−154.4…−157.3 dB against the control's −156.1), so there is no third error term.  Moving `r` a fraction α toward `r_exact` is worse at α = 1 in 8/8 points (2.7–3.4 dB block, 7.6–11.9 layered, 14.3–15.7 microstrip) and the optimum is **interior** and fixture- and frequency-dependent — ≈ 0.50 block, ≈ 0.25 layered and drifting down with frequency, at most 0 on the production microstrip — because it sits where the two magnitudes are *equal*, which needs a discrete mode solve per port per frequency, i.e. the band path; the shipped `r` sits near a cancellation optimum by accident.  So the minimax objective was wrong in kind (the floor is `|Γ_bnd − Γ_prof|`, not `|Γ_bnd|`), the coefficient is not calibrated on its own, and — the forward coupling worth remembering — a better *mode profile* alone would raise the floor too unless `r` moves in the same step.  Nothing under `src/` moves; the user-visible S11 is a third quantity and the −30 dB-class figures in DD-064, the docs and the runtime notices are untouched.  Certificate: `validation/qtem_mur_floor_decomposition.py`.

Older decisions: `design-decisions.md`.

## Working practices earned the hard way

* **Verify a numerics fix across the mesh-control range, not on the
  mesh that exposed it.**  DD-147 was checked at the setting that
  triggered it and at one neighbour; the same collapse was waiting two
  cell sizes away.  And a guard written as `== 0` against a computed
  quantity fires half the time — use a threshold (DD-149).
* **Derive a refinement law from the maximum over the parameter, not
  from a spot check.**  The section arc's sagitta at a *fixed* u is
  `r·du²/(8|cos u|)` and carries no tilt dependence at all, so a probe
  at one point argues for exponent 0; only the maximum over u — at
  u = ±π/2 — yields the exponent the law needs, 1 in place of the
  shipped 3, which had been over-spending points (DD-240).
* **A run that never advances looks exactly like a run that needs more
  steps.**  Compare `dt` against `courant_dt(mesh.grid)` (no material
  factor) and read `result.reference_signal`: a monotone 1e-18 ramp is
  a Gaussian tail that has not arrived.  The energy line's `0.0 dB`
  means "current = running maximum", which a barely-started run reports
  just like a resonant one (DD-147).
* **A cached OCC solid is shared mutable state.**  OCCT Booleans edit
  their arguments and a Boolean *result* shares sub-shapes with its
  operands, so damage propagates backwards into the user's bodies
  (DD-146); when geometry misbehaves only after something else ran,
  measure `BRep_Tool::Tolerance` over edges and vertices —
  `bounding_box()` stays right while the solid becomes unusable.
* **Stage hunks, never whole files.**  The working tree is shared: an
  uncommitted experiment (`energy_stop_db` 70 → 40) was once swept into
  a commit by a whole-file `git add` and broke 21 physics tests.
* **Worktree A/B runs need `PYTHONPATH=<worktree>/src`** — the editable
  install pins the main checkout's `src`, so without it the run imports
  the main tree and the attribution is meaningless.
* **A cost pinned on a loaded box is not a cost.**  The same port build
  measured 28.6 ms at load 0.15 and 1801 ms at load 68 on 16 cores —
  63x of spread at constant CPU work, which is how KB-040 was opened.
  Pin thread-limited CPU time, or run alone (DD-239).
* **Re-check the script directories after every API break.**  Nothing
  under `examples/`, `validation/`, `benchmarks/` or the internal
  `investigations/` dossiers has test coverage, so a rename breaks them
  silently — ten scripts once failed at import for months;
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

* **Band-pipeline runtime** — priced end to end at production size
  (DD-236): boundary convolution 59 %, kernel build 14 %, and on a
  201-point axis the postprocessing 51 %; the fixture's 12-13x modal
  (DD-234) has no production counterpart.  Leads: the exact blocked FFT
  convolution, then the arc-fan cut (3.12x, bit-identical, `k`-gated).
* **Band port floor (KB-038)** — diagnosed, not closed: both halves are
  the single-precision production default (numbers above).  What is
  open is a wordlength decision, not an absorber hunt — does the
  band-DTBC convolution state have to be double? — answered by one run
  with the solver in single and the port convolution forced to double.
  An integration test guards the degradation *rate*, one-sided, so a fix
  cannot fail it; CI never saw the defect because `tests/conftest.py`
  pins the suite to double, and nothing in DD-240 displaces it.
* **The quasi-static power-wave split (DD-239)** — the live QTEM item,
  in place of a better boundary.  `compute_s_parameters` values a QTEM
  channel at a frequency-flat quasi-static impedance: 99.5 % of the a/b
  split error, and the same mode injected is the drive-port error.  The
  ζ-pencil already computes Z(ω) and γ(ω) per port per frequency, so
  the candidate is arithmetic — feed those to the split, over the whole
  rfft axis rather than one bin.  Cost not measured; nor is the drive
  term separated into injected profile against the drive port's own
  re-absorption, nor the decomposition repeated on a homogeneous
  pair-certified TEM line, where the split should vanish.
* **Facet section engine (DD-240; KB-042, KB-043)** — the reach defect
  is closed for **planar and cylindrical faces**: analytic faces of a
  facetted shape are sectioned from their own geometry, and the
  pair-ladder coupler's ports terminate DTBC again (spreads 5.2164e-15
  / 1.3072e-13, slab defect 1.4795e-10, z_line 96.1625 Ω) with its
  loft in place — the model that failed is the model that passes, so
  the fixture question resolved itself and KB-039 is closed.  **KB-042** is the scope hole: cone, sphere
  and torus faces of a facetted shape still take the one-step
  parametric Newton lift, worst per-cell deviation against the kernel
  8.65e-03 / 9.32e-03 / 7.30e-03 against the repaired cylinder's
  2.17e-13 — seven to nine times what motivated the fix — measured with
  the free-form neighbour fused 40 mm from planes that never cross it,
  and every fillet or chamfer puts a torus in the model.  Closing it
  means extending the exact engine too, which admits only planes and
  cylinders.  **KB-043** is pre-existing and two-sided: within ~1e-7 m
  of a generatrix the kernel section collapses to 0.0 while the facet
  path books 44–64 % of truth (r = 2.30 mm, d = 1e-7: 2.7619e-07 facet /
  0.0 kernel / 4.2895e-07 true), so neither is trustworthy there and
  widening the tangency band would hand those planes to the worse one —
  which is why it stays a rounding guard.  Undecided beside it:
  `radians(5)·|c_n|` binds in every fixture tested and its origin is
  undocumented, so the corrected sagitta exponent is largely latent and
  the facet/exact bit-identity now measured is a consequence of that
  cap, not a structural guarantee; and `_FACET_REFINE_FRACTION = 0.1`
  leaves the facet path a 3.16x finer sagitta budget than the exact one.
  One guard test (`TestConicRunSurvivesAnUnbuildableArc`) flaked 4/4 in
  a tree-copy window and has passed 13/13 since — unreproduced.
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
* **Quasi-TEM de-embedding is quasi-static (KB-027).**  Channels on
  modal Mur carry no certified line parameters, so `result.deembed`
  removes `exp(jβd)` with the frequency-flat `ε_eff = C'/C'_0` of the
  Laplace mode and leaves the line's physical dispersion in the
  de-embedded matrix — measured 22° of S21 phase over 16 mm of 0.8 mm
  FR4 microstrip at 15 GHz, zero for ε_r = 1.  Needs a
  frequency-dependent quasi-TEM mode to close (the DD-239 successor
  above); until then keep such feeds short or compare raw S.
* **Mesh build** — the speed campaign closed 2026-08-29 (DD-201…
  DD-223): `benchmarks/bench_mesh_build.py` reads 16 Lange couplers
  9.6 s at 3.7 M cells, 240 posts 1.8 s, 16 × 16 patch array 6.4 s at
  1.8 M, every ladder row on its reference (`pool/hash_refs/`).
  Deferred work, A/B switches, traps: DD-223.  Open against it:
  KB-042 and KB-043.

Closed construction sites are tombstoned where they were decided (the
DD entry and `known-bugs.md`) and are not repeated here.

## Deferred / nice-to-have

* **Pulsed band-edge S-parameters on dispersive lines** are
  record-truncation limited; candidate: late-time AR estimation.
* **A third compute backend** — assessed, nothing built (DD-180);
  blocker is `xp is not np` as the capability test.  Metal rejected
  (no FP64), CuPy on ROCm is the candidate.
* **Residual GPU small-grid floor** (~0.41 ms/step at 10k cells, port
  feedback round trips — DD-092); **tensor (gyrotropic) μ** (the DD-089
  ADE is scalar per axis); **off-Yee field-monitor interpolation**
  (must preserve the DD-085 units).
* **Far-field accepted-power wiring on the streamed path** — the reader
  serves ``realized_gain``/``directivity``; ``gain`` raises until it
  wires ``1 − Σ|S|²`` (DD-070 follow-up; DD-198's ratio *is* derived).
