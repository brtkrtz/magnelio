# Magnelio — Project Status

*Last updated: 2026-08-25.*  Latest work: **the bulk cell size follows
the slab's wavelength** (DD-192, branch `feat/local-wavelength-rule`,
unmerged): `Mesh.from_geometry` used one bulk size from the densest
material anywhere, so the air box around a small ceramic or above a
thin substrate was meshed at the ceramic's wavelength.  Now each axis
interval between grid planes — a slab of the domain — is meshed for
the densest material whose analytic bounding box reaches into it
(background included; a shape without a box counts everywhere);
`MeshControl(wavelength_rule="global")` restores the old rule.
Feature refinement, grading, the DD-107 buffer and the DD-191 edge
floor (global reference) are unchanged; the PML depth follows the
boundary slab.  Measured: a 10 × 10 × 2 mm ε_r = 4.3 block in an
80 mm air box, 20 nodes/λ at 10 GHz: 1.43 M → 254 k cells, identical
cells inside the block.  A dense background now enters the wavelength
(it was silently ignored).  Tutorial re-run: see the DD-192 line.
Previous: **geometry-edge planes**
(DD-191, released 0.4.5): the mesher's
face pass reads planes, cylinders and spheres, so a chamfer (a cone) or
a fillet (a quarter cylinder whose tangents lie outside its trim) never
produced a grid plane, and — because the DD-051 material average is
taken over the dual face *transverse* to the edge — a feature varying
*along* the edges inside one cell layer had no effect until it crossed
the layer's midplane (the DR-filter worksheet's plateau-and-16-%-jump
chamfer, M4/M4a).  Now every sharp B-rep edge lying flat in an
axis-normal plane yields a *soft* plane: one cell across the feature
layer, floored at `h_max / MeshControl(max_edge_refinement=4)` (and at
`min_cell_size`), never outranking a material plane, one warning per
mesh naming the coarsest dropped edge and the ratio that keeps it;
`max_edge_refinement=0` gives the 0.4.4 meshes bit-exactly.  Excluded:
seams, degenerated edges and the split lines a Boolean fuse leaves
between coplanar sub-faces; a thin sheet's far face is dropped from
the edge pass as from the face pass; the DD-107 buffer no longer
triples a single-cell feature interval at a port-blind face.
Certificate `validation/edge_plane_chamfer_certificate.py` reproduces
M4 with the pass off and gives monotone f0 with it on.  Cost on the
tutorials: 13 +50–80 % cells (dz 1.0 → 0.49 mm at the chamfer), 18
+10 %, 10 +20 % (line/ring junction edges), the rest unchanged.
**Side finding:** tutorial 13's "f0 drifts 9.9 %, k 0.4 %" was mostly
the chamfer switching on between its two grids; with the chamfer
resolved on both, f0 moves 0.8 % (40 % of the passband) and k 0.8 %
(of the bandwidth) between the default grid and (mnpw 24, mcpf 6) — the tutorial is re-based on a grid pair that scales
both mesh knobs and argues from tolerances, not from "ratios are
converged".  New how-to
`plot_mesh_convergence.py`: the convergence loop as a drop-in recipe
(rung → `MeshControl` before the user's simulation, complex-ΔS stop
rule with 0.02 on two consecutive rungs after it; capacitive-patch
microstrip converges at mnpw 32, pillbox TM010 at 1 % on rung 24 with
0.7 % true error) — the answer to "adaptive refinement" as a recipe,
not a mesher feature.  Measured: on a 16 mm feed the complex ΔS is
S21 *phase* (grid dispersion), a bare-line ladder gives the same
numbers; magnitude-only Δ|S| is 3–4× smaller.
Previous: post-hoc de-embedding
(DD-187, branch `feat/port-deembedding`, merged): `result.deembed(
{"port": d})` shifts reference planes on the exact discrete chain
dispersion — the on-circle `lambda^{-d/dz}` from the certified
`(r, q, dz)` line records — and cancels a uniform feed line to the
run's own floor (measured −119.9 dB TEM / −67.4 dB TE10 at 8 cells/λ;
the continuum `exp(-γd)` would leave 3°–10° of grid dispersion, which
is why discrete is the default and continuum only the fallback for
uncertified channels).  Reference plane confirmed to sit exactly on
the port plane (no half-cell/half-step offset).  Works on RAM and
store results; returns an `SParameterResult` that now shares
`phase`/`plot_s` via the new `SDerivedAccessors` base; lumped ports
raise.  On top of it: the **How-to guides** gallery (DD-188,
`examples/howto/` → `docs/howto/`, unnumbered pages) with the
stripline pickup/kicker page moved out of the tutorials (old
`tutorials/plot_19_…` URL lapses, changelog notes it), the filter
capstone re-anchored as the close of tutorials 01–12, and the first
lumped-port termination guides (DD-189, three developer-review
amendments): **four pages** — *Lumped ports: investigations*
(principle + sensitivity sweeps for coax, microstrip and CPW) and one
compact *Lumped port tuning* pre-flight tool per line type (given →
knobs → derived → scoreboard, fixture cut as gallery thumbnail).
Waveguide port as measuring instrument, WG–WG reference run as
grid-exact phase ruler, phase polarity normalised to n·180° at the
low band edge, no de-embedding and no optimiser in the pages.  MS:
vertical trace-end port, position optimum ≈ −1.0·h_sub on the example
grid, dispersion tilts the compromise.  CPW (third amendment, after
the developer's reference model): symmetry-plane end-gap port — strip
ends an end gap short of the ground behind it, lumped port bridges it
longitudinally on the ``SymmetryPMC`` plane, PMC lid, coax knobs;
position optimum ≈ **+16·s beyond** the plane (21.4° → 0.74°, sign
opposite to coax/MS).  Side find: empty boolean results crashed
`plot()` via uncaught C++ exception → KB-026, closed by DD-190.
Unit suite 2234
passed / 3 skipped (DD-192 added 18, DD-191 22).  **Released v0.4.5
(2026-08-25)** with DD-191 and the mesh-convergence how-to; before
that v0.4.4 (2026-08-25) with DD-190, whose merge
had turned CI and Docs red first — VTK segfaults on GPU-less runners
(no EGL device, no libOSMesa; conda-forge `mesalib` is an empty
metapackage), fixed by `pyvista/setup-headless-display-action` in both
workflows.  Before that: released v0.4.2 (issue-#2
boilerplate cuts DD-185/186 + DD-184 export fix) and v0.4.3 (hotfix:
`Loft` missed the DD-185 name resolution); new versioning rule in
CLAUDE.md — while 0.x, PATCH covers everything backwards-compatible,
MINOR is reserved for breaking changes.  Before that: the
Touchstone/scikit-rf
export covers the excited channels instead of demanding the complete
square matrix (DD-184, issue #3) — an unexcited channel is matched by
its own port boundary, so the sub-matrix is the network seen with it
terminated; the `.sNp` extension is now checked against the exported
port count and filled in when absent, and an export that drops
*propagating* higher modes at a port it keeps warns.  Before that:
the stripline pickup/kicker page (DD-183; written as tutorial 19,
since DD-188 a how-to guide) documents the
pickup/kicker workflow of beam instrumentation on the public API
alone.  A stripline pair is dimensioned with 2-D port solves (strip
height for 50 Ω in the difference mode), driven as a kicker in its
sum and difference modes by the wall type on the plane between the
strips, and the beam-coupling figures follow from one line monitor:
beam voltage `∫E_z e^{-jk_B z}dz` (sign fixed by directivity, 25:1),
transverse kick by Panofsky–Wenzel from the gradient off the electric
wall, pickup transfer impedances by reciprocity.  Against the
ideal-stripline formulas of Goldberg/Lambertson with the feed-to-feed
electrical length: curve shapes reproduced, peaks 10–20 % lower
(transit-time factor of the extended end fields), position
sensitivity 7.2 %/mm vs 7.6 ideal.  The coax feed needs
`min_cells_per_feature=8` to sit within 5 % of 50 Ω — the
`min_cell_size` floor does not refine it.  Script: ~2:20 on the CPU
(two 3-D runs of 0.38 M cells plus ten 2-D port solves).  Before that: periodic
structures have an eigenmode solver (DD-182) and tutorial 18 (TESLA
mid-cell, 0-mode 1.2766 / π-mode 1.3010 GHz, coupling 1.89 % vs
1.87 % published); the boundary metric books a full dual cell on every
domain face, so the far plane must be stripped before the congruence.
The plane-wave tutorial remains deferred.

## Recent decisions

Newest first, one line each; the full record is the DD entry.

* **DD-192** (2026-08-25) — bulk cell size per axis interval from the wavelength of the densest material whose bounding box reaches into that slab (`MeshControl(wavelength_rule="local")`, default; `"global"` = old rule), background counted; feature refinement, grading, buffer and edge floor unchanged; ceramic-in-air 1.43 M → 254 k cells.
* **DD-191** (2026-08-25) — geometry-edge planes: a grid plane wherever a sharp B-rep edge lies flat in an axis-normal plane (chamfer/fillet onsets, loft sections, iris circles), as a soft class — one cell per feature layer, floored at `h_max / max_edge_refinement` (default 4) and `min_cell_size`, dropped edges reported once per mesh with the coarsest position and the ratio that keeps it, `0` = the old meshes.  Closes the DR-filter worksheet's invisible-chamfer artefact (M4/M4a: dual-face averaging is transverse-only — a feature varying *along* the edges has no lever until it crosses the cell midplane).  Traps recorded: Boolean-fuse split lines between coplanar sub-faces are not edges; the thin-sheet far face re-enters through the imprint's edges; the DD-107 buffer would triple a single-cell feature interval.
* **DD-190** (2026-08-25) — `model.plot()` rebuilt on PyVista: axis-aligned cutting plane from the widget toolbar (normal / slider / flip / undo / reset) that caps every solid and lays the exposed grid cells, coloured by assigned material, over the cut (the grid shows nowhere else); wires, ports, elements, symmetry planes and the domain box overlaid in mm; browser-side rendering by default (`mode=`), a VTK window in scripts, screenshots in the gallery (tutorials 01/02 now show the 3D view).  Transport is trame's own websocket — JupyterLab ≥ 4.5 executes comm messages in ipykernel-7 subshell threads, and VTK rendered there gave black frames / kernel aborts (proven by replaying the comm transport).  `pyvista` is a core dependency, `[jupyter]` extra for the widget; pythreejs path gone; closes KB-026 as a side effect.
* **DD-189** (2026-08-24, three amendments) — lumped-port termination guides, four pages: *Lumped ports: investigations* (principle + sweeps for all three line types) plus a compact *Lumped port tuning* tool each for coax/microstrip/CPW.  Waveguide port as instrument, WG–WG reference run as grid-exact phase ruler (no closed-form dispersion), phase polarity normalised to n·180° at the low band edge, knobs = end-gap geometry / position / port impedance as geometric re-run sweeps (de-embedding, the single overloaded page, and the CPW slot-port+resistor scheme all dropped on developer review).  CPW = coax picture on the symmetry plane: longitudinal end-gap port (`SymmetryPMC` half model, PMC lid), position optimum +16·s *beyond* the plane.  Kept as general (non-guide) knowledge: `elements=` for post-mesh lumped elements, single-mode test shields.  All three line types shipped; side find KB-026 (empty boolean crashes plot()).
* **DD-188** (2026-08-24) — second sphinx-gallery "How-to guides" (`examples/howto/` → `docs/howto/`, unnumbered pages, own toctree caption): task recipes separated from the numbered curriculum; stripline page moved out of the tutorials (old URL lapses), filter capstone re-anchored as the close of tutorials 01–12; renumbering the tutorials rejected (URL + prose breakage).
* **DD-187** (2026-08-24) — post-hoc reference-plane shift `result.deembed({"port": d})` on the exact discrete chain dispersion: `lambda^{-d/dz}` evaluated on the unit circle (passband magnitudes exactly untouched; the off-circle `lambda_symbol` offset would bias by `O(1e-8·d/dz)`), continuum `γ(ω)` fallback for uncertified channels, lumped ports raise; measured to cancel a uniform line to the run's own floor with zero reference-plane offset; `phase`/`plot_s` moved to `SDerivedAccessors`, shared by run results and `SParameterResult`.
* **DD-186** (2026-08-24) — the mesh carries the f_max it was built for: `Mesh.from_geometry` records it, `AnalysisScatteringTD(f_max=None)` defaults to it, an explicit value above it warns (undersampled grid), `from_grid` meshes keep requiring an explicit value; rejected the issue's session-global "last used f_max" buffer (execution-order-dependent).
* **DD-185** (2026-08-24) — built-in materials by name: `"air"`/`"vacuum"`/`"pec"` accepted (case-insensitive) at every public material argument, resolved at the call site to canonical shared instances; `"copper"` deferred to a curated material library; sticky last-used material and `bg=`/`mat=` aliases rejected (issue #2).
* **DD-184** (2026-08-23) — a Touchstone export is the square sub-matrix over the *excited* channels (issue #3): unexcited channels carry their reflection-free boundary all run, so dropping them is the matched-termination condition of the S-parameter definition, not a truncation; `channels=` selects a sub-network explicitly; a warning fires only for propagating modes dropped at a port that is itself exported (mode conversion missing from a file that looks complete); the `.sNp` extension must match the exported port count — Touchstone records it nowhere else — and is filled in when absent; supersedes the completeness rule of DD-112.
* **DD-183** (2026-08-22) — Tutorial 19, pickups and kickers as post-processing of a kicker run: beam voltage with `exp(-j k_B z)` for a +z particle (monitor convention, verified by directivity), symmetry plane between the strips selects sum/difference mode and drives both ports (2 W), `z_line_num` under it is the pair impedance, ideal-stripline reference with feed-to-feed length; no library change.
* **DD-182** (2026-08-21) — Bloch-periodic eigenmodes: a `"Periodic"` face pair plus `AnalysisEigenmode(phase_advance_deg=…)` solves the unit cell by a congruence `P^H A P` (far-plane edges = near-plane × e^{-iφ}; far plane stripped of its full-dual-cell metric first — the half-cell assumption was refuted by the empty box); real path for 0/π, complex Hermitian on SuperLU in between; verified against the discrete dispersion of the empty box (1e-8) and half-cell band edges of an iris pillbox (1e-3); CPML and unpaired Periodic now rejected instead of solved as PMC.
* **DD-181** (2026-08-21) — `Path.ellipse_to`/`Curve.ellipse_arc`: elliptical arcs as profile segments with the `arc_to` centre/normal vocabulary; OCC's major-first rule absorbed by an axis swap with a quarter-turn parameter shift.
* **DD-179** (2026-08-20) — Board import: `io.import_pcb` reads a Gerber/Excellon/`.gbrjob` fabrication set into one `ImportedSolid` per stackup layer plus one per plated barrel; own readers written from the Ucamco spec (no kernel dependency, unsupported constructs refused by file and line); 2-D face set per layer then a single extrusion, so no Boolean is three-dimensional; barrels fill their cuts exactly (union volume = sum of volumes); copper at its real thickness rides the DD-059/DD-124 thin-sheet path because a layer is one board-spanning solid; construction at a private `fine_detail_scale` power of two; dielectrics numbered `dielectric_n` because layout tools name every core after its material; loss tangent reported, never modelled.
* **DD-178** (2026-08-20) — CAD import: `io.import_step` (XCAF names, colours, file unit normalised to meters with the process-global setting restored; assemblies flattened into a `Group` with placements baked in) and `io.import_brep` (`unit=` mandatory); materials assigned by solid name with literal-beats-wildcard, errors on conflicting patterns and on keys matching nothing, unmapped solids stay construction bodies; `_LoadedShape` became the public `geo.ImportedSolid`; file colour is hue-only, opacity stays with the material; gate: STEP box meshes identically to the equivalent `Brick`.
* **DD-177** (2026-08-19) — the TF/SF correction is a coefficient table: `attach()` folds beta·metric·amplitude per box face and keeps the retardation separate (scalar or 1-D for axis-aligned k), so the time loop runs one array expression per face; 245.7 → 31.8 ms/step (CPU) and 2024.8 → 16.1 ms/step (GPU) at 75 200 boundary cells, injection no longer measurable against the source-free baseline; behaviour preserved to 3e-16 (double).
* **DD-176** (2026-08-19) — geometry arguments are checked where they are written: `geo/_validate.py` guards every constructor and verb (point/vector shape, sign and finiteness, operand type, edge selector), each message naming the argument; a scalar `Sphere(center=)` used to surface as `'float' object is not iterable` inside `pad_box` during `model.plot()`.
* **DD-175** (2026-08-18) — a field picture stands for a cell layer: `plot_cross_section(slab=)` raises the in-plane tolerance of volume-free features to half the displayed cell, filled by the plotting monitors from their grid; wires and discrete ports were absent from every field plot (measured: plane snapped 1.79 mm off the node they sit on).
* **DD-174** (2026-08-18) — pattern plots: `plots.plot_pattern_cut` (polar, antenna convention, dB floor) and `plots.plot_pattern_3d` (dB-radius surface); `FarFieldResult`/`MonitorFarField`/store reader delegate.
* **DD-173** (2026-08-18) — far field from a Huygens box: `monitors.MonitorFarField` (auto-placed node-plane surface DFT) + `post.ntff_transform`/`FarFieldResult`; PEC/PMC/symmetry faces via image theory (`mirror_sign` is the image-current table); effective-amplitude intensity |E|²/η; certificate: dipole 2.15 dBi, monopole +3 dB, P_rad closure 2 %.
* **DD-172** (2026-08-18) — a lumped element on a symmetry plane is half a device: full-model declaration, builder clips/scales internally (series cut Z/2, parallel cut 2Z), `LumpedPortReport` rides the existing √2-per-plane plumbing; invalid placements raise instead of clamping; certificate: exact restriction 5e-16.
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
the 19 gallery tutorials, no internal imports.  All run to completion
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
  ``MonitorFluxTime``, ``MonitorWallLoss``, ``MonitorFarField``),
  ``magnelio.circuit``
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
in tutorial 09.  A second convention came out of v0.2.1: **a tutorial
derives plot scales from the data, never from an absolute constant.**
The open-boundary tutorial pinned `vmax` to a number chosen when
monitor data was still field·seconds, so the 0.2.0 unit change put
every point above the ceiling and the panel rendered as one flat
colour block.  Relative parameters (`threshold`, `density`) came
through the same change untouched.  It surfaced only in CI, because
sphinx-gallery re-executes a tutorial when *its script* changes, not
when the library under it does — a local build without
`build_docs.sh --clean` shows cached figures from an older library.  Pillars: Tutorials (generated from
`examples/tutorials/*.py`, tutorials 01–19 shipped and given a
reader-perspective polish pass — full gallery build ~8:40, clean, of
which the board tutorial is 2.4 s;
tutorial 13, the DR-filter capstone, is deliberately the most
expensive page at ~5.5 min since the design path is the content),
API reference (high-level page + one page per component namespace),
Numerical methods (thirteen chapters, every method with citations,
in-house derivations marked in prose), Bibliography.
`docs/references.bib` holds 63 entries with bibliographic data only —
the citation-confidence bookkeeping lives exclusively in the
maintainers' internal records (internal record:
`reference_docs/provenance-ledger.md`); public docs and BibTeX carry
no verification labels or notes.

**Two published documentation channels (DD-171):** the site serves
`/stable/` (built from a `v*` tag) and `/dev/` (built from main), with
the root redirecting to stable, a version switcher in the navbar fed by
one shared `switcher.json` at the site root, and a banner on every dev
page.  Pages is served from the `gh-pages` branch — a file store the
Docs workflow clones shallow, writes its own channel into and pushes,
so a main push leaves the release docs untouched; the `.nojekyll`
marker there is load-bearing (Jekyll would hide `_static/`).  A build
knows its channel from `MAGNELIO_DOCS_CHANNEL`; unset — every local
build — is dev.

**Planned-run pre-registration (DD-070 follow-up):** multi-excitation
analyses pre-register every planned run as ``pending`` in the run
index (`ProjectStore.register_planned_runs`), so the project status
no longer flickers to ``"done"`` between sequential runs.  Reader
skips ``pending`` in aggregates, raises a clear error on per-run
access; watcher idiom: poll ``status``, skip ``state == "pending"``.

## Open construction sites

* **Symmetry planes — known limitations (DD-154/DD-155/DD-172).**
  Lumped ports/elements on a symmetry plane are corrected since
  DD-172 (full-model declaration, internal half-device scaling, exact
  restriction certified); ParaView's FlipAllInputArrays mirrors H
  like a polar vector (magnitude right, mirrored-half arrow sign
  inverted).  CPML min/max faces are not mirror images (KB-023) —
  full-vs-half parity of resonant open structures floors at ~1e-2.
* **Quasi-TEM de-embedding is quasi-static (KB-027).**  Channels on
  modal Mur carry no certified line parameters, so `result.deembed`
  removes `exp(jβd)` with the frequency-flat `ε_eff = C'/C'_0` of the
  Laplace mode and leaves the line's physical dispersion in the
  de-embedded matrix — measured 22° of S21 phase over 16 mm of
  0.8 mm FR4 microstrip at 15 GHz, growing with substrate thickness,
  zero for ε_r = 1.  Needs a frequency-dependent quasi-TEM mode to
  close; until then keep quasi-TEM feeds short or compare raw S.
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
* **A third compute backend** — assessed 2026-08-21, nothing built
  (DD-180).  The blocker is not the amount of CUDA (204 lines, no shared
  memory or intrinsics; 1.2 % of `src/` is backend-specific) but that
  `xp is not np` is the solver's capability test, so any third array
  module takes the CUDA path.  Metal is **rejected on bandwidth**: CPU
  and GPU share one memory on Apple Silicon and reach 1.0×…1.4× of each
  other, while the loop is bandwidth-bound — and Metal has no FP64, so
  `precision="double"` would be lost.  CuPy on ROCm is the candidate
  worth the effort (same array API, CUDA source translates nearly
  verbatim), but the launch geometry is measured on one Ada card and
  would need re-measuring.  Nothing merges without a run on the actual
  hardware: CI runs `tests/unit` only, so every cross-backend gate is a
  local run today.
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
* **Far-field accepted-power wiring on the streamed path** — DD-173
  wires ``1 − Σ|S|²`` into ``MonitorFarField`` after each in-RAM run;
  a store-streamed run serves ``realized_gain``/``directivity`` but
  ``gain`` raises until the reader wires accepted power (DD-070
  follow-up).
