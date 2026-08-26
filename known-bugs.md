# Known Bugs and Limitations

This file is the record of *investigated* defects: what was measured,
what characterises the defect, and why an open one stays open.  The
`KB-` numbers are stable anchors that code comments and design decisions
point at, so resolved entries stay as struck-through tombstones instead
of disappearing.

It is not where a bug gets reported.  That is the
[issue tracker](https://github.com/brtkrtz/magnelio/issues) — the
inbound channel, where a report arrives and its reproduction is settled.
A report that turns out to be a structural defect earns an entry here
and the issue links to it; most do not need one.

Resolved bugs are kept as short entries pointing at the design decision
that fixed them; the full record lives there.  Entries fixed without a
dedicated DD keep their record here.

**Four entries are open as of 2026-08-26: KB-022, KB-023, KB-027 and
KB-028.**  Everything else is struck through and resolved.

## KB-028: Four conformal reference tests fail since the DD-191 / DD-192 mesh changes — Open (2026-08-26)

Found while running the full integration suite for DD-196 (both
failures reproduce on `main` before that change, bisected with
`git bisect --first-parent`):

- `tests/integration/test_conformal_coax_sparams.py::test_conformal_coax_port_floor_and_impedance`
  reports the conformal coax line impedance at 48.941 Ω against the
  pinned 48.12 Ω ± 1 % (DD-053 measurement).  First bad commit:
  a188229, *Merge feat/local-wavelength-rule* (DD-192 bulk cell size
  per slab, DD-193 short-interval fill) — the coax mesh changed, and
  with it the staircase/conformal line impedance by 1.7 %.
- `tests/integration/test_conformal_convergence.py::TestDeyMittraTM010::
  test_dm_improves_over_conformal`, `::test_dm_improves_over_staircase`
  and `::TestDeyMittraConvergence::test_dm_preserves_convergence_order`
  fail since 6ca4049, *Merge feat/edge-feature-planes* (DD-191) — the
  cavity meshes gained edge planes and the TM010 error ordering the
  tests pin (Dey–Mittra < conformal < staircase, second-order
  convergence) no longer holds on the new grids.

Not yet investigated: whether the new grids are *worse* (a mesher
regression — a coarser cell at the coax conductor, an edge plane
that breaks the DD-107 buffer) or merely *different* (the pinned
numbers belong to the old grids and the tests need re-pinning with a
fresh convergence run).  The v0.4.6 release note counts the unit
suite only; the integration suite was not green at release.  Until
settled, the four tests are the known red set of the integration
suite.

## KB-027: De-embedding a quasi-TEM feed leaves the line's physical dispersion behind — Open (2026-08-25)

`result.deembed` removes the *discrete* chain propagation only on
channels the run certified with line parameters `(r, q)` — the DTBC
channels of homogeneous lines.  A quasi-TEM channel (microstrip, CPW,
any inhomogeneous cross-section) is terminated by modal Mur in the
default pipeline, carries no line parameters, and falls back to the
mode's continuum `γ(ω)`.  That `γ` is the **quasi-static** one of the
2D Laplace solve: `ε_eff = C'/C'_0` is frequency-flat, so the
fallback removes a dispersion-free phase from a line whose real
propagation constant rises with frequency.  The difference stays in
the de-embedded S-matrix and is attributed to the device under test.

Measured (internal record `investigations/port-deembedding/`): a
16 mm shielded microstrip (w = 1.2 mm, t = 0.2 mm on 0.8 mm ε_r = 4.3,
box 8 × 5 mm, PMC symmetry) de-embedded over its full length at
`min_nodes_per_wavelength = 32` leaves a residual S21 phase of
−1.1° / −7.9° / −22.4° at 5 / 10 / 15 GHz.  The residual is physics,
not grid: it is unchanged across the ladder 16 → 48 nodes/λ (−24.4 →
−22.0°, a 1/N² tail on top of a −21.6° limit), it vanishes for ε_r = 1
(−0.13° at 15 GHz, the mechanism itself is exact), and it scales with
the substrate — 12.7° / 22.4° / 39.9° at h = 0.4 / 0.8 / 1.6 mm — the
signature of microstrip dispersion (Getsinger's ε_eff(f) predicts the
same order and the same saturation with h).  The same quasi-static
`γ` sets the Mur reflection coefficient of the channel, which is part
of why QTEM channels sit at the −26…−39 dB floor.

Consequences: keep quasi-TEM feed lines short when de-embedding, or
judge mesh convergence on the raw S-matrix (the mesh-convergence
how-to does).  Closing it means a frequency-dependent quasi-TEM mode —
a full-wave 2D eigen-solve per frequency, or the band pipeline's
tracked mode families carrying their own `γ(ω)` into the shift.

## KB-026: ~~An empty boolean result crashes plot() with a C++ abort~~ — Resolved (2026-08-25)

**Resolution (DD-190).**  The rebuilt 3D viewer checks each shape's
bounding box before tessellating and skips a shape without extent with
a warning naming it; `plot()` no longer reaches the OCC call that
threw.  The mesher-side symptom (`GridLines.x must be a 1D array …`)
and the wish for validation at `add()` stand as recorded below.

*Original record:*

`GeometryModel.add(a - b)` accepts a `Difference` whose result is
empty (subtrahend covers the minuend, e.g. two equal bricks), and the
failure surfaces only downstream, twice removed from the cause:

- `model.plot()` **aborts the process** — the OCC tessellation of the
  empty shape throws `std::invalid_argument: "The deviation must be
  greater than 0"` (zero bounding-box diagonal → zero chordal
  deviation), the exception crosses the C++/Python boundary uncaught,
  and `terminate()` kills the interpreter.  In a notebook this reads
  as a kernel death with no traceback (found by the developer while
  building the CPW tuning model, internal notebook record).
- `Mesh.from_geometry` fails with `GridLines.x must be a 1D array
  with at least 2 elements` — technically an exception, but naming
  the mesher's internals instead of the empty shape.

Wanted: validate at `add()` (or at boolean construction) that a shape
has volume, and raise a `ValueError` naming the empty operand there —
the same early-error principle as the DD-176 argument validation.
Until then: a model that suddenly "has no geometry" after a boolean
edit is the signature; check the operands.

## KB-025: ~~A cross-section paints its holes shut~~ — Resolved (2026-08-20)

`plot_cross_section` drew every contour `cross_section_polygons`
returned as its own filled polygon.  That function returns "outer
boundaries and holes mixed together" with no winding convention, and
says so: the region is the set of points enclosed an *odd* number of
times, and consumers are to apply the even-odd rule.  Filling each
contour on its own applies no rule at all — a bore is painted in the
same colour as the material around it.

The consequence is not a cosmetic tint.  An opaque shape with a hole
covers everything that sits inside the hole, and shapes are drawn in
insertion order, so the visible picture depends on which body happens
to be added last.  Measured 2026-08-20 on a coaxial line (PEC pin,
PTFE dielectric, PEC shield, added in that order), sampling the
rendered image:

```
r = 0.0 mm (pin)         (166, 166, 166)
r = 1.4 mm (dielectric)  (166, 166, 166)
r = 2.8 mm (shield)      (166, 166, 166)
```

— one flat disc in the shield's colour.  It went unnoticed because the
coaxial tutorials add the inner conductor *last*, which paints it back
on top of the dielectric that had covered it; only a model whose
outermost body comes last shows the full effect.

Fixed in `post/plot_geometry.py`: the contours of one shape become a
single compound path, and each contour's direction is set from its
nesting depth — enclosed by an even number of others it bounds
material and runs counter-clockwise, by an odd number it is a hole and
runs the other way.  Matplotlib fills by the nonzero winding rule, so
that turns the odd-enclosure region into the filled one.  The outline
form air is drawn in stays contour by contour: the wall of a hole is a
wall too.

One trap sits inside the fix.  Matplotlib's `Path(vertices,
closed=True)` *drops* the last vertex to make room for its CLOSEPOLY
code, and section contours arrive without a repeated first point — so
it eats a real corner.  On a tessellated circle that is one chord out
of seventy and invisible; on the four-vertex contour of a rectangle it
leaves a triangle.  The closing segment is therefore written out
explicitly.

Regression cover in `tests/unit/test_plot_geometry.py`, sampled in the
rasterised image because the patch is one compound path either way and
only the renderer answers the question:
`test_a_hole_stays_open` (an annulus) and
`test_nested_contours_alternate` (a rectangular block with a bore and a
free-standing island in it — two contours deep, and rectangular, so it
catches the dropped corner the annulus cannot see).

## KB-024: ~~A missing pythonocc-core reads as an empty mesh, not as a missing dependency~~ — Resolved (2026-08-19)

The geometry backend raises a clear `ImportError` when pythonocc-core is
absent ("pythonocc-core is required for geometry operations.  Install
via: conda install -c conda-forge pythonocc-core"), but the mesher never
lets it through.  `extract_critical_planes_per_shape` wraps each shape
in a broad `except Exception` and skips it — a guard meant for OCC-less
and exotic shapes.  With the dependency missing, *every* shape is
skipped, the per-axis critical-plane lists stay empty, and the failure
surfaces two layers later as a complaint about grid line arrays.

Measured 2026-08-19 on the README's WR-90 quick-start model, with OCC
blocked by the `sys.meta_path` hook `release.yml` uses for its smoke
test:

```
import ok: 0.3.1
geo.Brick ok
Mesh.from_geometry FAILS: ValueError
  GridLines.x must be a 1D array with at least 2 elements
```

Importing the package and declaring geometry both succeed, so the report
a user can give is "meshing fails with an array error".  This is the
first thing a fresh install does, and the pip route is the only one
where the dependency can be absent — the conda-forge package pulls it in.

Fixed by separating the two causes the guards collect: `except
ImportError: raise` now stands ahead of the broad `except` at all three
sites (the bounding box and face queries in
`extract_critical_planes_per_shape`, and the analytic box in
`resolve_feature_gap`), so the backend's message reaches the caller
while an exotic shape is still skipped.  The same run now ends with

```
ImportError: pythonocc-core is required for geometry operations.
Install via: conda install -c conda-forge pythonocc-core
```

Regression cover in `tests/unit/test_geometry.py::TestMissingOccSurfaces`:
each site raises on `ImportError` and still skips on any other failure.
Nothing changes on an install that has the dependency.

## KB-023: CPML min and max faces are not mirror images — Open (2026-08-18)

The CPML profile (σ, κ, α) is sampled at cell centres and the same
per-cell coefficient drives both the ψ recursion of the node-registered
E components and the cell-registered H components.  On a staggered
grid that puts the effective profile half a cell off its staggered
sampling points, with opposite sign on min and max faces — so the two
absorbers of one axis are not mirror images and their residual
reflections differ.  The absorber still meets its `R_target`; only the
*symmetry* between opposing faces is broken.

Measured (internal record, DD-172 parity work): a mirror-symmetric
thin-wire dipole in a CPML box shows a field-level mirror asymmetry of
~1e-4 at the PML interface shortly after the pulse passes, growing to
several percent of the (decaying) local field in the resonant tail —
in double precision, so it is structural, not rounding.  Recycled
through the high-Q antenna it floors the full-vs-half S11 parity of
`validation/lumped_symmetry_parity_certificate.py` at ~2e-2 (gate B);
the same comparison in an all-PEC cavity is exact to 5e-16 (gate A),
and a plain vacuum port under CPML agrees to 3e-6 — the asymmetry only
matters where a resonator re-amplifies the residual.

Closing it means sampling the profile at the true staggered positions
(E at nodes, H at cell centres, measured from the physical interface),
which changes every CPML run's bit pattern and needs its own
reflection-floor re-certification — deferred until a use case needs
mirror-exact absorbers.

## KB-022: Pair coupling accepts ladder candidates 100x looser than the transparent-boundary gate — Open (2026-08-17)

Split out of KB-017, which DD-165 closed for the case that produced it.
The pairing calls two ladder targets equal at a relative `rtol = 1e-6`,
while the DTBC pair-spread gate certifies at 1e-8.  A port whose two
candidates differ anywhere inside that band reaches the gate with a
target that agreement did not pin down.  DD-165 resolves the choice by
conditioning — of two agreeing ladders, the one whose own partners
disagree less supplies the target — which is optimal but not a
guarantee: two jittered ladders would still pass the pairing and fail
the gate.  The failure mode is what makes it worth an entry: the
channel falls back to modal Mur-1st **silently**, trading a 1e-14
termination for a −30 dB-class reflection floor on that port alone,
while a geometrically identical port on the same model keeps the exact
one.

Measured on the stripline coupler (internal record): before DD-165 the
mirrored stub's port2 spread was 1.7e-8 against the 1e-8 tolerance,
where its unmirrored twin certified at 7e-15; after DD-165 the same
port reads 6.3e-14.  The conformal identity KB-017 originally blamed is
not involved — `eps_avg` and `f_A` agree to 3.9e-15 across all 19 244
conformal edges, since both integrals share one area budget.  The
jitter enters through the pairing tolerance.

Closing it means either tightening the pairing tolerance toward the
gate — at the risk of rejecting ladders that are merely coarse rather
than wrong — or making the target's provenance explicit, so that a
disagreement inside the band is reported instead of silently resolved.
Neither has been attempted: no model has been observed to fail this way
since DD-165, so the entry stands as a documented limitation rather
than a reproducible defect.

## KB-021: ~~Half a solid's cross-section goes missing with no warning~~ — Resolved (DD-168, 2026-08-15)

Recorded as a residual of DD-167 and read as the section operator
failing at grazing incidence.  It was not: the kernel produced every
edge, and the wire builder in `cross_section_polygons` lost them.
`BRepBuilderAPI_MakeWire` accepts an edge reaching *any* free end of
the wire so far — including a vertex that already joins two — and the
branched result is not a wire; `BRepTools_WireExplorer` walks one arm
and stops.  Measured on a stripline-coupler electrode (internal
record): fourteen section edges, eight added to one wire, one visited.
No open chain remained, so nothing warned, and thirty cells of metal
were meshed as vacuum.  Section edges are chained on an endpoint graph
now, with branches resolved by tangent continuity.

Worth remembering how the diagnosis went wrong the first time.  "The
section operator is degenerate here" is a plausible reading of a
halved cross-section and it survived a whole session, because both
plausible fixes — nudging further, tessellating finer — do nothing
against it.  What settled it was counting: edges out of the kernel
against edges reaching the tessellation.

## KB-020: ~~A near-tangent section plane drops a solid's whole cross-section on a fine mesh~~ — Resolved (DD-167, 2026-08-15)

Found on a stripline-coupler worksheet (internal record) whose mesher
printed two open-chain warnings with nothing a user could act on.  The
DD-157
retry that steps off a degenerate section plane took its step length
from the tessellation deflection, so the conformal-area pass — which
tessellates ten times finer than the cell classification on purpose —
inherited a ten times shorter reach and could no longer leave
near-tangency bands the classification pass cleared easily.  The two
passes then disagreed: cells classified conductor whose material
matrices saw nothing there.  The escape is now its own length, shared
by both passes.

The warning was the actionable part of the failure and it was not
actionable: it named no body, no amount, and no consequence.  It does
now.  Worth remembering that the natural reading — "the mesh is fine,
so the boundary should be nearly planar in every cell" — is exactly
inverted here: the mesher anchors a grid line on a feature's extreme,
so refining moves the neighbouring cell-centre plane *closer* to the
tangency, not away from it.

## KB-019: ~~The classifier never produces sub-cell data on a domain boundary face~~ — Resolved (2026-08-15)

The conformal candidate mask in `geo/_filling.py` was written only on
the interior index range of each transverse axis, so every partially
filled E-edge lying *in* a bbox face was rounded to fully free or fully
metal.  It is now written on the boundary indices too, with each
boundary edge's dual face clamped to `[wall, first dual line]` — see
DD-164.

The bug stayed open one session longer than the diagnosis, because the
fix had been written and measured and *no certificate improved*: the
pillbox quarter model is blind by construction (TM010's `E_z` vanishes
at the cylindrical wall it cuts), and the band-DTBC floor moved the
wrong way.  What closed it was a certificate with an exact identity and
a known target — a magnetic half model must reproduce its full model to
machine precision — on a fixture whose dielectric contour crosses the
symmetry plane where the mode's tangential E is maximal.  It read
-2.3e-03 and now reads 4.7e-15.  The DTBC floor turned out to be a
kernel-fit residual whose own spread under the fit's resolution is 30 to
52 dB, three to five times the ratio it was being asked to judge.

## KB-018: ~~2D mode profile carries several percent of spurious transverse field at a curved conductor~~ — Resolved (2026-08-15)

Mostly a plot defect, not a solver one: the mode profiles are FIT grid
quantities (edge and face voltages) and the picture read them as field
samples, so every arrow picked up the local cell size.  Dividing by the
edge metric removes the 17 % low reading at the contour and halves the
spurious tangential content — see DD-161, which also records the
measurement error in DD-160 that had pointed the other way.  The
residual (~2.5°, ~7 %) is the ordinary staircase discretisation of the
conductor contour in the 2D solve; it converges under refinement and is
not attributed further.

## KB-017: ~~Pair-coupling tolerance band lets a 7.5e-7 conformal jitter silently push a port channel to Mur~~ — Resolved (DD-165, 2026-08-15)

On the stripline coupler with its mirrored coax stub, port2's only TEM
channel fell back to modal Mur-1st with no warning: the DTBC pair-spread
gate measured 1.7e-8 against its 1e-8 tolerance, while the geometrically
identical stub on the unmirrored side certified at 7e-15.  Fixed by
DD-165: of two valid, agreeing ladders the one whose own partners
disagree less now supplies the target, instead of whichever axis was
listed first.  port2 reads 6.3e-14 and takes the exact termination.

The recorded root cause was wrong, and re-measuring is what showed it.
This entry blamed the classifier for deriving ``eps_avg`` and ``f_A``
from inconsistent integrals of one dual face; on the same model that
identity holds on all 19 244 conformal edges to 3.9e-15, with the
pairing error unchanged.  Both integrals share one area budget and
cannot disagree.

The structural gap the entry named does remain: the pairing calls
targets equal at ``rtol = 1e-6`` while the DTBC gate demands 1e-8.
DD-165 makes the choice inside that band optimal; it does not close the
band, and two jittered ladders would still get through.

## KB-016: ~~Frozen zero-M_eps edges seed NaN Mur coefficients on live complement-absorber edges~~ — Resolved (2026-08-14)

Degenerate conformal edges are clamped to ``M_eps == 0`` without
entering ``pec_mask_edges``; the volume update freezes them
(``live_E = M_eps > 0``), but the port complement absorber's live
mask only consulted the PEC mask.  Such an edge in a port window got
``eps_eff = 0`` → an infinite phase velocity → a NaN Mur coefficient
on a *live* edge (observed: four Ey edges of the stripline-coupler ZL
port sitting on the Boolean cut plane inside the curved electrode
shell).  Latent only because the absorber runs solely when a mode is
on Mur, and the affected port certified for the exact DTBC.  Fixed by
adding ``M_eps <= 0`` edges to the absorber's dead set with a finite
coefficient, mirroring the volume convention; the 0/0 chi-patch
census warning on ``f_A == 0`` edges was silenced the same way (the
isfinite guard already discarded those quotients).  Gate:
`tests/unit/test_port_edge_bc.py::TestComplementAbsorberFrozenEdges`.

## KB-015: ~~Open section chains book fantasy coverage — coax ports fall back to Mur under declared symmetry~~ — Resolved (DD-157, 2026-08-14)

On a plane in the near-tangent band of a curved face of a
tolerance-inflated Boolean union, `BRepAlgoAPI_Section` returns a
mutilated edge set; the wire assembly accepted the resulting OPEN
chains and the polygon consumers implicitly closed them — one
13-point chain spanning both coax bores of the stripline coupler
booked a bore-wall H face at 0.80 free instead of 0.19, broke the
feed-chain slab invariance (defect 0.43) and sent both coax ports to
modal Mur-1st.  Only the uncut full-model body triggered it, so it
surfaced when DD-154 symmetry declarations replaced manual Boolean
quarter cuts.  Fixed by the DD-157 closedness contract (open chains →
nudge retry → loud drop).  Certificate:
`validation/section_open_chain_guard_certificate.py`; gate:
`tests/unit/test_geometry.py::TestSectionAtFace`.

## KB-014: ~~A two-node phantom conductor shadows the real TEM mode~~ — Resolved (DD-156, 2026-08-14)

An isolated PEC staircase fragment above a curved electrode's apex
formed its own conductor group; its near-zero-gap TEM channel has an
enormous C', sorts first in the capacitance-ordered channel basis and
shadowed the real stripline mode at `n_modes=1` (reported z_line
0.95 Ω instead of ~46 Ω).  Fixed by DD-156 label fusion: PEC-cell
corner links decide which edge components are one conductor, without
adding nodes.  Gate:
`tests/unit/test_modal_factory_auto_conductors.py::TestSurfaceFragmentAbsorption`.

## KB-013: ~~A 50 nm domain-boundary offset sends every port channel to Mur~~ — Resolved (DD-151, 2026-08-13)

OCCT Booleans on interpenetrating operands inflate the bounding box by
`Precision::Confusion` (1e-7 model units); the mesher clustered the
inflated extent with the true face plane to their midpoint, the domain
boundary sat 50 nm past the geometry, and the resulting sliver fill
factor tripped the DTBC slab gate — every port channel on the face fell
back to modal Mur-1st, with a misleading warning.  Fixed by plane
provenance: face planes outrank bounding-box extents in the clustering
(full record, measurements and rejected alternatives in DD-151).
Gate: `tests/unit/test_mesh.py::TestPlaneProvenance`.

## KB-012: ~~`GeometryModel.plot()` changes the mesh of a model built afterwards~~ — Resolved (DD-152, 2026-08-13)

The 3D renderer tessellates the cached solids in place, and OCC
bounding-box reads default to `useTriangulation = True` — after a
`plot()`, face boxes came from triangle nodes plus deflection instead
of the analytic geometry, admitting extra critical planes (`N_y`
68 -> 75 on the same model).  All bbox reads feeding meshing and
classification now pass `useTriangulation = False` (full record and
ruled-out candidates in DD-152).  Gate:
`tests/unit/test_geometry.py::TestGeometryQueriesIgnoreTriangulation`.

## KB-011: ~~`AnalysisEigenmode` returns an empty list on sparsely filled high-contrast cavities~~ — Resolved (DD-138, 2026-08-12)

The auto shift assumed a *filled* dielectric cavity (`eps_r_max` over
the material library), so a ceramic puck filling ~1 % of the housing
pulled the shift 2.7× closer to the curl-curl null space than to the
first physical mode — ARPACK converged on null-space vectors and the
run silently returned 0 of 6 modes.  Fixed by the DD-138 escalation
ladder (filled-cavity lower bound, ×4 retries, B-metric merge of
attempts), and under-delivery now warns on every path.  Gate:
`test_analysis_eigenmode.py::TestSparseHighContrastCavity`.

## KB-010: ~~`test_coax_tem_vs_te_tm` fails intermittently~~ — Resolved (DD-142, 2026-08-12)

ARPACK's random start vector made the convergence residual of the
degenerate TE pair wander across the test's 1e-12 cross-projection
gate (measured 3.1e-16 … 1.1e-13 over 30 rebuilds).  Fixed by a fixed
generic start vector for both `eigsh` calls in
`ports/_modal/numerical_2d.py`; the measurement record is in DD-142.

## KB-009: ~~QTEM hybrid modes (n_modes ≥ 2) fail on x-normal port faces~~ — Resolved (2026-08-12)

Found 2026-08-10 during the Wilkinson tutorial groundwork (DD-123..125):
requesting `n_modes=2` on a `PortWaveguide` whose plane is
`xmin`/`xmax` and whose cross-section is transversally inhomogeneous
(shielded microstrip) raised
`RuntimeError: e_u and e_v families have different normal strides;
unsupported flat layout` in `zeta_pencil.build_period_blocks`.  Cause:
`PeriodChain` carried the one-period-inward flat-index offset of the
tangential trace as a single scalar, which only exists on z-normal
faces — there both tangential families stride by 1.  On x-/y-normal
faces `e_u` and `e_v` are different E components whose flat arrays
have different shapes, hence different normal strides (x-normal:
`Ny*(Nz+1)` for Ey vs `(Ny+1)*Nz` for Ez).  Fixed as the code's own
dead comment already sketched: `et_step` becomes a per-edge array when
the families differ; `period()` shifts elementwise, so the block
extraction, invariance certificate and Bloch-field synthesis are
untouched.  Gate:
`test_zeta_pencil.py::TestPeriodBlocks::test_x_and_y_normal_faces_match_z_reference`
(x- and y-normal chains on an axis-permuted fixture reproduce the
z-normal fundamental eigenpair to 1e-9).

## KB-006: ~~MonitorWallLoss crashes on the cupy backend~~ — Resolved (2026-08-10)

Found while fixing the same class of defect in the field/flux
monitors (DD-115): `MonitorWallLoss.record` called
`np.asarray(h_arrays[c])` on device arrays and fancy-indexed them
with NumPy index arrays — on the GPU backend (the production default
since DD-090) this raised `TypeError` at the first recorded step.
Fixed as sketched (found during the Tutorial-11 groundwork): the
wall samples are gathered on the device with device-resident index
arrays (cached per surface on the first record) and only the
per-surface sample vectors cross the bus; the reference-plane slabs
transfer per recorded step like the DD-115 field monitors.  The DFT
accumulators stay host-side, so CPU results are unchanged by
construction.  Gate:
`test_gpu_backend.py::TestWallLossMonitorGPU::test_fraction_matches_cpu`
(GPU fraction ≡ CPU fraction to 1e-12 on the DD-082 parallel-plate
fixture).

## KB-008: ~~`port_signal_stop_db="auto"` can never fire on band-edge cut-off plateaus~~ — Resolved (DD-122)

Found 2026-08-09 on the WR-90 magic tee: the E-arm drive leaves
band-edge ringing at the TE10 cut-off (vanishing group velocity) whose
modal-port |V| envelope plateaus near −56 dB — just above the −60 dB
``"auto"`` threshold — and the default unbounded run marched
indefinitely (>40 000 extra steps with no envelope movement; the
stored energy plateaus too, so ``energy_stop_db`` never fires either).
Root cause: cut-off content decays *algebraically*, not exponentially,
so any threshold below the plateau is unreachable.  DD-122 fixes this
with a stall watchdog (slope projection against the new
``max_time_steps`` runtime cap; accepts the plateau as the effective
floor with a ``RuntimeWarning`` and ``stop_reason =
"port_signal_stall"``) plus the cap itself as backstop.  Certificate:
``validation/wr90_tee_signal_stall_certificate.py`` (stall stop at
step 9101 instead of endless, max |ΔS| = 4.3e-4 vs the
``port_signal_stop_db=50`` workaround reference).

## KB-007: ~~Micron-scale geometry unusable (sub-100-nm features rejected, µm feature planes silently annihilated)~~ — Resolved (DD-120)

The absolute `min_feature_gap = 1e-6` and the OCC kernel's fixed
1e-7 model-unit precision limited reliable geometry to the mm regime.
DD-120's automatic power-of-two unit scaling plus relative tolerance
defaults lift both: micron models build at O(128) scaled units (the
effective feature limit is `1e-7 / s` meters) and the clustering
tolerance scales with the model.  Remaining documented limitation: a
model whose bounding-box diagonal is ≥ 1 mm keeps `s = 1`, so
sub-100-nm features inside such a model are still rejected — resolving
nm features across a mm domain is computationally infeasible anyway.

## KB-001: ~~No user-settable background material~~ — Resolved (DD-038)

## KB-002: ~~CSG shapes with holes~~ — Resolved (DD-038, even-odd rule)

## KB-003: ~~Curved shapes tangent to the domain boundary give cube modes~~ — Resolved (DD-049; wall-mask half superseded by DD-103)

Re-measured after DD-103 on the original fixture (air sphere R = 50 mm
in PEC background, bbox flush with the sphere): lowest eigenfrequency
2.6114 GHz on a 14³ grid and 2.6197 GHz on 22³, three-fold degenerate,
against the analytical sphere TM₁₀₁ at 2.6185 GHz (−0.27 % / +0.05 %,
converging).  The entry recorded 2.115 GHz — the cube TM₁₁₀ mode at
2.1199 GHz — so the rounded-cube cavity is gone.  The PEC padding this
entry prescribed is no longer needed.  Related but distinct and also
closed: bbox tangency as a wall-loss *registration* void (DD-099).

## KB-004: ~~Overlapping shapes "last wins"~~ — Resolved (DD-038, `allow_overlaps=False`)

## KB-005: ~~Conformal eps 2D vs 3D~~ — Resolved (DD-037, thin-box intersection)
