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

**Two entries are open as of 2026-08-31: KB-023 and KB-027.**
Everything else is struck through and resolved.

## KB-037: ~~Two builds of the same band port gave different Galerkin subspaces~~ — Resolved (2026-08-31)

`zeta_pencil.find_propagating_modes` called `spla.eigs` without a start
vector, so ARPACK began from a random one.  The band subspace is spanned
by the *traces* of the tracked mode families, so the randomness
propagated into it: two builds of the same port from the same mesh gave
projected exterior blocks differing by 35-113 % entrywise, with the
entrywise magnitudes agreeing to 1e-5 and every norm intact.  Most of
the difference was a per-basis-vector sign flip (the SVD gauge), but
~1e-5 of genuine numerical variation remained underneath it, so a sign
convention alone would not have fixed it.

This is the same defect as KB-010 (DD-142) in a third place: a fixed
start vector had been applied to `numerical_2d.py`'s two `eigsh` calls
and to `spectral_dt`, and the pencil eigensolve was missed.

Nothing measured wrong because of it — the subspace is a basis, and the
Galerkin projection is invariant under a change of basis to the accuracy
above.  What it blocked was **resume**: a resumed run rebuilds its
operators and reloads the boundary state from the checkpoint, and a
rebuilt subspace that differs from the recorded one makes the two
inconsistent while every norm still looks right.

Fixed by the shared `magnelio._arpack.arpack_v0`, which the three
callers now share.  Two builds of the same band port are now bit-
identical, in one process and across processes (measured 0.0e+00).
Measurement record: internal dossier `investigations/port-model-default/`
(`probe_band_reproducibility.py`, MEASUREMENTS.md section 12).

## KB-036: ~~Faces in a conductor's end wall blocked and the wall unbooked on grids below about 15 µm~~ — Resolved (DD-207, 2026-08-28)

A grid plane coinciding with a face of the model is sampled a small
step to either side (DD-106); the step was the section deflection, a
hundredth of the smallest cell.  On the Lange coupler's 6 µm grid that
is 60 nm — below the kernel's confusion (1e-7) and the edge tolerances
of a Boolean result (1.5e-7) — and there the section Boolean reports
the face the plane was meant to leave on *both* sides: the air body's
finger pocket (12.6 µm × 5 µm) appeared in the section outside its own
end wall, the spurious opening fell to the conducting background, and
every face lying in a finger's end wall read fully blocked with a wall
jump of zero (min = max, DD-106's min-convention had nothing to
choose from).  The threshold is the kernel's, not a plain distance: a
12.6-µm pocket on the body's bottom face answered correctly at the
same 60 nm, one in the interior did not; a single brick is protected
by the bounding-box screen.  Every model whose smallest cell is under
about 15 µm at scale 1 was exposed; the ε average on the same planes
was affected through the same mechanism.  Fix: the step is the larger
of the deflection and four times the largest B-Rep tolerance of the
model — which also puts the shifted planes past the planar engine's
tolerance screen, so they are answered exactly instead of by the
Boolean.  `tests/unit/test_section_slab_index.py`; measured in
`investigations/mesh-build-bench/MEASUREMENTS.md` (M10, internal
record).

## KB-035: ~~Far-field power deficit of about a tenth with a window port in an absorbing face~~ — Resolved (DD-204, 2026-08-27)

Misattributed when opened.  The deficit was not the window port's:
the same lossless patch element on a lumped port radiated 0.91 of its
accepted power too, and the Poynting flux through the Huygens box
reproduced the accepted power to a percent in every configuration —
the transform of those surface fields fell short, by 7 % with the
domain top 0.3 λ above the copper (the how-to's `h_box` of 12 mm) and
by nothing from 0.7 λ upward, independent of lateral clearance,
substrate extent, grid grading and angular resolution; halving the
cells took the shortfall to 3 %.  The box sits at the absorbing
faces, and 0.3 λ above a printed resonator its discrete near field is
not the outgoing free-space field the transform assumes.  The reading
in the original entry was also inverted: the pattern amplitude, hence
the realized gain, was 0.3–0.4 dB low, while directivity, normalised
to `P_rad` itself, was right.  The window port adds only the
documented few percent of outer-wall current beyond the box
(`P_surf/P_acc` 0.965 on the launch), which the balance does not see.
Fix: `FarFieldResult.surface_power`/`power_balance` and a warning
from `MonitorFarFieldFrequency.result` beyond 5 % imbalance; the how-to's
`h_box` is 0.7 λ.  Measured in `investigations/patch-array/MEASUREMENTS.md`
(M18, internal record); `tests/unit/test_far_field_closure.py`.

## KB-034: ~~Thin sheets and wires touching an absorbing face had no mask in the PML~~ — Resolved (DD-198 amendment, 2026-08-27)

DD-198 step 0 mirrors the first interior slab of the sub-cell data into
the PML extension so a conductor touching an absorbing face keeps its
PEC mask there.  It ran before the thin-wire and thin-sheet passes,
which paint `pec_mask_edges` afterwards, so a thin metallisation
reaching a CPML wall — a microstrip feed with a port window in that
wall — was conductor inside the domain and free space in the
extension.  A `PortWaveguide` window on the wall then saw a hollow
cross-section over substrate and air and was refused with the
"inhomogeneous or anisotropic filling" message; without a port the
sheet simply ended one cell short of the wall.  Fix: step 4c repeats
the mask-only extension after the sheet pass
(`tests/unit/test_pml_extension.py::test_pml_slabs_carry_a_thin_sheet_touching_the_face`,
`::test_microstrip_window_in_an_absorbing_face_resolves_as_a_line_mode`).
The Holland material correction of a thin wire is still not continued
into the extension — a wire ending on an absorbing face is not a
supported feed.

## KB-033: ~~The 3D viewer refused bodies of a few tens of micrometres~~ — Resolved (DD-201, 2026-08-27)

`plot_3d` tessellates every body with a linear deflection of 5e-4 of
its bounding-box diagonal, floored at 1e-12.  OCC rejects a deflection
below its confusion precision (1e-7 in kernel units) with a
`Standard_NumericError`, so a model at metre scale with a body under
~200 µm — the ribbon bonds of the Lange coupler, 66 µm across — made
`model.plot()` raise.  The ParaView exporter carried the same floor.
Fix: `_tessellate_shape` floors every deflection at 1.1e-7
(`tests/unit/test_plot_3d.py::TestTinyBodies`).

## KB-032: ~~Two thin sheets at one nominal height left a sliver anchor pair~~ — Resolved (DD-201, 2026-08-27)

Thin-metallisation planes are verbatim anchors of the plane merge, like
user-forced planes.  A brick and a Boolean-returned track on the same
substrate come back with one ulp of float wiggle between their
substrate-side faces (0.000254 against 0.00025399999999999994 m on the
Lange coupler), so the merge saw two anchors 5e-20 m apart, warned
"forced planes … closer than min_feature_gap … (user positions win)"
for planes no user had forced, and computed the singular-edge grading
from a feature size of 1.7e-21 m.  The grid itself deduplicated the
sliver downstream, so the damage was the misleading warning and a
growth-factor warning of ratio 1e14; the run that appeared to hang
alongside it had a different cause (a closed housing ringing in band).
Fix: `_unify_thin_sheet_positions` clusters sheet planes within the
feature gap before they become anchors — a user-forced plane within
reach wins, otherwise the lowest sheet — and updates the sheet specs
so their masks land on the shared node
(`tests/unit/test_thin_sheet_detection.py::TestThinSheetAnchorUnification`).

## KB-031: ~~Hollow conductors lost the conformal correction at their inner walls~~ — Resolved (DD-199, 2026-08-26)

The kernel Boolean returns the contours of a section without a
winding convention — a tube's bore and its rim came back with the same
sign — and `compute_face_material_areas` sums signed areas per shape.
A dual face inside the hole was therefore covered by the outer contour
and by the hole alike, booked fully PEC, and the sub-cell fractions at
the inner wall of every hollow conductor degraded to the staircase
value: a PEC tube in air on a 1.5 mm grid had a mean |f_A − exact| of
0.12 over its z-dual faces, the bore-wall faces at 0.000 against
0.997.  DD-102 had recorded the independent contour orientation and
judged it harmless.  DD-199 winds every contour by nesting parity
before the kernels see it (`orient_nested_contours`); the tube is at
4e-3 afterwards.  Dielectric bodies with a conductor in their hole
were shielded by the priority rule (the conductor claims its area
first), which is why coax-class models did not show it.  Dielectric
bodies with an *air* hole were not shielded: a ceramic ring's bore was
booked as ceramic, so the KB-011 fixture (ε_r = 45 ring, 4/2 mm,
resonating at 2.3279 GHz) moved to 2.6566 GHz on an unchanged grid
when the winding was fixed (solid puck 2.2302 GHz — the old value was
a nearly filled bore).  The fixture, the DD-191 chamfer certificate
and tutorial 13 were re-based on 2026-08-27.

## KB-030: ~~Monitors fed by a TE/TM port were normalised to the waveform, not to the incident power~~ — Resolved (DD-198, 2026-08-26)

The far-field and frequency monitors divided their bins by the
excitation waveform's spectrum, which equals the incident power wave
only for feeds with a frequency-flat wave impedance (lumped, TEM,
quasi-TEM).  A TE/TM port launches ``|a(f)|² = |W(f)|² Z(f_calc)/Z(f)``
per unit waveform, so gain and radiated power carried the shape of the
mode impedance: an open-ended 20 × 10 mm tube at 10 GHz reported
``P_rad / P_acc = 0.77`` with a PEC flange and 0.82 in an absorbing
box.  DD-198 wires the ratio ``|a(f)| / |W(f)|`` of the separated
incident wave into the monitors (0.91 / 0.97 afterwards, the remainder
being the feed-guide approximation of the far-field chapter); feeds
with flat impedance are untouched by construction.

## KB-029: ~~A conductor touching an absorbing face lost its PEC mask inside the absorber~~ — Resolved (DD-198, 2026-08-26)

The mesher continues the cell materials into the CPML extension slabs
(step 3b), but the conformal classifier works against the B-rep solids,
which end at the nominal bounding box: inside the extension every edge
read as free space, and the Cat-2 un-mask dropped the PEC mask of a
conductor's surface exactly in the slabs the absorber occupies
(measured on a 20 × 10 mm PEC tube, 2 mm grid: 156 of 284 Ey PEC edges
left in slab 0).  Staircase meshes and ``background="pec"`` (DD-049)
were correct.  Step 3d now copies the first fully interior slab's
sub-cell data into the extension — the same translation-invariant
continuation the materials already had.

## KB-028: ~~Four conformal reference tests fail since the DD-191 / DD-192 mesh changes~~ — Resolved (DD-191 amendment / DD-193 note, 2026-08-26)

Found while running the full integration suite for DD-196, bisected
with `git bisect --first-parent`, settled the same day — one
regression, one re-pin:

- The three Dey–Mittra TM010 tests of `test_conformal_convergence.py`
  failed since 6ca4049 (DD-191): the cylindrical cavity is inscribed
  in its PEC block, the Boolean splits wall and cylinder along the
  four touching lines, and the "two faces on one surface" skip of the
  edge pass let those four-face edges through as geometry — a plane
  through the cylinder axis, 7 × 7 → 8 × 8 cells with grid nodes on
  the tangency cusps, DM error 3.7 → 6.0 %.  A **mesher regression**;
  the edge pass now skips every edge at which each adjacent surface
  continues on both sides (DD-191 amendment; a latent `gp_Ax1.Distance`
  error in the coaxial-split test fixed alongside), the grids are
  bit-identical to 0.4.4 again.
- `test_conformal_coax_sparams.py` reported z_line 48.94 Ω against the
  pinned 48.12 since a188229 (DD-192 merge, but the mover is DD-193):
  the exact-fill grading turns the 0.121 / 0.168 / 0.121 mm ramp
  inside the inner conductor into 3 × 0.137 mm cells.  Closer to the
  analytic 49.97 Ω, port floor −131 → −135.6 dB — **merely different,
  and better**; re-pinned.

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

## KB-022: ~~Pair coupling accepts ladder candidates 100x looser than the transparent-boundary gate~~ — Resolved (DD-228, 2026-08-30)

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

Closed the second way (DD-228): the provenance is explicit and the
silence is gone.  The pairing records which accepted targets rest on a
residual above the gate's own 1e-8, the port build restricts that
record to the faces its gate reads, and a withheld exact termination
now warns — port, channel, measured spread, and the mesh-side cause
where there is one.  The warning covers the marginal band only
(1e-8 to 1e-4): further out the cross-section is genuinely
inhomogeneous, which is the model the user built, not a defect.  The
decision is also published per channel
(`ModeReport.termination` / `chain_spread`), so `solve_ports()`
answers the question before a run is paid for.

The first way was refuted by measurement.  Tightening the pairing
tolerance to the gate does not reject *wrong* ladders, it rejects
merely unpinned ones — and what replaces them is the Krietenstein
value, the wrong LC partner on a line.  On the coupler it drops 1 008
of 24 295 coupled targets and moves both ports' pair spread away from
the gate (0.1055 → 0.1180 and 0.1149 → 0.1175); on clean conformal
geometry the band is empty and the change is a no-op.

What is *not* closed is the underlying estimator: two jittered ladders
can still agree at `rtol` and fail the gate.  DD-165's conditioning
rule remains the best available choice, and the tolerance band remains
where jittered geometry lands.  The defect that made this an entry —
that it happened invisibly — is gone.

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
