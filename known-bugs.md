# Known Bugs and Limitations

Resolved bugs are kept as short entries pointing at the design decision
that fixed them; the full record lives there.  Entries fixed without a
dedicated DD keep their record here.

**No entry is open as of v0.2.1 (2026-08-17)** — every KB below is
struck through.  One item is documented rather than fixed and has no
entry of its own: pair coupling calls two ladder targets equal at a
relative 1e-6 while the transparent-boundary gate demands 1e-8, so a
port whose two candidates differ inside that band is decided by
conditioning rather than by agreement.  DD-165 makes that choice
optimal without closing the gap; KB-017 holds the measurement.

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
