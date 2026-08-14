# Known Bugs and Limitations

Resolved bugs are kept as short entries pointing at the design decision
that fixed them; the full record lives there.  Entries fixed without a
dedicated DD keep their record here.

## KB-017: Pair-coupling tolerance band lets a 7.5e-7 conformal jitter silently push a port channel to Mur — OPEN (2026-08-14)

On the stripline coupler with its mirrored coax stub
(`.mirrored(normal=(0,0,1))`), port2's only TEM channel falls back to
modal Mur-1st with **no warning**: the DTBC pair-spread gate measures
1.7e-8 against its 1e-8 tolerance while the feed chain itself is
perfectly slab-invariant (chain-slab defect 6.6e-11).  Full causal
chain (measured, probes in the internal dossier
`investigations/section-open-chains/`):

1. The sub-cell classifier marks one Ey edge next to the mirrored
   coax bore as category 1 with ``f_A = 1 - 7.5e-7`` while its
   ``eps_avg`` is exactly 1.0 — the two integrals over the same dual
   face disagree by the tessellation jitter of the mirrored surface,
   so ``eps_pair = eps_avg/f_A = 1 + 7.5e-7`` in vacuum.
2. In ``couple_face_material_pairs`` the transverse (z) ladder of the
   adjacent Hx face inherits that target; the ladder is internally
   inconsistent by the same 7.5e-7 but passes the ``rtol = 1e-6``
   agreement test using its jittered ``t1``.
3. The fixed ladder priority (``use_a = v_a & (~v_b | agree_ab)``)
   lets the transverse target overwrite the feed-direction target,
   which is translation-invariant to 1.6e-16.
4. The feed pair identity is violated by 7.5e-7 on that one edge →
   weighted pair spread 1.7e-8 → the 1e-8 DTBC gate withholds the
   exact termination, silently (only the chain-slab branch warns).

Structural issue: the pairing declares targets "equal" at 1e-6 while
the DTBC gate demands 1e-8 — any jitter inside that band produces
certified-looking masses that still fail the port.  Fix candidates,
in root-cause order: (a) make the classifier's ``eps_avg`` and
``f_A`` come from the same free-area integral so ``eps_pair`` is
exactly 1 in homogeneous material; (b) when both ladders are valid,
prefer the one with the smaller internal partner disagreement instead
of the fixed a-before-b priority.  Until then the fallback is the
DD-064 accepted default (−30 dB-class |S11| on that channel);
``port_model="band"`` provides the reflection-free path.

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
