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

**Two entries are open as of 2026-10-08: KB-038 and KB-043 on the default fast route.**
KB-043 has an explicit opt-in mitigation; other entries are resolved.

## KB-054: ~~Repeated edge tags reject a valid filleted CAD section~~ — Resolved (DD-287, 2026-10-08)

Repeated periodic/closing-edge parameter tags were counted as extra crossings,
rejecting valid filleted sections. DD-287 clusters aliases per endpoint without
widening tolerances or changing trace geometry. Independent fillet/circle
references pass conditioning 1/4/16/64; previously retries masked failures at
1/16 locally. Residual, CAD-tube and closure checks remain mandatory. OCCT
7.9.0 and hosted 8.0.1 acceptance pass, including the 320-rectangle certificate.
Evidence: `investigations/release-0.9.0/CAD-FIX.md` (internal record).

## KB-053: ~~Bounded-surface sections reject the repeated coaxial-cell CAD model~~ — Resolved (DD-287, 2026-10-06)

Embedded coedges, trim-vertex gaps and lost periodic/trace identities made
bounded sections reject valid repeated-cell CAD. DD-287 preserves native
domains, actual material wires and closed traces with the original residual/
topology/CAD-tube checks. The unchanged 51-cell/102-port model meshes with
bit-identical grid coordinates (8,582,496 cells); 361 recorded failing cuts,
63 independent area references and 320 material rectangles pass. The initial
1811.762 versus 641.465 s smoke comparison was not a controlled slowdown
measurement; no global physical-error percentage follows from these gates.
Evidence: `investigations/near-tangent-meshing-performance/hesr/REPAIR_MEASUREMENTS.md`
and `REPAIR_DERIVATION.md` (internal records).

## KB-052: ~~Project watching can omit the final writer state~~ — Resolved (2026-10-05)

The generator checked terminal state after yielding a running update. If
the writer completed while the consumer handled that update, iteration
ended before yielding the completed state. A deterministic paused-consumer
test reproduces the missing update. The terminal decision now precedes
the yield, so completion during consumption is observed in the next
iteration. All five watch tests pass. Discovery and reproduction:
`investigations/kb023-staggered-cpml/MEASUREMENTS.md` (internal record).

## KB-051: ~~Energy stopping can discard pending finite excitations~~ — Resolved (2026-10-05)

DD-282 guards energy stopping until every scheduled finite waveform and
synthesised port-drive buffer has completed, including effective delay and
plane-wave spatial retardation. Absolute completed steps preserve that
protection on resume; explicit duration bounds remain available. A delayed
second drive, an active single pulse, a translated plane wave and resumed
drives are covered by regressions. The narrower-band waveguide's complex
S21 error improves from 8.69e-4 to 2.41e-5. Full reproduction and acceptance:
`investigations/termination-accuracy/MEASUREMENTS.md` and `IMPLEMENTATION.md`
(internal records).

## KB-050: ~~Energy-stopped runs report a stale final port-signal level~~ — Resolved (2026-10-05)

DD-282 polls the current port-voltage interval before an energy return and
persists current energy/voltage peaks before periodic checkpoints. Pending
port-envelope state survives checkpoints between checks; old records remain
loadable. Three unaffected GPU fixtures retain identical S-parameters and
step counts while the reported diagnostic agrees with the final recorded
interval. Both field precisions and continuation are covered by regressions.
Evidence: `investigations/termination-accuracy/IMPLEMENTATION.md` (internal
record).

## KB-049: ~~Loaded geometry lacks the 3D `show()` method~~ — Resolved (2026-10-04)

DD-279 provides `LoadedGeometry.show()` through the same 3D scene builder
as live geometry. Both obsolete geometry `plot()` entries are removed in
this breaking naming revision. A loaded geometry scene was built headlessly
and the reader delegation is covered by the API acceptance tests.

## KB-048: ~~An absorbed material plane can send a straight coax TEM port to Mur~~ — Resolved (2026-10-03)

The longitudinal series pass mistook a changing PEC free area for dielectric
contrast at two curved-wall edges. DD-277 separates the two, routes those
edges through the line-solid classifier, and restores the HESR coax port's
exact DTBC certificate. Full diagnosis and measurement are in DD-277.

## KB-047: ~~`integrate_E` silently returns the real part of a complex field~~ — Resolved (2026-10-05)

DD-284 preserves the full complex voltage phasor while retaining the
existing float result and accumulation for real fields. The 7 mm,
`Ex=3+4j` counterexample now returns `0.021+0.028j` V. Uniform/graded
three-axis paths, reversal, closed-loop circulation, frequency snapshots
and real float32/float64 compatibility are covered. Physical FieldState
units remain V/m; raw FIT edge voltages are not the input. Methods and
the voltage-integral how-to document the contract. Initial discovery:
`investigations/oblique-lumped-path/DERIVATION.md`; closure evidence:
`investigations/kb047-complex-integrals/MEASUREMENTS.md` (internal records).

## KB-046: ~~Volume is quadrature-limited on rational B-spline faces~~ — Resolved (2026-10-05)

DD-275 WP4 replaced fixed volume quadrature with span-aware Gauss-Kronrod
integration. DD-285 closes the remaining precision gap by measuring a
centred, uniformly scaled copy of the completed CAD shape. The original
3/6 mm smoothstep taper now agrees with its analytic volume to about
1e-10 relative through public `volume()`, including automatic and explicit
build scales. Increasing/decreasing tapers at three physical sizes and
the polar paraboloid counterexample are covered. Construction and meshing
retain their geometry. Original record:
`investigations/taper-tangency/MEASUREMENTS.md`; closure:
`investigations/kb046-volume-quadrature/MEASUREMENTS.md` (internal records).

## KB-045: ~~The band-DTBC port does not run on the CuPy backend~~ — Resolved (2026-09-03)

Host NumPy mode profiles were multiplied by CuPy field slices, crashing the
first band-port projection. Shared gather_host/array_module_of gathers the
required samples and one fused scatter updates the device port plane; port
arithmetic remains host-side double. CPU length-law results are bit-identical
and the dedicated GPU gate agrees within 1e-6 dB. Default NumPy-pinned tests
had missed this combination; TestBandDTBCOnGPU skips without CUDA and does
not establish broad device-port coverage. This is the same transfer class
as KB-006; the independent KB-038 precision defect remains open.

Historical test/record anchors (private probes belong to the internal
records named above): `_backend/array_api.py`, `_modal/operator.py`, `band_dtbc.py`, `src/magnelio/ports/_modal/band_dtbc.py`, `test_qtem_band_dtbc_sparams.py`, `tests/conftest.py`, `tests/integration/test_gpu_backend.py`.

## KB-044: ~~The in-house section paths book a tenth of the deflection, the kernel path the whole of it~~ — Resolved (DD-243, 2026-09-02)

Kernel/exact sections used the full deflection while facet sections used
one tenth, so route changes altered curved-cell masses. DD-243 applies one
SECTION_CHORD_FRACTION=0.1 budget to all three paths. Against a delta/1000
reference, kernel worst-cell deviations improve from 7.0–9.3e-3 to
0.79–1.15e-3; cylinder route agreement reaches 8e-14/2e-16 of a cell.
The fillet-heavy build costs +19 % (5.41→6.43 s). Refining only the exact
engine was rejected because it moved the same inconsistency to delegated
kernel planes. Context: DD-199/DD-217 and KB-042. Full measured comparisons:
`investigations/kb042-analytic-facets/MEASUREMENTS.md` (internal record).

Historical test/record anchors (private probes belong to the internal
records named above): `_section_kernels.py`.

## KB-043: Fast sections can lose material near a cylinder generatrix — Opt-in mitigation (DD-287)

The Boolean section lost curves before polygon assembly; lifted facet
crossings missed a trim boundary. With the corrected bore-aware reference,
the facet path lost 25.1% / 49.7% at 100 nm / 1 nm, while the kernel returned
zero. The historical alleged 14% loss at 1 um omitted the bore and was not
a defect.

With `MeshControl(robust_sections=True)`, sensitive cuts use conditioned bounded CAD-surface traces, face trims
and shared-edge contour assembly. Holes retain their signed winding;
ordinary fast paths and coplanar interface semantics remain. Final residual,
CAD boundary-tube and closure checks reject unresolved sensitive sections
explicitly. Implemented on `fix/near-tangent-sections`.
The controlled repeated-cell comparison establishes substantial build cost
with very small finite-record S/field changes. The route is opt-in as of
2026-10-08; the default fast route remains affected. Actual wall-loss sensitivity
has not been measured, and sampling-weight diagnostics cannot certify it.

Acceptance: 31 new regressions, 63 independent production section cases,
320 material rectangles, mesh stress, port and S-parameter scale certificates.
Worst absolute coverage/epsilon errors are 2.602e-6 at 100 nm and 1.879e-7
at 1 nm. Full suite: **4405 passed, 39 skipped, no failures**.
Evidence: `investigations/kb043-near-tangency/IMPLEMENTATION.md` (internal
record). Reproduction, corrected references and rejected shortcuts remain
in that internal dossier; the repository certificate is
`validation/surface_section_tangency_certificate.py`.

## KB-042: ~~Cone, sphere and torus faces of a facetted shape keep the KB-041 reach defect~~ — Resolved (DD-242, 2026-09-02)

The suspected remote-body reach was refuted: moving the freeform neighbour
changes sphere/cone/torus answers by zero. Their 7–9e-3 cell difference was
the kernel tessellation budget, subsequently KB-044. The real defect was
an unprojected sphere-pole crossing (2.4e-6 m). DD-242 projects onto the
implicit analytic surface: residual 2e-18 m, pole-cell error 6.3e-4 and
section cost −30 %. The unrelated cylinder invariance defect is KB-041.

## KB-041: ~~A free-form body perturbs the conformal masses of cells that contain none of it~~ — Resolved (DD-240, 2026-09-01)

DD-199's freeform gate selected faceting per shape, including fused Boolean
tools, rather than per face. Analytic cylinder sections then lost axial
translation invariance; remote lofts reached an air body's bore through its
Difference tools. One mass was 42.5 % low and slab defect 8.4165e-2 failed
the 1e-8 port gate. DD-240 keeps analytic faces on their exact path; disabling
faceting on the identical geometry already demonstrated DTBC recovery.
Lost invariance, rather than generally worse approximation, caused the port
fallback. No contribution to the separate precision-floor shift was measured;
KB-038 remains open. Cone/sphere/torus and tangency limits were separated
as KB-042/043, not implicitly closed by this fix.

Related record anchors: KB-043.

## KB-040: ~~The 2D port mode solve is 7–78× slower than its pinned cost~~ — Resolved (DD-239, 2026-09-01)

Thread contention, not a solver regression, caused the apparent 7–78×
slowdown. Pinned CPU-time minima reproduce within 2 %; a loaded-box
wall clock varied 28.6→1801 ms (63×) at constant work. The compared full
mode call also included curl preparation outside the old timer window.
DD-239 closes the diagnosis: future cost certificates need CPU time or an
isolated machine and matching timed scope. The historical microstrip TD pin
overstated cost by at least 2.4×; the shipped modal build was not the costly
CW certificate instrument. No numerical or user-facing performance fix.

## KB-039: ~~The pair-ladder fixture's ports fall back to Mur since DD-199~~ — Resolved (DD-240, 2026-09-01)

The tangent loft triggered KB-041's per-shape faceting, not a bad fixture.
DD-240 restores the unchanged pair-ladder geometry: both DTBC ports certify,
pair spreads 5.2164e-15/1.3072e-13, slab defect 1.4795e-10 below 1e-8,
z_line 96.1625 ohms. Do not amputate the galvanic-feed loft to make a test
pass. The certificate's mass-based first stage, printed second-stage veto,
measured 18–20 s runtime and shipped 2e-6 gate were corrected. It remains
outside pytest coverage, like other DD-anchored certificates. Context DD-199/
DD-165; baseline `investigations/qtem-midpath/baseline/` (internal record).

Historical test/record anchors (private probes belong to the internal
records named above): `DRIFT.md`, `validation/pair_ladder_choice_certificate.py`.

Historical artifact anchors (internal records where applicable): `pair_ladder_choice_certificate.stdout`.

## KB-038: The band-DTBC port floor degrades with the length of the run — Open, cause located (2026-09-03)

Both halves of this entry — the level shift against the pinned floors
and the growth of the floor with the length of the run — are the
**single-precision production default**.  DD-094 made `float32` the
production time-loop precision; every pinned floor in this entry and in
the certificates it is measured against predates it and was taken in
double.  The defect is real and it is on the shipped default; what it
is not is an absorber regression.

**What was measured.**  `validation/qtem_band_dtbc_port_floors.py`
records, for the `layered` fundamental, floors of −159.6…−231.3 dB from
a 12288-step pulsed run with record end/peak 1e-11…1e-13.  The same
script on the same mesh gives **−114.1…−129.9 dB at end/peak 1.6e-06**,
and the microstrip case −146.6…−168.1 dB against a recorded
−171.1…−211.0 dB.  Lengthening the record makes it *worse*, which rules
out the finite-record limitation the header's own methodology note
describes (`layered`, everything else fixed):

    steps    n_kernel    record end/peak    |S11| worst    median
    12288      16384         1.59e-06         −114.1       −126.6
    24576      32768         3.96e-06         −106.4       −118.2
    49152      65536         8.67e-06          −99.6       −111.5

At 49152 steps the case fails its own −100 dB acceptance criterion, and
the degradation is ~7.5 dB per doubling of the run.  Neither the kernel
length nor the DC anchor is the cause: holding `n_kernel = 65536` while
varying only the record reproduces the same numbers to 0.1 dB (12288
steps −114.2 dB against −114.1; 24576 −106.4 against −106.4), so this
is not DD-234's kernel sizing, and building the microstrip ports with
`dc_anchor=False` gives −150.0 dB at end/peak 1.87e-08 against
−146.6 dB at 7.4e-09 with the anchor — the same class, the anchor
slightly better.

**The level shift: re-running the same fixtures at the same HEAD with
`MAGNELIO_PRECISION=double` recovers the pinned class exactly.**

    fixture                                  pin      single    double
    kg WR-90 TE10 at 1.01 f_c              −150.4    −124.5    −150.4
    dtbc_tem parallel plate, max           −138.7    −111.9    −145.6
    dtbc_tem parallel plate, median        −164.0    −133.9    −167.3
    dtbc_tem rect coax, max                −159.3    −140.9    −164.7
    dtbc_tem rect coax, median             −159.4    −162.5    −164.8
    qtem layered fundamental, 1.0 GHz      −244.6    −155.0    −248.5
    qtem block, 4.2 GHz                    −250.2    −173.3    −231.7
    qtem microstrip, 1.0 GHz               −250.8    −164.1    −227.5

— the WR-90 leg reproduces the pin to the printed digit, with a fit
residual of 2.4e-09 in double against 1.1e-07 in single.  The two
conformal round-WG legs did *not* drift (pinned −124…−132, measured
−123.8…−133.5): they are cross-section-limited, not wordlength-limited,
which is the control that keeps the reading honest.

**The length law, same code path, single against double.**  In single —
the production default — the floor loses 4.75 dB (worst) / 6.35 dB
(median) per doubling from 4064 to 8128 steps and 5.71 / 6.45 dB from
8128 to 16256, the same order as the ~7.5 dB per doubling recorded
above at full size.  In double the floor is **flat**: worst −149.12 →
−149.13 dB, median −185.09 → −185.81 dB, i.e. it improves.

**Why CI could never see it.**  `tests/conftest.py` pins the whole
suite to `MAGNELIO_PRECISION=double`, and the band test ran 4064 steps
against a −120 dB gate.  The defect exists only at the production
default, which no test exercised.  It does now:
`tests/integration/test_qtem_band_dtbc_sparams.py::TestBandDTBCLengthLaw`
runs in explicit single and asserts on the **degradation rate** rather
than an absolute floor — median below 8.0 dB per doubling (measured
6.35) and worst below 7.0 (measured 4.75), both one-sided, so a fix
cannot fail the test.

**Where the accumulation sits — answered 2026-09-03** (internal record
`investigations/kb038-wordlength/`).  The probe this entry proposed —
solver in single, the convolution state forced to double — is a no-op:
**the convolution state is already double and always was.**  Every piece
of boundary state is allocated with `dtype=float` irrespective of
`MAGNELIO_PRECISION`, which reaches only the solver's field arrays; read
off a running single-precision solve, `xt`, `w_hist`, `s_hist`, both
kernels and the coupling matrices are float64 while `e_flat`, `h_flat`
and `x1_prev` are float32.  The modal port is built the same way
(`_gather_host` + `np.asarray(..., dtype=float)`).

The only single-precision contact is a per-step round trip through the
field array in `PortOperatorBandDTBC.update_e`: the double
reconstruction `W @ xt` is rounded into a float32 store, and `x1_prev`
reads the first interior period back out of the float32 field.  The
separating experiment therefore runs the other way — solver in
**double**, only that interface quantised to float32.  On the CI band
fixture at 4064 / 8128 steps:

    arm                     4064            8128        rate w / m
    single              -128.72/-136.19  -123.97/-129.85  +4.75/+6.35
    double              -149.12/-185.09  -149.13/-185.81  -0.01/-0.71
    double, read q'd    -130.69/-135.24  -121.90/-127.27  +8.79/+7.97
    double, write q'd   -143.52/-146.07  -131.82/-137.77 +11.70/+8.30
    double, both q'd    -131.28/-140.87  -127.96/-134.32  +3.32/+6.55

(worst/median |S11| in dB; the two unmodified arms reproduce the
`TestBandDTBCLengthLaw` pins to the printed digit.)  Quantising nothing
but the interface reproduces the production length law — +6.55 dB per
doubling against single's +6.35 — and accounts for **84-92 %** of the
gap between the double and single floors.  The volume march carries the
remaining 8-16 %: it is worth 2.6 dB but does not flatten the law.  The
two sides partly cancel, so neither alone is the cause: quantising one
is *worse* than quantising both (+8.79 / +11.70 worst against +3.32),
because rounding one side puts the port's idea of the port plane out of
step with what the field holds, while rounding both is the consistent
state production is already in.

**What stays open** is therefore no longer a port-side wordlength
question — that one is answered, and widening the convolution further
is not available because it is already double.  What the values the
port reads were rounded to was decided by the solver when it marched
them, and cannot be recovered at the read.  Any fix is a *solver*
decision: carrying the port plane and its first interior period in
double while the bulk stays single.  Not priced.

Consequences while it is open: a long band run at the default precision
has a floor that depends on its own length, at roughly five to seven
decibels per doubling of the time steps, and a floor read from any of
these certificates means nothing without the wordlength it was taken
at.  The four certificates were re-pinned at HEAD `73f7c17` with the
previous value kept visible, they now print the resolved time-loop
precision next to their numbers, and the band figure in the user
documentation was re-measured on this tree.  Measurements: internal
record `investigations/port-model-default/` (`probe_record_length.py`,
`probe_prod_floor_cause.py`, `fixture_layered.log`,
`fixture_microstrip.log`); the wider certificate capture is internal
record `investigations/qtem-midpath/baseline/` (per-script stdout and
`DRIFT.md`).

**The length law is not a band-port property — measured 2026-09-03**
(internal record `investigations/precision-docs/`,
`probe_length_law_modal.py`).  The same erosion runs on the *ordinary*
exact TEM DTBC modal port.  Tutorial-01 parallel plate, 32768 cells,
`energy_stop_db=None`, worst / median |S11| over the band:

    steps    single worst  single med   double worst  double med
     2000       -125.76      -139.40       -166.48     -166.50
     4000       -119.98      -139.40       -166.39     -166.50
     8000       -113.02      -139.46       -166.03     -166.51
    16000       -112.60      -139.64       -166.01     -166.50

Single erodes at +5.78 / +6.95 dB per doubling — the same order as this
entry's 4.75-7.5 dB — while double is flat to +0.02…+0.35 dB.  Two
things follow.  First, the mechanism is the wordlength itself, not the
band boundary, so a fix aimed only at `PortOperatorBandDTBC.update_e`
would leave the modal port where it is.  Second, **the modal port
saturates and the band port does not**: from 8000 steps on the modal
floor stops at -112.6 dB, the float32 field floor
(20·log10(2e-6) ~ -114 dB), whereas this entry records -99.6 dB at
49152 steps — past that plateau.  So the band boundary carries an
*additional* mechanism on top of the general erosion, which is
consistent with the interface round trip located above.  Also worth
knowing when reading any of these numbers: the **median does not move**
(-139.40 -> -139.64 dB over eight times the record) — only the worst
frequency point erodes, so a band-averaged figure of merit shows
nothing.  This is now documented for users in
`docs/methods/precision.md`.

**Complete-suite audit, 2026-10-05.** The small fixture now reads median
-145.77/-136.56 dB at 4064/8128 steps: both better than the recorded
-136.19/-129.85 dB, but with a 9.21 dB difference. Historical source
`19ac93d3` on the current environment reproduces that difference, so it
does not establish a new code regression. The regression test retains the
original doubled-run ceilings, anchored to the recorded baseline, and adds
a short-run ceiling; cancellation lowering the short-run floor alone no
longer fails it. This does not fix the interface-wordlength defect or
establish its floor for arbitrary run lengths. Evidence:
`investigations/test-health-2026-10-05/MEASUREMENTS.md` (internal record).

## KB-037: ~~Two builds of the same band port gave different Galerkin subspaces~~ — Resolved (2026-08-31)

Unseeded zeta-pencil eigs rebuilt a different Galerkin basis: mostly sign
gauge, but about 1e-5 real numerical variation underneath. Basis-invariant
results concealed a checkpoint/resume incompatibility. Shared arpack_v0
now seeds the pencil, numerical_2d and spectral solves; repeated builds are
bit-identical in/across processes. Sign normalization alone was insufficient.
Same root cause as KB-010/DD-142. Evidence:
`investigations/port-model-default/` (internal dossier).

Historical test/record anchors (private probes belong to the internal
records named above): `MEASUREMENTS.md`, `numerical_2d.py`, `probe_band_reproducibility.py`.

## KB-036: ~~Faces in a conductor's end wall blocked and the wall unbooked on grids below about 15 µm~~ — Resolved (DD-207, 2026-08-28)

The two-sided DD-106 coplanar sampling step used deflection, falling below
CAD tolerances on fine grids (60 nm at 6 micrometres versus 1e-7 confusion/
1.5e-7 Boolean tolerances). Sections leaked pockets past conductor end walls,
corrupting both wall jumps and epsilon averages. DD-207 uses the larger of
deflection and four times the model's largest BREP tolerance; the planar
screen then answers outside the tolerance tube. The threshold is kernel/
geometry dependent, not a universal 15-micrometre physical-cell rule.
Evidence: `investigations/mesh-build-bench/MEASUREMENTS.md` (M10, internal record).

Historical test/record anchors (private probes belong to the internal
records named above): `tests/unit/test_section_slab_index.py`.

## KB-035: ~~Far-field power deficit of about a tenth with a window port in an absorbing face~~ — Resolved (DD-204, 2026-08-27)

The deficit was misattributed to window ports: lumped feeds showed it too,
while Huygens flux matched accepted power. At 0.3 wavelength top clearance
the discrete near field is unsuitable for a free-space transform: radiation
power was about 7 % low, halved cells about 3 %; realized gain was low but
directivity remained correct. At 0.7 wavelength clearance the deficit vanished.
DD-204 adds surface_power/power_balance and a >5 % closure warning; the
how-to uses 0.7 wavelength. Window launches still have documented currents
outside the box (surface/accepted power 0.965), which that balance cannot see.
Evidence: `investigations/patch-array/MEASUREMENTS.md` (M18, internal record).

Historical test/record anchors (private probes belong to the internal
records named above): `tests/unit/test_far_field_closure.py`.

## KB-034: ~~Thin sheets and wires touching an absorbing face had no mask in the PML~~ — Resolved (DD-198 amendment, 2026-08-27)

The DD-198 absorber continuation preceded thin-sheet/wire masking, so
their later PEC masks stopped at the physical wall and window ports saw the
wrong cross-section. Repeat mask-only continuation after the sheet pass;
thin-sheet/window-line regressions cover it. Thin-wire Holland material
correction still is not extended: a wire ending on an absorbing face is
not a supported feed. This limitation is not closed by the mask repair.

Historical test/record anchors (private probes belong to the internal
records named above): `tests/unit/test_pml_extension.py`.

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

One-ULP differences between nominally coincident sheet heights produced
spurious forced-plane and extreme-growth warnings (5e-20 m separation,
1e14 ratio). The downstream grid removed the sliver but not its diagnostics.
DD-201 clusters sheet anchors within the feature gap and updates masks to
the shared plane; a nearby user-forced plane wins, otherwise the lowest
sheet. A coincident long run was separate closed-housing ringing, not this
anchor defect. Thin-sheet anchor-unification regression covers the fix.

Historical test/record anchors (private probes belong to the internal
records named above): `tests/unit/test_thin_sheet_detection.py`.

## KB-031: ~~Hollow conductors lost the conformal correction at their inner walls~~ — Resolved (DD-199, 2026-08-26)

Kernel section contours had no signed nesting convention. Summing each as
positive filled conductor and dielectric bores, degrading inner-wall sub-cell
data. DD-199 winds by nesting parity; mean tube area error falls 0.12→4e-3.
Material priority had hidden the defect in coax cases whose holes contained
conductors, but not in ceramic air bores: correcting the ring changed
2.3279→2.6566 GHz on the same grid (solid puck 2.2302 GHz). DD-102's
earlier harmless-orientation verdict was wrong; KB-011, the DD-191 chamfer
certificate and Tutorial 13 were re-based on 2026-08-27.

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

One regression and one legitimate re-pin were separated. DD-191 admitted
four-face edges where touching cylindrical surfaces continued on both sides,
adding an axis plane and worsening TM010 error 3.7→6.0 %; the amended edge
skip restores the old grid and fixes a latent gp_Ax1 distance call. DD-193's
equal-fill grading instead improves conformal coax impedance 48.12→48.94
ohms toward analytic 49.97 and port floor −131→−135.6 dB; that test was
re-pinned, not forced back to a worse grid. Context DD-192/DD-196.

Historical test/record anchors (private probes belong to the internal
records named above): `test_conformal_coax_sparams.py`, `test_conformal_convergence.py`.

## KB-027: ~~De-embedding a quasi-TEM feed leaves the line's physical dispersion behind~~ — Resolved (DD-244, 2026-09-02)

Quasi-static continuum gamma could not remove a quasi-TEM line's physical
dispersion; de-embedding attributed the residual to the device. DD-244 saves
the port's restricted curl/feed/profiles and solves true discrete modes on
the result axis, also used by dispersion reports. On the 20 mm microstrip
at 25 nodes/wavelength, residual phase becomes +0.3/+1.5/+1.8 degrees at
5/10/15 GHz versus −1.5/−10.6/−29.5; DD-239's launch residue remains.
The original refinement/dielectric/thickness controls established a physical
dispersion mechanism, not a mesh error. Original evidence:
`investigations/port-deembedding/` (internal dossier).

## KB-026: ~~An empty boolean result crashes plot() with a C++ abort~~ — Resolved (2026-08-25)

An empty Boolean reached OCC tessellation at zero deflection and aborted
Python across the C++ boundary. DD-190's viewer checks bounding-box extent,
skips empty shapes with a named warning and prevents that abort. This does
not establish eager volume validation at model.add: the separate downstream
mesher array error and desired early empty-operand error remain outside this
resolution. Check Boolean operands when a model unexpectedly loses geometry;
DD-176 supplies the related argument-validation principle.

## KB-025: ~~A cross-section paints its holes shut~~ — Resolved (2026-08-20)

Drawing each contour as an independent fill painted holes shut and made
coax images insertion-order dependent. post/plot_geometry builds one compound
path with nesting-based opposite winding; the nonzero fill rule then realizes
the even-odd material region, preserving free islands inside holes. Air
outlines remain separate contours. CLOSEPOLY must explicitly repeat the first
point: Path(closed=True) otherwise consumes a real last vertex and turns a
rectangle into a triangle. Raster tests check annuli and rectangular
two-level nesting, since inspecting compound-path structure alone misses it.

Historical test/record anchors (private probes belong to the internal
records named above): `post/plot_geometry.py`, `tests/unit/test_plot_geometry.py`.

## KB-024: ~~A missing pythonocc-core reads as an empty mesh, not as a missing dependency~~ — Resolved (2026-08-19)

Broad exception guards swallowed missing pythonocc ImportError and later
reported invalid grid arrays instead of installation guidance. Re-raise
ImportError before generic guards at bounding-box/face feature extraction
and feature-gap analytic-box queries; exotic-shape failures still skip.
The README WR-90 missing-OCC probe now names the dependency rather than
GridLines. TestMissingOccSurfaces covers all three guards; available-OCC
behavior is unchanged.

Historical test/record anchors (private probes belong to the internal
records named above): `tests/unit/test_geometry.py`.

## KB-023: ~~CPML min and max faces are not mirror images~~ — Resolved (2026-10-05)

DD-286 samples electric updates at nodes and magnetic updates at cell
centres, using physical depth from the interface on graded grids. All
three axes and both precisions pass independent mirror tests. The resonant
full/half dipole max S11 difference improves from 0.02149 to 3.04e-5;
its gate tightens from 5e-2 to 1e-4. The vacuum pulse certificate covers
opposing faces and 8/16/24 cells in all three directions. Existing
checkpoints retain their former profiles and resume bit-exactly against
the original source; start a new run to use the corrected profiles.
Evidence: `investigations/kb023-staggered-cpml/MEASUREMENTS.md` (internal record).

## KB-022: ~~Pair coupling accepts ladder candidates 100x looser than the transparent-boundary gate~~ — Resolved (DD-228, 2026-08-30)

Pairing at rtol=1e-6 can agree without meeting the 1e-8 DTBC gate. DD-165
improved conditioning but did not close that band. Tightening pairing was
refuted: rejecting 1008/24295 targets substituted worse Krietenstein line
partners and increased both spreads. DD-228 instead publishes provenance,
termination/chain_spread and warns for withheld certificates in the marginal
1e-8–1e-4 band. Truly inhomogeneous models are not mislabeled defects.
Closure removes silent fallback, not estimator uncertainty: jittered ladders
can still fail. eps_avg/f_A were consistent to 3.9e-15 across 19244 edges;
the old inconsistent-integral diagnosis in KB-017 was wrong.

## KB-021: ~~Half a solid's cross-section goes missing with no warning~~ — Resolved (DD-168, 2026-08-15)

OCC produced all section edges, but MakeWire admitted a branch and
WireExplorer visited only one arm: fourteen edges, eight added, one visited,
thirty metal cells silently filled as air. DD-168 replaces assembly with an
endpoint graph and tangent-continuity branch choice. Nudging/tessellating
cannot fix this topology loss; count kernel edges versus consumed edges to
distinguish it from the wrongly suspected DD-167 grazing-section failure.

## KB-020: ~~A near-tangent section plane drops a solid's whole cross-section on a fine mesh~~ — Resolved (DD-167, 2026-08-15)

DD-157's recovery reach was tied to tessellation deflection, so the
ten-times-finer conformal pass could not leave near-tangency bands that cell
classification escaped. DD-167 gives both passes one independent recovery
length and warnings naming body, lost amount and consequence. Refinement
can move a neighboring section closer to an anchored extreme, not make
grazing geometry harmless. The separate lost-branch case is KB-021.

## KB-019: ~~The classifier never produces sub-cell data on a domain boundary face~~ — Resolved (2026-08-15)

The candidate mask excluded transverse boundary indices, rounding partially
filled domain-face edges to full free/metal. DD-164 includes those indices
and clamps dual faces to the physical boundary. The decisive magnetic-half
identity improves −2.3e-3→4.7e-15 where dielectric meets the symmetry plane;
the pillbox test was blind because its field vanished there. A moving band
floor was a separate kernel-fit sensitivity, not evidence against the fix.

Historical test/record anchors (private probes belong to the internal
records named above): `geo/_filling.py`.

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

DD-165 chooses the better-conditioned agreeing ladder rather than the
first axis: mirrored-coupler pair spread improves 1.7e-8→6.3e-14 and DTBC
returns. The blamed eps_avg/f_A mismatch was refuted (3.9e-15 consistency
over 19244 edges). The 1e-6 agreement versus 1e-8 certificate band still
exists; conditioning is optimal choice, not a guarantee that two jittered
ladders cannot fail. The remaining silent-fallback problem became KB-022.

## KB-016: ~~Frozen zero-M_eps edges seed NaN Mur coefficients on live complement-absorber edges~~ — Resolved (2026-08-14)

Zero-M_eps edges were frozen by volume stepping but remained live in a
Mur complement absorber that consulted only the PEC mask, producing infinite
speed/NaN coefficients. Add M_eps<=0 to the absorber's dead set with finite
coefficients; suppress equivalent already-discarded 0/0 chi census terms.
The four curved-cut Ey edges were latent while their port used exact DTBC.
The complement-absorber frozen-edge regression covers the Mur case.

Historical test/record anchors (private probes belong to the internal
records named above): `tests/unit/test_port_edge_bc.py`.

## KB-015: ~~Open section chains book fantasy coverage — coax ports fall back to Mur under declared symmetry~~ — Resolved (DD-157, 2026-08-14)

Near-tangent Boolean sections returned open edge chains that polygon
consumers closed implicitly, booking fantasy coverage (0.80 versus 0.19
free area, slab defect 0.43) and demoting symmetric coax ports to Mur.
DD-157 requires closed chains, retries with a nudge and loudly drops unresolved
chains. The uncut full model exposed it when DD-154 replaced manual quarter
cuts. Section-open-chain certificate and face-section regressions cover it.

Historical test/record anchors (private probes belong to the internal
records named above): `tests/unit/test_geometry.py`, `validation/section_open_chain_guard_certificate.py`.

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

PeriodChain assumed one normal flat stride for both E tangential families,
valid on z faces but not x/y faces whose arrays differ. et_step is now a
per-edge offset when necessary; elementwise period shifts retain the block,
certificate and Bloch synthesis algorithms. Axis-permuted x/y fixtures
reproduce the z-normal fundamental eigenpair to 1e-9. Found during the
DD-123/124/125 Wilkinson groundwork, not a physical orientation restriction.

Historical test/record anchors (private probes belong to the internal
records named above): `test_zeta_pencil.py`.

## KB-006: ~~MonitorWallLoss crashes on the cupy backend~~ — Resolved (2026-08-10)

MonitorWallLoss used NumPy coercion/indices on CuPy fields and crashed at
the first record. Gather wall samples on-device with cached device indices,
transfer only surface sample vectors and reference-plane slabs, and retain
host DFT accumulation. CPU arithmetic is unchanged; the DD-082 plate GPU
loss fraction matches CPU to 1e-12. Related monitor transfer work: DD-115;
automatic device selection: DD-090. Coverage remains the dedicated gated case.

Historical test/record anchors (private probes belong to the internal
records named above): `test_gpu_backend.py`.

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
