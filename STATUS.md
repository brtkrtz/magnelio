# Magnelio — Project Status

*Last updated: 2026-10-06.*  **Released v0.8.2** (2026-09-09): **DD-270**
the Poynting vector as a derived view of any recorded field (`poynting()`,
`"S"` in the component vocabulary), **DD-271** three more readings of an
S-matrix (`plot_balance` / `plot_smith` / `plot_polar`), **DD-272**
`result.extrapolate()`, which continues a truncated record by its own poles,
and **DD-273** the surface current `J_s = n × H`, read out of the wall-loss
booking (`Mesh.pec_surface` removed with it — never read by anything).
**Released v0.8.1** (2026-09-08): **DD-269** — a discrete port or lumped element follows an arbitrary path
(oblique, bent, or a curve) instead of an axis-parallel pair of
terminals, through the same canonical rasteriser thin wires already use;
its polarity now follows the declared direction (a behaviour change
judged to reach no user, hence a patch).  **KB-047** opened:
`integrate_E` drops the imaginary part of a complex frame.
**Released v0.8.0** (2026-09-08; a minor under the Cargo reading — every
complex frequency-domain field is the conjugate of what 0.7 returned,
`docs/migration-0.8.md`): **DD-268** one phasor convention for the library,
**DD-267** the phase of a picture is ωt, **DD-262…266** the ParaView export
on request and the 3D viewer after the review.
**Released v0.7.0** (2026-09-06; the field monitors' dictionary API is
gone, `docs/migration-0.7.md`): **DD-259/260/261** — monitors keep the
grid quantities and derive every view at access time (store schema 3.0),
energy and flux are identities on a recording, fields in the volume of
the 3D viewer; **DD-257** (CPU kernels 1.2× faster, bit-identical);
**DD-256** (a thin wire on a thin sheet).  Before it: **v0.6.0**
(2026-09-05, `Project.runs` hands out `Run` objects, DD-249…255) and **v0.5.0–v0.5.2** (2026-09-02…04, the API grammar, DD-224…248) — migration guides
`docs/migration-0.5.md` … `docs/migration-0.8.md`.

Open: KB-038 and KB-043 on the default fast route; opt-in DD-287 mitigates KB-043 and resolves KB-053 on `fix/near-tangent-sections`; KB-023/DD-286 and KB-052 are committed on the preceding `fix/cpml-staggered-profiles` branch (CPML commit `117c810f`). Latest full-suite acceptance: **4446 passed / 0 failed / 13 skipped**, 4459 collected, 1150.80 s, 132 warnings (2026-10-06), including the near-tangent section fix and existing integral, CAD, CPML and watch checks. Suite defaults pin NumPy/double; explicit single/GPU tests override them. Skipped paths are not verified. KB-047/DD-284 and KB-046/DD-285 are committed and backed up to private as `80eff88c` and `9fe0bdec`. Evidence: `investigations/near-tangent-meshing-performance/hesr/final-repair-tests.log` (internal record).
All seven initial failures are addressed: scattering-only S access again raises the correct ValueError without opening unrelated records; cap/resume and incident-field fixtures follow resolved pulse timing; the dipole radiation gate uses negligible net current to avoid a static-tail DFT boundary term (independent tail completion agrees to 2e-8); the band gate preserves its original doubled-record ceilings and adds a short-record bound rather than rejecting an improved floor through its ratio. That test-health fix left production field/source/monitor arithmetic unchanged; DD-286 subsequently corrects CPML sampling. KB-038 remains open. Fresh offline HTML docs (gallery execution disabled), Ruff, format, hygiene, DD and API gates pass. Evidence: `investigations/test-health-2026-10-05/MEASUREMENTS.md`, `DERIVATION.md` and `final-pytest.log` (internal records).
The foundation was merged to local `main` on 2026-10-02. WP0-WP5 and WP6.1/WP6.3-WP6.12 are implemented.
WP6.12 imports free CAD faces as Sheets without duplicating solid boundaries and exports selected solids/sheets through STEP/BREP with explicit unit rules. The 1-nm solid exchange gate stays valid. Tutorial 14, "Geometry toolbox", also absorbs the former CAD/PCB import tutorials; fresh offline Sphinx passes. The wire adapter supports pythonocc 7.9 and the 8.0.1 edge-return binding.
The feature branch is backed up to `private` through `926c1d91`; the foundation merge was pushed to `origin/main` on 2026-10-02. WP6.2 is cancelled. The foundation acceptance audit is recorded in `investigations/geo-api-foundation/FINAL-ACCEPTANCE.md` (internal record).
Pulse defaults are implemented in DD-281. DD-282 resolves KB-050/051: finite-drive completion guards, spatial plane-wave retardation and current decay diagnostics/checkpoint peaks. Four GPU acceptance runs pass; the narrow-band line error improves 36-fold. Growing spectral windows still falsely accept a passive narrow resonance, so no automatic S-accuracy claim is introduced. Evidence: `investigations/termination-accuracy/IMPLEMENTATION.md` and `SPECTRAL_STRATEGY.md` (internal records). Channels: GitHub, PyPI, conda-forge and the two docs channels below.
This file states what *is*.  Chronology: `git log --first-parent main`; reasoning: `design-decisions.md`; open bugs: `known-bugs.md`.  Measured floors regenerate from the `validation/` certificates their DDs name.
## Recent decisions
Newest first, one line each; the full record is the DD entry.
* **DD-287** (2026-10-06, `fix/near-tangent-sections`) — opt-in `MeshControl(robust_sections=True)` mitigates KB-043. Stationary-coordinate selection keeps regular fast paths; face trims and shared CAD edges retain thin material islands, holes and tangent contacts. Numerical residual and CAD boundary-consistency checks remain separate; unresolved sensitive cuts fail explicitly. All 46 section regressions, 63 independent production cuts and 320 material rectangles pass; the original stress and port certificates remain recorded. The unchanged 51-cell model now meshes, with all 361 recorded failures resolved. Methods and mesh-convergence how-to document the contract. Evidence: `investigations/kb043-near-tangency/IMPLEMENTATION.md` (internal record).
* **DD-286** (2026-10-05, commit `117c810f`) — staggered CPML profiles resolve KB-023. Full/half dipole max dS11 improves from 0.02149 to 3.04e-5; independent node sampling and mirror tests cover all axes, graded grids and both precisions. Pulse reflection certificates pass 8/16/24 cells in all directions. Actual old HDF5 checkpoints resume bit-exactly in single/double; missing sampling markers preserve legacy profiles. Tutorial 08 executes in fresh offline HTML. Full-suite acceptance: 4374 passed, 39 skipped, no failures. Evidence: `investigations/kb023-staggered-cpml/MEASUREMENTS.md` (internal record).
* **DD-284/DD-285** (2026-10-05; DD-284/DD-285 committed/private) — electric-field line integrals preserve full complex phase and the existing float result/accumulation for real fields; KB-047 closed. Uniform/graded three-axis conservative fields, reversal, closed-loop circulation, pure imaginary and zero-imaginary complex fields and frequency snapshots pass. Methods, the voltage-integral how-to and its executed snippet document physical units, normalization and path dependence. Fresh offline HTML, Ruff, format, hygiene, DD and API gates pass. DD-285 (`9fe0bdec`, backed up to private) closes KB-046: centred, scaled-copy CAD integration passes the public smoothstep-volume gate across radii, physical sizes and build scales; the polar-dish counterexample retains precision. Geometry runs: 491 and 590 passed (overlapping selections); fresh HTML executes Tutorial 14; all repository gates pass. The subsequent full solver suite passes with the CAD fix included. Evidence: `investigations/kb047-complex-integrals/MEASUREMENTS.md` and `investigations/kb046-volume-quadrature/MEASUREMENTS.md` (internal records).
* **DD-283** (2026-10-05, merged to main, unreleased) — stored-result metadata and time plots avoid S-matrix derivation; S/dB/phase evaluate selected runs and ports with reusable spectra, including custom axes. Coupled band projections remain intact; self-contained band records avoid mesh-operator reconstruction. Incident normalization and reference impedance use only required calibration/records. Index changes after completion and refresh invalidate derived caches. Shared DFT blocks cap kernel temporaries at 16 MiB and taper within blocks. HESR: first three-port time plot 0.44 s, selected S 0.55 s, cached S 0.00011 s, fresh full matrix 2.67 s versus 112.44 s baseline; maximum complex difference 1.73e-14. Validation: 217 targeted tests passed / 1 optional-tool skip; two pre-existing cap-warning failures reproduced on unchanged main and excluded. Methods and Tutorial 07 pass the executed Sphinx build; Ruff, format, hygiene, DD-reference and API gates pass. Evidence: `investigations/time-signal-first-call/MEASUREMENTS.md` (internal record).
* **DD-281/DD-282** (2026-10-05, merged to main, unreleased) — Gaussian upper-edge attenuation is configurable (default 25 dB), with a 4.5-tau peak and cached full-spectrum width. Resolved timing persists; legacy automatic and explicit pulses resume with their original samples. Finite-drive guards and current decay diagnostics resolve KB-050/051. Full Gaussian unit suite: 3808 passed / 5 skipped; 36 focused propagation/state and resume checks passed. Termination acceptance: focused suites of 50, 112 and 81 passed, four GPU comparisons and executed offline documentation passed. Methods and Tutorial 03 document the pulse. Evidence: `investigations/pulse-defaults/IMPLEMENTATION.md` and `investigations/termination-accuracy/IMPLEMENTATION.md` (internal records).
* **DD-280** (2026-10-05, merged to main, unreleased) — selected port-mode solves and bare-string excitations; dB/linear/phase S plots and calibrated a/b time plots; boundary overlays hidden by default, numeric cut input and mesh/layer steps, wrapping toolbars, searchable grouped solid visibility and configurable arrow length/density/thickness/colour/threshold and per-component numeric colour limits with automatic reset. Optional display-group metadata preserves project compatibility; solver physics is unchanged. Field-mirroring report deferred without a reproducer. Validation: 3832 passed / 5 skipped across the unit suite and selected integrations (full run plus corrected/final focused reruns); all 252 final focused tests pass; the colour-limit follow-up passes 146 viewer/API tests. Tutorials 02/03/04/10 pass the executed CPU Sphinx build with warnings treated as errors; browser checks cover menus/search/colour/length/cut steps and narrow-window wrapping. Ruff, format and public hygiene pass. Evidence: investigations/viewer-result-controls/MEASUREMENTS.md (internal record).
* **DD-279** (2026-10-04, merged to main, unreleased) — all 25 API naming decisions are implemented: physical Hz boundaries, explicit port/material coordinates, normalization and run scope, selectors, immutable verbs, public type homes, frequency/phase spelling and estimate_memory. LoadedGeometry.show closes KB-049; affected private scripts/notebook sources are migrated with developer authorization. Stored recipe keys and checkpoint physics remain unchanged; an original checkpoint resumes bit-identically. Tutorials 09/18 corrected after the executed CI gallery exposed two missed consumer spellings; both affected tutorials pass executed Sphinx-gallery checks (2026-10-05). Migration: docs/migration-api-naming.md. Acceptance: investigations/api-naming-review-2026-10-04/IMPLEMENTATION-ACCEPTANCE.md (internal record).
* **DD-277** (2026-10-03) — The absorbed-plane series pass now separates free-area loss to a curved PEC wall from intrinsic dielectric contrast and sends newly detected PEC edges through the line-solid classifier. On the HESR three-cell mesh, the affected interior air edges return to ε̄=1 and the x1 TEM port's pair spread drops from 1.763e-3 to 1.877e-14, restoring DTBC (KB-048).
* **DD-276** (2026-10-02) — PEC tangential-edge re-masking now requires a masked detour around an adjacent grid face; a remote inductive-loop bond no longer closes a local feed air gap. The reduced-model mesh and round-coax port gate pass; full TD response is unmeasured.
* **DD-275** (2026-09-28, merged to local `main` 2026-10-02, WP0-WP5 and WP6.1/WP6.3-WP6.12) — dimensional geometry, affine values, exact curves/Profile holes, owned topology and named histories are implemented. Path carries world poses and relative bends. Sweeps retain frame/twist/draft laws. Partition returns independent Solid/Sheet regions; Section returns exact curves or filled profiles. Directed Imprint splits receiver faces without volume change. Insert assigns material overlap by explicit priority and removes void tools. Bounded Curve projection supports parallel, perspective and closest-point policies. Offsets provide signed planar Curve placement, material-region Profile clearance/erosion and bounded Sheet normal placement. Bend maps existing geometry onto a freeform neutral surface with sampled strain/validity gates and smooth CAD faces. Wrap maps entire layered geometry onto a declared patch with bounded strain; planar tangent lofts supply G1 ends for matching extruded walls. Reusable placements preserve nested component members, materials and owned names; nonuniform scale/shear are rejected. Exact rectangular owned faces define eligible waveguide-port windows and field-frequency planes. CAD exchange preserves solid/sheet categories with explicit STEP/BREP units and documented metadata limits. Methods/API prose and Tutorial 14 cover these contracts. The foundation acceptance audit is complete; WP6.2 is cancelled. Record: `investigations/geo-api-foundation/FINAL-ACCEPTANCE.md` (internal record).
* **DD-274** (2026-09-23) — viewer destination is independent of its rendering backend.  A plain script now serves the complete Magnelio toolbar through trame and opens it in the system browser; `target="native"` retains the PyVista/VTK window.  `target="inline"` keeps the Jupyter widget.  Zed's `kernel-zed-*.json` identity selects the browser automatically despite its ipykernel; `plots.configure_viewer(target="browser")` is the once-per-kernel switch for other editor REPLs that do not render the asynchronously filled `VBox`.  The browser server binds loopback on a daemon thread and is reused; missing trame warns and falls back to native.  `mode` remains `client`/`server`/`trame`/`static`/`none`, with `none` still returning the plotter.
  CI compatibility: PyVista 0.49 stores viewers in `trame_pyvista.ui`; OCC 8 requires an explicit trace-clearance check beyond offset contour counting (DD-135).
* **DD-273** (2026-09-09, branch `feat/surface-current`) — the surface current is the wall-loss booking, read as a vector.  `surface_current(mesh)` on any recording, spectrum or `FieldState` gives `J_s = n × H` per wall patch; `enumerate_wall_patches` reads the [[DD-087]] enumeration the other way round (per patch: conformal area, outward normal `−w/‖w‖`, and *which* sample contributions it booked).  Magnitude from the loss booking (`|J_s|²A = Σw|H|²`, so `power_loss` **is** `MonitorWallLoss` — measured 0.9998 on a real run), direction from `n × H` with the weight-averaged samples.  Accuracy is inherited and does not refine away: coax inner conductor 1.3 %, shield 5.0 % against `√(P/Z₀)`, because the DD-098 pullback is calibrated on the quadratic loss while the current is linear in H; shipped deliberately, since the distribution is what a current picture is read for.  Four traps, all measured: grid quantities vs. physical fields ([[DD-085]], factor 1500), wall cells of a curved conductor are PEC-classified (dropping them loses 83 % of a mantle), a port plane holds the feed's *cross-section* and not a wall (+9.5 %, hence `exclude_faces=`), and a viewer display group must be registered in `_GROUPS` or its actor is built, filled and never shown.  `PECSurfaceData` removed: built for every conformal mesh, stored in every project, read by nothing, and it booked the *normal* H component.  Record `investigations/surface-current/` (internal dossier).
* **DD-272** (2026-09-09, branch `feat/extrapolate-decay`, merged `375578e`) — a truncated record is continued by its own poles, never automatically.  `ScatteringTDResult.extrapolate()` fits the free decay past the excitation with a matrix pencil (all channels of one excitation share one pole set, poles outside the unit circle dropped), continues every recorded V and I until the model has died away, and recomputes the S-matrix through the unchanged pipeline — so the discrete de-stagger, the band decomposition and the reference impedances need no second copy of the DFT convention.  The quality number is **out-of-sample** (fitted on the first half of the window, measured against the recorded second half), because a Prony model reproduces its own fit data almost by construction; above 0.3 the call warns.  Measured on an iris-coupled cavity, lossless with one port so `|S11| = 1` is exact and needs no reference run: a 30 ns march reports a unitarity defect of 0.69, its continuation 0.0048 (**144×**); at 9 ns the same fixture gives 50×.  Where the record is not a free decay of a few resonances the method says so and changes nothing: on a shorted line (a delay system) the residual comes out at 1.00.  A second lesson from the same fixture: the raw defect **grows** with march length (0.15 → 0.49 → 0.69), because a short run has not filled the cavity yet — truncation error is not monotone in run length.  Record `investigations/extrapolate-decay/` (internal dossier).
* **DD-271** (2026-09-09, branch `feat/sparameter-plots`, merged `c5e337a`) — three more readings of an S-matrix on `SDerivedAccessors`, so every result that answers `plot_s` answers these.  `plot_balance` sums |S|² over the observed channels per excitation (one on a lossless, fully exported network; `deficit=True` plots `10·log10(1 − Σ)`, which is an antenna's radiated power and a closed structure's convergence check — and the readable form wherever the balance sits near one); `plot_smith` draws the reflection channels on an in-house chart (no scikit-rf dependency for a standard picture) and **warns** when a channel's reference impedance moves with frequency, where the circles hold at no one normalisation ([[DD-244]]); `plot_polar` the same trajectory without the impedance grid.  An evanescent channel (NaN below cut-on) counts as zero — it carries no active power; what the sum cannot see is a channel missing from the result altogether, so the docs point at `is_complete`.
* **DD-270** (2026-09-09, branch `feat/poynting-vector`, merged `0ea515e`) — the Poynting vector as a derived view, in the spirit of [[DD-259]]: monitors keep storing the grid quantities, `poynting()` on `FieldState`/`FieldRecording`/`FieldSpectrum` forms `E × H` on the cell centres at access time, and `"S"`/`"Sx"` join the component vocabulary so the 2D pictures and the 3D viewer carry it without a new monitor, format or schema bump.  The dtype decides the reading (a real frame instantaneous, a complex one the time average `Re(E × H*)`), a power density has no instant so the phase of a picture must not act on it (two paths were corrected for that), and a frame that recorded only one field refuses rather than reading the unrecorded half as zeros.  Its parity across a symmetry plane is the product of the two field parities — normal odd, tangential even, on either kind of wall — gated against the `E`/`H` rules rather than restated.  Measured: the integral over a cross-section matches the `flux()` identity to **1e-8** once the boundary patches are booked, and is **13.3 % low** without — the same at every frequency, because a PMC wall sits half an outer cell beyond the last grid line.  Record `investigations/poynting-vector/` (internal dossier).
* **DD-269** (2026-09-08, branch `feat/oblique-lumped-paths`, released as v0.8.1) — a lumped port or element is a **path**.  `PortLumped` and `circuit.LumpedElement` take `path=` (points or a `Curve`); `start`/`end` stay as the two-point short form.  The restriction lived in one place, the degenerate two-point rasteriser [[DD-075]] and [[DD-079]] had both marked for subsumption; it is gone, so DD-076's "one canonical rasteriser" is finally true, and the runtime operator needed no change (DD-079 built it EdgePath-shaped for this).  A point path is densified here rather than routed through `Curve.polyline` — OCC-free, and a bare polyline would otherwise rasterise an oblique segment into an **L**, not a staircase.  Three deliberate behaviours: polarity follows `start` → `end` (the old resolution sorted the terminals, so a port declared backwards silently had its twin's polarity — single-port Z and S11 unaffected, a multi-port S-matrix picks up 180°); an edge may be traversed once (a two-terminal element is a series chain); PEC-shorted chain edges warn instead of silently shorting the device.  DD-172 symmetry lifted from a two-point chain to a polyline; a `Curve` reaching a plane is refused with the point form as the way out.  **Measured**, on area-matched congruent lattice loops so both orientations share one asserted-identical grid: the staircase adds `dL' = 58.17 nH/m · x^0.614` per unit chord (`x` = staircase/chord − 1), ten points over three families, ~32 pH/mm for a strongly oblique element.  It **does not refine away** (`∝ Δ^0.19`) and is **not proportional to the extra path length**.  Correction deferred, with reason: the coefficient differs 21 % between plain edges (~32 nH/m) and a PEC thin wire (~38 nH/m), and without it oblique lumped elements sit at exactly the accuracy `ThinWire` already ships at.  Certificate `validation/oblique_lumped_staircase_certificate.py` (3.893 %); record `investigations/oblique-lumped-path/` (internal dossier).
* **DD-268** (2026-09-07, branch `feat/phasor-convention-0.8`, for v0.8.0) — one phasor convention for the whole library.  Follows the developer's question after DD-267: *which* convention do the fields use?  The answer was "two".  The S-parameter path, `Signal1D.at_frequencies`, `Waveform.spectrum`, `cw_lockin_phasors` and the modal half-step `e^{+jω dt/2}` are all `e^{+jωt}`; only the monitors' running DFT summed `Σ F(t) e^{+jωt} dt` and handed out the conjugates, which is why DD-173's far-field transform had to conjugate in and out and why DD-183's transit phase for a beam toward `+z` read `e^{-jk_B z}`.  The accumulator (`monitors/_dft.py`, and the private copy in `wall_loss.py`) now sums `e^{-jωt}`; the three phasor evaluators go back to `Re(F e^{+jφ})` (DD-267's *meaning* — phase is ωt — is untouched); `post/far_field.py` drops its four conjugations and applies Balanis verbatim.  **Store:** schema stays 3.0 — a bump would refuse 0.7 stores whose data converts exactly.  `fields_freq.h5`, `wall_loss.h5` and `far_field.h5` carry `phasor_convention = "exp(+jwt)"`; absent means the older era and the bins (and the far field's stored divisor) are conjugated at the single read site of each file, an unknown value raises.  Exact in both directions, so a 0.7 run resumes bit-identically — verified by removing the hook, which fails all five legacy gates.  Invariant and unchanged: magnitudes, `Re Σ e·h*`, `|·|²`, gain, directivity, S-parameters.  Re-signed with the convention: the stripline how-to's `beam_voltage` and the Hertzian validation script (`arg(E_θ/j)` 180° → 0°).  Eight new gates (two on the kernel, one on the NTFF phase, five on legacy stores).
* **DD-267** (2026-09-07, patch after v0.7.0; **sign amended by DD-268**) — the phase of a picture is ωt.  Three findings of the developer's third viewer pass.  (1) "With the phase animated the wave runs backwards, toward the exciting port": the running DFT sums `Σ F(t) e^{+jωt} dt`, so a bin of `A·cos(ωt+φ)` is `(T/2)·A·e^{-jφ}` — a phasor of the **`e^{-jωt}` convention**, whose instant is `Re(F e^{-jωt})`.  Every picture applied `Re(F e^{+jφ})`, the same pattern run backwards.  Fixed in the three places that evaluate a phasor (`_frame_plots.at_phase`, `_FieldView._instant`, `_FieldSeries._snapshot`); measured on `cos(ωt-kx)` accumulated over 2000 steps — the old sign walked the crest toward `-x`, the new one tracks the analytic wave to a grid step — and seen in Chrome on an analytic `Ez = e^{+jkx}`, whose crests move a quarter wavelength *away from the port* at `phase = 90°`.  The convention itself was never in doubt (DD-173's far-field transform conjugates in and out for it, DD-183's transit phase reads it), only the reconstruction; magnitudes, S-parameters and far fields do not go through these three functions.  A picture at a phase other than 0/180° is now the mirror in time of what the same call gave before — changelog *Fixed*.  (2) PyVista's menu card is `height: 36px` with `flex-wrap: nowrap`, so *Cut* and *Show* were clipped along the top: DD-261's `:has()` rule keyed on the **field** row, which a geometry viewer has none of.  It now keys on `.mio-menu-row`, the cut row every viewer carries.  (3) `mesh=` **is** used for a monitor (PEC cut-out, symmetry planes, grid on the cut) but was dropped whole once the frames were mirrored, taking `show_grid` with it in silence — measured on the half-box: 99 grid cells at `mirror=False`, no actor at `mirror=True`; it now warns and names `mirror=False`.  `show()` on a monitor read back from a project describes `geometry=` and `mesh=` instead of pointing at `show_field`.  Gate `test_a_rising_phase_runs_the_wave_forward`; record `investigations/viewer-review-followup/MEASUREMENTS.md` (internal record).
* **DD-266** (2026-09-07, patch after v0.7.0) — one cutting plane for the ParaView session, and every set of arrows coloured when it is made.  Three findings from the developer's first 6.1 session: a `geometry_cut_<monitor>` plus a plane link per monitor; "most arrows have coloring = Solid Color" because only the first shown pair ever got a representation (ParaView auto-colours only what is shown, so every hidden set came up flat); and a `vtkContext2DScalarBarActor` printf-format warning.  Now: one `geometry_cut` and one `cut_plane` link over it and every `<monitor>_slice` (`vtkSMProxyLink` takes any number of proxies — measured: moving one slice drags the other and the clip); `coloured(proxy, array, visible, bar)` replaces `show_coloured` and builds the representation through `GetDisplayProperties`, scales the transfer function to the monitor's cap *before* attaching the array (so no unbuilt data is asked for its range) and then sets visibility — applied to the cut arrows, the imaginary part, the volume arrows and the field sheet, one scalar bar on the visible pair, `UseSeparateColorMap` because monitors share an array name but not a cap.  The scalar-bar warning is not ours: 6.1's defaults are already `std::format`, the printf spelling comes from a state baked by 6.0 (DD-265); a fresh 6.1 export raises none.  Cost: one pipeline update per glyph set at export (4.2 s for a three-monitor run incl. bake).  Gate `test_every_glyph_comes_up_coloured_by_its_field`.
* **DD-265** (2026-09-07, patch after v0.7.0) — a ParaView **state file is bound to the ParaView that baked it**.  The developer's move to 6.1 turned the session into dozens of `vtkPVGeometryFilter … Missing input data` / `vtkPVMetaClipDataSet … 0 connections`, solids without colour and a dead slice: `bake_pvsm` had baked with `/usr/bin/pvpython` (6.0.1), and `simple.Reflect` is `AxisAlignedReflectionFilter` on 6.0 but `ReflectionFilter` on 6.1 — the geometry mirrors are dropped on load and the clips below them lose their input.  Three parts: (1) `export_vtm(mirrors=)` clips each block to the simulated half and appends its reflection (`vtkReverseSense` — a mirror reverses every triangle's winding), so `geometry.vtm` is the **whole** model and the session builds no reflection at all (`geometry.vtm.json` records the planes and forces a rebuild when they differ); (2) `export_paraview(pvpython=)` / `MAGNELIO_PVPYTHON` choose the baking interpreter, and its version travels back on stdout into the header of `paraview_open.py`; (3) the chapter's new *Which of the two files to open* — the script is version-free, the state is not.  Measured on a copy of the developer's `pi_mode`: the **6.0.1-baked state now loads on 6.1.0**, all eight proxies with data, geometry whole; script path pixel-identical on both.  Not a defect: the 29 pipeline entries — that project has five monitors, DD-262's seven is per monitor.  Gates: `test_export_vtm_completes_the_model_across_symmetry_planes`, `test_the_baking_paraview_is_named_in_the_script`, `test_the_bake_interpreter_can_be_named`, `test_the_interpreter_may_be_a_launcher_command`, `test_a_command_quoted_as_a_whole_still_resolves`, `test_a_bake_that_cannot_run_says_so`.  **Amended same day:** the setting is a *command*, not only a path (`resolve_pvpython` returns the argv prefix) — a Flatpak ParaView has no `pvpython` file to point at and is reached as `flatpak run --command=pvpython org.paraview.ParaView`.  Its sandbox (`filesystems=home`) does not reach pytest's `/tmp`, so the three pvpython gates skip there with the reason naming it; with `--basetemp` inside `$HOME` all nine pass under 6.1.1.  Record `investigations/paraview-flatpak/MEASUREMENTS.md` (internal record).
* **DD-264** (2026-09-07, patch after v0.7.0) — the second viewer pass, five findings of the developer's follow-up review.  `AnalysisEigenmode(project=)` returned a `Project` that could not `plot`: `Project.__getattr__` now serves `frequencies`/`modes`/`n_modes`/`solver_info`/`field`/`show`/`plot` off `self.eigenmodes` (absent, not failing, on a project of another analysis), so `project=` no longer changes what the returned object can do.  The PNG button is taken **where the picture is drawn** — `trame.refs['view_…'].captureImage()` in the browser (the kernel is `suppress_rendering=True` there, hence *This plotter has not yet been set up*, and its camera is not the user's), the kernel attachment in server rendering.  The projection toggle follows PyVista's handler with `update_camera()`: `update()` pushes the scene, only `push_camera` carries `parallelProjection`.  Ruler (axes titled in the display unit) and HTML export are back; the tab is *Magnelio Viewer* (`trame__title` set after PyVista's `initialize`); `GeometryModel.show()` is the 3D view and `plot()` a deprecated alias.  Finding: `plotter._id_name` is `P_<address>_<len(_ALL_PLOTTERS)>` and both come back after a close — six plotters in a row read as one — so PyVista's forever-cache handed a toolbar a dead plotter (the "sometimes nothing happens" of the PNG button); `_install_viewer` replaces an entry that is not this plotter's.  Verified in Chrome against a trame server built from the same scene code.  Gates: `TestToolbar` (3 new), `TestEigenModeRoundTrip` (2 new), `test_obsolete_geometry_plot_is_removed`.
* **DD-263** (2026-09-07, branch `feat/viewer-review-0.7`, patch after v0.7.0) — the 3D viewer after the developer's review: the browser opens in parallel projection (trame's camera serialiser learns `parallelProjection`/`parallelScale`, rebound under the name its initialiser reads); an own toolbar class in PyVista's viewer cache (reset, iso, x/y/z, projection toggle starting in the scene's state, screenshot, pop-out via `window.open`, help dialog from `_HELP_ROWS`; no bounding box/edges/ruler/export); fields continued across the symmetry planes (`mirror=True`, `resolve_mirrors` on the mesh, one `_frames_from_states` for every source, PEC mask folded); symmetry sheets at the declared position; `EigenmodeResult.show()` with the modes as frames (`kind="mode"`, `frame=` picks the first); phase play; fixed-width readouts; centred glyphs with `glyph="cone"`, `glyph_width=`; line and point monitors draw (`_stencil_axis`).  vtk.js bindings read from the bundle (left rotate, middle/alt pan, right/wheel/ctrl zoom, alt+shift roll).  Gates: six new test classes; tutorial 18 shows the TESLA π-mode whole.  The developer's look followed as DD-264.
* **DD-262** (2026-09-07, branch `feat/paraview-on-demand`, patch after v0.7.0) — the ParaView export is a call, not a side effect: `Project.export_paraview()` / `export_paraview_eigenmodes()` are the only writers of the `.vtr` series, the session script, the baked `.pvsm` and (on first call) `geometry.vtm`; a run's close, a resume and the eigenmode solver write nothing (`ProjectStore.create(paraview=)` deprecated, no-op).  Reason: since DD-259 a time monitor's export is a materialised copy at twice the store's bytes per cell.  The per-monitor pipeline is **one `ProgrammableFilter`** (`<monitor>_field`: numpy mirroring with `mirror_sign` per array — H now continued as an axial vector — cell→point, `vtkResampleToImage`, `_mag`/`_len` arrays) feeding one `_slice`, `_arrows` (centred via `GlyphTransform`), the linked `geometry_cut_`, and hidden `_volume`/`_volume_arrows` (a thresholded point cloud): 7 proxies instead of ~28, one cut instead of three.  Findings: the ImageData output needs a `RequestInformationScript` or the script runs 4× per update; ParaView's filter preamble shadows `max/min/abs/sum/any/all` in `__main__`; ParaView 6.0.1 × numpy ≥ 2.4 kills every Python filter (`numpy.in1d`) — shimmed in the session header; a state file cannot be helped there at all, ParaView's own filter preamble runs before any script of ours (DD-265).  Gates: `TestFieldScript` (the script under the mio VTK with the shadowed namespace), `test_pvsm_reloads_with_field_on_the_cut` (pvpython), `validation/paraview_symmetry_certificate.py` PASS (H continuation 1e-15).
* **DD-261** (2026-09-06) — fields in the volume of the 3D viewer.  `show(volume="arrows"|"isosurface"|"both")`: arrows on an even 3D lattice over the kept half (`_lattice3`/`_resample3`, trilinear with the metal dropped from the stencil) and translucent isosurfaces (persistent `RectilinearGrid` → `cell_data_to_point_data` → `contour`, clipped open at the cut; ± levels for a signed component), both *Show* groups of every volume source; `_FieldFrames.volume` loader beside `layer`.  Arrows on the cut and in the volume now coloured by magnitude on the sheet's scale with a length floor of 0.3 spacing (`arrow_color=` for one colour).  Field controls in a **second toolbar row** (play, frame, phase, field, iso level 5–95 %, arrow density) via a scoped `:has()` style on PyVista's card — ends the one-row overflow.  Finding: `smooth_shading=True` hands the mapper a copy, so the persistent polydata was no longer what the browser drew.  Gates `TestVolume`, `TestVolumeControls`.  Seen in Chrome (second row, isosurface, level slider, coloured lattice arrows); **the developer's own browser look is still open.**
* **DD-257** (2026-09-05, branch `perf/fit-td-step-overhead`, patch after v0.6.0) — the CPU kernel sweeps plane by plane (every field array streams once per half-step), interior rows carry no boundary guards, a frozen PEC bbox face is not re-written, the energy check reduces without temporaries.  Bit-identical fields; solver step at 16.8 Mcells on eight threads **M1 Pro 91 → 111 GB/s, 7800X3D 40.6 → 50.0 GB/s**.  DD-180's "prefetch wall" is retracted (Numba STREAM on the M1: 133 GB/s, worst-stride stencil 125 GB/s — stride-blocking has ≤ 5 % to give); finer fusion *loses* on Apple Silicon.  Gate `test_numba_kernels.py`; record `investigations/fit-td-bandwidth/MEASUREMENTS.md`.  The M1 Pro MacBook is an arm64 test device (`ssh macbook`, `~/magnelio-dev`, editable; two `TestSectionBatch` bit-for-bit tests fail there by 1 ULP — FMA contraction, not a defect).

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
17 gallery tutorials, with no internal imports.
Released tutorials run on the GPU box on pure defaults (the DD-096 criterion is on by
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
classifier re-masks tangential-surface E edges with a local masked-face
detour (DD-276), so 2D mode solvers and the 3D update see
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
* **Core** — the top-level ``magnelio`` namespace (14 names, pinned
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
  ``analysis`` (result types), ``fields``, ``post``, ``plots``, ``io``, ``constants``.
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
`examples/tutorials/*.py`, 01–18 and 20 released, 21–27 on the foundation branch; of which
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

* **Large-grid TD setup memory** — DD-278 adds `analysis.estimate_memory()` phase/monitor budgets and total-cell/held-array mesh reporting in GiB/MiB/KiB; unknown auxiliary costs remain explicit. The CFL scaling defect stays open: approximately 112 GiB of curl-list payload at 168.7 million cells, plus roughly 48 GiB of mesh arrays and conversion/Lanczos workspaces. Two earlyoom terminations confirmed; allocation sites unknown. Evidence: `investigations/hesr-memory-termination/MEASUREMENTS.md` (internal record). The budgeted 51-cell HESR run completed on 2026-10-04: 8.58 million cells, explicit coax gap anchors, 144,901 steps to the 70 dB energy criterion, 13.91 GiB sampled peak RSS; project stored separately in the internal dossier. Record: `investigations/hesr-memory-budget/MEASUREMENTS.md` (internal record).
* **Band-pipeline runtime** — convolution (DD-245) and axis ranking
  (DD-247) closed: 314.9 s → 81.2 s on a 201-point axis, no item
  dominates.  Left: postprocessing is `eigs` + `splu` at 96.6 % over a
  per-frequency LU that cannot be amortised; one factorisation per
  channel per frequency and `k = 4` where one mode is consumed are
  unpriced; the like-for-like default-axis run remains unmeasured.
* **Band port floor (KB-038)** — wordlength question answered, defect
  not fixed.  The convolution state was **already double**; the
  single-precision contact is the per-step round trip through the field
  array in `update_e` (84-92 % of the double-to-single gap; quantising
  one side is worse than both).  What is left is a *solver* decision
  (port plane and first interior period in double, bulk single); not
  priced.  The law is **not band-specific** (the modal port erodes at
  the same rate but saturates at the float32 floor, −112.6 dB) — users:
  `docs/methods/precision.md`; dossier `investigations/kb038-wordlength/`.
* **Staircase correction for oblique paths (DD-269)** — excess 1.6–3.9 %; <0.16 % is a fit residual.
  NEC-2: 3.53094 GHz/72.02 Ω; staircase errors reach 21.1 %/36.0 %.
  The Guiffaut exact-edge reactance defect is mesh-persistent.  A reciprocal
  energy blend derives alpha = 0.97693 from translation invariance, unchanged
  at 2 mm; six 1 mm placements/orientations are within 0.44 %/1.65 % of NEC.
  A cycle-free shared-node lift reaches 0.17 % loop spread.  A planar NEC junction improves from 1.34 % to 0.68 % rotated-curve spread; a nonplanar four-way node gives 1.01 % curve/0.46 % current spread.  Absolute errors remain ~5 %.  Coupled CFL is exactly 0.972 of field-only here (0.98/0.99 unstable); a sparse row-sum rule certifies 0.697 with ~10 nonzeros/segment.  Homogeneous epsilon/mu scaling and static-conductivity passivity pass (lossy axis/diagonal curve spread 1.30 %).  A grounded wire normal to PEC passes an independent NEC monopole gate (~1 % resonance error; contact excess only 0.14 curve-percentage-points).  Tilted infinite-plane contact lifts preserve charge but miss NEC curves by 22--70 %; this is an auxiliary-model gate, not a defect in the shipped PEC staircase.  A finite-pad bond arch confines mask overlap to 1.07--2.38 mm per endpoint and touches no free-span segment.  Production `ThinWire` stays connected in finite-metal bond loops.  A controlled thin-sheet foot comparison shifts the parallel zero 1.23 % at 0.5 mm and 0.46 % at 0.25 mm; the earlier 6.5 % came from different whole-arm slopes and is not a contact error.  Two-pad loops also expose return-path dependence (6.1 % zero shift with conductor thickness).  A NEC2++ 31-patch finite-pad candidate fails its own foot-segmentation gate (2 GHz reactance -702/-8,629/-2,352 ohm for 1/2/3 segments), so it is not an accuracy reference.  Absolute bond accuracy still needs a resolved finite-pad reference; interfaces and further refinement remain open; dossier: `investigations/oblique-lumped-path/`.
* **Ports on the GPU** — only `TestBandDTBCOnGPU` (KB-045) exercises a
  port on a device; `tests/conftest.py` pins the suite to NumPy.
* **Modal decomposition overshoots unity transmission** (|S21| 1.0030
  frozen, 1.0078 dispersive, rank-independent, growing with f) — left by
  the closed launch pair DD-239 → DD-244 → DD-248, DD-244's to own.
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
* **Symmetry planes (DD-154/DD-155/DD-172/DD-286).**
  Lumped ports/elements on a symmetry plane are corrected since DD-172;
  the ParaView session mirrors H as the axial vector it is since DD-262.
  CPML min/max profiles now use the true staggered positions (KB-023);
  the resonant full/half dipole has max dS11 = 3.04e-5.
* **Ports with several signal conductors** report the channel's own
  reference in `dispersion()`, not a modal power–current impedance
  (DD-244); `TDResult` carries no reference impedances.
* **Mesh build** — bounded-surface sections (DD-287) are opt-in:
  `MeshControl(robust_sections=True)`, default False; per build and worker.
  Default fast paths retain KB-043: thin regions can be lost near tangency.
  Five controlled 51-cell pairs:638.6 ->1700.6s (+166.3%); identical8,582,496cells.
  Main RSS6.25 ->6.76GiB, tree10.24 ->10.68GiB; finite-record S/Ez changes tiny.
  Global area-error improvement unresolved; actual wall losses unmeasured.
  Curvature-weight diagnostics expose outliers. Further investigation deferred.
  Opt-in acceptance:255 focused tests; Ruff/API/hygiene/DD clean; offline HTML built.
  Evidence: `investigations/near-tangent-hesr-impact/` (internal dossier).

## Deferred / nice-to-have

* **Double-precision late energy growth:** understand the increase in both
  repeated-cell runs, already present before GPU resume; cause remains open.
* **Robust-section applicability:** find relevant geometries and indicators for
  targeted warnings/automatic selection. Actual wall losses remain uninvestigated
  and could be highly relevant; weight diagnostics are not loss-error measurements.
  Checklist: `investigations/near-tangent-hesr-impact/FOLLOW_UP.md` (internal dossier).

* **Pulsed band-edge S-parameters on dispersive lines** are record-
  truncation limited.  The candidate — late-time pole estimation — is
  built ([[DD-272]]) but untested on *this* case: a dispersive line's tail
  is not obviously a few resonances.
* **A third compute backend** — assessed, nothing built (DD-180);
  blocker is `xp is not np` as the capability test.  Metal rejected (no
  FP64), CuPy on ROCm is the candidate.
* **Temporal blocking of the CPU kernel** — the only route above the
  STREAM triad (DD-257); incompatible with the per-step hooks (ports,
  CPML, sources, recorder).  Not pursued.  A float32 curl accumulator
  on the E side would buy 3 % on Apple Silicon and is a numerics change.
* **Residual GPU small-grid floor** (~0.41 ms/step at 10k cells, port round
  trips — DD-092); **tensor (gyrotropic) μ** (DD-089's ADE is scalar per
  axis); **off-Yee field-monitor interpolation** (must preserve the DD-085
  units); **far-field accepted power on the streamed path** (``gain`` raises until it wires ``1 − Σ|S|²``, DD-070).
