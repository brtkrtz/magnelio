# Magnelio — Project Status

*Last updated: 2026-09-08.*  **Released v0.7.0** (2026-09-06; a minor
under the Cargo reading — the field monitors' dictionary API is gone,
`docs/migration-0.7.md`).  Unreleased on `main`: **v0.8.0** — the
phasor convention of the frequency-domain fields (DD-268,
`docs/migration-0.8.md`), the ParaView export on request (DD-262) and
the viewer review (DD-263…267).  In v0.7.0:
**DD-261** — fields in the volume of the 3D viewer (lattice arrows,
isosurfaces, coloured arrows, a second toolbar row; browser review
open); **DD-260** — energy and flux from a recording (`recording.energy()`,
`.flux()`; a monitor carries its region's operators, in the store too);
**DD-259** complete — field monitors keep the grid quantities
(`monitor.recording`/`.spectrum`, store schema 3.0, the dictionary API
gone, `docs/migration-0.7.md`), fields in the 3D viewer, a recorded
frame as the start of a run (`SourceFieldInitial.from_recording`);
**DD-257** (CPU kernels 1.2× faster, bit-identical); **DD-256** (a thin
wire on a thin sheet).  **Released v0.6.0** (2026-09-05; `Project.runs`
hands out `Run` objects, `docs/migration-0.6.md`): the usability series
**DD-253/254/255** (timed marches with clock, rate and ETA; live `Run`
objects, `aborted`/`stale`, reprs without arrays; `plot_energy`,
`watch`, `follow`, `monitor`), **DD-249/250** (`lofted(blend="tangent")`),
**DD-251** (in-place notebook progress), **DD-252** (`refine_port_modes`).
Before it: v0.5.0–v0.5.2 (2026-09-02…04) DD-224…DD-248 — the API
grammar and its phases, the content gate, `docs/migration-0.5.md`.

Open: KB-023, KB-038, KB-043 and KB-046.  Unit and integration: 3581 passed / 13 skipped
(2026-09-08 on merged `main`, NumPy backend; the four GPU tests need
`CUPY_ACCELERATORS=""` outside the sandbox).
Channels: GitHub, PyPI, conda-forge and the two docs channels below.

This file states what *is*.  Chronology: `git log --first-parent main`;
reasoning: `design-decisions.md`; open bugs: `known-bugs.md`.  Measured
floors regenerate from the `validation/` certificates their DDs name.

## Recent decisions

Newest first, one line each; the full record is the DD entry.

* **DD-268** (2026-09-07, branch `feat/phasor-convention-0.8`, for v0.8.0) — one phasor convention for the whole library.  Follows the developer's question after DD-267: *which* convention do the fields use?  The answer was "two".  The S-parameter path, `Signal1D.at_frequencies`, `Waveform.spectrum`, `cw_lockin_phasors` and the modal half-step `e^{+jω dt/2}` are all `e^{+jωt}`; only the monitors' running DFT summed `Σ F(t) e^{+jωt} dt` and handed out the conjugates, which is why DD-173's far-field transform had to conjugate in and out and why DD-183's transit phase for a beam toward `+z` read `e^{-jk_B z}`.  The accumulator (`monitors/_dft.py`, and the private copy in `wall_loss.py`) now sums `e^{-jωt}`; the three phasor evaluators go back to `Re(F e^{+jφ})` (DD-267's *meaning* — phase is ωt — is untouched); `post/far_field.py` drops its four conjugations and applies Balanis verbatim.  **Store:** schema stays 3.0 — a bump would refuse 0.7 stores whose data converts exactly.  `fields_freq.h5`, `wall_loss.h5` and `far_field.h5` carry `phasor_convention = "exp(+jwt)"`; absent means the older era and the bins (and the far field's stored divisor) are conjugated at the single read site of each file, an unknown value raises.  Exact in both directions, so a 0.7 run resumes bit-identically — verified by removing the hook, which fails all five legacy gates.  Invariant and unchanged: magnitudes, `Re Σ e·h*`, `|·|²`, gain, directivity, S-parameters.  Re-signed with the convention: the stripline how-to's `beam_voltage` and the Hertzian validation script (`arg(E_θ/j)` 180° → 0°).  Eight new gates (two on the kernel, one on the NTFF phase, five on legacy stores).
* **DD-267** (2026-09-07, patch after v0.7.0; **sign amended by DD-268**) — the phase of a picture is ωt.  Three findings of the developer's third viewer pass.  (1) "With the phase animated the wave runs backwards, toward the exciting port": the running DFT sums `Σ F(t) e^{+jωt} dt`, so a bin of `A·cos(ωt+φ)` is `(T/2)·A·e^{-jφ}` — a phasor of the **`e^{-jωt}` convention**, whose instant is `Re(F e^{-jωt})`.  Every picture applied `Re(F e^{+jφ})`, the same pattern run backwards.  Fixed in the three places that evaluate a phasor (`_frame_plots.at_phase`, `_FieldView._instant`, `_FieldSeries._snapshot`); measured on `cos(ωt-kx)` accumulated over 2000 steps — the old sign walked the crest toward `-x`, the new one tracks the analytic wave to a grid step — and seen in Chrome on an analytic `Ez = e^{+jkx}`, whose crests move a quarter wavelength *away from the port* at `phase = 90°`.  The convention itself was never in doubt (DD-173's far-field transform conjugates in and out for it, DD-183's transit phase reads it), only the reconstruction; magnitudes, S-parameters and far fields do not go through these three functions.  A picture at a phase other than 0/180° is now the mirror in time of what the same call gave before — changelog *Fixed*.  (2) PyVista's menu card is `height: 36px` with `flex-wrap: nowrap`, so *Cut* and *Show* were clipped along the top: DD-261's `:has()` rule keyed on the **field** row, which a geometry viewer has none of.  It now keys on `.mio-menu-row`, the cut row every viewer carries.  (3) `mesh=` **is** used for a monitor (PEC cut-out, symmetry planes, grid on the cut) but was dropped whole once the frames were mirrored, taking `show_grid` with it in silence — measured on the half-box: 99 grid cells at `mirror=False`, no actor at `mirror=True`; it now warns and names `mirror=False`.  `show()` on a monitor read back from a project describes `geometry=` and `mesh=` instead of pointing at `show_field`.  Gate `test_a_rising_phase_runs_the_wave_forward`; record `investigations/viewer-review-followup/MEASUREMENTS.md` (internal record).
* **DD-266** (2026-09-07, patch after v0.7.0) — one cutting plane for the ParaView session, and every set of arrows coloured when it is made.  Three findings from the developer's first 6.1 session: a `geometry_cut_<monitor>` plus a plane link per monitor; "most arrows have coloring = Solid Color" because only the first shown pair ever got a representation (ParaView auto-colours only what is shown, so every hidden set came up flat); and a `vtkContext2DScalarBarActor` printf-format warning.  Now: one `geometry_cut` and one `cut_plane` link over it and every `<monitor>_slice` (`vtkSMProxyLink` takes any number of proxies — measured: moving one slice drags the other and the clip); `coloured(proxy, array, visible, bar)` replaces `show_coloured` and builds the representation through `GetDisplayProperties`, scales the transfer function to the monitor's cap *before* attaching the array (so no unbuilt data is asked for its range) and then sets visibility — applied to the cut arrows, the imaginary part, the volume arrows and the field sheet, one scalar bar on the visible pair, `UseSeparateColorMap` because monitors share an array name but not a cap.  The scalar-bar warning is not ours: 6.1's defaults are already `std::format`, the printf spelling comes from a state baked by 6.0 (DD-265); a fresh 6.1 export raises none.  Cost: one pipeline update per glyph set at export (4.2 s for a three-monitor run incl. bake).  Gate `test_every_glyph_comes_up_coloured_by_its_field`.
* **DD-265** (2026-09-07, patch after v0.7.0) — a ParaView **state file is bound to the ParaView that baked it**.  The developer's move to 6.1 turned the session into dozens of `vtkPVGeometryFilter … Missing input data` / `vtkPVMetaClipDataSet … 0 connections`, solids without colour and a dead slice: `bake_pvsm` had baked with `/usr/bin/pvpython` (6.0.1), and `simple.Reflect` is `AxisAlignedReflectionFilter` on 6.0 but `ReflectionFilter` on 6.1 — the geometry mirrors are dropped on load and the clips below them lose their input.  Three parts: (1) `export_vtm(mirrors=)` clips each block to the simulated half and appends its reflection (`vtkReverseSense` — a mirror reverses every triangle's winding), so `geometry.vtm` is the **whole** model and the session builds no reflection at all (`geometry.vtm.json` records the planes and forces a rebuild when they differ); (2) `export_paraview(pvpython=)` / `MAGNELIO_PVPYTHON` choose the baking interpreter, and its version travels back on stdout into the header of `paraview_open.py`; (3) the chapter's new *Which of the two files to open* — the script is version-free, the state is not.  Measured on a copy of the developer's `pi_mode`: the **6.0.1-baked state now loads on 6.1.0**, all eight proxies with data, geometry whole; script path pixel-identical on both.  Not a defect: the 29 pipeline entries — that project has five monitors, DD-262's seven is per monitor.  Gates: `test_export_vtm_completes_the_model_across_symmetry_planes`, `test_the_baking_paraview_is_named_in_the_script`, `test_the_bake_interpreter_can_be_named`, `test_the_interpreter_may_be_a_launcher_command`, `test_a_command_quoted_as_a_whole_still_resolves`, `test_a_bake_that_cannot_run_says_so`.  **Amended same day:** the setting is a *command*, not only a path (`resolve_pvpython` returns the argv prefix) — a Flatpak ParaView has no `pvpython` file to point at and is reached as `flatpak run --command=pvpython org.paraview.ParaView`.  Its sandbox (`filesystems=home`) does not reach pytest's `/tmp`, so the three pvpython gates skip there with the reason naming it; with `--basetemp` inside `$HOME` all nine pass under 6.1.1.  Record `investigations/paraview-flatpak/MEASUREMENTS.md` (internal record).
* **DD-264** (2026-09-07, patch after v0.7.0) — the second viewer pass, five findings of the developer's follow-up review.  `AnalysisEigenmode(project=)` returned a `Project` that could not `plot`: `Project.__getattr__` now serves `frequencies`/`modes`/`n_modes`/`solver_info`/`field`/`show`/`plot` off `self.eigenmodes` (absent, not failing, on a project of another analysis), so `project=` no longer changes what the returned object can do.  The PNG button is taken **where the picture is drawn** — `trame.refs['view_…'].captureImage()` in the browser (the kernel is `suppress_rendering=True` there, hence *This plotter has not yet been set up*, and its camera is not the user's), the kernel attachment in server rendering.  The projection toggle follows PyVista's handler with `update_camera()`: `update()` pushes the scene, only `push_camera` carries `parallelProjection`.  Ruler (axes titled in the display unit) and HTML export are back; the tab is *Magnelio Viewer* (`trame__title` set after PyVista's `initialize`); `GeometryModel.show()` is the 3D view and `plot()` a deprecated alias.  Finding: `plotter._id_name` is `P_<address>_<len(_ALL_PLOTTERS)>` and both come back after a close — six plotters in a row read as one — so PyVista's forever-cache handed a toolbar a dead plotter (the "sometimes nothing happens" of the PNG button); `_install_viewer` replaces an entry that is not this plotter's.  Verified in Chrome against a trame server built from the same scene code.  Gates: `TestToolbar` (3 new), `TestEigenModeRoundTrip` (2 new), `test_plot_is_a_deprecated_alias_of_show`.
* **DD-263** (2026-09-07, branch `feat/viewer-review-0.7`, patch after v0.7.0) — the 3D viewer after the developer's review: the browser opens in parallel projection (trame's camera serialiser learns `parallelProjection`/`parallelScale`, rebound under the name its initialiser reads); an own toolbar class in PyVista's viewer cache (reset, iso, x/y/z, projection toggle starting in the scene's state, screenshot, pop-out via `window.open`, help dialog from `_HELP_ROWS`; no bounding box/edges/ruler/export); fields continued across the symmetry planes (`mirror=True`, `resolve_mirrors` on the mesh, one `_frames_from_states` for every source, PEC mask folded); symmetry sheets at the declared position; `EigenmodeResult.show()` with the modes as frames (`kind="mode"`, `frame=` picks the first); phase play; fixed-width readouts; centred glyphs with `glyph="cone"`, `glyph_width=`; line and point monitors draw (`_stencil_axis`).  vtk.js bindings read from the bundle (left rotate, middle/alt pan, right/wheel/ctrl zoom, alt+shift roll).  Gates: six new test classes; tutorial 18 shows the TESLA π-mode whole.  The developer's look followed as DD-264.
* **DD-262** (2026-09-07, branch `feat/paraview-on-demand`, patch after v0.7.0) — the ParaView export is a call, not a side effect: `Project.export_paraview()` / `export_paraview_eigenmodes()` are the only writers of the `.vtr` series, the session script, the baked `.pvsm` and (on first call) `geometry.vtm`; a run's close, a resume and the eigenmode solver write nothing (`ProjectStore.create(paraview=)` deprecated, no-op).  Reason: since DD-259 a time monitor's export is a materialised copy at twice the store's bytes per cell.  The per-monitor pipeline is **one `ProgrammableFilter`** (`<monitor>_field`: numpy mirroring with `mirror_sign` per array — H now continued as an axial vector — cell→point, `vtkResampleToImage`, `_mag`/`_len` arrays) feeding one `_slice`, `_arrows` (centred via `GlyphTransform`), the linked `geometry_cut_`, and hidden `_volume`/`_volume_arrows` (a thresholded point cloud): 7 proxies instead of ~28, one cut instead of three.  Findings: the ImageData output needs a `RequestInformationScript` or the script runs 4× per update; ParaView's filter preamble shadows `max/min/abs/sum/any/all` in `__main__`; ParaView 6.0.1 × numpy ≥ 2.4 kills every Python filter (`numpy.in1d`) — shimmed in the session header; a state file cannot be helped there at all, ParaView's own filter preamble runs before any script of ours (DD-265).  Gates: `TestFieldScript` (the script under the mio VTK with the shadowed namespace), `test_pvsm_reloads_with_field_on_the_cut` (pvpython), `validation/paraview_symmetry_certificate.py` PASS (H continuation 1e-15).
* **DD-261** (2026-09-06) — fields in the volume of the 3D viewer.  `show(volume="arrows"|"isosurface"|"both")`: arrows on an even 3D lattice over the kept half (`_lattice3`/`_resample3`, trilinear with the metal dropped from the stencil) and translucent isosurfaces (persistent `RectilinearGrid` → `cell_data_to_point_data` → `contour`, clipped open at the cut; ± levels for a signed component), both *Show* groups of every volume source; `_FieldFrames.volume` loader beside `layer`.  Arrows on the cut and in the volume now coloured by magnitude on the sheet's scale with a length floor of 0.3 spacing (`arrow_color=` for one colour).  Field controls in a **second toolbar row** (play, frame, phase, field, iso level 5–95 %, arrow density) via a scoped `:has()` style on PyVista's card — ends the one-row overflow.  Finding: `smooth_shading=True` hands the mapper a copy, so the persistent polydata was no longer what the browser drew.  Gates `TestVolume`, `TestVolumeControls`.  Seen in Chrome (second row, isosurface, level slider, coloured lattice arrows); **the developer's own browser look is still open.**
* **DD-260** (2026-09-06) — energy and flux are identities on a recording.  A field monitor takes the region's cut of the solver's `M_ε`/`M_μ` at attach (`attach_operators`), plus the kind of each region end (`cut`/`wall`/`pmc`) and the symmetry faces (`fields/_operators.py`, `RegionOperators`); `FieldState`/`FieldRecording`/`FieldSpectrum` get `energy()` and `flux(normal, position)`, full-model booking as the flux monitor.  The conserved energy needs no second frame: `h(n−½) = h(n+½) + β_H·(C e)` makes the DD-225 pairing `½hMμh + (dt/2)(Ce)·h` from one frame.  Measured: energy trace reproduced to 1e-9, flux monitor bit for bit (whole domain and one-cell layer), |S21|² to 1.4e-5 on the plate line, half box behind an electric symmetry plane = full box to 1e-9 (the plane must be booked as a cut), two regions add to their union.  A spectrum is an **RMS phasor** (DD-078) — read as peak it gave half a watt.  Store: operator datasets pre-declared (SWMR), filled by the sink after setup, `valid` flag; schema-additive on 3.0.  Fixed on the way: a project run without ports crashed at its first flush.  Gates `test_field_energy.py` (33), `test_recording_energy.py` (4).
* **DD-259** (2026-09-05, *Proposed*; **step 0 merged `7a59be1`** after the developer's browser review — the first cut re-created the actors per frame and froze the browser after 75 frames, now updated in place; **step 1 merged `5b71c52`**: `fields.FieldRecording`/`FieldSpectrum`, `FieldState.mirrored` exact on PEC and PMC planes, the cell-centre averaging moved to `fields._interp`; **step 2 merged `ddd40e8`**: the monitors record the grid quantities on the region's Yee positions, `monitor.recording`/`.spectrum` hand out the containers — bit-identical to the old record-time average because a region carries its own dual widths; **step 3 merged `fa4bde8`**: `.data`/`.data_raw`/`.component`/`.region` removed on monitors and readers, pictures drawn through `monitors/_frame_plots.py` averaging only the drawn layer, lazy store reader, `docs/migration-0.7.md`; a frame's `t` is the electric instant `t + dt` (was one step early), the frequency monitor keeps the port recorder's phase convention; store schema **3.0** (`layout="yee"`, `dual_x/y/z`, 2.x monitors refused), `fields.xdmf` and `io/xdmf.py` gone, time monitors exported as `.vtr` series; **step 4** documented (chapter section *ParaView export*, tutorial 07); **step 5 merged**: `SourceFieldInitial.from_recording(recording, name=, t=|frame=)` — a frame is the march's own leapfrog pair, so the source carries `h_lead` (lead of its H samples) and `attach` takes the Faraday step of the difference `(½ − h_lead/dt)`; a run resumed from a whole-domain frame on the same grid and step continues the recorded one **bit for bit** (gate `test_recorded_frame_continues_the_march`); energy/flux from a recording left to their own DD) — field monitors keep the grid quantities and derive every view at access time.  The cell-centre averaging at record time is DD-014's ParaView layout (2026-03-11, before the first monitor existed), a filter that cannot be undone: conductor faces smeared, no energy/flux from a recording, no replay as an initial field, H labelled half a step early.  Five steps for 0.7.0 after v0.6.0 and the DD-256 patch: containers (`FieldRecording`/`FieldSpectrum`), raw recording + store schema bump (old stores refused), hard break of `.data`/`.region`, ParaView as VTR export, replay.  **Step 0 shipped:** the 3D viewer lays a field on its cutting plane (`monitor.show()`, `field.show()`, `plots.show_field`) — the exposed cell layer as a coloured sheet (|E|/|H| or one signed component), arrows on an even lattice, frame/phase sliders and a component selector in the toolbar, PEC cells cut out with `mesh=`; the frame source is a protocol (`_FieldFrames`), so step 1 swaps the storage underneath.  Gates `test_field_3d.py` (23; the controls driven through trame's state), tutorials 07/20, chapter *3D viewer → Fields on the cut*.
* **DD-258** (2026-09-05, merged `c006d39`) — the solver's field container is `FieldArrays`; `FieldState` names the public one only (the double name hid the grid-quantity vs. field distinction DD-085 exists to keep visible).  Rename only, no user-visible change.
* **DD-257** (2026-09-05, branch `perf/fit-td-step-overhead`, patch after v0.6.0) — the CPU kernel sweeps plane by plane (every field array streams once per half-step), interior rows carry no boundary guards, a frozen PEC bbox face is not re-written, the energy check reduces without temporaries.  Bit-identical fields; solver step at 16.8 Mcells on eight threads **M1 Pro 91 → 111 GB/s, 7800X3D 40.6 → 50.0 GB/s**.  DD-180's "prefetch wall" is retracted (Numba STREAM on the M1: 133 GB/s, worst-stride stencil 125 GB/s — stride-blocking has ≤ 5 % to give); finer fusion *loses* on Apple Silicon.  Gate `test_numba_kernels.py`; record `investigations/fit-td-bandwidth/MEASUREMENTS.md`.  The M1 Pro MacBook is an arm64 test device (`ssh macbook`, `~/magnelio-dev`, editable; two `TestSectionBatch` bit-for-bit tests fail there by 1 ULP — FMA contraction, not a defect).
* **DD-256** (2026-09-05) — a thin wire lands on a thin sheet the way it lands on a solid: vertices inside the sheet's thickness collapse onto the sheet plane (no sliver, no spurious endpoint warning), the foot ring composes ``m`` with the sub-cell value instead of yielding.  Monopole on a sheet vs. on a solid, one grid: **+0.6 % / −2.2 %** (was +2.3 %).  Radius rule untouched — the Lange bonds stay bricks.
* **DD-255** (2026-09-05) — a run is watched by polling the store, and one figure is what everyone watches.  `Project.watch(interval, on_change=, timeout=)` — a generator yielding the project at every change (signature: index stamp + per-run `(state, n_energy_samples)`) until `done`/`aborted`/`stale`; **polling on purpose** (no inotify: dependency, blind on network mounts and to half flushes).  `plot_energy()` on `TDResult`/`ScatteringTDResult`/`Run`/`Project`: dB below peak, criterion dashed, `plot_s` conventions — the same number as the progress line and the table.  `Project.monitor()`: `VBox(HTML, Image)` refreshed by a daemon thread that sets **widget state only** (DD-251's rule) and renders on its own Agg canvas, never pyplot.  `Run.n_steps` moves while marching (latest energy sample's step).  **Amendments from the first notebook session:** `Project.follow(interval, plot=, timeout=, stream=)` — the watch loop ready-made, its display *replacing itself* (notebook `clear_output`+`display`, terminal `ESC[nA`, log appended; a bare expression in a loop shows nothing, and the inline backend flushes figures only at cell end, so `plot=True`/`plot=callable(project, ax)` renders the picture per change as PNG), probed in a real ipykernel via `jupyter_client`; the energy axis runs from ten dB below the criterion to +5 dB (`floor_db=`), because the empty grid's first samples read −3000 dB.  How-to *Watching a simulation that is still running* (solver on a thread, real output), Tutorial 07's nine-statement energy block is `proj.plot_energy()`.  Gates `test_plot_energy.py`, `test_project_monitor.py`, `test_project_watch.py`.
* **DD-254** (2026-09-05) — a run is an object, a project knows whether anyone is still writing it, and nothing prints its arrays.  **The 0.6.0 break:** `Project.runs` is a mapping of live `Run` views (`.state .n_steps .energy_db .energy_trace .elapsed .result() .monitors`; channel keys tuples; `docs/migration-0.6.md`).  `meta` follows `project.json` by `(mtime, inode, size)` until the stored status is terminal, so a live watcher needs no `refresh()`; `_load_run` keys its cache on `(n_steps, finished)`.  Status rule fixed (any aborted → **aborted**, was `running` forever); **`stale`** derived from the writer's pid on the same host (POSIX; elsewhere unknown → `running`).  Repr principle in `_repr.py` (what, how big, what state — never arrays): `Project` prints a summary plus run table and **cannot raise**, `CheckpointState` is a `Mapping` that prints sizes, `TDResult`/`ScatteringTDResult`/`SParameterResult`/`RunSettings` summarise, HTML tables in notebooks.  `check_api_surface.py` pin now lists DD-246's verbosity switch (had drifted).  Gates `TestRunObjects`, `TestProjectStatus`, `TestCheckpointState`, `test_repr.py`.
* **DD-253** (2026-09-05) — every march is timed, and the time loop says what runs and how long it has run.  The clock lives in `FITTimeDomainSolver.run()` (a wrapper around the loop with a `finally`, so all five exits and exceptions are covered) and reaches the result objects (`started`/`finished`/`elapsed`, marching only, in the result contract) and the store (`_RunSink.close(elapsed=)`, `_finalize_run` **accumulates** over resumes, `reopen_run` stamps `resumed`; `pid`/`host` on every run entry and as `meta["writer"]`; `meta["analysis"]` for the whole call).  The line: `step 2900/∞ | 0.7 s | energy -58.4/-70 dB | 3.9k steps/s`, closing line in the same slots; **ETA only on a fixed step count** — the run-length estimate is a 25-transit scale, an ETA on it would overstate a TEM run several-fold, so the header states the *rule* (`stops at energy -70 dB or port signal -60 dB, cap 388480 steps`) instead of a number.  `run | finished in 2.6 s (2 runs)` per `run()`/`resume()`; seven bare prints now go through `Reporter.note` (multi-line aware).  Gates `TestDurations`, `TestMarchLines`, `TestRunTiming`.
* **DD-252** (2026-09-04) — `refine_port_modes` converges what the mode family defines.  The DD-244 ladder defaulted to `z_line`, and the round port of a rectangular-to-circular taper — whose cut-off the grid reads **0.33 % low**, amplified by β to 2 % at 11.9 GHz — answered `has no line impedance`.  `target="auto"` (default) resolves from the level-0 report: line impedance for TEM/quasi-TEM, cut-off otherwise.
* **DD-251** (2026-09-04) — a notebook is an in-place stream, and the viewer starts its trame server on the running loop.  *Run All* aborted at the first `plot()` (`RuntimeError: cannot enter context … is already entered`): PyVista's `elegantly_launch` nests a loop into the kernel's via `nest_asyncio2`, and under ipykernel 7 that nested loop picks up the *next queued cell* — reproduced without a browser by queueing four requests through `jupyter_client`.  The server is now a task on the loop that already runs, with an empty widget in the cell until `server.ready`.  DD-246 had filed the notebook with the logs (`isatty()` is false), so a one-minute GPU run printed its first line at **86 %** of the march; an `ipykernel` stream is an in-place stream at **0.5 s**.  Gates `TestNotebookStream`; record `investigations/viewer3d/runall_repro.py`.
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

* **3D viewer, browser review (DD-261, DD-263, DD-264, DD-267)** — the toolbar, the volume representations, the mirrored frames and the phase play are gated through trame's state; the toolbar was driven in Chrome for DD-264 (tab name, projection both ways, ruler, PNG with the browser camera, HTML export) and for DD-267 (the card growing with the cut row, the phase moving the crests away from the port), the mouse bindings in the help dialog are still read from the vtk.js bundle, not clicked through.  DD-268 conjugated the bins under all of it — both the sum and the reconstruction flipped, so the direction a wave travels on screen is unchanged, and the developer confirmed the viewer after the change (2026-09-08).
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
* **Ports on the GPU** — only `TestBandDTBCOnGPU` (KB-045) exercises a
  port on a device; `tests/conftest.py` pins the suite to NumPy.
* **The launch pair (DD-239 → DD-244 → DD-248) — closed.**  Left
  behind it: the decomposition **overshoots unity transmission**
  (|S21| 1.0030 frozen, 1.0078 dispersive, rank-independent, growing
  with f) — DD-244's to own.
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
  guarantee; `_FACET_REFINE_FRACTION = 0.1` leaves the facet path a
  3.16x finer sagitta budget than the exact one.
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
  the ParaView session mirrors H as the axial vector it is since DD-262.
  CPML min/max faces are not mirror images (KB-023) — full-vs-half
  parity of resonant open structures floors at ~1e-2.
* **Ports with several signal conductors** report the channel's own
  reference in `dispersion()`, not a modal power–current impedance
  (DD-244); `TDResult` carries no reference impedances.
* **Mesh build** — speed campaign closed 2026-08-29 (DD-201…DD-223;
  deferred work, A/B switches and traps in DD-223).  Open against it:
  KB-043.

Closed construction sites are tombstoned where they were decided and
are not repeated here.

## Deferred / nice-to-have

* **Pulsed band-edge S-parameters on dispersive lines** are record-
  truncation limited; candidate: late-time AR estimation.
* **A third compute backend** — assessed, nothing built (DD-180);
  blocker is `xp is not np` as the capability test.  Metal rejected (no
  FP64), CuPy on ROCm is the candidate.
* **Temporal blocking of the CPU kernel** — the only route above the
  STREAM triad (DD-257); incompatible with the per-step hooks (ports,
  CPML, sources, recorder).  Not pursued.  A float32 curl accumulator
  on the E side would buy 3 % on Apple Silicon and is a numerics change.
* **Residual GPU small-grid floor** (~0.41 ms/step at 10k cells, port
  round trips — DD-092); **tensor (gyrotropic) μ** (DD-089's ADE is
  scalar per axis); **off-Yee field-monitor interpolation** (must
  preserve the DD-085 units); **far-field accepted power on the streamed
  path** (``gain`` raises until it wires ``1 − Σ|S|²``, DD-070).
