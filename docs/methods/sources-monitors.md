# Sources, monitors and post-processing

## Sources, waveforms and excitations

A time-domain drive is described by three objects with three
distinct jobs:

- A **source** is a model object — declared on the
  `GeometryModel` before meshing with `add_source`, exactly like a
  port with `add_port` — that says *where* and *how* energy enters
  the domain: a plane wave or any other incident field on a
  total-field/scattered-field box (`sources.SourcePlaneWave`,
  `sources.SourceFieldIncident`), a field present at `t = 0`
  (`sources.SourceFieldInitial`), a recorded Huygens surface
  (`sources.SourceFieldSurface`), an impressed current along a curve
  (`sources.SourceCurrentPath`), and in later releases charged
  beams.  Ports are sources *and* loads and keep
  their own namespace, `magnelio.ports`.  Sources travel with the
  mesh (`mesh.sources`, next to `mesh.ports` and `mesh.elements`;
  `Mesh.with_sources` attaches them to a grid built without a model)
  and share one name namespace with ports and elements.
- A **waveform** (`magnelio.signals`) is a pure, unit-peak function
  of time that knows its own bandwidth: `WaveformGaussian(f_max)` for
  DC-inclusive drives, `WaveformGaussianModulated(f_min, f_max)` for a
  band-limited pulse on a carrier, `WaveformSine` and `WaveformStep`
  for continuous-wave and step drives (infinite duration, so the run
  needs an explicit length), `WaveformTable` for measured or imported
  samples and `WaveformFunction` for an arbitrary callable.  Every
  waveform reports `f_min`, `f_max`, `f_center` (the carrier, or
  `None` for baseband forms) and `t_end`, can be sampled into a
  `Signal1D`, and gives its spectrum in closed form where one exists.
  Amplitude, delay and phase are *not* part of a waveform — one
  waveform object can drive several ports and sources.
- An **excitation** (`magnelio.Excitation`) binds a port channel or a
  source, *by name*, to a waveform and a weight: `amplitude` in the
  source's natural unit (`sqrt(W)` — the incident power wave — for
  ports, `V/m` for a plane wave; every source publishes it as
  `amplitude_unit`), a `delay`, and on carrier waveforms a `phase`,
  which is applied as a delay of `phase / (360 · f_center)`.  Two
  modes of one port at 90° make a circularly polarised feed; a phase
  on a baseband pulse is rejected because such a pulse has none.

The scattering analysis excites *channels* — one `(port, mode)` per
independent run, listed in `run(excited=...)` — and derives the
waveform per excited mode from the analysis band: a Gaussian
pulse over `[0, f_max]` for TEM and lumped ports, a modulated
Gaussian over `[max(f_cutoff, f_min), f_max]` above a mode's
cut-off, so that no pulse energy sits below cut-off where it would
be totally reflected.  `AnalysisScatteringTD(waveform=...)` overrides
that choice with any waveform; a waveform reaching above the
analysis band warns.  Simultaneous excitations of several ports and
sources in one run (`excitations=[Excitation(...), ...]`) are the
business of the general time-domain analysis, not of the scattering
analysis, which rejects the argument.

At the component level a bound waveform is what a port operator or
a source injects: `operator.set_excitation(mode, waveform)` and
`source.set_excitation(waveform, amplitude=..., delay=...)` — the
solver-facing form of the same triad.

## The general time-domain analysis

`AnalysisTD` is the problem class behind every time-domain run: one
leapfrog march under a list of excitations applied *simultaneously*,
returning a `TDResult` — the recorded modal voltage and current of
every port channel (`result.signal(port, mode)`), the sampled drive of
every excitation (`result.excitation_signal(name)`), the power waves
`a(t)` / `b(t)` where a port defines them, the stored-energy trace and
the monitors.  No S-parameters: a scattering matrix needs one channel
per run, and that is what `AnalysisScatteringTD` — a subclass — does
with `run(excited=[...])`, on the very same engine.

Three rules follow from the waveforms:

- The run length is estimated from the drives: the time the last
  excitation has died out, `max(delay + waveform.t_end)`, plus a
  generous number of diagonal transits of the domain.  The estimate
  sets the energy-check cadence and the checkpoint stride; the run
  itself ends on the energy or port-signal criterion.
- A continuous-wave waveform (`WaveformSine`, a `WaveformStep`
  without `fall_time`) never decays, so such a run needs an explicit
  length — `run(t_end=…)` in seconds or `total_time_steps=` — and
  the decay criteria are switched off.
- Frequency-domain monitors keep their raw transient bins.  With
  several drives there is no single reference spectrum to divide out,
  so `result.renormalize(name)` names the excitation the monitors
  should refer to; the scattering analysis does this for you with the
  excited channel's waveform.

Several modes of one port may be driven in the same run (two `TE11`
modes at 90° make a circularly polarised feed), and a run may combine
ports and sources — an antenna under plane-wave illumination is one
`excitations=[...]` list.  With `project=` the run streams into a
project store under a name (`run(name=…)`, default `run_<n>`), resumes
with `resume(project, name)`, and `project.result(name)` rebuilds its
`TDResult`.

## Incident fields on a total-field/scattered-field box

Illumination from outside uses the **total-field/scattered-field
(TF/SF)** technique: the domain is split by a virtual box; consistency
corrections on the six box faces inject the incident wave into the
total-field region while the exterior carries only scattered field
(DD-013).  The TF/SF formulation is due to
Merewether, Fisher and Smith {cite}`merewether1980` and
Umashankar and Taflove {cite}`umashankartaflove1982`;
textbook treatment in {cite}`taflovehagness2005`.
The incident samples are converted to FIT grid quantities per
edge/face, so amplitudes are physical (V/m) on any grid — an
in-house calibration (DD-085).  `SourcePlaneWave(name, direction,
polarization, corners)` declares the wave and its box; the excitation
that names it supplies the waveform and the peak field.  Propagation
is along a grid axis.

Any other incident field is `SourceFieldIncident(name, field,
corners)`, whose `field(x, y, z, t, drive)` returns `((Ex, Ey, Ez),
(Hx, Hy, Hz))` at the sample positions of one box face — a
superposition of waves, a tabulated measured field, a beam.  `drive`
is the excitation's time function, so a retarded field reads
`drive(t - k·r/c0)`.  The plane wave is the special case in which the
retardation collapses to one delay per face, which is why it has its
own class and evaluates the waveform on a handful of values per step
instead of on the face.

The incident field must solve the free-space Maxwell equations by
itself; the box corrections assume it does.  An E/H pair that does not
(a transversally tapered "beam" whose longitudinal components are
dropped, say) radiates the inconsistency into the scattered-field
region, where it is indistinguishable from a scattered wave.

## Initial fields

A transient run can also start *from* a field instead of being driven
into one.  `SourceFieldInitial(name, field)` carries a
`magnelio.fields.FieldState` that becomes the state at `t = 0`;
`SourceFieldInitial.from_project(project, name=…, mode=…)` takes it
from the eigenmodes of a stored project, `from_recording(recording,
name=…, t=…)` from a frame of a time monitor's `FieldRecording` —
live or read back from a project — and `from_function` and
`from_arrays` from a formula or from data.  Its excitation has no
waveform — the amplitude alone scales the field (`amplitude_unit` is
`"1"`) — and the run then rings down freely, which is how a Q is
measured (how-to *Ring-down*).

The march holds the magnetic field half a leapfrog step ahead of the
electric one.  A field given at one instant — an eigenmode, a formula
— is moved there by a half discrete Faraday step, so a mode of the
discrete operator starts as exactly that mode and oscillates without
a transient.  A recorded frame already *is* such a pair: its E stands
at the frame's instant, its H half a time step later (`times_h`), and
the source takes both as they are (`h_lead`, the lead of the magnetic
samples, is the recording's half step).  A frame of a monitor covering
the whole domain, replayed on the same grid with the same time step,
therefore continues the recorded run bit for bit from that frame — a
ring-down cut short resumes from its last frame, a state reached under
one excitation is handed to another model.  Several initial fields in
one run superpose.

Nothing about the model restricts an initial field.  Where the run
carries state besides the fields — an absorber's convolutions, the
pole currents of a dispersive material, the branch currents of a
surface-impedance wall, a port's boundary history — that state starts
at zero, which is the quiescent condition: the absorber is empty, the
material unpolarised, the wall carries no current, the exterior of a
port was quiet.  That is a well-defined initial-value problem, and a
stable one, but it is not the *steady* state a mode would have built
up around itself.

In practice the difference is small, because those states relax on
their own time scale.  A copper-walled cavity rung down with
`wall_model="sibc"` — whose branch currents start at zero — returns a
wall Q within a third of a percent of the perturbative
surface-resistance evaluation on the same mode, two independent routes
to the same loss.

Ports are what make the ring-down of a coupled resonator possible: a
waveguide port's transparent boundary assumes only a quiet *exterior*
before `t = 0`, and the field reaching it during the run is the whole
point of an external-Q measurement.  A discrete port is a resistor
without memory and has no start condition at all.  An initial field
sitting *on* a port plane is allowed too; what it costs is worth
knowing, since it leaves as a prompt burst (about four times the
steady port signal at 20 % of the field peak) and biases a fitted
external Q by roughly 0.4 % at −14 dB and 2 % at −6 dB, with nothing
measurable below −26 dB.

Whether the field you load is the mode you meant, and whether the
decay you see is the one you wanted to measure, are modelling
questions — not something the library decides for you.

## Field sources: reusing what a model radiates

A radiator that has been simulated once does not have to be simulated
again.  The equivalence principle says that the tangential **E** and
**H** on a closed surface stand for everything inside it: replayed on
that surface in a second model, they radiate the same field outwards
and nothing inwards.  Magnelio spells this as a pair —
`MonitorFieldSurface` records, `SourceFieldSurface` replays — with a
`magnelio.fields.SurfaceRecording` between them:

```python
# Run 1 — the antenna, on its own, in a small domain
box = monitors.MonitorFieldSurface(name="antenna", corners=((-8e-3,) * 3, (8e-3,) * 3))
...
box.recording().save("antenna.h5")

# Run 2 — the platform, driven by the antenna without meshing it
model.add_source(
    sources.SourceFieldSurface.from_file(
        "antenna.h5", name="antenna", position=(0.0, 0.0, 0.25), rotation=("z", 90)
    )
)
```

The recording file is the whole interface: the two models share no
grid, no geometry and no project.  The replaying box may sit anywhere
and be turned by a multiple of 90°; free angles are refused rather
than rounded, because the box is spanned by grid-node planes and a
tilted recording would cross the target grid obliquely, with no
samples where the corrections are applied.  The excitation that drives
the source carries no waveform — the recording *is* the time function
— only a scale factor and a delay.

What the surface must enclose is every source of the field it stands
for.  A conductor cutting through a face carries current across it and
the recording is then short of exactly that part; the monitor says so.
A ground plane is the licit exception: it closes a domain face, so the
box is left open there, and the recording is valid in a second model
that continues that plane.

Two things decide the accuracy.  The first is the sampling rate.  The
replay interpolates the recording linearly in time, and that error
falls with the square of the sample spacing rather than obeying the
sampling theorem, so the default records eight samples per period at
`f_max` — twice Nyquist is measurably not enough.  What matters is the
bandwidth the field actually carries, which is not always the model's
`f_max`: a field started from a sharply localised initial state
carries energy far above it, and then the rate has to follow the field
rather than the setting.

The second is the grid.  Replayed on the grid it was recorded on, and
in the same place, the interpolation reads its own samples and the
replay is exact — measured at the single-precision floor, with the
inside of the box quiet to −97 dB.  Across grids it is the spatial
interpolation that limits: a scattered field recorded at twelve nodes
per wavelength and replayed on a different grid reproduces the outside
field to about a percent, with the inside quiet to roughly −40 dB.

## Impressed currents on a path

Where the current distribution is known, the current itself is the
source.  `SourceCurrentPath` prescribes `I(t)` along a curve through
the model, and the fields it radiates follow: a short filament is a
Hertzian dipole, a closed one a magnetic dipole standing in for a
coil, and a long one an injection probe on a harness or a lightning
channel.  The excitation carries the current, in amperes:

```python
model.add_source(
    sources.SourceCurrentPath(name="dip", path=[(0, 0, -1e-3), (0, 0, 1e-3)])
)
analysis.run(excitations=[mio.Excitation("dip", waveform=wf, amplitude=1e-3)])
```

The path is either a sequence of points or any `geo.Curve` — a
polyline, an arc, a spline, a helix, or a chain of them — and is
rasterised onto the grid edges by the same rasteriser that lumped
elements and voltage probes use, so a filament and a voltage probe can
never disagree about which edges a curve occupies.  The vertices of
the path become grid planes, so a filament ends where it was declared
to end rather than at the nearest node.

The current is prescribed, not solved for.  That is the whole
difference between this source and a wire: nothing the model does
changes `I(t)`, so a current path is the right description of a driven
coil or a known interference current, and the wrong one for an antenna
whose current distribution is the answer — feed that with a port.

Two properties follow from the discrete operators rather than from any
correction applied on top of them, and they are worth knowing because
they bound what the source can and cannot do.

*The dipole moment is exact.*  An oblique path is walked as a
staircase, but the staircase runs monotonically from the start node to
the end node, so the signed sum of its steps is exactly the chord
between them.  The first moment of the current is therefore what was
asked for on any grid; only the higher multipoles see the steps.

*The charge at the ends is exact.*  The discrete divergence of a
discrete curl vanishes identically, so an open filament accumulates
precisely the time integral of its current on its two end nodes and no
charge anywhere else — to the last bit of double precision.  An open
current path *is* a consistent oscillating dipole, with the right
near-zone electrostatic part; there is nothing to clean up.

Edges the solver holds at zero — inside a perfect conductor, or
tangential to a PEC wall — cannot take an impressed current.  A path
crossing such a region radiates less than it was asked to, and says so
rather than doing it quietly.  Under a symmetry declaration the mesh
is the kept half: give the path in that half and let the symmetry wall
supply its image.

Measured against the closed-form short dipole — a 2.5 mm filament at
10 GHz in an air cube — the radiated power approaches the analytic
value from below as the grid is refined, 0.964 at ten nodes per
wavelength to 0.983 at eighteen, with the residual tracking the
near-to-far-field box's own closure and not the source.  The pattern
is `sin θ` and the peak directivity 1.5 throughout.

## Field containers

Three objects in `magnelio.fields` carry a field between analyses,
plots and files, all on the Yee positions of one grid — nothing is
averaged until a picture asks for it:

- **`FieldState`** — one snapshot: the six components at their own
  positions (`component`, `positions`), at arbitrary points (`at`),
  averaged onto cell centres (`cell_centred`), sliced (`plot`), in 3D
  (`show`), continued across the model's symmetry planes
  (`mirrored`), and written back into a run as an initial field.
  Eigenmodes hand one out per mode.
- **`FieldRecording`** — frames of a transient field on one grid:
  `times` for the electric field and `times_h`, half a leapfrog step
  later, for the magnetic one; `frame(i)` and `at_time(t)` are
  `FieldState`s, `component(name)` the whole stack; a frame becomes
  the start of a new run (`SourceFieldInitial.from_recording`).
- **`FieldSpectrum`** — complex frames, one per frequency:
  `at_frequency(f)` is the complex pattern, `snapshot(f, phase=…)` its
  real field at an instant.

Both series may hold a subset of the six components (`components`);
a frame fills the rest with zeros.  A recording's grid is whatever
region was recorded — a plane monitor's is one cell thick, with the
normal component on one node plane and the tangential ones on two,
which is the layer a field picture stands for.  The field monitors
record into these containers (`monitor.recording`, `monitor.spectrum`),
and they are the vocabulary for fields you assemble or resample
yourself.

### Energy and flux

A monitor's frames are the solver's own samples, so the two FIT
identities the march itself relies on hold on them: the stored energy
$\tfrac12 e^{\mathsf T} M_\varepsilon e + \tfrac12 h^{\mathsf T} M_\mu h$
and the Poynting flux $\sum e\,h$ through a plane.  A recording or
spectrum from a monitor carries what they need — the material
operators of its region, cut from the solver's own when the monitor
was attached, and how its edges are booked — so `recording.energy()`
gives the joules of every frame and `recording.flux("z", position)`
the watts through the region's cross-section at a plane;
`spectrum.energy()` and `spectrum.flux(...)` are the time averages of
a frequency monitor's pattern, per watt incident; `frame.energy()` and
`frame.flux(...)` do the same for one `FieldState`.  The numbers are
the march's: a recording of the whole domain reproduces the run's
energy trace — the leapfrog's own conserved pairing of the two
magnetic half-steps, formed from a single frame through the discrete
Faraday law — a recording's flux is what a `MonitorFluxTime` on the
same plane records, bit for bit, and on a matched line a spectrum's
flux is $|S_{21}|^2$.  Where a region is cut out of the domain, the
dual patches on the cut count by the half that lies inside, so two
adjoining regions add up to their union; a model solved behind
symmetry planes reports the whole model, as the flux monitor does.
A field assembled by hand or an eigenmode carries no operators and
says so.  A project store keeps the operators with the monitor, so a
recording read back in another session states its energy too.

## Field, flux and frequency monitors

- **MonitorFieldTime** — time snapshots of E/H in a region, streamed
  to the on-disk store.
- **MonitorFieldSurface** — the tangential fields on a closed box over
  time, for reuse as a field source (above).
- **MonitorFieldFrequency** — running (accumulated) discrete Fourier
  transform of the fields at selected frequencies during the march;
  the running-DFT-during-timestepping technique is standard practice
  in time-domain solvers {cite}`taflovehagness2005`.
- **MonitorFluxTime** — Poynting flux through a plane,
  $\sum \hat e \cdot \hat h$ in the FIT pairing (physical Watt after
  DD-085).
- **MonitorWallLoss** — see the
  [conductor-losses chapter](conductor-losses.md).

All monitors return physical SI units; the calibration (C = 1 pinned
at the excitation source) is in-house bookkeeping (DD-085).

A field monitor records the *grid quantities on the Yee positions* of
its region — a copy of the solver's own samples, nothing averaged — and
hands them out as a `FieldRecording` (`monitor.recording`) or, for the
running transform, a `FieldSpectrum` (`monitor.spectrum`); a
cell-centred array is one call away (`recording.cell_centred(...)`),
but nothing is averaged until asked for.  The recording's
`times` are the instants of the electric field, `times_h` those of the
magnetic one, half a step later.  A project store keeps the same
staggered frames, one dataset per component, and the ParaView export
converts them at export time.  A field monitor plots on a slice
(`plot`, `interact`) and opens in the 3D viewer (`show`), where the
field lies on the viewer's cutting plane and the position slider walks
it through the recorded volume — see [the viewer chapter](viewer.md).

A plotted field plane is one *layer* of cells, sampled at their
centres, not a mathematical plane — the plane coordinate printed in the
title is the cell-centre coordinate the request snapped to.  Geometry
overlays follow the same rule: thin wires, discrete ports and lumped
elements are drawn when they lie inside the displayed layer, so a wire
declared on the grid nodes half a cell away still appears in the
picture of the field around it (DD-175).

## Signal processing

Excitation waveforms are Gaussian-family pulses with prescribed
spectral occupancy; S-parameters divide recorded spectra by the
excitation spectrum (standard practice
{cite}`taflovehagness2005`).  CW measurements use
lock-in phasor extraction over an integer number of periods after
settling (`cw_lockin_phasors`) — standard signal processing.

A frequency monitor is divided by the same spectrum, and for the same
reason.  Its running sum $\sum_n F(t_n)\,e^{+j\omega t_n}\,\Delta t$ is
the transient folded with the excitation, so it carries an extra factor
of time and the pulse's own spectral shape.  Since the excitation
waveform *is* the incident power-wave amplitude $a(t)$ in $\sqrt{\rm W}$
(DD-078), dividing it out leaves the field of a **1 W CW excitation** at
each monitor frequency — E in V/m, H in A/m, per $\sqrt{\rm W}$ of
incident power.  A run performs that division on its own monitors, so
`.spectrum` is in those units from the moment the run returns;
`.spectrum_raw` exposes the undivided transform for callers who want
the transient itself.

The sign in that exponent fixes the library's phasor convention:
frequency-domain fields are those of the $e^{-j\omega t}$ convention,
so the instant of a pattern at time $t$ is
$\mathrm{Re}\left(F\,e^{-j\omega t}\right)$.  That is what the
`phase` argument of every picture means — degrees of $\omega t$,
advancing with time — and what the viewer's phase play animates, so a
travelling wave moves the way it ran in the simulation.  The
far-field transform (whose textbook formulas are written for
$e^{+j\omega t}$) conjugates on the way in and back out, so its
patterns are phasors of the same convention as everything else.
For a TE/TM feed the waveform launches a frequency-dependent power —
the mode's wave impedance varies across the band — and the run divides
additionally by the ratio $|a(f)|/|W(f)|$ of the incident wave it
separated at the port to the waveform spectrum, so the statement holds
for every feed type (DD-198).

## Project store, checkpointing, resume

Runs stream results append-only into an HDF5-based on-disk project
store (SWMR single-writer/multi-reader), with periodic checkpoints
and bit-exact resume (DD-070).  File formats: HDF5 for the store, VTK
series for field visualisation.  This is engineering infrastructure,
not a research method; the formats are community standards.

(paraview-export)=
## ParaView export

A field monitor's frames leave the store as VTK files when you ask
for them: `project.export_paraview()` writes one run's set,
`project.export_paraview_eigenmodes()` the eigenmodes', and both
tessellate the solids into `geometry.vtm` on their first call.  Under
`runs/<run>/paraview/` there is one rectilinear `.vtr` per frame of a
time monitor (`<monitor>/t_0000.vtr`, …) or per frequency of a
frequency monitor (`f_0000.vtr`, …), collected by a `<monitor>.pvd`
whose axis is the frame's electric instant, or its frequency.  The
files hold *cell data*: the recorded components averaged onto the
cell centres at export time — the numbers `recording.cell_centred()`
returns — and the vectors `E` and `H` where a group was recorded
whole.  ParaView reads plain VTK and never touches the staggered
frames in `results.h5` or `fields_freq.h5`.  The copy takes about as
much disk as the monitor's own data, which is why it is a call and
not a side effect of the run.

Beside the data, `paraview_open.py` builds a `paraview.simple`
pipeline and, when `pvpython` is on the path, `paraview.pvsm` is
baked from it as a double-clickable state.  What the pipeline browser
shows per monitor — the count below is *per monitor*, so a run with
five of them carries five such groups — is the reader, `<monitor>_field`
— a Python filter
that mirrors the recorded half across the model's symmetry planes
with the continuation rules of the monitor plots (`E` and `H` each
with their own parity), averages the cells onto the points and
resamples them onto an even lattice, handing out `<field>`,
`<field>_mag` and `<field>_len` (the arrow length, saturating at a
high percentile of the magnitude so edge singularities keep their
direction without dictating the scale) — then `<monitor>_slice`, one
cut normal to the region's shortest extent whose plane widget turns
it to any other, and `<monitor>_arrows` on the cut, centred on their
sample points.  `<monitor>_volume` and `<monitor>_volume_arrows` wait
hidden for the whole volume, thresholded to the cells carrying field;
a frequency monitor's `<monitor>_arrows_im` holds the field a quarter
period later.

Beside the monitors there is exactly **one** `geometry_cut`: the
solids clipped by the session's cutting plane.  That plane is shared —
`geometry_cut` and every monitor's slice are linked, so dragging any
one of them drags them all and the solids are always opened where the
field is shown.

Every set of arrows is coloured by its field's magnitude when the
session is built, whether it is shown at once or waits hidden, so a
set switched on later comes up on the same scale rather than in a flat
colour.  The colour carries the true magnitude while the arrow length
is compressed, and each monitor gets its own transfer function,
because the cap it is scaled to is its own.  Eigenmodes take the same shape one directory up, one `.vtr`
per mode.  Tutorial 07 opens such a session.

`geometry.vtm` holds the **whole** model: a geometry declared behind
symmetry planes is clipped to the simulated half and mirrored across
each plane while the file is written, not by a filter in the session.
The planes it was built for are recorded in `geometry.vtm.json`, so a
file written for other planes is made again rather than shown as a
half model.

The export is a step *after* the run, not a live view; after a
resume, call it again and the set is regenerated.  Watching a run
while it marches is what `watch`, `follow` and the notebook viewer
are for ([Projects and runs](projects-and-runs.md), the how-to
*Watching a simulation that is still running*).

### Which of the two files to open

`paraview_open.py` is the robust one.  It builds the session live, so
it works on whatever ParaView runs it:

```bash
paraview --script=paraview_open.py
```

`paraview.pvsm` is the convenience — a double-click, no command line —
and it is **bound to the ParaView that baked it**.  A state file names
its proxies the way that release spelled them, and a renamed proxy is
dropped on load together with everything downstream of it, which reads
as missing solids and errors about filters without input rather than
as a version mismatch.  The version that baked it is written into the
header of `paraview_open.py` beside it.  On a machine carrying more
than one ParaView, name the one that will open the session:

```python
project.export_paraview(pvpython="/opt/ParaView-6.1/bin/pvpython")
```

or set `MAGNELIO_PVPYTHON` once; `bake_state=False` skips the state
file altogether.  Two build notes: ParaView 6.0 bundles a
`numpy_interface` older than numpy 2.4, which stops every Python
filter — the generated script shims that for `paraview --script` and
for the bake, but a state file opened on such a build cannot be
helped, since ParaView's own filter preamble runs before any script of
ours.  Use `paraview --script=` there, or a newer ParaView.
