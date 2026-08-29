# Sources, monitors and post-processing

## Sources, waveforms and excitations

A time-domain drive is described by three objects with three
distinct jobs:

- A **source** is a model object — declared on the
  `GeometryModel` before meshing with `add_source`, exactly like a
  port with `add_port` — that says *where* and *how* energy enters
  the domain: a plane wave on a total-field/scattered-field box
  (`sources.SourcePlaneWave`), and in later releases incident fields,
  current paths and beams.  Ports are sources *and* loads and keep
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
from the eigenmodes of a stored project, `from_function` and
`from_arrays` from a formula or from data.  Its excitation has no
waveform — the amplitude alone scales the field (`amplitude_unit` is
`"1"`) — and the run then rings down freely, which is how a Q is
measured (how-to *Ring-down*).

The electric field is written on the primal edges and the magnetic
field half a leapfrog step ahead, following the discrete Faraday law,
so a mode of the discrete operator starts as exactly that mode and
oscillates without a transient.  Several initial fields in one run
superpose.

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

## Field, flux and frequency monitors

- **MonitorFieldTime** — time snapshots of E/H in a region, streamed
  to the on-disk store.
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
`.data` is in those units from the moment the run returns; `.data_raw`
exposes the undivided bins for callers who want the transient itself.
For a TE/TM feed the waveform launches a frequency-dependent power —
the mode's wave impedance varies across the band — and the run divides
additionally by the ratio $|a(f)|/|W(f)|$ of the incident wave it
separated at the port to the waveform spectrum, so the statement holds
for every feed type (DD-198).

## Project store, checkpointing, resume

Runs stream results append-only into an HDF5-based on-disk project
store (SWMR single-writer/multi-reader), with periodic checkpoints
and bit-exact resume (DD-070).  File formats: HDF5, XDMF/VTK for
field visualisation.  This is engineering infrastructure, not a
research method; the formats are community standards.
