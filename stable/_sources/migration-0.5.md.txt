# Upgrading from 0.4.x

This page is for anyone with a script, a notebook or a saved project
written against Magnelio 0.4.x.  Release 0.5 renames a number of
classes and arguments, removes a few, and bumps the project-store
format.  Nothing here is a silent change of meaning except where this
page says so; almost every rename fails loudly at import time or at
the first call, and a 0.4.x store stops at the first read of its
metadata.  The one rename that can stay silent — the waveform
functions under their old submodule path — is called out where it
arises.

The whole list follows from one change.  In 0.4.x every object that
could be driven carried its own drive: a plane-wave source took a
waveform string, a band and an amplitude; a port spec took an
`excitation` field; the scattering analysis took an `ExcitationSpec`.
The excitation is now one concept, bound in one place.  A **source**
(and a **port**) says only *what* is there and *where* — geometry, no
time function, no strength.  A **waveform**
({class}`~magnelio.signals.Waveform` and its subclasses) is a pure
unit-peak function of time that knows its own band and duration, and
carries no amplitude, so one waveform object can drive several
channels.  An **excitation** ({class}`~magnelio.Excitation`) binds a
waveform to a named channel or source, with an amplitude in that
source's own unit, a delay, and a phase.  Someone who has internalised
that separation can derive every rename below without reading the
table: whatever described *when* and *how strongly* moved off the
source and onto the excitation, and whatever described a *band* became
a waveform class.


## Quick reference

| 0.4.x | 0.5 | If you leave it alone |
|---|---|---|
| `sources.PlaneWaveSource` | {class}`~magnelio.sources.SourcePlaneWave` | `ImportError` / `AttributeError` |
| `PlaneWaveSource(…)` without a name | `SourcePlaneWave(name=…, …)` | `TypeError`: missing `name` |
| `PlaneWaveSource(waveform=…, f_max=…, f_center=…, amplitude=…)` | a waveform and an amplitude on the excitation | `TypeError` on each keyword |
| `ports.ExcitationSpec(f_min=…, f_max=…)` | `WaveformGaussianModulated(f_min=…, f_max=…)` | `AttributeError` / `ImportError` |
| `ExcitationSpec(mode_index=n)` | `Excitation(port, mode=n)`, or the `(port, n)` pair in `run(excited=…)` | — |
| `PortSpecCoax/RectWG/Numerical/MultiConductor(excitation=…)` | `operator.set_excitation(mode, waveform)` on the built operator | `TypeError` on `excitation` |
| `AnalysisScatteringTD(excitation=…)` | `AnalysisScatteringTD(waveform=…)` | `TypeError` on `excitation` |
| `signals.gaussian(t, f_max)` | {class}`~magnelio.signals.WaveformGaussian` | `AttributeError` on `magnelio.signals` |
| `signals.modulated_gaussian(t, f_max, f_min)` | {class}`~magnelio.signals.WaveformGaussianModulated` — **argument order flips** | `AttributeError` on `magnelio.signals` |
| `signals.waveform_for_mode(f_max, omega_c, f_min)` | the analysis default; pass no waveform | `AttributeError` on `magnelio.signals` |
| `monitors.MonitorFarField` | {class}`~magnelio.monitors.MonitorFarFieldFrequency` | `AttributeError` / `ImportError` |
| `MonitorFluxTime(plane=("z", z0))` | `MonitorFluxTime(normal="z", position=z0)` | `TypeError` on `plane` |
| `MonitorWallLoss(reference_plane=("z", z0))` | `MonitorWallLoss(normal="z", position=z0)` | `TypeError` on `reference_plane` |
| a project store written by 0.4.x | re-run the analysis | `ProjectSchemaError` |

None of the old names has a compatibility shim.  Where the table says
`AttributeError` or `ImportError`, the name is gone from the namespace
the table names.


## Sources, waveforms and excitations

### The plane wave

`PlaneWaveSource` is now
{class}`~magnelio.sources.SourcePlaneWave`.  Only the name changed —
`magnelio.sources` and the `magnelio.sources.plane_wave` submodule
both carried the old class and both carry the new one — and the new
class requires a `name`, the name an excitation uses to address it.
`direction`, `polarization` and `corners` keep their meaning and their
defaults, and `SourcePlaneWave.from_ranges(name=…, x1=…, x2=…, …)`
still exists.

At the component level, where the source is handed to the solver
directly:

```python
# 0.4.x
from magnelio.sources.plane_wave import PlaneWaveSource

src = PlaneWaveSource(
    direction=(0.0, 0.0, 1.0),
    polarization=(1.0, 0.0, 0.0),
    corners=tf_box,
    f_max=f_max,
    waveform="gaussian",
)
```

```python
# 0.5
from magnelio.signals import WaveformGaussian
from magnelio.sources import SourcePlaneWave

src = SourcePlaneWave(
    name="pw",
    direction=(0.0, 0.0, 1.0),
    polarization=(1.0, 0.0, 0.0),
    corners=tf_box,
)
src.set_excitation(WaveformGaussian(f_max=f_max))
```

Everything downstream is unchanged: the source still goes to the
solver as `sources=[src]`.

Fixing the class name alone is not enough — the removed keywords fail
next, one at a time:
`SourcePlaneWave.__init__() got an unexpected keyword argument
'waveform'`, likewise `f_max`, `f_center` and `amplitude`.

### The high-level form

A source is now a model object: declare it on the
{class}`~magnelio.GeometryModel` before meshing, and drive it by name
at run time.

```python
model.add_source(
    sources.SourcePlaneWave(
        name="pw",
        direction=(0.0, 0.0, 1.0),
        polarization=(1.0, 0.0, 0.0),
        corners=((-box, -box, -box), (box, box, box)),
    )
)
...
result = analysis.run(
    excitations=[
        mio.Excitation("pw", waveform=signals.WaveformGaussian(f_max=f_max), amplitude=1.0),
    ],
    energy_stop_db=60.0,
)
```

`Excitation.amplitude` is the peak incident field in V/m here; every
source publishes the unit its amplitude is read in as
`amplitude_unit`.

:::{admonition} The one trap in this section
:class: warning

If you carry `f_max` straight across from the old plane wave into
`WaveformGaussian`, the pulse changes shape and nothing complains.
The old plane-wave Gaussian was
`exp(-(t - t0)² / (2 σ²))`, while `WaveformGaussian` is
`exp(-(t - t0)² / σ²)` with the same `σ = 2 / (π f_max)` and
`t0 = 4 / f_max`.  The old pulse is therefore √2 wider in time and its
spectrum √2 narrower: at `f_max` = 5 GHz and `t = t0 ± t0/2` the old
value is 0.007192 and the new one 5.2·10⁻⁵.  A migrated run
illuminates a wider band than before.

Usually that is harmless or an improvement.  Where you need the old
pulse sample for sample — comparing against archived results, say —
state the old closed form explicitly:

```python
waveform = signals.WaveformFunction(
    fn=lambda t: np.exp(-((t - 4 / f_max) ** 2) / (2 * (2 / (math.pi * f_max)) ** 2)),
    f_max=f_max / math.sqrt(2),
    t_end=8 / f_max,
)
```
:::

### Continuous-wave drives

`PlaneWaveSource(waveform="sine", f_center=…)` becomes
{class}`~magnelio.signals.WaveformSine` on the excitation:

```python
src = sources.SourcePlaneWave(name="pw", direction=(0, 0, 1), polarization=(1, 0, 0))
src.set_excitation(signals.WaveformSine(f=1e9), amplitude=2.5)
# or, by name, at run level:
mio.Excitation("pw", waveform=signals.WaveformSine(f=1e9), amplitude=2.5)
```

Numerically the two agree — both are `A·sin(2π f t)` with a hard
switch-on — and `WaveformSine` adds a `rise_time=` for a raised-cosine
switch-on that the old waveform string had no way to express.  Note
that its `t_end` is infinite, so a run driven by it needs an explicit
length (`run(t_end=…)` or `total_time_steps=`) and cannot stop on
energy decay.

This path was never used in a published example or tutorial; if none
of your scripts mentions `f_center`, there is nothing to do here.

### `ExcitationSpec`

`magnelio.ports.ExcitationSpec` is removed.  It held three kinds of
information, and each goes somewhere different:

- `f_min`, `f_max` and the `waveform` string become a waveform class —
  `waveform="modulated_gaussian"` is
  `WaveformGaussianModulated(f_min, f_max)`, `waveform="gaussian"` is
  `WaveformGaussian(f_max)`;
- `mode_index` becomes the channel that is driven: the first argument
  of `set_excitation`, or `Excitation(port, mode=…)`, or the
  `(port, mode)` pair in `run(excited=…)`;
- the object itself becomes {class}`~magnelio.Excitation` at run
  level, which additionally carries the `amplitude`, `delay` and
  `phase` that `ExcitationSpec` had no field for.

```python
# 0.4.x
from magnelio.ports._modal import (
    BoxFace,
    ExcitationSpec,
    PortSpecRectWG,
    build_modal_port,
)

excitation = ExcitationSpec(f_min=F_MIN, f_max=F_MAX, mode_index=0)
```

```python
# 0.5
from magnelio.ports._modal import BoxFace, PortSpecRectWG, build_modal_port
from magnelio.signals import WaveformGaussianModulated

op_src.set_excitation(0, WaveformGaussianModulated(f_min=F_MIN, f_max=F_MAX))
```

`ExcitationSpec` was public and rendered into the API reference, but
it appeared in no example or tutorial: its real call sites are
hand-written expert scripts.

### Excitations on port specs

The `PortSpec*` classes lost their `excitation` field.  A spec is a
description of a port; a bound waveform is run-time state, and it is
set on the built operator:

```python
# 0.4.x
excitation = ExcitationSpec(f_min=F_MIN, f_max=F_MAX, mode_index=0)
spec_src = PortSpecRectWG(
    name="port1",
    plane=BoxFace.X_MIN,
    width_a=WR90_A,
    height_b=WR90_B,
    n_modes=1,
    excitation=excitation,
)
op_src = build_modal_port(spec_src, mesh, m_eps, m_mu, dt=dt, f_calc=F_CALC)
```

```python
# 0.5
spec_src = PortSpecRectWG(
    name="port1",
    plane=BoxFace.X_MIN,
    width_a=WR90_A,
    height_b=WR90_B,
    n_modes=1,
)
op_src = build_modal_port(spec_src, mesh, m_eps, m_mu, dt=dt, f_calc=F_CALC)
op_src.set_excitation(0, WaveformGaussianModulated(f_min=F_MIN, f_max=F_MAX))
```

The rewrite is mechanical because 0.4.x's builder already did exactly
this internally, one line below where it read the field.  The specs
are frozen dataclasses, so the stale keyword is a loud `TypeError` at
construction rather than an ignored field; a port that is never given
a waveform stays absorber-only, which is what a `None` excitation
always meant.  `PortSpecLumped` is unaffected — it never had the
field.

Two things are new rather than merely relocated: a second
`set_excitation` call on a *different* mode drives both modes at once
(a second call on the same mode replaces its waveform), and
`clear_excitation()` drops them all.  Neither was expressible with one
excitation field per spec.

### The waveform functions

`signals.gaussian`, `signals.modulated_gaussian` and
`signals.waveform_for_mode` are gone from `magnelio.signals`, replaced
by the waveform classes.  A bare callable cannot report its own
bandwidth, and that is what the run-length estimate and the band
warnings read.

This is the one rename on the page that need not fail loudly.  The
three functions still exist in the module that defines them, so
`from magnelio.signals.waveforms import modulated_gaussian` keeps
resolving and a script written that way keeps running — on a path
that is no longer part of the public surface.  Only the
`magnelio.signals` spelling raises.  Grep for `signals.waveforms` as
well as for the three names.

```python
# 0.4.x
from magnelio.signals.waveforms import modulated_gaussian

ref_values = np.array([float(modulated_gaussian(float(t), F_MAX, F_MIN)) for t in ref_t])
```

```python
# 0.5
from magnelio.signals import WaveformGaussianModulated

waveform = WaveformGaussianModulated(f_min=F_MIN, f_max=F_MAX)
ref_values = waveform(ref_t)          # scalar or array, same closed form
```

The closed form is unchanged: called on an array, the class returns
the old function's values bit for bit, and the scalar loop above
differs only in the last bit — `math.exp` against `np.exp`.  But
**the argument order flips**: the function was
`modulated_gaussian(t, f_max, f_min)`, with `f_max` first, and the
class is `WaveformGaussianModulated(f_min, f_max)`, with `f_min`
first.  A positional conversion swaps the band edges.  The class
validates `f_max > f_min`, so it usually raises rather than running
with a mirrored band — but pass the two by keyword and the question
does not arise.

`waveform_for_mode` chose between the two forms per mode.  That choice
is now what the analysis does by default, so the migration is usually
to delete the call:

```python
# Default: pass no waveform and let the analysis pick, per excited mode.
analysis = mio.AnalysisScatteringTD(mesh=mesh, ports=specs, f_min=8.2e9, f_max=12.4e9)

# Explicit, if you want to name the choice yourself:
f_lo = max(f_cutoff, f_min)
waveform = (
    signals.WaveformGaussianModulated(f_min=f_lo, f_max=f_max)
    if f_lo > 0.0
    else signals.WaveformGaussian(f_max=f_max)
)
```

The selection rule is preserved exactly: effective lower edge
`max(f_cutoff, f_min)`; a zero edge gives the DC-inclusive
`WaveformGaussian`, a positive edge the modulated Gaussian over
`[max(f_cutoff, f_min), f_max]`.  Note that the old signature took
`omega_c` in rad/s while the classes take frequencies in Hz — divide
by 2π.

Where an {class}`~magnelio.Excitation` is concerned a bare callable is
no longer accepted at all: `Excitation("p1", waveform=lambda t: 0.0)`
raises `TypeError: Excitation.waveform must be a
magnelio.signals.Waveform (or None); got function`.  Wrap it in
`signals.WaveformFunction(fn=…, f_max=…)`.  The operator-level
`set_excitation` still takes a plain callable.


## Monitors

Monitors adopted the library's naming grammar, and nothing about what
they measure changed.  Two rules cover all three changes.

A monitor's class name states what it records and in which domain, so
the far-field monitor — which has always accumulated a running DFT at
requested frequencies — is `MonitorFarFieldFrequency`, next to
`MonitorFieldFrequency` and `MonitorFieldTime`.  It is a pure rename:
`freqs`, `name`, `margin_cells` and the whole result API (`result`,
`plot_cut`, `plot_3d`, `renormalize`) are untouched.

```python
# 0.4.x
farfield = monitors.MonitorFarField(freqs=[f0], name="farfield")

# 0.5
farfield = monitors.MonitorFarFieldFrequency(freqs=[f0], name="farfield")

analysis = mio.AnalysisScatteringTD(mesh=mesh, f_min=f_min, monitors=(farfield,), verbose=False)
result = analysis.run(f_axis=f_axis, excited=["feed"])
pattern = farfield.result(f0)
```

And an axis-aligned plane is spelled as two arguments everywhere,
`normal="z"` and `position=5e-3`, never as a packed `(axis, position)`
pair.  That vocabulary was retired from sources and from the flux and
wall-loss monitors in one pass, so `plane=` and `reference_plane=` are
gone:

```python
# 0.4.x
flux = MonitorFluxTime(plane=("z", 5e-3), name="flux_z")

# 0.5
flux = MonitorFluxTime(normal="z", position=5e-3, name="flux_z")
```

```python
# 0.4.x
monitor = monitors.MonitorWallLoss(
    freqs=np.linspace(f_lo, f_hi, 9),
    reference_plane=("z", 5e-3),
    sigma=sigma_steel,
    bc_faces=("xmin", "xmax", "ymin", "ymax"),
)

# 0.5
monitor = monitors.MonitorWallLoss(
    freqs=np.linspace(f_lo, f_hi, 9),
    normal="z",
    position=5e-3,
    sigma=sigma_steel,
    bc_faces=("xmin", "xmax", "ymin", "ymax"),
)
```

Full-cross-section integration, node snapping, the symmetry factor,
`sibc`, `masked_faces`, `wall_overrides` and every recorded quantity
are as before.

A keyword call fails cleanly.  A *positional* call is the one to watch:
the old pair lands in the new first parameter and the diagnostic then
points at the value rather than at the renamed argument —
`MonitorFluxTime(("z", 5e-3), "flux_z")` raises
`ValueError: MonitorFluxTime normal must be 'x', 'y' or 'z'; got
('z', 0.005)`, and in `MonitorWallLoss` every later argument shifts by
one, so the conductivity ends up in `position`.  Post-run code that
reads `mon.plane` or `mon.reference_plane` also fails: those
attributes no longer exist.


## The analysis classes

### The waveform override

The scattering analysis takes a waveform instead of an
`ExcitationSpec`:

```python
# 0.4.x
excitation = ExcitationSpec(f_min=8.2e9, f_max=12.4e9)

analysis = AnalysisScatteringTD(
    mesh=mesh.with_boundary_conditions(_lateral_pec_bcs()),
    ports=_wr90_specs(),
    f_max=12.4e9,
    f_min=8.2e9,
    excitation=excitation,
    verbose=False,
)
```

```python
# 0.5
from magnelio.signals import WaveformGaussianModulated

analysis = AnalysisScatteringTD(
    mesh=mesh.with_boundary_conditions(_lateral_pec_bcs()),
    ports=_wr90_specs(),
    f_max=12.4e9,
    f_min=8.2e9,
    waveform=WaveformGaussianModulated(f_min=8.2e9, f_max=12.4e9),
    verbose=False,
)
```

The semantics are unchanged.  The argument is still an optional
override applied to every excited `(port, mode)`, and omitting it
still derives a waveform per excited mode from the analysis band —
Gaussian for TEM and lumped ports, modulated Gaussian above a mode's
cut-off.  The two mappings are exact, pulse for pulse:

| 0.4.x | 0.5 |
|---|---|
| `ExcitationSpec(f_min=A, f_max=B)` | `WaveformGaussianModulated(f_min=A, f_max=B)` |
| `ExcitationSpec(f_min=A, f_max=B, waveform="gaussian")` | `WaveformGaussian(f_max=B)` |

`ExcitationSpec.mode_index` had no effect here in 0.4.x either — the
override always applied to every excited mode — so nothing is lost by
its absence.  Which channels are excited is still `run(excited=[…])`,
unchanged.

Two guards are new: the value must be a
{class}`~magnelio.signals.Waveform` instance, and a waveform whose
`f_max` reaches above the analysis band now warns — *"waveform f_max =
2e+10 Hz exceeds the analysis band f_max = 1.24e+10 Hz: the pulse
carries energy the grid and the frequency axis do not resolve."*

Like `ExcitationSpec` itself, `excitation=` appears in no 0.4.x
tutorial or example.  This one bites hand-written expert scripts, not
code that followed the published pages.

### What is new rather than changed

{class}`~magnelio.AnalysisTD` — one march under a list of
*simultaneous* excitations — is new, and `AnalysisScatteringTD` is now
a subclass of it running on the same engine.  Its own API is unchanged
apart from `waveform=`.  Also new: `Project.result(name)` rebuilds the
`TDResult` of any run, a scattering channel run included, and the run
names of a scattering project are still the 0.4.x `"port1_mode0"`
form.  {func}`~magnelio.resume` keeps its exact 0.4.x signature; only
the meaning of its second argument widened, from a `(port, mode)` pair
to that *or* a run name.  Nothing in existing `resume` calls changes.

One footnote, because the new vocabulary invites the confusion:
`AnalysisTD.run()` takes `excitations=[…]` — drives applied together —
while `AnalysisScatteringTD.run()` takes `excited=[…]` — channels, one
independent run each.  These are different things, not two spellings
of one; `excited=` was never renamed.  The scattering analysis rejects
the wrong one with a message saying which it wants.

Driving several modes of one port in the same run is likewise new
capability, not a change; existing single-mode code is unaffected.  It
is worth knowing about here for one reason: it is why a port
checkpoint now holds one source-history buffer per excited mode, which
is part of what the store bump below is.


## Saved projects must be re-run

This is the one change that bites without touching a line of your
code.

**A project store written by 0.4.x cannot be opened by 0.5, and there
is no in-place migration.**  The store format went from schema 1.0 to
2.0: `results.h5` now names a run by its excitations, `mesh.h5`
records the mesh's element type, and port checkpoints hold one
source-history buffer per excited mode.  There is no converter, and
the recipe readers that used to tolerate the pre-0.5 spellings were
retired in the same bump.

What you see is a hard, complete stop the first time you touch project
metadata:

```text
magnelio.io._schema.ProjectSchemaError: wr90_run/project.json: schema
version '1.0' is not supported (current: '2.0'). This store was
written by another magnelio release — re-run the simulation to
regenerate it.
```

Note *when* it arrives.  {func}`~magnelio.open_project` itself
succeeds — it only checks that `project.json` exists, and the metadata
is read lazily — so the error comes one line later, at whatever first
reads the store: `project.status`, `project.setup`, `project.runs`,
`project.s_params`, `project.result(name)`,
{func}`~magnelio.resume`, and `repr(project)`, so even typing the bare
name at a REPL prompt raises.

One part survives, and it is the useful part: `project.mesh`,
`project.grid` and `project.geometry` still read.  `mesh.h5` carries
no schema stamp and is not gated, so the *model* of an old project can
be recovered and re-run without meshing it again.

Do not work around the gate by editing the version stamp.  Forcing
`project.json` and the two HDF5 files to `"2.0"` makes a plain modal
scattering project appear to read back — status, runs, S-parameters
all return values — and that is exactly what makes it dangerous.  The
stored 1.0 recipe key `excitation` is not read by the 2.0 recipe
reader, so the rebuilt analysis silently falls back to the
auto-derived per-mode waveform, without a warning.  A resume from that
state splices a different pulse onto the recorded one.  Re-running the
analysis is the only supported path, and it is what the error message
asks for.


## Changes that are not API breaks

Three changes move numbers slightly without renaming anything.  They
are listed here so a result that shifts after the upgrade has an
explanation.

**The auto-sized run length now follows the pulse.**  The estimate for
a band-limited drive is the drive's own duration (`waveform.t_end`,
plus any delay) instead of a fixed `8 / f_max`.  A TE- or TM-fed
scattering run therefore checks its stop criterion on a slightly
different cadence and may stop a few steps later than before.  On the
reference waveguides the in-band `|S|` changes by less than 5·10⁻⁴.
TEM and lumped-port runs are unchanged.

**The default total-field box of a plane wave moved.**  Where a box
side is not given, it now falls two bulk cells inside the *physical*
domain — past the absorber cells the mesher appends — rather than
being counted from the edge of the padded grid.  This applies per
side, so it also reaches a `corners=` pair with `None` or infinite
components.  A box stated in full is unaffected.

**A fallback channel no longer reports a chain floor.**
`analysis.solve_ports()[name].modes[i].chain_floor_db` is a property
of the exact transparent termination, so it is now `None` on a channel
that fell back to the first-order absorber, where it used to read
around −13 to −10 dB.  That number was neither the reflection floor of
that channel nor a bound on anything, and code that plotted or
compared it will now find `None` there.  What such a channel does
publish is its cross-section measurement, `chain_spread`.
