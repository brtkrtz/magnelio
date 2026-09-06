# Upgrading from 0.6.x

Release 0.7 changes what a field monitor hands out.  A time or
frequency monitor used to average the solver's staggered samples onto
cell centres the moment it recorded them and to return those averages
as a dictionary of arrays.  It now keeps the solver's own samples — the
grid quantities on the Yee positions of its region — and hands them out
as a field container; the averages are one call away and computed only
where you ask for them.  The project store changed with it, so runs
written by 0.6.x cannot be read by 0.7 and are refused with a message.

## Quick reference

| 0.6.x | 0.7 | If you leave it alone |
|---|---|---|
| `mon.data["Ez"]` (time monitor) | `mon.recording.cell_centred(["Ez"], squeeze=True)["Ez"]` | `AttributeError: 'MonitorFieldTime' object has no attribute 'data'` |
| `mon.data["Ez"]` (frequency monitor) | `mon.spectrum.cell_centred(["Ez"], squeeze=True)["Ez"]` | `AttributeError` |
| `mon.data_raw["Ez"]` | `mon.spectrum_raw.cell_centred(["Ez"], squeeze=True)["Ez"]` | `AttributeError` |
| `mon.component("Ez")` | `mon.recording.cell_centred(["Ez"], squeeze=True)["Ez"]`, or `mon.recording.component("Ez")` for the staggered samples | `AttributeError` |
| `mon.region.xc`, `.yc`, `.zc` | `mon.recording.cell_centres[0]`, `[1]`, `[2]` (`.spectrum.cell_centres` for a frequency monitor) | `AttributeError` |
| `mon.region.ndim` | `sum(n > 1 for n in mon.recording.shape)` | `AttributeError` |
| `mon.t` | unchanged — but see below | — |
| `project.monitors[...]` readers: `.data`, `.component(...)` | `.recording` / `.spectrum` as above; `.frame(i)` reads one frame | `AttributeError` |
| `mon.plot(...)`, `mon.interact(...)`, `mon.show(...)` | unchanged | — |
| `runs/<name>/fields.xdmf` | `runs/<name>/paraview/<monitor>.pvd` with one `.vtr` per frame | the XDMF file is not written |
| a project written by 0.6.x | re-run it | `ProjectSchemaError: schema version '2.0' is not supported` |

`squeeze=True` drops the spatial axes of length one, which is what the
old dictionaries did on their own: a plane monitor reads
`(n_frames, nu, nv)`, a line `(n_frames, n)`, a point `(n_frames,)`.
Without it every array is `(n_frames, nx, ny, nz)`.

## What the containers give you

`FieldRecording` (time) and `FieldSpectrum` (frequency) live in
`magnelio.fields` and are described in the chapter
{doc}`methods/sources-monitors`.  In short:

- `frame(i)`, `at_time(t)`, `at_frequency(f)` — one instant as a
  `FieldState`, the same object an eigenmode hands out, so a recorded
  frame goes straight into `SourceFieldInitial` for a restart or a
  ring-down.
- `component(name)` — the staggered samples themselves,
  `(n_frames, *Yee shape)`, E in V/m and H in A/m.
- `cell_centred(...)`, `cell_centres` — the averages and their
  positions, for whole frames or a sub-box.
- `times`, `times_h` — the instants of the electric and of the magnetic
  field; the latter lies half a time step later, which the old
  dictionary silently ignored.
- `plot(...)`, `show(...)` — the slice plots and the 3D view, frame by
  frame.

## Two numbers that moved

**The instant of a frame.**  The solver calls the monitors after it
has stepped the electric field to `t + dt`; a 0.6.x time monitor
labelled that frame `t`, one step early.  A frame is now stamped with
the electric field's own instant, and `times_h` states the magnetic
one.  A request at `t = 0` is served by the first frame the solver
hands out, at `dt` — the field at `t = 0` is identically zero and never
recorded.  A recording schedule finer than the time step yields one
frame per step, where 0.6.x repeated the same snapshot under several
labels.

**The dual widths of a region.**  The samples of `H` at the two boundary
nodes of a region cut out of the grid carry the dual lengths of the
*full* grid; a container built from such a region keeps them, so the
cell-centred averages are what they were.  You will not see a change
here; it is what made the first point possible without one.

The frequency monitor's phase convention is unchanged: its transform
uses the same time stamps as the port recorder, so a renormalised
pattern is phase-consistent with the run's S-parameters, as before.

## Why

The record-time average was a filter, and a filter is not undone.  It
smeared the tangential field on a conductor face and the normal field
across a dielectric interface into the neighbouring cells; it left no
way to compute the stored energy or the Poynting flux from a recording,
both of which are identities on the staggered samples; and it made a
recorded frame unusable as the initial field of another run.  Keeping
the solver's own samples removes all three limits at once, and it is
what the commercial suites do as well — store the solver's result,
interpolate for the picture.  The averaging still exists, exactly as it
was; it moved from the recording to the reading.
