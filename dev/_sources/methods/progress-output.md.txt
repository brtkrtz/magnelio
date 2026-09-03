# Progress output

A full-wave run spends most of its wall time before the first field is
updated: the mesh is built, the stability time step is measured, and
every port's mode problem is solved.  On a two-million-cell model that
is well over half a minute of setup.  Magnelio reports each of those
phases as it runs, so that a long-running call can be told apart from a
stalled one.

## The switch

Every long-running operation takes a `verbose` argument, and the
process-wide default behind it is set once:

```python
import magnelio as mio

mio.set_verbosity(False)      # a batch sweep: no output at all
mio.set_verbosity(True)       # the default
```

The argument overrides the default for one call:

```python
mesh = mio.Mesh.from_geometry(model, control, f_max=12e9, verbose=False)
analysis = mio.AnalysisScatteringTD(mesh=mesh, verbose=True)
```

Leaving `verbose` unset (`None`) means *follow the process-wide
setting*.  That is what makes the setting reach nested work: a port
refinement builds a mesh and solves ports once per rung, and those
inner calls inherit your setting rather than being silenced.

## What the phases mean

Output lines are prefixed with the operation reporting them.

```
  mesh | materials | done (1.6 s)
  mesh | conformal cells | 795 sections
  mesh | conformal cells | done (20.2 s)
  mesh | PEC masks | done (0.5 s)
  mesh | 131 x 122 x 130 cells (22.4 s total)
```

`mesh` follows the mesh build: locating the planes the geometry
demands, generating the grid lines, filling material identities from
cross-sections, classifying partially filled cells, and building the
conductor masks.  The two phases missing from the listing above —
finding the feature planes and generating the grid lines — finished
too quickly to be worth a line; see *Short phases* below.

On a model with curved or free-form surfaces the **conformal cell
classification dominates**, and by a wide margin: the run above is a
reflector antenna where it takes 20 of the 22 seconds, because every
partially filled cell needs its own cross-section through a curved
face.  A running count of those cross-sections appears while it works.
It is a count and not a percentage on purpose — the phase makes
several passes for different material properties, and a percentage
would run to 100 and start over, which reads as a stall followed by a
restart.

```
  setup | CFL eigenvalue | done (14.1 s)
  setup | port 'feed' | done (8.6 s)
```

`setup` covers the work between a finished mesh and the first time
step, whether it runs inside `run` or on its own through
`solve_ports`.  **The CFL
eigenvalue is often the surprise here.**  Magnelio measures the exact
stability limit with an iteration over the whole update operator rather
than estimating it from the cell sizes, which buys a time step several
times larger than the geometric estimate — but it is proportional to
the size of the model, not to the number of ports.  A port report on a
large model therefore takes noticeably longer than the ports alone
would suggest.  The measured value is cached on the mesh, so inspecting
the ports first and running afterwards pays for it only once:

```python
analysis = mio.AnalysisScatteringTD(mesh=mesh, f_min=8.5e9)
print(analysis.solve_ports()["feed"])     # pays the eigenvalue
result = analysis.run(f_axis=f_axis)      # does not pay it again
```

```
  FIT-TD | time step 2701/∞ | stored energy [dB] -70.0/-70 | done (energy criterion)
```

`FIT-TD` is the time loop.  While it marches, the line reports the step
count and the quantity the active stop criterion watches — stored
energy in dB below the run peak, or the port-signal envelope — so the
distance to the finish is visible even on an open-ended run.

```
  eigen | factorising at sigma=2.012e+21 (7.14 GHz) | done (14.0 s)
  eigen | eigensolve at sigma=2.012e+21 (7.14 GHz) | done (2.7 s)
```

The shift is reported both as the solver holds it — an eigenvalue of
the curl-curl operator, so `(2*pi*f)**2` — and as the frequency it
targets, which is the number to check against your model.

`eigen` is the cavity eigenmode solver.  It reports phases rather than
a percentage, because an eigensolver converges when it converges — but
the phases are informative on their own: the factorisation of the
shifted operator regularly costs several times the iteration that
follows it, and it scales with the mesh, so a slow eigenmode run is
usually asking for a coarser grid rather than more patience.

```
  refine | level 1/1: meshing and solving the port slab
  refine | level 1: 684 plane cells, f_cutoff = 6.55527e+09  Δ +0.085 %
```

`refine` is the port-plane convergence ladder.  Each rung is announced
before it runs and reported with its value and change afterwards, and
the mesh build and port solve of the rung in progress report inside it.

## Short phases

A phase that finishes in under half a second does not report that it
finished.  Nothing is gained by learning that five separate steps each
took no measurable time, and a small model — every tutorial, most
tests — would otherwise print a wall of `done (0.0 s)` lines that say
nothing.  A small mesh build therefore reports only its result:

```
  mesh | 14 x 7 x 4 cells
```

The closing line of an operation carries its total wall time whenever
that total is itself worth reporting, so a build with several long
phases needs no addition in your head.

## Terminals, logs and notebooks

On a terminal, progress is one line that updates in place.  Everywhere
else — a log file, a CI job, a captured pipe, a Jupyter notebook —
Magnelio writes whole lines at a slow cadence instead, because
overwriting depends on carriage returns that a log file records
literally and concatenates into one unreadable row.  You do not
configure this; it follows from where the output is going.

Two consequences worth knowing:

* Redirecting a run to a file gives you a readable record, not a
  transcript of every refresh.
* In a notebook you see a phase announced when it begins and again when
  it ends, rather than a single updating line.

Work running inside worker processes stays silent.  Mesh sectioning and
the band port kernel are computed by process pools, and every worker
reporting to one terminal would interleave into noise; only the parent
process reports.
