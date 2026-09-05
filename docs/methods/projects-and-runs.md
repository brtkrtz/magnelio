# Projects and runs

A simulation that finishes in a minute can live in memory.  One that
takes an afternoon should not: the process may be a notebook kernel
that gets restarted, the results are wanted in another session or on
another machine, and someone will want to look at the run while it is
still marching.  Magnelio's answer is the **project**: a directory the
analysis writes into as it goes, readable by any other process at any
time, and resumable from its last checkpoint.

```python
proj = mio.AnalysisScatteringTD(mesh=mesh, project="magic_tee").run(excited=["port3", "port4"])
```

Given a `project=` directory, `run()` streams everything into it and
returns the reader of that directory — the same object
{func}`~magnelio.open_project` returns, so a post-processing script
and the return value of `run()` are one and the same thing:

```python
proj = mio.open_project("magic_tee")
```

This chapter is the vocabulary of that directory: what a project and a
run are, the states they pass through, what the clock on them means,
and how to read one that is still being written.


## Vocabulary

**Project.**  A directory holding `project.json` (the index: setup,
status, the run table), the mesh, the geometry when one was given, and
one sub-directory per run.  The handle is a
{class}`~magnelio.io.Project`.  It is also a scattering result — `S`,
`db`, `plot_s`, the Touchstone export — so a script written for the
in-RAM result runs unchanged against it.

**Run.**  One march of the time-domain solver.  A scattering analysis
makes one run per excited channel, named after it (`port3_mode0`); the
general time-domain analysis names its runs `run_1`, `run_2`, … or
whatever `name=` it was given.  The handle is a
{class}`~magnelio.io.Run`, handed out by `proj.runs[name]`; it carries
the run's state, its step count, the stop criteria and the reason the
marching ended, its clock, its energy trace, and gives access to the
run's result and monitors.

**Run states.**  A run is `pending` from the moment the analysis plans
it (no directory on disk yet), `running` while the solver marches, and
ends `done` or `aborted`.  A fifth reading, `stale`, is not stored: it
is `running` on disk with nobody writing — the solver process the run
names no longer exists on this host, because the kernel died or the
machine was rebooted.  Resume the run, or run the analysis again.

**Project status.**  `created` before the first run, `running` while
any run is planned or marching, then `done` when every planned run is
done, `aborted` when one ended on a graceful stop or an error (the
analysis call goes with it, so its planned siblings never start), and
`stale` under the same rule as for a run.


## Reading a project

Typing the project's name at a prompt, or leaving it as the last
expression of a notebook cell, prints its state and a table of its
runs:

```
Project wr90_demo
  analysis   AnalysisScatteringTD
  status     done
  last call  finished 2026-09-05 08:55:15 in 0.5 s
  created    2026-09-05 08:55:15
  runs       2
run       excited  state  steps    energy  elapsed  stop reason
────────  ───────  ─────  ─────  ────────  ───────  ───────────
p1_mode0  p1:0     done     701  -47.0 dB    0.2 s  energy
p2_mode0  p2:0     done     701  -47.0 dB    0.2 s  energy
```

(A WR-90 section with two ports, excited from each side in turn and
stopped 40 dB below the energy peak — a small run, so the clocks are
short.)  `proj.runs` is a read-only mapping of run names to run
objects; each run prints as a summary of its own, and its attributes
are the columns of the table and more:

```python
run = proj.runs["p1_mode0"]
run.state, run.n_steps, run.stop_reason      # 'done', 701, 'energy'
run.energy_db                                # the last energy sample, dB below the peak
run.energy_trace                             # the whole trace: step, time, energy
run.result()                                 # the run as a TDResult
run.monitors                                 # the run's monitors by name
```

```
Run 'p1_mode0'
  state        done
  excited      p1:0
  steps        701
  stops at     energy -40 dB or port signal -60 dB
  energy       -47.0 dB below peak
  stop reason  energy
  started      2026-09-05 08:55:15
  finished     2026-09-05 08:55:15
  elapsed      0.2 s
  dt           2.154e-12
  checkpoint   yes
```

The S-parameters of a scattering project come off the project itself,
derived from the stored port signals: `proj.S("port1", "port3")`,
`proj.plot_s(...)`.  The general time-domain analysis has no S-matrix;
its runs are read through `run.result()`.


## A project that is still being written

Every energy sample the solver takes goes to disk at once, and the run
index is replaced atomically whenever a run starts or ends, so a
second process may open the project while the solver marches and see
its current state.  Until the project is finished, the reader re-reads
the index whenever the file changed — typing `proj` again shows the
current step count, energy and elapsed time without any call on your
side.  Once the project is `done` or `aborted` the parsed index is
kept; {meth}`~magnelio.io.Project.refresh` re-reads it, for a run that
was resumed elsewhere after you opened the project.

A `running` run's `elapsed` counts the time since its current march
started on top of what earlier marches booked, so the clock in the
table keeps moving while the solver does.


## Time

Every run carries its wall clock: `started` and `finished` (UTC), and
`elapsed`, the wall time of the *marching* — summed over the marches
of a resumed run.  The same three numbers are on every result object
and on the project, where they span its runs.  The analysis call that
produced the runs, setup included, is a separate figure: the
`finished in` line the call prints, and the *last call* entry at the
top of the project's summary.  The two differ by the setup — the
stability time step and the port mode solves — which the progress
output accounts for phase by phase.


## Checkpoints and resume

A streamed run writes a resume checkpoint about eight times over its
expected length, once more when it finishes, and once on a graceful
stop (Ctrl-C).  {func}`~magnelio.resume` continues a run from its last
checkpoint — to finish an aborted one, or to march a finished one
further under a deeper stop criterion — and appends to the same
streams, bit-exact with an uninterrupted run of the same length.

`proj.checkpoint_state(name)` (or `run.checkpoint_state()`) reads the
checkpoint back as a {class}`~magnelio.io.CheckpointState`: a
read-only mapping with the solver's own state layout — the completed
step, the peak energy and port signal, the field vectors, and a group
per boundary, port and monitor.  Printing it shows the step and the
sizes, not the field vectors.


## What prints

A result, a run, a project or a checkpoint answers a bare name at the
prompt with what it is, how large it is and what state it is in — never
with its arrays.  In a notebook the same summaries render as tables.
The arrays are one attribute away (`result.energy_trace`,
`s_params.matrix`); the summary is the part meant to be read.
