# Upgrading from 0.5.x

Release 0.6 changes one thing a 0.5.x script can trip over: the run
index of a project store hands out run *objects* instead of the raw
dictionaries of `project.json`.  Everything else in the release is
additive — new attributes, new lines of progress output, new pages.

## Quick reference

| 0.5.x | 0.6 | If you leave it alone |
|---|---|---|
| `proj.runs["port1_mode0"]["n_steps"]` | `proj.runs["port1_mode0"].n_steps` | `TypeError: 'Run' object is not subscriptable` |
| `proj.runs[name]["state"]`, `["stop_reason"]`, `["dt"]`, … | the attribute of the same name | `TypeError` |
| `proj.runs[name]["excited"]` → `["port1", 0]` (a list) | `.excited` → `("port1", 0)` (a tuple) | `TypeError` |
| `proj.runs[name]["excitations"]` → `[["p1", 0], …]` | `.excitations` → `(("p1", 0), …)` | `TypeError` |
| `for name, info in proj.runs.items(): info["state"]` | `for name, run in proj.runs.items(): run.state` | `TypeError` |
| `list(proj.runs)`, `len(proj.runs)`, `name in proj.runs` | unchanged | — |
| `proj.checkpoint_state(name)` → `dict` | a read-only mapping, indexed the same way | — |
| `proj.status` ∈ `created / running / done` | may also read `aborted` or `stale` | a check for `!= "done"` keeps working |

The raw dictionary is still there for anyone who needs it —
`proj.meta["runs"][name]` — but every field it holds has an attribute
on the run object, with the types a script wants (tuples for channel
keys, `datetime` for the stamps, floats for the durations).

## Why

The dictionary was the store's own bookkeeping leaking through: its
keys were whatever the writer had needed, its channel keys were lists
because JSON has no tuples, and it knew nothing a user could not read
off `project.json` with a text editor.  A run object knows more — its
energy trace and the latest energy level below the peak, its wall
clock (still moving while the run marches), whether its writer process
is still alive — and prints as a summary rather than a dump.  See
*Projects and runs* in the technical description.

## Also new, nothing to change

* Every result carries `started`, `finished` and `elapsed`; a project
  books them per run and for the analysis call.
* The time loop's progress line reports a running clock and the step
  rate; two header lines say what runs and what ends it; every call
  closes with `finished in …`.
* `ScatteringTDResult` keeps the energy trace of every excitation
  (`energy_traces`).
* Results, runs, projects, S-parameter matrices and checkpoints print
  as short summaries, and as tables in a notebook.
* `plot_energy()` on results, runs and projects; `Project.watch()`,
  `Project.follow()` and `Project.monitor()` to follow a project
  another process is writing — see *Projects and runs* and the how-to
  *Watching a simulation that is still running*.
