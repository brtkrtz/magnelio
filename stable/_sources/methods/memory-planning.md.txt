# Memory planning

The mesh is only part of a time-domain simulation's memory requirement.
Analysis setup constructs material operators and estimates a stable time
step before allocating the fields. Temporary setup arrays can exceed the
memory held during time integration.

`AnalysisTD.estimate_memory()` and `AnalysisScatteringTD.estimate_memory()` report
allocation budgets without building material matrices, solving ports,
computing a CFL eigenvalue, attaching monitors or creating project files:

```python
estimate = analysis.estimate_memory(excited=["input"])
print(estimate)
```

`excited=` applies to scattering analyses. General TD analyses represent
one run and omit it. The returned `MemoryEstimate` exposes byte counts
through `mesh_bytes`, `fields_bytes`, `coefficients_bytes`, `phases` and
`monitors`. Printed units are binary (`GiB = 2³⁰ bytes`,
`MiB = 2²⁰ bytes`, `KiB = 2¹⁰ bytes`), matching tools such as `free -h`.

## Phase budgets

Each phase states simultaneous RAM and, for a GPU scenario, VRAM storage.
RAM numbers include the mesh already in memory. Do not sum the phase
budgets or add the mesh again. Sequential scattering runs reuse solver
storage; raw monitor output on disk is multiplied by the number of
excited channels. Results retained by multiple in-memory runs can increase
RAM and leave the upper budget unresolved.

Field storage follows the staggered Yee shapes and requested precision,
including the `MAGNELIO_PRECISION` default. CFL and material operator
calculations remain in double precision. The CFL budget includes the
current explicit sparse curl construction, its Python index lists,
conversion temporaries, the absolute-value matrix copy and Lanczos vectors.
A cached CFL eigenvalue removes the corresponding construction budget.

Ranges model allocation paths in the current implementation, not measured
RSS or guaranteed process-memory limits. Conservative workspaces cover
material/operator construction, field updates and CPML. CAD/native objects,
Python metadata, allocator pools and checkpoint/I/O workspaces are excluded.
Mesh arrays and views sharing a backing allocation are counted once.

Port/element mode construction, sparse factorisation and convolution
histories, source state, dispersive/SIBC auxiliary states and monitor
types other than rectangular time/frequency field monitors currently
have no allocation model. When present, they are listed explicitly and
the upper budget is `None`; known phase budgets still identify costly
setup stages. `peak_ram_bytes` and `peak_vram_bytes` are consequently
optional; `minimum_peak_ram_bytes` exposes the largest modelled lower
endpoint. A missing upper endpoint must not be treated as zero.

An unresolved `backend="auto"` uses a CPU scenario without probing CUDA.
Use `estimate(backend="cupy")` to budget the GPU scenario independently.
This does not change the backend used by `run()`.

## Time monitor recordings

Every rectangular time monitor reports `bytes_per_snapshot` for the
requested raw staggered components. A slice, line or plane therefore
uses its own component shapes, including boundary samples.

| Schedule | Recording budget per run |
| --- | --- |
| `times=[...]` | At most the requested number of snapshots |
| `interval=...`, finite step bound and known `dt` | Snapshot count through that bound |
| `interval=...`, finite step bound but unknown `dt` | At most one snapshot per step |
| `interval=...`, no resolved bound | Unknown total, plus bytes per simulated second |

With a known time step the estimator also coalesces explicit times that
would be consumed in the same solver step. Early termination can reduce
the recording below every stated count. For interval recording the growth
rate uses the requested schedule; when the time step is known, it is
capped at one snapshot per step.

Supply the same explicit limits intended for the run:

```python
estimate = analysis.estimate_memory(total_time_steps=200_000)
estimate = analysis.estimate_memory(max_time_steps=500_000)
```

The automatic runtime cap depends on prepared ports and waveforms. The
estimator leaves it unresolved instead of performing that setup. A physical
duration requires the step to account for rounding and the monitor's
half-step acceptance window:

```python
estimate = analysis.estimate_memory(t_end=20e-9, dt=1e-12)
```

`dt=` is an assumed scenario, not a solver override. If omitted, a cached
CFL eigenvalue supplies the normal-accuracy step; otherwise no new
eigenvalue is computed. Without a known step, a duration alone cannot
certify the number of rounded recordings. `t_end` is a run option on
`AnalysisTD`; on a scattering estimate it is only a planning scenario.

Without `project=`, time snapshots accumulate in RAM, including a possible
second copy when frames are stacked. With `project=`, the raw total
belongs to disk storage; RAM holds the pending batch and its stacking
copy, at most 100 snapshots between the current solver's flushes.
`monitor_disk_bytes` is the raw field-monitor payload across all runs,
excluding geometry, mesh files, port signals, checkpoints and HDF5 overhead.

Frequency monitors hold one complex128 array per recorded component and
frequency, independently of run length and field precision. They continue
to need those accumulators in RAM on a project-backed run.

The returned `MemoryEstimate.n_runs` counts sequential runs; collections of
actual runs remain `runs`. The method is `estimate_memory()` on both
`AnalysisTD` and `AnalysisScatteringTD`. Its `dt`, `backend` and `t_end`
assumptions describe an estimate scenario and do not configure a later run.
