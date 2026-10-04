"""Memory budgets for TD analyses without constructing solver operators."""

from __future__ import annotations

import math
import os
import struct
import sys
from dataclasses import dataclass

import numpy as np

from magnelio._backend.array_api import resolve_precision
from magnelio._memory import array_bytes, format_bytes
from magnelio.fields.state import _yee_shapes
from magnelio.monitors.base import _expand_field_list, resolve_region
from magnelio.monitors.field_frequency import MonitorFieldFrequency
from magnelio.monitors.field_time import MonitorFieldTime


@dataclass(frozen=True)
class MemoryPhase:
    """Estimated simultaneous array storage for one execution phase.

    Attributes
    ----------
    name : str
        Execution phase.
    ram_lower_bytes, ram_upper_bytes : int or None
        Host-memory range, including the mesh. An absent upper endpoint
        means an unbounded recording or an unmodelled allocation.
    vram_lower_bytes, vram_upper_bytes : int or None
        Device-memory range, independent of host RAM.
    """

    name: str
    ram_lower_bytes: int
    ram_upper_bytes: int | None
    vram_lower_bytes: int = 0
    vram_upper_bytes: int | None = 0


@dataclass(frozen=True)
class MonitorMemoryEstimate:
    """One field monitor's recording and resident-storage budget.

    Attributes
    ----------
    name : str
        Monitor name.
    bytes_per_snapshot : int or None
        Raw Yee samples of the requested components, excluding metadata.
        Frequency monitors report one complex frequency bin instead.
    max_snapshots : int or None
        Upper bound on recorded frames, or number of frequency bins.
        Time schedules finer than a step are coalesced by the solver.
    recording_bytes : int or None
        Upper bound on raw recording payload per run. ``None`` means
        the run horizon is unknown.
    ram_lower_bytes, ram_upper_bytes : int or None
        Additional resident storage including buffering and operators.
    disk_bytes : int or None
        Raw recording payload per run with ``project=``; zero otherwise.
        HDF5 metadata, checkpoints and compression are not included.
    bytes_per_second : float or None
        Interval-form recording growth per simulated second. With a known
        step, the schedule rate is capped at one snapshot per time step.
    kind : {"time", "frequency", "unknown"}
        Recording type, distinguishing time frames from frequency bins.
    """

    name: str
    bytes_per_snapshot: int | None
    max_snapshots: int | None
    recording_bytes: int | None
    ram_lower_bytes: int
    ram_upper_bytes: int | None
    disk_bytes: int | None
    bytes_per_second: float | None = None
    kind: str = "time"


@dataclass(frozen=True)
class MemoryEstimate:
    """Structured storage budget returned by a TD analysis's ``estimate``.

    Attributes
    ----------
    mesh_bytes : int
        Distinct NumPy backing allocations already retained by the mesh.
    fields_bytes, coefficients_bytes : int
        Six field arrays and six update/energy diagonals in solver precision.
    backend, precision : str
        Backend scenario and resolved scalar dtype.
    runs : int
        Sequential runs. Solver storage is reused; disk recordings scale
        with this count.
    phases : tuple of MemoryPhase
        Simultaneous host and device budgets by phase, not summed phases.
    monitors : tuple of MonitorMemoryEstimate
        Field monitor budgets per run.
    notes : tuple of str
        Assumptions, missing contributions and unresolved run limits.
    """

    mesh_bytes: int
    fields_bytes: int
    coefficients_bytes: int
    backend: str
    precision: str
    runs: int
    phases: tuple[MemoryPhase, ...]
    monitors: tuple[MonitorMemoryEstimate, ...]
    notes: tuple[str, ...]

    @property
    def peak_ram_bytes(self) -> int | None:
        """Estimated host peak, or ``None`` when an upper budget is missing."""
        return _upper_peak(self.phases, "ram_upper_bytes")

    @property
    def peak_vram_bytes(self) -> int | None:
        """Estimated device peak, or ``None`` when a contribution is unknown."""
        return _upper_peak(self.phases, "vram_upper_bytes")

    @property
    def minimum_peak_ram_bytes(self) -> int:
        """Largest lower endpoint among the host phase budgets."""
        return max(p.ram_lower_bytes for p in self.phases)

    @property
    def monitor_disk_bytes(self) -> int | None:
        """Raw field-monitor payload for all runs, excluding other project files."""
        if any(m.disk_bytes is None for m in self.monitors):
            return None
        return self.runs * sum(m.disk_bytes for m in self.monitors)

    def __str__(self) -> str:
        def span(lo, hi):
            if hi is None:
                return f">= {format_bytes(lo)} (upper unknown)"
            if lo == hi:
                return format_bytes(lo)
            return f"{format_bytes(lo)} - {format_bytes(hi)}"

        lines = [
            f"Memory estimate | {self.backend} | {self.precision} | {self.runs} run(s)",
            f"  mesh already held: {format_bytes(self.mesh_bytes)}",
            f"  fields: {format_bytes(self.fields_bytes)}; "
            f"coefficients: {format_bytes(self.coefficients_bytes)}",
        ]
        for phase in self.phases:
            line = f"  {phase.name}: RAM {span(phase.ram_lower_bytes, phase.ram_upper_bytes)}"
            if self.backend == "cupy":
                line += f" | VRAM {span(phase.vram_lower_bytes, phase.vram_upper_bytes)}"
            lines.append(line)
        for mon in self.monitors:
            count = "unknown" if mon.max_snapshots is None else f"<= {mon.max_snapshots:,}"
            payload = (
                "unknown" if mon.recording_bytes is None else format_bytes(mon.recording_bytes)
            )
            frame = (
                "unknown"
                if mon.bytes_per_snapshot is None
                else format_bytes(mon.bytes_per_snapshot)
            )
            unit = "bin" if mon.kind == "frequency" else "frame"
            lines.append(
                f"  monitor {mon.name}: {frame}/{unit} | {unit}s {count} | recording/run {payload}"
            )
            disk = "unknown" if mon.disk_bytes is None else format_bytes(mon.disk_bytes)
            lines.append(
                f"    RAM {span(mon.ram_lower_bytes, mon.ram_upper_bytes)} | disk/run {disk}"
            )
            if mon.bytes_per_second is not None:
                lines.append(
                    f"    recording growth: {mon.bytes_per_second / 2**30:.3g} GiB/s simulated"
                )
        disk = self.monitor_disk_bytes
        lines.append(
            "  field-monitor disk payload, all runs: "
            + ("unknown" if disk is None else format_bytes(disk))
        )
        lines.extend(f"  Note: {note}" for note in self.notes)
        return "\n".join(lines)


def _upper_peak(phases, attribute):
    values = [getattr(p, attribute) for p in phases]
    return None if any(v is None for v in values) else max(values)


def _positive(value, name, *, integer=False):
    if value is None:
        return None
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    if integer and int(value) != value:
        raise ValueError(f"{name} must be an integer")
    return int(value) if integer else float(value)


def _frame_count(mon, steps, dt, horizon):
    # Match the target acceptance window in MonitorFieldTime.record.
    if steps is not None and dt is not None:
        horizon = (steps + 0.5) * dt
    elif horizon is not None and dt is not None:
        steps = max(1, int(math.ceil(horizon / dt - 1e-9)))
        horizon = (steps + 0.5) * dt
    elif horizon is not None:
        # t_end rounds up to a step, and the monitor accepts targets half
        # a step ahead. Without dt, its duration alone cannot certify a
        # schedule count. Keep the unconditional explicit-times bound.
        horizon = None
    if mon.times is not None:
        if dt is None:
            return len(mon.times) if steps is None else min(len(mon.times), steps)
        last_step = 0
        count = 0
        for target in mon.times:
            step = max(1, int(math.ceil(float(target) / dt - 0.5)))
            if horizon is not None and target > horizon:
                break
            if step != last_step:
                count += 1
                last_step = step
        return count
    if horizon is None:
        return steps  # one frame per step is an unconditional upper bound
    count = max(0, int(math.floor((horizon - mon.start) / mon.interval)) + 1)
    return count if steps is None else min(count, steps)


def _monitor_budget(mon, grid, itemsize, steps, dt, horizon, streamed):
    region = resolve_region(mon.corners, grid)
    dims = [s.stop - s.start for s in (region.ix, region.iy, region.iz)]
    shapes = _yee_shapes(*dims)
    components = _expand_field_list(mon.fields)
    samples = sum(math.prod(shapes[c]) for c in components)
    # RegionOperators store all six diagonals in float64. A contiguous
    # cut can retain the whole parent allocation, particularly in single
    # precision (conversion to float64 precedes slicing).
    operators = sum(math.prod(s) for s in _yee_shapes(grid.Nx, grid.Ny, grid.Nz).values()) * 8
    if isinstance(mon, MonitorFieldFrequency):
        frame = samples * 16  # DFT bins always complex128
        count = len(mon.freqs)
        payload = frame * count
        # DFT update and normalisation materialise complex products.
        return MonitorMemoryEstimate(
            mon.name,
            frame,
            count,
            payload,
            payload,
            operators + 3 * payload,
            payload if streamed else 0,
            kind="frequency",
        )
    frame = samples * itemsize
    count = _frame_count(mon, steps, dt, horizon)
    payload = None if count is None else count * frame
    # Store flush cadence is at most 100 steps; one frame per step.
    # pop_pending stacks copies alongside the pending snapshots.
    buffered = min(100, count) if count is not None else 100
    ram_upper = (
        operators + 2 * buffered * frame
        if streamed
        else None
        if payload is None
        else operators + 2 * payload
    )
    return MonitorMemoryEstimate(
        mon.name,
        frame,
        count,
        payload,
        0,
        ram_upper,
        payload if streamed else 0,
        frame / max(mon.interval, dt or 0.0) if mon.interval is not None else None,
    )


def estimate_td(analysis, *, excited, total_time_steps, max_time_steps, t_end, dt, backend):
    """Account for current TD allocation paths without running their setup."""
    steps = _positive(total_time_steps, "total_time_steps", integer=True)
    horizon = _positive(t_end, "t_end")
    dt = _positive(dt, "dt")
    if horizon is not None and steps is not None:
        raise ValueError("pass either t_end or total_time_steps")
    if isinstance(max_time_steps, str):
        if max_time_steps != "auto":
            raise ValueError("max_time_steps must be 'auto', None or a positive integer")
        cap = None
    else:
        cap = _positive(max_time_steps, "max_time_steps", integer=True)
    if steps is None and horizon is None:
        steps = cap
    notes = []
    scenario = analysis.backend if backend is None else backend
    if scenario == "auto":
        scenario = os.environ.get("MAGNELIO_BACKEND", "auto")
    if scenario == "auto":
        scenario = "numpy"
        notes.append("Auto backend is unresolved: CPU scenario shown; pass backend='cupy' for GPU.")
    if scenario not in ("numpy", "cupy"):
        raise ValueError("backend must be 'auto', 'numpy' or 'cupy'")
    real, _ = resolve_precision(analysis.precision)
    real = np.dtype(real)
    mesh = analysis.mesh
    if dt is None and getattr(mesh, "_spectral_lambda_max", None) is not None:
        from magnelio.solver.stability import SAFETY_FACTORS  # noqa: PLC0415

        dt = SAFETY_FACTORS["normal"] * 2 / math.sqrt(mesh._spectral_lambda_max)
        notes.append("Snapshot counts use the cached CFL step at accuracy='normal'.")
    if dt is None:
        notes.append(
            "CFL step unknown: time-to-step counts remain unresolved; dt= supplies a scenario."
        )
    if steps is None and horizon is None and max_time_steps == "auto":
        notes.append(
            "Automatic step cap depends on prepared ports/waveforms and is not solved here."
        )
    if horizon is not None and dt is None:
        notes.append(
            "t_end cannot bound rounded snapshot times without dt; recording growth is shown."
        )
    resolver = getattr(analysis, "_resolve_excited", None)
    if resolver is None:
        if excited is not None:
            raise ValueError("excited= is only supported by AnalysisScatteringTD")
        runs = 1
    else:
        runs = len(resolver(excited))
    if runs < 1:
        raise ValueError("at least one run is required")
    shapes = _yee_shapes(mesh.Nx, mesh.Ny, mesh.Nz)
    ne = sum(math.prod(s) for c, s in shapes.items() if c.startswith("E"))
    nh = sum(math.prod(s) for c, s in shapes.items() if c.startswith("H"))
    held = array_bytes(mesh)
    fields = (ne + nh) * real.itemsize
    coefficients = 3 * fields
    mass = (ne + nh) * 8
    all_pec = mesh.pec_mask_edges.shape[1] >= max(
        math.prod(shapes[c]) for c in ("Ex", "Ey", "Ez")
    ) and all(
        mesh.pec_mask_edges[i, : math.prod(shapes[c])].all()
        for i, c in enumerate(("Ex", "Ey", "Ez"))
    )
    monitors = []
    unknown = []
    for mon in analysis.monitors:
        if isinstance(mon, (MonitorFieldTime, MonitorFieldFrequency)):
            monitors.append(
                _monitor_budget(
                    mon,
                    mesh.grid,
                    real.itemsize,
                    steps,
                    dt,
                    horizon,
                    analysis.project is not None,
                )
            )
        else:
            unknown.append(f"Monitor {mon.name!r} ({type(mon).__name__}) needs a separate budget.")
            monitors.append(
                MonitorMemoryEstimate(
                    mon.name,
                    None,
                    None,
                    None,
                    0,
                    None,
                    None if analysis.project is not None else 0,
                    kind="unknown",
                )
            )
    if analysis.ports or analysis.elements:
        unknown.append(
            "Port/element modes, factorisation, convolution histories "
            "and signal records are unbudgeted."
        )
    if analysis.sources:
        unknown.append("Source state and injection buffers are unbudgeted.")
    if analysis.wall_model == "sibc" or any(
        getattr(mat, "dispersion", None) is not None
        or getattr(mat, "dispersion_mu", None) is not None
        for mat in mesh.material_library.values()
    ):
        unknown.append("Dispersive or surface-impedance auxiliary states are unbudgeted.")
    # Count CPML slab states; coefficients/masks and their setup copies
    # are covered by a multiplier. No initialize() or backend probing.
    pml = 0
    from magnelio.boundaries.cpml import CPMLBoundary  # noqa: PLC0415

    for bc in analysis._resolve_bc().values():
        if isinstance(bc, CPMLBoundary):
            dims = [mesh.Nx, mesh.Ny, mesh.Nz]
            axis = "xyz".index(bc.face[0])
            width = min(bc.thickness_cells, dims[axis])
            tangential = [dims[a] for a in range(3) if a != axis]
            a, b = tangential
            pml += 2 * width * (a * (b + 1) + (a + 1) * b) * real.itemsize
        elif type(bc).__module__.startswith("magnelio.") is False:
            unknown.append(f"Custom boundary {type(bc).__name__} has unbudgeted state.")
    # Material builders retain three staircase property/geometry/mass
    # components plus concatenations and conformal replacement vectors.
    phases = [MemoryPhase("material matrices", held + mass, held + 12 * mass)]
    if getattr(mesh, "_spectral_lambda_max", None) is None and not all_pec:
        nnz = 4 * nh
        pointer = struct.calcsize("P")
        integer = sys.getsizeof(max(ne, nh))
        list_payload = 12 * nh * pointer + 5 * nh * integer
        index_bytes = 8 if max(nnz, ne, nh) > np.iinfo(np.int32).max else 4
        csr = nnz * (8 + index_bytes) + (nh + 1) * index_bytes
        # The list build and sparse conversion overlap. Lanczos defaults
        # to ncv=20; using every E edge bounds its active vector dimension.
        cfl_lo = held + mass + list_payload
        cfl_hi = (
            held
            + 2 * mass
            + max(
                math.ceil(list_payload * 1.15) + 72 * nnz,
                3 * csr + 28 * ne * 8 + 4 * nh * 8,
            )
        )
        phases.append(MemoryPhase("CFL setup", cfl_lo, cfl_hi))
    elif all_pec:
        phases.append(MemoryPhase("CFL setup (no live edges)", held + mass, held + 2 * mass))
    else:
        phases.append(MemoryPhase("CFL setup (cached)", held + mass, held + mass))
    if analysis.ports or analysis.elements:
        phases.append(MemoryPhase("port/element setup", held + mass, None))
    monitor_lo = sum(m.ram_lower_bytes for m in monitors)
    monitor_hi = (
        None
        if any(m.ram_upper_bytes is None for m in monitors)
        else sum(m.ram_upper_bytes for m in monitors)
    )
    resident = fields + coefficients + pml
    host = held + mass + monitor_lo
    upper = None if unknown or monitor_hi is None else held + mass + monitor_hi
    if scenario == "numpy":
        phases.extend(
            [
                MemoryPhase(
                    "solver setup",
                    host + resident,
                    None if upper is None else upper + resident + 16 * mass + 4 * pml,
                ),
                MemoryPhase(
                    "time integration",
                    host + resident,
                    None if upper is None else upper + resident + 2 * fields + 4 * pml,
                ),
            ]
        )
    else:
        phases.extend(
            [
                MemoryPhase(
                    "solver setup",
                    host,
                    None if upper is None else upper + resident + 16 * mass + 4 * pml,
                    resident,
                    None if unknown else 2 * resident + 2 * fields + 4 * pml,
                ),
                MemoryPhase(
                    "time integration",
                    host,
                    upper,
                    resident,
                    None if unknown else resident + 2 * fields + 4 * pml,
                ),
            ]
        )
    if analysis.project is None and runs > 1:
        notes.append("Prior run results retained in RAM are unbudgeted; per-run phases shown.")
        phases[-1] = MemoryPhase(
            phases[-1].name,
            phases[-1].ram_lower_bytes,
            None,
            phases[-1].vram_lower_bytes,
            phases[-1].vram_upper_bytes,
        )
    notes.extend(unknown)
    notes.append(
        "Ranges budget current allocation paths, not measured RSS or guaranteed process limits."
    )
    notes.append(
        "CAD/native objects, Python metadata, allocator pools "
        "and checkpoint/I/O workspaces are excluded."
    )
    return MemoryEstimate(
        held,
        fields,
        coefficients,
        scenario,
        real.name,
        runs,
        tuple(phases),
        tuple(monitors),
        tuple(notes),
    )
