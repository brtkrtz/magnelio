"""Budgets must match recording storage without constructing a solver."""

import numpy as np
import pytest

import magnelio as mio
from magnelio._fields.field_arrays import FieldArrays
from magnelio._memory import array_bytes
from magnelio.analysis.memory import _frame_count
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.monitors import MonitorFieldFrequency, MonitorFieldTime
from magnelio.sources import SourcePlaneWave


def _analysis(monitors=(), **kwargs):
    mesh = Mesh.from_grid(
        GridLines(x=np.linspace(0, 0.01, 5), y=np.linspace(0, 0.02, 6), z=np.linspace(0, 0.03, 7)),
    ).with_sources([SourcePlaneWave(name="wave")])
    return mio.AnalysisTD(mesh=mesh, f_max=1e9, monitors=list(monitors), **kwargs)


def test_estimate_never_builds_operators_or_attaches_monitors(monkeypatch):
    from magnelio._operators import curl, material_matrices
    from magnelio.solver import stability

    def forbidden(*args, **kwargs):
        pytest.fail("estimate constructed a solver operator")

    mon = MonitorFieldFrequency(freqs=[1e9], fields=["E"])
    analysis = _analysis([mon], precision="double")
    for module, name in (
        (curl, "build_curl_matrix"),
        (material_matrices, "build_M_eps"),
        (stability, "spectral_dt"),
        (mon, "attach"),
    ):
        monkeypatch.setattr(module, name, forbidden)
    estimate = analysis.estimate()
    assert estimate.mesh_bytes == array_bytes(analysis.mesh)
    fields = FieldArrays.zeros(4, 5, 6, dtype=np.float64)
    assert estimate.fields_bytes == sum(
        getattr(fields, c).nbytes for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    )
    assert estimate.coefficients_bytes == 3 * estimate.fields_bytes
    assert mon._region is None
    assert not hasattr(analysis.mesh, "_spectral_lambda_max")
    assert any(p.name == "CFL setup" for p in estimate.phases)


@pytest.mark.parametrize("precision,dtype", [("single", np.float32), ("double", np.float64)])
def test_snapshot_payload_matches_actual_staggered_region(precision, dtype):
    mon = MonitorFieldTime(
        corners=((0.005, None, None), (0.005, None, None)), times=[0, 1, 2], fields=["Ez", "H"]
    )
    analysis = _analysis([mon], precision=precision)
    report = analysis.estimate(total_time_steps=4, dt=1)
    mon.attach(analysis.mesh)
    mon.record(FieldArrays.zeros(4, 5, 6, dtype=dtype), 0, 0, 1)
    actual = sum(arr.nbytes for arr in mon._snapshots[0].values())
    assert report.monitors[0].bytes_per_snapshot == actual
    assert report.monitors[0].recording_bytes == 2 * actual


@pytest.mark.parametrize("times", [[0, 0.1, 0.5, 1.5, 1.6, 3.5], [10, 20], [0, 0, 2, 4]])
def test_explicit_count_matches_recording_coalescing(times):
    mon = MonitorFieldTime(times=times)
    analysis = _analysis([mon])
    planned = _frame_count(mon, 4, 1, None)
    mon.attach(analysis.mesh)
    fields = FieldArrays.zeros(4, 5, 6)
    for step in range(4):
        mon.record(fields, step, step, 1)
    assert planned == mon._n_recorded


def test_open_interval_distinguishes_ram_growth_and_streaming():
    mon = MonitorFieldTime(interval=0.25, fields=["E"])
    memory = _analysis([mon], precision="double").estimate(max_time_steps=None)
    streamed = _analysis([mon], precision="double", project="unused").estimate(
        max_time_steps=None,
    )
    a, b = memory.monitors[0], streamed.monitors[0]
    assert a.max_snapshots is b.max_snapshots is None
    assert a.recording_bytes is b.recording_bytes is None
    assert a.ram_upper_bytes is None
    assert b.ram_upper_bytes is not None
    assert a.disk_bytes == 0
    assert b.disk_bytes is None
    assert a.bytes_per_second == a.bytes_per_snapshot / 0.25


def test_duration_and_explicit_cap_bound_interval():
    mon = MonitorFieldTime(interval=2, start=2)
    analysis = _analysis([mon])
    assert analysis.estimate(t_end=10, dt=1).monitors[0].max_snapshots == 5
    assert analysis.estimate(max_time_steps=10, dt=1).monitors[0].max_snapshots == 5
    assert (
        analysis.estimate(total_time_steps=4, max_time_steps=2, dt=1).monitors[0].max_snapshots == 2
    )
    assert analysis.estimate(t_end=10).monitors[0].max_snapshots is None
    assert analysis.estimate(max_time_steps=10).monitors[0].max_snapshots == 10


def test_interval_growth_is_capped_by_known_step():
    mon = MonitorFieldTime(interval=0.25)
    budget = _analysis([mon]).estimate(dt=1).monitors[0]
    assert budget.bytes_per_second == budget.bytes_per_snapshot


def test_fully_masked_mesh_does_not_budget_curl_construction():
    analysis = _analysis()
    analysis.mesh.pec_mask_edges[:] = True
    report = analysis.estimate()
    assert any(p.name == "CFL setup (no live edges)" for p in report.phases)


def test_frequency_budget_matches_bins_and_is_always_complex128():
    mon = MonitorFieldFrequency(freqs=[1e8, 2e8], fields=["Ez"])
    analysis = _analysis([mon], precision="single", project="unused")
    report = analysis.estimate()
    mon.attach(analysis.mesh)
    actual = sum(acc._bins.nbytes for acc in mon._accumulators.values())
    assert report.monitors[0].recording_bytes == actual
    assert report.monitors[0].bytes_per_snapshot * 2 == actual
    assert report.monitor_disk_bytes == actual


def test_precision_changes_fields_but_not_cfl_lists():
    single = _analysis(precision="single").estimate()
    double = _analysis(precision="double").estimate()
    assert single.fields_bytes * 2 == double.fields_bytes
    a = next(p for p in single.phases if p.name == "CFL setup")
    b = next(p for p in double.phases if p.name == "CFL setup")
    assert a == b


def test_gpu_scenario_does_not_initialise_cuda(monkeypatch):
    from magnelio._backend import array_api

    monkeypatch.setattr(array_api, "resolve_backend", lambda *_: pytest.fail("CUDA probe"))
    report = _analysis().estimate(backend="cupy")
    phase = next(p for p in report.phases if p.name == "time integration")
    assert phase.vram_lower_bytes >= report.fields_bytes + report.coefficients_bytes
    assert "VRAM" in str(report)


def test_cached_cfl_removes_matrix_construction_budget():
    analysis = _analysis()
    analysis.mesh._spectral_lambda_max = 1e18
    estimate = analysis.estimate()
    assert any(p.name == "CFL setup (cached)" for p in estimate.phases)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt": 0},
        {"total_time_steps": 1.5},
        {"t_end": 1, "total_time_steps": 2},
        {"max_time_steps": "invalid"},
    ],
)
def test_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        _analysis().estimate(**kwargs)


def test_unknown_monitors_keep_upper_budget_open():
    class CustomMonitor:
        name = "custom"

    report = _analysis([CustomMonitor()]).estimate()
    assert report.peak_ram_bytes is None
    assert any("CustomMonitor" in note for note in report.notes)


def test_scattering_counts_sequential_runs_and_disk_payload():
    from magnelio.ports import PortLumped

    mesh = _analysis().mesh.with_ports(
        [
            PortLumped(name="p1", start=(0.002, 0.003, 0.003), end=(0.004, 0.003, 0.003)),
            PortLumped(name="p2", start=(0.002, 0.003, 0.02), end=(0.004, 0.003, 0.02)),
        ]
    )
    mon = MonitorFieldTime(times=[0, 1, 2], fields=["Ez"])
    analysis = mio.AnalysisScatteringTD(
        mesh=mesh,
        f_max=1e9,
        project="unused",
        monitors=[mon],
    )
    one = analysis.estimate(excited=["p1"], total_time_steps=4, dt=1)
    two = analysis.estimate(excited=["p1", "p2"], total_time_steps=4, dt=1)
    assert one.runs == 1 and two.runs == 2
    assert two.monitor_disk_bytes == 2 * one.monitor_disk_bytes
    assert two.fields_bytes == one.fields_bytes
    assert two.phases == one.phases


def test_large_grid_cfl_cost_is_visible_without_large_allocations(monkeypatch):
    analysis = _analysis()
    analysis.mesh.grid = GridLines(
        x=np.linspace(0, 1, 185),
        y=np.linspace(0, 1, 185),
        z=np.linspace(0, 1, 4985),
    )

    # Only the topology changes: the estimator must not attempt to build
    # a full-grid operator from these deliberately small backing arrays.
    def forbidden(*args, **kwargs):
        pytest.fail("full-grid allocation")

    monkeypatch.setattr(np, "zeros", forbidden)
    monkeypatch.setattr(np, "empty", forbidden)
    estimate = analysis.estimate()
    cfl = next(p for p in estimate.phases if p.name == "CFL setup")
    assert cfl.ram_lower_bytes > 100 * 2**30
    assert cfl.ram_upper_bytes > cfl.ram_lower_bytes
