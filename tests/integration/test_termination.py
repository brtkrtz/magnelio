"""Finite-drive completion, decay diagnostics and continuation (DD-282)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import magnelio as mio
from magnelio import Excitation, signals
from magnelio.mesh.grid import GridLines
from magnelio.ports import PortWaveguide


def _analysis(*, project=None, precision="double", z_offset=0.0, sources=()):
    grid = GridLines(
        x=np.linspace(-5e-3, 5e-3, 5),
        y=np.linspace(-2.5e-3, 2.5e-3, 5),
        z=np.linspace(0, 20e-3, 21) + z_offset,
    )
    mesh = (
        mio.Mesh.from_grid(
            grid,
            boundary_conditions={
                "xmin": "PMC",
                "xmax": "PMC",
                "ymin": "PEC",
                "ymax": "PEC",
                "zmin": "PEC",
                "zmax": "PEC",
            },
        )
        .with_ports(
            [
                PortWaveguide(name="p1", plane="zmin"),
                PortWaveguide(name="p2", plane="zmax"),
            ]
        )
        .with_sources(sources)
    )
    return mio.AnalysisTD(
        mesh=mesh,
        f_max=12e9,
        project=project,
        backend="numpy",
        precision=precision,
        verbose=False,
    )


def _drives():
    pulse = signals.WaveformGaussian(f_max=12e9)
    return [Excitation("p1", waveform=pulse), Excitation("p2", waveform=pulse, delay=1.5e-9)]


@pytest.mark.parametrize("precision", ["single", "double"])
def test_decay_waits_for_delayed_drive_and_matches_full_record(precision):
    analysis, drives = _analysis(precision=precision), _drives()
    result = analysis.run(excitations=drives)
    assert result.stop_reason == "energy"
    assert result.n_steps >= math.ceil((drives[1].delay + drives[1].waveform.t_end) / result.dt)
    assert np.max(abs(result.excitation_signal("p2").values)) > 0.99
    reference = analysis.run(
        excitations=drives,
        total_time_steps=result.n_steps,
        energy_stop_db=None,
        port_signal_stop_db=None,
    )
    for port in ("p1", "p2"):
        for kind in ("V", "I"):
            np.testing.assert_array_equal(
                result.signal(port, kind=kind).values, reference.signal(port, kind=kind).values
            )


def test_energy_decay_waits_for_active_single_pulse():
    pulse = signals.WaveformGaussian(f_max=12e9)
    result = _analysis().run(excitations=[Excitation("p1", waveform=pulse)], energy_stop_db=20.0)
    assert result.stop_reason == "energy"
    assert result.n_steps >= math.ceil(pulse.t_end / result.dt)


def test_decay_waits_for_spatially_retarded_plane_wave():
    offset = 0.45
    source = mio.sources.SourcePlaneWave(
        name="pw",
        direction=(0, 0, 1),
        polarization=(0, 1, 0),
        corners=((-2.5e-3, -1.25e-3, offset + 2e-3), (2.5e-3, 1.25e-3, offset + 18e-3)),
    )
    pulse = signals.WaveformGaussian(f_max=12e9)
    analysis = _analysis(z_offset=offset, sources=[source])
    drives = [Excitation("p1", waveform=pulse), Excitation("pw", waveform=pulse)]
    result = analysis.run(excitations=drives, total_time_steps=2800)
    assert result.n_steps * result.dt >= pulse.t_end + (offset + 18e-3) / 299_792_458.0
    # The retarded drive must actually enter the record after the first
    # port pulse has left, rather than merely stretching a duration bound.
    late = int(1.5e-9 / result.dt)
    control = analysis.run(
        excitations=drives[:1],
        total_time_steps=result.n_steps,
        energy_stop_db=None,
        port_signal_stop_db=None,
    )
    # p2 is outside the TF box: only its finite-grid leakage reaches it.
    assert np.max(abs(result.signal("p2").values[late:])) > 1e-8
    assert np.max(abs(control.signal("p2").values[late:])) < 1e-10
    reference = analysis.run(
        excitations=drives,
        total_time_steps=result.n_steps,
        energy_stop_db=None,
        port_signal_stop_db=None,
    )
    np.testing.assert_array_equal(result.signal("p2").values, reference.signal("p2").values)


def test_explicit_step_bound_can_end_before_pending_drive():
    result = _analysis().run(excitations=_drives(), total_time_steps=100)
    assert result.n_steps == 100 and result.stop_reason == "steps"
    assert result.n_steps * result.dt < 1.5e-9


def test_resume_in_quiet_gap_keeps_absolute_drive_end(tmp_path):
    drives = _drives()
    reference = _analysis().run(excitations=drives)
    split = int(1e-9 / reference.dt)
    path = tmp_path / "delayed"
    project = _analysis(project=path).run(
        excitations=drives,
        name="delayed",
        total_time_steps=split,
        energy_stop_db=None,
        port_signal_stop_db=None,
        checkpoint_interval=100,
    )
    project = mio.resume(
        project, "delayed", energy_stop_db=70.0, port_signal_stop_db=60.0, verbose=False
    )
    resumed = project.result("delayed")
    assert resumed.stop_reason == "energy" and resumed.n_steps == reference.n_steps
    assert np.max(abs(resumed.excitation_signal("p2").values)) > 0.99
    for port in ("p1", "p2"):
        for kind in ("V", "I"):
            np.testing.assert_array_equal(
                resumed.signal(port, kind=kind).values, reference.signal(port, kind=kind).values
            )


@pytest.mark.parametrize("precision", ["single", "double"])
def test_energy_stop_reports_current_port_interval(precision):
    pulse = signals.WaveformGaussian(f_max=3e9)
    result = _analysis(precision=precision).run(excitations=[Excitation("p1", waveform=pulse)])
    assert result.stop_reason == "energy"
    voltages = [result.signal(port).values for port in ("p1", "p2")]
    peak = max(np.max(abs(v)) for v in voltages)
    interval = int(result.energy_trace["step"][-1] - result.energy_trace["step"][-2])
    current_db = 20 * np.log10(max(np.max(abs(v[-interval:])) for v in voltages) / peak)
    previous_db = 20 * np.log10(
        max(np.max(abs(v[-2 * interval : -interval])) for v in voltages) / peak
    )
    assert abs(current_db - previous_db) > 1.0
    tolerance = 1e-5 if precision == "single" else 1e-9
    assert result.settings.final_port_signal_db == pytest.approx(current_db, abs=tolerance)


def test_periodic_checkpoints_keep_current_decay_peaks(tmp_path, monkeypatch):
    from magnelio.io.project import _RunSink

    snapshots = []
    original = _RunSink.write_checkpoint

    def capture(sink, step=None):
        state = sink._checkpoint_fn()
        snapshots.append((state["n_completed"], state["peak_energy"], state["peak_signal"]))
        return original(sink, step)

    monkeypatch.setattr(_RunSink, "write_checkpoint", capture)
    result = (
        _analysis(project=tmp_path / "peaks")
        .run(
            excitations=[_drives()[0]],
            name="peaks",
            total_time_steps=601,
            energy_stop_db=None,
            checkpoint_interval=100,
        )
        .result("peaks")
    )
    assert len(snapshots) >= 3
    trace = result.energy_trace
    for completed, peak_energy, peak_signal in snapshots:
        prefix = trace["energy"][trace["step"] < completed]
        assert peak_energy == max(prefix)
        expected = max(np.max(abs(result.signal(port).values[:completed])) for port in ("p1", "p2"))
        assert peak_signal == pytest.approx(expected, rel=1e-12)
