"""AnalysisTD end to end (DD-224 Phase B).

A port channel driven through the general time-domain analysis records
the same V/I as the scattering analysis' channel run, bit for bit; two
modes of one port drive at once; a plane wave drives a run without any
port; a continuous-wave run needs its length; the project store carries
an AnalysisTD run, serves it back as a TDResult and resumes it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import magnelio as mio
from magnelio import Excitation, monitors, signals, sources
from magnelio.analysis import TDResult
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.ports import PortWaveguide

pytest.importorskip("OCC.Core.BRepPrimAPI")

F_MAX = 20e9


def _guide_mesh(n_modes: int = 1):
    """10 × 2 × 20 mm PEC box — a rectangular guide with TE10 at 15 GHz."""
    from magnelio.geo import Brick, GeometryModel

    a, b, L = 10e-3, 2e-3, 20e-3
    model = GeometryModel()
    model.add(Brick(origin=(-a / 2, -b / 2, -L / 2), size=(a, b, L), material="air"))
    model.add_port(PortWaveguide(name="p1", plane="zmin", n_modes=n_modes))
    model.add_port(PortWaveguide(name="p2", plane="zmax", n_modes=n_modes))
    return Mesh.from_geometry(model, mio.MeshControl(min_nodes_per_wavelength=8), f_max=F_MAX)


def _pec_box_with_plane_wave(n=10):
    grid = GridLines(
        x=np.linspace(0.0, 10e-3, n + 1),
        y=np.linspace(0.0, 10e-3, n + 1),
        z=np.linspace(0.0, 20e-3, 2 * n + 1),
    )
    pw = sources.SourcePlaneWave(
        name="pw",
        direction=(0, 0, 1),
        polarization=(1, 0, 0),
        corners=(
            (grid.x[2], grid.y[2], grid.z[2]),
            (grid.x[n - 2], grid.y[n - 2], grid.z[2 * n - 2]),
        ),
    )
    return Mesh.from_grid(grid).with_sources([pw])


class TestPortDrive:
    def test_channel_run_matches_the_scattering_analysis_bit_for_bit(self):
        mesh = _guide_mesh()
        scattering = mio.AnalysisScatteringTD(
            mesh=mesh, f_max=F_MAX, verbose=False, backend="numpy"
        )
        s_res = scattering.run(excited=["p1"], energy_stop_db=40.0)
        general = mio.AnalysisTD(mesh=mesh, f_max=F_MAX, verbose=False, backend="numpy")
        td = general.run(excitations=["p1"], energy_stop_db=40.0)

        assert isinstance(td, TDResult)
        assert td.n_steps == s_res.n_actual_steps
        assert td.stop_reason == s_res.settings.stop_reason
        assert td.settings.excitations == (("p1", 0),)
        for chan, (v, i) in s_res.signals[("p1", 0)].items():
            np.testing.assert_array_equal(td.signal(*chan).values, v.values)
            np.testing.assert_array_equal(td.signal(*chan, kind="I").values, i.values)
        np.testing.assert_array_equal(
            td.excitation_signal("p1").values, s_res.reference_signal.values
        )
        # the resolved waveform is the per-mode default of the channel
        wf = td.excitations[0].waveform
        assert isinstance(wf, signals.WaveformGaussianModulated)
        assert wf.f_max == F_MAX and wf.f_min > 0.0
        # power waves agree with the scattering result's
        np.testing.assert_array_equal(
            td.a("p1", f_ref=18e9).values, s_res.a("p1", f_ref=18e9).values
        )
        np.testing.assert_array_equal(
            td.b("p2", f_ref=18e9).values, s_res.b("p2", f_ref=18e9).values
        )
        assert td.energy_trace is not None and td.energy_trace["energy"].size > 0

    def test_amplitude_and_delay_scale_and_shift_the_drive(self):
        mesh = _guide_mesh()
        wf = signals.WaveformGaussianModulated(f_min=15.5e9, f_max=F_MAX)
        general = mio.AnalysisTD(mesh=mesh, f_max=F_MAX, verbose=False, backend="numpy")
        base = general.run(excitations=[Excitation("p1", waveform=wf)], total_time_steps=300)
        shifted = general.run(
            excitations=[Excitation("p1", waveform=wf, amplitude=2.0, delay=50 * base.dt)],
            total_time_steps=300,
        )
        e0 = base.excitation_signal("p1").values
        e1 = shifted.excitation_signal("p1").values
        np.testing.assert_allclose(e1[50:], 2.0 * e0[:-50], rtol=1e-12, atol=1e-15)
        assert shifted.stop_reason == "steps" and shifted.n_steps == 300
        v0 = base.signal("p2").values
        v1 = shifted.signal("p2").values
        # linear system: the delayed, doubled drive gives the delayed, doubled response
        np.testing.assert_allclose(v1[50:], 2.0 * v0[:-50], rtol=1e-5, atol=1e-6 * np.abs(v0).max())

    def test_two_modes_of_one_port_drive_simultaneously(self):
        mesh = _guide_mesh(n_modes=2)
        general = mio.AnalysisTD(mesh=mesh, f_max=F_MAX, verbose=False, backend="numpy")
        wf = signals.WaveformGaussianModulated(f_min=15.5e9, f_max=F_MAX)
        one = general.run(excitations=[Excitation("p1", mode=0, waveform=wf)], total_time_steps=250)
        two = general.run(
            excitations=[
                Excitation("p1", mode=0, waveform=wf),
                Excitation("p1", mode=1, waveform=wf),
            ],
            total_time_steps=250,
        )
        assert set(two.excitation_signals) == {("p1", 0), ("p1", 1)}
        assert two.signal("p1", 1).values.max() > 0.0
        assert two.settings.excitations == (("p1", 0), ("p1", 1))
        # the mode-0 drive is the same in both runs — the superposed run
        # sees mode 1 on top, so the mode-0 signal differs from the single
        # run only by what mode 1 scatters into it (small but non-zero)
        np.testing.assert_array_equal(
            two.excitation_signal("p1", 0).values, one.excitation_signal("p1", 0).values
        )


class TestSourceDrive:
    def test_plane_wave_drives_a_run_without_ports(self):
        mesh = _pec_box_with_plane_wave()
        probe = monitors.MonitorFieldTime(
            corners=((5e-3, 5e-3, 10e-3), (5e-3, 5e-3, 10e-3)),
            interval=5e-12,
            fields=["E"],
            name="probe",
        )
        analysis = mio.AnalysisTD(
            mesh=mesh, f_max=F_MAX, monitors=[probe], verbose=False, backend="numpy"
        )
        assert analysis.ports == []
        assert analysis._resolve_port_signal_stop("auto") is None
        res = analysis.run(excitations=[Excitation("pw", amplitude=2.0)], total_time_steps=120)
        assert res.signals == {}
        drive = res.excitation_signal("pw")
        assert drive.values.max() == pytest.approx(2.0, rel=2e-3)
        assert isinstance(res.excitations[0].waveform, signals.WaveformGaussian)
        assert res.monitors["probe"] is probe
        ex = np.asarray(probe.component("Ex"))
        assert np.max(np.abs(ex)) > 1e-3
        assert res.energy_trace is not None

    def test_continuous_wave_needs_a_length(self):
        mesh = _pec_box_with_plane_wave()
        analysis = mio.AnalysisTD(mesh=mesh, f_max=F_MAX, verbose=False, backend="numpy")
        cw = Excitation("pw", waveform=signals.WaveformSine(f=10e9, rise_time=0.2e-9))
        with pytest.raises(ValueError, match="continuous-wave"):
            analysis.run(excitations=[cw])
        res = analysis.run(excitations=[cw], t_end=0.3e-9)
        assert res.n_steps == math.ceil(0.3e-9 / res.dt - 1e-9)
        assert res.settings.energy_stop_db is None
        assert res.settings.port_signal_stop_db is None
        assert res.settings.t_end == 0.3e-9
        assert res.stop_reason == "steps"


class TestProjectStore:
    def test_store_round_trip_and_resume(self, tmp_path):
        mesh = _guide_mesh()
        wf = signals.WaveformGaussianModulated(f_min=15.5e9, f_max=F_MAX)
        excs = [Excitation("p1", waveform=wf), Excitation("p2", waveform=wf, delay=20e-12)]
        ram = mio.AnalysisTD(mesh=mesh, f_max=F_MAX, verbose=False, backend="numpy").run(
            excitations=excs, total_time_steps=240
        )
        proj = mio.AnalysisTD(
            mesh=mesh, f_max=F_MAX, verbose=False, backend="numpy", project=str(tmp_path / "proj")
        ).run(excitations=excs, total_time_steps=240, name="both")
        assert proj.setup["analysis"] == "AnalysisTD"
        assert list(proj.runs) == ["both"]
        assert proj.runs["both"]["excited"] is None
        assert proj.runs["both"]["excitations"] == [["p1", 0], ["p2", 0]]
        assert proj.runs["both"]["state"] == "done"

        stored = proj.result("both")
        assert isinstance(stored, TDResult) and stored.name == "both"
        assert stored.n_steps == ram.n_steps == 240
        assert [e.source for e in stored.excitations] == ["p1", "p2"]
        assert stored.excitations[1].delay == 20e-12
        for key in ram.signals:
            np.testing.assert_array_equal(stored.signal(*key).values, ram.signal(*key).values)
        for key in ram.excitation_signals:
            np.testing.assert_array_equal(
                stored.excitation_signal(*key).values, ram.excitation_signal(*key).values
            )
        np.testing.assert_array_equal(stored.a("p1").values, ram.a("p1").values)
        assert stored.stop_reason == "steps"
        with pytest.raises(ValueError, match="S-parameters are a scattering result"):
            proj.S("p2", "p1")

        # resume: run longer, then the extended record equals an uninterrupted run
        longer = mio.resume(proj, "both", total_time_steps=300)
        ext = longer.result("both")
        assert ext.n_steps == 300
        ref = mio.AnalysisTD(mesh=mesh, f_max=F_MAX, verbose=False, backend="numpy").run(
            excitations=excs, total_time_steps=300
        )
        np.testing.assert_array_equal(ext.signal("p2").values, ref.signal("p2").values)
        np.testing.assert_array_equal(
            ext.excitation_signal("p2").values, ref.excitation_signal("p2").values
        )

    def test_run_names_and_kind_guard(self, tmp_path):
        mesh = _guide_mesh()
        a = mio.AnalysisTD(
            mesh=mesh, f_max=F_MAX, verbose=False, backend="numpy", project=str(tmp_path / "p")
        )
        proj = a.run(excitations=["p1"], total_time_steps=60)
        assert list(proj.runs) == ["run_1"]
        proj = a.run(excitations=["p2"], total_time_steps=60)
        assert list(proj.runs) == ["run_1", "run_2"]
        with pytest.raises(ValueError, match="already taken"):
            a.run(excitations=["p1"], total_time_steps=60, name="run_1")
        with pytest.raises(ValueError, match="one project, one analysis kind"):
            mio.AnalysisScatteringTD(
                mesh=mesh, f_max=F_MAX, verbose=False, backend="numpy", project=str(tmp_path / "p")
            ).run(excited=["p1"], total_time_steps=60)
        assert proj.result("run_2").excitations[0].source == "p2"
        with pytest.raises(ValueError, match="pass the run name"):
            proj.result()

    def test_scattering_project_serves_channel_runs_as_td_results(self, tmp_path):
        mesh = _guide_mesh()
        proj = mio.AnalysisScatteringTD(
            mesh=mesh, f_max=F_MAX, verbose=False, backend="numpy", project=str(tmp_path / "s")
        ).run(excited=["p1"], energy_stop_db=40.0)
        assert proj.runs["p1_mode0"]["excited"] == ["p1", 0]
        assert proj.runs["p1_mode0"]["excitations"] == [["p1", 0]]
        td = proj.result(("p1", 0))
        assert td.name == "p1_mode0"
        np.testing.assert_array_equal(
            td.excitation_signal("p1").values, proj.reference_signal.values
        )
        assert proj.S("p2", "p1").shape == (201,)
