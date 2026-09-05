"""AnalysisTD — construction, excitation resolution, run-length rules (DD-224 Phase B)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import magnelio as mio
from magnelio import signals, sources
from magnelio.analysis import TDResult
from magnelio.analysis._recipe import excitation_from_dict, excitation_to_dict
from magnelio.analysis.excitation import Excitation
from magnelio.analysis.time_domain import AnalysisTD, _PreparedRun
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.ports import PortLumped, PortWaveguide


def _grid(n=8):
    return GridLines(
        x=np.linspace(0.0, 8e-3, n + 1),
        y=np.linspace(0.0, 4e-3, n // 2 + 1),
        z=np.linspace(0.0, 12e-3, n + 5),
    )


def _mesh(*, ports=True, source=True):
    mesh = Mesh.from_grid(_grid())
    if ports:
        mesh = mesh.with_ports([PortWaveguide(name="p", plane="zmin")])
    if source:
        mesh = mesh.with_sources(
            [sources.SourcePlaneWave(name="pw", direction=(0, 0, 1), polarization=(1, 0, 0))]
        )
    return mesh


def _prepared(*excitations):
    return _PreparedRun(
        excitations=tuple(excitations),
        operators=[],
        element_ops=[],
        sources=[],
        recorder=None,
        drives={},
        n_steps_estimate=100,
        port_modes={},
        port_normal_dx={},
        port_line_params={},
    )


# ── namespace ────────────────────────────────────────────────────────────────


def test_core_and_analysis_exports():
    assert mio.AnalysisTD is AnalysisTD
    assert "AnalysisTD" in mio.__all__
    assert "TDResult" in mio.analysis.__all__
    assert TDResult is mio.analysis.TDResult


# ── construction ─────────────────────────────────────────────────────────────


class TestConstruction:
    def test_picks_up_ports_and_sources_from_the_mesh(self):
        a = AnalysisTD(mesh=_mesh(), f_max=10e9, verbose=False)
        assert [s.name for s in a.ports] == ["p"]
        assert [s.name for s in a.sources] == ["pw"]
        assert a.resolved_method == "fit"

    def test_sources_alone_are_enough(self):
        a = AnalysisTD(mesh=_mesh(ports=False), f_max=10e9, verbose=False)
        assert a.ports == []
        assert [s.name for s in a.sources] == ["pw"]

    def test_nothing_to_excite_raises(self):
        with pytest.raises(ValueError, match="nothing to excite"):
            AnalysisTD(mesh=_mesh(ports=False, source=False), f_max=10e9, verbose=False)

    def test_rejects_algebraic_solver_and_other_methods(self):
        with pytest.raises(ValueError, match="no algebraic solver"):
            AnalysisTD(mesh=_mesh(), f_max=10e9, solver="arpack", verbose=False)
        with pytest.raises(ValueError, match="tetrahedral"):
            AnalysisTD(mesh=_mesh(), f_max=10e9, method="fem", verbose=False)
        with pytest.raises(ValueError, match="method must be"):
            AnalysisTD(mesh=_mesh(), f_max=10e9, method="tlm", verbose=False)

    def test_band_pipeline_belongs_to_the_scattering_analysis(self):
        with pytest.raises(ValueError, match="port_model"):
            AnalysisTD(mesh=_mesh(), f_max=10e9, port_model="band", verbose=False)

    def test_scattering_analysis_derives_from_td(self):
        assert issubclass(mio.AnalysisScatteringTD, AnalysisTD)
        a = mio.AnalysisScatteringTD(mesh=_mesh(), f_max=10e9, verbose=False)
        assert a.f_min == 0.0 and a.n_freq == 201
        with pytest.raises(TypeError, match="excited="):
            a.run(excitations=["p"])

    def test_repr(self):
        r = repr(AnalysisTD(mesh=_mesh(), f_max=10e9, verbose=False))
        assert r.startswith("AnalysisTD(ports=['p'], sources=['pw']")


# ── excitations ──────────────────────────────────────────────────────────────


class TestResolveExcitations:
    def _a(self):
        return AnalysisTD(mesh=_mesh(), f_max=10e9, verbose=False)

    def test_shorthands_and_objects(self):
        out = self._a()._resolve_excitations(["p", ("p", 1), Excitation("pw", amplitude=3.0)])
        assert [(e.source, e.mode, e.amplitude) for e in out] == [
            ("p", 0, 1.0),
            ("p", 1, 1.0),
            ("pw", 0, 3.0),
        ]

    def test_single_entry_without_a_list(self):
        assert [e.source for e in self._a()._resolve_excitations("pw")] == ["pw"]
        assert [e.mode for e in self._a()._resolve_excitations(("p", 1))] == [1]

    def test_none_and_empty_are_rejected(self):
        with pytest.raises(TypeError, match="excitations="):
            self._a()._resolve_excitations(None)
        with pytest.raises(ValueError, match="must not be empty"):
            self._a()._resolve_excitations([])

    def test_unknown_name_and_duplicates(self):
        with pytest.raises(ValueError, match="neither a port"):
            self._a()._resolve_excitations(["nope"])
        with pytest.raises(ValueError, match="duplicate"):
            self._a()._resolve_excitations(["p", Excitation("p")])

    def test_source_has_no_modes(self):
        with pytest.raises(ValueError, match="no modes"):
            self._a()._resolve_excitations([("pw", 1)])

    def test_waveform_above_the_band_warns(self):
        with pytest.warns(UserWarning, match="exceeds the analysis band"):
            self._a()._resolve_excitations(
                [Excitation("pw", waveform=signals.WaveformGaussian(f_max=12e9))]
            )

    def test_codec_round_trip(self):
        exc = Excitation(
            "p",
            mode=1,
            waveform=signals.WaveformGaussianModulated(f_min=2e9, f_max=8e9),
            amplitude=0.5,
            delay=1e-9,
            phase=90.0,
        )
        d = excitation_to_dict(exc)
        assert d["waveform"]["type"] == "WaveformGaussianModulated"
        assert excitation_from_dict(d) == exc


# ── run-length rules ─────────────────────────────────────────────────────────


class TestDurationRules:
    def test_t_end_becomes_a_step_count(self):
        prep = _prepared(Excitation("p", waveform=signals.WaveformGaussian(f_max=10e9)))
        steps, esd, psd = AnalysisTD._duration_rules(prep, 1e-12, 2.5e-9, None, 70.0, 60.0)
        assert steps == math.ceil(2.5e-9 / 1e-12)
        assert (esd, psd) == (70.0, 60.0)

    def test_t_end_and_total_time_steps_are_exclusive(self):
        prep = _prepared(Excitation("p", waveform=signals.WaveformGaussian(f_max=10e9)))
        with pytest.raises(ValueError, match="both fix the run length"):
            AnalysisTD._duration_rules(prep, 1e-12, 2.5e-9, 100, 70.0, 60.0)
        with pytest.raises(ValueError, match="positive duration"):
            AnalysisTD._duration_rules(prep, 1e-12, 0.0, None, 70.0, 60.0)

    def test_continuous_wave_needs_a_length_and_has_no_decay_stop(self):
        prep = _prepared(Excitation("p", waveform=signals.WaveformSine(f=5e9)))
        with pytest.raises(ValueError, match="continuous-wave"):
            AnalysisTD._duration_rules(prep, 1e-12, None, None, 70.0, 60.0)
        steps, esd, psd = AnalysisTD._duration_rules(prep, 1e-12, 4e-9, None, 70.0, 60.0)
        assert steps == 4000 and esd is None and psd is None
        steps, esd, psd = AnalysisTD._duration_rules(prep, 1e-12, None, 500, 70.0, 60.0)
        assert steps == 500 and esd is None and psd is None

    def test_pulse_duration_is_the_last_finite_end(self):
        g = signals.WaveformGaussian(f_max=10e9)
        excs = [
            Excitation("p", waveform=g),
            Excitation("pw", waveform=g, delay=3e-9),
            Excitation("q", waveform=signals.WaveformSine(f=1e9)),
        ]
        assert AnalysisTD._pulse_duration(excs) == pytest.approx(3e-9 + 8.0 / 10e9)
        assert AnalysisTD._pulse_duration(excs[2:]) == 0.0

    def test_step_estimate_reproduces_the_gaussian_rule(self):
        grid = _grid()
        dt = 1e-12
        f_max = 10e9
        L = math.sqrt(8e-3**2 + 4e-3**2 + 12e-3**2)
        expected = math.ceil((2.0 * (4.0 / f_max) + 25 * L / (0.5 * 299_792_458.0)) / dt)
        assert AnalysisTD._estimate_steps(grid, 8.0 / f_max, dt) == expected


class TestStopAndNames:
    def test_port_signal_auto_needs_a_modal_port(self):
        with_modal = AnalysisTD(mesh=_mesh(), f_max=10e9, verbose=False)
        assert with_modal._resolve_port_signal_stop("auto") == 60.0
        assert with_modal._resolve_port_signal_stop(None) is None
        assert with_modal._resolve_port_signal_stop(40.0) == 40.0
        source_only = AnalysisTD(mesh=_mesh(ports=False), f_max=10e9, verbose=False)
        assert source_only._resolve_port_signal_stop("auto") is None
        lumped = Mesh.from_grid(_grid()).with_ports(
            [PortLumped(name="lp", start=(0, 0, 0), end=(0, 0, 1e-3), Z0=50.0)]
        )
        assert (
            AnalysisTD(mesh=lumped, f_max=10e9, verbose=False)._resolve_port_signal_stop("auto")
            is None
        )
        with pytest.raises(ValueError, match="'auto'"):
            with_modal._resolve_port_signal_stop("later")

    def test_run_names(self):
        assert AnalysisTD._next_run_name([], None) == "run_1"
        assert AnalysisTD._next_run_name(["run_1", "p_mode0"], None) == "run_3"
        assert AnalysisTD._next_run_name(["run_1", "run_3"], None) == "run_4"
        assert AnalysisTD._next_run_name(["run_1"], "sweep") == "sweep"
        with pytest.raises(ValueError, match="already taken"):
            AnalysisTD._next_run_name(["p_mode0"], "p_mode0")


# ── TDResult ─────────────────────────────────────────────────────────────────


class TestTDResult:
    def _result(self):
        dt = 1e-12
        t = np.arange(5) * dt
        sig = signals.Signal1D(t=t, values=np.arange(5.0), dt=dt)
        return TDResult(
            excitations=(Excitation("p", waveform=signals.WaveformGaussian(f_max=10e9)),),
            dt=dt,
            n_steps=5,
            signals={("p", 0): (sig, sig)},
            excitation_signals={("p", 0): sig},
        )

    def test_accessors(self):
        r = self._result()
        np.testing.assert_array_equal(r.t, np.arange(5) * 1e-12)
        assert r.signal("p").values[-1] == 4.0
        assert r.signal("p", kind="I") is r.signals[("p", 0)][1]
        assert r.excitation_signal("p") is r.excitation_signals[("p", 0)]
        assert r.stop_reason is None
        with pytest.raises(KeyError, match="not recorded"):
            r.signal("q")
        with pytest.raises(KeyError, match="no excitation"):
            r.excitation_signal("q")
        with pytest.raises(ValueError, match="'V' or 'I'"):
            r.signal("p", kind="P")
        # A repr says what it is and what state it is in — never its arrays.
        text = repr(r)
        assert text.startswith("TDResult")
        assert "steps" in text
        assert "array(" not in text

    def test_power_waves_need_port_modes(self):
        with pytest.raises(ValueError, match="no port_modes"):
            self._result().a("p")
