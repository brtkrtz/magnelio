"""The energy plot shows the progress figure: dB below the peak over time."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from magnelio.post._plot_energy import plot_energy_traces


def _trace(energies, dt=1e-12, stride=100):
    trace = np.empty(len(energies), dtype=[("step", int), ("time", float), ("energy", float)])
    trace["step"] = np.arange(len(energies)) * stride
    trace["time"] = trace["step"] * dt
    trace["energy"] = energies
    return trace


class TestPlotEnergyTraces:
    def test_curve_peaks_at_zero_db_and_axes_read_time(self):
        fig, ax = plot_energy_traces({"run_1": _trace([0.1, 1.0, 0.01])})
        (line,) = ax.get_lines()
        y = line.get_ydata()
        assert y.max() == pytest.approx(0.0)
        assert y[0] == pytest.approx(-10.0)
        assert y[-1] == pytest.approx(-20.0)
        assert ax.get_xlabel() == "t / ns"
        assert line.get_xdata()[-1] == pytest.approx(200 * 1e-12 * 1e9)
        plt.close(fig)

    def test_step_axis_and_stop_line(self):
        fig, ax = plot_energy_traces({"a": _trace([1.0, 0.5])}, energy_stop_db=40, x="step")
        assert ax.get_xlabel() == "time step"
        assert len(ax.get_lines()) == 2  # the curve and the criterion
        assert ax.get_lines()[1].get_ydata()[0] == pytest.approx(-40.0)
        assert ax.get_legend() is not None
        plt.close(fig)

    def test_reuses_axes_and_labels_every_trace(self):
        fig, ax = plt.subplots()
        out_fig, out_ax = plot_energy_traces(
            {"p1:0": _trace([1.0, 0.1]), "p2:0": _trace([1.0, 0.2])}, ax=ax
        )
        assert out_fig is fig and out_ax is ax
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["p1:0", "p2:0"]
        plt.close(fig)

    def test_rejects_an_unknown_axis_and_skips_an_empty_trace(self):
        with pytest.raises(ValueError, match="x must be"):
            plot_energy_traces({"a": _trace([1.0])}, x="energy")
        with pytest.warns(UserWarning, match="empty"):
            fig, ax = plot_energy_traces({"a": _trace([])})
        assert ax.get_lines() == []
        plt.close(fig)


class TestPlotEnergyMethods:
    """The same picture off a result, a scattering result, a run and a project."""

    def test_td_result_draws_its_trace_and_criterion(self):
        from magnelio.analysis import TDResult
        from magnelio.analysis.result_interface import RunSettings

        result = TDResult(
            excitations=(),
            dt=1e-12,
            n_steps=300,
            signals={},
            excitation_signals={},
            energy_trace=_trace([0.5, 1.0, 0.01]),
            settings=RunSettings(energy_stop_db=40.0),
            name="run_1",
        )
        fig, ax = result.plot_energy(x="step")
        assert len(ax.get_lines()) == 2
        assert ax.get_lines()[0].get_label() == "run_1"
        plt.close(fig)

    def test_scattering_result_draws_one_curve_per_excitation(self):
        from magnelio.analysis import ScatteringTDResult
        from magnelio.post.sparameter_result import SParameterResult
        from magnelio.signals.signal_1d import Signal1D

        f = np.linspace(8e9, 12e9, 11)
        s_params = SParameterResult(
            f_axis=f,
            channels=(("p1", 0), ("p2", 0)),
            excitations=(("p1", 0), ("p2", 0)),
            matrix=np.zeros((11, 2, 2), dtype=complex),
        )
        ref = Signal1D(t=np.arange(4) * 1e-12, values=np.zeros(4), dt=1e-12, label="excitation")
        result = ScatteringTDResult(
            s_params=s_params,
            signals={},
            reference_signal=ref,
            dt=1e-12,
            n_actual_steps=300,
            energy_traces={("p1", 0): _trace([1.0, 0.1]), ("p2", 0): _trace([1.0, 0.2])},
        )
        fig, ax = result.plot_energy()
        labels = [line.get_label() for line in ax.get_lines()]
        assert labels == ["p1:0", "p2:0"]
        plt.close(fig)


class TestAxisFloor:
    """The empty grid's first samples must not drag the axis to −3000 dB."""

    def test_floor_sits_ten_db_below_the_criterion(self):
        fig, ax = plot_energy_traces({"a": _trace([0.0, 1e-33, 1.0, 1e-4])}, energy_stop_db=40)
        assert ax.get_ylim() == (-50.0, 5.0)
        plt.close(fig)

    def test_floor_follows_a_run_that_got_deeper(self):
        fig, ax = plot_energy_traces({"a": _trace([1.0, 1e-9])}, energy_stop_db=40)
        assert ax.get_ylim() == pytest.approx((-100.0, 5.0))
        plt.close(fig)

    def test_default_floor_without_a_criterion_and_explicit_floor(self):
        fig, ax = plot_energy_traces({"a": _trace([0.0, 1.0, 0.1])})
        assert ax.get_ylim() == (-100.0, 5.0)
        plt.close(fig)
        fig, ax = plot_energy_traces({"a": _trace([0.0, 1.0, 0.1])}, floor_db=-30)
        assert ax.get_ylim() == (-30.0, 5.0)
        plt.close(fig)
