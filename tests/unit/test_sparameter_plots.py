"""Balance, Smith and polar plots of an S-matrix (DD-271).

The three pictures an S-parameter result was missing: where the power
that does not come back out of the ports went, and the complex
trajectory behind the magnitude.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from magnelio.post.sparameter_result import SParameterResult  # noqa: E402


def _two_port(loss: float = 0.0, nf: int = 101) -> SParameterResult:
    """A reciprocal two-port: |S11|² + |S21|² = 1 − loss at every frequency."""
    f = np.linspace(1e9, 10e9, nf)
    w = 2 * np.pi * f
    g = 0.5 * np.exp(-1j * w / 3e9)
    t = np.sqrt(np.maximum(1.0 - loss - np.abs(g) ** 2, 0.0)) * np.exp(-1j * w / 1.5e9)
    m = np.zeros((f.size, 2, 2), dtype=complex)
    m[:, 0, 0] = m[:, 1, 1] = g
    m[:, 1, 0] = m[:, 0, 1] = t
    return SParameterResult(
        f_axis=f,
        channels=(("p1", 0), ("p2", 0)),
        excitations=(("p1", 0), ("p2", 0)),
        matrix=m,
    )


class TestBalance:
    def test_a_lossless_network_sums_to_one(self):
        res = _two_port()
        _fig, ax = res.plot_balance()
        traces = [
            np.asarray(ln.get_ydata())
            for ln in ax.get_lines()
            if np.size(ln.get_ydata()) == res.f_axis.size
        ]
        assert traces
        for y in traces:
            np.testing.assert_allclose(y, 1.0, atol=1e-12)

    def test_the_deficit_reads_the_loss(self):
        """A tenth of the power lost is −10 dB on the deficit trace."""
        res = _two_port(loss=0.1)
        _fig, ax = res.plot_balance(deficit=True)
        traces = [
            np.asarray(ln.get_ydata()) for ln in ax.get_lines() if np.size(ln.get_ydata()) == 101
        ]
        assert traces
        for y in traces:
            np.testing.assert_allclose(y, 10.0 * np.log10(0.1), atol=1e-9)

    def test_one_excitation_can_be_selected(self):
        res = _two_port()
        _fig, ax = res.plot_balance("p2")
        assert len(ax.get_lines()) == 2  # the trace and the unity line
        assert "p2" in ax.get_legend().get_texts()[0].get_text()

    def test_an_evanescent_channel_counts_as_zero(self):
        """NaN is a channel that carries no propagating mode: no active power."""
        res = _two_port()
        res.matrix[:, 1, 0] = np.nan
        _fig, ax = res.plot_balance("p1")
        y = ax.get_lines()[0].get_ydata()
        assert np.all(np.isfinite(y))
        np.testing.assert_allclose(y, 0.25, atol=1e-12)


class TestSmith:
    def test_the_trace_is_the_reflection_coefficient(self):
        res = _two_port()
        _fig, ax = res.plot_smith()
        traces = [ln for ln in ax.get_lines() if ln.get_label().startswith("S(")]
        assert len(traces) == 2
        x, y = traces[0].get_xdata(), traces[0].get_ydata()
        s11 = res.S("p1", "p1")
        np.testing.assert_allclose(x, np.real(s11))
        np.testing.assert_allclose(y, np.imag(s11))

    def test_the_grid_is_a_chart(self):
        """The r = 1 circle passes through the origin and through (1, 0)."""
        res = _two_port()
        _fig, ax = res.plot_smith(labels=False)
        pts = np.concatenate(
            [np.column_stack([ln.get_xdata(), ln.get_ydata()]) for ln in ax.get_lines()]
        )
        pts = pts[np.isfinite(pts).all(axis=1)]
        assert np.min(np.linalg.norm(pts - [0.0, 0.0], axis=1)) < 1e-6
        assert np.min(np.linalg.norm(pts - [1.0, 0.0], axis=1)) < 1e-6
        assert np.max(np.linalg.norm(pts, axis=1)) <= 1.0 + 1e-9

    def test_marks_land_on_the_named_frequencies(self):
        res = _two_port()
        _fig, ax = res.plot_smith((("p1", "p1")), mark=[5e9])
        marked = [ln for ln in ax.get_lines() if ln.get_marker() == "o"]
        assert marked
        i = int(np.argmin(np.abs(res.f_axis - 5e9)))
        s11 = res.S("p1", "p1")
        np.testing.assert_allclose(marked[0].get_xdata()[0], np.real(s11[i]))

    def test_a_moving_reference_warns(self):
        res = _two_port()
        z = np.linspace(40.0, 60.0, res.f_axis.size)
        res = SParameterResult(
            f_axis=res.f_axis,
            channels=res.channels,
            excitations=res.excitations,
            matrix=res.matrix,
            reference_impedances={("p1", 0): z, ("p2", 0): z},
        )
        with pytest.warns(UserWarning, match="varies with frequency"):
            res.plot_smith()


class TestPolar:
    def test_every_channel_is_drawn_by_default(self):
        res = _two_port()
        _fig, ax = res.plot_polar()
        traces = [ln for ln in ax.get_lines() if ln.get_label().startswith("S(")]
        assert len(traces) == 4

    def test_radius_is_the_magnitude_and_angle_the_phase(self):
        res = _two_port()
        _fig, ax = res.plot_polar((("p2", "p1")))
        line = [ln for ln in ax.get_lines() if ln.get_label().startswith("S(")][0]
        s21 = res.S("p2", "p1")
        np.testing.assert_allclose(line.get_ydata(), np.abs(s21))
        np.testing.assert_allclose(line.get_xdata(), np.angle(s21))

    def test_a_cartesian_axes_is_refused(self):
        import matplotlib.pyplot as plt

        res = _two_port()
        _fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="polar axes"):
            res.plot_polar(ax=ax)
