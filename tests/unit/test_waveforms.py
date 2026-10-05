"""Unit tests for the Waveform classes (DD-224)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from magnelio.signals import (
    Signal1D,
    Waveform,
    WaveformFunction,
    WaveformGaussian,
    WaveformGaussianModulated,
    WaveformSine,
    WaveformStep,
    WaveformTable,
)
from magnelio.signals.waveforms import gaussian, modulated_gaussian


class TestGaussian:
    def test_matches_closed_form(self):
        w = WaveformGaussian(f_max=10e9)
        t = np.linspace(0.0, 2e-9, 41)
        np.testing.assert_array_equal(w(t), gaussian(t, 10e9))
        for tt in t:
            assert w(float(tt)) == gaussian(float(tt), 10e9)
            assert isinstance(w(float(tt)), float)

    def test_band_and_duration(self):
        w = WaveformGaussian(f_max=10e9)
        assert w.f_max == 10e9
        assert w.f_min == 0.0
        assert w.f_center is None
        assert w.tau == pytest.approx(math.sqrt(25 * math.log(10) / 20) / (math.pi * 10e9))
        assert w.peak_time == 4.5 * w.tau
        assert w.t_end == 2 * w.peak_time
        assert w(w.t_end) == pytest.approx(math.exp(-(4.5**2)))
        assert w(w.peak_time) == pytest.approx(1.0)

    def test_validation(self):
        with pytest.raises(ValueError, match="f_max"):
            WaveformGaussian(f_max=0.0)
        with pytest.raises(ValueError, match="f_max"):
            WaveformGaussian(f_max=math.inf)

    def test_frozen_and_hashable(self):
        w = WaveformGaussian(f_max=10e9)
        with pytest.raises(AttributeError):
            w.f_max = 1.0  # type: ignore[misc]
        assert w == WaveformGaussian(f_max=10e9)
        assert hash(w) == hash(WaveformGaussian(f_max=10e9))

    def test_analytic_spectrum_matches_numeric(self):
        w = WaveformGaussian(f_max=10e9)
        f = np.linspace(0.0, 15e9, 7)
        analytic = w.spectrum(f)
        numeric = Waveform.spectrum(w, f)
        # The sampled finite interval omits envelope tails of about 1.6e-9.
        np.testing.assert_allclose(analytic, numeric, rtol=1e-8, atol=5e-10 * abs(analytic).max())
        # The spectrum meets the documented amplitude ratio at the upper edge.
        assert abs(w.spectrum(np.array([10e9]))[0]) / abs(w.spectrum(np.array([0.0]))[0]) == (
            pytest.approx(10 ** (-25 / 20))
        )

    def test_sample_is_signal(self):
        w = WaveformGaussian(f_max=10e9)
        sig = w.sample(1e-12, 50, label="exc")
        assert isinstance(sig, Signal1D)
        assert sig.dt == 1e-12 and len(sig) == 50 and sig.label == "exc"
        np.testing.assert_array_equal(sig.values, w(np.arange(50) * 1e-12))
        with pytest.raises(ValueError):
            w.sample(1e-12, 0)


class TestGaussianModulated:
    def test_matches_closed_form(self):
        w = WaveformGaussianModulated(f_min=8.2e9, f_max=12.4e9)
        t = np.linspace(0.0, 4e-9, 81)
        np.testing.assert_array_equal(w(t), modulated_gaussian(t, 12.4e9, 8.2e9))
        assert w(1e-9) == modulated_gaussian(1e-9, 12.4e9, 8.2e9)

    def test_band_and_duration(self):
        w = WaveformGaussianModulated(f_min=8.2e9, f_max=12.4e9)
        assert w.f_center == pytest.approx(10.3e9)
        assert w.peak_time == 4.5 * w.tau
        assert w.t_end == 2 * w.peak_time
        assert abs(w(w.t_end)) <= math.exp(-(4.5**2)) * (1 + 1e-12)
        assert w(w.peak_time) == pytest.approx(1.0)

    def test_validation(self):
        with pytest.raises(ValueError, match="exceed"):
            WaveformGaussianModulated(f_min=10e9, f_max=10e9)
        with pytest.raises(ValueError, match="f_min"):
            WaveformGaussianModulated(f_min=-1.0, f_max=10e9)

    def test_analytic_spectrum_matches_numeric(self):
        w = WaveformGaussianModulated(f_min=8.2e9, f_max=12.4e9)
        f = np.linspace(0.0, 20e9, 9)
        analytic = w.spectrum(f)
        numeric = Waveform.spectrum(w, f)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-8, atol=5e-10 * abs(analytic).max())
        s_c = abs(w.spectrum(np.array([w.f_center]))[0])
        assert abs(w.spectrum(np.array([w.f_max]))[0]) / s_c == pytest.approx(10 ** (-25 / 20))
        assert abs(w.spectrum(np.array([2e9]))[0]) / s_c < 1e-6


@pytest.mark.parametrize("attenuation", [0.1, 1.0, 6.0, 25.0, 60.0])
@pytest.mark.parametrize("ratio", [0.0, 1e-8, 0.1, 0.5, 0.9999])
def test_modulated_full_spectrum_edge(attenuation, ratio):
    w = WaveformGaussianModulated(ratio * 1e9, 1e9, edge_attenuation_db=attenuation)
    f = np.linspace(w.f_min, w.f_max, 501)
    ratios = abs(w.spectrum(f)) / abs(w.spectrum(w.f_center))
    target = 10 ** (-attenuation / 20)
    assert ratios[-1] == pytest.approx(target, rel=1e-12)
    assert min(ratios) >= target * (1 - 1e-12)


@pytest.mark.parametrize(
    "factory", [WaveformGaussian, lambda f, **kw: WaveformGaussianModulated(0, f, **kw)]
)
@pytest.mark.parametrize("invalid", [0, -1, math.inf, math.nan])
def test_edge_attenuation_validation(factory, invalid):
    with pytest.raises(ValueError, match="edge_attenuation_db"):
        factory(1e9, edge_attenuation_db=invalid)


@pytest.mark.parametrize("modulated", [False, True])
def test_nondefault_width_matches_numerical_time_integral(modulated):
    from scipy.integrate import quad

    w = (
        WaveformGaussianModulated(0.01e9, 1e9, edge_attenuation_db=6)
        if modulated
        else WaveformGaussian(1e9, edge_attenuation_db=6)
    )
    # Integrate the actual time function, independently of its closed transform.
    for f in [0.0, 0.3e9, 1e9]:

        def integrand(u):
            t = w.peak_time + w.tau * u
            return w(t) * np.exp(-2j * math.pi * f * t)

        numeric = w.tau * (
            quad(lambda u: integrand(u).real, -4.5, 4.5, epsabs=1e-12)[0]
            + 1j * quad(lambda u: integrand(u).imag, -4.5, 4.5, epsabs=1e-12)[0]
        )
        assert abs(numeric - w.spectrum(f)) < 5e-10 * abs(w.spectrum(w.f_center or 0.0))


def test_scalar_evaluation_uses_cached_width(monkeypatch):
    w = WaveformGaussianModulated(1e6, 1e9)
    monkeypatch.setattr(
        "magnelio.signals.waveforms.brentq", lambda *a, **kw: pytest.fail("recomputed width")
    )
    assert w(w.peak_time) == 1.0
    assert np.all(np.isfinite(w(np.linspace(0, w.t_end, 10))))


class TestSine:
    def test_values_and_causality(self):
        w = WaveformSine(f=1e9, phase_deg=90.0)
        assert w(0.0) == pytest.approx(1.0)  # sin(90°)
        assert w(-1e-12) == 0.0
        t = np.array([-1e-9, 0.25e-9, 0.5e-9])
        np.testing.assert_allclose(w(t), [0.0, 0.0, -1.0], atol=1e-12)

    def test_cw_attributes(self):
        w = WaveformSine(f=1e9)
        assert w.f_max == w.f_min == w.f_center == 1e9
        assert w.t_end == math.inf
        with pytest.raises(ValueError, match="continuous-wave"):
            w.spectrum(np.array([1e9]))

    def test_raised_cosine_rise(self):
        w = WaveformSine(f=1e9, rise_time=2e-9)
        # The envelope reaches 1/2 at half the rise time and 1 after it.
        t_half = 1e-9
        assert w(t_half) == pytest.approx(0.5 * math.sin(2 * math.pi * 1e9 * t_half))
        t_late = 2.25e-9
        assert w(t_late) == pytest.approx(math.sin(2 * math.pi * 1e9 * t_late))

    def test_validation(self):
        with pytest.raises(ValueError, match="rise_time"):
            WaveformSine(f=1e9, rise_time=0.0)


class TestStep:
    def test_step_shape(self):
        w = WaveformStep(rise_time=1e-10)
        assert w(-1e-12) == 0.0
        assert w(0.0) == 0.0
        assert w(0.5e-10) == pytest.approx(0.5)
        assert w(1e-10) == pytest.approx(1.0)
        assert w(5e-9) == 1.0
        assert w.t_end == math.inf
        assert w.f_max == pytest.approx(1e10)
        assert w.f_min == 0.0 and w.f_center is None

    def test_pulse_shape(self):
        w = WaveformStep(rise_time=1e-10, hold=1e-9)
        assert w.fall_time == 1e-10
        assert w.t_end == pytest.approx(1.2e-9)
        t = np.array([0.0, 0.5e-10, 1e-10, 5e-10, 1.1e-9, 1.15e-9, 1.2e-9, 2e-9])
        np.testing.assert_allclose(w(t), [0, 0.5, 1, 1, 1, 0.5, 0, 0], atol=1e-12)
        assert w.spectrum(np.array([0.0]))[0].real == pytest.approx(1.1e-9, rel=1e-3)

    def test_validation(self):
        with pytest.raises(ValueError, match="hold"):
            WaveformStep(rise_time=1e-10, fall_time=1e-10)


class TestTable:
    def test_interpolates_and_is_zero_outside(self):
        t = np.linspace(0.0, 2e-9, 201)
        ref = WaveformGaussian(f_max=5e9)
        w = WaveformTable(t=t, values=ref(t))
        assert w(1e-9) == pytest.approx(ref(1e-9), rel=1e-6)
        assert w(0.5 * (t[10] + t[11])) == pytest.approx(0.5 * (ref(t[10]) + ref(t[11])))
        assert w(-1e-12) == 0.0 and w(3e-9) == 0.0
        assert w.t_end == 2e-9
        # The estimated band edge is where the table's spectrum falls 40 dB.
        assert 3e9 < w.f_max < 8e9

    def test_explicit_band(self):
        t = np.linspace(0.0, 1e-9, 11)
        w = WaveformTable(t=t, values=np.ones(11), f_max=3e9, f_min=1e9, f_center=2e9)
        assert (w.f_max, w.f_min, w.f_center) == (3e9, 1e9, 2e9)

    def test_validation(self):
        with pytest.raises(ValueError, match="increasing"):
            WaveformTable(t=[0.0, 1e-9, 1e-9], values=[0, 1, 0])
        with pytest.raises(ValueError, match="same length"):
            WaveformTable(t=[0.0, 1e-9], values=[0, 1, 0])
        with pytest.raises(ValueError, match="start at or after 0"):
            WaveformTable(t=[-1e-9, 1e-9], values=[0, 1])
        with pytest.raises(ValueError, match="all zero"):
            WaveformTable(t=[0.0, 1e-9], values=[0.0, 0.0])


class TestFunction:
    def test_scalar_function_broadcasts(self):
        w = WaveformFunction(lambda t: math.sin(1e10 * t), f_max=2e9)
        assert w(1e-10) == pytest.approx(math.sin(1.0))
        t = np.array([1e-10, 2e-10])
        np.testing.assert_allclose(w(t), np.sin(1e10 * t))
        assert w.t_end == math.inf and w.f_min == 0.0 and w.f_center is None

    def test_array_function_used_directly(self):
        calls = []

        def fn(t):
            calls.append(np.ndim(t))
            return np.exp(-((t - 1e-9) ** 2) / 1e-20)

        w = WaveformFunction(fn, f_max=2e9, t_end=2e-9)
        w(np.linspace(0, 2e-9, 5))
        assert calls == [1]
        assert abs(w.spectrum(np.array([0.0]))[0]) > 0.0

    def test_validation(self):
        with pytest.raises(TypeError, match="callable"):
            WaveformFunction(3.0, f_max=1e9)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="f_max"):
            WaveformFunction(lambda t: 0.0, f_max=0.0)
        with pytest.raises(ValueError, match="t_end"):
            WaveformFunction(lambda t: 0.0, f_max=1e9, t_end=-1.0)


class TestABC:
    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            Waveform()  # type: ignore[abstract]

    def test_all_are_waveforms(self):
        for w in (
            WaveformGaussian(1e9),
            WaveformGaussianModulated(1e9, 2e9),
            WaveformSine(1e9),
            WaveformStep(1e-10),
            WaveformTable([0.0, 1e-9], [0.0, 1.0], f_max=1e9),
            WaveformFunction(lambda t: 0.0, f_max=1e9),
        ):
            assert isinstance(w, Waveform)
            assert callable(w)

    def test_plot_returns_axes(self):
        import matplotlib

        matplotlib.use("Agg")
        ax = WaveformGaussianModulated(1e9, 2e9).plot()
        assert ax.get_xlabel() == "t [ns]"
        ax = WaveformSine(1e9).plot(ax=ax)
        assert len(ax.lines) == 2
