"""Unit tests for the signals package (Signal1D, waveforms)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from magnelio.signals.signal_1d import Signal1D
from magnelio.signals.waveforms import gaussian, modulated_gaussian, waveform_for_mode

# ======================================================================
# Signal1D
# ======================================================================


class TestSignal1D:
    def _make_signal(self, n=128, dt=1e-12):
        t = np.arange(n) * dt
        values = np.sin(2 * np.pi * 1e9 * t)
        return Signal1D(t=t, values=values, dt=dt, label="test")

    def test_construction(self):
        sig = self._make_signal()
        assert len(sig) == 128
        assert sig.dt == 1e-12
        assert sig.label == "test"

    def test_frozen(self):
        sig = self._make_signal()
        with pytest.raises(AttributeError):
            sig.dt = 2e-12  # type: ignore[misc]

    def test_frequency_axis(self):
        sig = self._make_signal(n=128, dt=1e-12)
        f = sig.f
        assert f[0] == 0.0
        # Nyquist should be 1/(2*dt) = 500 GHz
        assert f[-1] == pytest.approx(500e9, rel=1e-6)

    def test_spectrum_shape(self):
        sig = self._make_signal(n=128)
        spec = sig.spectrum
        assert len(spec) == 128 // 2 + 1

    def test_spectrum_cached(self):
        sig = self._make_signal()
        s1 = sig.spectrum
        s2 = sig.spectrum
        assert s1 is s2  # same object, not recomputed

    def test_at_frequencies_dc(self):
        """DC signal should have all energy at f=0."""
        dt = 1e-12
        n = 64
        t = np.arange(n) * dt
        values = np.ones(n)
        sig = Signal1D(t=t, values=values, dt=dt)
        # At DC the spectrum should be n (sum of all ones)
        result = sig.at_frequencies(np.array([0.0]))
        assert abs(result[0] - n) < 1e-10

    def test_at_frequencies_interpolation(self):
        sig = self._make_signal(n=256, dt=1e-12)
        f_target = np.linspace(0, 100e9, 50)
        result = sig.at_frequencies(f_target)
        assert result.shape == (50,)
        assert np.iscomplexobj(result)

    def test_addition(self):
        sig1 = self._make_signal()
        sig2 = self._make_signal()
        result = sig1 + sig2
        np.testing.assert_allclose(result.values, 2 * sig1.values)

    def test_subtraction(self):
        sig = self._make_signal()
        result = sig - sig
        np.testing.assert_allclose(result.values, 0.0, atol=1e-15)

    def test_scalar_multiply(self):
        sig = self._make_signal()
        result = sig * 3.0
        np.testing.assert_allclose(result.values, 3.0 * sig.values)

    def test_rmul(self):
        sig = self._make_signal()
        result = 2.0 * sig
        np.testing.assert_allclose(result.values, 2.0 * sig.values)

    def test_negation(self):
        sig = self._make_signal()
        result = -sig
        np.testing.assert_allclose(result.values, -sig.values)

    def test_repr(self):
        sig = self._make_signal()
        r = repr(sig)
        assert "Signal1D" in r
        assert "N=128" in r


# ======================================================================
# Waveforms
# ======================================================================


class TestWaveforms:
    def test_gaussian_peak(self):
        """Gaussian peaks at t0 = 4/f_max with value 1."""
        f_max = 1e9
        t0 = 4.0 / f_max
        assert gaussian(t0, f_max) == pytest.approx(1.0)

    def test_gaussian_zero_at_origin(self):
        """Gaussian is near zero at t=0 for reasonable f_max."""
        f_max = 1e9
        val = gaussian(0.0, f_max)
        assert abs(val) < 1e-6

    def test_gaussian_array(self):
        f_max = 1e9
        t = np.linspace(0, 10e-9, 100)
        result = gaussian(t, f_max)
        assert isinstance(result, np.ndarray)
        assert result.shape == (100,)

    def test_modulated_gaussian_peak(self):
        """Modulated Gaussian has peak envelope = 1 at t0 = 4/bandwidth."""
        f_max = 10e9
        f_min = 5e9
        bandwidth = f_max - f_min
        t0 = 4.0 / bandwidth
        # At t0 the envelope is 1, cos factor is cos(0) = 1
        val = modulated_gaussian(t0, f_max, f_min)
        assert val == pytest.approx(1.0)

    def test_modulated_gaussian_array(self):
        t = np.linspace(0, 2e-9, 50)
        result = modulated_gaussian(t, 10e9, 5e9)
        assert isinstance(result, np.ndarray)
        assert result.shape == (50,)

    def test_matches_port2d_tem(self):
        """gaussian() formula: bandwidth = f_max, t0 = 4/f_max."""
        f_max = 1e9
        sigma = 2.0 / (math.pi * f_max)
        t0 = 4.0 / f_max
        for t in [0.0, 1e-9, 2e-9, 4e-9, 6e-9]:
            x = (t - t0) / sigma
            expected = math.exp(-x * x)
            assert gaussian(t, f_max) == pytest.approx(expected, rel=1e-12)

    def test_matches_port2d_te(self):
        """modulated_gaussian() bandwidth = f_max - f_min, t0 = 4/bandwidth."""
        f_max = 10e9
        f_min = 5e9
        bandwidth = f_max - f_min
        sigma = 2.0 / (math.pi * bandwidth)
        t0 = 4.0 / bandwidth
        f_center = 0.5 * (f_min + f_max)
        for t in [0.0, 0.5e-9, 1e-9, 1.5e-9, 2e-9]:
            x = (t - t0) / sigma
            expected = math.exp(-x * x) * math.cos(2 * math.pi * f_center * (t - t0))
            assert modulated_gaussian(t, f_max, f_min) == pytest.approx(expected, rel=1e-12)

    def test_waveform_for_mode_tem(self):
        f_max = 1e9
        wf = waveform_for_mode(f_max, omega_c=0.0)
        t0 = 4.0 / f_max
        assert wf(t0) == pytest.approx(1.0)
        assert wf(0.0) == pytest.approx(gaussian(0.0, f_max))

    def test_waveform_for_mode_te(self):
        f_max = 10e9
        omega_c = 2 * math.pi * 5e9
        wf = waveform_for_mode(f_max, omega_c)
        f_cutoff = omega_c / (2 * math.pi)
        # waveform_for_mode forwards f_cutoff as the modulated-Gaussian f_min
        t0 = 4.0 / (f_max - f_cutoff)
        assert wf(t0) == pytest.approx(modulated_gaussian(t0, f_max, f_cutoff))

    def test_waveform_for_mode_f_min_overrides_dc(self):
        """f_min argument should produce a bandpass even for TEM modes."""
        f_max = 12.4e9
        f_min = 8.2e9
        wf = waveform_for_mode(f_max, omega_c=0.0, f_min=f_min)
        t0 = 4.0 / (f_max - f_min)
        # At t0 the envelope peaks → value is 1 (cos(0) = 1)
        assert wf(t0) == pytest.approx(1.0)
        # f_min=0 forces plain DC-Gaussian
        wf_dc = waveform_for_mode(f_max, omega_c=0.0, f_min=0.0)
        assert wf_dc(4.0 / f_max) == pytest.approx(1.0)

    def test_waveform_for_mode_zero_omega_uses_gaussian(self):
        """omega_c=0 should fall back to a plain Gaussian."""
        wf = waveform_for_mode(1e9, omega_c=0.0)
        t0 = 4.0 / 1e9
        assert wf(t0) == pytest.approx(1.0)
