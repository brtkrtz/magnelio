"""Continuing a truncated record past the end of the march (DD-272).

The matrix pencil on a free decay: the poles it recovers, the honesty
of its out-of-sample number, what it refuses, and the spectrum it
repairs.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio.post._extrapolate import (
    ExtrapolationReport,
    continuation,
    excitation_end,
    extend_records,
    fit_poles,
    residues,
)

DT = 1e-12
N = 600
FREQS = np.array([3.1e9, 5.4e9, 8.9e9])
TAUS = np.array([4e-9, 2.5e-9, 1.2e-9])
AMPS = np.array([1.0, 0.6, 0.35])
PHASES = np.array([0.3, 1.1, -0.7])


def ringdown(t, amps=AMPS, phases=PHASES):
    """Three lightly damped resonances — a cavity cut off far too early."""
    return sum(
        a * np.exp(-t / tau) * np.cos(2 * np.pi * f * t + p)
        for a, f, tau, p in zip(amps, FREQS, TAUS, phases)
    )


def _record(n=N, noise=0.0, seed=3, **kw):
    t = np.arange(n) * DT
    x = ringdown(t, **kw)
    if noise:
        x = x + noise * np.random.default_rng(seed).standard_normal(n)
    return x


def _dft(x, f_axis, dt):
    t = np.arange(np.size(x)) * dt
    return np.sum(np.asarray(x) * np.exp(-2j * np.pi * f_axis[:, None] * t[None, :]), axis=1) * dt


class TestThePoleFit:
    def test_the_order_comes_out_of_the_singular_values(self):
        """Three resonances are six poles — nobody has to say so."""
        z, order = fit_poles(_record(noise=3e-5), tol=1e-3)
        assert order == 6
        assert np.all(np.abs(z) <= 1.0)

    def test_the_resonances_are_recovered(self):
        z, _ = fit_poles(_record(noise=3e-5), tol=1e-3)
        s = np.log(z.astype(complex)) / DT
        got_f = np.sort(s[s.imag > 0].imag / (2 * np.pi))
        got_tau = np.array([-1 / s[s.imag > 0][i].real for i in np.argsort(s[s.imag > 0].imag)])
        np.testing.assert_allclose(got_f, FREQS, rtol=1e-3)
        np.testing.assert_allclose(got_tau, TAUS, rtol=5e-3)

    def test_an_asked_for_order_is_used(self):
        _z, order = fit_poles(_record(), order=4)
        assert order == 4

    def test_a_record_too_short_to_fit_is_refused(self):
        with pytest.raises(ValueError, match="too short"):
            fit_poles(np.zeros(5))

    def test_the_channels_share_one_pole_set(self):
        """Two ports see one structure: the poles are fitted jointly."""
        a = _record(noise=1e-5)
        b = _record(noise=1e-5, seed=9, amps=AMPS * np.array([0.2, 1.4, 0.8]))
        z, order = fit_poles(np.vstack([a, b]), tol=1e-3)
        assert order == 6
        s = np.log(z.astype(complex)) / DT
        np.testing.assert_allclose(np.sort(s[s.imag > 0].imag / (2 * np.pi)), FREQS, rtol=1e-3)


class TestTheContinuation:
    def test_it_repairs_the_truncated_spectrum(self):
        """The point of the exercise, in one number."""
        rec = _record(noise=3e-5)
        extended, report = extend_records([rec], DT, decay_db=80.0)
        f_axis = np.linspace(2e9, 10e9, 401)
        truth = ringdown(np.arange(np.size(extended[0])) * DT)
        ref = np.abs(_dft(truth, f_axis, DT))
        before = np.max(np.abs(np.abs(_dft(rec, f_axis, DT)) - ref)) / ref.max()
        after = np.max(np.abs(np.abs(_dft(extended[0], f_axis, DT)) - ref)) / ref.max()
        assert before > 0.5, "the truncated record must be visibly wrong for this to mean anything"
        assert after < 1e-3
        assert after < before / 100.0
        assert report.residual < 2e-2

    def test_the_report_names_the_resonances(self):
        _ext, report = extend_records([_record(noise=3e-5)], DT)
        assert isinstance(report, ExtrapolationReport)
        assert report.order == 6
        np.testing.assert_allclose(np.sort(report.frequencies), FREQS, rtol=1e-3)
        np.testing.assert_allclose(np.sort(report.decay_times), np.sort(TAUS), rtol=5e-3)
        assert "GHz" in repr(report)

    def test_it_stops_when_the_model_has_decayed(self):
        """The horizon is the slowest pole, not a fixed length."""
        _ext, short = extend_records([_record(noise=3e-5)], DT, decay_db=40.0)
        _ext, long = extend_records([_record(noise=3e-5)], DT, decay_db=80.0)
        assert long.added > short.added
        # 40 dB more of an exp(-t/4ns) decay is 4 ns * ln(100) ≈ 18.4 ns
        np.testing.assert_allclose(long.added - short.added, TAUS[0] * np.log(100.0), rtol=0.05)

    def test_the_cap_holds_and_says_so(self):
        _ext, report = extend_records([_record(noise=3e-5)], DT, decay_db=400.0, max_factor=3.0)
        assert report.added == pytest.approx(3.0 * N * DT, rel=1e-9)
        assert report.capped
        assert "capped" in repr(report)

    def test_an_uncapped_run_says_so_too(self):
        _ext, report = extend_records([_record(noise=3e-5)], DT, decay_db=60.0)
        assert not report.capped

    def test_every_record_is_continued_on_one_axis(self):
        a, b = _record(noise=1e-5), _record(noise=1e-5, seed=9)
        extended, _ = extend_records([a, b], DT)
        assert extended[0].size == extended[1].size > N
        np.testing.assert_array_equal(extended[0][:N], a)


class TestWhatItRefuses:
    def test_noise_is_reported_as_noise(self):
        """White noise has no small pole set; the out-of-sample number says so."""
        noise = np.random.default_rng(5).standard_normal(N)
        with pytest.warns(UserWarning, match="describing noise"):
            _ext, report = extend_records([noise], DT, tol=1e-2)
        assert report.residual > 0.3

    def test_a_window_too_short_is_refused(self):
        with pytest.raises(ValueError, match="no free decay|at least 16"):
            extend_records([_record(n=100)], DT, fit_from=90)

    def test_a_growing_record_has_no_decaying_pole(self):
        t = np.arange(200) * DT
        growing = np.exp(t / 1e-9) * np.cos(2 * np.pi * 3e9 * t)
        with pytest.raises(ValueError, match="no decaying pole"):
            extend_records([growing], DT, order=2)


class TestTheFitWindow:
    def test_the_window_opens_after_the_excitation(self):
        t = np.arange(500) * DT
        pulse = np.exp(-(((t - 50 * DT) / (10 * DT)) ** 2))
        end = excitation_end(pulse, floor_db=60.0)
        # sigma = 10 samples, 60 dB is sqrt(ln 1000) = 2.63 sigma past the peak
        assert 70 < end < 100
        assert np.all(np.abs(pulse[end:]) < 1e-3 * pulse.max())

    def test_a_silent_reference_opens_at_zero(self):
        assert excitation_end(np.zeros(50)) == 0

    def test_fitting_across_the_source_is_worse_than_after_it(self):
        """Why the window matters: the source's shape is not a resonance."""
        t = np.arange(N) * DT
        driven = np.exp(-(((t - 40 * DT) / (12 * DT)) ** 2)) * 4.0 + ringdown(t)
        start = excitation_end(np.exp(-(((t - 40 * DT) / (12 * DT)) ** 2)))
        _e, clean = extend_records([driven], DT, fit_from=start, tol=1e-3)
        with pytest.warns(UserWarning, match="describing noise"):
            _e, dirty = extend_records([driven], DT, fit_from=0, tol=1e-3)
        assert clean.residual < dirty.residual


class TestTheBuildingBlocks:
    def test_residues_and_continuation_reproduce_the_record(self):
        rec = _record()
        z, _ = fit_poles(rec, tol=1e-6)
        z = z[np.abs(z) < 1.0]
        r = residues(rec, z)
        # The model continues the record it was fitted to, seamlessly.
        tail = continuation(N, 50, z, r)
        truth = ringdown((N + np.arange(50)) * DT)
        np.testing.assert_allclose(tail, truth, atol=1e-4 * np.abs(rec).max())
