"""Tests for the trapezoidal RLC companion models (Cluster 3, 3b).

Each element is exercised through a mini source-resistor circuit that mirrors
the discrete-port solve exactly — a step source ``Vs`` behind a series
resistance ``Rs`` (the analogues of ``v_src`` and ``Σβ``)::

    i = (Vs − v_hist) / (Rs + r_eq)
    V_element = r_eq·i + v_hist
    element.advance(i, V_element, dt)

so the transients validate both the companion algebra and the update
structure the solver will use.  Comparisons are against closed-form analytic
responses.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from magnelio.circuit.companion import ParallelRLC, SeriesRLC


def _drive(element, Rs, Vs_of_t, dt, n_steps):
    """Run the mini source-resistor circuit; return (t, i, v_element)."""
    i_arr = np.empty(n_steps)
    v_arr = np.empty(n_steps)
    for n in range(n_steps):
        t = (n + 1) * dt  # element solved at t^{n+1}
        vh = element.v_hist(dt)
        req = element.r_eq(dt)
        i = (Vs_of_t(t) - vh) / (Rs + req)
        v = req * i + vh
        element.advance(i, v, dt)
        i_arr[n] = i
        v_arr[n] = v
    t_arr = (np.arange(n_steps) + 1) * dt
    return t_arr, i_arr, v_arr


def test_reqs_and_hist_match_analytic():
    dt = 2e-12
    assert SeriesRLC(R=50.0).r_eq(dt) == pytest.approx(50.0)
    assert SeriesRLC(L=1e-8).r_eq(dt) == pytest.approx(2 * 1e-8 / dt)
    assert SeriesRLC(C=1e-12).r_eq(dt) == pytest.approx(dt / (2 * 1e-12))
    assert SeriesRLC(R=10, L=1e-8, C=1e-12).r_eq(dt) == pytest.approx(
        10 + 2 * 1e-8 / dt + dt / (2 * 1e-12)
    )
    # a pure resistor has no history
    assert SeriesRLC(R=50.0).v_hist(dt) == 0.0
    # parallel R only → r_eq = R, no history
    assert ParallelRLC(R=50.0).r_eq(dt) == pytest.approx(50.0)
    assert ParallelRLC(R=50.0).v_hist(dt) == 0.0


def _step_error(element_factory, Rs, Vs, tau, exact_of_t, dt, *, use_v=False):
    """Max |numeric − analytic| / scale over 6τ, for a hard-step source.

    A hard step is the trapezoidal rule's worst case (an O(dt) response at
    the discontinuity that decays over τ), so correctness is asserted by
    *convergence* — the error must fall ~proportionally as dt shrinks.
    """
    n = int(round(6.0 * tau / dt))
    elem = element_factory()
    t, i, v = _drive(elem, Rs, lambda _t: Vs, dt, n)
    got = v if use_v else i
    return float(np.max(np.abs(got - exact_of_t(t))))


def test_series_rl_step_converges_to_exponential():
    """Series RL, step Vs: i(t) → i∞(1 − e^{−t/τ}); error converges O(dt)."""
    Vs, Rs, R, L = 1.0, 50.0, 50.0, 1e-8
    tau = L / (Rs + R)
    i_inf = Vs / (Rs + R)
    exact = lambda t: i_inf * (1.0 - np.exp(-t / tau))
    e_coarse = _step_error(lambda: SeriesRLC(R=R, L=L), Rs, Vs, tau, exact, tau / 50)
    e_fine = _step_error(lambda: SeriesRLC(R=R, L=L), Rs, Vs, tau, exact, tau / 200)
    assert e_fine < 0.4 * e_coarse  # ~1st-order convergence at the step
    assert e_fine < 5e-3 * i_inf  # absolute sanity


def test_series_rc_step_converges_to_exponential():
    """Series RC, step Vs: i(t) → i0·e^{−t/τ}; error converges O(dt)."""
    Vs, Rs, R, C = 1.0, 50.0, 50.0, 1e-12
    tau = (Rs + R) * C
    i0 = Vs / (Rs + R)
    exact = lambda t: i0 * np.exp(-t / tau)
    e_coarse = _step_error(lambda: SeriesRLC(R=R, C=C), Rs, Vs, tau, exact, tau / 50)
    e_fine = _step_error(lambda: SeriesRLC(R=R, C=C), Rs, Vs, tau, exact, tau / 200)
    assert e_fine < 0.4 * e_coarse
    assert e_fine < 5e-3 * i0


def test_series_rlc_underdamped_ringing_frequency():
    """Underdamped series RLC: current rings at ω_d = √(1/LC − (R_tot/2L)²)."""
    Vs, Rs, R, L, C = 1.0, 5.0, 5.0, 1e-8, 1e-12
    R_tot = Rs + R
    w0 = 1.0 / math.sqrt(L * C)
    alpha = R_tot / (2.0 * L)
    assert alpha < w0, "test fixture must be underdamped"
    wd = math.sqrt(w0 * w0 - alpha * alpha)
    dt = (2 * math.pi / wd) / 200.0  # ~200 steps/period
    n = 4000
    elem = SeriesRLC(R=R, L=L, C=C)
    t, i, _ = _drive(elem, Rs, lambda _t: Vs, dt, n)

    # dominant ringing frequency from the current spectrum
    I = np.abs(np.fft.rfft(i - i.mean()))
    freqs = np.fft.rfftfreq(n, dt)
    f_peak = freqs[np.argmax(I)]
    assert f_peak == pytest.approx(wd / (2 * math.pi), rel=0.02)


def test_parallel_rc_step_matches_exponential():
    """Parallel RC through Rs, step Vs: V(t) = V∞(1 − e^{−t/τ}).

    Element voltage V across the parallel R‖C: V∞ = Vs·R/(R+Rs),
    τ = Rs·R·C/(R+Rs).
    """
    Vs, Rs, R, C = 1.0, 50.0, 100.0, 1e-12
    v_inf = Vs * R / (R + Rs)
    tau = Rs * R * C / (R + Rs)
    exact = lambda t: v_inf * (1.0 - np.exp(-t / tau))
    e_coarse = _step_error(lambda: ParallelRLC(R=R, C=C), Rs, Vs, tau, exact, tau / 50, use_v=True)
    e_fine = _step_error(lambda: ParallelRLC(R=R, C=C), Rs, Vs, tau, exact, tau / 200, use_v=True)
    assert e_fine < 0.4 * e_coarse
    assert e_fine < 5e-3 * v_inf


def test_reset_and_state_roundtrip():
    dt = 2e-12
    elem = SeriesRLC(R=10.0, L=1e-8, C=1e-12)
    _drive(elem, 50.0, lambda _t: 1.0, dt, 50)
    saved = elem.state_dict()
    assert elem.v_hist(dt) != 0.0
    elem.reset()
    assert elem.v_hist(dt) == 0.0
    elem.load_state_dict(saved)
    assert elem.state_dict() == saved


def test_validation_errors():
    with pytest.raises(ValueError, match="at least one"):
        SeriesRLC()
    with pytest.raises(ValueError, match="positive"):
        SeriesRLC(R=-1.0)
    with pytest.raises(ValueError, match="at least one"):
        ParallelRLC()
    with pytest.raises(ValueError, match="positive"):
        ParallelRLC(C=0.0)
