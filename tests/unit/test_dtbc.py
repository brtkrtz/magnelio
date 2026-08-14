"""Unit tests for the exact discrete transparent boundary condition.

Validates the WP-R2 production core (``magnelio.ports._modal.dtbc``)
against the analytic properties established at the WP-R1 gate:

* the outgoing-root symbol (branch selection, quadratic identity,
  passband modulus);
* the contour-integration kernel (``l_0 = 0``, symbol reconstruction);
* the a-priori reflection bound (decay with kernel length);
* the discrete de-stagger exponent (dispersion relation, continuum
  limit);
* the :class:`DTBCTermination` state (reference-convolution equality,
  chain transparency at the -100 dB level, exact ghost-plane
  injection, kernel auto-extension).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from magnelio.ports._modal.dtbc import (
    DTBCTermination,
    destagger_theta,
    dtbc_kernel,
    dtbc_wave_impedance,
    lambda_symbol,
    reflection_bound,
)

WORKING_POINTS = [(0.30, 0.0), (0.70, 0.0), (0.99, 0.0), (0.90, 0.20)]


def _band(r: float, q: float) -> tuple[float, float]:
    """Evaluation band [1.01 w_c, w at beta*dz = pi/2] as w*dt."""
    w_c = 2.0 * math.asin(q / 2.0) if q > 0 else 0.0
    w_half = 2.0 * math.asin(math.sqrt(r * r * 0.5 + q * q / 4.0))
    return max(1.01 * w_c, 0.02), w_half


class TestLambdaSymbol:
    @pytest.mark.parametrize("r,q", WORKING_POINTS)
    def test_outgoing_branch_bounded(self, r, q):
        z = 1.3 * np.exp(1j * np.linspace(0.0, 2.0 * np.pi, 97))
        lam = lambda_symbol(z, r, q)
        assert np.all(np.abs(lam) <= 1.0 + 1e-12)

    @pytest.mark.parametrize("r,q", WORKING_POINTS)
    def test_quadratic_identity(self, r, q):
        z = (1.0 + 1e-8) * np.exp(1j * np.linspace(0.01, np.pi, 51))
        lam = lambda_symbol(z, r, q)
        A = 1.0 + (z - 2.0 + 1.0 / z + q * q) / (2.0 * r * r)
        resid = lam * lam - 2.0 * A * lam + 1.0
        assert np.max(np.abs(resid)) < 1e-10

    def test_passband_unimodular_evanescent_contracting(self):
        r, q = 0.9, 0.2
        w_c = 2.0 * math.asin(q / 2.0)
        z_pass = (1.0 + 1e-12) * np.exp(
            1j
            * np.linspace(
                1.05 * w_c,
                2.0 * math.asin(math.sqrt(r * r + q * q / 4.0)) * 0.98,
                25,
            )
        )
        lam_pass = lambda_symbol(z_pass, r, q)
        assert np.allclose(np.abs(lam_pass), 1.0, atol=1e-6)
        z_evan = (1.0 + 1e-12) * np.exp(1j * np.linspace(0.1, 0.9, 9) * w_c)
        lam_evan = lambda_symbol(z_evan, r, q)
        assert np.all(np.abs(lam_evan) < 1.0 - 1e-4)

    def test_tem_dc_limit(self):
        lam = lambda_symbol(np.array([1.0 + 1e-8 + 0j]), 0.7, 0.0)
        assert abs(lam[0] - 1.0) < 1e-3


class TestKernel:
    @pytest.mark.parametrize("r,q", WORKING_POINTS)
    def test_l0_vanishes(self, r, q):
        kern = dtbc_kernel(r, q, 1024)
        assert abs(kern[0]) < 1e-12

    @pytest.mark.parametrize("r,q", WORKING_POINTS)
    def test_reconstructs_symbol_off_circle(self, r, q):
        kern = dtbc_kernel(r, q, 4096)
        z = 1.05 * np.exp(1j * np.linspace(0.0, np.pi, 17))
        m = np.arange(kern.size)
        lam_t = np.array([np.sum(kern * zz ** (-m)) for zz in z])
        lam = lambda_symbol(z, r, q)
        assert np.max(np.abs(lam_t - lam)) < 1e-12

    def test_reflection_bound_decays_with_length(self):
        r, q = 0.7, 0.0
        w_lo, w_hi = _band(r, q)
        w = np.linspace(w_lo, w_hi, 61)
        g_small = reflection_bound(r, q, dtbc_kernel(r, q, 2048), w).max()
        g_large = reflection_bound(r, q, dtbc_kernel(r, q, 16384), w).max()
        # n^{-3/2} predicts a factor 8^{3/2} ~ 22.6; require > 10.
        assert g_small / g_large > 10.0


class TestDestaggerTheta:
    @pytest.mark.parametrize("r,q", WORKING_POINTS)
    def test_matches_discrete_dispersion_in_passband(self, r, q):
        w_lo, w_hi = _band(r, q)
        w = np.linspace(w_lo * 1.05, w_hi * 0.95, 21)
        theta = destagger_theta(w, r, q)
        # sin(w dt/2) = sqrt(r^2 sin^2(beta dz/2) + q^2/4)
        beta_dz = 2.0 * np.arcsin(
            np.sqrt(
                (np.sin(w / 2.0) ** 2 - q * q / 4.0) / (r * r),
            )
        )
        assert np.max(np.abs(theta.real)) < 1e-6
        assert np.max(np.abs(2.0 * theta.imag - beta_dz)) < 1e-6

    def test_continuum_limit_tem(self):
        r = 0.6
        w = np.array([1e-3])
        theta = destagger_theta(w, r, 0.0)
        # gamma*dz/2 = i*(w dt)/(2 r) for the continuum TEM line.
        assert abs(theta[0] - 1j * w[0] / (2.0 * r)) < 1e-7

    def test_evanescent_real_positive(self):
        r, q = 0.9, 0.2
        w_c = 2.0 * math.asin(q / 2.0)
        theta = destagger_theta(np.array([0.5 * w_c]), r, q)
        assert theta[0].real > 0.0
        assert abs(theta[0].imag) < 1e-6


class TestDTBCWaveImpedance:
    """Discrete wave impedance closed form (WP-R3)."""

    @pytest.mark.parametrize("r,q", [(0.70, 0.10), (0.90, 0.20), (0.56, 0.0846)])
    @pytest.mark.parametrize("kind", ["TE", "TM"])
    def test_matches_lambda_based_evaluation(self, r, q, kind):
        """s/rad equals sin(w dt/2)/(r sin(beta_hat dz/2)) with
        beta_hat from the lambda symbol — one analytic function."""
        w_lo, w_hi = _band(r, q)
        w = np.linspace(w_lo * 1.05, w_hi * 0.95, 25)
        z0 = 3.7
        Z = dtbc_wave_impedance(w, q, z0, kind)
        theta = destagger_theta(w, r, q)
        sin_b2 = np.sin(theta.imag)  # sin(beta_hat dz / 2)
        shape = np.sin(w / 2.0) / (r * sin_b2)
        Z_ref = z0 * shape if kind == "TE" else z0 / shape
        assert np.max(np.abs(Z / Z_ref - 1.0)) < 1e-9

    def test_te_continuum_shape_at_fine_sampling(self):
        """For w dt << 1 the discrete form approaches the continuum
        omega/sqrt(omega^2 - omega_c^2) shape."""
        q = 1e-3
        w = np.array([1.5e-3, 3e-3, 8e-3])
        Z = dtbc_wave_impedance(w, q, 1.0, "TE").real
        cont = w / np.sqrt(w**2 - q**2)
        assert np.max(np.abs(Z / cont - 1.0)) < 1e-5

    def test_below_cutoff_reactance_signs(self):
        """Evanescent branch: Z_TE inductive (+j), Z_TM capacitive
        (-j) — the continuum reactances."""
        q = 0.2
        w = np.array([0.5 * 2.0 * math.asin(q / 2.0)])
        z_te = dtbc_wave_impedance(w, q, 1.0, "TE")[0]
        z_tm = dtbc_wave_impedance(w, q, 1.0, "TM")[0]
        assert abs(z_te.real) < 1e-12 and z_te.imag > 0.0
        assert abs(z_tm.real) < 1e-12 and z_tm.imag < 0.0

    def test_rejects_unknown_kind(self):
        with pytest.raises(ValueError, match="mode_kind"):
            dtbc_wave_impedance(np.array([0.3]), 0.1, 1.0, "TEM")


def _run_chain_with_dtbc(
    K: int,
    n_steps: int,
    r: float,
    q: float,
    src_interior: np.ndarray,
    k_src: int,
    probes: tuple[int, ...],
    src_ghost: np.ndarray | None = None,
    n_kernel_init: int = 4096,
) -> np.ndarray:
    """Leapfrog chain, sites 0..K-1 interior, site K = DTBCTermination.

    Mirrors the production ordering: the interior update to t^{n+1}
    sees the boundary value u_K^n, then the boundary advances using
    u_{K-1}^n.  The left end (site 0) also carries a DTBC so left-going
    energy leaves the domain.
    """
    term_r = DTBCTermination(r, q, n_kernel_init=n_kernel_init)
    term_l = DTBCTermination(r, q, n_kernel_init=n_kernel_init)
    u_prev = np.zeros(K)
    u = np.zeros(K)
    out = np.zeros((n_steps, len(probes)))
    probes_arr = np.array(probes, dtype=int)
    r2, q2 = r * r, q * q
    for n in range(n_steps):
        lap = np.empty(K)
        lap[1:-1] = u[2:] - 2.0 * u[1:-1] + u[:-2]
        lap[0] = u[1] - 2.0 * u[0] + term_l.u_boundary
        lap[-1] = term_r.u_boundary - 2.0 * u[-1] + u[-2]
        u_next = 2.0 * u - u_prev + r2 * lap - q2 * u
        u_next[k_src] += src_interior[n]
        s_n = 0.0 if src_ghost is None else float(src_ghost[n])
        term_r.advance(u[-1], s_n)
        term_l.advance(u[0], 0.0)
        u_prev, u = u, u_next
        out[n] = u[probes_arr]
    return out


def _bandlimited_pulse(n_steps: int, w_lo: float, w_hi: float) -> np.ndarray:
    """Second-differenced modulated Gaussian (R1 spike drive)."""
    n = np.arange(n_steps)
    w0 = 0.5 * (w_lo + w_hi)
    sig = 6.0 / max(w_hi - w_lo, 1e-9)
    t0 = 6.0 * sig
    env = np.exp(-0.5 * ((n - t0) / sig) ** 2)
    pulse = env * np.cos(w0 * (n - t0))
    out = np.zeros(n_steps)
    out[2:] = np.diff(pulse, 2)
    return out


class TestDTBCTermination:
    def test_advance_matches_reference_convolution(self):
        r, q = 0.8, 0.1
        rng = np.random.default_rng(3)
        n_steps = 200
        u_int = rng.standard_normal(n_steps)
        src = rng.standard_normal(n_steps)
        term = DTBCTermination(r, q, n_kernel_init=4096)
        kern = dtbc_kernel(r, q, 4096)

        u_hist = [0.0]  # u_K^0
        w_hist: list[float] = []
        s_hist: list[float] = []
        u_prev = 0.0
        got = np.empty(n_steps)
        for n in range(n_steps):
            u_inc = sum(kern[m] * s_hist[n - m] for m in range(1, n + 1))
            ghost = src[n] + sum(kern[m] * w_hist[n - m] for m in range(1, n + 1))
            w_hist.append(u_hist[n] - u_inc)
            s_hist.append(src[n])
            u_new = (
                2.0 * u_hist[n]
                - u_prev
                + r * r * (ghost - 2.0 * u_hist[n] + u_int[n])
                - q * q * u_hist[n]
            )
            u_prev = u_hist[n]
            u_hist.append(u_new)
            got[n] = term.advance(u_int[n], src[n])
        assert np.allclose(got, u_hist[1:], rtol=0.0, atol=1e-12)

    @pytest.mark.parametrize("r,q", [(0.70, 0.0), (0.99, 0.0), (0.90, 0.20)])
    def test_chain_transparency_minus_100_db(self, r, q):
        w_lo, w_hi = _band(r, q)
        K, n_steps = 120, 3000
        src = np.zeros(n_steps)
        src[:2048] = _bandlimited_pulse(2048, max(w_lo, 1.05 * w_lo), w_hi)
        probe = (K - 8,)

        rec = _run_chain_with_dtbc(K, n_steps, r, q, src, 40, probe)

        # Reference: same source and probe geometry embedded in a chain
        # long enough that no end reflection reaches the probe within
        # n_steps (the infinite line the DTBC domain emulates on BOTH
        # sides).
        K_ref = 2 * (K + n_steps) + 16
        c = K_ref // 2
        k_src_ref, k_probe_ref = c, c + (probe[0] - 40)
        u_prev = np.zeros(K_ref)
        u = np.zeros(K_ref)
        ref = np.zeros(n_steps)
        r2, q2 = r * r, q * q
        for n in range(n_steps):
            lap = np.zeros(K_ref)
            lap[1:-1] = u[2:] - 2.0 * u[1:-1] + u[:-2]
            u_next = 2.0 * u - u_prev + r2 * lap - q2 * u
            u_next[k_src_ref] += src[n]
            u_prev, u = u, u_next
            ref[n] = u[k_probe_ref]

        err = np.max(np.abs(rec[:, 0] - ref))
        peak = np.max(np.abs(ref))
        assert peak > 0.0
        assert err / peak < 1e-5, f"boundary reflection {20 * math.log10(err / peak):.1f} dB"

    def test_ghost_injection_launches_and_leaves_cleanly(self):
        r, q = 0.7, 0.0
        K, n_steps = 80, 6000
        w_lo, w_hi = 0.05, 0.8
        src_ghost = np.zeros(n_steps)
        pulse_len = 2048
        n = np.arange(pulse_len)
        w0 = 0.5 * (w_lo + w_hi)
        sig = 6.0 / (w_hi - w_lo)
        t0 = 6.0 * sig
        src_ghost[:pulse_len] = np.exp(-0.5 * ((n - t0) / sig) ** 2) * np.cos(w0 * (n - t0))
        rec = _run_chain_with_dtbc(
            K,
            n_steps,
            r,
            q,
            np.zeros(n_steps),
            1,
            (K // 2,),
            src_ghost=src_ghost,
        )
        peak = np.max(np.abs(rec[:, 0]))
        tail = np.max(np.abs(rec[-500:, 0]))
        # The incident amplitude arrives at interior probes at O(source
        # amplitude) (|lambda| = 1 in the passband) ...
        assert peak > 0.3
        # ... and after the pulse has crossed, both DTBC ends drain the
        # domain to the float-noise class.
        assert tail / peak < 1e-9

    def test_kernel_auto_extension_matches_large_kernel(self):
        r, q = 0.85, 0.0
        rng = np.random.default_rng(11)
        n_steps = 300
        u_int = rng.standard_normal(n_steps)
        src = rng.standard_normal(n_steps)
        small = DTBCTermination(r, q, n_kernel_init=64)
        large = DTBCTermination(r, q, n_kernel_init=1024)
        got_small = np.array([small.advance(ui, s) for ui, s in zip(u_int, src)])
        got_large = np.array([large.advance(ui, s) for ui, s in zip(u_int, src)])
        assert np.allclose(got_small, got_large, rtol=0.0, atol=1e-11)

    def test_initialize_guards(self):
        term = DTBCTermination(0.5)
        term.initialize(2.0)
        assert term.u_boundary == 2.0
        term.advance(0.0)
        with pytest.raises(RuntimeError):
            term.initialize(1.0)

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            DTBCTermination(0.0)
        with pytest.raises(ValueError):
            DTBCTermination(1.5)
        with pytest.raises(ValueError):
            DTBCTermination(0.5, -0.1)
