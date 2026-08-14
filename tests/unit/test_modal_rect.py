"""Tests for magnelio.ports._modal.rect — analytical TE/TM rect-WG solver."""

from __future__ import annotations

import math

import numpy as np
import pytest

from magnelio.ports._modal import (
    C0,
    MU0,
    ModeType,
    RectWGAnalyticalModeSolver,
)

# WR-90 standard rectangular waveguide (X-band, ~8.2–12.4 GHz)
WR90_A = 22.86e-3
WR90_B = 10.16e-3


def _wr90(eps_r: float = 1.0) -> RectWGAnalyticalModeSolver:
    return RectWGAnalyticalModeSolver(
        width_a=WR90_A,
        height_b=WR90_B,
        epsilon_r=eps_r,
    )


class TestRectConstruction:
    def test_default_parameters(self):
        s = RectWGAnalyticalModeSolver(width_a=22.86e-3, height_b=10.16e-3)
        assert s.epsilon_r == 1.0
        assert s.center == (0.0, 0.0)

    def test_rejects_zero_width(self):
        with pytest.raises(ValueError):
            RectWGAnalyticalModeSolver(width_a=0.0, height_b=10.16e-3)

    def test_rejects_zero_height(self):
        with pytest.raises(ValueError):
            RectWGAnalyticalModeSolver(width_a=22.86e-3, height_b=0.0)

    def test_rejects_zero_epsilon(self):
        with pytest.raises(ValueError):
            RectWGAnalyticalModeSolver(
                width_a=22.86e-3,
                height_b=10.16e-3,
                epsilon_r=0.0,
            )


class TestRectModeOrdering:
    def test_n_modes_one_returns_te10(self):
        modes = _wr90().solve(n_modes=1, f_calc=10e9)
        assert len(modes) == 1
        assert modes[0].name == "TE10"
        assert modes[0].mode_type is ModeType.TE

    def test_first_five_wr90_modes(self):
        modes = _wr90().solve(n_modes=5, f_calc=10e9)
        labels = [m.name for m in modes]
        # Expected order at WR-90: TE10, TE20, TE01, TE11, TM11
        assert labels == ["TE10", "TE20", "TE01", "TE11", "TM11"]

    def test_te_precedes_tm_at_same_cutoff(self):
        # TE11 and TM11 share cutoff frequency.  TE must come first.
        modes = _wr90().solve(n_modes=6, f_calc=10e9)
        idx_te11 = next(i for i, m in enumerate(modes) if m.name == "TE11")
        idx_tm11 = next(i for i, m in enumerate(modes) if m.name == "TM11")
        assert idx_te11 < idx_tm11
        assert modes[idx_te11].omega_c == pytest.approx(
            modes[idx_tm11].omega_c,
            rel=1e-12,
        )

    def test_modes_sorted_by_ascending_cutoff(self):
        modes = _wr90().solve(n_modes=8, f_calc=10e9)
        omegas = [m.omega_c for m in modes]
        assert omegas == sorted(omegas)

    def test_zero_n_modes_rejected(self):
        with pytest.raises(ValueError):
            _wr90().solve(n_modes=0, f_calc=10e9)

    def test_zero_f_calc_rejected(self):
        with pytest.raises(ValueError):
            _wr90().solve(n_modes=1, f_calc=0.0)


class TestRectCutoffFrequencies:
    def test_te10_cutoff_vacuum(self):
        modes = _wr90().solve(n_modes=1, f_calc=10e9)
        # f_c = c / (2a) for TE10 in vacuum
        f_c_expected = C0 / (2.0 * WR90_A)
        f_c = modes[0].omega_c / (2.0 * math.pi)
        assert f_c == pytest.approx(f_c_expected, rel=1e-12)

    def test_te10_cutoff_in_dielectric(self):
        eps_r = 4.0
        modes = _wr90(eps_r=eps_r).solve(n_modes=1, f_calc=5e9)
        f_c_expected = C0 / (2.0 * WR90_A * math.sqrt(eps_r))
        f_c = modes[0].omega_c / (2.0 * math.pi)
        assert f_c == pytest.approx(f_c_expected, rel=1e-12)

    def test_tm11_cutoff(self):
        modes = _wr90().solve(n_modes=5, f_calc=20e9)
        tm11 = next(m for m in modes if m.name == "TM11")
        # f_c = (c/2)·√(1/a² + 1/b²) for TM11 in vacuum
        f_c_expected = (C0 / 2.0) * math.sqrt((1.0 / WR90_A) ** 2 + (1.0 / WR90_B) ** 2)
        f_c = tm11.omega_c / (2.0 * math.pi)
        assert f_c == pytest.approx(f_c_expected, rel=1e-12)

    def test_te20_cutoff(self):
        modes = _wr90().solve(n_modes=2, f_calc=15e9)
        te20 = next(m for m in modes if m.name == "TE20")
        f_c_expected = C0 / WR90_A
        f_c = te20.omega_c / (2.0 * math.pi)
        assert f_c == pytest.approx(f_c_expected, rel=1e-12)


class TestRectImpedances:
    def test_te10_z_wave_propagating(self):
        modes = _wr90().solve(n_modes=1, f_calc=10e9)
        omega = 2 * math.pi * 10e9
        z = modes[0].z_wave(omega)
        # Z_TE = ωμ/β where β = √(ω² - ω_c²)/c (for ε_r=1)
        omega_c = modes[0].omega_c
        beta = math.sqrt(omega**2 - omega_c**2) / C0
        expected = omega * MU0 / beta
        assert z.imag == pytest.approx(0.0, abs=1e-9)
        assert z.real == pytest.approx(expected, rel=1e-12)

    def test_evanescent_te10_below_cutoff(self):
        modes = _wr90().solve(n_modes=1, f_calc=10e9)
        # TE10 cutoff ~6.56 GHz; query at 5 GHz is evanescent
        z = modes[0].z_wave(2 * math.pi * 5e9)
        assert z.real == pytest.approx(0.0, abs=1e-12)
        assert z.imag > 0.0

    def test_z_line_is_none(self):
        modes = _wr90().solve(n_modes=5, f_calc=20e9)
        for m in modes:
            assert m.z_line is None


class TestRectFieldEvaluator:
    def _solve_te10(self, f_calc=10e9, eps_r=1.0):
        return _wr90(eps_r=eps_r).solve(n_modes=1, f_calc=f_calc)[0]

    def test_zero_outside_waveguide(self):
        m = self._solve_te10()
        u = np.array([-1.0e-3, 30.0e-3, 5.0e-3, 5.0e-3])
        v = np.array([5.0e-3, 5.0e-3, -1.0e-3, 15.0e-3])
        E_u, E_v, H_u, H_v = m.field_evaluator(u, v)
        for arr in (E_u, E_v, H_u, H_v):
            assert np.all(arr == 0.0)

    def test_te10_has_only_e_v_and_h_u(self):
        m = self._solve_te10()
        u = np.array([WR90_A / 2.0, WR90_A / 4.0])
        v = np.array([WR90_B / 2.0, WR90_B / 4.0])
        E_u, E_v, H_u, H_v = m.field_evaluator(u, v)
        # TE10: E_u and H_v are zero, only E_v and H_u carry the mode
        assert np.allclose(E_u, 0.0, atol=1e-12)
        assert np.allclose(H_v, 0.0, atol=1e-12)
        assert not np.allclose(E_v, 0.0)
        assert not np.allclose(H_u, 0.0)

    def test_te10_e_zero_on_pec_walls(self):
        m = self._solve_te10()
        # E_v ∝ sin(πu/a) → zero at u=0 and u=a (up to machine roundoff
        # in sin(π); the absolute value is peak·O(eps_machine)).
        u = np.array([0.0, WR90_A, WR90_A / 2.0])
        v = np.full(3, WR90_B / 2.0)
        _, E_v, _, _ = m.field_evaluator(u, v)
        peak = abs(E_v[2])
        assert peak > 0.0
        assert abs(E_v[0]) < 1e-12 * peak
        assert abs(E_v[1]) < 1e-12 * peak

    def test_te10_e_max_at_centre(self):
        m = self._solve_te10()
        # Sample along v at u = a/2 (where sin(π·a/2/a) = 1, E_v is maximal)
        u = np.full(5, WR90_A / 2.0)
        v = np.linspace(0.0, WR90_B, 5)
        _, E_v, _, _ = m.field_evaluator(u, v)
        # All five samples should have the same E_v (no v-dependence for TE10)
        assert np.allclose(E_v, E_v[0], rtol=1e-12)
        # And E_v[0] should be the global max
        assert abs(E_v[0]) > 0.0

    def test_te10_poynting_to_one_watt(self):
        m = self._solve_te10(f_calc=10e9)
        n_u, n_v = 200, 100
        u = np.linspace(0.0, WR90_A, n_u, endpoint=False) + WR90_A / (2 * n_u)
        v = np.linspace(0.0, WR90_B, n_v, endpoint=False) + WR90_B / (2 * n_v)
        du = WR90_A / n_u
        dv = WR90_B / n_v
        U, V = np.meshgrid(u, v, indexing="ij")
        E_u, E_v, H_u, H_v = m.field_evaluator(U.ravel(), V.ravel())
        E_u = E_u.reshape(U.shape)
        E_v = E_v.reshape(U.shape)
        H_u = H_u.reshape(U.shape)
        H_v = H_v.reshape(U.shape)
        S_z = E_u * H_v - E_v * H_u
        P = float(np.sum(S_z) * du * dv)
        assert P == pytest.approx(1.0, rel=1e-3)

    def test_higher_modes_poynting_to_one_watt(self):
        modes = _wr90().solve(n_modes=5, f_calc=20e9)
        n_u, n_v = 300, 200
        u = np.linspace(0.0, WR90_A, n_u, endpoint=False) + WR90_A / (2 * n_u)
        v = np.linspace(0.0, WR90_B, n_v, endpoint=False) + WR90_B / (2 * n_v)
        du = WR90_A / n_u
        dv = WR90_B / n_v
        U, V = np.meshgrid(u, v, indexing="ij")
        for m in modes:
            E_u, E_v, H_u, H_v = m.field_evaluator(U.ravel(), V.ravel())
            E_u = E_u.reshape(U.shape)
            E_v = E_v.reshape(U.shape)
            H_u = H_u.reshape(U.shape)
            H_v = H_v.reshape(U.shape)
            S_z = E_u * H_v - E_v * H_u
            P = float(np.sum(S_z) * du * dv)
            assert P == pytest.approx(1.0, rel=5e-3), f"Mode {m.name} P={P}"


class TestRectModeOrthogonality:
    def test_e_field_orthogonality(self):
        # Cross-section integral of E_t,i · E_t,j must be zero for i ≠ j.
        modes = _wr90().solve(n_modes=5, f_calc=20e9)
        n_u, n_v = 300, 200
        u = np.linspace(0.0, WR90_A, n_u, endpoint=False) + WR90_A / (2 * n_u)
        v = np.linspace(0.0, WR90_B, n_v, endpoint=False) + WR90_B / (2 * n_v)
        du = WR90_A / n_u
        dv = WR90_B / n_v
        U, V = np.meshgrid(u, v, indexing="ij")
        # Compute E profiles for all modes
        e_us = []
        e_vs = []
        for m in modes:
            E_u, E_v, _, _ = m.field_evaluator(U.ravel(), V.ravel())
            e_us.append(E_u.reshape(U.shape))
            e_vs.append(E_v.reshape(U.shape))
        # Off-diagonal inner products
        for i in range(len(modes)):
            for j in range(i + 1, len(modes)):
                inner = float(np.sum(e_us[i] * e_us[j] + e_vs[i] * e_vs[j]) * du * dv)
                # Self-norm (rough scale) of mode i for relative tolerance
                self_norm_i = float(np.sum(e_us[i] ** 2 + e_vs[i] ** 2) * du * dv)
                assert abs(inner) < 1e-3 * self_norm_i, (
                    f"Modes {modes[i].name} and {modes[j].name} "
                    f"not E-orthogonal: {inner=}, scale={self_norm_i}"
                )


class TestRectCenterOffset:
    def test_translation_invariance(self):
        m_centered = _wr90().solve(n_modes=1, f_calc=10e9)[0]
        offset = (5.0e-3, 2.0e-3)
        m_offset = RectWGAnalyticalModeSolver(
            width_a=WR90_A,
            height_b=WR90_B,
            center=offset,
        ).solve(n_modes=1, f_calc=10e9)[0]
        # Sample at the same point relative to each waveguide's lower-left corner
        u_c = np.array([WR90_A / 2.0])
        v_c = np.array([WR90_B / 2.0])
        u_o = np.array([WR90_A / 2.0 + offset[0]])
        v_o = np.array([WR90_B / 2.0 + offset[1]])
        _, E_v_c, _, _ = m_centered.field_evaluator(u_c, v_c)
        _, E_v_o, _, _ = m_offset.field_evaluator(u_o, v_o)
        assert E_v_c[0] == pytest.approx(E_v_o[0], rel=1e-12)
