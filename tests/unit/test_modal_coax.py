"""Tests for magnelio.ports._modal.coax — analytical TEM coax mode solver."""

from __future__ import annotations

import math

import numpy as np
import pytest

from magnelio.ports._modal import (
    C0,
    ETA0,
    CoaxAnalyticalModeSolver,
    Mode,
    ModeType,
)


class TestCoaxConstruction:
    def test_default_parameters(self):
        s = CoaxAnalyticalModeSolver(inner_radius=0.5e-3, outer_radius=1.5e-3)
        assert s.epsilon_r == 1.0
        assert s.center == (0.0, 0.0)

    def test_rejects_zero_inner(self):
        with pytest.raises(ValueError):
            CoaxAnalyticalModeSolver(inner_radius=0.0, outer_radius=1.5e-3)

    def test_rejects_negative_inner(self):
        with pytest.raises(ValueError):
            CoaxAnalyticalModeSolver(inner_radius=-1.0, outer_radius=1.5e-3)

    def test_rejects_inverted_radii(self):
        with pytest.raises(ValueError):
            CoaxAnalyticalModeSolver(inner_radius=1.5e-3, outer_radius=0.5e-3)

    def test_rejects_equal_radii(self):
        with pytest.raises(ValueError):
            CoaxAnalyticalModeSolver(inner_radius=1.0e-3, outer_radius=1.0e-3)

    def test_rejects_zero_epsilon(self):
        with pytest.raises(ValueError):
            CoaxAnalyticalModeSolver(
                inner_radius=0.5e-3,
                outer_radius=1.5e-3,
                epsilon_r=0.0,
            )


class TestCoaxModeProperties:
    def _solve(self, eps_r=1.0, r_i=0.5e-3, r_o=1.5e-3, center=(0.0, 0.0)):
        return CoaxAnalyticalModeSolver(
            inner_radius=r_i,
            outer_radius=r_o,
            epsilon_r=eps_r,
            center=center,
        ).solve(n_modes=1)[0]

    def test_returns_one_mode(self):
        modes = CoaxAnalyticalModeSolver(
            inner_radius=0.5e-3,
            outer_radius=1.5e-3,
        ).solve()
        assert len(modes) == 1
        assert isinstance(modes[0], Mode)

    def test_n_modes_above_one_unsupported(self):
        with pytest.raises(NotImplementedError):
            CoaxAnalyticalModeSolver(
                inner_radius=0.5e-3,
                outer_radius=1.5e-3,
            ).solve(n_modes=2)

    def test_metadata(self):
        m = self._solve()
        assert m.name == "TEM"
        assert m.mode_type is ModeType.TEM
        assert m.omega_c == 0.0
        assert m.epsilon_r == 1.0

    def test_z_line_vacuum_50_ohm_geometry(self):
        # 50-Ω-ish geometry (approximately): Z₀ = (η₀/2π)·ln(r_o/r_i)
        # For r_o/r_i = e^(50·2π/η₀), but here we just test the formula.
        m = self._solve()
        expected = ETA0 * math.log(3.0) / (2.0 * math.pi)
        assert m.z_line == pytest.approx(expected, rel=1e-12)

    def test_z_line_dielectric(self):
        m = self._solve(eps_r=2.1)
        eta = ETA0 / math.sqrt(2.1)
        expected = (eta / (2.0 * math.pi)) * math.log(3.0)
        assert m.z_line == pytest.approx(expected, rel=1e-12)

    def test_z_modal_returns_z_line(self):
        m = self._solve(eps_r=2.1)
        # z_modal uses z_line for TEM modes (frequency-independent)
        omega1 = 2 * math.pi * 1e9
        omega2 = 2 * math.pi * 50e9
        assert m.z_modal(omega1) == complex(m.z_line)
        assert m.z_modal(omega2) == complex(m.z_line)

    def test_z_wave_equals_eta(self):
        m = self._solve(eps_r=2.1)
        z = m.z_wave(2 * math.pi * 10e9)
        assert z.real == pytest.approx(ETA0 / math.sqrt(2.1), rel=1e-12)

    def test_gamma_propagating(self):
        m = self._solve(eps_r=2.1)
        omega = 2 * math.pi * 10e9
        gamma = m.gamma(omega)
        # γ = j·ω·√ε_r/c₀
        assert gamma.imag == pytest.approx(omega * math.sqrt(2.1) / C0, rel=1e-12)


class TestCoaxFieldEvaluator:
    def _solve(self, **kwargs):
        params = dict(inner_radius=0.5e-3, outer_radius=1.5e-3, epsilon_r=1.0)
        params.update(kwargs)
        return CoaxAnalyticalModeSolver(**params).solve()[0]

    def test_zero_inside_inner_conductor(self):
        m = self._solve()
        u = np.array([0.0, 0.3e-3])
        v = np.array([0.0, 0.0])
        E_u, E_v, H_u, H_v = m.field_evaluator(u, v)
        for arr in (E_u, E_v, H_u, H_v):
            assert np.all(arr == 0.0)

    def test_zero_outside_outer_conductor(self):
        m = self._solve()
        u = np.array([2.0e-3, -3.0e-3])
        v = np.array([0.0, 1.0e-3])
        E_u, E_v, H_u, H_v = m.field_evaluator(u, v)
        for arr in (E_u, E_v, H_u, H_v):
            assert np.all(arr == 0.0)

    def test_radial_e_along_u_axis(self):
        m = self._solve()
        u = np.array([1.0e-3])
        v = np.array([0.0])
        E_u, E_v, H_u, H_v = m.field_evaluator(u, v)
        # On +u-axis: E points purely in +u, H purely in +v
        assert E_v[0] == pytest.approx(0.0, abs=1e-12)
        assert E_u[0] > 0.0
        assert H_u[0] == pytest.approx(0.0, abs=1e-12)
        assert H_v[0] > 0.0

    def test_e_h_ratio_equals_eta(self):
        m = self._solve(epsilon_r=2.1)
        u = np.array([1.0e-3])
        v = np.array([0.0])
        E_u, _, _, H_v = m.field_evaluator(u, v)
        expected = ETA0 / math.sqrt(2.1)
        assert (E_u[0] / H_v[0]) == pytest.approx(expected, rel=1e-12)

    def test_one_over_r_dependence(self):
        m = self._solve()
        u = np.array([0.7e-3, 1.4e-3])
        v = np.zeros(2)
        E_u, _, _, _ = m.field_evaluator(u, v)
        # E_r ∝ 1/r → E(r=0.7)/E(r=1.4) = 2
        assert (E_u[0] / E_u[1]) == pytest.approx(2.0, rel=1e-12)

    def test_azimuthal_orientation(self):
        m = self._solve()
        # Sample at four cardinal points r=1 mm
        u = np.array([1.0e-3, 0.0, -1.0e-3, 0.0])
        v = np.array([0.0, 1.0e-3, 0.0, -1.0e-3])
        E_u, E_v, H_u, H_v = m.field_evaluator(u, v)
        # At +u: E=(+,0), H=(0,+)
        assert E_u[0] > 0 and abs(E_v[0]) < 1e-12
        assert abs(H_u[0]) < 1e-12 and H_v[0] > 0
        # At +v: E=(0,+), H=(-,0)
        assert abs(E_u[1]) < 1e-12 and E_v[1] > 0
        assert H_u[1] < 0 and abs(H_v[1]) < 1e-12
        # At -u: E=(-,0), H=(0,-)
        assert E_u[2] < 0 and abs(E_v[2]) < 1e-12
        assert abs(H_u[2]) < 1e-12 and H_v[2] < 0
        # At -v: E=(0,-), H=(+,0)
        assert abs(E_u[3]) < 1e-12 and E_v[3] < 0
        assert H_u[3] > 0 and abs(H_v[3]) < 1e-12

    def test_poynting_normalisation_to_one_watt(self):
        m = self._solve(epsilon_r=2.1)
        n_r, n_phi = 800, 360
        r = np.linspace(0.5e-3, 1.5e-3, n_r)
        phi = np.linspace(0.0, 2 * math.pi, n_phi, endpoint=False)
        dr = r[1] - r[0]
        dphi = 2 * math.pi / n_phi
        R, P = np.meshgrid(r, phi, indexing="ij")
        U = R * np.cos(P)
        V = R * np.sin(P)
        E_u, E_v, H_u, H_v = m.field_evaluator(U.ravel(), V.ravel())
        E_u = E_u.reshape(R.shape)
        E_v = E_v.reshape(R.shape)
        H_u = H_u.reshape(R.shape)
        H_v = H_v.reshape(R.shape)
        # Axial Poynting S_z = E_u·H_v − E_v·H_u (peak amplitude, no ½ factor)
        S_z = E_u * H_v - E_v * H_u
        P_total = np.sum(S_z * R * dr * dphi)
        assert P_total == pytest.approx(1.0, rel=2e-3)

    def test_center_offset_translation_invariance(self):
        m_centered = self._solve()
        m_offset = self._solve(center=(2.0e-3, 1.0e-3))
        # Same point relative to each center
        u_c = np.array([1.0e-3])
        v_c = np.zeros(1)
        E_u_c, _, _, _ = m_centered.field_evaluator(u_c, v_c)
        u_o = np.array([3.0e-3])
        v_o = np.array([1.0e-3])
        E_u_o, _, _, _ = m_offset.field_evaluator(u_o, v_o)
        assert E_u_c[0] == pytest.approx(E_u_o[0], rel=1e-12)
