"""Tests for magnelio.ports._modal.mode — Mode dataclass and impedance formulas."""

from __future__ import annotations

import math

import numpy as np
import pytest

from magnelio.ports._modal import C0, EPS0, ETA0, MU0, Mode, ModeType


def _dummy_evaluator(u, v):
    z = np.zeros_like(np.asarray(u, dtype=float))
    return z, z, z, z


def _make(mode_type, omega_c=0.0, epsilon_r=1.0, z_line=None):
    return Mode(
        name="dummy",
        mode_type=mode_type,
        omega_c=omega_c,
        epsilon_r=epsilon_r,
        field_evaluator=_dummy_evaluator,
        z_line=z_line,
    )


class TestGamma:
    def test_tem_propagates_at_speed_of_light_vacuum(self):
        m = _make(ModeType.TEM, omega_c=0.0, epsilon_r=1.0)
        omega = 2 * math.pi * 10e9
        gamma = m.gamma(omega)
        assert gamma.real == pytest.approx(0.0, abs=1e-6)
        assert gamma.imag == pytest.approx(omega / C0, rel=1e-12)

    def test_tem_in_dielectric(self):
        m = _make(ModeType.TEM, omega_c=0.0, epsilon_r=4.0)
        omega = 2 * math.pi * 10e9
        gamma = m.gamma(omega)
        # γ = j·ω·√(με) = j·ω·√ε_r/c₀
        assert gamma.imag == pytest.approx(2.0 * omega / C0, rel=1e-12)

    def test_te_propagating_above_cutoff(self):
        # WR-90 TE10: a = 22.86 mm, f_c = c/2a = 6.557 GHz
        a = 22.86e-3
        f_c = C0 / (2.0 * a)
        omega_c = 2 * math.pi * f_c
        m = _make(ModeType.TE, omega_c=omega_c, epsilon_r=1.0)
        omega = 2 * math.pi * 10e9
        gamma = m.gamma(omega)
        # β = √(k² - kc²) = (1/c) √(ω² - ω_c²)
        expected_beta = math.sqrt(omega**2 - omega_c**2) / C0
        assert gamma.real == pytest.approx(0.0, abs=1e-3)
        assert gamma.imag == pytest.approx(expected_beta, rel=1e-12)

    def test_te_evanescent_below_cutoff(self):
        omega_c = 2 * math.pi * 10e9
        m = _make(ModeType.TE, omega_c=omega_c, epsilon_r=1.0)
        omega = 2 * math.pi * 5e9
        gamma = m.gamma(omega)
        # α = (1/c)·√(ω_c² - ω²) (real, positive)
        expected_alpha = math.sqrt(omega_c**2 - omega**2) / C0
        assert gamma.imag == pytest.approx(0.0, abs=1e-6)
        assert gamma.real == pytest.approx(expected_alpha, rel=1e-12)

    def test_at_cutoff_gamma_zero(self):
        omega_c = 2 * math.pi * 10e9
        m = _make(ModeType.TE, omega_c=omega_c, epsilon_r=1.0)
        gamma = m.gamma(omega_c)
        assert abs(gamma) < 1e-3


class TestZWave:
    def test_tem_equals_eta(self):
        m = _make(ModeType.TEM, epsilon_r=2.1)
        z = m.z_wave(2 * math.pi * 10e9)
        assert z.real == pytest.approx(ETA0 / math.sqrt(2.1), rel=1e-12)
        assert z.imag == 0.0

    def test_tem_frequency_independent(self):
        m = _make(ModeType.TEM, epsilon_r=1.0)
        z_low = m.z_wave(2 * math.pi * 1e9)
        z_high = m.z_wave(2 * math.pi * 100e9)
        assert z_low == z_high

    def test_te_propagating_real_positive(self):
        omega_c = 2 * math.pi * 6.557e9
        m = _make(ModeType.TE, omega_c=omega_c, epsilon_r=1.0)
        omega = 2 * math.pi * 10e9
        z = m.z_wave(omega)
        # Z_TE = ω·μ/β where β = √(ω²με − ω_c²με)
        beta = math.sqrt(omega**2 - omega_c**2) / C0
        expected = omega * MU0 / beta
        assert z.imag == pytest.approx(0.0, abs=1e-9)
        assert z.real == pytest.approx(expected, rel=1e-12)

    def test_tm_propagating_real_positive(self):
        omega_c = 2 * math.pi * 8e9
        m = _make(ModeType.TM, omega_c=omega_c, epsilon_r=1.0)
        omega = 2 * math.pi * 10e9
        z = m.z_wave(omega)
        # Z_TM = β/(ω·ε)
        beta = math.sqrt(omega**2 - omega_c**2) / C0
        expected = beta / (omega * EPS0)
        assert z.imag == pytest.approx(0.0, abs=1e-9)
        assert z.real == pytest.approx(expected, rel=1e-12)

    def test_te_evanescent_purely_imaginary(self):
        omega_c = 2 * math.pi * 10e9
        m = _make(ModeType.TE, omega_c=omega_c, epsilon_r=1.0)
        z = m.z_wave(2 * math.pi * 5e9)
        # Z_TE = j·ω·μ/α (purely imaginary, positive imag)
        assert z.real == pytest.approx(0.0, abs=1e-12)
        assert z.imag > 0.0

    def test_tm_evanescent_purely_imaginary(self):
        omega_c = 2 * math.pi * 10e9
        m = _make(ModeType.TM, omega_c=omega_c, epsilon_r=1.0)
        z = m.z_wave(2 * math.pi * 5e9)
        # Z_TM = α/(j·ω·ε) = -j·α/(ω·ε) (purely imaginary, negative imag)
        assert z.real == pytest.approx(0.0, abs=1e-12)
        assert z.imag < 0.0

    def test_z_wave_zero_omega_rejected(self):
        m = _make(ModeType.TEM)
        with pytest.raises(ValueError):
            m.z_wave(0.0)


class TestZModal:
    def test_falls_back_to_z_wave_when_z_line_none(self):
        m = _make(ModeType.TE, omega_c=2 * math.pi * 8e9)
        omega = 2 * math.pi * 10e9
        assert m.z_modal(omega) == m.z_wave(omega)

    def test_returns_z_line_when_set(self):
        m = _make(ModeType.TEM, z_line=50.0)
        z_at_low = m.z_modal(2 * math.pi * 1e9)
        z_at_high = m.z_modal(2 * math.pi * 100e9)
        assert z_at_low == complex(50.0)
        assert z_at_high == complex(50.0)

    def test_z_line_overrides_even_for_te(self):
        # If a solver chose to set z_line on a TE mode (unusual but allowed),
        # z_modal returns it.
        m = _make(ModeType.TE, omega_c=2 * math.pi * 8e9, z_line=75.0)
        z = m.z_modal(2 * math.pi * 10e9)
        assert z == complex(75.0)


class TestModeImmutability:
    def test_frozen_dataclass(self):
        m = _make(ModeType.TEM)
        with pytest.raises(AttributeError):
            m.name = "other"  # type: ignore[misc]


class TestModeValidityInvariant:
    """Phase-2 Variant B (architecture doc §2.5): exactly one of
    field_evaluator or all four discrete_*_profile fields must be set."""

    def _profile(self, n: int) -> np.ndarray:
        return np.linspace(0.1, 1.0, n)

    def test_numerical_path_constructs(self):
        n_u, n_v = 5, 4
        m = Mode(
            name="num",
            mode_type=ModeType.TE,
            omega_c=2 * math.pi * 8e9,
            epsilon_r=1.0,
            field_evaluator=None,
            discrete_e_u_profile=self._profile(n_u),
            discrete_e_v_profile=self._profile(n_v),
            discrete_h_u_profile=self._profile(n_v),
            discrete_h_v_profile=self._profile(n_u),
        )
        assert m.field_evaluator is None
        assert m.discrete_e_u_profile.shape == (n_u,)

    def test_neither_path_set_rejected(self):
        with pytest.raises(ValueError, match="either field_evaluator"):
            Mode(
                name="empty",
                mode_type=ModeType.TEM,
                omega_c=0.0,
                epsilon_r=1.0,
                field_evaluator=None,
            )

    def test_both_paths_set_rejected(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            Mode(
                name="both",
                mode_type=ModeType.TEM,
                omega_c=0.0,
                epsilon_r=1.0,
                field_evaluator=_dummy_evaluator,
                discrete_e_u_profile=self._profile(3),
                discrete_e_v_profile=self._profile(3),
                discrete_h_u_profile=self._profile(3),
                discrete_h_v_profile=self._profile(3),
            )

    def test_partial_profiles_rejected(self):
        with pytest.raises(ValueError, match="all four"):
            Mode(
                name="partial",
                mode_type=ModeType.TE,
                omega_c=2 * math.pi * 8e9,
                epsilon_r=1.0,
                field_evaluator=None,
                discrete_e_u_profile=self._profile(3),
                discrete_e_v_profile=self._profile(3),
                discrete_h_u_profile=self._profile(3),
                # discrete_h_v_profile missing
            )

    def test_mismatched_u_profile_shapes_rejected(self):
        with pytest.raises(ValueError, match="discrete_e_u_profile"):
            Mode(
                name="bad_u",
                mode_type=ModeType.TE,
                omega_c=2 * math.pi * 8e9,
                epsilon_r=1.0,
                field_evaluator=None,
                discrete_e_u_profile=self._profile(5),
                discrete_e_v_profile=self._profile(4),
                discrete_h_u_profile=self._profile(4),
                discrete_h_v_profile=self._profile(6),  # mismatch with e_u
            )

    def test_mismatched_v_profile_shapes_rejected(self):
        with pytest.raises(ValueError, match="discrete_e_v_profile"):
            Mode(
                name="bad_v",
                mode_type=ModeType.TE,
                omega_c=2 * math.pi * 8e9,
                epsilon_r=1.0,
                field_evaluator=None,
                discrete_e_u_profile=self._profile(5),
                discrete_e_v_profile=self._profile(4),
                discrete_h_u_profile=self._profile(7),  # mismatch with e_v
                discrete_h_v_profile=self._profile(5),
            )
