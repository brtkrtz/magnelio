"""Tests for magnelio.ports._modal.discrete — discretisation & B-orthonormalisation."""

from __future__ import annotations

import numpy as np
import pytest

from magnelio._operators.material_matrices import build_M_eps
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.ports._modal import (
    BoxFace,
    CoaxAnalyticalModeSolver,
    PortPlane,
    RectWGAnalyticalModeSolver,
    discretize_modes,
    gram_matrix,
)

# ----------------------------------------------------------------------
# Test fixtures: small box with port at X_MIN
# ----------------------------------------------------------------------

WR90_A = 22.86e-3
WR90_B = 10.16e-3


def _wr90_mesh(Nu: int = 14, Nv: int = 7, Nz: int = 5, length: float = 30e-3) -> Mesh:
    """Mesh sized to fit a WR-90 cross-section.  X is propagation, Y is u, Z is v."""
    grid = GridLines(
        x=np.linspace(0.0, length, Nz + 1),
        y=np.linspace(0.0, WR90_A, Nu + 1),
        z=np.linspace(0.0, WR90_B, Nv + 1),
    )
    return Mesh.from_grid(grid)


def _coax_mesh(R_outer: float = 1.5e-3, length: float = 5e-3, N_uv: int = 12, N_z: int = 6) -> Mesh:
    """Square cross-section that bounds a coaxial cable centred at origin."""
    half = R_outer * 1.05
    grid = GridLines(
        x=np.linspace(0.0, length, N_z + 1),
        y=np.linspace(-half, half, N_uv + 1),
        z=np.linspace(-half, half, N_uv + 1),
    )
    return Mesh.from_grid(grid)


# ----------------------------------------------------------------------
# Single-mode case: gram matrix is [[1.0]]
# ----------------------------------------------------------------------


class TestSingleModeNormalisation:
    def test_coax_tem_normalises_to_unity(self):
        mesh = _coax_mesh()
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        m_eps = build_M_eps(mesh)
        modes = CoaxAnalyticalModeSolver(
            inner_radius=0.5e-3,
            outer_radius=1.5e-3,
        ).solve(n_modes=1)
        discrete = discretize_modes(modes, plane, m_eps)
        assert len(discrete) == 1
        G = gram_matrix(discrete, plane, m_eps)
        assert G.shape == (1, 1)
        assert G[0, 0] == pytest.approx(1.0, rel=1e-12)

    def test_rect_te10_normalises_to_unity(self):
        mesh = _wr90_mesh()
        # Port at X_MIN: u=y, v=z.  WR-90 has width along y = 22.86 mm,
        # height along z = 10.16 mm; the mesh y, z extents match these.
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        m_eps = build_M_eps(mesh)
        modes = RectWGAnalyticalModeSolver(
            width_a=WR90_A,
            height_b=WR90_B,
        ).solve(n_modes=1, f_calc=10e9)
        discrete = discretize_modes(modes, plane, m_eps)
        G = gram_matrix(discrete, plane, m_eps)
        assert G[0, 0] == pytest.approx(1.0, rel=1e-12)


# ----------------------------------------------------------------------
# Multi-mode: gram matrix = identity (cross-mode orthogonality)
# ----------------------------------------------------------------------


class TestMultiModeOrthonormality:
    def test_wr90_five_modes_gram_is_identity(self):
        mesh = _wr90_mesh()
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        m_eps = build_M_eps(mesh)
        modes = RectWGAnalyticalModeSolver(
            width_a=WR90_A,
            height_b=WR90_B,
        ).solve(n_modes=5, f_calc=15e9)
        discrete = discretize_modes(modes, plane, m_eps)
        G = gram_matrix(discrete, plane, m_eps)
        # Diagonal: ones to 1e-12.  Off-diagonal: zero to 1e-10 (after MGS).
        np.testing.assert_allclose(np.diag(G), 1.0, rtol=1e-12)
        off = G - np.eye(len(modes))
        assert np.max(np.abs(off)) < 1e-10, (
            f"Max off-diagonal Gram entry: {np.max(np.abs(off)):.3e}"
        )

    def test_wr90_eight_modes_gram_is_identity(self):
        # Higher mode count → more chances for Gram-Schmidt to lose accuracy
        mesh = _wr90_mesh(Nu=20, Nv=10, Nz=5)
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        m_eps = build_M_eps(mesh)
        modes = RectWGAnalyticalModeSolver(
            width_a=WR90_A,
            height_b=WR90_B,
        ).solve(n_modes=8, f_calc=25e9)
        discrete = discretize_modes(modes, plane, m_eps)
        G = gram_matrix(discrete, plane, m_eps)
        np.testing.assert_allclose(np.diag(G), 1.0, rtol=1e-12)
        off = G - np.eye(len(modes))
        assert np.max(np.abs(off)) < 1e-10


# ----------------------------------------------------------------------
# Profile properties after discretisation
# ----------------------------------------------------------------------


class TestDiscreteModeProperties:
    def test_profile_shapes_match_plane(self):
        mesh = _wr90_mesh()
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        m_eps = build_M_eps(mesh)
        modes = RectWGAnalyticalModeSolver(
            width_a=WR90_A,
            height_b=WR90_B,
        ).solve(n_modes=3, f_calc=15e9)
        discrete = discretize_modes(modes, plane, m_eps)
        n_u = plane.e_u_indices.shape[0]
        n_v = plane.e_v_indices.shape[0]
        for d in discrete:
            assert d.e_u_profile.shape == (n_u,)
            assert d.e_v_profile.shape == (n_v,)
            assert d.h_v_profile.shape == (n_u,)  # co-located with u-edges
            assert d.h_u_profile.shape == (n_v,)  # co-located with v-edges

    def test_te10_e_u_is_zero(self):
        # TE10 has no E_u component (E lies along v-axis)
        mesh = _wr90_mesh()
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        m_eps = build_M_eps(mesh)
        modes = RectWGAnalyticalModeSolver(
            width_a=WR90_A,
            height_b=WR90_B,
        ).solve(n_modes=1, f_calc=10e9)
        discrete = discretize_modes(modes, plane, m_eps)
        np.testing.assert_allclose(discrete[0].e_u_profile, 0.0, atol=1e-12)
        # E_v must be non-zero
        assert np.max(np.abs(discrete[0].e_v_profile)) > 0.0

    def test_h_eu_ratio_preserves_z_modal(self):
        """After GS, |E| / |H| at peak should still be Z_modal (within roundoff)."""
        mesh = _wr90_mesh()
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        m_eps = build_M_eps(mesh)
        modes = RectWGAnalyticalModeSolver(
            width_a=WR90_A,
            height_b=WR90_B,
        ).solve(n_modes=1, f_calc=10e9)
        discrete = discretize_modes(modes, plane, m_eps)
        # For TE10 the peak |E_v| / |H_u| should equal Z_TE at f_calc.
        # GS scales both by the same factor, so the ratio is preserved.
        peak_e = np.max(np.abs(discrete[0].e_v_profile))
        # H_u co-located at v-edges (= Ez edge midpoints).  For TE10 this
        # is the dominant H component.
        peak_h = np.max(np.abs(discrete[0].h_u_profile))
        z_te = float(modes[0].z_wave(2 * np.pi * 10e9).real)
        ratio = peak_e / peak_h
        assert ratio == pytest.approx(z_te, rel=1e-10)


# ----------------------------------------------------------------------
# Failure modes
# ----------------------------------------------------------------------


class TestDiscretizeErrors:
    def test_zero_norm_mode_rejected(self):
        # Construct a "mode" whose field_evaluator returns zero on the plane.
        from magnelio.ports._modal.mode import Mode, ModeType

        def zero_eval(u, v):
            z = np.zeros_like(np.asarray(u, dtype=float))
            return z, z, z, z

        mesh = _wr90_mesh()
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        m_eps = build_M_eps(mesh)
        bogus = Mode(
            name="zero",
            mode_type=ModeType.TEM,
            omega_c=0.0,
            epsilon_r=1.0,
            field_evaluator=zero_eval,
        )
        with pytest.raises(ValueError, match="linearly dependent"):
            discretize_modes([bogus], plane, m_eps)


# ----------------------------------------------------------------------
# Phase-2 numerical pass-through path (Variant B, architecture §2.5)
# ----------------------------------------------------------------------


class TestDiscretizeNumericalPassThrough:
    """Modes carrying discrete_*_profile arrays bypass Gram-Schmidt."""

    def _make_numerical_mode(
        self,
        label: str,
        n_u: int,
        n_v: int,
        scale: float = 1.0,
    ):
        from magnelio.ports._modal.mode import Mode, ModeType

        # Distinct deterministic profiles per mode so we can verify they
        # land in the DiscreteMode unchanged (no projection, no rescale).
        rng = np.random.default_rng(seed=hash(label) & 0xFFFF)
        return Mode(
            name=label,
            mode_type=ModeType.TE,
            omega_c=2 * np.pi * 8e9,
            epsilon_r=1.0,
            field_evaluator=None,
            discrete_e_u_profile=scale * rng.standard_normal(n_u),
            discrete_e_v_profile=scale * rng.standard_normal(n_v),
            discrete_h_u_profile=scale * rng.standard_normal(n_v),
            discrete_h_v_profile=scale * rng.standard_normal(n_u),
        )

    def test_pass_through_preserves_profiles_byte_for_byte(self):
        mesh = _wr90_mesh()
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        m_eps = build_M_eps(mesh)
        n_u = int(plane.e_u_indices.size)
        n_v = int(plane.e_v_indices.size)

        modes = [
            self._make_numerical_mode("num0", n_u, n_v),
            self._make_numerical_mode("num1", n_u, n_v, scale=2.7),
        ]
        discrete = discretize_modes(modes, plane, m_eps)

        assert len(discrete) == 2
        for m, d in zip(modes, discrete):
            np.testing.assert_array_equal(d.e_u_profile, m.discrete_e_u_profile)
            np.testing.assert_array_equal(d.e_v_profile, m.discrete_e_v_profile)
            np.testing.assert_array_equal(d.h_u_profile, m.discrete_h_u_profile)
            np.testing.assert_array_equal(d.h_v_profile, m.discrete_h_v_profile)

    def test_pass_through_returns_copies_not_views(self):
        """Mutating the DiscreteMode profile must not propagate to the Mode."""
        mesh = _wr90_mesh()
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        m_eps = build_M_eps(mesh)
        n_u = int(plane.e_u_indices.size)
        n_v = int(plane.e_v_indices.size)

        m = self._make_numerical_mode("num", n_u, n_v)
        original = m.discrete_e_u_profile.copy()
        discrete = discretize_modes([m], plane, m_eps)
        discrete[0].e_u_profile[0] = 999.0
        np.testing.assert_array_equal(m.discrete_e_u_profile, original)

    def test_mixed_list_rejected(self):

        mesh = _wr90_mesh()
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        m_eps = build_M_eps(mesh)
        n_u = int(plane.e_u_indices.size)
        n_v = int(plane.e_v_indices.size)

        analytical = RectWGAnalyticalModeSolver(
            width_a=WR90_A,
            height_b=WR90_B,
        ).solve(n_modes=1, f_calc=10e9)[0]
        numerical = self._make_numerical_mode("num", n_u, n_v)

        with pytest.raises(ValueError, match="mixes analytical"):
            discretize_modes([analytical, numerical], plane, m_eps)

    def test_wrong_u_shape_rejected(self):
        mesh = _wr90_mesh()
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        m_eps = build_M_eps(mesh)
        n_u = int(plane.e_u_indices.size)
        n_v = int(plane.e_v_indices.size)

        # Build a mode with u-profiles one element too short.
        bad = self._make_numerical_mode("bad_u", n_u - 1, n_v)
        with pytest.raises(ValueError, match="u-edge profiles"):
            discretize_modes([bad], plane, m_eps)

    def test_wrong_v_shape_rejected(self):
        mesh = _wr90_mesh()
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        m_eps = build_M_eps(mesh)
        n_u = int(plane.e_u_indices.size)
        n_v = int(plane.e_v_indices.size)

        bad = self._make_numerical_mode("bad_v", n_u, n_v + 1)
        with pytest.raises(ValueError, match="v-edge profiles"):
            discretize_modes([bad], plane, m_eps)

    def test_empty_list_returns_empty(self):
        mesh = _wr90_mesh()
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        m_eps = build_M_eps(mesh)
        assert discretize_modes([], plane, m_eps) == []
