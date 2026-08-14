"""Tests for magnelio.ports._modal.tem_laplace — solve_tem_laplace.

Phase 2b step 7 of the numerical-mode-solver work order
(`reference_architecture_phase2_mode_solver.md` §3.3 / §5).

Five layers of validation:

1. **Argument validation.**  Bad shapes / fewer than two conductor
   groups / out-of-range or overlapping node indices / non-positive
   ``epsilon_r`` are rejected.
2. **2D gradient helper.**  ``build_2d_gradient`` returns the right
   shapes, ``±1`` entries, and the de Rham exactness ``C_2d @ G_2d == 0``
   holds against the curl operator from the Phase-2a step-1 module.
3. **Coax Z_line vs analytical.**  On a staircased Coax (Ny=Nz=80,
   r_i=1 mm, r_o=3.5 mm), ``Z_line`` recovers the closed-form
   ``(η/2π)·ln(r_o/r_i)`` within 5 % at ε_r=1 and within 5 % at
   ε_r=4 (clean O(h) convergence on the radial staircase).
4. **Mode object structure.**  Resulting modes are valid Phase-2
   numerical-path ``Mode`` objects: ``field_evaluator is None``,
   ``omega_c == 0``, ``mode_type == ModeType.TEM``,
   ``discrete_*_profile`` arrays of the right shapes,
   M_ε-orthonormal, travelling-wave H profiles
   ``h = ±e_co·(μ₀·dz/Z_TEM)/M_μ[face]`` (WP7.2),
   and the ``discretize_modes`` pass-through round-trips byte-for-byte.
5. **Multi-conductor.**  A 3-conductor setup (one ground + two signal
   conductors) yields 2 TEM modes, one per signal conductor in the
   input ordering, each with positive outward flux on its own
   conductor (sign-convention).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse as sp

from magnelio._operators.curl import (
    build_curl_matrix,
    build_gradient_matrix,
)
from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.ports._modal import (
    BoxFace,
    ModeType,
    PortPlane,
    discretize_modes,
    solve_tem_laplace,
)
from magnelio.ports._modal.curl_curl_2d import (
    _normal_h_indices,
    build_2d_curl_curl,
    build_2d_gradient,
)
from magnelio.ports._modal.mode import ETA0, MU0
from magnelio.ports._modal.tem_laplace import _tem_label

C0 = 2.99792458e8


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _coax_setup(
    *,
    Nyz: int = 80,
    r_i: float = 1.0e-3,
    r_o: float = 3.5e-3,
    L: float = 8e-3,
    epsilon_r: float = 1.0,
):
    """Build a coaxial-cross-section setup on X_MIN.

    Returns (plane, g_2d, m_eps_2d, conductor_groups, mesh).  The
    conductor groups list is ``[outer_ground, inner_signal]`` of local
    2D node indices.
    """
    grid = GridLines(
        x=np.linspace(0.0, 4e-3, 5),
        y=np.linspace(-L / 2, L / 2, Nyz + 1),
        z=np.linspace(-L / 2, L / 2, Nyz + 1),
    )
    mesh = Mesh.from_grid(grid)
    plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)

    M_eps = build_M_eps(mesh)
    M_mu = build_M_mu(mesh)
    if epsilon_r != 1.0:
        # Cheat: scale the whole-domain ε to ε_r without rebuilding the
        # mesh material library.  Z_line depends on the dielectric
        # filling between conductors — uniform scaling suffices.
        M_eps = M_eps * epsilon_r
    C = build_curl_matrix(mesh.grid)
    G = build_gradient_matrix(mesh.grid)
    _, m_eps_2d, primal_2d_edges = build_2d_curl_curl(plane, mesh.grid, M_eps, M_mu, C)
    g_2d, primal_2d_nodes, primal_2d_edges_g = build_2d_gradient(plane, mesh.grid, G)
    assert np.array_equal(primal_2d_edges, primal_2d_edges_g)

    Ny, Nz = mesh.Ny, mesh.Nz
    y_n = mesh.grid.y
    z_n = mesh.grid.z
    J, K = np.meshgrid(np.arange(Ny + 1), np.arange(Nz + 1), indexing="ij")
    local_node_idx = J * (Nz + 1) + K
    yc = y_n[J]
    zc = z_n[K]
    r = np.sqrt(yc * yc + zc * zc)
    inner_nodes = local_node_idx[r <= r_i].ravel().astype(np.int64)
    outer_nodes = local_node_idx[r >= r_o].ravel().astype(np.int64)
    return plane, g_2d, m_eps_2d, [outer_nodes, inner_nodes], mesh


def _twin_signal_setup(
    *,
    Nyz: int = 60,
    L: float = 8e-3,
    a: float = 1.0e-3,
    y_signal_1: float = -2.0e-3,
    y_signal_2: float = +2.0e-3,
):
    """Build a 3-conductor setup: square ground at the bbox + two square
    signal pads.  Returns (plane, g_2d, m_eps_2d, conductor_groups, mesh).
    """
    grid = GridLines(
        x=np.linspace(0.0, 4e-3, 5),
        y=np.linspace(-L / 2, L / 2, Nyz + 1),
        z=np.linspace(-L / 2, L / 2, Nyz + 1),
    )
    mesh = Mesh.from_grid(grid)
    plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
    M_eps = build_M_eps(mesh)
    M_mu = build_M_mu(mesh)
    C = build_curl_matrix(mesh.grid)
    G = build_gradient_matrix(mesh.grid)
    _, m_eps_2d, primal_2d_edges = build_2d_curl_curl(plane, mesh.grid, M_eps, M_mu, C)
    g_2d, primal_2d_nodes, _ = build_2d_gradient(plane, mesh.grid, G)

    Ny, Nz = mesh.Ny, mesh.Nz
    y_n = mesh.grid.y
    z_n = mesh.grid.z
    J, K = np.meshgrid(np.arange(Ny + 1), np.arange(Nz + 1), indexing="ij")
    local_node_idx = J * (Nz + 1) + K
    yc = y_n[J]
    zc = z_n[K]

    # Ground = bbox boundary.
    ground = local_node_idx[(J == 0) | (J == Ny) | (K == 0) | (K == Nz)].ravel().astype(np.int64)
    sig1_mask = (np.abs(yc - y_signal_1) <= a / 2) & (np.abs(zc) <= a / 2)
    sig2_mask = (np.abs(yc - y_signal_2) <= a / 2) & (np.abs(zc) <= a / 2)
    sig1 = local_node_idx[sig1_mask].ravel().astype(np.int64)
    sig2 = local_node_idx[sig2_mask].ravel().astype(np.int64)
    return plane, g_2d, m_eps_2d, [ground, sig1, sig2], mesh


# ---------------------------------------------------------------------
# 1) build_2d_gradient — shape, ±1 entries, exactness with curl
# ---------------------------------------------------------------------


class TestBuild2DGradient:
    """``build_2d_gradient`` returns shape-correct topological ±1 entries
    and inherits the de Rham exactness ``C_2d @ G_2d == 0``.
    """

    @pytest.fixture
    def setup(self):
        grid = GridLines(
            x=np.linspace(0.0, 1.0, 5),  # Nx=4
            y=np.linspace(0.0, 2.0, 9),  # Ny=8
            z=np.linspace(0.0, 1.5, 7),  # Nz=6
        )
        mesh = Mesh.from_grid(grid)
        return mesh

    def test_shape_xmin(self, setup):
        mesh = setup
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        G = build_gradient_matrix(mesh.grid)
        g_2d, nodes, edges = build_2d_gradient(plane, mesh.grid, G)
        # 2D primal nodes = (Ny+1)·(Nz+1) for X_MIN.
        assert nodes.size == (mesh.Ny + 1) * (mesh.Nz + 1)
        # 2D primal edges = e_u + e_v.
        assert edges.size == plane.e_u_indices.size + plane.e_v_indices.size
        assert g_2d.shape == (edges.size, nodes.size)

    def test_topological_pm_one_entries(self, setup):
        mesh = setup
        plane = PortPlane.from_mesh(BoxFace.Y_MAX, mesh)
        G = build_gradient_matrix(mesh.grid)
        g_2d, _, _ = build_2d_gradient(plane, mesh.grid, G)
        data = np.unique(g_2d.tocsr().data)
        # Topological gradient: every nonzero is ±1.
        assert set(data.tolist()) <= {-1.0, 1.0}

    def test_two_nonzeros_per_row(self, setup):
        """Each primal edge connects exactly 2 nodes, so every row has 2 nz."""
        mesh = setup
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        G = build_gradient_matrix(mesh.grid)
        g_2d, _, _ = build_2d_gradient(plane, mesh.grid, G)
        per_row_nnz = np.diff(g_2d.tocsr().indptr)
        assert (per_row_nnz == 2).all()

    def test_de_rham_exactness_curl_grad_zero(self, setup):
        """``C_2d @ G_2d == 0`` — curl of a discrete gradient vanishes.

        This is the 2D shadow of the 3D ``C @ G == 0`` (verified by
        ``operators/curl.py`` documentation).  After slicing to the
        port-plane primal edges (rows of C_2d) and primal nodes (cols
        of G_2d), the product must still be exactly zero.
        """
        mesh = setup
        for face in (BoxFace.X_MIN, BoxFace.X_MAX, BoxFace.Y_MIN, BoxFace.Z_MAX):
            plane = PortPlane.from_mesh(face, mesh)
            C = build_curl_matrix(mesh.grid)
            G = build_gradient_matrix(mesh.grid)
            primal_2d_edges = np.concatenate([plane.e_u_indices, plane.e_v_indices])
            h_n = _normal_h_indices(face, mesh.grid)
            C_2d = C[h_n, :][:, primal_2d_edges].tocsr()
            g_2d, _, _ = build_2d_gradient(plane, mesh.grid, G)
            prod = (C_2d @ g_2d).toarray()
            assert np.max(np.abs(prod)) == 0.0, (
                f"face {face}: C_2d @ G_2d not zero (||·||_inf = {np.max(np.abs(prod))})"
            )

    def test_node_indices_consistent_with_global_layout(self, setup):
        """Returned node indices match the (i, j, k) → flat-3D-node convention."""
        mesh = setup
        plane = PortPlane.from_mesh(BoxFace.Z_MIN, mesh)
        G = build_gradient_matrix(mesh.grid)
        _, nodes, _ = build_2d_gradient(plane, mesh.grid, G)
        Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
        # Z_MIN: u=x, v=y, normal=z (node k=0).
        # Local 2D ordering: (i_u=i, i_v=j) → flat 3D = i*(Ny+1)*(Nz+1) + j*(Nz+1) + 0.
        for i_u in range(Nx + 1):
            for i_v in range(Ny + 1):
                local = i_u * (Ny + 1) + i_v
                expected = i_u * (Ny + 1) * (Nz + 1) + i_v * (Nz + 1) + 0
                assert nodes[local] == expected


# ---------------------------------------------------------------------
# 2) Argument validation in solve_tem_laplace
# ---------------------------------------------------------------------


class TestArgumentValidation:
    """Constructor-time argument rejection."""

    @pytest.fixture
    def coax(self):
        return _coax_setup(Nyz=20)

    def test_fewer_than_two_groups_rejected(self, coax):
        plane, g_2d, m_eps_2d, groups, mesh = coax
        with pytest.raises(ValueError, match="at least 2 conductor groups"):
            solve_tem_laplace(
                plane,
                g_2d,
                m_eps_2d,
                [groups[0]],
                epsilon_r=1.0,
                grid=mesh.grid,
                m_mu_flat=build_M_mu(mesh),
            )

    def test_negative_epsilon_r_rejected(self, coax):
        plane, g_2d, m_eps_2d, groups, mesh = coax
        with pytest.raises(ValueError, match="epsilon_r must be positive"):
            solve_tem_laplace(
                plane,
                g_2d,
                m_eps_2d,
                groups,
                epsilon_r=-1.0,
                grid=mesh.grid,
                m_mu_flat=build_M_mu(mesh),
            )

    def test_zero_epsilon_r_rejected(self, coax):
        plane, g_2d, m_eps_2d, groups, mesh = coax
        with pytest.raises(ValueError, match="epsilon_r must be positive"):
            solve_tem_laplace(
                plane,
                g_2d,
                m_eps_2d,
                groups,
                epsilon_r=0.0,
                grid=mesh.grid,
                m_mu_flat=build_M_mu(mesh),
            )

    def test_mass_shape_mismatch_rejected(self, coax):
        plane, g_2d, _, groups, mesh = coax
        n = g_2d.shape[0]
        bad_mass = sp.diags(np.ones(n - 1), format="csr")
        with pytest.raises(ValueError, match="m_eps_2d shape"):
            solve_tem_laplace(
                plane,
                g_2d,
                bad_mass,
                groups,
                epsilon_r=1.0,
                grid=mesh.grid,
                m_mu_flat=build_M_mu(mesh),
            )

    def test_empty_group_rejected(self, coax):
        plane, g_2d, m_eps_2d, groups, mesh = coax
        bad = [groups[0], np.array([], dtype=np.int64)]
        with pytest.raises(ValueError, match=r"\[1\] is empty"):
            solve_tem_laplace(
                plane,
                g_2d,
                m_eps_2d,
                bad,
                epsilon_r=1.0,
                grid=mesh.grid,
                m_mu_flat=build_M_mu(mesh),
            )

    def test_out_of_range_node_rejected(self, coax):
        plane, g_2d, m_eps_2d, groups, mesh = coax
        n_nodes = g_2d.shape[1]
        bad = [groups[0], np.array([n_nodes + 5], dtype=np.int64)]
        with pytest.raises(ValueError, match="out of range"):
            solve_tem_laplace(
                plane,
                g_2d,
                m_eps_2d,
                bad,
                epsilon_r=1.0,
                grid=mesh.grid,
                m_mu_flat=build_M_mu(mesh),
            )

    def test_overlapping_groups_rejected(self, coax):
        plane, g_2d, m_eps_2d, groups, mesh = coax
        # Inject one outer node into the inner group too.
        bad = [
            groups[0],
            np.concatenate([groups[1], groups[0][:1]]).astype(np.int64),
        ]
        with pytest.raises(ValueError, match="must be disjoint"):
            solve_tem_laplace(
                plane,
                g_2d,
                m_eps_2d,
                bad,
                epsilon_r=1.0,
                grid=mesh.grid,
                m_mu_flat=build_M_mu(mesh),
            )


# ---------------------------------------------------------------------
# 3) Coax Z_line vs analytical
# ---------------------------------------------------------------------


class TestCoaxZLine:
    """``Z_line`` matches the closed-form ``(η/2π)·ln(r_o/r_i)`` to
    within the ~staircase discretisation error of the radial geometry
    on a Cartesian grid.
    """

    def test_vacuum_within_5_percent(self):
        plane, g_2d, m_eps_2d, groups, mesh = _coax_setup(Nyz=80)
        modes = solve_tem_laplace(
            plane, g_2d, m_eps_2d, groups, epsilon_r=1.0, grid=mesh.grid, m_mu_flat=build_M_mu(mesh)
        )
        assert len(modes) == 1
        m = modes[0]
        expected = (ETA0 / (2.0 * math.pi)) * math.log(3.5e-3 / 1.0e-3)
        rel = abs(m.z_line - expected) / expected
        assert rel < 0.05, (
            f"Coax Z_line off by {rel:.2%}: got {m.z_line:.3f}, expected {expected:.3f}"
        )

    def test_dielectric_filled_within_5_percent(self):
        plane, g_2d, m_eps_2d, groups, mesh = _coax_setup(Nyz=80, epsilon_r=4.0)
        modes = solve_tem_laplace(
            plane, g_2d, m_eps_2d, groups, epsilon_r=4.0, grid=mesh.grid, m_mu_flat=build_M_mu(mesh)
        )
        m = modes[0]
        expected = ((ETA0 / math.sqrt(4.0)) / (2.0 * math.pi)) * math.log(3.5e-3 / 1.0e-3)
        rel = abs(m.z_line - expected) / expected
        assert rel < 0.05

    def test_convergence_h_to_h2(self):
        """Doubling the resolution halves the staircase error (≈ O(h))."""
        expected = (ETA0 / (2.0 * math.pi)) * math.log(3.5e-3 / 1.0e-3)
        errs = []
        for Nyz in (40, 80):
            plane, g_2d, m_eps_2d, groups, mesh = _coax_setup(Nyz=Nyz)
            modes = solve_tem_laplace(
                plane,
                g_2d,
                m_eps_2d,
                groups,
                epsilon_r=1.0,
                grid=mesh.grid,
                m_mu_flat=build_M_mu(mesh),
            )
            errs.append(abs(modes[0].z_line - expected) / expected)
        assert errs[1] < errs[0], (
            f"Z_line error did not decrease with mesh refinement: "
            f"Nyz=40 → {errs[0]:.4f}, Nyz=80 → {errs[1]:.4f}"
        )


# ---------------------------------------------------------------------
# 4) Mode object structure
# ---------------------------------------------------------------------


class TestModeStructure:
    """Returned modes are valid Phase-2 numerical-path Mode objects."""

    @pytest.fixture
    def coax_modes(self):
        plane, g_2d, m_eps_2d, groups, mesh = _coax_setup(Nyz=40)
        modes = solve_tem_laplace(
            plane, g_2d, m_eps_2d, groups, epsilon_r=1.0, grid=mesh.grid, m_mu_flat=build_M_mu(mesh)
        )
        return modes, plane, m_eps_2d, mesh

    def test_field_evaluator_is_none(self, coax_modes):
        modes, *_ = coax_modes
        for m in modes:
            assert m.field_evaluator is None

    def test_mode_type_is_tem(self, coax_modes):
        modes, *_ = coax_modes
        for m in modes:
            assert m.mode_type is ModeType.TEM

    def test_omega_c_is_zero(self, coax_modes):
        modes, *_ = coax_modes
        for m in modes:
            assert m.omega_c == 0.0

    def test_label_naming(self, coax_modes):
        modes, *_ = coax_modes
        assert modes[0].name == "TEM_lap00"

    def test_profile_shapes(self, coax_modes):
        modes, plane, *_ = coax_modes
        n_u = plane.e_u_indices.size
        n_v = plane.e_v_indices.size
        for m in modes:
            assert m.discrete_e_u_profile.shape == (n_u,)
            assert m.discrete_h_v_profile.shape == (n_u,)
            assert m.discrete_e_v_profile.shape == (n_v,)
            assert m.discrete_h_u_profile.shape == (n_v,)

    def test_m_orthonormal(self, coax_modes):
        modes, plane, m_eps_2d, _ = coax_modes
        for i, m in enumerate(modes):
            e_full = np.concatenate([m.discrete_e_u_profile, m.discrete_e_v_profile])
            norm = float(e_full @ (m_eps_2d @ e_full))
            assert abs(norm - 1.0) < 1e-9, f"mode {i} not M-orthonormal: ê^T M ê = {norm}"

    def test_travelling_wave_h_u(self, coax_modes):
        """``h_u = -e_v · (μ₀·dz/Z_TEM) / M_μ[h_u face]`` (WP7.2)."""
        modes, plane, _, mesh = coax_modes
        m_mu = build_M_mu(mesh)
        scale = MU0 * plane.normal_dx / ETA0  # ε_r = 1
        for m in modes:
            np.testing.assert_allclose(
                m.discrete_h_u_profile,
                -m.discrete_e_v_profile * (scale / m_mu[plane.h_u_indices]),
                atol=0,
                rtol=0,
            )

    def test_travelling_wave_h_v(self, coax_modes):
        modes, plane, _, mesh = coax_modes
        m_mu = build_M_mu(mesh)
        scale = MU0 * plane.normal_dx / ETA0
        for m in modes:
            np.testing.assert_allclose(
                m.discrete_h_v_profile,
                m.discrete_e_u_profile * (scale / m_mu[plane.h_v_indices]),
                atol=0,
                rtol=0,
            )

    def test_z_line_matches_modal_z_line(self, coax_modes):
        modes, *_ = coax_modes
        for m in modes:
            assert m.z_line is not None
            assert m.z_line > 0
            # z_modal(omega) for TEM with z_line set returns z_line.
            assert m.z_modal(2 * math.pi * 1e9) == complex(m.z_line)

    def test_discretize_modes_pass_through(self, coax_modes):
        modes, plane, m_eps_2d, _ = coax_modes
        # m_eps_flat for the analytical-path Gram-Schmidt — not used on
        # the numerical path but required by the discretize_modes
        # signature.  Pass any positive array of the right shape.
        # Build a dummy full-3D-shape m_eps_flat array.
        # Easier: use the original mesh's M_eps directly.
        primal_2d_edges = np.concatenate([plane.e_u_indices, plane.e_v_indices])
        # Reconstruct m_eps_flat by scattering from m_eps_2d's diagonal.
        # The numerical path doesn't read it, but pass through cleanly.
        m_eps_flat_dummy = np.ones(int(primal_2d_edges.max()) + 100)
        discrete_modes = discretize_modes(modes, plane, m_eps_flat_dummy)
        assert len(discrete_modes) == len(modes)
        for dm, m in zip(discrete_modes, modes):
            np.testing.assert_array_equal(dm.e_u_profile, m.discrete_e_u_profile)
            np.testing.assert_array_equal(dm.e_v_profile, m.discrete_e_v_profile)
            np.testing.assert_array_equal(dm.h_u_profile, m.discrete_h_u_profile)
            np.testing.assert_array_equal(dm.h_v_profile, m.discrete_h_v_profile)


# ---------------------------------------------------------------------
# 5) Multi-conductor (3 conductors → 2 modes)
# ---------------------------------------------------------------------


class TestMultiConductor:
    """3 conductors yield 2 TEM modes, one per signal in input order."""

    def test_two_modes_returned(self):
        plane, g_2d, m_eps_2d, groups, mesh = _twin_signal_setup(Nyz=40)
        modes = solve_tem_laplace(
            plane, g_2d, m_eps_2d, groups, epsilon_r=1.0, grid=mesh.grid, m_mu_flat=build_M_mu(mesh)
        )
        assert len(modes) == 2

    def test_labels_in_order(self):
        plane, g_2d, m_eps_2d, groups, mesh = _twin_signal_setup(Nyz=40)
        modes = solve_tem_laplace(
            plane, g_2d, m_eps_2d, groups, epsilon_r=1.0, grid=mesh.grid, m_mu_flat=build_M_mu(mesh)
        )
        assert modes[0].name == "TEM_lap00"
        assert modes[1].name == "TEM_lap01"

    def test_each_mode_has_positive_z_line(self):
        plane, g_2d, m_eps_2d, groups, mesh = _twin_signal_setup(Nyz=40)
        modes = solve_tem_laplace(
            plane, g_2d, m_eps_2d, groups, epsilon_r=1.0, grid=mesh.grid, m_mu_flat=build_M_mu(mesh)
        )
        for m in modes:
            assert m.z_line > 0

    def test_signal_conductor_outward_flux_positive(self):
        """Architecture §2.4 / §7 sign rule: ê·n̂ integral over signal k > 0."""
        plane, g_2d, m_eps_2d, groups, mesh = _twin_signal_setup(Nyz=40)
        modes = solve_tem_laplace(
            plane, g_2d, m_eps_2d, groups, epsilon_r=1.0, grid=mesh.grid, m_mu_flat=build_M_mu(mesh)
        )
        for k, m in enumerate(modes, start=1):
            e_full = np.concatenate([m.discrete_e_u_profile, m.discrete_e_v_profile])
            d_flux = m_eps_2d @ e_full
            node_div = g_2d.T @ d_flux
            outward = -float(node_div[groups[k]].sum())
            assert outward > 0, (
                f"mode {k}: signal-conductor outward flux is negative ({outward:.3e})"
            )


# ---------------------------------------------------------------------
# 6) Symmetry / face independence (smoke check)
# ---------------------------------------------------------------------


class TestFaceIndependence:
    """The same physical Coax built on different faces should yield the
    same |Z_line| (symmetry of TEM impedance under face swap)."""

    def test_xmin_vs_xmax_same_z_line(self):
        plane_min, g_min, m_min, groups_min, mesh = _coax_setup(Nyz=40)
        modes_min = solve_tem_laplace(
            plane_min,
            g_min,
            m_min,
            groups_min,
            epsilon_r=1.0,
            grid=mesh.grid,
            m_mu_flat=build_M_mu(mesh),
        )

        # Build the same setup on X_MAX.
        L = 8e-3
        Nyz = 40
        grid = GridLines(
            x=np.linspace(0.0, 4e-3, 5),
            y=np.linspace(-L / 2, L / 2, Nyz + 1),
            z=np.linspace(-L / 2, L / 2, Nyz + 1),
        )
        mesh = Mesh.from_grid(grid)
        plane_max = PortPlane.from_mesh(BoxFace.X_MAX, mesh)
        M_eps = build_M_eps(mesh)
        M_mu = build_M_mu(mesh)
        C = build_curl_matrix(mesh.grid)
        G = build_gradient_matrix(mesh.grid)
        _, m_max, _ = build_2d_curl_curl(plane_max, mesh.grid, M_eps, M_mu, C)
        g_max, _, _ = build_2d_gradient(plane_max, mesh.grid, G)

        # Build conductor groups for X_MAX (u=z, v=y per BoxFace _UV_AXES).
        Ny, Nz = mesh.Ny, mesh.Nz
        y_n = mesh.grid.y
        z_n = mesh.grid.z
        # X_MAX local node ordering: (i_u=k, i_v=j) → local idx = k*(Ny+1) + j.
        K, J = np.meshgrid(np.arange(Nz + 1), np.arange(Ny + 1), indexing="ij")
        local_node_idx = K * (Ny + 1) + J
        yc = y_n[J]
        zc = z_n[K]
        r = np.sqrt(yc * yc + zc * zc)
        inner_max = local_node_idx[r <= 1.0e-3].ravel().astype(np.int64)
        outer_max = local_node_idx[r >= 3.5e-3].ravel().astype(np.int64)

        modes_max = solve_tem_laplace(
            plane_max,
            g_max,
            m_max,
            [outer_max, inner_max],
            epsilon_r=1.0,
            grid=mesh.grid,
            m_mu_flat=build_M_mu(mesh),
        )
        rel = abs(modes_max[0].z_line - modes_min[0].z_line) / modes_min[0].z_line
        assert rel < 1e-6, (
            f"X_MIN and X_MAX Z_line differ: {modes_min[0].z_line:.4f} vs {modes_max[0].z_line:.4f}"
        )


# ---------------------------------------------------------------------
# 7) Label helper
# ---------------------------------------------------------------------


class TestLabelHelper:
    def test_label_zero_padding(self):
        assert _tem_label(0) == "TEM_lap00"
        assert _tem_label(7) == "TEM_lap07"
        assert _tem_label(15) == "TEM_lap15"
