"""Tests for magnelio.ports._modal.numerical_2d — Numerical2DModeSolver.

Phase 2a step 3 of the numerical-mode-solver work order
(`reference_architecture_phase2_mode_solver.md` §3.1 / §5).

Three layers of validation:

1. **Constructor invariants.**  Shape-mismatch arguments are rejected
   at ``__post_init__``; ``solve()`` rejects bad ``n_modes`` / ``f_calc``.
2. **Test 1N (TE cutoff) — WR-90.**  The solver returns TE_10 / TE_20 /
   TE_01 angular cutoffs within 1 % of the analytical
   ``omega_c = c·k_c`` at Ny=60, Nz=30 resolution.  PEC walls applied
   via ``pec_edge_mask``.
3. **Mode object structure.**  Resulting modes have
   ``field_evaluator=None``, the four ``discrete_*_profile`` arrays
   filled with the right shapes, and the eigenvectors are
   M-orthonormal (``v^T M v = 1`` to machine precision).  H-profiles
   satisfy the modal Ohm's law ``H_u = -E_v / Z`` and the resulting
   Poynting integral is positive (forward orientation).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse as sp

from magnelio._operators.curl import build_curl_matrix, build_gradient_matrix
from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.ports._modal import (
    BoxFace,
    Mode,
    ModeType,
    Numerical2DModeSolver,
    PortPlane,
    discretize_modes,
)
from magnelio.ports._modal.curl_curl_2d import (
    build_2d_curl_curl,
    build_2d_gradient,
    build_2d_tm_curl_curl,
)

C0 = 2.99792458e8
WR90_A = 22.86e-3
WR90_B = 10.16e-3


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _wr90_mesh(*, Ny: int = 60, Nz: int = 30, Nx: int = 4, L_x: float = 10e-3) -> Mesh:
    """Hollow WR-90 cross-section (Ny, Nz) extruded by Nx cells along x."""
    grid = GridLines(
        x=np.linspace(0.0, L_x, Nx + 1),
        y=np.linspace(0.0, WR90_A, Ny + 1),
        z=np.linspace(0.0, WR90_B, Nz + 1),
    )
    return Mesh.from_grid(grid)


def _build_KM(mesh: Mesh, face: BoxFace = BoxFace.X_MIN):
    plane = PortPlane.from_mesh(face, mesh)
    M_eps = build_M_eps(mesh)
    M_mu = build_M_mu(mesh)
    C = build_curl_matrix(mesh.grid)
    K, M, primal_2d = build_2d_curl_curl(plane, mesh.grid, M_eps, M_mu, C)
    return plane, K, M, primal_2d, M_mu


def _wall_pec_mask(plane: PortPlane, *, a: float, b: float) -> np.ndarray:
    """Boolean mask over (e_u, e_v) — True for edges tangent to a PEC wall.

    Mirrors the helper in ``test_modal_curl_curl_2d.py``.  For a hollow
    rectangular WG with PEC walls at u ∈ {0, a} and v ∈ {0, b}, the
    tangential E-edges are u-edges whose v-node is on the v-wall and
    v-edges whose u-node is on the u-wall.
    """
    eps = 1e-9
    u_v = plane.u_edge_uv[:, 1]
    u_pec = (np.abs(u_v) < eps) | (np.abs(u_v - b) < eps)
    v_u = plane.v_edge_uv[:, 0]
    v_pec = (np.abs(v_u) < eps) | (np.abs(v_u - a) < eps)
    return np.concatenate([u_pec, v_pec])


def _wr90_solver(face: BoxFace = BoxFace.X_MIN) -> Numerical2DModeSolver:
    mesh = _wr90_mesh()
    plane, K, M, primal_2d, m_mu = _build_KM(mesh, face=face)
    pec = _wall_pec_mask(plane, a=WR90_A, b=WR90_B)
    return Numerical2DModeSolver(
        plane=plane,
        K=K,
        M=M,
        primal_2d_indices=primal_2d,
        m_mu_flat=m_mu,
        pec_edge_mask=pec,
        epsilon_r=1.0,
        mode_type=ModeType.TE,
    )


def _wr90_tm_solver(
    *,
    Ny: int = 60,
    Nz: int = 30,
    face: BoxFace = BoxFace.X_MIN,
    epsilon_r: float = 1.0,
) -> tuple[Numerical2DModeSolver, Mesh]:
    """Build a TM-path solver on a hollow WR-90 cross-section.

    Returns the solver plus the mesh (the latter so tests can also
    construct curl operators for the curl-free TM E_t check).
    """
    mesh = _wr90_mesh(Ny=Ny, Nz=Nz)
    plane = PortPlane.from_mesh(face, mesh)
    M_eps = build_M_eps(mesh)
    M_mu = build_M_mu(mesh)
    C = build_curl_matrix(mesh.grid)
    G = build_gradient_matrix(mesh.grid)
    K, M, primal_2d = build_2d_curl_curl(plane, mesh.grid, M_eps, M_mu, C)
    g_2d, _, _ = build_2d_gradient(plane, mesh.grid, G)
    L_node, M_node, pec_node_mask, _ = build_2d_tm_curl_curl(
        plane,
        mesh.grid,
        M_eps,
        M_mu,
        C,
    )
    pec_edge = _wall_pec_mask(plane, a=WR90_A, b=WR90_B)
    solver = Numerical2DModeSolver(
        plane=plane,
        K=K,
        M=M,
        primal_2d_indices=primal_2d,
        m_mu_flat=M_mu,
        pec_edge_mask=pec_edge,
        epsilon_r=epsilon_r,
        mode_type=ModeType.TM,
        g_2d=g_2d,
        L_node=L_node,
        M_node=M_node,
        pec_node_mask=pec_node_mask,
    )
    return solver, mesh


# ---------------------------------------------------------------------
# 1) Constructor invariants
# ---------------------------------------------------------------------


class TestConstructorInvariants:
    def test_K_shape_mismatch_rejected(self):
        plane, K, M, primal_2d, m_mu = _build_KM(_wr90_mesh(Ny=8, Nz=4))
        bad_K = sp.csr_matrix((primal_2d.size + 1, primal_2d.size + 1))
        with pytest.raises(ValueError, match="K shape"):
            Numerical2DModeSolver(
                plane=plane,
                K=bad_K,
                M=M,
                primal_2d_indices=primal_2d,
                m_mu_flat=m_mu,
            )

    def test_M_shape_mismatch_rejected(self):
        plane, K, M, primal_2d, m_mu = _build_KM(_wr90_mesh(Ny=8, Nz=4))
        bad_M = sp.csr_matrix((primal_2d.size + 1, primal_2d.size + 1))
        with pytest.raises(ValueError, match="M shape"):
            Numerical2DModeSolver(
                plane=plane,
                K=K,
                M=bad_M,
                primal_2d_indices=primal_2d,
                m_mu_flat=m_mu,
            )

    def test_pec_mask_shape_mismatch_rejected(self):
        plane, K, M, primal_2d, m_mu = _build_KM(_wr90_mesh(Ny=8, Nz=4))
        bad_mask = np.zeros(primal_2d.size + 5, dtype=bool)
        with pytest.raises(ValueError, match="pec_edge_mask shape"):
            Numerical2DModeSolver(
                plane=plane,
                K=K,
                M=M,
                primal_2d_indices=primal_2d,
                m_mu_flat=m_mu,
                pec_edge_mask=bad_mask,
            )

    def test_pec_mask_non_bool_rejected(self):
        plane, K, M, primal_2d, m_mu = _build_KM(_wr90_mesh(Ny=8, Nz=4))
        bad_mask = np.zeros(primal_2d.size, dtype=int)
        with pytest.raises(ValueError, match="boolean"):
            Numerical2DModeSolver(
                plane=plane,
                K=K,
                M=M,
                primal_2d_indices=primal_2d,
                m_mu_flat=m_mu,
                pec_edge_mask=bad_mask,
            )

    def test_negative_epsilon_rejected(self):
        plane, K, M, primal_2d, m_mu = _build_KM(_wr90_mesh(Ny=8, Nz=4))
        with pytest.raises(ValueError, match="epsilon_r"):
            Numerical2DModeSolver(
                plane=plane,
                K=K,
                M=M,
                primal_2d_indices=primal_2d,
                m_mu_flat=m_mu,
                epsilon_r=-1.0,
            )

    def test_tem_mode_type_without_conductor_groups_rejected(self):
        """``mode_type=TEM`` requires ``conductor_node_groups`` (TEM is
        computed via the Laplace dispatch, not the curl-curl eigsh)."""
        plane, K, M, primal_2d, m_mu = _build_KM(_wr90_mesh(Ny=8, Nz=4))
        with pytest.raises(ValueError, match="conductor_node_groups"):
            Numerical2DModeSolver(
                plane=plane,
                K=K,
                M=M,
                primal_2d_indices=primal_2d,
                m_mu_flat=m_mu,
                mode_type=ModeType.TEM,
            )

    def test_solve_rejects_zero_n_modes(self):
        solver = _wr90_solver()
        with pytest.raises(ValueError, match="n_modes"):
            solver.solve(n_modes=0, f_calc=10e9)

    def test_solve_rejects_zero_f_calc(self):
        solver = _wr90_solver()
        with pytest.raises(ValueError, match="f_calc"):
            solver.solve(n_modes=1, f_calc=0.0)


# ---------------------------------------------------------------------
# 2) Test 1N — WR-90 TE cutoffs vs analytical
# ---------------------------------------------------------------------


class TestWR90Cutoffs:
    """TE_10 / TE_20 / TE_01 cutoffs within 1 % at Ny=60, Nz=30."""

    @pytest.fixture(scope="class")
    def modes(self) -> list[Mode]:
        solver = _wr90_solver()
        return solver.solve(n_modes=4, f_calc=10e9)

    def test_returns_requested_count(self, modes):
        assert len(modes) == 4

    def test_modes_sorted_by_omega_c(self, modes):
        omega_c = [m.omega_c for m in modes]
        assert omega_c == sorted(omega_c)

    def test_te10_cutoff_within_one_percent(self, modes):
        f_c_analytical = C0 / (2 * WR90_A)
        f_c_num = modes[0].omega_c / (2 * math.pi)
        rel_err = abs(f_c_num - f_c_analytical) / f_c_analytical
        assert rel_err < 0.01, (
            f"TE_10 f_c off by {rel_err:.4%}: numerical {f_c_num / 1e9:.4f} GHz, "
            f"analytical {f_c_analytical / 1e9:.4f} GHz"
        )

    def test_te20_cutoff_within_one_percent(self, modes):
        # The next TE_m0 is TE_20 at f_c = c/a.  In the lowest-3-modes
        # ordering, TE_20 sits at index 1 if (a > 2b), else TE_01.
        # WR-90: a = 22.86 mm, b = 10.16 mm — a/b ≈ 2.25 > 2, so TE_01
        # cutoff (c/2b = 14.74 GHz) is above TE_20 (c/a = 13.11 GHz).
        f_c_analytical = C0 / WR90_A
        f_c_num = modes[1].omega_c / (2 * math.pi)
        rel_err = abs(f_c_num - f_c_analytical) / f_c_analytical
        assert rel_err < 0.01, f"TE_20 f_c off by {rel_err:.4%}"

    def test_te01_cutoff_within_two_percent(self, modes):
        # TE_01 sits at index 2 for WR-90 (Nz=30 is the lower-resolution
        # axis, so the 2 % tolerance per Test 1N).
        f_c_analytical = C0 / (2 * WR90_B)
        f_c_num = modes[2].omega_c / (2 * math.pi)
        rel_err = abs(f_c_num - f_c_analytical) / f_c_analytical
        assert rel_err < 0.02


# ---------------------------------------------------------------------
# 3) Mode object structure
# ---------------------------------------------------------------------


class TestModeStructure:
    """Each Mode is a valid Phase-2 numerical-path Mode object."""

    @pytest.fixture(scope="class")
    def solver_and_modes(self):
        solver = _wr90_solver()
        modes = solver.solve(n_modes=3, f_calc=10e9)
        return solver, modes

    def test_field_evaluator_is_none(self, solver_and_modes):
        _, modes = solver_and_modes
        for m in modes:
            assert m.field_evaluator is None

    def test_discrete_profiles_have_correct_shapes(self, solver_and_modes):
        solver, modes = solver_and_modes
        n_u = solver.plane.e_u_indices.size
        n_v = solver.plane.e_v_indices.size
        for m in modes:
            assert m.discrete_e_u_profile.shape == (n_u,)
            assert m.discrete_h_v_profile.shape == (n_u,)
            assert m.discrete_e_v_profile.shape == (n_v,)
            assert m.discrete_h_u_profile.shape == (n_v,)

    def test_modes_are_M_orthonormal(self, solver_and_modes):
        """``v^T M v = 1`` per mode (eigsh's default for the gen. problem)."""
        solver, modes = solver_and_modes
        M_diag = solver.M.diagonal()
        for m in modes:
            v = np.concatenate([m.discrete_e_u_profile, m.discrete_e_v_profile])
            norm = float(np.dot(M_diag, v * v))
            assert abs(norm - 1.0) < 1e-9

    def test_modes_are_M_orthogonal(self, solver_and_modes):
        """``v_i^T M v_j = 0`` for i != j (eigsh's gen.-eigvec property)."""
        solver, modes = solver_and_modes
        M_diag = solver.M.diagonal()
        for i, mi in enumerate(modes):
            for j, mj in enumerate(modes):
                if i == j:
                    continue
                vi = np.concatenate([mi.discrete_e_u_profile, mi.discrete_e_v_profile])
                vj = np.concatenate([mj.discrete_e_u_profile, mj.discrete_e_v_profile])
                cross = float(np.dot(M_diag, vi * vj))
                # Tolerance 1e-7 covers ARPACK's residual on shift-invert.
                assert abs(cross) < 1e-7, f"mode {i}, {j} not M-orthogonal (cross={cross:.3e})"

    def test_h_profile_travelling_wave_form(self, solver_and_modes):
        """``h = ±e_co·(μ₀·dz/Z)/M_μ[face]`` at f_calc (WP7.2)."""
        from magnelio.ports._modal.mode import MU0

        solver, modes = solver_and_modes
        plane = solver.plane
        m_mu = solver.m_mu_flat
        omega = 2 * math.pi * 10e9
        for m in modes:
            z_real = abs(m.z_wave(omega))
            scale = MU0 * plane.normal_dx / z_real
            np.testing.assert_allclose(
                m.discrete_h_u_profile,
                -m.discrete_e_v_profile * (scale / m_mu[plane.h_u_indices]),
                rtol=1e-12,
                atol=0,
            )
            np.testing.assert_allclose(
                m.discrete_h_v_profile,
                m.discrete_e_u_profile * (scale / m_mu[plane.h_v_indices]),
                rtol=1e-12,
                atol=0,
            )

    def test_poynting_forward_oriented(self, solver_and_modes):
        """Field-level Poynting integral S_n > 0 (forward propagation)."""
        _, modes = solver_and_modes
        for m in modes:
            # S_n = E_u·H_v - E_v·H_u   (sum over edges, no weighting needed
            # for the sign, since both terms are sums of squares divided
            # by Z > 0).
            s_n = float(
                np.dot(m.discrete_e_u_profile, m.discrete_h_v_profile)
                - np.dot(m.discrete_e_v_profile, m.discrete_h_u_profile)
            )
            assert s_n > 0.0


# ---------------------------------------------------------------------
# 4) discretize_modes pass-through compatibility
# ---------------------------------------------------------------------


class TestDiscretizeModesPassThrough:
    """The numerical solver's modes flow through discretize_modes byte-for-byte."""

    def test_round_trip_through_discretize_modes(self):
        mesh = _wr90_mesh()
        plane, K, M, primal_2d, m_mu = _build_KM(mesh)
        pec = _wall_pec_mask(plane, a=WR90_A, b=WR90_B)
        solver = Numerical2DModeSolver(
            plane=plane,
            K=K,
            M=M,
            primal_2d_indices=primal_2d,
            m_mu_flat=m_mu,
            pec_edge_mask=pec,
            epsilon_r=1.0,
            mode_type=ModeType.TE,
        )
        modes = solver.solve(n_modes=2, f_calc=10e9)

        m_eps_flat = build_M_eps(mesh)
        discrete = discretize_modes(modes, plane, m_eps_flat)

        assert len(discrete) == 2
        for src, dst in zip(modes, discrete):
            np.testing.assert_array_equal(
                dst.e_u_profile,
                src.discrete_e_u_profile,
            )
            np.testing.assert_array_equal(
                dst.e_v_profile,
                src.discrete_e_v_profile,
            )


# ---------------------------------------------------------------------
# 5) Sign-flip determinism
# ---------------------------------------------------------------------


class TestSignFlipDeterminism:
    """``_resolve_sign`` makes the eigenvector sign deterministic across
    repeated solves, even though ``eigsh`` starts from a random vector.

    The historical assertion (largest-magnitude entry positive, the
    pre-reciprocity-fix argmax convention) is ill-defined for modes
    with two equal-magnitude antiphase peaks: TE20's two extrema are
    float-equal up to eigsh start-vector noise, so *which* one carries
    the largest magnitude is random even though the port-symmetric
    overlap criterion fixes the physical sign deterministically
    (measured: linear-overlap decision margin ≥ 0.31 of the
    Cauchy-Schwarz bound vs ~1e-15 noise on the orthogonal test
    functions).  The old test failed ~7 % of runs on the TE20 mode.
    """

    def test_sign_deterministic_across_solves(self):
        runs = []
        for _ in range(2):
            solver = _wr90_solver()
            modes = solver.solve(n_modes=3, f_calc=10e9)
            runs.append(
                [np.concatenate([m.discrete_e_u_profile, m.discrete_e_v_profile]) for m in modes]
            )
        for k, (va, vb) in enumerate(zip(*runs)):
            corr = float(np.dot(va, vb)) / (np.linalg.norm(va) * np.linalg.norm(vb))
            assert corr > 0.999, (
                f"mode {k}: eigenvector not deterministic across solves "
                f"(normalised correlation {corr:+.6f}; -1 means a sign "
                f"flip slipped through _resolve_sign)"
            )


# ---------------------------------------------------------------------
# 6) TEM dispatch (Phase 2b step 8)
# ---------------------------------------------------------------------


COAX_RI = 1.0e-3
COAX_RO = 3.5e-3
COAX_BBOX = 8.0e-3  # half-edge length in (y, z)


def _coax_solver_setup(*, Nyz: int = 80, epsilon_r: float = 1.0):
    """Build a Numerical2DModeSolver pre-loaded for the TEM dispatch.

    Mirrors the helper in ``test_modal_tem_laplace.py``: cell-centred
    square (y, z) grid with Coax conductors identified by radius
    threshold on the 2D primal nodes.
    """
    from magnelio._operators.curl import build_gradient_matrix
    from magnelio.ports._modal.curl_curl_2d import build_2d_gradient

    grid = GridLines(
        x=np.linspace(0.0, 4e-3, 5),
        y=np.linspace(-COAX_BBOX / 2, COAX_BBOX / 2, Nyz + 1),
        z=np.linspace(-COAX_BBOX / 2, COAX_BBOX / 2, Nyz + 1),
    )
    mesh = Mesh.from_grid(grid)
    plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
    M_eps = build_M_eps(mesh) * epsilon_r  # uniform-scale for ε_r ≠ 1
    M_mu = build_M_mu(mesh)
    C = build_curl_matrix(mesh.grid)
    G = build_gradient_matrix(mesh.grid)
    K, M_2d, primal_2d_edges = build_2d_curl_curl(plane, mesh.grid, M_eps, M_mu, C)
    g_2d, _, _ = build_2d_gradient(plane, mesh.grid, G)

    Ny, Nz = mesh.Ny, mesh.Nz
    y_n = mesh.grid.y
    z_n = mesh.grid.z
    J, K2 = np.meshgrid(np.arange(Ny + 1), np.arange(Nz + 1), indexing="ij")
    local_idx = J * (Nz + 1) + K2
    yc = y_n[J]
    zc = z_n[K2]
    r = np.sqrt(yc * yc + zc * zc)
    inner = local_idx[r <= COAX_RI].ravel().astype(np.int64)
    outer = local_idx[r >= COAX_RO].ravel().astype(np.int64)

    solver = Numerical2DModeSolver(
        plane=plane,
        K=K,
        M=M_2d,
        primal_2d_indices=primal_2d_edges,
        m_mu_flat=M_mu,
        epsilon_r=epsilon_r,
        g_2d=g_2d,
        conductor_node_groups=[outer, inner],
        grid=mesh.grid,
    )
    return solver, mesh, M_eps


class TestTEMDispatchClassify:
    """``_classify`` returns the right path identifier per construction inputs."""

    def test_classify_te_tm_when_no_conductor_groups(self):
        # ``mode_type=TE`` (default) routes to the curl-curl eigsh path.
        # ``mode_type=TM`` routes to the node-Laplace eigsh path; that is
        # exercised separately by the TM unit tests below.
        solver = _wr90_solver()
        assert solver._classify() == "te"

    def test_classify_tem_when_conductor_groups_set(self):
        solver, _, _ = _coax_solver_setup(Nyz=20)
        assert solver._classify() == "tem"


class TestTEMDispatchValidation:
    """Construction-time validation of the new TEM-path inputs."""

    def test_g2d_without_conductor_groups_rejected(self):
        from magnelio._operators.curl import build_gradient_matrix
        from magnelio.ports._modal.curl_curl_2d import build_2d_gradient

        mesh = _wr90_mesh(Ny=8, Nz=4)
        plane, K, M, primal_2d, m_mu = _build_KM(mesh)
        G = build_gradient_matrix(mesh.grid)
        g_2d, _, _ = build_2d_gradient(plane, mesh.grid, G)
        # ``g_2d`` is shared between the TEM Laplace path (which also
        # needs ``conductor_node_groups``) and the TM eigsh path (which
        # needs the node-Laplace inputs).  With neither downstream
        # consumer set, construction must fail.
        with pytest.raises(ValueError, match="without a downstream consumer"):
            Numerical2DModeSolver(
                plane=plane,
                K=K,
                M=M,
                primal_2d_indices=primal_2d,
                m_mu_flat=m_mu,
                g_2d=g_2d,
                conductor_node_groups=None,
                grid=mesh.grid,
            )

    def test_conductor_groups_without_g2d_rejected(self):
        mesh = _wr90_mesh(Ny=8, Nz=4)
        plane, K, M, primal_2d, m_mu = _build_KM(mesh)
        with pytest.raises(ValueError, match="conductor_node_groups requires g_2d"):
            Numerical2DModeSolver(
                plane=plane,
                K=K,
                M=M,
                primal_2d_indices=primal_2d,
                m_mu_flat=m_mu,
                g_2d=None,
                conductor_node_groups=[
                    np.array([0, 1], dtype=np.int64),
                    np.array([2, 3], dtype=np.int64),
                ],
                grid=mesh.grid,
            )

    def test_single_conductor_group_rejected(self):
        from magnelio._operators.curl import build_gradient_matrix
        from magnelio.ports._modal.curl_curl_2d import build_2d_gradient

        mesh = _wr90_mesh(Ny=8, Nz=4)
        plane, K, M, primal_2d, m_mu = _build_KM(mesh)
        G = build_gradient_matrix(mesh.grid)
        g_2d, _, _ = build_2d_gradient(plane, mesh.grid, G)
        with pytest.raises(ValueError, match="at least 2 groups"):
            Numerical2DModeSolver(
                plane=plane,
                K=K,
                M=M,
                primal_2d_indices=primal_2d,
                m_mu_flat=m_mu,
                g_2d=g_2d,
                conductor_node_groups=[np.array([0, 1], dtype=np.int64)],
                grid=mesh.grid,
            )

    def test_g2d_row_count_mismatch_rejected(self):

        mesh = _wr90_mesh(Ny=8, Nz=4)
        plane, K, M, primal_2d, m_mu = _build_KM(mesh)
        # Wrong-shape g_2d (one extra row).
        bad_g = sp.csr_matrix((primal_2d.size + 1, (mesh.Ny + 1) * (mesh.Nz + 1)))
        with pytest.raises(ValueError, match="g_2d row count"):
            Numerical2DModeSolver(
                plane=plane,
                K=K,
                M=M,
                primal_2d_indices=primal_2d,
                m_mu_flat=m_mu,
                g_2d=bad_g,
                conductor_node_groups=[
                    np.array([0, 1], dtype=np.int64),
                    np.array([2, 3], dtype=np.int64),
                ],
                grid=mesh.grid,
            )

    def test_tem_path_accepts_mode_type_tem(self):
        """``mode_type=TEM`` is now permitted alongside the TEM dispatch."""
        solver, _, _ = _coax_solver_setup(Nyz=20)
        # Reconstruct the solver explicitly with mode_type=TEM (no error).
        Numerical2DModeSolver(
            plane=solver.plane,
            K=solver.K,
            M=solver.M,
            primal_2d_indices=solver.primal_2d_indices,
            m_mu_flat=solver.m_mu_flat,
            epsilon_r=solver.epsilon_r,
            mode_type=ModeType.TEM,
            g_2d=solver.g_2d,
            conductor_node_groups=solver.conductor_node_groups,
            grid=solver.grid,
        )


class TestTEMDispatchSolve:
    """End-to-end TEM dispatch on a Coax setup."""

    def test_returns_one_mode_for_two_conductors(self):
        solver, _, _ = _coax_solver_setup(Nyz=40)
        modes = solver.solve(n_modes=1, f_calc=10e9)
        assert len(modes) == 1

    def test_result_mode_type_is_tem(self):
        solver, _, _ = _coax_solver_setup(Nyz=40)
        modes = solver.solve(n_modes=1, f_calc=10e9)
        assert modes[0].mode_type is ModeType.TEM
        assert modes[0].omega_c == 0.0
        assert modes[0].field_evaluator is None

    def test_n_modes_too_large_raises(self):
        """``n_modes > K - 1`` (here 2 > 1) raises RuntimeError."""
        solver, _, _ = _coax_solver_setup(Nyz=20)
        with pytest.raises(RuntimeError, match="only supports 1"):
            solver.solve(n_modes=2, f_calc=10e9)

    def test_f_calc_zero_allowed_on_tem_path(self):
        """TEM is non-dispersive — ``f_calc=0`` is acceptable on this path
        (in contrast to the TE/TM path which still rejects it)."""
        solver, _, _ = _coax_solver_setup(Nyz=20)
        # No exception expected.
        modes = solver.solve(n_modes=1, f_calc=0.0)
        assert len(modes) == 1

    def test_dispatch_matches_standalone_solve_tem_laplace(self):
        """The Numerical2DModeSolver TEM path is a thin wrapper around
        ``solve_tem_laplace``: byte-for-byte identical Mode profiles."""
        from magnelio.ports._modal.tem_laplace import solve_tem_laplace

        solver, _, _ = _coax_solver_setup(Nyz=40)
        via_dispatch = solver.solve(n_modes=1, f_calc=10e9)[0]
        via_standalone = solve_tem_laplace(
            solver.plane,
            solver.g_2d,
            solver.M,
            solver.conductor_node_groups,
            solver.epsilon_r,
            grid=solver.grid,
            m_mu_flat=solver.m_mu_flat,
        )[0]
        np.testing.assert_array_equal(
            via_dispatch.discrete_e_u_profile,
            via_standalone.discrete_e_u_profile,
        )
        np.testing.assert_array_equal(
            via_dispatch.discrete_e_v_profile,
            via_standalone.discrete_e_v_profile,
        )
        np.testing.assert_array_equal(
            via_dispatch.discrete_h_u_profile,
            via_standalone.discrete_h_u_profile,
        )
        np.testing.assert_array_equal(
            via_dispatch.discrete_h_v_profile,
            via_standalone.discrete_h_v_profile,
        )
        assert via_dispatch.z_line == via_standalone.z_line


class TestNumericalCoaxVsAnalytical:
    """Numerical Coax TEM via the dispatch matches the analytical Coax
    closed-form to discretisation accuracy."""

    def test_z_line_within_5_percent_at_Nyz_80(self):
        from magnelio.ports._modal.coax import CoaxAnalyticalModeSolver

        solver, _, _ = _coax_solver_setup(Nyz=80)
        m_num = solver.solve(n_modes=1, f_calc=10e9)[0]
        m_ana = CoaxAnalyticalModeSolver(
            inner_radius=COAX_RI,
            outer_radius=COAX_RO,
            epsilon_r=1.0,
        ).solve(n_modes=1)[0]
        rel = abs(m_num.z_line - m_ana.z_line) / m_ana.z_line
        assert rel < 0.05, (
            f"Numerical Coax Z_line off by {rel:.2%}: "
            f"got {m_num.z_line:.3f}, analytical {m_ana.z_line:.3f}"
        )

    def test_dielectric_z_line_within_5_percent_at_Nyz_60(self):
        from magnelio.ports._modal.coax import CoaxAnalyticalModeSolver

        solver, _, _ = _coax_solver_setup(Nyz=60, epsilon_r=4.0)
        m_num = solver.solve(n_modes=1, f_calc=10e9)[0]
        m_ana = CoaxAnalyticalModeSolver(
            inner_radius=COAX_RI,
            outer_radius=COAX_RO,
            epsilon_r=4.0,
        ).solve(n_modes=1)[0]
        rel = abs(m_num.z_line - m_ana.z_line) / m_ana.z_line
        assert rel < 0.06

    def test_classify_qtem_when_vacuum_mass_set(self):
        """``_classify`` returns ``"qtem"`` when vacuum mass is supplied."""
        from magnelio._operators.curl import build_gradient_matrix
        from magnelio.ports._modal.curl_curl_2d import build_2d_gradient

        mesh = _wr90_mesh(Ny=8, Nz=4)
        plane, K, M, primal_2d, m_mu = _build_KM(mesh)
        G = build_gradient_matrix(mesh.grid)
        g_2d, _, _ = build_2d_gradient(plane, mesh.grid, G)
        # Use a copy of M as the "vacuum" mass for the smoke test.
        M_vacuum = M.copy()
        solver = Numerical2DModeSolver(
            plane=plane,
            K=K,
            M=M,
            primal_2d_indices=primal_2d,
            m_mu_flat=m_mu,
            g_2d=g_2d,
            conductor_node_groups=[
                np.array([0, 1, 2], dtype=np.int64),
                np.array([3, 4, 5], dtype=np.int64),
            ],
            m_eps_2d_vacuum=M_vacuum,
            grid=mesh.grid,
        )
        assert solver._classify() == "qtem"

    def test_qtem_dispatch_via_solve(self):
        """``solve()`` routes QTEM-class inputs to ``solve_qtem_laplace``;
        verifying the mode name as the simplest cross-check."""

        # Use the Coax-with-half-substrate setup from
        # test_modal_qtem_laplace.py — but build inline to avoid
        # cross-test imports.
        from magnelio._operators.curl import build_gradient_matrix
        from magnelio.ports._modal.curl_curl_2d import build_2d_gradient

        # Tiny Coax without a substrate scaling: QTEM with vacuum=actual
        # → ε_eff = 1, mode label "QTEM_lap00".
        Nyz = 20
        L = 8e-3
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
        K, M_2d, primal_2d_edges = build_2d_curl_curl(plane, mesh.grid, M_eps, M_mu, C)
        g_2d, _, _ = build_2d_gradient(plane, mesh.grid, G)

        Ny, Nz = mesh.Ny, mesh.Nz
        y_n = mesh.grid.y
        z_n = mesh.grid.z
        J, K2 = np.meshgrid(np.arange(Ny + 1), np.arange(Nz + 1), indexing="ij")
        local_idx = J * (Nz + 1) + K2
        yc = y_n[J]
        zc = z_n[K2]
        r = np.sqrt(yc * yc + zc * zc)
        inner = local_idx[r <= COAX_RI].ravel().astype(np.int64)
        outer = local_idx[r >= COAX_RO].ravel().astype(np.int64)

        solver = Numerical2DModeSolver(
            plane=plane,
            K=K,
            M=M_2d,
            primal_2d_indices=primal_2d_edges,
            m_mu_flat=M_mu,
            g_2d=g_2d,
            conductor_node_groups=[outer, inner],
            m_eps_2d_vacuum=M_2d,  # vacuum = actual → ε_eff = 1
            grid=mesh.grid,
        )
        modes = solver.solve(n_modes=1, f_calc=10e9)
        assert len(modes) == 1
        assert modes[0].name == "QTEM_lap00"
        assert abs(modes[0].epsilon_r - 1.0) < 1e-12

    def test_qtem_n_modes_too_large_raises(self):
        from magnelio._operators.curl import build_gradient_matrix
        from magnelio.ports._modal.curl_curl_2d import build_2d_gradient

        Nyz = 20
        L = 8e-3
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
        K, M_2d, primal_2d_edges = build_2d_curl_curl(plane, mesh.grid, M_eps, M_mu, C)
        g_2d, _, _ = build_2d_gradient(plane, mesh.grid, G)

        y_n = mesh.grid.y
        z_n = mesh.grid.z
        J, K2 = np.meshgrid(np.arange(mesh.Ny + 1), np.arange(mesh.Nz + 1), indexing="ij")
        local_idx = J * (mesh.Nz + 1) + K2
        yc = y_n[J]
        zc = z_n[K2]
        r = np.sqrt(yc * yc + zc * zc)
        inner = local_idx[r <= COAX_RI].ravel().astype(np.int64)
        outer = local_idx[r >= COAX_RO].ravel().astype(np.int64)

        solver = Numerical2DModeSolver(
            plane=plane,
            K=K,
            M=M_2d,
            primal_2d_indices=primal_2d_edges,
            m_mu_flat=M_mu,
            g_2d=g_2d,
            conductor_node_groups=[outer, inner],
            m_eps_2d_vacuum=M_2d,
            grid=mesh.grid,
        )
        with pytest.raises(RuntimeError, match="QTEM dispatch.*only supports 1"):
            solver.solve(n_modes=2, f_calc=10e9)

    def test_vacuum_mass_without_conductor_groups_rejected(self):
        mesh = _wr90_mesh(Ny=8, Nz=4)
        plane, K, M, primal_2d, m_mu = _build_KM(mesh)
        with pytest.raises(ValueError, match="m_eps_2d_vacuum requires"):
            Numerical2DModeSolver(
                plane=plane,
                K=K,
                M=M,
                primal_2d_indices=primal_2d,
                m_mu_flat=m_mu,
                m_eps_2d_vacuum=M,
            )

    def test_vacuum_mass_shape_mismatch_rejected(self):
        from magnelio._operators.curl import build_gradient_matrix
        from magnelio.ports._modal.curl_curl_2d import build_2d_gradient

        mesh = _wr90_mesh(Ny=8, Nz=4)
        plane, K, M, primal_2d, m_mu = _build_KM(mesh)
        G = build_gradient_matrix(mesh.grid)
        g_2d, _, _ = build_2d_gradient(plane, mesh.grid, G)
        bad_vacuum = sp.diags(np.ones(primal_2d.size + 1), format="csr")
        with pytest.raises(ValueError, match="m_eps_2d_vacuum shape"):
            Numerical2DModeSolver(
                plane=plane,
                K=K,
                M=M,
                primal_2d_indices=primal_2d,
                m_mu_flat=m_mu,
                g_2d=g_2d,
                conductor_node_groups=[
                    np.array([0, 1, 2], dtype=np.int64),
                    np.array([3, 4, 5], dtype=np.int64),
                ],
                m_eps_2d_vacuum=bad_vacuum,
                grid=mesh.grid,
            )

    def test_profile_overlap_with_analytical_above_0_95(self):
        """After both modes flow through ``discretize_modes`` (analytical
        → sample + Gram-Schmidt; numerical → pass-through), the
        M-orthonormal profiles overlap to within the staircase
        discretisation: ``|⟨ê_num, ê_ana⟩_M| > 0.95`` at Ny=Nz=80.

        This is the "matches CoaxAnalyticalModeSolver to discretisation
        accuracy" criterion of architecture §5 step 8.
        """
        from magnelio.ports._modal.coax import CoaxAnalyticalModeSolver

        solver, mesh, M_eps = _coax_solver_setup(Nyz=80)
        m_num = solver.solve(n_modes=1, f_calc=10e9)[0]
        m_ana = CoaxAnalyticalModeSolver(
            inner_radius=COAX_RI,
            outer_radius=COAX_RO,
            epsilon_r=1.0,
        ).solve(n_modes=1)[0]

        dm_num = discretize_modes([m_num], solver.plane, M_eps)[0]
        dm_ana = discretize_modes([m_ana], solver.plane, M_eps)[0]

        me_u = M_eps[solver.plane.e_u_indices]
        me_v = M_eps[solver.plane.e_v_indices]
        overlap = float(
            np.sum(me_u * dm_num.e_u_profile * dm_ana.e_u_profile)
            + np.sum(me_v * dm_num.e_v_profile * dm_ana.e_v_profile)
        )
        assert abs(overlap) > 0.95, (
            f"Numerical-vs-analytical Coax M-overlap = {abs(overlap):.4f}, expected > 0.95"
        )


# ---------------------------------------------------------------------
# TM-path tests (scalar node-Laplace dispatch).  These cover the
# Phase-2a step-3 follow-up that fixed the spurious-TM-mode bug:
# the curl-curl operator only resolves TE modes (TM E_t is a gradient
# field and lives in its null-space), so TM mode_type now dispatches
# to a separate node-Laplace eigsh built by ``build_2d_node_laplace``.
# ---------------------------------------------------------------------


class TestTMPathDispatch:
    """``_classify`` and validation for the new TM-eigsh dispatch."""

    def test_classify_tm_when_mode_type_TM(self):
        solver, _ = _wr90_tm_solver(Ny=8, Nz=4)
        assert solver._classify() == "tm"

    def test_tm_without_node_laplace_inputs_rejected(self):
        plane, K, M, primal_2d, m_mu = _build_KM(_wr90_mesh(Ny=8, Nz=4))
        with pytest.raises(ValueError, match="TM eigenproblem inputs"):
            Numerical2DModeSolver(
                plane=plane,
                K=K,
                M=M,
                primal_2d_indices=primal_2d,
                m_mu_flat=m_mu,
                pec_edge_mask=_wall_pec_mask(plane, a=WR90_A, b=WR90_B),
                epsilon_r=1.0,
                mode_type=ModeType.TM,
            )


class TestTMCutoffWR90:
    """TM11 in hollow WR-90 has analytical cutoff
    f_c = (c/2)·sqrt((1/a)² + (1/b)²) ≈ 16.156 GHz.  The Phase-2a
    numerical solver must reproduce it within ≤1% on a moderately
    refined mesh and converge to it under refinement."""

    F_TM11_ANALYTICAL = (C0 / 2.0) * math.sqrt((1.0 / WR90_A) ** 2 + (1.0 / WR90_B) ** 2)

    def test_tm11_cutoff_within_1_percent_at_60x30(self):
        solver, _ = _wr90_tm_solver(Ny=60, Nz=30)
        mode = solver.solve(n_modes=1, f_calc=18e9)[0]
        f_num = mode.omega_c / (2.0 * math.pi)
        rel_err = abs(f_num - self.F_TM11_ANALYTICAL) / self.F_TM11_ANALYTICAL
        assert rel_err < 0.01, (
            f"TM11 cutoff {f_num / 1e9:.4f} GHz vs analytical "
            f"{self.F_TM11_ANALYTICAL / 1e9:.4f} GHz: rel err = {rel_err:.4%}"
        )

    def test_tm11_mode_type_stamped(self):
        solver, _ = _wr90_tm_solver(Ny=30, Nz=15)
        mode = solver.solve(n_modes=1, f_calc=18e9)[0]
        assert mode.mode_type is ModeType.TM
        assert mode.name.startswith("TM_num")

    def test_tm11_convergence_h_to_h2(self):
        """Refining the mesh should bring numerical TM11 closer to
        analytical at approximately O(h²)."""
        solver_coarse, _ = _wr90_tm_solver(Ny=15, Nz=7)
        solver_fine, _ = _wr90_tm_solver(Ny=30, Nz=15)
        f_coarse = solver_coarse.solve(n_modes=1, f_calc=18e9)[0].omega_c / (2.0 * math.pi)
        f_fine = solver_fine.solve(n_modes=1, f_calc=18e9)[0].omega_c / (2.0 * math.pi)
        err_coarse = abs(f_coarse - self.F_TM11_ANALYTICAL)
        err_fine = abs(f_fine - self.F_TM11_ANALYTICAL)
        assert err_fine < err_coarse, (
            f"Refinement should reduce error: coarse={err_coarse / 1e9:.4f}, "
            f"fine={err_fine / 1e9:.4f} GHz"
        )


class TestTMModeStructure:
    """TM mode profile must be a gradient field (E_t = -j·β/k_c² · ∇E_z),
    so its 2D curl is ≈ 0 — the very property that prevents TM modes from
    living in the curl-curl operator's spectrum.  This test verifies the
    new TM path actually delivers a curl-free E_t.
    """

    def test_tm_profile_is_curl_free(self):
        solver, mesh = _wr90_tm_solver(Ny=30, Nz=15)
        mode = solver.solve(n_modes=1, f_calc=18e9)[0]
        # Reconstruct full 2D edge vector
        e_t = np.concatenate(
            [
                mode.discrete_e_u_profile,
                mode.discrete_e_v_profile,
            ]
        )
        # Build the 2D curl operator (same as the K factor in the TE problem).
        # The curl of a gradient field is zero in continuous theory; on
        # the discrete FIT mesh the residual reflects only floating-point
        # round-off (G_2d^T C_2d^T = 0 by construction of the de Rham
        # complex).
        from magnelio.ports._modal.curl_curl_2d import _normal_h_indices

        h_n = _normal_h_indices(solver.plane.face, mesh.grid)
        C_3d = build_curl_matrix(mesh.grid)
        C_2d = C_3d[h_n, :][:, solver.primal_2d_indices].tocsr()
        curl_e_t = C_2d @ e_t
        rel_curl = float(np.linalg.norm(curl_e_t)) / max(
            float(np.linalg.norm(e_t)),
            1e-30,
        )
        assert rel_curl < 1e-10, f"TM E_t should be curl-free; relative curl norm = {rel_curl:.3e}"

    def test_tm_profile_M_eps_orthonormal(self):
        solver, _ = _wr90_tm_solver(Ny=30, Nz=15)
        mode = solver.solve(n_modes=1, f_calc=18e9)[0]
        m_eps_diag = solver.M.diagonal()
        e_t = np.concatenate(
            [
                mode.discrete_e_u_profile,
                mode.discrete_e_v_profile,
            ]
        )
        norm_sq = float(np.sum(e_t * m_eps_diag * e_t))
        assert abs(norm_sq - 1.0) < 1e-10, (
            f"TM mode is not M_eps-orthonormal: e^T M e = {norm_sq:.6f}"
        )

    def test_tm_h_profile_travelling_wave_form(self):
        """``h = ±e_co·(μ₀·dz/Z_TM)/M_μ[face]`` at f_calc (WP7.2)."""
        from magnelio.ports._modal.mode import MU0

        solver, _ = _wr90_tm_solver(Ny=30, Nz=15)
        f_calc = 18e9
        mode = solver.solve(n_modes=1, f_calc=f_calc)[0]
        z = mode.z_wave(2.0 * math.pi * f_calc)
        z_real = abs(z.real) + abs(z.imag)
        plane = solver.plane
        m_mu = solver.m_mu_flat
        scale = MU0 * plane.normal_dx / z_real
        np.testing.assert_allclose(
            mode.discrete_h_u_profile,
            -mode.discrete_e_v_profile * (scale / m_mu[plane.h_u_indices]),
            rtol=1e-12,
            atol=0,
        )
        np.testing.assert_allclose(
            mode.discrete_h_v_profile,
            +mode.discrete_e_u_profile * (scale / m_mu[plane.h_v_indices]),
            rtol=1e-12,
            atol=0,
        )


class TestSignFlipPortSymmetry:
    """The sign convention should be invariant under port-swap so that
    reciprocity holds for symmetric modes (TE20 was the historical
    failure case — its two equal-magnitude antiphase peaks defeated the
    legacy ``argmax(|v|)`` flip).  The fix is a port-symmetric criterion
    that overlaps the eigenvector with test functions of global
    tangential coordinates; this test exercises that criterion directly.
    """

    def test_resolve_sign_is_deterministic_under_input_negation(self):
        """The same eigenvector and its negation must produce opposite
        flips so the resolved sign is the same in both cases — the
        fundamental property of any well-defined sign convention.
        """
        solver = _wr90_solver()
        modes = solver.solve(n_modes=2, f_calc=14e9)
        e_te20 = np.concatenate(
            [
                modes[1].discrete_e_u_profile,
                modes[1].discrete_e_v_profile,
            ]
        )
        flipped = solver._resolve_sign(-e_te20.copy())
        np.testing.assert_allclose(flipped, e_te20, rtol=0, atol=1e-30)

    def test_te20_sign_consistent_at_X_MIN_and_X_MAX(self):
        """TE20 has antiphase symmetry sin(2πu/a) — the legacy
        ``argmax(|v|)`` flip picked between equal-magnitude peaks via
        numerical noise, giving inconsistent signs at X_MIN vs X_MAX
        and breaking reciprocity at +6 dB on the diagonal.  The
        global-tangential-coordinate convention must give the same
        sign at both faces for the same physical mode.
        """
        solver_min = _wr90_solver(face=BoxFace.X_MIN)
        solver_max = _wr90_solver(face=BoxFace.X_MAX)
        modes_min = solver_min.solve(n_modes=2, f_calc=14e9)
        modes_max = solver_max.solve(n_modes=2, f_calc=14e9)
        # TE20 is mode index 1 at both faces (TE10 is index 0).
        # X_MIN: u=y, v=z → TE20's E_z lives in e_v_profile (along y).
        # X_MAX: u=z, v=y → TE20's E_z lives in e_u_profile (along y).
        # Pick the (y, z)-coordinate of each profile's first global-y
        # half-period (y < a/2): the dominant component should have
        # the same sign at the same physical (y, z) on both ports.
        plane_min = solver_min.plane
        plane_max = solver_max.plane
        # X_MIN e_v lives on v-edges at (u_local=y, v_local=z).
        # X_MAX e_u lives on u-edges at (u_local=z, v_local=y).
        y_min = plane_min.v_edge_uv[:, 0]  # global y from u_local
        y_max = plane_max.u_edge_uv[:, 1]  # global y from v_local
        # Find an edge in each profile with y near a/4 (positive-peak
        # region of sin(2πy/a)) and at the cross-section centre line.
        z_min = plane_min.v_edge_uv[:, 1]
        z_max = plane_max.u_edge_uv[:, 0]
        target_y = WR90_A / 4.0
        target_z = WR90_B / 2.0
        idx_min = int(np.argmin((y_min - target_y) ** 2 + (z_min - target_z) ** 2))
        idx_max = int(np.argmin((y_max - target_y) ** 2 + (z_max - target_z) ** 2))
        sign_min = np.sign(modes_min[1].discrete_e_v_profile[idx_min])
        sign_max = np.sign(modes_max[1].discrete_e_u_profile[idx_max])
        assert sign_min == sign_max != 0, (
            f"TE20 sign mismatch at (y≈a/4, z≈b/2): "
            f"X_MIN={sign_min}, X_MAX={sign_max}.  Port-symmetric sign "
            f"convention failed; reciprocity will not hold."
        )
