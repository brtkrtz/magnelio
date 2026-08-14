"""Tests for magnelio.ports._modal.curl_curl_2d.

Phase 2a, step 1 of the numerical-mode-solver work order
(`reference_architecture_phase2_mode_solver.md` §3.2 / §5).

Three layers of validation:

1. **Index helper.**  The normal-H index helper produces the correct
   flat-H indices on each of the six bbox faces, with the right counts
   and the correct H-component block offset.
2. **Slicing consistency.**  On a small mesh, ``C_2D`` applied to a
   randomly-filled 2D edge vector reproduces ``C_3d`` applied to the
   same vector embedded in the global E-state (read out at the same
   normal-H indices).  This is the byte-for-byte sanity check that the
   row/column slicing is consistent with the 3D operator.
3. **Eigenvalues against analytical TE/TM cutoffs.**  For a hollow
   WR-90 cross-section with PEC walls (applied as Dirichlet by removing
   the wall-tangent edges), the lowest non-trivial eigenvalues of
   ``K · ê = λ · M · ê`` match the analytical ``ω_c²`` of TE_10, TE_20,
   TE_01 to ≤ 1 % at typical resolution.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse.linalg import eigsh

from magnelio._operators.curl import build_curl_matrix
from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.ports._modal.curl_curl_2d import (
    _normal_h_indices,
    build_2d_curl_curl,
)
from magnelio.ports._modal.port_plane import BoxFace, PortPlane

C0 = 2.99792458e8
WR90_A = 22.86e-3  # broad WR-90 dimension (m)
WR90_B = 10.16e-3  # narrow WR-90 dimension (m)


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
    return K, M, primal_2d, plane


def _wall_pec_mask(plane: PortPlane, *, a: float, b: float) -> np.ndarray:
    """Boolean mask over (e_u, e_v) — True for edges tangent to a PEC wall.

    For a hollow WG with PEC walls at u ∈ {0, a} and v ∈ {0, b}, the
    tangential E-edges are:

    - ``E_u`` edges whose v-coordinate (the *node* coordinate, see
      ``PortPlane`` docstring) lies on v = 0 or v = b.
    - ``E_v`` edges whose u-coordinate (also a node coordinate) lies on
      u = 0 or u = a.
    """
    eps = 1e-9
    # u-edges: midpoints stored as (u_centre, v_node); wall test on v_node.
    u_v = plane.u_edge_uv[:, 1]
    u_pec = (np.abs(u_v) < eps) | (np.abs(u_v - b) < eps)
    # v-edges: midpoints stored as (u_node, v_centre); wall test on u_node.
    v_u = plane.v_edge_uv[:, 0]
    v_pec = (np.abs(v_u) < eps) | (np.abs(v_u - a) < eps)
    return np.concatenate([u_pec, v_pec])


# ---------------------------------------------------------------------
# 1) Normal-H index helper
# ---------------------------------------------------------------------


class TestNormalHIndices:
    """`_normal_h_indices` — correct counts and offsets per face."""

    @pytest.fixture
    def grid(self):
        return GridLines(
            x=np.linspace(0.0, 1.0, 5),  # Nx = 4
            y=np.linspace(0.0, 1.0, 4),  # Ny = 3
            z=np.linspace(0.0, 1.0, 3),  # Nz = 2
        )

    def _component_offsets(self, grid):
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        n_Hx = (Nx + 1) * Ny * Nz
        n_Hy = Nx * (Ny + 1) * Nz
        n_Hz = Nx * Ny * (Nz + 1)
        return n_Hx, n_Hy, n_Hz

    def test_xmin_count_and_offset(self, grid):
        Ny, Nz = grid.Ny, grid.Nz
        idx = _normal_h_indices(BoxFace.X_MIN, grid)
        assert idx.shape == (Ny * Nz,)
        # X_MIN: Hx[0, j, k] for j ∈ [0, Ny), k ∈ [0, Nz) — all in the
        # Hx block (offset 0..n_Hx).
        n_Hx, _, _ = self._component_offsets(grid)
        assert (idx >= 0).all() and (idx < n_Hx).all()

    def test_xmax_count_and_offset(self, grid):
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        idx = _normal_h_indices(BoxFace.X_MAX, grid)
        assert idx.shape == (Ny * Nz,)
        n_Hx, _, _ = self._component_offsets(grid)
        # X_MAX: Hx[Nx, j, k] — still in Hx block, but *upper* slab.
        assert (idx >= Nx * Ny * Nz).all() and (idx < n_Hx).all()

    def test_ymin_in_hy_block(self, grid):
        Nx, Nz = grid.Nx, grid.Nz
        idx = _normal_h_indices(BoxFace.Y_MIN, grid)
        assert idx.shape == (Nx * Nz,)
        n_Hx, n_Hy, _ = self._component_offsets(grid)
        assert (idx >= n_Hx).all() and (idx < n_Hx + n_Hy).all()

    def test_zmax_in_hz_block(self, grid):
        Nx, Ny = grid.Nx, grid.Ny
        idx = _normal_h_indices(BoxFace.Z_MAX, grid)
        assert idx.shape == (Nx * Ny,)
        n_Hx, n_Hy, n_Hz = self._component_offsets(grid)
        assert (idx >= n_Hx + n_Hy).all()
        assert (idx < n_Hx + n_Hy + n_Hz).all()

    def test_indices_unique(self, grid):
        for face in BoxFace:
            idx = _normal_h_indices(face, grid)
            assert np.unique(idx).size == idx.size, f"{face} has duplicates"


# ---------------------------------------------------------------------
# 2) Slicing consistency vs. the 3D primal-curl
# ---------------------------------------------------------------------


class TestSliceConsistency:
    """``C_2D · ê`` matches ``C_3d · e_global`` at the normal-H indices.

    On a tiny mesh, embed a random 2D edge vector into the global E-state
    (with all other components zero), apply the 3D curl, and read out the
    H-values at the normal-H indices.  This must equal the result of
    applying ``C_2D = C_3d[h_n, primal_2d]`` to the same 2D vector.
    """

    @pytest.fixture
    def setup(self):
        mesh = _wr90_mesh(Ny=8, Nz=4, Nx=3)
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        C = build_curl_matrix(mesh.grid)
        h_n = _normal_h_indices(BoxFace.X_MIN, mesh.grid)
        primal_2d = np.concatenate([plane.e_u_indices, plane.e_v_indices])
        return mesh, plane, C, h_n, primal_2d

    def test_curl_slice_matches_3d_application(self, setup):
        mesh, plane, C, h_n, primal_2d = setup
        # Build C_2D by slicing.
        C_2d = C[h_n, :][:, primal_2d].tocsr()

        # Random 2D vector (positive seed for determinism).
        rng = np.random.default_rng(42)
        e_2d = rng.standard_normal(primal_2d.size)

        # Embed into global E-state with everything else zero.
        n_E = C.shape[1]
        e_full = np.zeros(n_E)
        e_full[primal_2d] = e_2d

        # Apply 3D curl, read out at normal-H indices.
        h_full = C @ e_full
        ref = h_full[h_n]

        # Apply 2D curl directly.
        got = C_2d @ e_2d

        np.testing.assert_allclose(got, ref, atol=0, rtol=0)

    def test_C2d_row_count_equals_normal_h(self, setup):
        mesh, plane, C, h_n, primal_2d = setup
        C_2d = C[h_n, :][:, primal_2d].tocsr()
        assert C_2d.shape == (h_n.size, primal_2d.size)

    def test_C2d_each_row_has_at_most_four_nonzeros(self, setup):
        """The 2D primal curl on a Yee grid has 4 entries per H-face row.

        ``(curl E)_x = dE_z/dy − dE_y/dz`` reads exactly two ``E_z``
        edges and two ``E_y`` edges — four nonzeros per row.  If any
        boundary H-face has fewer than four neighbours, the count drops
        but never exceeds four.
        """
        mesh, plane, C, h_n, primal_2d = setup
        C_2d = C[h_n, :][:, primal_2d].tocsr()
        per_row_nnz = np.diff(C_2d.indptr)
        assert per_row_nnz.max() <= 4
        # On a hollow rectangular grid the *interior* rows are exactly 4.
        assert per_row_nnz.max() == 4


# ---------------------------------------------------------------------
# 3) K, M structural properties
# ---------------------------------------------------------------------


class TestStructuralProperties:
    """K is symmetric; M is diagonal; K is positive semi-definite."""

    def test_K_symmetric(self):
        mesh = _wr90_mesh(Ny=20, Nz=10)
        K, _, _, _ = _build_KM(mesh)
        diff = K - K.T
        # ||K - K.T||_max — sparse-friendly check.
        assert abs(diff).max() < 1e-12 * abs(K).max()

    def test_M_is_diagonal(self):
        mesh = _wr90_mesh(Ny=20, Nz=10)
        _, M, primal_2d, _ = _build_KM(mesh)
        # sp.diags(...) returns a DIA-shape; converted to CSR it has
        # nnz == n.
        assert M.shape == (primal_2d.size, primal_2d.size)
        # Off-diagonal entries are zero.
        Md = M.toarray()
        assert np.allclose(Md - np.diag(np.diag(Md)), 0.0)
        # Diagonal entries are positive (M_ε > 0 for vacuum).
        assert (np.diag(Md) > 0).all()

    def test_K_sparsity_bounded_by_2D_stencil(self):
        """``K = C_2D^T diag(...) C_2D`` has ≤ 9 entries per row.

        Each primal 2D edge is touched by at most two adjacent normal-H
        faces (one on each side along the perpendicular tangent), and
        each of those faces contributes a row of ``C_2D`` with up to 4
        edges.  After accounting for the shared edge, the union has
        ≤ 9 unique columns per row.
        """
        mesh = _wr90_mesh(Ny=20, Nz=10)
        K, _, primal_2d, _ = _build_KM(mesh)
        per_row_nnz = np.diff(K.tocsr().indptr)
        assert per_row_nnz.max() <= 9
        # Average is far below 9 because of boundary thinning.
        assert K.nnz < 10 * primal_2d.size

    def test_K_positive_semidefinite_by_construction(self):
        """``K = X^T diag(D) X`` with ``D > 0`` entrywise ⇒ PSD.

        ``D = M_μ⁻¹`` is the inverse of the magnetic-mass diagonal at
        the normal-H faces, which is strictly positive (vacuum here).
        Sample-vector ``x^T K x ≥ 0`` confirms the property numerically
        without relying on ARPACK's behaviour on a singular operator
        (which is brittle for the gradient null-space).
        """
        mesh = _wr90_mesh(Ny=12, Nz=6)
        K, _, _, _ = _build_KM(mesh)
        from magnelio._operators.material_matrices import build_M_mu

        M_mu = build_M_mu(mesh)
        h_n = _normal_h_indices(BoxFace.X_MIN, mesh.grid)
        assert (M_mu[h_n] > 0).all()
        rng = np.random.default_rng(0)
        scale_K = abs(K).max()
        for _ in range(10):
            x = rng.standard_normal(K.shape[0])
            quad = float(x @ (K @ x))
            assert quad > -1e-8 * (x @ x) * scale_K


# ---------------------------------------------------------------------
# 4) Eigenvalues vs. analytical WR-90 TE_mn cutoffs
# ---------------------------------------------------------------------


class TestEigenvaluesWR90:
    """Lowest non-trivial eigenvalues match analytical TE_mn cutoffs.

    For hollow WR-90 (a × b) with PEC lateral walls (applied via
    edge-removal), the 2D curl-curl spectrum has a large gradient
    null-space at λ = 0, and physical eigenvalues at
    ``λ = ω_c²(TE_mn)``.  Using shift-invert at a positive ``sigma``
    just below TE_10 cleanly separates the physical modes from the
    null-space.
    """

    @staticmethod
    def _solve(mesh: Mesh, n_modes: int = 6, sigma_factor: float = 0.7) -> np.ndarray:
        """Return the lowest ``n_modes`` non-trivial eigenvalues, sorted."""
        K, M, primal_2d, plane = _build_KM(mesh, face=BoxFace.X_MIN)
        pec_mask = _wall_pec_mask(plane, a=WR90_A, b=WR90_B)
        free = np.where(~pec_mask)[0]
        K_f = K.tocsr()[free, :][:, free]
        M_f = M.tocsr()[free, :][:, free]

        # Sigma between the null space and TE_10 — closest physical
        # eigenvalues are the real modes.
        analytical_te10_sq = (C0 * np.pi / WR90_A) ** 2
        sigma = sigma_factor * analytical_te10_sq

        omega2, _ = eigsh(K_f, k=n_modes, M=M_f, sigma=sigma)
        return np.sort(omega2)

    def test_te10_within_one_percent(self):
        mesh = _wr90_mesh(Ny=60, Nz=30)
        eigs = self._solve(mesh, n_modes=4)
        physical = eigs[eigs > 1e15]
        analytical = (C0 * np.pi / WR90_A) ** 2
        rel_err = abs(physical[0] - analytical) / analytical
        assert rel_err < 0.01, (
            f"TE_10 ω_c² off by {rel_err:.4%}: got {physical[0]:.4e}, analytical {analytical:.4e}"
        )

    def test_te10_cutoff_frequency(self):
        """f_c(TE_10) = c / (2a) ≈ 6.557 GHz at WR-90."""
        mesh = _wr90_mesh(Ny=60, Nz=30)
        eigs = self._solve(mesh, n_modes=4)
        physical = eigs[eigs > 1e15]
        f_c = np.sqrt(physical[0]) / (2 * np.pi)
        f_c_analytical = C0 / (2 * WR90_A)
        assert abs(f_c - f_c_analytical) / f_c_analytical < 0.01

    def test_te20_within_one_percent(self):
        """TE_20: ω_c² = (2π·c/a)² = 4 × TE_10² ≈ 6.79e21 (rad/s)²."""
        mesh = _wr90_mesh(Ny=60, Nz=30)
        # TE_20 is the second physical eigenvalue; bracket the search
        # at sigma above TE_10 to land near TE_20.
        K, M, primal_2d, plane = _build_KM(mesh, face=BoxFace.X_MIN)
        pec_mask = _wall_pec_mask(plane, a=WR90_A, b=WR90_B)
        free = np.where(~pec_mask)[0]
        K_f = K.tocsr()[free, :][:, free]
        M_f = M.tocsr()[free, :][:, free]
        analytical_te20 = (2 * C0 * np.pi / WR90_A) ** 2
        eigs, _ = eigsh(K_f, k=4, M=M_f, sigma=0.95 * analytical_te20)
        eigs = np.sort(eigs)
        # The closest eigenvalue to sigma = 0.95·TE_20 is TE_20 itself
        # (TE_10 is far below, TE_01 is above).
        idx = np.argmin(np.abs(eigs - analytical_te20))
        rel_err = abs(eigs[idx] - analytical_te20) / analytical_te20
        assert rel_err < 0.01, f"TE_20 ω_c² off by {rel_err:.4%}"

    def test_te01_within_two_percent(self):
        """TE_01: ω_c² = (π·c/b)² ≈ 8.61e21 (rad/s)².

        Two-percent tolerance because Nz = 30 across b = 10.16 mm gives
        a coarser TE_01 (which varies along v=z) than TE_10.
        """
        mesh = _wr90_mesh(Ny=60, Nz=30)
        K, M, primal_2d, plane = _build_KM(mesh, face=BoxFace.X_MIN)
        pec_mask = _wall_pec_mask(plane, a=WR90_A, b=WR90_B)
        free = np.where(~pec_mask)[0]
        K_f = K.tocsr()[free, :][:, free]
        M_f = M.tocsr()[free, :][:, free]
        analytical = (C0 * np.pi / WR90_B) ** 2
        eigs, _ = eigsh(K_f, k=4, M=M_f, sigma=0.95 * analytical)
        eigs = np.sort(eigs)
        idx = np.argmin(np.abs(eigs - analytical))
        rel_err = abs(eigs[idx] - analytical) / analytical
        assert rel_err < 0.02
