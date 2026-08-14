"""Unit tests for the matrix-free CurlCurlOperator.

Verifies that CurlCurlOperator.matvec(x) produces exactly the same
result as the explicit sparse matrix A = C^T diag(M_mu_inv) C.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from magnelio._operators.curl import build_curl_matrix
from magnelio._operators.curl_curl_operator import CurlCurlOperator
from magnelio._operators.material_matrices import build_M_mu
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.solver._eigenmode_3d import _build_pec_dof_mask


def _make_cavity(Nx, Ny, Nz, a=30e-3, b=20e-3, c=15e-3):
    """Build a simple PEC cavity mesh."""
    grid = GridLines(
        x=np.linspace(0, a, Nx + 1),
        y=np.linspace(0, b, Ny + 1),
        z=np.linspace(0, c, Nz + 1),
    )
    return Mesh.from_grid(grid)


class TestCurlCurlOperator:
    """Test matrix-free operator against explicit sparse matrix."""

    @pytest.mark.parametrize(
        "Nx,Ny,Nz",
        [
            (4, 3, 2),
            (6, 5, 4),
            (10, 8, 6),
        ],
    )
    def test_matvec_matches_explicit(self, Nx, Ny, Nz):
        mesh = _make_cavity(Nx, Ny, Nz)
        grid = mesh.grid

        # Build explicit sparse matrix A = C^T M_mu_inv C
        C = build_curl_matrix(grid)
        M_mu = build_M_mu(mesh)
        M_mu_inv = 1.0 / np.where(M_mu > 0, M_mu, 1.0)
        A = C.T @ sp.diags(M_mu_inv) @ C

        # PEC mask and reduced system
        bcs = {}  # all-PEC
        pec_mask = _build_pec_dof_mask(grid, bcs)
        free_idx = np.where(~pec_mask)[0]
        A_f = A[np.ix_(free_idx, free_idx)].toarray()

        # Matrix-free operator
        op = CurlCurlOperator(Nx, Ny, Nz, M_mu_inv, pec_mask)

        # Test with random vector
        rng = np.random.default_rng(42)
        x = rng.standard_normal(len(free_idx))

        y_explicit = A_f @ x
        y_stencil = op.matvec(x)

        np.testing.assert_allclose(
            y_stencil,
            y_explicit,
            rtol=1e-12,
            atol=1e-15,
            err_msg=f"CurlCurlOperator.matvec mismatch at grid {Nx}x{Ny}x{Nz}",
        )

    def test_linear_operator_interface(self):
        Nx, Ny, Nz = 4, 3, 2
        mesh = _make_cavity(Nx, Ny, Nz)
        grid = mesh.grid

        M_mu = build_M_mu(mesh)
        M_mu_inv = 1.0 / np.where(M_mu > 0, M_mu, 1.0)
        pec_mask = _build_pec_dof_mask(grid, {})

        op = CurlCurlOperator(Nx, Ny, Nz, M_mu_inv, pec_mask)
        A_op = op.as_linear_operator()

        assert A_op.shape == (op.n_free, op.n_free)

        rng = np.random.default_rng(123)
        x = rng.standard_normal(op.n_free)

        y1 = op.matvec(x)
        y2 = A_op @ x

        np.testing.assert_allclose(y1, y2, rtol=1e-14)

    def test_symmetry(self):
        """A should be symmetric: x^T A y == y^T A x."""
        Nx, Ny, Nz = 6, 5, 4
        mesh = _make_cavity(Nx, Ny, Nz)
        grid = mesh.grid

        M_mu = build_M_mu(mesh)
        M_mu_inv = 1.0 / np.where(M_mu > 0, M_mu, 1.0)
        pec_mask = _build_pec_dof_mask(grid, {})

        op = CurlCurlOperator(Nx, Ny, Nz, M_mu_inv, pec_mask)

        rng = np.random.default_rng(7)
        x = rng.standard_normal(op.n_free)
        y = rng.standard_normal(op.n_free)

        Ax = op.matvec(x)
        Ay = op.matvec(y)

        lhs = float(x @ Ay)
        rhs = float(y @ Ax)

        np.testing.assert_allclose(lhs, rhs, rtol=1e-12, err_msg="CurlCurlOperator not symmetric")

    def test_positive_semidefinite(self):
        """x^T A x >= 0 for any x (PSD after PEC elimination)."""
        Nx, Ny, Nz = 6, 5, 4
        mesh = _make_cavity(Nx, Ny, Nz)
        grid = mesh.grid

        M_mu = build_M_mu(mesh)
        M_mu_inv = 1.0 / np.where(M_mu > 0, M_mu, 1.0)
        pec_mask = _build_pec_dof_mask(grid, {})

        op = CurlCurlOperator(Nx, Ny, Nz, M_mu_inv, pec_mask)

        rng = np.random.default_rng(99)
        for _ in range(5):
            x = rng.standard_normal(op.n_free)
            Ax = op.matvec(x)
            assert float(x @ Ax) >= -1e-14, "Operator should be PSD"
