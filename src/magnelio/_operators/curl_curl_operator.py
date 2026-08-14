"""Matrix-free curl-curl operator A = C^T diag(M_mu_inv) C.

Computes the action of the curl-curl operator on a vector via stencil
operations, without assembling the explicit sparse matrix.  This enables
LOBPCG eigensolves for grids with millions of cells where the sparse
matrix assembly and LU factorisation become infeasible.

Works on both NumPy (CPU) and CuPy (GPU) arrays — the array module
is detected from the input or passed explicitly.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator

from magnelio._operators.curl import curl_e_stencil, curl_h_stencil


def _component_sizes(Nx, Ny, Nz):
    """Return (n_Ex, n_Ey, n_Ez, n_Hx, n_Hy, n_Hz) for a given grid."""
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    n_Ez = (Nx + 1) * (Ny + 1) * Nz
    n_Hx = (Nx + 1) * Ny * Nz
    n_Hy = Nx * (Ny + 1) * Nz
    n_Hz = Nx * Ny * (Nz + 1)
    return n_Ex, n_Ey, n_Ez, n_Hx, n_Hy, n_Hz


class CurlCurlOperator:
    """Matrix-free curl-curl operator for the FIT eigenvalue problem.

    Computes ``A @ x = C^T diag(M_mu_inv) C @ x`` using stencil operations
    from :mod:`magnelio._operators.curl`.  Operates in the reduced (free-DOF)
    subspace where PEC-constrained DOFs are eliminated.

    Parameters
    ----------
    Nx, Ny, Nz : int
        Grid cell counts.
    M_mu_inv : np.ndarray
        Inverse of the magnetic mass matrix diagonal, shape ``(n_H,)``.
        Ordering: ``[Hx | Hy | Hz]``.
    pec_mask : np.ndarray
        Boolean mask of PEC-constrained E-DOFs, shape ``(n_E,)``.
    xp : module
        Array module (numpy or cupy).
    """

    def __init__(self, Nx, Ny, Nz, M_mu_inv, pec_mask, xp=None):
        if xp is None:
            xp = np
        self._xp = xp
        self._Nx, self._Ny, self._Nz = Nx, Ny, Nz

        sizes = _component_sizes(Nx, Ny, Nz)
        n_Ex, n_Ey, n_Ez, n_Hx, n_Hy, n_Hz = sizes
        self._n_Ex, self._n_Ey, self._n_Ez = n_Ex, n_Ey, n_Ez
        self._n_E = n_Ex + n_Ey + n_Ez

        # PEC DOF handling
        self._pec_mask = xp.asarray(pec_mask)
        self._free_idx = xp.where(~self._pec_mask)[0]
        self._n_free = int(self._free_idx.shape[0])

        # Split M_mu_inv into per-component 3D views
        M_mu_inv = xp.asarray(M_mu_inv)
        self._mu_inv_x = M_mu_inv[:n_Hx].reshape(Nx + 1, Ny, Nz)
        self._mu_inv_y = M_mu_inv[n_Hx : n_Hx + n_Hy].reshape(Nx, Ny + 1, Nz)
        self._mu_inv_z = M_mu_inv[n_Hx + n_Hy :].reshape(Nx, Ny, Nz + 1)

        # Pre-allocate workspace buffers
        self._Ex = xp.zeros((Nx, Ny + 1, Nz + 1))
        self._Ey = xp.zeros((Nx + 1, Ny, Nz + 1))
        self._Ez = xp.zeros((Nx + 1, Ny + 1, Nz))
        self._Hx = xp.zeros((Nx + 1, Ny, Nz))
        self._Hy = xp.zeros((Nx, Ny + 1, Nz))
        self._Hz = xp.zeros((Nx, Ny, Nz + 1))
        self._oEx = xp.zeros((Nx, Ny + 1, Nz + 1))
        self._oEy = xp.zeros((Nx + 1, Ny, Nz + 1))
        self._oEz = xp.zeros((Nx + 1, Ny + 1, Nz))
        self._e_full = xp.zeros(self._n_E)

    def matvec(self, x_free):
        """Compute A @ x in the free-DOF subspace.

        Parameters
        ----------
        x_free : ndarray, shape (n_free,)
            Input vector (free DOFs only).

        Returns
        -------
        ndarray, shape (n_free,)
            Result of curl-curl operator application.
        """
        xp = self._xp
        Nx, Ny, Nz = self._Nx, self._Ny, self._Nz
        n_Ex, n_Ey = self._n_Ex, self._n_Ey

        x_free = xp.asarray(x_free).ravel()

        # 1. Expand to full E-DOF space (PEC DOFs = 0)
        e = self._e_full
        e[:] = 0.0
        e[self._free_idx] = x_free

        # 2. Reshape to 3D component views
        Ex = e[:n_Ex].reshape(Nx, Ny + 1, Nz + 1)
        Ey = e[n_Ex : n_Ex + n_Ey].reshape(Nx + 1, Ny, Nz + 1)
        Ez = e[n_Ex + n_Ey :].reshape(Nx + 1, Ny + 1, Nz)

        # 3. C @ e: primal curl (E → H)
        Hx, Hy, Hz = self._Hx, self._Hy, self._Hz
        curl_e_stencil(Ex, Ey, Ez, Hx, Hy, Hz)

        # 4. M_mu_inv @ (C @ e): element-wise multiply
        Hx *= self._mu_inv_x
        Hy *= self._mu_inv_y
        Hz *= self._mu_inv_z

        # 5. C^T @ (M_mu_inv @ C @ e): dual curl (H → E)
        oEx, oEy, oEz = self._oEx, self._oEy, self._oEz
        curl_h_stencil(Hx, Hy, Hz, oEx, oEy, oEz)

        # 6. Flatten and extract free DOFs
        result = xp.concatenate([oEx.ravel(), oEy.ravel(), oEz.ravel()])
        return result[self._free_idx]

    def as_linear_operator(self):
        """Return a scipy LinearOperator wrapping :meth:`matvec`."""
        n = self._n_free
        return LinearOperator(
            shape=(n, n),
            matvec=self.matvec,
            dtype=float,
        )

    @property
    def n_free(self):
        """Number of free (non-PEC) DOFs."""
        return self._n_free
