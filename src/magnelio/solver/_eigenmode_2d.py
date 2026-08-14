"""
2D Eigenmode Solver.

Solves the 2D curl-curl eigenvalue problem on a rectangular cross-section
(typically a port face) to find guided modes of a waveguide.

Uses scipy.sparse.linalg.eigsh (ARPACK) — see design-decisions.md DD-007.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


@dataclass
class EigenmodeSolver2D:
    """2D eigenmode solver for waveguide cross-sections.

    Args:
        n_modes: Number of modes to compute.
        sigma:   ARPACK shift (eigenvalue near which to find modes).
                 Use a small positive value to avoid spurious zero modes.
    """

    n_modes: int = 5
    sigma: float = 1e-10

    def solve(
        self,
        A: sp.spmatrix,
        B: sp.spmatrix | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve the generalized eigenvalue problem A·x = λ·B·x.

        Args:
            A: Curl-curl operator matrix, shape (N, N).
            B: Mass matrix (positive definite). If None, uses identity (standard EVP).

        Returns:
            ``(omega, modes)`` where:
            - ``omega``: Array of angular frequencies (sqrt of eigenvalues), shape (k,).
            - ``modes``: Eigenvector matrix, shape (N, k). Column i is mode i.
        """
        k = min(self.n_modes, A.shape[0] - 2)

        if B is None:
            eigenvalues, eigenvectors = eigsh(A, k=k, which="SM", sigma=self.sigma)
        else:
            eigenvalues, eigenvectors = eigsh(A, M=B, k=k, which="SM", sigma=self.sigma)

        # Sort ascending
        order = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # Physical eigenvalues are non-negative; clip numerical noise
        eigenvalues = np.maximum(eigenvalues, 0.0)
        omega = np.sqrt(eigenvalues.real)

        return omega, eigenvectors
