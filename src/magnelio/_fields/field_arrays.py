"""
FieldState: Structure-of-Arrays storage for Yee-staggered E and H fields.

E fields are on primary grid edges; H fields are on dual grid face centers.
See spec.md for field layout and design-decisions.md DD-002.

Yee grid staggering (grid size Nx × Ny × Nz cells):
    Ex: shape (Nx,   Ny+1, Nz+1)
    Ey: shape (Nx+1, Ny,   Nz+1)
    Ez: shape (Nx+1, Ny+1, Nz  )
    Hx: shape (Nx+1, Ny,   Nz  )
    Hy: shape (Nx,   Ny+1, Nz  )
    Hz: shape (Nx,   Ny,   Nz+1)

Internal storage uses two contiguous flat arrays (_e, _h).  The component
properties (Ex, Ey, …) return reshaped **views** — zero-copy, so all
slice writes (``fields.Ex[:, :, 0] = 0.0``) modify _e in place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

try:
    import cupy as _cp
except ImportError:
    _cp = None


def _array_module(arr):
    """Return numpy or cupy depending on array type."""
    if _cp is not None and isinstance(arr, _cp.ndarray):
        return _cp
    return np


class FieldState:
    """Yee-staggered electromagnetic field arrays with flat backing store.

    All component attributes (Ex, Ey, …) are views into the contiguous
    flat arrays ``e_flat`` and ``h_flat``.  This avoids concatenate / ravel
    overhead in the time-stepping loop.
    """

    __slots__ = (
        "_e",
        "_h",
        "_xp",
        "_n_Ex",
        "_n_Ey",
        "_n_Ez",
        "_n_Hx",
        "_n_Hy",
        "_n_Hz",
        "_shape_Ex",
        "_shape_Ey",
        "_shape_Ez",
        "_shape_Hx",
        "_shape_Hy",
        "_shape_Hz",
    )

    def __init__(
        self,
        Ex: np.ndarray,
        Ey: np.ndarray,
        Ez: np.ndarray,
        Hx: np.ndarray,
        Hy: np.ndarray,
        Hz: np.ndarray,
    ) -> None:
        self._shape_Ex = Ex.shape
        self._shape_Ey = Ey.shape
        self._shape_Ez = Ez.shape
        self._shape_Hx = Hx.shape
        self._shape_Hy = Hy.shape
        self._shape_Hz = Hz.shape

        self._n_Ex = Ex.size
        self._n_Ey = Ey.size
        self._n_Ez = Ez.size
        self._n_Hx = Hx.size
        self._n_Hy = Hy.size
        self._n_Hz = Hz.size

        # Build contiguous flat storage on same device as inputs.  The flat
        # store follows the component dtype, so FieldState is precision-
        # transparent: float32 components -> float32 backing store (plan WP1).
        xp = _array_module(Ex)
        self._xp = xp
        dtype = Ex.dtype

        self._e = xp.empty(self._n_Ex + self._n_Ey + self._n_Ez, dtype=dtype)
        self._e[: self._n_Ex] = Ex.ravel()
        self._e[self._n_Ex : self._n_Ex + self._n_Ey] = Ey.ravel()
        self._e[self._n_Ex + self._n_Ey :] = Ez.ravel()

        self._h = xp.empty(self._n_Hx + self._n_Hy + self._n_Hz, dtype=dtype)
        self._h[: self._n_Hx] = Hx.ravel()
        self._h[self._n_Hx : self._n_Hx + self._n_Hy] = Hy.ravel()
        self._h[self._n_Hx + self._n_Hy :] = Hz.ravel()

    # -- Flat access (zero-copy) -------------------------------------------

    @property
    def e_flat(self) -> np.ndarray:
        """Contiguous 1-D array [Ex | Ey | Ez]."""
        return self._e

    @property
    def h_flat(self) -> np.ndarray:
        """Contiguous 1-D array [Hx | Hy | Hz]."""
        return self._h

    # -- Component views (zero-copy reshape) --------------------------------

    @property
    def Ex(self) -> np.ndarray:
        return self._e[: self._n_Ex].reshape(self._shape_Ex)

    @property
    def Ey(self) -> np.ndarray:
        return self._e[self._n_Ex : self._n_Ex + self._n_Ey].reshape(self._shape_Ey)

    @property
    def Ez(self) -> np.ndarray:
        return self._e[self._n_Ex + self._n_Ey :].reshape(self._shape_Ez)

    @property
    def Hx(self) -> np.ndarray:
        return self._h[: self._n_Hx].reshape(self._shape_Hx)

    @property
    def Hy(self) -> np.ndarray:
        return self._h[self._n_Hx : self._n_Hx + self._n_Hy].reshape(self._shape_Hy)

    @property
    def Hz(self) -> np.ndarray:
        return self._h[self._n_Hx + self._n_Hy :].reshape(self._shape_Hz)

    # -- Setters (for backward compat with `fields.Ex = array`) -------------

    @Ex.setter
    def Ex(self, value) -> None:
        self._e[: self._n_Ex] = self._xp.asarray(value).ravel()

    @Ey.setter
    def Ey(self, value) -> None:
        self._e[self._n_Ex : self._n_Ex + self._n_Ey] = self._xp.asarray(value).ravel()

    @Ez.setter
    def Ez(self, value) -> None:
        self._e[self._n_Ex + self._n_Ey :] = self._xp.asarray(value).ravel()

    @Hx.setter
    def Hx(self, value) -> None:
        self._h[: self._n_Hx] = self._xp.asarray(value).ravel()

    @Hy.setter
    def Hy(self, value) -> None:
        self._h[self._n_Hx : self._n_Hx + self._n_Hy] = self._xp.asarray(value).ravel()

    @Hz.setter
    def Hz(self, value) -> None:
        self._h[self._n_Hx + self._n_Hy :] = self._xp.asarray(value).ravel()

    # -- Factory ------------------------------------------------------------

    @classmethod
    def zeros(cls, Nx: int, Ny: int, Nz: int, xp=None, dtype=float) -> "FieldState":
        """Allocate a zero-initialized FieldState for a grid of size Nx×Ny×Nz.

        Args:
            Nx, Ny, Nz: Number of cells in each direction.
            xp: Array module to use (default: numpy). Pass ``get_xp()`` for
                backend-agnostic allocation.
            dtype: Field scalar dtype (default: ``float`` = float64). Pass
                ``float32`` for the single-precision time loop (plan WP1);
                the backend (``xp``) and precision (``dtype``) axes are
                orthogonal.
        """
        if xp is None:
            xp = np
        return cls(
            Ex=xp.zeros((Nx, Ny + 1, Nz + 1), dtype=dtype),
            Ey=xp.zeros((Nx + 1, Ny, Nz + 1), dtype=dtype),
            Ez=xp.zeros((Nx + 1, Ny + 1, Nz), dtype=dtype),
            Hx=xp.zeros((Nx + 1, Ny, Nz), dtype=dtype),
            Hy=xp.zeros((Nx, Ny + 1, Nz), dtype=dtype),
            Hz=xp.zeros((Nx, Ny, Nz + 1), dtype=dtype),
        )

    # -- Energy helpers -----------------------------------------------------

    def energy_E(self, M_eps) -> float:
        """Total electric field energy (approximate, using flat arrays)."""
        return 0.5 * float((M_eps * self._e) @ self._e)

    def energy_H(self, M_mu) -> float:
        """Total magnetic field energy (approximate, using flat arrays)."""
        return 0.5 * float((M_mu * self._h) @ self._h)

    def __repr__(self) -> str:
        return (
            f"FieldState("
            f"Ex{self._shape_Ex}, Ey{self._shape_Ey}, Ez{self._shape_Ez}, "
            f"Hx{self._shape_Hx}, Hy{self._shape_Hy}, Hz{self._shape_Hz})"
        )
