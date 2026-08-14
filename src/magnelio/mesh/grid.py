"""
GridLines: Stores sorted node positions for the non-uniform hexahedral grid.

See spec.md for the data-structure specification.
"""

# Design: DD-028 (two-scale h_max / h_fine grid-line generation algorithm).

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GridLines:
    """Sorted node positions for a structured non-uniform hexahedral grid.

    All coordinates are in meters.

    The number of cells in each direction is ``len(x) - 1``, etc.

    Parameters
    ----------
    x : np.ndarray
        Sorted array of x node positions, shape ``(Nx+1,)``.
    y : np.ndarray
        Sorted array of y node positions, shape ``(Ny+1,)``.
    z : np.ndarray
        Sorted array of z node positions, shape ``(Nz+1,)``.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    def __post_init__(self) -> None:
        for name, arr in (("x", self.x), ("y", self.y), ("z", self.z)):
            arr = np.asarray(arr, dtype=float)
            if arr.ndim != 1 or len(arr) < 2:
                raise ValueError(f"GridLines.{name} must be a 1D array with at least 2 elements")
            if not np.all(np.diff(arr) > 0):
                raise ValueError(f"GridLines.{name} must be strictly increasing")
            object.__setattr__(self, name, arr)

    # ------------------------------------------------------------------
    # Derived quantities (computed, not stored)
    # ------------------------------------------------------------------

    @property
    def Nx(self) -> int:
        """Number of cells in x direction."""
        return len(self.x) - 1

    @property
    def Ny(self) -> int:
        """Number of cells in y direction."""
        return len(self.y) - 1

    @property
    def Nz(self) -> int:
        """Number of cells in z direction."""
        return len(self.z) - 1

    @property
    def dx(self) -> np.ndarray:
        """Cell sizes in x, shape (Nx,)."""
        return np.diff(self.x)

    @property
    def dy(self) -> np.ndarray:
        """Cell sizes in y, shape (Ny,)."""
        return np.diff(self.y)

    @property
    def dz(self) -> np.ndarray:
        """Cell sizes in z, shape (Nz,)."""
        return np.diff(self.z)

    @property
    def dx_min(self) -> float:
        return float(self.dx.min())

    @property
    def dy_min(self) -> float:
        return float(self.dy.min())

    @property
    def dz_min(self) -> float:
        return float(self.dz.min())

    @property
    def n_cells(self) -> int:
        """Total number of cells."""
        return self.Nx * self.Ny * self.Nz

    @property
    def courant_dt_max(self) -> float:
        """Maximum stable time step (Courant limit, safety factor = 1.0).

        Use ``stability.courant_dt(grid, accuracy)`` for production use.
        """
        from magnelio.constants import C0  # noqa: PLC0415

        return 1.0 / (C0 * np.sqrt(1 / self.dx_min**2 + 1 / self.dy_min**2 + 1 / self.dz_min**2))

    def __repr__(self) -> str:
        return (
            f"GridLines(Nx={self.Nx}, Ny={self.Ny}, Nz={self.Nz}, "
            f"n_cells={self.n_cells}, "
            f"x=[{self.x[0]:.3g}, {self.x[-1]:.3g}] m, "
            f"y=[{self.y[0]:.3g}, {self.y[-1]:.3g}] m, "
            f"z=[{self.z[0]:.3g}, {self.z[-1]:.3g}] m)"
        )
