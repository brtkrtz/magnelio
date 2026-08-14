"""
Periodic boundary condition.

Enforces field periodicity by copying ghost-cell values from the opposite face
after each E and H update.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from magnelio.mesh.grid import GridLines

if TYPE_CHECKING:
    from magnelio._fields.field_arrays import FieldState


class PeriodicBoundary:
    """Periodic boundary along one axis pair.

    Parameters
    ----------
    axis : str
        ``'x'``, ``'y'``, or ``'z'`` — the periodic axis.
    grid : GridLines
        The simulation grid.
    """

    def __init__(self, axis: str, grid: GridLines) -> None:
        if axis not in ("x", "y", "z"):
            raise ValueError(f"axis must be 'x', 'y', or 'z'; got {axis!r}")
        self.axis = axis
        self.grid = grid

    def apply_E(self, E: "FieldState") -> None:
        """Apply periodic BC to E field (copy from opposite face)."""
        if self.axis == "x":
            E.Ey[0, :, :] = E.Ey[-2, :, :]
            E.Ez[0, :, :] = E.Ez[-2, :, :]
        elif self.axis == "y":
            E.Ex[:, 0, :] = E.Ex[:, -2, :]
            E.Ez[:, 0, :] = E.Ez[:, -2, :]
        elif self.axis == "z":
            E.Ex[:, :, 0] = E.Ex[:, :, -2]
            E.Ey[:, :, 0] = E.Ey[:, :, -2]

    def apply_H(self, H: "FieldState") -> None:
        """Apply periodic BC to H field (copy from opposite face)."""
        if self.axis == "x":
            H.Hy[-1, :, :] = H.Hy[1, :, :]
            H.Hz[-1, :, :] = H.Hz[1, :, :]
        elif self.axis == "y":
            H.Hx[:, -1, :] = H.Hx[:, 1, :]
            H.Hz[:, -1, :] = H.Hz[:, 1, :]
        elif self.axis == "z":
            H.Hx[:, :, -1] = H.Hx[:, :, 1]
            H.Hy[:, :, -1] = H.Hy[:, :, 1]

    def __repr__(self) -> str:
        return f"PeriodicBoundary(axis={self.axis!r})"
