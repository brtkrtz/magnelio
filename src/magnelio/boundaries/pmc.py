"""
Perfect Magnetic Conductor (PMC) boundary condition.

PMC enforces H_tangential = 0 on a magnetic wall.  On the staggered
FIT grid this is the *natural* boundary condition of the free curl
operators: the E-update at the outermost primal grid plane accumulates
only the in-domain tangential-H faces (missing faces contribute zero),
and the mass matrices assign the boundary E-edges a *full* dual cell
(``_build_avg_d``).  Together this is the mirror closure
``H_tan(-Δ/2) = 0`` — a magnetic wall Δ/2 *outside* the outermost grid
line — identical to the natural-BC wall of the 2D port mode solver and
``EigenmodeSolver3D``.

Consequently the runtime BC object performs no field surgery at all:
it exists as the face-coverage marker for the solver's bbox audit and
to keep the six-face BC dict explicit.  (The previous implementation
zeroed tangential H on the first/last *cell-centre* layer, which put
the TD wall Δ/2 *inside* — one full cell off both mode solvers, and
measurably non-passive S-matrices as a consequence.)
"""

# Design: PORT_MODES_PLAN.md WP-U0 (measured non-passive S-matrix of the old
# cell-centre wall).

from __future__ import annotations

from typing import TYPE_CHECKING

from magnelio.mesh.grid import GridLines

if TYPE_CHECKING:
    from magnelio._fields.field_arrays import FieldState


class PMCBoundary:
    """PMC boundary on one face of the simulation domain.

    The magnetic wall is realised by the natural boundary condition of
    the free FIT operators (see module docstring); it sits Δ/2 outside
    the outermost primal grid line of this face.  ``apply_E`` /
    ``apply_H`` are therefore no-ops — the instance only marks the
    face as closed.

    Parameters
    ----------
    face : str
        One of ``'xmin'``, ``'xmax'``, ``'ymin'``, ``'ymax'``,
        ``'zmin'``, ``'zmax'``.
    grid : GridLines
        The simulation grid.
    """

    _VALID_FACES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")

    def __init__(self, face: str, grid: GridLines) -> None:
        if face not in self._VALID_FACES:
            raise ValueError(f"Unknown face: {face!r}")
        self.face = face
        self.grid = grid

    def apply_E(self, fields: "FieldState") -> None:
        return

    def apply_H(self, fields: "FieldState") -> None:
        return

    def __repr__(self) -> str:
        return f"PMCBoundary(face={self.face!r})"
