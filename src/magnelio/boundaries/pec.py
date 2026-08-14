"""
Perfect Electric Conductor (PEC) boundary condition.

PEC enforces E_tangential = 0 on the boundary faces.
Implemented by zeroing out tangential E-field components after each update.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from magnelio._fields.field_arrays import FieldState


class PECBoundary:
    """PEC boundary on one face of the simulation domain.

    Parameters
    ----------
    face : str
        One of ``'xmin'``, ``'xmax'``, ``'ymin'``, ``'ymax'``,
        ``'zmin'``, ``'zmax'``.
    wall_sigma : float, optional
        Conductivity [S/m] of THIS face's wall for the wall-loss
        models.  ``None`` (default) falls back to the analysis-level
        ``wall_sigma``.  The boundary condition carries the wall
        material, following large-suite convention; the field update
        itself stays ideal PEC either way (losses enter through
        ``wall_model``).
    wall_mu : float
        Relative permeability accompanying ``wall_sigma`` (default
        1.0; only consulted when ``wall_sigma`` is set).
    wall_roughness : SurfaceRoughness, optional
        Surface-roughness model for this face's wall (only consulted
        when ``wall_sigma`` is set).
    """

    def __init__(
        self,
        face: str,
        wall_sigma: float | None = None,
        wall_mu: float = 1.0,
        wall_roughness: object = None,
    ) -> None:
        self.face = face
        self.wall_sigma = wall_sigma
        self.wall_mu = wall_mu
        self.wall_roughness = wall_roughness

    def apply(self, E: "FieldState") -> None:
        """Zero tangential E-field on this PEC face.

        Parameters
        ----------
        E : FieldState
            :class:`~magnelio._fields.field_arrays.FieldState` to
            modify in place.
        """
        face = self.face
        if face == "xmin":
            E.Ey[0, :, :] = 0.0
            E.Ez[0, :, :] = 0.0
        elif face == "xmax":
            E.Ey[-1, :, :] = 0.0
            E.Ez[-1, :, :] = 0.0
        elif face == "ymin":
            E.Ex[:, 0, :] = 0.0
            E.Ez[:, 0, :] = 0.0
        elif face == "ymax":
            E.Ex[:, -1, :] = 0.0
            E.Ez[:, -1, :] = 0.0
        elif face == "zmin":
            E.Ex[:, :, 0] = 0.0
            E.Ey[:, :, 0] = 0.0
        elif face == "zmax":
            E.Ex[:, :, -1] = 0.0
            E.Ey[:, :, -1] = 0.0
        else:
            raise ValueError(f"Unknown face: {self.face!r}")

    # PEC is an E-constraint: it must run after the E-update.  ``apply_E``
    # is the dispatching alias used by the FIT-TD inner loop; ``apply_H``
    # is a no-op so the solver can call both unconditionally on every
    # boundary type and let each type opt in to the appropriate stage.
    def apply_E(self, fields: "FieldState") -> None:
        self.apply(fields)

    def apply_H(self, fields: "FieldState") -> None:
        return

    def __repr__(self) -> str:
        return f"PECBoundary(face={self.face!r})"
