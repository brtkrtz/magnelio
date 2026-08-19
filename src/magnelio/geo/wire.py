"""
ThinWire — a sub-cell wire on grid edges.

A :class:`ThinWire` is a physical thin conductor along a
:class:`~magnelio.geo.curves.Curve`: implicitly PEC, with a radius far
below the cell size.  It is *not* a solid — the mesher rasterises its curve
onto the grid E-edges (the canonical curve rasteriser), masks that edge
chain PEC, and applies the Holland/Simpson sub-cell correction of
:mod:`magnelio.mesh._thin_wire` so the wire presents the physical per-length
inductance ``L' = (mu/2pi)·ln(delta/a)`` instead of the bare-grid value.
"""

# Design: DD-080 (ThinWire sub-cell model), DD-076 (canonical curve rasteriser).

from __future__ import annotations

from dataclasses import dataclass

from magnelio.geo._validate import positive
from magnelio.geo.curves import Curve
from magnelio.materials.material import Material


@dataclass
class ThinWire:
    """A thin PEC wire along a Curve (radius far below the cell size).

    A new leaf-node category in the geometry-object protocol: like a solid
    it exposes ``.material`` (always PEC), ``._occ_shape()`` (the curve's
    ``TopoDS_Wire``) and :meth:`bounding_box` — so persistence, validation
    and critical-plane extraction work unchanged — but it has no volume:
    the mesher never fills or classifies it, it applies the thin-wire
    sub-cell model instead.

    Parameters
    ----------
    curve : Curve
        The wire's path (polyline / arc / spline / helix).  Rasterised to
        an axis-aligned edge staircase at meshing time.
    radius : float
        Physical wire radius [m].  Must be positive and stay below ~0.3 of
        the smallest transverse cell along the path (checked at meshing
        time — the thin-wire model is a sub-cell model; a fatter conductor
        should be a resolved cylinder instead).
    name : str, optional
        Optional label used in warnings and error messages.
    """

    curve: Curve
    radius: float
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.curve, Curve):
            raise TypeError(f"ThinWire.curve must be a Curve, got {type(self.curve).__name__}.")
        self.radius = positive(self.radius, "ThinWire.radius")

    @property
    def material(self) -> Material:
        """The wire's material — always PEC in v1."""
        return Material.pec()

    def _occ_shape(self, scale=1.0):
        return self.curve._occ_shape(scale)

    def bounding_box(
        self, scale: float | None = None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Bounding box of the wire's *curve* (the radius is NOT included).

        The radius is a sub-cell model parameter: inflating the box by ~a
        would introduce feature planes 2a apart and collapse the mesher's
        fine cell size.
        """
        return self.curve.bounding_box(scale)

    def _analytic_bbox(self):
        return self.curve._analytic_bbox()
