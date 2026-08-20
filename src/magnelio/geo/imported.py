"""Kernel-shape-backed geometry: solids that came from outside.

A shape built with the CSG API knows its own construction — a
:class:`~magnelio.geo.Brick` can be rebuilt at any model scale from its
parameters.  A solid read from a CAD file or from a project store
cannot: all that exists is its boundary representation.
:class:`ImportedSolid` is the wrapper that turns such a boundary
representation into a full citizen of the geometry API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from magnelio.geo.shape import Shape

if TYPE_CHECKING:
    from magnelio.materials.material import Material


class ImportedSolid(Shape):
    """A solid whose geometry came from a CAD file or a project store.

    Instances are produced by :func:`~magnelio.io.import_step`,
    :func:`~magnelio.io.import_brep` and by reading back a project's
    stored geometry; there is no reason to build one by hand.

    An imported solid is a :class:`Shape` like any other — Boolean
    operators, the chainable verbs, :meth:`~magnelio.geo.Shape.volume`
    and :meth:`bounding_box` all work on it, and it can be added to a
    :class:`~magnelio.geo.GeometryModel` once it carries a material.
    What it is *not* is the original CSG construction: boundary
    representation is a finished solid, so the parameters it was drawn
    from (a radius, an extrusion length) are gone and cannot be edited.

    Attributes
    ----------
    material : Material or None
        Material of the solid.  ``None`` marks a construction body — a
        solid that was imported but not assigned a material; it can act
        as a Boolean operand but is rejected by
        :meth:`~magnelio.geo.GeometryModel.add`.
    name : str or None
        Name the solid carried in its source file.  This is the key
        material assignments are made against, so it survives a
        re-export of the same CAD model.
    color : tuple of float or None
        Display colour ``(r, g, b)`` in 0–1 from the source file, or
        ``None``.  Purely cosmetic: plots and the ParaView export use it
        when the material prescribes no colour of its own; it never
        affects the physics.
    """

    def __init__(
        self,
        shape,
        material: "Material | None" = None,
        name: str | None = None,
        color: tuple[float, float, float] | None = None,
    ) -> None:
        self._shape = shape
        self.material = material
        self.name = name
        self.color = color
        self._scaled: dict[float, object] = {}

    def _occ_shape(self, scale: float = 1.0):
        # The stored BRep is meter-space; a non-unity DD-120 scale is
        # realised as a lazy uniform transform (cached per scale).
        if scale == 1.0:
            return self._shape
        cached = self._scaled.get(scale)
        if cached is None:
            from magnelio.geo._occ_backend import occ_scale  # noqa: PLC0415

            cached = occ_scale(self._shape, float(scale), (0.0, 0.0, 0.0))
            self._scaled[scale] = cached
        return cached

    def _analytic_bbox(self):
        # OCC-free is impossible for a BRep-only shape, but no build at
        # an unknown scale is involved either: the meter-space shape
        # already exists, so its BRep bbox is a valid (exact) source
        # for the DD-120 scale choice.
        from magnelio.geo._occ_backend import bounding_box  # noqa: PLC0415

        return bounding_box(self._shape)

    def bounding_box(self, scale: float | None = None):
        """Axis-aligned bounding box ``(min_corner, max_corner)`` [meters]."""
        from magnelio.geo._occ_backend import bounding_box  # noqa: PLC0415

        return bounding_box(self._shape)

    def __repr__(self) -> str:
        mat = getattr(self.material, "name", None)
        return f"ImportedSolid(name={self.name!r}, material={mat!r})"
