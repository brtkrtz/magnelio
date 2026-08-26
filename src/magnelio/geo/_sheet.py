"""The sheet marker bases.

A *sheet* is a zero-thickness region: a planar one — a
:class:`~magnelio.geo.Face` (an axis-normal polygon) or the result of
:meth:`~magnelio.geo.Curve.covered` (a free planar region bounded by a
closed curve) — or a curved one, a :class:`~magnelio.geo.Surface`
sampled from a parametric map.  All of them are profiles first and
objects second, so the whole package needs one predicate to tell them
apart from solids, and a second one for the verbs that need a plane
(revolving, sweeping, lofting).

This module sits directly above ``shape`` in the import graph and must
not import anything else from the package.
"""

from __future__ import annotations

from magnelio.geo.shape import Shape


class Sheet(Shape):
    """Base class of every zero-thickness sheet, planar or curved.

    A sheet *is* its own profile: :meth:`~magnelio.geo.Shape.extruded`
    takes it directly and needs no ``face_near`` to pick a face, and
    :meth:`~magnelio.geo.Shape.thickened` grows it into a solid of
    constant thickness — the direct way from a drawn or computed
    surface to a metal shell such as a reflector.

    Carrying a material makes a sheet a *thin sheet* rather than a
    construction profile; thin-sheet physics is not wired yet, so a
    standalone sheet cannot be meshed either way.
    """


class PlanarSheet(Sheet):
    """Base class of every zero-thickness *planar* profile.

    Besides what every :class:`Sheet` can do, a planar sheet is the
    profile of :meth:`~magnelio.geo.Shape.revolved` and
    :meth:`~magnelio.geo.Shape.swept`, and a section of a
    :class:`~magnelio.geo.Loft` — the verbs that need a plane to turn,
    move or interpolate.
    """
