"""The planar-sheet marker base.

A *planar sheet* is a zero-thickness planar region: a
:class:`~magnelio.geo.Face` (an axis-normal polygon) or the result of
:meth:`~magnelio.geo.Curve.covered` (a free planar region bounded by a
closed curve).  Both are profiles first and objects second, so the whole
package needs one predicate to tell them apart from solids.

This module sits directly above ``shape`` in the import graph and must
not import anything else from the package.
"""

from __future__ import annotations

from magnelio.geo.shape import Shape


class PlanarSheet(Shape):
    """Base class of every zero-thickness planar profile.

    A planar sheet *is* its own profile: :meth:`~magnelio.geo.Shape.extruded`,
    :meth:`~magnelio.geo.Shape.revolved` and :meth:`~magnelio.geo.Shape.swept`
    take it directly and need no ``face_near`` to pick a face, and
    :meth:`~magnelio.geo.Shape.thickened` grows it into a solid slab.

    Carrying a material makes a sheet a *thin sheet* rather than a
    construction profile; thin-sheet physics is not wired yet, so a
    standalone sheet cannot be meshed either way.
    """
