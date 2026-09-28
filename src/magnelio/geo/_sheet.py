"""Standalone sheets and planar profiles."""

from __future__ import annotations

from dataclasses import dataclass

from magnelio.geo._cache import cached_occ_shape
from magnelio.geo.shape import Shape


class Sheet(Shape):
    """Base class of zero-thickness planar and curved sheets.

    Sheets can be extruded or thickened into solids. A standalone sheet
    cannot yet be meshed; give it a resolved physical thickness first.
    """


@dataclass
class Profile(Sheet):
    """A bounded planar region with one outer boundary and optional holes.

    Use :meth:`polygon`, :meth:`rectangle`, :meth:`circle`, or
    :meth:`from_wires` to construct a profile. All coordinates are world
    coordinates. Boundaries are closed, planar, and non-intersecting;
    holes lie strictly inside the outer boundary and do not nest.

    An optional material is inherited by the solid produced by extrusion,
    revolution, sweep, or loft. Profiles themselves cannot be meshed.
    """

    _outer: object
    _holes: tuple = ()
    material: object = None
    name: str | None = None

    def __post_init__(self):
        from magnelio.geo._scaling import choose_scale  # noqa: PLC0415
        from magnelio.geo.curves import Curve  # noqa: PLC0415
        from magnelio.materials.material import resolve_material  # noqa: PLC0415

        self._holes = tuple(self._holes)
        for curve in (self._outer, *self._holes):
            if not isinstance(curve, Curve):
                raise TypeError("Profile.from_wires needs Curve boundaries.")
            if curve._ends is not None and not curve.is_closed:
                raise ValueError("Profile.from_wires needs closed Curve boundaries.")
        self.material = resolve_material(self.material, "Profile.material")
        scale = choose_scale(*self._analytic_bbox())
        self._occ_shape(scale)

    @classmethod
    def from_wires(cls, outer, holes=(), *, material=None, name=None):
        """Return the region inside *outer*, excluding all *holes*.

        Parameters
        ----------
        outer : Curve
            Closed planar outer boundary.
        holes : iterable of Curve, optional
            Closed boundaries strictly inside *outer*. Winding is corrected
            automatically. Touching, intersecting, or nested holes are invalid.
        material : Material or str, optional
            Material inherited by subsequent solid construction.
        name : str, optional
            Optional label.

        Returns
        -------
        Profile
            Validated planar region, retaining analytic boundary curves.

        Raises
        ------
        TypeError
            If any boundary is not a Curve.
        ValueError
            If boundaries are open, non-planar, intersecting, or incorrectly nested.
        """
        try:
            holes = tuple(holes)
        except TypeError:
            raise TypeError(
                "Profile.from_wires holes must be an iterable of Curve boundaries."
            ) from None
        return cls(outer, holes, material, name)

    @classmethod
    def polygon(cls, points, *, material=None, name=None):
        """Return a planar polygon through three-dimensional world points.

        Parameters
        ----------
        points : sequence of tuple of float
            At least three coplanar vertices [meters]. Closure is automatic;
            an explicit repeated last vertex is also accepted.
        material : Material or str, optional
            Material inherited by subsequent solid construction.
        name : str, optional
            Optional label.

        Returns
        -------
        Profile
            The bounded polygon region.
        """
        from magnelio.geo._validate import point_list  # noqa: PLC0415
        from magnelio.geo.curves import Curve  # noqa: PLC0415

        pts = point_list(points, "Profile.polygon(points)", dim=3, minimum=3)
        if pts[-1] != pts[0]:
            pts += (pts[0],)
        if any(a == b for a, b in zip(pts, pts[1:])):
            raise ValueError("Profile.polygon needs distinct consecutive vertices.")
        return cls.from_wires(Curve.polyline(pts), material=material, name=name)

    @classmethod
    def rectangle(cls, center, size, *, normal="z", x_direction=None, material=None, name=None):
        """Return an oriented rectangle centred at a world point.

        Parameters
        ----------
        center : tuple of float
            World centre [meters].
        size : tuple of float
            Positive width and height [meters].
        normal : str or sequence of float, optional
            Plane normal; defaults to ``"z"``.
        x_direction : str or sequence of float, optional
            In-plane width direction. Its normal component is discarded.
            Defaults to the least parallel world axis projected into the plane.
            Height follows ``normal x x_direction``.
        material : Material or str, optional
            Material inherited by subsequent solid construction.
        name : str, optional
            Optional label.

        Returns
        -------
        Profile
            The bounded rectangle region.
        """
        import numpy as np  # noqa: PLC0415

        from magnelio.geo._axes import normalize_axis  # noqa: PLC0415
        from magnelio.geo._validate import point3, positive  # noqa: PLC0415

        c = np.array(point3(center, "Profile.rectangle(center)"))
        try:
            width, height = size
        except (TypeError, ValueError):
            raise ValueError("Profile.rectangle(size) needs two positive lengths.") from None
        width, height = positive(width, "size[0]"), positive(height, "size[1]")
        n = np.array(normalize_axis(normal, "Profile.rectangle(normal)"))
        if x_direction is None:
            x_direction = np.eye(3)[np.argmin(np.abs(n))]
        u = np.array(normalize_axis(x_direction, "Profile.rectangle(x_direction)"))
        u -= np.dot(u, n) * n
        if np.linalg.norm(u) <= 1e-9:
            raise ValueError("x_direction must not be parallel to normal.")
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        pts = [
            tuple(c + sx * width / 2 * u + sy * height / 2 * v)
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]
        return cls.polygon(pts, material=material, name=name)

    @classmethod
    def circle(cls, center, radius, *, normal="z", material=None, name=None):
        """Return the planar disc bounded by an exact circle.

        Parameters
        ----------
        center : tuple of float
            World centre [meters].
        radius : float
            Positive radius [meters].
        normal : str or sequence of float, optional
            Plane normal; defaults to ``"z"``.
        material : Material or str, optional
            Material inherited by subsequent solid construction.
        name : str, optional
            Optional label.

        Returns
        -------
        Profile
            The bounded circular disc.
        """
        from magnelio.geo.curves import Curve  # noqa: PLC0415

        return cls.from_wires(
            Curve.circle(center, radius, normal=normal), material=material, name=name
        )

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        from magnelio.geo._occ_backend import make_profile_face  # noqa: PLC0415

        return make_profile_face(
            self._outer._occ_shape(scale), [h._occ_shape(scale) for h in self._holes]
        )

    def _analytic_bbox(self):
        return self._outer._analytic_bbox()

    def boundary(self):
        """Return detached boundary Curves, outer first followed by holes.

        Returns
        -------
        tuple of Curve
            Independent, closed geometry values in the profile's world placement.
            Tuple order fixes hole correspondence when lofting multiple profiles.
        """
        if hasattr(self, "_inner"):
            return tuple(self._transform @ curve for curve in self._inner.boundary())
        return (self._outer, *self._holes)

    @property
    def area(self):
        """Area of the planar region, excluding holes [square meters]."""
        from OCC.Core.BRepGProp import brepgprop  # noqa: PLC0415
        from OCC.Core.GProp import GProp_GProps  # noqa: PLC0415

        from magnelio.geo._scaling import choose_scale  # noqa: PLC0415

        scale = choose_scale(*self._analytic_bbox())
        props = GProp_GProps()
        brepgprop.SurfaceProperties(self._occ_shape(scale), props)
        return props.Mass() / scale**2
