"""
CSG geometry primitives.

Each primitive stores its geometric parameters and an associated material.
The OCC shape is built lazily via occ_backend.py when needed for meshing.

Supported primitives (v1.0): Brick, Sphere, Cylinder, Cone, Torus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from magnelio.materials.material import Material


from magnelio.geo._cache import cached_occ_shape
from magnelio.geo._sheet import PlanarSheet
from magnelio.geo.shape import Shape


@dataclass
class _BaseShape(Shape):
    """Common base for all CSG shapes.

    ``material`` is optional.  A solid that carries one is a *physical*
    object and can be added to a
    :class:`~magnelio.geo.GeometryModel`; a solid without one is a
    **construction solid** — a body that exists only to shape other
    bodies through Boolean operations, the volumetric sibling of the
    material-less :class:`Face` profile::

        ring = Cylinder(radius=r_out, height=t, material=pec) - Cylinder(
            origin=(0, 0, -1), radius=r_in, height=2
        )

    The cutting cylinder never becomes part of the model, so demanding a
    material for it would be noise: ``Difference``/``Union``/
    ``Intersection`` take their material from the base (resp. first)
    operand and ignore the tools' entirely.  Adding a material-less
    solid to a model is rejected by
    :meth:`~magnelio.geo.GeometryModel.add`, so the omission cannot
    silently reach the mesher.
    """

    material: "Material | None" = None
    name: str | None = None

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        """Build and return the OpenCASCADE TopoDS_Shape at *scale*.

        *scale* is the DD-120 model scale factor: the OCC shape lives in
        scaled units (meters x scale); the backend converts every query
        result back to meters.  Implemented by subclasses.
        """
        raise NotImplementedError

    def _analytic_bbox(self):
        """Conservative OCC-free bounding box [m] (DD-120 scale choice)."""
        raise NotImplementedError


@dataclass
class Brick(_BaseShape):
    """Axis-aligned rectangular box (cuboid).

    Parameters
    ----------
    origin : tuple of float
        ``(x, y, z)`` of the minimum-coordinate corner [meters].
    size : tuple of float
        ``(dx, dy, dz)`` — extents in each direction [meters].
    material : Material, optional
        Material filling this volume.  Omit it for a construction solid
        used only as a Boolean operand.
    name : str, optional
        Optional label.
    """

    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: tuple[float, float, float] = (1.0, 1.0, 1.0)

    @classmethod
    def from_corners(cls, p1, p2, *, material=None, name=None) -> "Brick":
        """Build an axis-aligned Brick from two opposite corners.

        The corners may be given in any order; each axis is normalised so
        that the resulting ``origin`` holds the minimum coordinate and
        ``size`` is non-negative.  The ``origin``/``size`` fields are
        populated as for a directly constructed :class:`Brick`, so the two
        forms are interchangeable downstream.

        Parameters
        ----------
        p1, p2 : tuple of float
            Two opposite corners ``(x, y, z)`` of the box [meters], in any
            order.
        material : Material, optional
            Material filling this volume.  Omit it for a construction
            solid used only as a Boolean operand — the two-corner form
            is the natural spelling for a cutting box.
        name : str, optional
            Optional label.

        Returns
        -------
        Brick

        Examples
        --------
        >>> Brick.from_corners((3e-3, 0, 2e-3), (0, 4e-3, 0), material=pec)
        Brick(origin=(0.0, 0.0, 0.0), size=(3e-3, 4e-3, 2e-3), ...)
        """
        lo = tuple(min(a, b) for a, b in zip(p1, p2))
        hi = tuple(max(a, b) for a, b in zip(p1, p2))
        size = tuple(h - l for l, h in zip(lo, hi))
        return cls(origin=lo, size=size, material=material, name=name)

    @classmethod
    def from_ranges(
        cls,
        *,
        x1=None,
        x2=None,
        dx=None,
        y1=None,
        y2=None,
        dy=None,
        z1=None,
        z2=None,
        dz=None,
        material=None,
        name=None,
    ) -> "Brick":
        """Build an axis-aligned Brick from one coordinate range per axis.

        Each axis is given by exactly two of its three keywords: the two
        bounds (``x1``, ``x2``), a bound and an extent (``x1``, ``dx`` or
        ``x2``, ``dx``).  Extents may be negative and bounds may come in
        either order; the result is normalised exactly as in
        :meth:`from_corners`, so both forms produce identical fields and
        are interchangeable downstream.

        This is the spelling to reach for when a box is described by the
        planes it lies between — a substrate between two metal layers, a
        cutting slab spanning a whole domain — where naming the two
        opposite corners would interleave the axes.

        Parameters
        ----------
        x1, x2, dx : float, optional
            Lower bound, upper bound and extent along x [meters].  Supply
            exactly two of them.
        y1, y2, dy : float, optional
            The same for y.
        z1, z2, dz : float, optional
            The same for z.
        material : Material, optional
            Material filling this volume.  Omit it for a construction
            solid used only as a Boolean operand.
        name : str, optional
            Optional label.

        Returns
        -------
        Brick

        Raises
        ------
        ValueError
            If an axis is given fewer or more than two of its keywords.

        See Also
        --------
        Brick.from_corners : Build the same box from two opposite corners.

        Examples
        --------
        A substrate of thickness ``h`` under a ground plane at ``z = 0``::

            Brick.from_ranges(
                x1=0, dx=w, y1=0, dy=length, z2=0, dz=h, material=fr4
            )

        >>> Brick.from_ranges(x1=0, x2=3e-3, y1=0, dy=4e-3, z1=2e-3, z2=0)
        Brick(origin=(0.0, 0.0, 0.0), size=(3e-3, 4e-3, 2e-3), ...)
        """
        from magnelio.geo._ranges import axis_range

        lo, hi = zip(
            axis_range("x", x1, x2, dx),
            axis_range("y", y1, y2, dy),
            axis_range("z", z1, z2, dz),
        )
        size = tuple(h - low for low, h in zip(lo, hi))
        return cls(origin=lo, size=size, material=material, name=name)

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        from magnelio.geo._occ_backend import make_brick

        return make_brick(self.origin, self.size, scale=scale)

    def _analytic_bbox(self):
        from magnelio.geo._scaling import box_of_points

        corner = tuple(o + s for o, s in zip(self.origin, self.size))
        return box_of_points([self.origin, corner])


@dataclass
class Sphere(_BaseShape):
    """Sphere.

    Parameters
    ----------
    center : tuple of float
        ``(x, y, z)`` center position [meters].
    radius : float
        Radius [meters].
    material : Material, optional
        Material filling this volume.  Omit it for a construction solid
        used only as a Boolean operand.
    name : str, optional
        Optional label.
    """

    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 1.0

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        from magnelio.geo._occ_backend import make_sphere

        return make_sphere(self.center, self.radius, scale=scale)

    def _analytic_bbox(self):
        from magnelio.geo._scaling import pad_box

        return pad_box((self.center, self.center), self.radius)


@dataclass
class Cylinder(_BaseShape):
    """Right circular cylinder, optionally hollow and optionally a segment.

    Left alone, ``inner_radius`` and ``angle_deg`` give a plain solid
    cylinder.  Setting ``inner_radius`` bores it out into a tube, setting
    ``angle_deg`` cuts a wedge out of the full turn, and setting both
    gives the curved slab that a segmented electrode or a septum of a
    circular structure is made of.

    Angles are measured about *axis* in the same right-handed sense as
    :meth:`~magnelio.geo.Shape.rotated`.  Zero lies on the first
    coordinate direction perpendicular to *axis* — ``+x`` for an
    ``'z'`` axis, ``+y`` for an ``'x'`` axis.

    Parameters
    ----------
    origin : tuple of float
        ``(x, y, z)`` center of the bottom face [meters].
    radius : float
        Outer radius [meters].
    height : float
        Height [meters].  A negative height extrudes along ``-axis``
        from the origin.
    axis : str or tuple of float
        Axis direction: ``'x'``/``'y'``/``'z'`` or any 3-vector
        (default ``'z'``).
    inner_radius : float
        Radius of the axial bore [meters]; 0 (default) for a solid
        cylinder, otherwise less than ``radius``.
    angle_deg : float or tuple of float, optional
        Angular extent [degrees]: a single value for a segment starting
        at zero, or ``(start, end)`` for one anywhere.  ``None``
        (default) is the full turn.
    material : Material, optional
        Material filling this volume.  Omit it for a construction solid
        used only as a Boolean operand.
    name : str, optional
        Optional label.

    Examples
    --------
    A 20 degree segment of a hollow cylinder — a curved electrode::

            electrode = Cylinder(
                radius=12e-3, inner_radius=10e-3, height=30e-3,
                angle_deg=(0, 20), material=pec,
            )
    """

    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 1.0
    height: float = 1.0
    axis: "str | tuple[float, float, float]" = "z"
    inner_radius: float = 0.0
    angle_deg: "float | tuple[float, float] | None" = None

    def __post_init__(self):
        if self.inner_radius < 0.0:
            raise ValueError(
                f"Cylinder inner_radius must not be negative; got {self.inner_radius}."
            )
        if self.inner_radius >= self.radius:
            raise ValueError(
                f"Cylinder inner_radius ({self.inner_radius}) must be smaller "
                f"than radius ({self.radius}) — there would be no wall left."
            )
        self._angle_span()

    def _angle_span(self):
        """``(start, end)`` in degrees, or None for a full cylinder."""
        if self.angle_deg is None:
            return None
        try:
            start, end = self.angle_deg
        except TypeError:
            start, end = 0.0, float(self.angle_deg)
        start, end = float(start), float(end)
        if not 0.0 < end - start <= 360.0:
            raise ValueError(
                f"Cylinder angle_deg must sweep more than 0 and at most "
                f"360 degrees; ({start}, {end}) sweeps {end - start}."
            )
        return start, end

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        from magnelio.geo._occ_backend import make_cylinder

        return make_cylinder(
            self.origin,
            self.radius,
            self.height,
            self.axis,
            inner_radius=self.inner_radius,
            angle_deg=self._angle_span(),
            scale=scale,
        )

    def _analytic_bbox(self):
        from magnelio.geo._axes import cross, normalize_axis, reference_dir
        from magnelio.geo._scaling import axis_segment_box, box_of_points, sector_uv_points

        span = self._angle_span()
        direction = normalize_axis(self.axis)
        if span is None:
            # A full turn fills the circumscribed cylinder, bore or not.
            return axis_segment_box(self.origin, direction, self.height, self.radius)

        ref = reference_dir(direction)
        quarter = cross(direction, ref)
        base = tuple(o + min(0.0, self.height) * a for o, a in zip(self.origin, direction))
        top = tuple(b + abs(self.height) * a for b, a in zip(base, direction))
        points = []
        for u, v in sector_uv_points(*span, self.radius, self.inner_radius):
            offset = tuple(u * r + v * q for r, q in zip(ref, quarter))
            points.append(tuple(c + o for c, o in zip(base, offset)))
            points.append(tuple(c + o for c, o in zip(top, offset)))
        return box_of_points(points)


@dataclass
class Cone(_BaseShape):
    """Right circular cone (or truncated cone if top_radius > 0).

    Parameters
    ----------
    origin : tuple of float
        ``(x, y, z)`` center of the bottom face [meters].
    bottom_radius : float
        Bottom radius [meters].
    top_radius : float
        Top radius [meters].  Use 0 for a full cone.
    height : float
        Height [meters].  A negative height extrudes along ``-axis``
        from the origin.
    axis : str or tuple of float
        Axis direction: ``'x'``/``'y'``/``'z'`` or any 3-vector.
    material : Material, optional
        Material filling this volume.  Omit it for a construction solid
        used only as a Boolean operand.
    name : str, optional
        Optional label.
    """

    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bottom_radius: float = 1.0
    top_radius: float = 0.0
    height: float = 1.0
    axis: "str | tuple[float, float, float]" = "z"

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        from magnelio.geo._occ_backend import make_cone

        return make_cone(
            self.origin, self.bottom_radius, self.top_radius, self.height, self.axis, scale=scale
        )

    def _analytic_bbox(self):
        from magnelio.geo._axes import normalize_axis
        from magnelio.geo._scaling import axis_segment_box

        radius = max(self.bottom_radius, self.top_radius)
        return axis_segment_box(self.origin, normalize_axis(self.axis), self.height, radius)


@dataclass
class Face(PlanarSheet):
    """A standalone planar polygon face.

    A Face lives in an axis-normal plane and carries an **optional**
    material:

    - **no material** (default) — a *construction profile*: the input to
      :meth:`~magnelio.geo.Shape.extruded`,
      :meth:`~magnelio.geo.Shape.revolved`,
      :meth:`~magnelio.geo.Shape.swept` or
      :meth:`~magnelio.geo.Shape.thickened`, each of which turns it into
      a solid.  A material-less Face is not a physical object and is not
      meshed on its own.
    - **with a material** — a *thin sheet*.  The object is free to carry the
      material field, but thin-sheet *physics* wiring is deferred,
      so a material-carrying Face cannot yet be added to a
      :class:`~magnelio.geo.GeometryModel` for meshing.

    The polygon is given as in-plane ``(u, v)`` points; ``(u, v)`` map to
    the two axes orthogonal to *normal* following the package convention
    (normal ``'x'`` → u=y, v=z; ``'y'`` → u=x, v=z; ``'z'`` → u=x, v=y),
    the same frame :func:`cross_section_polygons` uses.

    Parameters
    ----------
    normal : str
        Plane normal axis: ``'x'``, ``'y'``, or ``'z'``.
    points : sequence of (float, float)
        In-plane ``(u, v)`` vertices [meters]; at least 3, without
        self-intersection.  The polygon is closed automatically.
    position : float
        Position of the plane along the normal axis [meters] (default 0).
    material : Material, optional
        Material of the thin sheet.  ``None`` (default) = construction
        profile.
    name : str, optional
        Optional label.
    """

    normal: str
    points: tuple
    position: float = 0.0
    material: "Material | None" = None
    name: str | None = None

    def __post_init__(self):
        if self.normal not in ("x", "y", "z"):
            raise ValueError(f"Face.normal must be 'x', 'y', or 'z'; got {self.normal!r}")
        if len(self.points) < 3:
            raise ValueError(f"A Face needs at least 3 points; got {len(self.points)}.")

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        from magnelio.geo._occ_backend import make_face

        return make_face(self.normal, self.position, self.points, scale=scale)

    def _analytic_bbox(self):
        from magnelio.geo._scaling import box_of_points

        uv_axes = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}[self.normal]
        normal_axis = {"x": 0, "y": 1, "z": 2}[self.normal]
        pts3 = []
        for u, v in self.points:
            p = [0.0, 0.0, 0.0]
            p[normal_axis] = self.position
            p[uv_axes[0]] = u
            p[uv_axes[1]] = v
            pts3.append(tuple(p))
        return box_of_points(pts3)


@dataclass
class Torus(_BaseShape):
    """Torus.

    Parameters
    ----------
    center : tuple of float
        ``(x, y, z)`` center of the torus [meters].
    major_radius : float
        Major radius (center of tube to center of torus) [meters].
    minor_radius : float
        Minor radius (tube radius) [meters].
    axis : str or tuple of float
        Symmetry axis: ``'x'``/``'y'``/``'z'`` or any 3-vector.
    material : Material, optional
        Material filling this volume.  Omit it for a construction solid
        used only as a Boolean operand.
    name : str, optional
        Optional label.
    """

    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    major_radius: float = 1.0
    minor_radius: float = 0.25
    axis: "str | tuple[float, float, float]" = "z"

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        from magnelio.geo._occ_backend import make_torus

        return make_torus(self.center, self.major_radius, self.minor_radius, self.axis, scale=scale)

    def _analytic_bbox(self):
        from magnelio.geo._scaling import pad_box

        return pad_box((self.center, self.center), self.major_radius + self.minor_radius)
