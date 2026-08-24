"""
Shape modifications: chamfer, fillet, extrude, loft.

These functions modify or derive shapes from existing geometry.
Chamfer/fillet use edge selection; extrude/loft use face selection
— both based on point proximity (``face_near``/``near``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from magnelio.geo._cache import cached_occ_shape
from magnelio.geo._sheet import PlanarSheet
from magnelio.geo._validate import finite, nonzero, point3, positive, vector3
from magnelio.geo.shape import Shape
from magnelio.materials.material import resolve_material


def _check_edge_selector(verb, near, face_near, edges):
    """Reject an edge selection that names zero or several modes.

    The kernel-side selector checks this too, but only when the solid is
    finally built — long after the call that got it wrong, and typically
    inside a plot or a mesh build.
    """
    modes = sum(x is not None for x in (near, face_near, edges))
    if modes != 1:
        given = [
            name
            for name, value in (("near", near), ("face_near", face_near), ("edges", edges))
            if value is not None
        ]
        got = f"got {', '.join(given)}" if given else "got none of them"
        raise ValueError(
            f"{verb}() takes exactly one of near=, face_near= or edges= to "
            f"select the edges to work on; {got}."
        )
    if edges is not None and edges != "all":
        raise ValueError(f"{verb}(edges=...) takes only 'all'; got {edges!r}.")


def chamfer(shape, *, near=None, face_near=None, edges=None, distance):
    """Implementation of :meth:`magnelio.geo.Shape.chamfered`.

    Exactly one of *near*, *face_near* or *edges* must be specified.
    """
    _check_edge_selector("chamfered", near, face_near, edges)
    if isinstance(distance, (tuple, list)):
        if len(distance) != 2:
            raise ValueError(
                f"chamfered(distance=...) takes a single value or a "
                f"(d1, d2) pair; got {len(distance)} values."
            )
        distance = tuple(positive(d, "chamfered(distance)") for d in distance)
    else:
        distance = positive(distance, "chamfered(distance)")
    return _ChamferedShape(shape, near, face_near, edges, distance)


def fillet(shape, *, near=None, face_near=None, edges=None, radius):
    """Implementation of :meth:`magnelio.geo.Shape.filleted`.

    Exactly one of *near*, *face_near* or *edges* must be specified.
    """
    _check_edge_selector("filleted", near, face_near, edges)
    radius = positive(radius, "filleted(radius)")
    return _FilletedShape(shape, near, face_near, edges, radius)


@dataclass
class _ChamferedShape(Shape):
    _inner: object
    _near: object
    _face_near: object
    _edges: object
    _distance: object

    @property
    def material(self):
        return self._inner.material

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        from magnelio.geo._occ_backend import make_chamfer, resolve_edges  # noqa: PLC0415

        occ_shape = self._inner._occ_shape(scale)
        selected = resolve_edges(
            occ_shape,
            near=self._near,
            face_near=self._face_near,
            edges=self._edges,
            scale=scale,
        )
        return make_chamfer(occ_shape, selected, self._distance, scale=scale)

    def _analytic_bbox(self):
        return self._inner._analytic_bbox()


def extrude(shape, *, vector, face_near=None, material=None):
    """Implementation of :meth:`magnelio.geo.Shape.extruded`.

    Uses ``BRepPrimAPI_MakePrism``.
    """
    from magnelio.geo._sheet import PlanarSheet  # noqa: PLC0415

    material = resolve_material(material, "extruded(material=...)")
    if isinstance(shape, PlanarSheet):
        if material is None and shape.material is None:
            raise ValueError(
                "Extruding a construction profile (material=None) requires "
                "an explicit material= for the resulting solid."
            )
    elif face_near is None:
        raise ValueError(
            "extrude() on a solid requires face_near= to select the face "
            "to extrude (only a standalone planar sheet may omit it)."
        )
    vector = vector3(vector, "extruded(vector)", nonzero=True)
    return _ExtrudedFaceShape(shape, face_near, vector, material)


def cover(curve, *, material=None, name=None):
    """Implementation of :meth:`magnelio.geo.Curve.covered`.

    Uses ``BRepBuilderAPI_MakeFace`` on the curve's wire.
    """
    material = resolve_material(material, "covered(material=...)")
    if curve._ends is not None and not curve.is_closed:
        gap = math.dist(curve._ends[0], curve._ends[1])
        raise ValueError(
            f"covered() needs a closed curve, but this one ends {gap:.3e} m "
            f"away from where it starts.  Chain the segments with joined() "
            f"so the last end meets the first start, or build the profile "
            f"with Path(...).closed()."
        )
    return _CoveredSheet(curve, material, name)


@dataclass
class _CoveredSheet(PlanarSheet):
    _curve: object
    material: object = None
    name: str | None = None

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        from magnelio.geo._occ_backend import make_wire_face  # noqa: PLC0415

        return make_wire_face(self._curve._occ_shape(scale))

    def _analytic_bbox(self):
        # Exact: a planar face lies inside the convex hull of its own
        # boundary, whose AABB is the boundary's AABB.
        return self._curve._analytic_bbox()


def trace(curve, *, width, thickness, caps="round", normal=None, material=None, name=None):
    """Implementation of :meth:`magnelio.geo.Curve.traced`.

    Offsets the centreline within its plane, then extrudes the outline.
    """
    material = resolve_material(material, "traced(material=...)")
    width = positive(width, "traced(width)")
    thickness = nonzero(thickness, "traced(thickness)")
    if caps not in ("round", "flat"):
        raise ValueError(f"caps must be 'round' or 'flat'; got {caps!r}")
    return _TracedCurveShape(curve, width, thickness, caps, normal, material, name)


@dataclass
class _TracedCurveShape(Shape):
    _curve: object
    _width: float
    _thickness: float
    _caps: str
    _normal: object
    material: object = None
    name: str | None = None

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        from magnelio.geo._axes import normalize_axis  # noqa: PLC0415
        from magnelio.geo._occ_backend import make_trace  # noqa: PLC0415

        return make_trace(
            self._curve._occ_shape(scale),
            self._width,
            self._thickness,
            caps=self._caps,
            normal=None if self._normal is None else normalize_axis(self._normal),
            scale=scale,
        )

    def _analytic_bbox(self):
        from magnelio.geo._scaling import pad_box  # noqa: PLC0415

        # The plane normal is unknown without the kernel, so pad by the
        # full half-width plus thickness in every direction.
        return pad_box(self._curve._analytic_bbox(), 0.5 * self._width + abs(self._thickness))


#: Blend modes shared by :func:`loft` and :class:`Loft` (DD-144).
#: ``"tangent"`` needs face normals to aim at and is therefore only open
#: to the former.  With two sections ``"spline"`` and ``"ruled"`` build
#: the same surface — the mode only starts to matter from three on.
_BLEND_MODES = ("spline", "ruled", "tangent")
#: Hermite's own convention: interior control points one third of the
#: way along.  Past ~0.7 the blend overshoots into a bulge (DD-144).
_DEFAULT_TENSION = 1.0 / 3.0


def _check_blend(blend, *, allow_tangent):
    """Validate a blend mode and say what the alternatives are."""
    allowed = _BLEND_MODES if allow_tangent else tuple(m for m in _BLEND_MODES if m != "tangent")
    if blend not in allowed:
        options = ", ".join(repr(mode) for mode in allowed)
        extra = ""
        if blend == "tangent":
            extra = (
                "  'tangent' aims at the normals of two faces, which free "
                "cross-sections do not have — use Shape.lofted() for it."
            )
        raise ValueError(f"blend must be one of {options}; got {blend!r}.{extra}")


def _check_tension(tension, *, blend):
    """Normalise *tension* to a ``(t_a, t_b)`` pair of floats."""
    if blend != "tangent":
        if tension is not None:
            raise ValueError(
                f"tension= shapes the tangent blend's spine and has no "
                f"meaning for blend={blend!r}; pass blend='tangent' or drop it."
            )
        return None
    if tension is None:
        tension = _DEFAULT_TENSION
    values = tension if isinstance(tension, (tuple, list)) else (tension, tension)
    if len(values) != 2:
        raise ValueError(
            f"tension must be a single value or a (start, end) pair; got {len(values)} values."
        )
    values = tuple(float(v) for v in values)
    if any(v <= 0.0 for v in values):
        raise ValueError(f"tension must be positive; got {values}.")
    return values


def loft(
    shape_a, face_near_a, shape_b, face_near_b, *, material=None, blend="spline", tension=None
):
    """Implementation of :meth:`magnelio.geo.Shape.lofted`.

    Extracts the outer wire of each selected face and bridges them with
    ``BRepOffsetAPI_ThruSections`` (``"spline"``/``"ruled"``) or, for
    ``"tangent"``, sweeps one wire into the other along a Bezier spine
    with ``BRepOffsetAPI_MakePipeShell``.
    """
    material = resolve_material(material, "lofted(material=...)")
    _check_blend(blend, allow_tangent=True)
    tension = _check_tension(tension, blend=blend)
    return _LoftedShape(shape_a, face_near_a, shape_b, face_near_b, material, blend, tension)


def revolve(profile, *, axis, angle_deg=360.0, origin=(0.0, 0.0, 0.0), material=None):
    """Implementation of :meth:`magnelio.geo.Shape.revolved`.

    Uses ``BRepPrimAPI_MakeRevol``.
    """
    from magnelio.geo._axes import normalize_axis  # noqa: PLC0415
    from magnelio.geo._sheet import PlanarSheet  # noqa: PLC0415

    material = resolve_material(material, "revolved(material=...)")
    if isinstance(profile, PlanarSheet) and material is None and profile.material is None:
        raise ValueError(
            "Revolving a construction profile (material=None) requires an "
            "explicit material= for the resulting solid."
        )
    normalize_axis(axis, "revolved(axis)")
    angle_deg = finite(angle_deg, "revolved(angle_deg)")
    if not -360.0 <= angle_deg <= 360.0 or angle_deg == 0.0:
        raise ValueError(
            f"revolved(angle_deg=...) must be a non-zero sweep of at most a "
            f"full turn; got {angle_deg}."
        )
    origin = point3(origin, "revolved(origin)")
    return _RevolvedShape(profile, axis, angle_deg, origin, material)


def shell(shape, *, thickness, opening_face_near=None):
    """Implementation of :meth:`magnelio.geo.Shape.shelled`.

    Uses ``BRepOffsetAPI_MakeThickSolid`` with an inward offset.
    """
    if isinstance(shape, PlanarSheet):
        raise TypeError(
            "shelled() hollows a solid, but this is a planar sheet. To "
            "grow a sheet into a solid slab use thickened()."
        )
    thickness = positive(thickness, "shelled(thickness)")
    return _ShelledShape(shape, thickness, opening_face_near)


def thicken(sheet, *, thickness, direction="forward", material=None):
    """Implementation of :meth:`magnelio.geo.Shape.thickened`.

    A prism along the sheet's own plane normal.
    """
    if not isinstance(sheet, PlanarSheet):
        raise TypeError(
            f"thickened() grows a planar sheet into a solid, but this is a "
            f"{type(sheet).__name__}. To hollow a solid use shelled()."
        )
    material = resolve_material(material, "thickened(material=...)")
    thickness = positive(thickness, "thickened(thickness)")
    if direction not in ("forward", "backward", "symmetric"):
        raise ValueError(
            f"direction must be 'forward', 'backward' or 'symmetric'; got {direction!r}"
        )
    if material is None and sheet.material is None:
        raise ValueError(
            "Thickening a construction profile (material=None) requires an "
            "explicit material= for the resulting solid."
        )
    return _ThickenedSheet(sheet, thickness, direction, material)


@dataclass
class _ShelledShape(Shape):
    _inner: object
    _thickness: float
    _opening_face_near: object

    @property
    def material(self):
        return self._inner.material

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        from magnelio.geo._occ_backend import make_thick_solid, resolve_faces  # noqa: PLC0415

        occ_shape = self._inner._occ_shape(scale)
        openings = resolve_faces(occ_shape, self._opening_face_near, scale=scale)
        return make_thick_solid(occ_shape, openings, self._thickness, scale=scale)

    def _analytic_bbox(self):
        # Exact: hollowing keeps the outer surface where it was.
        return self._inner._analytic_bbox()


@dataclass
class _ThickenedSheet(Shape):
    _inner: object
    _thickness: float
    _direction: str
    _material: object

    @property
    def material(self):
        if self._material is not None:
            return self._material
        return self._inner.material

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        from magnelio.geo._occ_backend import (  # noqa: PLC0415
            face_plane_normal,
            make_extrude,
            occ_translate,
        )

        face = self._inner._occ_shape(scale)
        normal = face_plane_normal(face)
        if self._direction == "backward":
            normal = tuple(-c for c in normal)
        if self._direction == "symmetric":
            face = occ_translate(
                face, tuple(-0.5 * self._thickness * c for c in normal), scale=scale
            )
        return make_extrude(face, tuple(self._thickness * c for c in normal), scale=scale)

    def _analytic_bbox(self):
        from magnelio.geo._scaling import pad_box  # noqa: PLC0415

        # The plane normal is not known without the kernel, so pad in
        # every direction — conservative for any orientation.
        return pad_box(self._inner._analytic_bbox(), self._thickness)


def sweep(profile, spine, *, material=None):
    """Implementation of :meth:`magnelio.geo.Shape.swept`.

    Uses ``BRepOffsetAPI_MakePipe``, orienting the result by the pipe
    trihedron.
    """
    from magnelio.geo.curves import Curve  # noqa: PLC0415

    material = resolve_material(material, "swept(material=...)")
    if material is None and getattr(profile, "material", None) is None:
        raise ValueError(
            "Sweeping a construction profile (material=None) requires an "
            "explicit material= for the resulting solid."
        )
    if not isinstance(spine, Curve):
        raise TypeError(
            f"swept() needs a Curve as its spine, but got a "
            f"{type(spine).__name__}. Build the path with Curve.polyline / "
            f"Curve.arc / Curve.spline / Curve.helix, or draw it with Path."
        )
    return _SweptShape(profile, spine, material)


@dataclass
class _ExtrudedFaceShape(Shape):
    _inner: object
    _face_near: object
    _vector: object
    _material: object

    @property
    def material(self):
        if self._material is not None:
            return self._material
        return self._inner.material

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        from magnelio.geo._occ_backend import (  # noqa: PLC0415
            find_nearest_face,
            make_extrude,
        )
        from magnelio.geo._sheet import PlanarSheet  # noqa: PLC0415

        if isinstance(self._inner, PlanarSheet):
            # A standalone sheet _is_ the profile — no face selection needed.
            face = self._inner._occ_shape(scale)
        else:
            occ_shape = self._inner._occ_shape(scale)
            face = find_nearest_face(occ_shape, self._face_near, scale=scale)
        return make_extrude(face, self._vector, scale=scale)

    def _analytic_bbox(self):
        from magnelio.geo._scaling import translate_box, union_boxes  # noqa: PLC0415

        base = self._inner._analytic_bbox()
        return union_boxes([base, translate_box(base, self._vector)])


@dataclass
class _LoftedShape(Shape):
    _shape_a: object
    _face_near_a: object
    _shape_b: object
    _face_near_b: object
    _material: object
    _blend: str
    _tension: object

    @property
    def material(self):
        if self._material is not None:
            return self._material
        return self._shape_a.material

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        from magnelio.geo._occ_backend import (  # noqa: PLC0415
            extract_face_wire,
            find_nearest_face,
            make_loft,
            make_tangent_blend,
        )

        face_a = find_nearest_face(self._shape_a._occ_shape(scale), self._face_near_a, scale=scale)
        face_b = find_nearest_face(self._shape_b._occ_shape(scale), self._face_near_b, scale=scale)
        if self._blend == "tangent":
            # The spine is derived from the faces themselves, so it comes
            # out in whatever unit they carry — no rescaling needed here.
            return make_tangent_blend(face_a, face_b, self._tension)
        wire_a = extract_face_wire(face_a)
        wire_b = extract_face_wire(face_b)
        return make_loft([wire_a, wire_b], is_solid=True, is_ruled=self._blend == "ruled")

    def _analytic_bbox(self):
        from magnelio.geo._scaling import box_diagonal, pad_box, union_boxes  # noqa: PLC0415

        # A smooth or tangent loft may overshoot the hull of its two
        # profiles; pad generously — only the order of magnitude matters.
        # A tangent blend bows out along the normals, so its reach grows
        # with the tension and the padding has to follow.
        box = union_boxes([self._shape_a._analytic_bbox(), self._shape_b._analytic_bbox()])
        slack = 0.25
        if self._blend == "tangent":
            slack = max(slack, 1.5 * max(self._tension))
        return pad_box(box, slack * box_diagonal(box))


@dataclass
class Loft(Shape):
    """A solid interpolating an ordered series of cross-sections.

    The way to build a transition no primitive covers: a horn flaring
    from a waveguide mouth to a wider aperture, a taper from a round
    cross-section to a square one, a matching section that steps through
    several intermediate outlines.  Where
    :meth:`~magnelio.geo.Shape.lofted` bridges one face of a solid to a
    face of another, ``Loft`` takes the profiles themselves and as many
    of them as the shape needs.

    Parameters
    ----------
    *sections : Face, covered Curve, or closed Curve
        At least two cross-sections, in the order the solid passes
        through them.  Sheets contribute their outer boundary.  The
        sections should wind the same way — a reversed one produces a
        twisted, self-intersecting solid rather than an error.
    blend : {'spline', 'ruled'}
        How consecutive sections are joined.  ``'spline'`` (default)
        passes one smooth surface through all of them; ``'ruled'`` joins
        them with straight surfaces, so the solid is a stack of frusta.
    material : Material, optional
        Material of the lofted solid.  ``None`` (default) makes it a
        construction body, since cross-sections carry no volume material
        to inherit.
    name : str, optional
        Optional label.

    Raises
    ------
    TypeError
        If a section is a solid (use :meth:`~magnelio.geo.Shape.lofted`)
        or a :class:`~magnelio.geo.Group`.
    ValueError
        If fewer than two sections are given, a Curve section is not
        closed, or *blend* is not one of the two modes above.

    Examples
    --------
    A horn flaring from a square throat to a wider square mouth::

        throat = geo.Face(normal="z", points=[...], position=0.0)
        mouth = geo.Face(normal="z", points=[...], position=60e-3)
        horn = geo.Loft(throat, mouth, blend="ruled", material=pec)
    """

    sections: tuple
    blend: str = "spline"
    material: "object | None" = None
    name: str | None = None

    def __init__(self, *sections, blend="spline", material=None, name=None):
        from magnelio.geo.curves import Curve  # noqa: PLC0415
        from magnelio.geo.operations import _reject_group  # noqa: PLC0415

        _reject_group(sections, "Loft")
        _check_blend(blend, allow_tangent=False)
        if len(sections) < 2:
            raise ValueError(f"A Loft needs at least 2 cross-sections; got {len(sections)}.")
        for index, section in enumerate(sections):
            if isinstance(section, Curve):
                if section._ends is not None and not section.is_closed:
                    raise ValueError(
                        f"Loft section {index} is an open curve.  Every "
                        f"cross-section must be a closed outline — chain it "
                        f"with joined() or draw it with Path(...).closed()."
                    )
            elif not isinstance(section, PlanarSheet):
                raise TypeError(
                    f"Loft section {index} is a {type(section).__name__}. "
                    f"Sections must be planar sheets (a Face or a covered "
                    f"Curve) or closed curves; to loft between faces of two "
                    f"existing solids use the lofted() verb instead."
                )
        self.sections = sections
        self.blend = blend
        self.material = resolve_material(material, "Loft(material=...)")
        self.name = name

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        from magnelio.geo._occ_backend import extract_face_wire, make_loft  # noqa: PLC0415
        from magnelio.geo.curves import Curve  # noqa: PLC0415

        wires = [
            s._occ_shape(scale) if isinstance(s, Curve) else extract_face_wire(s._occ_shape(scale))
            for s in self.sections
        ]
        return make_loft(wires, is_solid=True, is_ruled=self.blend == "ruled")

    def _analytic_bbox(self):
        from magnelio.geo._scaling import box_diagonal, pad_box, union_boxes  # noqa: PLC0415

        box = union_boxes([s._analytic_bbox() for s in self.sections])
        if self.blend == "ruled":
            # Exact: every ruled point is a convex combination of section
            # points, and a box containing all sections contains those.
            return box
        return pad_box(box, 0.25 * box_diagonal(box))


@dataclass
class _FilletedShape(Shape):
    _inner: object
    _near: object
    _face_near: object
    _edges: object
    _radius: object

    @property
    def material(self):
        return self._inner.material

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        from magnelio.geo._occ_backend import make_fillet, resolve_edges  # noqa: PLC0415

        occ_shape = self._inner._occ_shape(scale)
        selected = resolve_edges(
            occ_shape,
            near=self._near,
            face_near=self._face_near,
            edges=self._edges,
            scale=scale,
        )
        return make_fillet(occ_shape, selected, self._radius, scale=scale)

    def _analytic_bbox(self):
        return self._inner._analytic_bbox()


_AXIS_VECTORS = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


@dataclass
class _RevolvedShape(Shape):
    _profile: object
    _axis: object
    _angle_deg: float
    _origin: object
    _material: object

    @property
    def material(self):
        if self._material is not None:
            return self._material
        return self._profile.material

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        import math  # noqa: PLC0415

        from magnelio.geo._axes import normalize_axis  # noqa: PLC0415
        from magnelio.geo._occ_backend import make_revolve  # noqa: PLC0415

        return make_revolve(
            self._profile._occ_shape(scale),
            self._origin,
            normalize_axis(self._axis),
            math.radians(self._angle_deg),
            scale=scale,
        )

    def _analytic_bbox(self):
        from magnelio.geo._axes import normalize_axis  # noqa: PLC0415
        from magnelio.geo._scaling import (  # noqa: PLC0415
            box_of_points,
            corners_of_box,
            pad_box,
        )

        # Every revolved point stays within r_max of the axis segment
        # spanned by the profile's axial projections.
        axis = normalize_axis(self._axis)
        corners = corners_of_box(self._profile._analytic_bbox())
        t_vals, r_max = [], 0.0
        for p in corners:
            rel = tuple(c - o for c, o in zip(p, self._origin))
            t = sum(r * a for r, a in zip(rel, axis))
            perp = tuple(r - t * a for r, a in zip(rel, axis))
            t_vals.append(t)
            r_max = max(r_max, sum(c * c for c in perp) ** 0.5)
        ends = [
            tuple(o + t * a for o, a in zip(self._origin, axis)) for t in (min(t_vals), max(t_vals))
        ]
        return pad_box(box_of_points(ends), r_max)


@dataclass
class _SweptShape(Shape):
    _profile: object
    _spine: object
    _material: object

    @property
    def material(self):
        if self._material is not None:
            return self._material
        return self._profile.material

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        from magnelio.geo._occ_backend import make_sweep  # noqa: PLC0415

        return make_sweep(self._profile._occ_shape(scale), self._spine._occ_shape(scale))

    def _analytic_bbox(self):
        from magnelio.geo._scaling import box_diagonal, pad_box  # noqa: PLC0415

        # The profile is re-positioned onto the spine start, so its
        # absolute location is irrelevant — pad the spine box by the
        # profile's full diagonal (conservative for any orientation).
        return pad_box(
            self._spine._analytic_bbox(),
            box_diagonal(self._profile._analytic_bbox()),
        )
