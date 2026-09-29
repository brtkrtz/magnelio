"""
Shape modifications: chamfer, fillet, extrude, loft.

These functions modify or derive shapes from existing geometry.
Chamfer/fillet use edge selection; extrude/loft use face selection
— owned references are the core; point selectors delegate to semantic selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from magnelio.geo._cache import cached_occ_shape
from magnelio.geo._sheet import Profile, Sheet
from magnelio.geo._topology_history import finish
from magnelio.geo._validate import finite, nonzero, point3, positive, vector3
from magnelio.geo.shape import Solid
from magnelio.materials.material import resolve_material


def _require_solid(shape, verb):
    if not isinstance(shape, Solid):
        raise TypeError(f"{verb}() requires a Solid; got {type(shape).__name__}.")


def _profile_input(shape, verb, *, planar=False):
    from magnelio.geo.topology import FaceRef

    if isinstance(shape, FaceRef):
        shape = shape.detached()
    if not isinstance(shape, Sheet):
        hint = (
            " Use shelled() for a Solid."
            if verb == "thickened"
            else " Select a face and use lofted()."
            if verb == "Loft"
            else ""
        )
        raise TypeError(
            f"{verb}() requires a Profile, eligible Sheet or FaceRef; "
            f"got {type(shape).__name__}.{hint}"
        )
    if planar and not isinstance(shape, Profile):
        from magnelio.geo.topology import _is_planar, _scale

        scale = _scale(shape)
        raw = shape._occ_shape(scale)
        if not _is_planar(raw):
            raise ValueError(f"{verb}() requires a planar Sheet or FaceRef.")
        shape = FaceRef(shape, raw, scale).detached()
    return shape


def _refs(shape, value, classes, verb):
    from magnelio.geo.topology import EdgeSetRef, FaceSetRef

    if isinstance(value, classes):
        refs = tuple(value) if isinstance(value, (EdgeSetRef, FaceSetRef)) else (value,)
    elif isinstance(value, (tuple, list)):
        refs = tuple(r for item in value for r in _refs(shape, item, classes, verb))
    else:
        accepted = ", ".join(cls.__name__ for cls in classes)
        raise TypeError(f"{verb}() selection requires {accepted}; got {type(value).__name__}.")
    if not refs:
        raise ValueError(f"{verb}() selection must not be empty.")
    if any(ref.owner is not shape for ref in refs):
        raise ValueError(f"{verb}() references must belong to this exact Solid owner.")
    if len({ref._scale for ref in refs}) != 1:
        raise ValueError(f"{verb}() references must share a model scale.")
    return refs


def _points(value, verb):
    try:
        return (point3(value, verb),)
    except (TypeError, ValueError) as error:
        try:
            points = tuple(value)
        except TypeError:
            raise error from None
        if not points:
            raise ValueError(f"{verb} requires a non-empty sequence of world points.")
        return tuple(point3(p, verb) for p in points)


def _edge_selection(shape, verb, near, face_near, edges, faces):
    from magnelio.geo.topology import EdgeRef, EdgeSetRef, FaceRef, FaceSetRef

    _require_solid(shape, verb)
    modes = sum(x is not None for x in (near, face_near, edges, faces))
    if modes != 1:
        raise ValueError(
            f"{verb}() takes exactly one of near=, face_near= or edges= "
            "(or faces=) to select the edges to work on."
        )
    if near is not None:
        return tuple(shape.edge(near=p) for p in _points(near, f"{verb}(near)"))
    if face_near is not None:
        faces = shape.face(near=point3(face_near, f"{verb}(face_near)"))
    if faces is not None:
        return tuple(e for f in _refs(shape, faces, (FaceRef, FaceSetRef), verb) for e in f.edges)
    if isinstance(edges, str):
        if edges != "all":
            raise ValueError(f"{verb}(edges=...) takes only 'all' as a string; got {edges!r}.")
        return tuple(shape.edges())
    return _refs(shape, edges, (EdgeRef, EdgeSetRef), verb)


def _selected_edges(refs):
    if refs:
        from magnelio.geo._topology_history import _unique

        return _unique([ref._shape for ref in refs])
    return []


def chamfer(shape, *, near=None, face_near=None, edges=None, faces=None, distance):
    """Implementation of :meth:`magnelio.geo.Shape.chamfered`.

    Exactly one of *near*, *face_near* or *edges* must be specified.
    """
    if isinstance(distance, (tuple, list)):
        if len(distance) != 2:
            raise ValueError(
                f"chamfered(distance=...) takes a single value or a "
                f"(d1, d2) pair; got {len(distance)} values."
            )
        distance = tuple(positive(d, "chamfered(distance)") for d in distance)
    else:
        distance = positive(distance, "chamfered(distance)")
    edges = _edge_selection(shape, "chamfered", near, face_near, edges, faces)
    return finish(_ChamferedShape(shape, None, None, edges, distance))


def fillet(shape, *, near=None, face_near=None, edges=None, faces=None, radius):
    """Implementation of :meth:`magnelio.geo.Shape.filleted`.

    Exactly one of *near*, *face_near* or *edges* must be specified.
    """
    radius = positive(radius, "filleted(radius)")
    edges = _edge_selection(shape, "filleted", near, face_near, edges, faces)
    return finish(_FilletedShape(shape, None, None, edges, radius))


@dataclass
class _ChamferedShape(Solid):
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
        selected = (
            _selected_edges(self._edges)
            if not isinstance(self._edges, str) and self._edges is not None
            else resolve_edges(
                occ_shape,
                near=self._near,
                face_near=self._face_near,
                edges=self._edges,
                scale=scale,
            )
        )
        return make_chamfer(occ_shape, selected, self._distance, scale=scale)

    def _analytic_bbox(self):
        return self._inner._analytic_bbox()


def extrude(shape, *, vector, face_near=None, material=None):
    """Implementation of :meth:`magnelio.geo.Shape.extruded`.

    Uses ``BRepPrimAPI_MakePrism``.
    """
    material = resolve_material(material, "extruded(material=...)")
    if isinstance(shape, Solid):
        if face_near is None:
            raise ValueError("extruded() on a Solid requires face_near= or use a FaceRef.")
        selected = shape.face(near=point3(face_near, "extruded(face_near)"))
        vector = vector3(vector, "extruded(vector)", nonzero=True)
        return finish(_ExtrudedFaceShape(shape, selected, vector, material))
    elif face_near is not None:
        raise ValueError("extruded(face_near=) applies only to a Solid.")
    shape = _profile_input(shape, "extruded")
    face_near = None
    vector = vector3(vector, "extruded(vector)", nonzero=True)
    return finish(_ExtrudedFaceShape(shape, face_near, vector, material))


def trace(curve, *, width, thickness, caps="round", normal=None, material=None, name=None):
    """Implementation of :meth:`magnelio.geo.Curve.traced`.

    Offsets the centreline within its plane, then extrudes the outline.
    """
    material = resolve_material(material, "traced(material=...)")
    width = positive(width, "traced(width)")
    thickness = nonzero(thickness, "traced(thickness)")
    if caps not in ("round", "flat"):
        raise ValueError(f"caps must be 'round' or 'flat'; got {caps!r}")
    return finish(_TracedCurveShape(curve, width, thickness, caps, normal, material, name))


@dataclass
class _TracedCurveShape(Solid):
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
    values = tuple(positive(v, "tension") for v in values)
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
    _require_solid(shape_a, "lofted")
    _require_solid(shape_b, "lofted")
    a = shape_a.face(near=point3(face_near_a, "lofted(face_near)"))
    b = shape_b.face(near=point3(face_near_b, "lofted(other_face_near)"))
    sections = [_profile_input(ref, "lofted", planar=True) for ref in (a, b)]
    if len(_boundaries(sections[0])) != len(_boundaries(sections[1])):
        raise ValueError("lofted() profiles must have the same number of holes.")
    return finish(_LoftedShape(shape_a, a, shape_b, b, material, blend, tension))


def revolve(profile, *, axis, angle_deg=360.0, origin=(0.0, 0.0, 0.0), material=None):
    """Implementation of :meth:`magnelio.geo.Shape.revolved`.

    Uses ``BRepPrimAPI_MakeRevol``.
    """
    from magnelio.geo._axes import normalize_axis  # noqa: PLC0415

    material = resolve_material(material, "revolved(material=...)")
    profile = _profile_input(profile, "revolved", planar=True)
    normalize_axis(axis, "revolved(axis)")
    angle_deg = finite(angle_deg, "revolved(angle_deg)")
    if not -360.0 <= angle_deg <= 360.0 or angle_deg == 0.0:
        raise ValueError(
            f"revolved(angle_deg=...) must be a non-zero sweep of at most a "
            f"full turn; got {angle_deg}."
        )
    origin = point3(origin, "revolved(origin)")
    return finish(_RevolvedShape(profile, axis, angle_deg, origin, material))


def shell(shape, *, thickness, opening_face_near=None, openings=None):
    """Implementation of :meth:`magnelio.geo.Shape.shelled`.

    Uses ``BRepOffsetAPI_MakeThickSolid`` with an inward offset.
    """
    from magnelio.geo.topology import FaceRef, FaceSetRef

    if not isinstance(shape, Solid):
        raise TypeError("shelled() requires a Solid; grow a Sheet with thickened().")
    if opening_face_near is not None and openings is not None:
        raise ValueError("shelled() takes either openings= or opening_face_near=.")
    if opening_face_near is not None:
        openings = tuple(
            shape.face(near=p) for p in _points(opening_face_near, "shelled(opening_face_near)")
        )
    openings = (
        None if openings is None else _refs(shape, openings, (FaceRef, FaceSetRef), "shelled")
    )
    thickness = positive(thickness, "shelled(thickness)")
    return finish(_ShelledShape(shape, thickness, openings))


def thicken(sheet, *, thickness, direction="forward", material=None):
    """Implementation of :meth:`magnelio.geo.Shape.thickened`.

    A prism along the sheet's own plane normal for a planar sheet, a
    normal offset for a curved one.
    """
    sheet = _profile_input(sheet, "thickened")
    material = resolve_material(material, "thickened(material=...)")
    thickness = positive(thickness, "thickened(thickness)")
    if direction not in ("forward", "backward", "symmetric"):
        raise ValueError(
            f"direction must be 'forward', 'backward' or 'symmetric'; got {direction!r}"
        )
    if not isinstance(sheet, Profile):
        from magnelio.geo.topology import FaceRef, _is_planar, _scale

        scale = _scale(sheet)
        raw = sheet._occ_shape(scale)
        if _is_planar(raw):
            sheet = FaceRef(sheet, raw, scale).detached()
        elif direction == "symmetric":
            raise ValueError("thickened(direction='symmetric') requires a planar Sheet or FaceRef.")
    return finish(_ThickenedSheet(sheet, thickness, direction, material))


@dataclass
class _ShelledShape(Solid):
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
        openings = (
            _selected_edges(self._opening_face_near)
            if self._opening_face_near and not isinstance(self._opening_face_near[0], (tuple, list))
            else resolve_faces(occ_shape, self._opening_face_near, scale=scale)
        )
        return make_thick_solid(occ_shape, openings, self._thickness, scale=scale)

    def _analytic_bbox(self):
        # Exact: hollowing keeps the outer surface where it was.
        return self._inner._analytic_bbox()


@dataclass
class _ThickenedSheet(Solid):
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
            is_planar_face,
            make_extrude,
            make_thick_face,
            occ_translate,
        )

        face = self._inner._occ_shape(scale)
        if not is_planar_face(face):
            return make_thick_face(face, self._thickness, self._direction, scale=scale)
        from magnelio.geo.topology import _normal

        normal = _normal(face)
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


def sweep(
    profile,
    spine,
    *,
    face_near=None,
    material=None,
    frame="corrected_frenet",
    binormal=None,
    twist_deg=0.0,
    draft_deg=0.0,
    tolerance=None,
):
    """Implementation of :meth:`magnelio.geo.Shape.swept`.

    Zero laws retain the pipe builders; nonzero laws fit arc-length sections.
    """
    from magnelio.geo.curves import Curve  # noqa: PLC0415

    twist_deg = finite(twist_deg, "swept(twist_deg)")
    draft_deg = finite(draft_deg, "swept(draft_deg)")
    if abs(draft_deg) >= 90:
        raise ValueError("swept(draft_deg) must be strictly between -90 and 90 degrees.")
    if tolerance is not None:
        tolerance = positive(tolerance, "swept(tolerance)")
    modes = ("corrected_frenet", "frenet", "fixed", "fixed_binormal")
    if frame not in modes:
        raise ValueError(f"swept(frame=...) must be one of {modes}; got {frame!r}.")
    if frame == "fixed_binormal":
        if binormal is None:
            raise ValueError("swept(frame='fixed_binormal') requires binormal=.")
        from magnelio.geo._axes import normalize_axis

        if not isinstance(binormal, str):
            binormal = vector3(binormal, "swept(binormal)", nonzero=True)
            magnitude = max(abs(component) for component in binormal)
            binormal = tuple(component / magnitude for component in binormal)
        binormal = normalize_axis(binormal, "swept(binormal)")
    elif binormal is not None:
        raise ValueError("swept(binormal=...) requires frame='fixed_binormal'.")
    material = resolve_material(material, "swept(material=...)")
    if isinstance(profile, Solid):
        if face_near is None:
            raise ValueError("swept() on a Solid requires face_near= or use a FaceRef.")
        profile = profile.face(near=point3(face_near, "swept(face_near)"))
    elif face_near is not None:
        raise ValueError("swept(face_near=) applies only to a Solid.")
    profile = _profile_input(profile, "swept", planar=True)
    if not isinstance(spine, Curve):
        raise TypeError(
            f"swept() needs a Curve as its spine, but got a "
            f"{type(spine).__name__}. Build the path with Curve.polyline / "
            f"Curve.arc / Curve.spline / Curve.helix, or draw it with Path."
        )
    if frame == "fixed_binormal":
        from OCC.Core.BRepAdaptor import BRepAdaptor_CompCurve
        from OCC.Core.gp import gp_Pnt, gp_Vec

        from magnelio.geo.topology import _scale

        curve = BRepAdaptor_CompCurve(spine._occ_shape(_scale(spine)))
        tangent = gp_Vec()
        curve.D1(curve.FirstParameter(), gp_Pnt(), tangent)
        if tangent.Magnitude() <= 1e-30:
            raise ValueError("swept() requires a non-zero initial spine tangent.")
        direction = gp_Vec(*binormal)
        if tangent.Crossed(direction).Magnitude() <= 1e-12 * tangent.Magnitude():
            raise ValueError("swept(binormal=...) must not be parallel to the spine tangent.")
    return finish(
        _SweptShape(profile, spine, material, frame, binormal, twist_deg, draft_deg, tolerance)
    )


@dataclass
class _ExtrudedFaceShape(Solid):
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
        from magnelio.geo._sheet import Sheet  # noqa: PLC0415

        if isinstance(self._inner, Sheet):
            # A standalone sheet _is_ the profile — no face selection needed.
            face = self._inner._occ_shape(scale)
        else:
            occ_shape = self._inner._occ_shape(scale)
            face = (
                self._face_near._shape
                if hasattr(self._face_near, "owner")
                else find_nearest_face(occ_shape, self._face_near, scale=scale)
            )
        return make_extrude(face, self._vector, scale=scale)

    def _analytic_bbox(self):
        from magnelio.geo._scaling import translate_box, union_boxes  # noqa: PLC0415

        base = self._inner._analytic_bbox()
        return union_boxes([base, translate_box(base, self._vector)])


@dataclass
class _LoftedShape(Solid):
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
            make_tangent_blend,
        )

        face_a = (
            self._face_near_a._shape
            if hasattr(self._face_near_a, "owner")
            else find_nearest_face(self._shape_a._occ_shape(scale), self._face_near_a, scale=scale)
        )
        face_b = (
            self._face_near_b._shape
            if hasattr(self._face_near_b, "owner")
            else find_nearest_face(self._shape_b._occ_shape(scale), self._face_near_b, scale=scale)
        )
        if hasattr(self._face_near_b, "owner") and self._face_near_b._scale != scale:
            from magnelio.geo.topology import _cast, _rescale_copy

            face_b = _cast(_rescale_copy(face_b, self._face_near_b._scale, scale), "face")
        if self._blend == "tangent":
            # The spine is derived from the faces themselves, so it comes
            # out in whatever unit they carry — no rescaling needed here.
            return make_tangent_blend(face_a, face_b, self._tension)
        from magnelio.geo._occ_backend import _shape_wires, make_profile_loft

        def boundaries(face):
            outer = extract_face_wire(face)
            return [outer] + [w for w in _shape_wires(face) if not w.IsSame(outer)]

        return make_profile_loft(
            [boundaries(face_a), boundaries(face_b)], is_ruled=self._blend == "ruled"
        )

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
class Loft(Solid):
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
    *sections : Profile or Sheet or FaceRef or Curve
        At least two planar cross-sections, in the order the solid passes
        through them. Profiles contribute their outer boundary and all holes,
        matched in boundary order. Every section needs the same hole count.
        The sections should wind the same way — a reversed one produces a
        twisted, self-intersecting solid rather than an error.
    blend : {'spline', 'ruled'}
        How consecutive sections are joined.  ``'spline'`` (default)
        passes one smooth surface through all of them; ``'ruled'`` joins
        them with straight surfaces, so the solid is a stack of frusta.
    material : Material, optional
        Material of the lofted solid. Defaults to the first
        section's material; without one the result is a construction solid.
    name : str, optional
        Optional label.

    Raises
    ------
    TypeError
        If a section is a solid (select a FaceRef first)
        or a :class:`~magnelio.geo.Group`.
    ValueError
        If fewer than two sections are given, a Curve section is not
        closed, or *blend* is not one of the two modes above.

    Examples
    --------
    A horn flaring from a square throat to a wider square mouth::

        throat = geo.Profile.rectangle((0, 0, 0), (10e-3, 10e-3))
        mouth = geo.Profile.rectangle((0, 0, 60e-3), (30e-3, 30e-3))
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
        normalized = []
        for index, section in enumerate(sections):
            if isinstance(section, Curve):
                if section._ends is not None and not section.is_closed:
                    raise ValueError(
                        f"Loft section {index} is an open curve; use a closed outline."
                    )
                section = Profile.from_wires(section)
            normalized.append(_profile_input(section, "Loft", planar=True))
        sections = tuple(normalized)
        self.sections = sections
        self.blend = blend
        self.material = resolve_material(material, "Loft(material=...)")
        if self.material is None:
            self.material = getattr(sections[0], "material", None)
        counts = {len(_boundaries(s)) for s in sections}
        if len(counts) != 1:
            raise ValueError("Loft profiles must have the same number of holes.")
        self.name = name

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        from magnelio.geo._occ_backend import make_profile_loft  # noqa: PLC0415

        wires = [[c._occ_shape(scale) for c in _boundaries(s)] for s in self.sections]
        return make_profile_loft(wires, is_ruled=self.blend == "ruled")

    def _analytic_bbox(self):
        from magnelio.geo._scaling import box_diagonal, pad_box, union_boxes  # noqa: PLC0415

        box = union_boxes([s._analytic_bbox() for s in self.sections])
        if self.blend == "ruled":
            # Exact: every ruled point is a convex combination of section
            # points, and a box containing all sections contains those.
            return box
        return pad_box(box, 0.25 * box_diagonal(box))


@dataclass
class _FilletedShape(Solid):
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
        selected = (
            _selected_edges(self._edges)
            if not isinstance(self._edges, str) and self._edges is not None
            else resolve_edges(
                occ_shape,
                near=self._near,
                face_near=self._face_near,
                edges=self._edges,
                scale=scale,
            )
        )
        return make_fillet(occ_shape, selected, self._radius, scale=scale)

    def _analytic_bbox(self):
        return self._inner._analytic_bbox()


_AXIS_VECTORS = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


@dataclass
class _RevolvedShape(Solid):
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
class _SweptShape(Solid):
    _profile: object
    _spine: object
    _material: object
    _frame: str = "corrected_frenet"
    _binormal: object = None
    _twist_deg: float = 0.0
    _draft_deg: float = 0.0
    _tolerance: float | None = None

    @property
    def material(self):
        if self._material is not None:
            return self._material
        return self._profile.material

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        from magnelio.geo._occ_backend import make_sweep  # noqa: PLC0415

        return make_sweep(
            self._profile._occ_shape(scale),
            self._spine._occ_shape(scale),
            frame=self._frame,
            binormal=self._binormal,
            twist_deg=self._twist_deg,
            draft_deg=self._draft_deg,
            tolerance=None if self._tolerance is None else self._tolerance * scale,
        )

    def _analytic_bbox(self):
        from magnelio.geo._scaling import box_diagonal, pad_box  # noqa: PLC0415

        # The profile is re-positioned onto the spine start, so its
        # absolute location is irrelevant — pad the spine box by the
        # profile's full diagonal (conservative for any orientation).
        padding = box_diagonal(self._profile._analytic_bbox())
        if self._draft_deg:
            from magnelio.geo._sweep_laws import draft_radius
            from magnelio.geo.topology import _scale

            scale = _scale(self._profile)
            padding += (
                draft_radius(
                    self._profile._occ_shape(scale),
                    self._spine.length * math.tan(math.radians(self._draft_deg)) * scale,
                )
                / scale
            )
        return pad_box(
            self._spine._analytic_bbox(),
            padding,
        )


def _boundaries(sheet):
    from magnelio.geo.topology import FaceRef, _scale

    if isinstance(sheet, Profile):
        return sheet.boundary()
    return FaceRef(sheet, sheet._occ_shape(_scale(sheet)), _scale(sheet)).detached().boundary()


def loft_profiles(start, end, *, material=None, blend="spline", tension=None):
    """Build a two-section loft from the same categories as other profile verbs."""
    from magnelio.geo._topology_history import finish

    _check_blend(blend, allow_tangent=True)
    tension = _check_tension(tension, blend=blend)
    start = _profile_input(start, "lofted", planar=True)
    end = _profile_input(end, "lofted", planar=True)
    if len(_boundaries(start)) != len(_boundaries(end)):
        raise ValueError("lofted() profiles must have the same number of holes.")
    if blend != "tangent":
        return Loft(start, end, blend=blend, material=material)
    return finish(
        _ProfileBlend(start, end, resolve_material(material, "lofted(material=...)"), tension)
    )


@dataclass
class _ProfileBlend(Solid):
    _shape_a: object
    _shape_b: object
    _material: object
    _tension: tuple

    @property
    def material(self):
        return self._material if self._material is not None else self._shape_a.material

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        from magnelio.geo._occ_backend import make_tangent_blend

        return make_tangent_blend(
            self._shape_a._occ_shape(scale), self._shape_b._occ_shape(scale), self._tension
        )

    def _analytic_bbox(self):
        from magnelio.geo._scaling import box_diagonal, pad_box, union_boxes

        box = union_boxes([self._shape_a._analytic_bbox(), self._shape_b._analytic_bbox()])
        return pad_box(box, max(0.25, 1.5 * max(self._tension)) * box_diagonal(box))
