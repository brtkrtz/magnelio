"""Owner-bound topology views and semantic selection."""

from __future__ import annotations

from dataclasses import dataclass

from magnelio.geo._axes import normalize_axis
from magnelio.geo._cache import cached_occ_shape
from magnelio.geo._validate import point3
from magnelio.geo.shape import Solid


class TopologySelectionError(ValueError):
    """No owned topology satisfies the requested selection."""


class AmbiguousTopologyError(TopologySelectionError):
    """Several subshapes satisfy a singular selection equally well."""


class TopologyEvolutionError(ValueError):
    """Construction cannot prove the successor of a named selection."""


def _scale(owner):
    from magnelio.geo._scaling import choose_scale

    if "_topology_model_scale" not in owner.__dict__:
        owner._topology_model_scale = choose_scale(*owner._analytic_bbox())
    return owner._topology_model_scale


def _kind(kind):
    from OCC.Core import TopAbs

    return getattr(TopAbs, f"TopAbs_{kind.upper()}")


def _cast(shape, kind):
    from OCC.Core.TopoDS import topods

    return getattr(topods, kind.title())(shape)


def _index(shape, kind):
    from OCC.Core.TopExp import topexp
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape

    result = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, _kind(kind), result)
    return result


def _subshapes(shape, kind):
    index = _index(shape, kind)
    return tuple(_cast(index.FindKey(i), kind) for i in range(1, index.Size() + 1))


def _inventory(owner, scale, kind):
    cache = owner.__dict__.setdefault("_topology_inventory", {})
    key = (scale, kind)
    if key not in cache:
        index = _index(owner._occ_shape(scale), kind)
        cache[key] = (
            index,
            tuple(_cast(index.FindKey(i), kind) for i in range(1, index.Size() + 1)),
        )
    return cache[key]


def _type(shape, kind):
    from OCC.Core import GeomAbs
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface

    names = (
        ("plane", "cylinder", "cone", "sphere", "torus", "bspline")
        if kind == "face"
        else ("line", "circle", "ellipse", "hyperbola", "parabola", "bezier", "bspline")
    )
    enums = (
        ("Plane", "Cylinder", "Cone", "Sphere", "Torus", "BSplineSurface")
        if kind == "face"
        else ("Line", "Circle", "Ellipse", "Hyperbola", "Parabola", "BezierCurve", "BSplineCurve")
    )
    adaptor = BRepAdaptor_Surface(shape) if kind == "face" else BRepAdaptor_Curve(shape)
    value = adaptor.GetType()
    return next(
        (name for name, enum in zip(names, enums) if value == getattr(GeomAbs, f"GeomAbs_{enum}")),
        "other",
    )


def _is_planar(face):
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.GeomLib import GeomLib_IsPlanarSurface

    return (
        _type(face, "face") == "plane"
        or GeomLib_IsPlanarSurface(BRep_Tool.Surface(face), 1e-7).IsPlanar()
    )


def _distance(shape, point, scale):
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
    from OCC.Core.gp import gp_Pnt

    probe = BRepBuilderAPI_MakeVertex(gp_Pnt(*(c * scale for c in point))).Vertex()
    measure = BRepExtrema_DistShapeShape(probe, shape)
    measure.Perform()
    if not measure.IsDone() or measure.NbSolution() == 0:
        raise RuntimeError("Topology selection distance failed.")
    return measure.Value(), measure.PointOnShape2(1)


def _normal(face, point=None, scale=1.0):
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepLProp import BRepLProp_SLProps
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCC.Core.TopAbs import TopAbs_REVERSED

    if point is None:
        from OCC.Core.BRepTools import breptools

        u0, u1, v0, v1 = breptools.UVBounds(face)
        u, v = (u0 + u1) / 2, (v0 + v1) / 2
    else:
        distance, on_face = _distance(face, point, scale)
        if distance > 1e-7:
            raise ValueError("normal_at(point) requires a point on the selected face.")
        surface = BRep_Tool.Surface(face)
        projection = GeomAPI_ProjectPointOnSurf(on_face, surface)
        if projection.NbPoints() == 0:
            raise ValueError("The selected point has no surface parameters.")
        u, v = projection.LowerDistanceParameters()
    properties = BRepLProp_SLProps(BRepAdaptor_Surface(face), u, v, 1, 1e-9)
    if not properties.IsNormalDefined():
        raise ValueError("The selected point has no defined surface normal.")
    direction = properties.Normal()
    sign = -1 if face.Orientation() == TopAbs_REVERSED else 1
    return tuple(sign * c for c in (direction.X(), direction.Y(), direction.Z()))


def _measure(shape, dimension, scale):
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps

    properties = GProp_GProps()
    getattr(brepgprop, "SurfaceProperties" if dimension == 2 else "LinearProperties")(
        shape, properties
    )
    return properties.Mass() / scale**dimension, tuple(
        c / scale for c in properties.CentreOfMass().Coord()
    )


@dataclass(frozen=True, eq=False)
class TopologyRef:
    """Read-only subshape view strongly bound to one immutable Solid.

    References are produced by Solid selectors, not constructed by users.
    They cannot be transformed, meshed or used as Boolean operands.

    Attributes
    ----------
    owner : Solid
        Body whose topology this view describes.
    """

    owner: object
    _shape: object
    _scale: float
    _origin: object = None

    def bounding_box(self):
        """Return world-coordinate minimum and maximum corners in metres."""
        from magnelio.geo._occ_backend import bounding_box

        return bounding_box(self._shape, scale=self._scale)


class VertexRef(TopologyRef):
    """Owned vertex with a read-only world-coordinate point."""

    @property
    def point(self):
        """tuple of float: Vertex coordinates in metres."""
        from OCC.Core.BRep import BRep_Tool

        return tuple(c / self._scale for c in BRep_Tool.Pnt(self._shape).Coord())


class EdgeRef(TopologyRef):
    """Owned edge; detach explicitly with :meth:`as_curve`."""

    @property
    def length(self):
        """float: Exact edge length in metres."""
        return _measure(self._shape, 1, self._scale)[0]

    @property
    def vertices(self):
        """tuple of VertexRef: Unique vertices connected to this edge."""
        return tuple(
            VertexRef(self.owner, s, self._scale) for s in _subshapes(self._shape, "vertex")
        )

    @property
    def start(self):
        """tuple of float: Start point following the edge orientation."""
        from OCC.Core.TopExp import topexp

        return VertexRef(self.owner, topexp.FirstVertex(self._shape, True), self._scale).point

    @property
    def end(self):
        """tuple of float: End point following the edge orientation."""
        from OCC.Core.TopExp import topexp

        return VertexRef(self.owner, topexp.LastVertex(self._shape, True), self._scale).point

    def as_curve(self):
        """Return an independent Curve with the edge's exact world geometry.

        Returns
        -------
        Curve
            Standalone curve with no owner or registered selections.
        """
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire

        from magnelio.geo.curves import Curve

        return Curve(
            _build=lambda scale: _rescale_copy(
                BRepBuilderAPI_MakeWire(self._shape).Wire(), self._scale, scale
            ),
            _bounds=self.bounding_box(),
            _ends=(self.start, self.end),
        )


class FaceRef(TopologyRef):
    """Owned face; detach explicitly with :meth:`detached`."""

    @property
    def centroid(self):
        """tuple of float: Area-weighted world centroid in metres."""
        return _measure(self._shape, 2, self._scale)[1]

    @property
    def area(self):
        """float: Face area excluding holes, in square metres."""
        return _measure(self._shape, 2, self._scale)[0]

    @property
    def is_planar(self):
        """bool: Whether the surface is geometrically planar at CAD tolerance."""
        return _is_planar(self._shape)

    @property
    def normal(self):
        """tuple of float: Constant outward normal of a planar face."""
        if not self.is_planar:
            raise ValueError("A curved face has no constant normal; use normal_at(point).")
        return _normal(self._shape)

    def normal_at(self, point):
        """Return the outward unit normal at a point on this face.

        Parameters
        ----------
        point : tuple of float
            World coordinates on the trimmed face, in metres.

        Returns
        -------
        tuple of float
            Oriented unit normal. Singular surface points raise ValueError.
        """
        return _normal(self._shape, point3(point, "normal_at(point)"), self._scale)

    @property
    def edges(self):
        """EdgeSetRef: Unique boundary edges, including inner boundaries."""
        return EdgeSetRef(
            self.owner,
            _subshapes(self._shape, "edge"),
            self._scale,
            {"parent": self, "connection": "edges"},
        )

    @property
    def vertices(self):
        """tuple of VertexRef: Unique vertices on all boundaries."""
        return tuple(
            VertexRef(self.owner, s, self._scale) for s in _subshapes(self._shape, "vertex")
        )

    def detached(self):
        """Return an independent sheet retaining placement, holes and material.

        Returns
        -------
        Profile or Surface
            Profile for planar faces, Surface for curved faces. Registered
            topology names remain on the owner and are not detached.
        """
        from magnelio.geo._sheet import Profile
        from magnelio.geo.surfaces import Surface

        shape = self._shape
        planar = self.is_planar
        if planar and _type(shape, "face") != "plane":
            # A planar spline patch is still a Profile. Re-cover its exact
            # wires with a plane so all Profile constructors can consume it.
            from magnelio.geo._occ_backend import (
                _shape_wires,
                extract_face_wire,
                make_profile_face,
            )

            outer = extract_face_wire(shape)
            holes = [wire for wire in _shape_wires(shape) if not wire.IsSame(outer)]
            shape = make_profile_face(outer, holes)
            if sum(a * b for a, b in zip(_normal(shape), self.normal)) < 0:
                shape.Reverse()
        return _detached_class(Profile if planar else Surface)(
            shape, self._scale, self.bounding_box(), self.owner.material
        )

    def extruded(self, vector, *, material=None):
        """Return a standalone Solid extruded from this face, retaining holes.

        Parameters
        ----------
        vector : tuple of float
            World extrusion vector in metres, non-zero.
        material : Material or str, optional
            Override the owner's material; otherwise it is inherited.

        Returns
        -------
        Solid
            Independent continuation. Owner registrations remain on the owner.
        """
        from magnelio.geo.modifications import extrude

        return extrude(self, vector=vector, material=material)

    def revolved(self, axis, angle_deg=360.0, *, origin=(0.0, 0.0, 0.0), material=None):
        """Return a Solid of revolution from this planar face.

        Parameters
        ----------
        axis : str or tuple of float
            Revolution axis letter or non-zero vector.
        angle_deg : float, optional
            Non-zero angle of at most a full turn, in degrees.
        origin : tuple of float, optional
            World point on the revolution axis, in metres.
        material : Material or str, optional
            Override the owner's material.

        Returns
        -------
        Solid
            Independent solid retaining the face's holes.
        """
        from magnelio.geo.modifications import revolve

        return revolve(self, axis=axis, angle_deg=angle_deg, origin=origin, material=material)

    def swept(
        self,
        spine,
        *,
        material=None,
        frame="corrected_frenet",
        binormal=None,
        twist_deg=0.0,
        draft_deg=0.0,
        tolerance=None,
    ):
        """Return a Solid by sweeping this planar face along a Curve.

        Parameters
        ----------
        spine : Curve
            World sweep path. The centroid moves to its start and the outward
            normal aligns with its tangent by the shortest rotation. In-plane
            roll is retained; an already aligned face stays in its actual pose.
        material : Material or str, optional
            Override the owner's material.

        frame : {'corrected_frenet', 'frenet', 'fixed', 'fixed_binormal'}, optional
            Transport of the initially aligned section. Fixed keeps sections
            parallel in world space; fixed binormal preserves their angular
            relation to a supplied world direction.
        binormal : str or tuple of float, optional
            Required only for fixed binormal transport; it must not be parallel
            to the spine tangent.
        twist_deg : float, optional
            Total additional right-hand roll about the transported section
            normal [degrees], uniform over arc length.
        draft_deg : float, optional
            Constant section-offset angle [degrees]. Positive values grow the
            exterior and shrink holes; absolute values must be below 90.
        tolerance : float, optional
            Sampled fitting tolerance [m], default one millionth of the profile diagonal.

        Returns
        -------
        Solid
            Independent continuation with every hole retained.
        """
        from magnelio.geo.modifications import sweep

        return sweep(
            self,
            spine,
            material=material,
            frame=frame,
            binormal=binormal,
            twist_deg=twist_deg,
            draft_deg=draft_deg,
            tolerance=tolerance,
        )

    def thickened(self, thickness, *, direction="forward", material=None):
        """Return a Solid offset from this face.

        Parameters
        ----------
        thickness : float
            Positive thickness in metres.
        direction : {'forward', 'backward', 'symmetric'}, optional
            Forward follows the outward normal. Symmetric requires planarity.
        material : Material or str, optional
            Override the owner's material.

        Returns
        -------
        Solid
            Independent slab or curved offset.
        """
        from magnelio.geo.modifications import thicken

        return thicken(self, thickness=thickness, direction=direction, material=material)

    def lofted(self, other, *, material=None, blend="spline", tension=None):
        """Return a Solid connecting this planar face to another section.

        Parameters
        ----------
        other : Profile or Sheet or FaceRef
            Planar end section. Hole counts must agree; boundary order fixes
            correspondence. Tangent blending uses the oriented face normals.
        material : Material or str, optional
            Override the start owner's material.
        blend : {'spline', 'ruled', 'tangent'}, optional
            Surface interpolation or a transition leaving both face normals.
        tension : float or tuple of float, optional
            Positive tangent reach fractions, only for tangent blending.

        Returns
        -------
        Solid
            Independent transition retaining all holes.
        """
        from magnelio.geo.modifications import loft_profiles

        return loft_profiles(self, other, material=material, blend=blend, tension=tension)


class _RefSet(TopologyRef):
    def __len__(self):
        return len(self._shape)

    def __iter__(self):
        return (
            self._member_type(self.owner, s, self._scale, {"parent": self, "member_shape": s})
            for s in self._shape
        )

    def bounding_box(self):
        """Return a world bounding box enclosing every member, in metres."""
        from magnelio.geo._scaling import union_boxes

        return union_boxes([ref.bounding_box() for ref in self])


class EdgeSetRef(_RefSet):
    """Deliberate, iterable collection of edges sharing one Solid owner."""

    _member_type = EdgeRef


class FaceSetRef(_RefSet):
    """Deliberate, iterable collection of faces sharing one Solid owner."""

    _member_type = FaceRef


def _rescale_copy(shape, source_scale, target_scale):
    from magnelio.geo._occ_backend import occ_scale

    return occ_scale(shape, target_scale / source_scale, (0, 0, 0))


_DETACHED_CLASSES = {}


def _detached_class(category):
    if category not in _DETACHED_CLASSES:
        _DETACHED_CLASSES[category] = type(
            f"Detached{category.__name__}", (_DetachedSheet, category), {}
        )
    return _DETACHED_CLASSES[category]


class _DetachedSheet:
    def __init__(self, shape, scale, bounds, material):
        self._shape = shape
        self._source_scale = scale
        self._bounds = bounds
        self.material = material
        self.name = None
        self._occ_shape_cache = {}

    def __repr__(self):
        return f"{type(self).__name__}(material={self.material!r})"

    def _occ_shape(self, scale=1.0):
        if scale not in self._occ_shape_cache:
            self._occ_shape_cache[scale] = _cast(
                _rescale_copy(self._shape, self._source_scale, scale), "face"
            )
        return self._occ_shape_cache[scale]

    def _analytic_bbox(self):
        return self._bounds

    def boundary(self):
        from magnelio.geo._occ_backend import _shape_wires, extract_face_wire
        from magnelio.geo.curves import Curve

        outer = extract_face_wire(self._shape)
        wires = [outer] + [wire for wire in _shape_wires(self._shape) if not wire.IsSame(outer)]
        return tuple(
            Curve(
                _build=lambda scale, wire=wire: _rescale_copy(wire, self._source_scale, scale),
                _bounds=self._bounds,
            )
            for wire in wires
        )


def _nearest(owner, scale, kind, members, near):
    import numpy as np

    from magnelio.geo._occ_backend import bounding_box

    index, all_members = _inventory(owner, scale, kind)
    cache = owner.__dict__.setdefault("_topology_bounds", {})
    key = (scale, kind)
    if key not in cache:
        cache[key] = np.array([bounding_box(shape) for shape in all_members])
    indices = np.array([index.FindIndex(shape) - 1 for shape in members])
    boxes = cache[key][indices]
    point = np.array(near) * scale
    delta = np.maximum(np.maximum(boxes[:, 0] - point, point - boxes[:, 1]), 0)
    lower_bounds = np.linalg.norm(delta, axis=1)
    best = float("inf")
    measured = []
    for slot in np.argsort(lower_bounds, kind="stable"):
        if lower_bounds[slot] > best + 1e-7:
            break
        distance = _distance(members[slot], near, scale)[0]
        best = min(best, distance)
        measured.append((members[slot], distance))
    return tuple(shape for shape, distance in measured if distance <= best + 1e-7)


def _candidate_summary(members, kind, scale):
    summaries = []
    for shape in members[:8]:
        if kind == "vertex":
            point = VertexRef(None, shape, scale).point
            summaries.append(f"vertex at {point}")
        else:
            _, center = _measure(shape, 2 if kind == "face" else 1, scale)
            summaries.append(f"{_type(shape, kind)} centred at {center}")
    return "; ".join(sorted(summaries)) + ("; ..." if len(members) > 8 else "")


def select(
    owner,
    kind,
    name=None,
    *,
    near=None,
    normal=None,
    surface_type=None,
    curve_type=None,
    plural=False,
    _model_scale=None,
):
    from magnelio.geo._topology_history import names

    scale = _scale(owner) if _model_scale is None else _model_scale
    if name is not None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "A topology name must be a non-empty string; numeric indices are not supported."
            )
        if any(value is not None for value in (near, normal, surface_type, curve_type)):
            raise ValueError("Named lookup cannot be combined with semantic constraints.")
        registered = names(owner, scale).get((kind, name))
        if registered is None:
            raise TopologySelectionError(f"No named {kind} selection {name!r} on this owner.")
        members, is_set = registered
        if plural != is_set:
            raise TopologySelectionError(
                f"Selection {name!r} is "
                + (
                    "a set; use the plural selector."
                    if is_set
                    else "singular; use the singular selector."
                )
            )
    else:
        if not plural and all(value is None for value in (near, normal, surface_type, curve_type)):
            raise ValueError("A singular selection requires near, normal or type constraints.")
        if near is not None:
            near = point3(near, "near")
        if normal is not None:
            normal = normalize_axis(normal, "normal")
        wanted = surface_type if kind == "face" else curve_type
        allowed = (
            {"plane", "cylinder", "cone", "sphere", "torus", "bspline", "other"}
            if kind == "face"
            else {
                "line",
                "circle",
                "ellipse",
                "hyperbola",
                "parabola",
                "bezier",
                "bspline",
                "other",
            }
        )
        if wanted is not None and wanted not in allowed:
            raise ValueError(f"Unsupported {kind} type {wanted!r}; expected {sorted(allowed)}.")
        members = _inventory(owner, scale, kind)[1]
        if wanted is not None:
            members = tuple(s for s in members if _type(s, kind) == wanted)
        if normal is not None:
            filtered = []
            for s in members:
                planar = _is_planar(s)
                if not planar and near is None:
                    continue
                if planar:
                    actual = _normal(s)
                else:
                    _, p = _distance(s, near, scale)
                    actual = _normal(s, tuple(c / scale for c in p.Coord()), scale)
                if sum((a - b) ** 2 for a, b in zip(actual, normal)) <= 1e-12:
                    filtered.append(s)
            members = tuple(filtered)
        if near is not None and members:
            members = _nearest(owner, scale, kind, members, near)
        if not members:
            raise TopologySelectionError(f"No {kind} satisfies the supplied semantic constraints.")
        if not plural and len(members) != 1:
            raise AmbiguousTopologyError(
                f"{len(members)} equally eligible {kind} candidates: "
                f"{_candidate_summary(members, kind, scale)}. "
                "Disambiguate with near, normal or type constraints, "
                "or select a deliberate set."
            )
    cls = {
        ("face", False): FaceRef,
        ("edge", False): EdgeRef,
        ("vertex", False): VertexRef,
        ("face", True): FaceSetRef,
        ("edge", True): EdgeSetRef,
    }[(kind, plural)]
    origin = {
        "kind": kind,
        "name": name,
        "plural": plural,
        "selectors": {
            key: value
            for key, value in (
                ("near", near),
                ("normal", normal),
                ("surface_type", surface_type),
                ("curve_type", curve_type),
            )
            if value is not None
        },
    }
    return cls(owner, members if plural else members[0], scale, origin)


def tag(owner, kind, name, *, plural=False, **selectors):
    from magnelio.geo._topology_history import finish, names

    if not isinstance(name, str) or not name.strip():
        raise ValueError("A topology name must be a non-empty string.")
    selectors = {key: value for key, value in selectors.items() if value is not None}
    if "near" in selectors:
        selectors["near"] = point3(selectors["near"], "near")
    if "normal" in selectors:
        selectors["normal"] = normalize_axis(selectors["normal"], "normal")
    scale = _scale(owner)
    if (kind, name) in names(owner, scale):
        raise ValueError(f"A {kind} selection named {name!r} already exists on this owner.")
    selected = select(owner, kind, plural=plural, **selectors)
    count = len(selected) if plural else 1
    return finish(_TaggedSolid(owner, kind, name, selectors, plural, count))


class _TaggedSolid(Solid):
    def __init__(self, inner, kind, label, selectors, plural, count):
        self._inner = inner
        self._registration = (kind, label, selectors, plural, count)

    @property
    def material(self):
        return self._inner.material

    @property
    def name(self):
        return getattr(self._inner, "name", None)

    @property
    def color(self):
        return getattr(self._inner, "color", None)

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        return self._inner._occ_shape(scale)

    def _analytic_bbox(self):
        return self._inner._analytic_bbox()
