"""Finite freeform bends of existing solids and sheets."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from magnelio.geo._axes import normalize_axis
from magnelio.geo._cache import cached_occ_shape
from magnelio.geo._scaling import box_diagonal, pad_box, union_boxes
from magnelio.geo._sheet import Sheet
from magnelio.geo._topology_history import finish, result
from magnelio.geo._validate import point3, positive
from magnelio.geo.shape import Shape, Solid


def _unit(vector):
    magnitude = np.linalg.norm(vector)
    if not math.isfinite(magnitude) or magnitude <= 0:
        raise ValueError("Bend directions and surface tangents must be nonzero and finite.")
    return vector / magnitude


def _box_corners(box):
    low, high = box
    return np.array(
        [
            (x, y, z)
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ],
        dtype=float,
    )


@dataclass(frozen=True)
class Bend:
    """Reusable finite bend around an explicit neutral surface.

    The source coordinates are measured from *origin* along orthonormal
    *along* and *across* directions; their right-handed normal is the layer
    direction.  The target sheet's increasing surface parameters correspond
    linearly to the declared ``u`` and ``v`` intervals.  A point at normal
    distance ``w`` maps to the target point plus ``w`` times its unit normal.

    Parameters
    ----------
    target : Sheet
        Single-face neutral surface with a regular rectangular parameter
        chart. Its start edge must coincide with the source neutral plane.
    origin : sequence of float
        World origin of the source frame [m].
    along, across : str or sequence of float
        Perpendicular world directions of the source chart.
    u, v : pair of float
        Affected longitudinal interval and transverse chart interval [m].
        Geometry before ``u[0]`` stays fixed; geometry after ``u[1]`` follows
        the rigid end frame. The target covers the complete transverse span.
    max_strain : float
        Greatest permitted principal in-plane stretch or compression of the
        neutral surface, relative to the flat source chart. Required.
    tolerance : float, optional
        Absolute sampled CAD approximation budget [m]. The default is one
        millionth of the source/target extent, subject to CAD resolution.

    Notes
    -----
    The normal extension can change volume. It is checked for folds and
    invalid CAD topology; the tolerance is a sampled error bound rather than
    a global Hausdorff certificate.
    """

    target: Sheet
    origin: tuple[float, float, float]
    along: tuple[float, float, float]
    across: tuple[float, float, float]
    u: tuple[float, float]
    v: tuple[float, float]
    max_strain: float
    tolerance: float | None = None

    def __post_init__(self):
        operation = type(self).__name__
        if not isinstance(self.target, Sheet):
            raise TypeError(
                f"{operation} target must be a Sheet; got {type(self.target).__name__}."
            )
        origin = point3(self.origin, f"{operation}(origin)")
        along = np.asarray(normalize_axis(self.along, f"{operation}(along)"))
        across = np.asarray(normalize_axis(self.across, f"{operation}(across)"))
        if abs(float(np.dot(along, across))) > 1e-10:
            raise ValueError(f"{operation} along and across directions must be perpendicular.")
        for label in ("u", "v"):
            span = getattr(self, label)
            if len(span) != 2 or not all(math.isfinite(float(x)) for x in span):
                raise ValueError(f"{operation} {label} must be a finite (start, end) pair.")
            start, end = map(float, span)
            if end <= start:
                raise ValueError(f"{operation} {label} interval must increase.")
            object.__setattr__(self, label, (start, end))
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "along", tuple(float(x) for x in along))
        object.__setattr__(self, "across", tuple(float(x) for x in across))
        object.__setattr__(
            self, "max_strain", positive(self.max_strain, f"{operation}(max_strain)")
        )
        if self.tolerance is not None:
            object.__setattr__(
                self, "tolerance", positive(self.tolerance, f"{operation}(tolerance)")
            )

    def __matmul__(self, other):
        """Bend a Solid, Sheet or material-preserving Group independently."""
        from magnelio.geo.operations import Group

        if isinstance(other, Group):
            return Group(*(self @ member for member in other.shapes), name=other.name)
        if isinstance(other, Solid):
            return finish(_BentSolid(other, self))
        if isinstance(other, Sheet):
            return finish(_BentSheet(other, self))
        raise TypeError(f"Bend accepts Solid, Sheet or Group; got {type(other).__name__}.")

    def _frame(self):
        along = np.asarray(self.along)
        across = np.asarray(self.across)
        return np.stack((along, across, np.cross(along, across)), axis=1)

    def _context(self, source, scale, *, finite=True):
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.BRepTools import breptools
        from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE
        from OCC.Core.TopoDS import topods

        operation = "Bend" if finite else "Wrap"
        faces = members_of(self.target._occ_shape(scale), TopAbs_FACE, topods.Face)
        if len(faces) != 1:
            raise ValueError(f"{operation} target must have one continuous surface face.")
        face = faces[0]
        if len(members_of(face, TopAbs_WIRE, topods.Wire)) != 1:
            raise ValueError(f"{operation} target chart must be continuous without holes.")
        bounds = np.asarray(breptools.UVBounds(face), dtype=float)
        if not np.all(np.isfinite(bounds)) or bounds[1] <= bounds[0] or bounds[3] <= bounds[2]:
            raise ValueError(f"{operation} target needs a finite, regular surface chart.")
        frame = self._frame()
        source_box = source._analytic_bbox()
        extent = max(box_diagonal(source_box), box_diagonal(self.target._analytic_bbox()))
        tolerance = self.tolerance
        if tolerance is None:
            tolerance = max(extent * 1e-6, 1e-7 / scale)
        context = _BendContext(
            self, BRepAdaptor_Surface(face), face, bounds, frame, scale, tolerance, finite
        )
        context.check(source._occ_shape(scale))
        return context


class _BendContext:
    def __init__(self, bend, surface, face, uv, frame, scale, tolerance, finite):
        self.bend = bend
        self.surface = surface
        self.face = face
        self.uv = uv
        self.frame = frame
        self.scale = scale
        self.tolerance = tolerance
        self.finite = finite
        self.operation = "Bend" if finite else "Wrap"
        self.origin = np.asarray(bend.origin)
        self.end_frame = None
        self.end_anchor = None

    def _parameters(self, u, v):
        b = self.bend
        U = self.uv[0] + (u - b.u[0]) * (self.uv[1] - self.uv[0]) / (b.u[1] - b.u[0])
        V = self.uv[2] + (v - b.v[0]) * (self.uv[3] - self.uv[2]) / (b.v[1] - b.v[0])
        return U, V

    def neutral(self, u, v):
        from OCC.Core.gp import gp_Pnt, gp_Vec

        U, V = self._parameters(u, v)
        point, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
        self.surface.D1(U, V, point, du, dv)
        du = np.asarray(du.Coord()) * (self.uv[1] - self.uv[0]) / (self.bend.u[1] - self.bend.u[0])
        dv = np.asarray(dv.Coord()) * (self.uv[3] - self.uv[2]) / (self.bend.v[1] - self.bend.v[0])
        cross = np.cross(du, dv)
        normal = _unit(cross)
        return np.asarray(point.Coord()) / self.scale, du / self.scale, dv / self.scale, normal

    def coordinates(self, point):
        return self.frame.T @ (np.asarray(point) - self.origin)

    def point(self, point):
        u, v, w = self.coordinates(point)
        if self.finite and u <= self.bend.u[0]:
            return np.asarray(point)
        if self.finite and u >= self.bend.u[1]:
            return self.end_anchor + self.end_frame @ np.array((u - self.bend.u[1], v, w))
        neutral, _, _, normal = self.neutral(u, v)
        return neutral + w * normal

    def occ_point(self, point):
        from OCC.Core.gp import gp_Pnt

        return gp_Pnt(*(self.point(np.asarray(point.Coord()) / self.scale) * self.scale))

    def _boundary(self, u, *, start):
        b = self.bend
        reference = self.origin + self.frame @ np.array((u, 0.0, 0.0))
        candidates = []
        for v in np.linspace(*b.v, 33):
            point, du, dv, normal = self.neutral(u, v)
            candidates.append((point, du, dv, normal))
        base = candidates[0][0] - b.v[0] * _unit(candidates[0][2])
        axes = np.stack(
            (_unit(candidates[0][1]), _unit(candidates[0][2]), candidates[0][3]), axis=1
        )
        length_scale = max(b.u[1] - b.u[0], b.v[1] - b.v[0])
        position_limit = max(self.tolerance * 4, length_scale * 1e-8)
        derivative_limit = max(4 * self.tolerance / length_scale, 1e-7)
        if np.linalg.norm(axes.T @ axes - np.eye(3)) > derivative_limit:
            raise ValueError("Bend target boundary has no rigid tangent frame.")
        for v, (point, du, dv, normal) in zip(np.linspace(*b.v, 33), candidates):
            if (
                np.linalg.norm(point - (base + v * axes[:, 1])) > position_limit
                or np.linalg.norm(_unit(du) - axes[:, 0]) > derivative_limit
                or np.linalg.norm(_unit(dv) - axes[:, 1]) > derivative_limit
                or np.linalg.norm(normal - axes[:, 2]) > derivative_limit
            ):
                raise ValueError(
                    "Bend target boundary cannot join a rigid continuation tangentially."
                )
        if start and (
            np.linalg.norm(base - reference) > position_limit
            or np.linalg.norm(axes - self.frame) > derivative_limit
        ):
            raise ValueError(
                "Bend target start must meet the unchanged source in position and frame."
            )
        return base, axes

    def check(self, source_shape):
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
        from OCC.Core.BRepClass import BRepClass_FaceClassifier
        from OCC.Core.gp import gp_Pnt2d, gp_Trsf
        from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON

        from magnelio.geo._occ_backend import bounding_box

        b = self.bend
        rotation = self.frame.T
        translation = -rotation @ self.origin * self.scale
        transform = gp_Trsf()
        transform.SetValues(
            *(float(value) for row in np.column_stack((rotation, translation)) for value in row)
        )
        local_shape = BRepBuilderAPI_Transform(source_shape, transform, True).Shape()
        self.local_bounds = bounding_box(local_shape, scale=self.scale)
        coordinates = _box_corners(self.local_bounds)
        if (
            np.min(coordinates[:, 1]) < b.v[0] - self.tolerance
            or np.max(coordinates[:, 1]) > b.v[1] + self.tolerance
        ):
            raise ValueError(
                f"{self.operation} target does not cover the source's transverse extent."
            )
        if self.finite:
            _, _ = self._boundary(b.u[0], start=True)
            self.end_anchor, self.end_frame = self._boundary(b.u[1], start=False)
        elif (
            np.min(coordinates[:, 0]) < b.u[0] - self.tolerance
            or np.max(coordinates[:, 0]) > b.u[1] + self.tolerance
        ):
            raise ValueError("Wrap target does not cover the source's longitudinal extent.")
        w_min, w_max = np.min(coordinates[:, 2]), np.max(coordinates[:, 2])
        length = max(b.u[1] - b.u[0], b.v[1] - b.v[0])
        step = length * 1e-5
        for iu, u in enumerate(np.linspace(*b.u, 33)):
            for iv, v in enumerate(np.linspace(*b.v, 33)):
                U, V = self._parameters(u, v)
                classifier = BRepClass_FaceClassifier(self.face, gp_Pnt2d(U, V), 1e-9)
                if classifier.State() not in (TopAbs_IN, TopAbs_ON):
                    raise ValueError(
                        f"{self.operation} target does not cover the declared surface chart."
                    )
                _, du, dv, normal = self.neutral(u, v)
                if (
                    np.max(abs(np.linalg.svd(np.stack((du, dv), axis=1), compute_uv=False) - 1))
                    > b.max_strain + 1e-8
                ):
                    raise ValueError(
                        f"{self.operation} target exceeds the permitted neutral-surface strain."
                    )
                if iu % 4 or iv % 4 or u in b.u or v in b.v:
                    continue
                for w in (w_min, w_max):
                    pu = self.point(self.origin + self.frame @ np.array((u + step, v, w)))
                    mu = self.point(self.origin + self.frame @ np.array((u - step, v, w)))
                    pv = self.point(self.origin + self.frame @ np.array((u, v + step, w)))
                    mv = self.point(self.origin + self.frame @ np.array((u, v - step, w)))
                    jac = np.linalg.det(
                        np.stack(((pu - mu) / (2 * step), (pv - mv) / (2 * step), normal), axis=1)
                    )
                    if jac <= 1e-8:
                        raise ValueError(
                            f"{self.operation} normal extension folds through the "
                            "occupied thickness."
                        )

    def rigid_surface(self, face, side):
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.Geom import Geom_Surface
        from OCC.Core.gp import gp_Trsf

        if side == "start":
            return BRep_Tool.Surface(face)
        rotation = self.end_frame @ self.frame.T
        source_anchor = self.origin + self.bend.u[1] * self.frame[:, 0]
        offset = (self.end_anchor - rotation @ source_anchor) * self.scale
        transform = gp_Trsf()
        transform.SetValues(
            *(float(value) for row in np.column_stack((rotation, offset)) for value in row)
        )
        return Geom_Surface.DownCast(BRep_Tool.Surface(face).Transformed(transform))


def _fit_surface(face, context):
    from OCC.Core.Approx import Approx_IsoParametric
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepTools import breptools
    from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface
    from OCC.Core.TColgp import TColgp_Array2OfPnt
    from OCC.Core.TColStd import TColStd_Array1OfReal

    source = BRepAdaptor_Surface(face)
    u0, u1, v0, v1 = breptools.UVBounds(face)
    if not all(map(math.isfinite, (u0, u1, v0, v1))) or u0 >= u1 or v0 >= v1:
        raise ValueError(f"{context.operation} source face needs a finite regular parameter chart.")
    for count in (9, 17, 33, 65):
        points = TColgp_Array2OfPnt(1, count, 1, count)
        for i in range(1, count + 1):
            for j in range(1, count + 1):
                u = u0 + (u1 - u0) * (i - 1) / (count - 1)
                v = v0 + (v1 - v0) * (j - 1) / (count - 1)
                points.SetValue(i, j, context.occ_point(source.Value(u, v)))
        fit = GeomAPI_PointsToBSplineSurface()
        fit.Interpolate(points, Approx_IsoParametric)
        if not fit.IsDone():
            raise ValueError(f"{context.operation} could not interpolate a source CAD face.")
        surface = fit.Surface()
        for axis, start, end in (("U", u0, u1), ("V", v0, v1)):
            n = getattr(surface, f"Nb{axis}Knots")()
            knots = getattr(surface, f"{axis}Knots")()
            first, last = knots.Value(1), knots.Value(n)
            new = TColStd_Array1OfReal(1, n)
            for k in range(1, n + 1):
                new.SetValue(k, start + (end - start) * (knots.Value(k) - first) / (last - first))
            getattr(surface, f"Set{axis}Knots")(new)
        error = 0.0
        for i in range(count - 1):
            for j in range(count - 1):
                for fu, fv in ((0.25, 0.25), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.75, 0.75)):
                    u = u0 + (u1 - u0) * (i + fu) / (count - 1)
                    v = v0 + (v1 - v0) * (j + fv) / (count - 1)
                    exact = context.occ_point(source.Value(u, v))
                    error = max(error, surface.Value(u, v).Distance(exact) / context.scale)
        if error <= context.tolerance:
            return surface
    raise ValueError(
        f"{context.operation} could not meet the sampled CAD approximation tolerance "
        f"({error:.3g} m > {context.tolerance:.3g} m)."
    )


def _build_bent(source, bend, scale):
    from OCC.Core.BRep import BRep_Builder, BRep_Tool
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Check
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.BRepLib import breplib
    from OCC.Core.BRepTools import breptools
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.TopoDS import topods

    context = bend._context(source, scale)
    prepared = (
        _split_at_boundaries(source._occ_shape(scale), context)
        if context.finite
        else source._occ_shape(scale)
    )
    copy = BRepBuilderAPI_Copy(prepared)
    shape = copy.Shape()

    def members(kind, cast):
        unique = []
        explorer = TopExp_Explorer(shape, kind)
        while explorer.More():
            item = cast(explorer.Current())
            if not any(item.IsSame(other) for other in unique):
                unique.append(item)
            explorer.Next()
        return unique

    faces = members(TopAbs_FACE, topods.Face)
    edges = members(TopAbs_EDGE, topods.Edge)
    vertices = members(TopAbs_VERTEX, topods.Vertex)
    saved = []
    for face in faces:
        coordinates = np.array(
            [
                context.coordinates(np.asarray(BRep_Tool.Pnt(vertex).Coord()) / scale)
                for vertex in members_of(face, TopAbs_VERTEX, topods.Vertex)
            ]
        )
        if (
            context.finite
            and len(coordinates)
            and np.max(coordinates[:, 0]) <= bend.u[0] + context.tolerance
        ):
            surface = context.rigid_surface(face, "start")
        elif (
            context.finite
            and len(coordinates)
            and np.min(coordinates[:, 0]) >= bend.u[1] - context.tolerance
        ):
            surface = context.rigid_surface(face, "end")
        else:
            surface = _fit_surface(face, context)
        curves = []
        for edge in members_of(face, TopAbs_EDGE, topods.Edge):
            first = BRep_Tool.CurveOnSurface(edge, face)[0]
            second = (
                BRep_Tool.CurveOnSurface(topods.Edge(edge.Reversed()), face)[0]
                if BRep_Tool.IsClosed(edge, face)
                else None
            )
            curves.append((edge, first, second))
        saved.append((face, surface, curves))
    builder = BRep_Builder()
    for vertex in vertices:
        builder.UpdateVertex(
            vertex, context.occ_point(BRep_Tool.Pnt(vertex)), context.tolerance * scale
        )
    for face, _, curves in saved:
        for edge, _, second in curves:
            if second is None:
                builder.UpdateEdge(edge, None, face, context.tolerance * scale)
            else:
                builder.UpdateEdge(edge, None, None, face, context.tolerance * scale)
    for edge in edges:
        builder.UpdateEdge(edge, None, context.tolerance * scale)
    for face, surface, _ in saved:
        builder.UpdateFace(face, surface, TopLoc_Location(), context.tolerance * scale)
    for face, _, curves in saved:
        for edge, first, second in curves:
            if second is None:
                builder.UpdateEdge(edge, first, face, context.tolerance * scale)
            else:
                builder.UpdateEdge(edge, first, second, face, context.tolerance * scale)
    for edge in edges:
        if not breplib.BuildCurve3d(edge):
            raise ValueError(f"{context.operation} could not rebuild a shared CAD edge.")
    breptools.Clean(shape)
    if not BRepCheck_Analyzer(shape).IsValid():
        raise ValueError(
            f"{context.operation} produced invalid CAD topology or inconsistent shared edges."
        )
    if not BRepAlgoAPI_Check(shape, False, True).IsValid():
        raise ValueError(f"{context.operation} produced self-intersecting CAD geometry.")
    result(copy)
    return shape


def _split_at_boundaries(shape, context):
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.BRepFeat import BRepFeat_SplitShape
    from OCC.Core.gp import gp_Dir, gp_Pln, gp_Pnt
    from OCC.Core.TopAbs import TopAbs_EDGE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods
    from OCC.Core.TopTools import TopTools_SequenceOfShape

    from magnelio.geo._occ_backend import _shape_list, bounding_box, keep_operands_intact

    corners = _box_corners(bounding_box(shape, scale=context.scale))
    along = (context.local_bounds[0][0], context.local_bounds[1][0])
    extent = max(np.ptp(corners, axis=0))
    width = 4 * extent * context.scale
    edges = TopTools_SequenceOfShape()
    for station in context.bend.u:
        if station <= min(along) + context.tolerance or station >= max(along) - context.tolerance:
            continue
        center = context.origin + station * context.frame[:, 0]
        plane = gp_Pln(gp_Pnt(*(center * context.scale)), gp_Dir(*context.frame[:, 0]))
        cutter = BRepBuilderAPI_MakeFace(plane, -width, width, -width, width).Face()
        section = BRepAlgoAPI_Section()
        section.SetArguments(_shape_list((shape,)))
        section.SetTools(_shape_list((cutter,)))
        keep_operands_intact(section)
        section.Build()
        if not section.IsDone():
            raise ValueError("Bend could not split the source at its interval boundary.")
        explorer = TopExp_Explorer(section.Shape(), TopAbs_EDGE)
        while explorer.More():
            edges.Append(topods.Edge(explorer.Current()))
            explorer.Next()
    if edges.Length() == 0:
        return shape
    splitter = BRepFeat_SplitShape(shape)
    if not splitter.Add(edges):
        raise ValueError("Bend could not assign interval seams to source faces.")
    splitter.Build()
    if not splitter.IsDone() or not BRepCheck_Analyzer(splitter.Shape()).IsValid():
        raise ValueError("Bend could not form valid interval seams.")
    result(splitter)
    return splitter.Shape()


def members_of(shape, kind, cast):
    from OCC.Core.TopExp import TopExp_Explorer

    found = []
    explorer = TopExp_Explorer(shape, kind)
    while explorer.More():
        item = cast(explorer.Current())
        if not any(item.IsSame(other) for other in found):
            found.append(item)
        explorer.Next()
    return found


class _BentBase(Shape):
    def __init__(self, inner, bend):
        self._inner = inner
        self._bend = bend

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
        return _build_bent(self._inner, self._bend, scale)

    def _analytic_bbox(self):
        source = self._inner._analytic_bbox()
        target = self._bend.target._analytic_bbox()
        span = max(box_diagonal(source), box_diagonal(target))
        return pad_box(union_boxes((source, target)), span)


class _BentSolid(_BentBase, Solid):
    pass


class _BentSheet(_BentBase, Sheet):
    def thickened(self, thickness, *, direction="forward", material=None):
        from OCC.Core.TopAbs import TopAbs_FACE
        from OCC.Core.TopoDS import topods

        from magnelio.geo.topology import _scale

        faces = members_of(self._occ_shape(_scale(self)), TopAbs_FACE, topods.Face)
        if len(faces) != 1:
            raise ValueError(
                "This bent Sheet crosses interval seams and has multiple CAD faces. "
                "Thicken the source Sheet first, then apply the same Bend to that Solid."
            )
        return super().thickened(thickness, direction=direction, material=material)
