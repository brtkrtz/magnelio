"""Conditioned bounded-face sections for sensitive planes (DD-287).

Imported only after OpenCASCADE is available. Geometry is never rebuilt.
"""

from __future__ import annotations

import itertools
import math
from collections import OrderedDict, defaultdict

import numpy as np
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepTools import breptools
from OCC.Core.GCPnts import GCPnts_TangentialDeflection
from OCC.Core.Geom import Geom_Plane, Geom_RectangularTrimmedSurface, Geom_TrimmedCurve
from OCC.Core.Geom2d import Geom2d_Curve, Geom2d_Line, Geom2d_TrimmedCurve
from OCC.Core.Geom2dAdaptor import geom2dadaptor
from OCC.Core.Geom2dAPI import (
    Geom2dAPI_InterCurveCurve,
    Geom2dAPI_Interpolate,
    Geom2dAPI_ProjectPointOnCurve,
)
from OCC.Core.GeomAbs import (
    GeomAbs_BSplineCurve,
    GeomAbs_BSplineSurface,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_SurfaceOfExtrusion,
    GeomAbs_Torus,
)
from OCC.Core.GeomAdaptor import GeomAdaptor_Curve, GeomAdaptor_Surface
from OCC.Core.GeomAPI import GeomAPI_IntCS, GeomAPI_Interpolate, GeomAPI_ProjectPointOnCurve
from OCC.Core.GeomConvert import GeomConvert_BSplineSurfaceToBezierSurface
from OCC.Core.GeomInt import GeomInt_IntSS
from OCC.Core.gp import gp_Dir, gp_Dir2d, gp_Pln, gp_Pnt, gp_Pnt2d, gp_Trsf, gp_Vec, gp_Vec2d
from OCC.Core.IntRes2d import IntRes2d_End, IntRes2d_Head, IntRes2d_Undecided
from OCC.Core.ProjLib import ProjLib_ProjectedCurve
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
from OCC.Core.TColgp import TColgp_HArray1OfPnt, TColgp_HArray1OfPnt2d
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_HArray1OfReal
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_EXTERNAL, TopAbs_INTERNAL, TopAbs_REVERSED
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopoDS import topods
from OCC.Core.TopTools import TopTools_IndexedMapOfShape
from scipy.optimize import brentq, root
from scipy.signal import convolve2d

from magnelio.geo._occ_backend import _FaceSlabIndex
from magnelio.geo._scaling import fine_detail_scale


class _SectionFailure(RuntimeError):
    """A bounded-face section could not meet its geometric contract."""


class _CoplanarSection(_SectionFailure):
    """The established face-region semantics must handle this plane."""


class _PrecisionFailure(_SectionFailure):
    """The measurement copy needs more kernel-unit precision headroom."""


_RAW_ROUTERS = OrderedDict()


def _regular_normal(surface, u, v, poles=(), pole_tolerance=0.0):
    p, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
    surface.D1(float(u), float(v), p, du, dv)
    normal = _coordinates(du.Crossed(dv))
    derivative_scale = max(np.linalg.norm(_coordinates(du)), np.linalg.norm(_coordinates(dv)))
    limiting = min(np.linalg.norm(_coordinates(du)), np.linalg.norm(_coordinates(dv))) <= (
        128 * np.finfo(float).eps * derivative_scale
    )
    a, b, c, d = surface.Bounds()
    param_round = np.sqrt(np.finfo(float).eps)
    limiting |= any(
        abs((u if axis == 0 else v) - value) <= param_round * ((b - a) if axis == 0 else (d - c))
        for axis, value in poles
    )
    adaptor = GeomAdaptor_Surface(surface)
    if adaptor.GetType() == GeomAbs_BSplineSurface:
        spline = adaptor.BSpline()
        a, b, c, d = surface.Bounds()
        param_round = np.sqrt(np.finfo(float).eps)
        row_tol = max(
            pole_tolerance,
            128
            * np.finfo(float).eps
            * (
                np.linalg.norm(_coordinates(du)) * (b - a)
                + np.linalg.norm(_coordinates(dv)) * (d - c)
            ),
        )
        rows = []
        if min(abs(u - a), abs(u - b)) <= param_round * (b - a):
            i = 1 if abs(u - a) <= abs(u - b) else spline.NbUPoles()
            rows.append([_coordinates(spline.Pole(i, j)) for j in range(1, spline.NbVPoles() + 1)])
        if min(abs(v - c), abs(v - d)) <= param_round * (d - c):
            j = 1 if abs(v - c) <= abs(v - d) else spline.NbVPoles()
            rows.append([_coordinates(spline.Pole(i, j)) for i in range(1, spline.NbUPoles() + 1)])
        limiting |= any(
            np.max(np.linalg.norm(np.array(row) - row[0], axis=1)) <= row_tol for row in rows
        )
    if limiting:
        duu, dvv, duv = gp_Vec(), gp_Vec(), gp_Vec()
        surface.D2(float(u), float(v), p, du, dv, duu, dvv, duv)
        normal = max(
            (_coordinates(du.Crossed(duv)), _coordinates(duv.Crossed(dv))), key=np.linalg.norm
        )
    return normal / np.linalg.norm(normal) if np.linalg.norm(normal) else None


def _bernstein_product(left, right):
    p, q = np.array(left.shape) - 1
    r, s = np.array(right.shape) - 1
    a = np.outer([math.comb(p, i) for i in range(p + 1)], [math.comb(q, j) for j in range(q + 1)])
    b = np.outer([math.comb(r, i) for i in range(r + 1)], [math.comb(s, j) for j in range(s + 1)])
    c = np.outer(
        [math.comb(p + r, i) for i in range(p + r + 1)],
        [math.comb(q + s, j) for j in range(q + s + 1)],
    )
    return convolve2d(left * a, right * b) / c


def _stationary_seeds(spline, bounds, axis):
    """Reject knot spans whose rational coordinate derivative cannot vanish."""
    conversion = GeomConvert_BSplineSurfaceToBezierSurface(spline)
    uk = TColStd_Array1OfReal(1, conversion.NbUPatches() + 1)
    vk = TColStd_Array1OfReal(1, conversion.NbVPatches() + 1)
    conversion.UKnots(uk)
    conversion.VKnots(vk)
    for i in range(1, conversion.NbUPatches() + 1):
        for j in range(1, conversion.NbVPatches() + 1):
            low = np.maximum((uk.Value(i), vk.Value(j)), (bounds[0], bounds[2]))
            high = np.minimum((uk.Value(i + 1), vk.Value(j + 1)), (bounds[1], bounds[3]))
            if (high <= low).any():
                continue
            patch = conversion.Patch(i, j)
            p, q = patch.UDegree(), patch.VDegree()
            weights = np.array(
                [[patch.Weight(a + 1, b + 1) for b in range(q + 1)] for a in range(p + 1)]
            )
            coordinates = np.array(
                [
                    [patch.Pole(a + 1, b + 1).Coord(axis + 1) for b in range(q + 1)]
                    for a in range(p + 1)
                ]
            )
            numerator = (coordinates - coordinates[0, 0]) * weights
            gu = _bernstein_product(p * np.diff(numerator, axis=0), weights) - _bernstein_product(
                numerator, p * np.diff(weights, axis=0)
            )
            gv = _bernstein_product(q * np.diff(numerator, axis=1), weights) - _bernstein_product(
                numerator, q * np.diff(weights, axis=1)
            )
            guard = 128 * np.finfo(float).eps * max(np.max(np.abs(gu)), np.max(np.abs(gv)))
            if gu.min() > guard or gu.max() < -guard or gv.min() > guard or gv.max() < -guard:
                continue
            for u, v in itertools.product(
                np.linspace(low[0], high[0], p + 1), np.linspace(low[1], high[1], q + 1)
            ):
                yield np.array(
                    (
                        (u - bounds[0]) / (bounds[1] - bounds[0]),
                        (v - bounds[2]) / (bounds[3] - bounds[2]),
                    )
                )


def _numeric_stationary(surface, bounds, axis, poles=(), pole_tolerance=0.0):
    """Find stationary coordinates with a regular physical surface normal."""
    u0, u1, v0, v1 = bounds
    span = np.array((u1 - u0, v1 - v0))
    if not np.isfinite(bounds).all() or min(span) <= 0:
        return []
    values = []
    adaptor = GeomAdaptor_Surface(surface)
    if adaptor.GetType() == GeomAbs_SurfaceOfExtrusion:
        if abs(_coordinates(adaptor.Direction())[axis]) > 32 * np.finfo(float).eps:
            return []
        curve = adaptor.BasisCurve()
        grid = np.linspace(u0, u1, 65)

        def derivative(t):
            p, d = gp_Pnt(), gp_Vec()
            curve.D1(float(t), p, d)
            return d.Coord(axis + 1)

        for left, right in zip(grid[:-1], grid[1:]):
            a, b = derivative(left), derivative(right)
            if a == 0:
                t = left
            elif a * b < 0:
                t = brentq(derivative, left, right, xtol=32 * np.finfo(float).eps * span[0])
            else:
                continue
            normal = _regular_normal(surface, t, (v0 + v1) / 2, poles, pole_tolerance)
            if normal is not None and 1 - abs(normal[axis]) <= 1e-10:
                values.append(surface.Value(t, (v0 + v1) / 2).Coord(axis + 1))
        return values

    def gradient(q):
        uv = np.array((u0, v0)) + q * span
        p, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
        surface.D1(float(uv[0]), float(uv[1]), p, du, dv)
        return np.array((du.Coord(axis + 1), dv.Coord(axis + 1))) * span

    seeds = itertools.product(np.linspace(0, 1, 7), np.linspace(0, 1, 7))
    if adaptor.GetType() == GeomAbs_BSplineSurface:
        seeds = _stationary_seeds(adaptor.BSpline(), bounds, axis)
    for seed in seeds:
        solution = root(gradient, seed)
        q = solution.x
        if not np.isfinite(q).all() or (q < -1e-10).any() or (q > 1 + 1e-10).any():
            continue
        uv = np.array((u0, v0)) + np.clip(q, 0, 1) * span
        normal = _regular_normal(surface, *uv, poles, pole_tolerance)
        if normal is not None and 1 - abs(normal[axis]) <= 1e-10:
            values.append(surface.Value(float(uv[0]), float(uv[1])).Coord(axis + 1))
    return list(set(values))


def _sensitive_section(shape, axis, position, deflection, scale, slabs=None):
    key = (hash(shape), float(scale), float(deflection))
    router = _RAW_ROUTERS.get(key)
    if router is None or not router.shape.IsSame(shape):
        router = _SurfaceRouter(shape, scale, deflection, slabs)
        _RAW_ROUTERS[key] = router
        while len(_RAW_ROUTERS) > 16:
            _RAW_ROUTERS.popitem(last=False)
    else:
        _RAW_ROUTERS.move_to_end(key)
    return router.section(axis, position) if router.sensitive(axis, position) else None


class _SurfaceRouter:
    """Cheap face-slab and stationary-coordinate index; prepare only on demand."""

    def __init__(self, shape, scale, deflection, slabs=None):
        self.shape = shape
        self.scale = float(scale)
        self.deflection = max(float(deflection) * self.scale, 1e-7) / self.scale
        self.slabs = _FaceSlabIndex(shape) if slabs is None else slabs
        face_tol = max((BRep_Tool.Tolerance(f) for f in self.slabs.faces), default=1e-7)
        self.band = max(8 * self.deflection * self.scale, 4 * face_tol, 4e-7)
        self.engine = None
        self._engines = {}
        entries = [[], [], []]
        for fi, face in enumerate(self.slabs.faces):
            surface = BRepAdaptor_Surface(face)
            kind = surface.GetType()
            if kind == GeomAbs_Plane:
                continue
            if kind == GeomAbs_Cylinder:
                value = surface.Cylinder()
                frame = value.Position()
                c = _coordinates(frame.Location())
                a = _coordinates(frame.Direction())
                radial = np.hypot(
                    _coordinates(frame.XDirection()), _coordinates(frame.YDirection())
                )
                for axis in range(3):
                    if abs(a[axis]) <= 32 * np.finfo(float).eps:
                        entries[axis].extend(
                            (c[axis] + sign * value.Radius() * radial[axis], fi) for sign in (-1, 1)
                        )
            elif kind == GeomAbs_Torus:
                value = surface.Torus()
                frame = value.Position()
                c = _coordinates(frame.Location())
                radial = np.hypot(
                    _coordinates(frame.XDirection()), _coordinates(frame.YDirection())
                )
                for axis in range(3):
                    entries[axis].extend(
                        (
                            c[axis]
                            + s * value.MajorRadius() * radial[axis]
                            + t * value.MinorRadius(),
                            fi,
                        )
                        for s in (-1, 1)
                        for t in (-1, 1)
                    )
            elif kind == GeomAbs_Cone:
                value = surface.Cone()
                frame = value.Position()
                a = _coordinates(frame.Direction())
                if abs(math.sin(value.SemiAngle())) > np.finfo(float).eps:
                    apex = (
                        _coordinates(frame.Location())
                        - value.RefRadius() / math.tan(value.SemiAngle()) * a
                    )
                    for axis in range(3):
                        if (
                            abs(abs(a[axis]) - abs(math.sin(value.SemiAngle())))
                            <= 32 * np.finfo(float).eps
                        ):
                            entries[axis].append((apex[axis], fi))
            elif kind == GeomAbs_Sphere:
                value = surface.Sphere()
                c = _coordinates(value.Location())
                for axis in range(3):
                    entries[axis].extend((c[axis] + sign * value.Radius(), fi) for sign in (-1, 1))
            else:
                geometry = BRep_Tool.Surface(face)
                bounds = breptools.UVBounds(face)
                poles = []
                edges = TopExp_Explorer(face, TopAbs_EDGE)
                while edges.More():
                    edge = topods.Edge(edges.Current())
                    edges.Next()
                    if not BRep_Tool.Degenerated(edge):
                        continue
                    pc, a, b = BRep_Tool.CurveOnSurface(edge, face)
                    start, end = pc.Value(a), pc.Value(b)
                    if start.X() == end.X():
                        poles.append((0, start.X()))
                    if start.Y() == end.Y():
                        poles.append((1, start.Y()))
                for axis in range(3):
                    entries[axis].extend(
                        (p, fi)
                        for p in _numeric_stationary(
                            geometry, bounds, axis, poles, BRep_Tool.Tolerance(face)
                        )
                    )
        self.critical = []
        self.owners = []
        for entries_axis in entries:
            pairs = sorted(entries_axis)
            self.critical.append(np.array([p for p, _ in pairs], dtype=float))
            self.owners.append(np.array([i for _, i in pairs], dtype=int))

    def sensitive(self, axis, position):
        pos = position * self.scale
        critical = self.critical[axis]
        lo, hi = np.searchsorted(critical, (pos - self.band, pos + self.band))
        faces = self.owners[axis][lo:hi]
        return bool(
            (
                (self.slabs.lo[faces, axis] <= pos + self.band)
                & (self.slabs.hi[faces, axis] >= pos - self.band)
            ).any()
        )

    def section(self, axis, position):
        try:
            return self.resolved_section(axis, position)
        except _CoplanarSection:
            return None
        except _SectionFailure as error:
            raise RuntimeError(
                f"Cannot resolve the near-tangent cross-section at {'xyz'[axis]}="
                f"{position:.12g} m: {error}. No partial material coverage was accepted. "
                "Check the CAD body's validity and boundary tolerances."
            ) from error

    def resolved_section(self, axis, position):
        if self.engine is None:
            self.engine = _SurfaceSectionEngine(self.shape, self.scale)
        self._engines[self.engine.conditioning] = self.engine
        # Kernel accuracy need not improve monotonically with measurement scale.
        # Keep four bounded choices and retry the baseline before increasing scale.
        choices = [self.engine.conditioning]
        choices.extend(c for c in (1, 4, 16, 64) if c not in choices)
        for index, conditioning in enumerate(choices):
            if conditioning not in self._engines:
                self._engines[conditioning] = _SurfaceSectionEngine(
                    self.shape, self.scale, conditioning
                )
            self.engine = self._engines[conditioning]
            try:
                return self.engine.section(axis, position, self.deflection * 0.1)[0]
            except _PrecisionFailure:
                if index == len(choices) - 1:
                    raise


class _PreparedGeometry:
    def __init__(self, occ):
        self.occ = occ
        self.slabs = _FaceSlabIndex(occ)
        self.edges = TopTools_IndexedMapOfShape()
        topexp.MapShapes(occ, TopAbs_EDGE, self.edges)
        self.surfaces = []
        for face in self.slabs.faces:
            surface = BRep_Tool.Surface(face)
            # Keep the native support domain for intersection; use its basis
            # periodicity only when evaluating and wrapping face pcurves.
            while surface.DynamicType().Name() == "Geom_RectangularTrimmedSurface":
                surface = Geom_RectangularTrimmedSurface.DownCast(surface).BasisSurface()
            self.surfaces.append(surface)
        self.bounds = [breptools.UVBounds(face) for face in self.slabs.faces]
        self.borders = []
        for face in self.slabs.faces:
            borders = []
            ex = TopExp_Explorer(face, TopAbs_EDGE)
            while ex.More():
                edge = topods.Edge(ex.Current())
                ex.Next()
                # Embedded coedges are imprint data, not material trim boundaries.
                if edge.Orientation() in (TopAbs_INTERNAL, TopAbs_EXTERNAL):
                    continue
                pc, first, last = BRep_Tool.CurveOnSurface(edge, face)
                if pc is not None and last > first:
                    borders.append(
                        (
                            self.edges.FindIndex(edge) - 1,
                            pc,
                            first,
                            last,
                            Geom2d_TrimmedCurve(pc, first, last),
                        )
                    )
            self.borders.append(borders)


def _coordinates(point):
    return np.array((point.X(), point.Y(), point.Z()))


def _range_on_box(curve, low, high):
    first, last = (curve.FirstParameter(), curve.LastParameter())
    if abs(first) < 1e50 and abs(last) < 1e50:
        return (first, last)
    params = []
    for corner in itertools.product(*zip(low, high)):
        proj = GeomAPI_ProjectPointOnCurve(gp_Pnt(*corner), curve)
        if proj.NbPoints():
            params.append(proj.LowerDistanceParameter())
    if not params:
        raise _SectionFailure("Unbounded trace has no finite parameter enclosure")
    width = max(params) - min(params)
    return (min(params) - 0.01 * width, max(params) + 0.01 * width)


def _tessellate(curve, lo, hi, budget):
    if not math.isfinite(lo + hi + budget) or hi <= lo or budget <= 0:
        raise _SectionFailure("Invalid trace interval or chord budget")
    adaptor = GeomAdaptor_Curve(curve, lo, hi)
    discretizer = GCPnts_TangentialDeflection(adaptor, math.radians(5), budget * 0.5)
    if discretizer.NbPoints() < 2:
        raise _SectionFailure("Trace tessellation returned no interval")
    return np.array(
        [_coordinates(discretizer.Value(i)) for i in range(1, discretizer.NbPoints() + 1)]
    )


def _strict_inside(face, u, v, tol, borders=None):
    # Only a transverse, even full-line crossing count defines a closed trim.
    # Try another direction when a ray hits a tangent, vertex gap or overlap.
    bounds = breptools.UVBounds(face)
    if u < bounds[0] - tol or u > bounds[1] + tol or v < bounds[2] - tol or v > bounds[3] + tol:
        return False
    if borders is None:
        borders = []
        explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while explorer.More():
            edge = topods.Edge(explorer.Current())
            explorer.Next()
            if edge.Orientation() in (TopAbs_INTERNAL, TopAbs_EXTERNAL):
                continue
            pc, first, last = BRep_Tool.CurveOnSurface(edge, face)
            if pc is not None and last > first:
                borders.append((-1, pc, first, last, Geom2d_TrimmedCurve(pc, first, last)))
    for dx, dy in (
        (0, 1),
        (1, 0),
        (1, 1),
        (1, -1),
        (1, 2),
        (2, -1),
        (1, np.sqrt(2)),
        (np.sqrt(3), 1),
    ):
        direction = gp_Dir2d(dx, dy)
        ray = Geom2d_Line(gp_Pnt2d(u, v), direction)
        total = forward = 0
        ambiguous = False
        for _, pc, first, last, border in borders:
            result = Geom2dAPI_InterCurveCurve(border, ray, tol)
            algo = result.Intersector()
            if not algo.IsDone():
                raise _SectionFailure("UV parity intersection failed")
            if result.NbSegments():
                ambiguous = True
                break
            for i in range(1, result.NbPoints() + 1):
                point = algo.Point(i)
                transition = point.TransitionOfFirst()
                if transition.IsTangent() or transition.TransitionType() == IntRes2d_Undecided:
                    ambiguous = True
                    break
                position = transition.PositionOnCurve()
                count = True
                if position in (IntRes2d_Head, IntRes2d_End):
                    p, tangent = gp_Pnt2d(), gp_Vec2d()
                    pc.D1(first if position == IntRes2d_Head else last, p, tangent)
                    side = -direction.Y() * tangent.X() + direction.X() * tangent.Y()
                    if abs(side) <= 64 * np.finfo(float).eps * tangent.Magnitude():
                        ambiguous = True
                        break
                    count = side > 0 if position == IntRes2d_Head else side < 0
                total += int(count)
                forward += int(count and point.ParamOnSecond() > tol)
            if ambiguous:
                break
        if not ambiguous and total % 2 == 0:
            return bool(forward % 2)
    raise _SectionFailure("No unambiguous UV parity ray")


def _checked_pcurve(curve, surface, lo, hi, tol, coupled=None):
    projection = ProjLib_ProjectedCurve(
        GeomAdaptor_Surface(surface), GeomAdaptor_Curve(curve, lo, hi), tol
    )
    initial = geom2dadaptor.MakeCurve(projection)

    def residual(pc, params):
        errors = []
        for t in params:
            uv = pc.Value(float(t))
            errors.append(
                np.linalg.norm(
                    _coordinates(surface.Value(uv.X(), uv.Y()))
                    - _coordinates(curve.Value(float(t)))
                )
            )
        return max(errors)

    adaptor = GeomAdaptor_Curve(curve, lo, hi)
    breaks = [lo, hi]
    if adaptor.GetType() == GeomAbs_BSplineCurve:
        spline = adaptor.BSpline()
        breaks.extend(
            spline.Knot(i) for i in range(1, spline.NbKnots() + 1) if lo < spline.Knot(i) < hi
        )
    breaks = np.unique(breaks)
    checks = np.unique(
        np.concatenate(
            [
                np.linspace(lo, hi, 17),
                *(breaks[:-1] + f * np.diff(breaks) for f in (0.25, 0.5, 0.75)),
            ]
        )
    )
    if coupled is not None and residual(coupled, checks) <= tol:
        return coupled
    if residual(initial, checks) <= tol:
        return initial
    analysis = ShapeAnalysis_Surface(surface)
    best_error = float("inf")
    for n in (17, 33, 65, 129, 257, 513):
        params = np.unique(np.concatenate([np.linspace(lo, hi, n), breaks]))
        n_points = len(params)
        previous = (
            coupled.Value(lo) if coupled is not None else analysis.ValueOfUV(curve.Value(lo), tol)
        )
        values_list = []
        for t in params:
            previous = analysis.NextValueOfUV(previous, curve.Value(float(t)), tol)
            values_list.append((previous.X(), previous.Y()))
        values = np.array(values_list)
        if surface.IsUPeriodic():
            values[:, 0] = np.unwrap(values[:, 0], period=surface.UPeriod())
        if surface.IsVPeriodic():
            values[:, 1] = np.unwrap(values[:, 1], period=surface.VPeriod())
        pts = TColgp_HArray1OfPnt2d(1, n_points)
        ts = TColStd_HArray1OfReal(1, n_points)
        for i in range(n_points):
            pts.SetValue(i + 1, gp_Pnt2d(*values[i]))
            ts.SetValue(i + 1, float(params[i]))
        interpolate = Geom2dAPI_Interpolate(pts, ts, False, 1e-14)
        interpolate.Perform()
        if not interpolate.IsDone():
            raise _SectionFailure("Adaptive pcurve interpolation failed")
        pc = interpolate.Curve()
        check = np.concatenate([params[:-1] + f * np.diff(params) for f in (0.25, 0.5, 0.75)])
        best_error = residual(pc, np.unique(np.concatenate([check, checks])))
        if best_error <= tol:
            return pc
    raise _PrecisionFailure(f"Adaptive pcurve residual remains {best_error:.6g}")


def _refine_trace(curve, surface, seed_curve, lo, hi, axis, position, tol, budget, nodes=None):
    """Refine an approximate kernel trace on the surface/plane intersection."""
    analysis = ShapeAnalysis_Surface(surface)
    breaks = [lo, hi]
    adaptor = GeomAdaptor_Curve(curve, lo, hi)
    if adaptor.GetType() == GeomAbs_BSplineCurve:
        spline = adaptor.BSpline()
        breaks.extend(
            spline.Knot(i) for i in range(1, spline.NbKnots() + 1) if lo < spline.Knot(i) < hi
        )
    breaks = np.unique(breaks)
    params = np.unique(
        np.concatenate(
            [
                np.linspace(lo, hi, 17),
                breaks,
                *(breaks[:-1] + f * np.diff(breaks) for f in (0.25, 0.5, 0.75)),
            ]
        )
    )
    for _ in range(16):
        count = len(params)
        if count > 16384:
            break
        values, points = [], []
        previous = None
        for t in params:
            target = curve.Value(float(t))
            if t in (lo, hi):
                inside = lo + 0.01 * (hi - lo) if t == lo else hi - 0.01 * (hi - lo)
                reference = analysis.ValueOfUV(curve.Value(inside), tol)
                seed = analysis.NextValueOfUV(reference, target, tol)
            elif seed_curve is not None:
                seed = seed_curve.Value(float(t))
            elif previous is None:
                seed = analysis.ValueOfUV(target, tol)
            else:
                seed = analysis.NextValueOfUV(previous, target, tol)
            uv = np.array((seed.X(), seed.Y()))
            if nodes and nodes.get(t):
                options = []
                for edge_id, edge_param, border, degenerate in nodes[t]:
                    if degenerate:
                        projection = Geom2dAPI_ProjectPointOnCurve(gp_Pnt2d(*uv), border)
                        option = border.Value(projection.LowerDistanceParameter())
                    else:
                        option = border.Value(edge_param)
                    options.append(np.array((option.X(), option.Y())))
                uv = min(options, key=lambda option: float(np.linalg.norm(option - uv))).copy()
            for _ in range(32):
                p, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
                surface.D1(float(uv[0]), float(uv[1]), p, du, dv)
                residual = p.Coord(axis + 1) - position
                # Leave residual headroom for interpolation between solved samples.
                if abs(residual) <= tol / 16:
                    break
                gradient = np.array((du.Coord(axis + 1), dv.Coord(axis + 1)))
                denominator = float(np.dot(gradient, gradient))
                if denominator <= np.finfo(float).tiny:
                    raise _PrecisionFailure("Stationary unresolved surface/plane trace")
                uv -= residual * gradient / denominator
            else:
                raise _PrecisionFailure("Surface/plane trace refinement did not converge")
            point = _coordinates(surface.Value(float(uv[0]), float(uv[1])))
            normal = _coordinates(du.Crossed(dv))
            inclination = np.linalg.norm(np.delete(normal, axis)) / max(
                np.linalg.norm(normal), np.finfo(float).tiny
            )
            # A kernel trace's surface-position error is amplified by
            # 1/sin(angle) when projected onto a nearly tangent plane.
            # This guards branch displacement, not the final residual.
            neighbourhood = 8 * budget / max(inclination, np.sqrt(np.finfo(float).eps))
            if np.linalg.norm(point - _coordinates(target)) > neighbourhood:
                raise _PrecisionFailure(
                    "Surface/plane refinement left its kernel trace neighbourhood"
                )
            previous = gp_Pnt2d(*uv)
            values.append(uv)
            points.append(point)
        values = np.array(values)
        if surface.IsUPeriodic():
            values[:, 0] = np.unwrap(values[:, 0], period=surface.UPeriod())
        if surface.IsVPeriodic():
            values[:, 1] = np.unwrap(values[:, 1], period=surface.VPeriod())
        array2 = TColgp_HArray1OfPnt2d(1, count)
        array3 = TColgp_HArray1OfPnt(1, count)
        parameters = TColStd_HArray1OfReal(1, count)
        for i, (uv, point, t) in enumerate(zip(values, points, params)):
            array2.SetValue(i + 1, gp_Pnt2d(*uv))
            array3.SetValue(i + 1, gp_Pnt(*point))
            parameters.SetValue(i + 1, float(t))
        point_spacing = float(np.min(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        if point_spacing == 0:
            raise _PrecisionFailure("Trace interpolation samples coincide at measurement precision")
        # OCC's constructor tolerance checks sample coincidence, not curve accuracy.
        # Retain distinct adaptive samples; residual checks below keep the same budget.
        fit2 = Geom2dAPI_Interpolate(array2, parameters, False, 1e-14)
        fit3 = GeomAPI_Interpolate(array3, parameters, False, min(tol * 0.01, point_spacing / 4))
        fit2.Perform()
        fit3.Perform()
        if not fit2.IsDone() or not fit3.IsDone():
            raise _PrecisionFailure("Refined trace interpolation failed")
        pc, refined = fit2.Curve(), fit3.Curve()
        worst = 0.0
        insert = []
        for fraction in (0.25, 0.5, 0.75):
            for t in params[:-1] + fraction * np.diff(params):
                uv = pc.Value(float(t))
                exact = _coordinates(surface.Value(uv.X(), uv.Y()))
                approx = _coordinates(refined.Value(float(t)))
                error = max(abs(exact[axis] - position), np.linalg.norm(exact - approx))
                worst = max(worst, error)
                if error > tol:
                    insert.append(t)
        if worst <= tol:
            return refined, pc
        params = np.unique(np.concatenate([params, insert]))
    raise _PrecisionFailure(f"Refined surface/plane trace residual remains {worst:.6g}")


def _trace_pieces(
    curve, lo, hi, face, surface, prepared, face_index, plane, axis, position, tol, budget
):
    """Split a support trace at closing and singular B-Rep boundaries."""
    nodes = []
    for edge_id, pc, first, last, border in prepared.borders[face_index]:
        edge = topods.Edge(prepared.edges.FindKey(edge_id + 1))
        degenerate = BRep_Tool.Degenerated(edge)
        if not degenerate and not BRep_Tool.IsClosed(edge, face):
            continue
        if degenerate:
            point = BRep_Tool.Pnt(topexp.FirstVertex(edge))
            if abs(point.Coord(axis + 1) - position) > tol:
                continue
            candidates = [(0.0, point)]
        else:
            edge_curve, a, b = BRep_Tool.Curve(edge)
            inter = GeomAPI_IntCS(Geom_TrimmedCurve(edge_curve, a, b), plane)
            if not inter.IsDone():
                raise _SectionFailure("Closing-edge/plane intersection failed")
            candidates = [
                (inter.Parameters(i)[2], inter.Point(i)) for i in range(1, inter.NbPoints() + 1)
            ]
        for parameter, point in candidates:
            if not degenerate:
                for _ in range(32):
                    uv, derivative = gp_Pnt2d(), gp_Vec2d()
                    pc.D1(parameter, uv, derivative)
                    p, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
                    surface.D1(uv.X(), uv.Y(), p, du, dv)
                    residual = p.Coord(axis + 1) - position
                    if abs(residual) <= tol:
                        point = p
                        break
                    slope = (
                        du.Coord(axis + 1) * derivative.X() + dv.Coord(axis + 1) * derivative.Y()
                    )
                    if abs(slope) <= np.finfo(float).tiny:
                        raise _PrecisionFailure("Stationary unresolved closing-edge crossing")
                    parameter = min(max(parameter - residual / slope, first), last)
                else:
                    raise _PrecisionFailure("Closing-edge/plane refinement failed")
            projection = GeomAPI_ProjectPointOnCurve(point, curve, lo, hi)
            if not projection.NbPoints():
                distance, t = min(
                    (np.linalg.norm(_coordinates(curve.Value(t)) - _coordinates(point)), t)
                    for t in (lo, hi)
                )
                if distance > max(BRep_Tool.Tolerance(edge), tol):
                    continue
            else:
                if projection.LowerDistance() > 8 * budget:
                    continue
                t = projection.LowerDistanceParameter()
            rounding = 64 * np.finfo(float).eps * max(1, abs(lo), abs(hi))
            if abs(t - lo) <= rounding:
                t = lo
            elif abs(t - hi) <= rounding:
                t = hi
            nodes.append((t, edge_id, parameter, border, degenerate))
            for endpoint in (lo, hi):
                if np.linalg.norm(_coordinates(curve.Value(endpoint)) - _coordinates(point)) <= tol:
                    nodes.append((endpoint, edge_id, parameter, border, degenerate))
    cuts = sorted(set([lo, hi, *(n[0] for n in nodes)]))
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b <= a:
            continue
        ends = {
            t: [
                (eid, parameter, border, degenerate)
                for cut, eid, parameter, border, degenerate in nodes
                if cut == t
            ]
            for t in (a, b)
        }
        yield a, b, ends


def _trimmed_segments(occ, axis, position, tol=1e-12, budget=2.5e-08, uv_tol=1e-12, prepared=None):
    prepared = _PreparedGeometry(occ) if prepared is None else prepared
    slabs = prepared.slabs
    origin = np.zeros(3)
    origin[axis] = position
    normal = np.zeros(3)
    normal[axis] = 1
    plane = Geom_Plane(gp_Pln(gp_Pnt(*origin), gp_Dir(*normal)))
    segments = []
    records = []
    diagnostics = []
    candidates = np.flatnonzero(
        (slabs.lo[:, axis] <= position + tol) & (slabs.hi[:, axis] >= position - tol)
    )
    for index in candidates:
        face = slabs.faces[index]
        low = slabs.lo[index]
        high = slabs.hi[index]
        if position < low[axis] - tol or position > high[axis] + tol:
            continue
        surface = prepared.surfaces[index]
        solver = GeomInt_IntSS()
        solver.Perform(BRep_Tool.Surface(face), plane, tol, True, True, True)
        if not solver.IsDone():
            raise _SectionFailure(f"Support intersection failed on face {index}")
        bounds = prepared.bounds[index]
        for line_no in range(1, solver.NbLines() + 1):
            curve = solver.Line(line_no)
            lo, hi = _range_on_box(curve, low, high)
            coupled = solver.LineOnS1(line_no) if solver.HasLineOnS1(line_no) else None
            kernel_curve = curve
            for lo, hi, nodes in _trace_pieces(
                kernel_curve,
                lo,
                hi,
                face,
                surface,
                prepared,
                index,
                plane,
                axis,
                position,
                tol,
                budget,
            ):
                curve = kernel_curve
                try:
                    pc = _checked_pcurve(curve, surface, lo, hi, tol, coupled)
                except _PrecisionFailure:
                    curve, pc = _refine_trace(
                        curve, surface, coupled, lo, hi, axis, position, tol, budget, nodes
                    )
                if pc is None:
                    raise _SectionFailure(f"Trace projection failed on face {index}")
                shifts_u = (
                    (-surface.UPeriod(), 0, surface.UPeriod()) if surface.IsUPeriodic() else (0,)
                )
                shifts_v = (
                    (-surface.VPeriod(), 0, surface.VPeriod()) if surface.IsVPeriodic() else (0,)
                )
                projected_copies = []
                for du, dv in itertools.product(shifts_u, shifts_v):
                    copy = Geom2d_Curve.DownCast(pc.Copy())
                    copy.Translate(gp_Vec2d(du, dv))
                    projected_copies.append(Geom2d_TrimmedCurve(copy, lo, hi))
                cuts = [lo, hi]
                cut_tags = [
                    (t, eid, 0.0 if degenerate else parameter)
                    for t, tags in nodes.items()
                    for eid, parameter, _, degenerate in tags
                ]
                for edge_id, _, first, last, boundary in prepared.borders[index]:
                    for trimmed_pc in projected_copies:
                        inter = Geom2dAPI_InterCurveCurve(trimmed_pc, boundary, uv_tol)
                        algo = inter.Intersector()
                        if not algo.IsDone():
                            raise _SectionFailure(f"Trim intersection failed on face {index}")
                        for i in range(1, inter.NbPoints() + 1):
                            point = algo.Point(i)
                            cuts.append(point.ParamOnFirst())
                            edge = topods.Edge(prepared.edges.FindKey(edge_id + 1))
                            parameter = (
                                0.0 if BRep_Tool.Degenerated(edge) else point.ParamOnSecond()
                            )
                            cut_tags.append((point.ParamOnFirst(), edge_id, parameter))
                cuts.sort()
                unique = [cuts[0]]
                for t in cuts[1:]:
                    if abs(t - unique[-1]) > uv_tol:
                        unique.append(t)
                accepted = []
                classifications = []
                for t0, t1 in zip(unique[:-1], unique[1:]):
                    uv = pc.Value((t0 + t1) / 2)
                    u, v = (uv.X(), uv.Y())
                    if surface.IsUPeriodic():
                        u += (
                            round(((bounds[0] + bounds[1]) / 2 - u) / surface.UPeriod())
                            * surface.UPeriod()
                        )
                    if surface.IsVPeriodic():
                        v += (
                            round(((bounds[2] + bounds[3]) / 2 - v) / surface.VPeriod())
                            * surface.VPeriod()
                        )
                    strict = _strict_inside(face, u, v, uv_tol, prepared.borders[index])
                    classifications.append({"range": (t0, t1), "uv": (u, v), "strict": strict})
                    if not strict:
                        continue
                    normals = []
                    for t in (t0 + 0.25 * (t1 - t0), (t0 + t1) / 2, t0 + 0.75 * (t1 - t0)):
                        sample_uv = pc.Value(t)
                        point, du, dv = (gp_Pnt(), gp_Vec(), gp_Vec())
                        surface.D1(sample_uv.X(), sample_uv.Y(), point, du, dv)
                        n = _coordinates(du.Crossed(dv))
                        if np.linalg.norm(n) == 0:
                            raise _SectionFailure("Singular surface normal on trace")
                        normals.append(np.linalg.norm(np.cross(normal, n / np.linalg.norm(n))))
                    if max(normals) < 1e-12:
                        continue
                    segment = _tessellate(curve, t0, t1, budget)
                    uv_mid = pc.Value((t0 + t1) / 2)
                    exact_mid = _coordinates(surface.Value(uv_mid.X(), uv_mid.Y()))
                    correspondence_error = np.linalg.norm(
                        exact_mid - _coordinates(curve.Value((t0 + t1) / 2))
                    )
                    if correspondence_error > tol:
                        raise _PrecisionFailure(
                            f"Trace residual {correspondence_error:.6g} on face {index}; "
                            f"kernel reached {solver.TolReached3d():.6g}/"
                            f"{solver.TolReached2d():.6g}"
                        )
                    segments.append(segment)
                    endpoint_tags = [
                        [(eid, param) for cut, eid, param in cut_tags if abs(cut - t) <= uv_tol]
                        for t in (t0, t1)
                    ]
                    sample_point, du, dv, tangent = (gp_Pnt(), gp_Vec(), gp_Vec(), gp_Vec())
                    surface.D1(uv_mid.X(), uv_mid.Y(), sample_point, du, dv)
                    curve.D1((t0 + t1) / 2, sample_point, tangent)
                    outward = _coordinates(du.Crossed(dv))
                    if face.Orientation() == TopAbs_REVERSED:
                        outward = -outward
                    direction = float(np.dot(np.cross(normal, outward), _coordinates(tangent)))
                    if direction == 0:
                        raise _SectionFailure("Trace orientation is singular")
                    records.append(
                        {
                            "curve": curve,
                            "params": [t0, t1],
                            "face": index,
                            "line": line_no,
                            "edge_tags": endpoint_tags,
                            "direction": direction,
                            "axis": axis,
                            "support_closed": kernel_curve.IsClosed(),
                            "support_range": (
                                kernel_curve.FirstParameter(),
                                kernel_curve.LastParameter(),
                            ),
                        }
                    )
                    accepted.append((t0, t1))
                diagnostics.append(
                    {
                        "face": index,
                        "surface_kind": int(BRepAdaptor_Surface(face).GetType()),
                        "line": line_no,
                        "bounds": bounds,
                        "cuts": unique,
                        "kept": accepted,
                        "classifications": classifications,
                    }
                )
    return (segments, diagnostics, records)


def _conservative_face_bounds(face, surface, uv):
    adaptor = BRepAdaptor_Surface(face)
    kind = adaptor.GetType()
    if kind == GeomAbs_Plane:
        points = [_coordinates(surface.Value(u, v)) for u in uv[:2] for v in uv[2:]]
        return (np.min(points, axis=0), np.max(points, axis=0))
    if kind in (GeomAbs_Cylinder, GeomAbs_Cone):
        value = adaptor.Cylinder() if kind == GeomAbs_Cylinder else adaptor.Cone()
        frame = value.Position()
        centre = _coordinates(frame.Location())
        axis = _coordinates(frame.Direction())
        radial = np.hypot(_coordinates(frame.XDirection()), _coordinates(frame.YDirection()))
        lows, highs = ([], [])
        for v in uv[2:]:
            if kind == GeomAbs_Cylinder:
                c, radius = (centre + v * axis, value.Radius())
            else:
                angle = value.SemiAngle()
                c = centre + v * np.cos(angle) * axis
                radius = abs(value.RefRadius() + v * np.sin(angle))
            lows.append(c - radius * radial)
            highs.append(c + radius * radial)
        return (np.min(lows, axis=0), np.max(highs, axis=0))
    if kind == GeomAbs_Sphere:
        sphere = adaptor.Sphere()
        c = _coordinates(sphere.Location())
        return (c - sphere.Radius(), c + sphere.Radius())
    if kind == GeomAbs_Torus:
        torus = adaptor.Torus()
        frame = torus.Position()
        c = _coordinates(frame.Location())
        radial = np.hypot(_coordinates(frame.XDirection()), _coordinates(frame.YDirection()))
        extent = torus.MajorRadius() * radial + torus.MinorRadius()
        return (c - extent, c + extent)
    if kind == GeomAbs_BSplineSurface:
        surface = adaptor.BSpline()
        if any(
            (
                surface.Weight(i, j) <= 0
                for i in range(1, surface.NbUPoles() + 1)
                for j in range(1, surface.NbVPoles() + 1)
            )
        ):
            return None
        points = [
            _coordinates(surface.Pole(i, j))
            for i in range(1, surface.NbUPoles() + 1)
            for j in range(1, surface.NbVPoles() + 1)
        ]
        return (np.min(points, axis=0), np.max(points, axis=0))
    return None


def _endpoint_partners(records):
    occurrences = defaultdict(list)
    for i, record in enumerate(records):
        for end, tags in enumerate(record["edge_tags"]):
            for edge_id, param in set(tags):
                occurrences[record["face"], edge_id].append((param, 2 * i + end))
    groups = defaultdict(set)
    for (_, edge_id), items in occurrences.items():
        unique = []
        for param, endpoint in sorted(items):
            rounding = 64 * np.finfo(float).eps * max(1, abs(param))
            if not unique or abs(param - unique[-1][0]) > rounding:
                unique.append((param, {endpoint}))
            else:
                unique[-1][1].add(endpoint)
        for rank, (_, endpoints) in enumerate(unique):
            groups[edge_id, rank].update(endpoints)
    candidates = defaultdict(set)
    shared_edges = {}
    for (edge_id, rank), endpoints in groups.items():
        if len(endpoints) == 2:
            pairs = [sorted(endpoints)]
        elif len(endpoints) > 2 and len(endpoints) % 2 == 0:
            incoming, outgoing = [], []
            for endpoint in endpoints:
                record = records[endpoint // 2]
                enters = (record["direction"] > 0) == (endpoint % 2 == 1)
                (incoming if enters else outgoing).append(endpoint)
            if len(incoming) != len(outgoing):
                raise _SectionFailure("Unbalanced directed branches at a shared edge")
            axis = records[next(iter(endpoints)) // 2]["axis"]
            uv = ((1, 2), (2, 0), (0, 1))[axis]

            def angle(endpoint):
                record = records[endpoint // 2]
                p, tangent = gp_Pnt(), gp_Vec()
                record["curve"].D1(record["params"][endpoint % 2], p, tangent)
                t = _coordinates(tangent) * np.sign(record["direction"])
                return math.atan2(t[uv[1]], t[uv[0]])

            pairs = []
            unused = set(outgoing)
            for i in sorted(incoming):
                back = angle(i) + math.pi
                j = min(unused, key=lambda j: (back - angle(j)) % (2 * math.pi))
                unused.remove(j)
                pairs.append((i, j))
        else:
            # A trim crossing can disappear at the kernel's absolute precision.
            # Retry the same CAD/plane on the existing higher-scale measurement copy.
            raise _PrecisionFailure(
                f"Shared edge {edge_id}, crossing {rank}: {len(endpoints)} endpoints"
            )
        for i, j in pairs:
            candidates[i].add(j)
            candidates[j].add(i)
            shared_edges[min(i, j), max(i, j)] = edge_id
    cyclic = defaultdict(set)
    for i, record in enumerate(records):
        curve = record["curve"]
        if not record.get("support_closed", curve.IsClosed()):
            continue
        first, last = record.get("support_range", (curve.FirstParameter(), curve.LastParameter()))
        rounding = 64 * np.finfo(float).eps * max(1, abs(first), abs(last))
        for end in (0, 1):
            if record["edge_tags"][end]:
                continue
            parameter = record["params"][end]
            if min(abs(parameter - first), abs(parameter - last)) <= rounding:
                cyclic[record["face"], record["line"]].add(2 * i + end)
    for endpoints in cyclic.values():
        if len(endpoints) == 2:
            i, j = sorted(endpoints)
            candidates[i].add(j)
            candidates[j].add(i)
    for i, record in enumerate(records):
        if not record["edge_tags"][0] and (not record["edge_tags"][1]):
            curve = record["curve"]
            span = record["params"][1] - record["params"][0]
            support_lo, support_hi = record.get(
                "support_range", (curve.FirstParameter(), curve.LastParameter())
            )
            full_range = support_hi - support_lo
            p = _coordinates(curve.Value(curve.FirstParameter()))
            q = _coordinates(curve.Value(curve.LastParameter()))
            closes_at_rounding = np.linalg.norm(p - q) <= 64 * np.finfo(float).eps * max(
                np.linalg.norm(p), np.linalg.norm(q)
            )
            if (record.get("support_closed", curve.IsClosed()) or closes_at_rounding) and abs(
                span - full_range
            ) <= 64 * np.finfo(float).eps * max(1, abs(span)):
                candidates[2 * i].add(2 * i + 1)
                candidates[2 * i + 1].add(2 * i)
    partners = np.full(2 * len(records), -1, dtype=int)
    for i in range(len(partners)):
        if len(candidates[i]) != 1:
            raise _SectionFailure(f"Topological endpoint {i}: {len(candidates[i])} continuations")
        partners[i] = next(iter(candidates[i]))
    if not np.array_equal(partners[partners], np.arange(len(partners))):
        raise _SectionFailure("Non-reciprocal topological links")
    return (partners, shared_edges)


def _resolve_nodes(prepared, records, partners, shared_edges, axis, budget, solve_tol):
    uv = [i for i in range(3) if i != axis]
    vertices = {}
    updates = []
    for i, j in enumerate(partners):
        if j < i:
            continue
        ri, rj = (records[i // 2], records[j // 2])
        ti, tj = (ri["params"][i % 2], rj["params"][j % 2])
        initial_i = _coordinates(ri["curve"].Value(ti))
        initial_j = _coordinates(rj["curve"].Value(tj))
        gap = np.linalg.norm(initial_i - initial_j)
        if gap <= solve_tol:
            vertices[i] = vertices[j] = initial_i
            continue

        def fun(params):
            return (
                _coordinates(ri["curve"].Value(params[0]))
                - _coordinates(rj["curve"].Value(params[1]))
            )[uv]

        def jac(params):
            p, vi, vj = (gp_Pnt(), gp_Vec(), gp_Vec())
            ri["curve"].D1(params[0], p, vi)
            rj["curve"].D1(params[1], p, vj)
            return np.column_stack((_coordinates(vi)[uv], -_coordinates(vj)[uv]))

        solved = root(fun, (ti, tj), jac=jac, options={"xtol": 1e-11})
        residual = np.linalg.norm(fun(solved.x))
        if residual > solve_tol:
            raise _SectionFailure(f"Adjacent support traces do not meet: {residual:.6g}")
        point = _coordinates(ri["curve"].Value(solved.x[0]))
        edge_id = shared_edges[i, j]
        edge = topods.Edge(prepared.edges.FindKey(edge_id + 1))
        edge_curve, lo, hi = BRep_Tool.Curve(edge)
        projection = GeomAPI_ProjectPointOnCurve(gp_Pnt(*point), edge_curve, lo, hi)
        tolerance = max(
            BRep_Tool.Tolerance(edge),
            BRep_Tool.Tolerance(prepared.slabs.faces[ri["face"]]),
            BRep_Tool.Tolerance(prepared.slabs.faces[rj["face"]]),
            BRep_Tool.Tolerance(topexp.FirstVertex(edge)),
            BRep_Tool.Tolerance(topexp.LastVertex(edge)),
        )
        if not projection.NbPoints() or projection.LowerDistance() > tolerance + solve_tol:
            distance = projection.LowerDistance() if projection.NbPoints() else float("inf")
            raise _SectionFailure(
                f"Resolved node left shared edge {edge_id}'s tolerance tube "
                f"({distance:.6g} > {tolerance + solve_tol:.6g})"
            )
        ri["params"][i % 2], rj["params"][j % 2] = solved.x
        vertices[i] = vertices[j] = point
        updates.append({"edge": edge_id, "gap_before": float(gap), "residual": float(residual)})
    segments = []
    for i, record in enumerate(records):
        segment = _tessellate(record["curve"], *record["params"], budget)
        segment[0] = vertices[2 * i]
        segment[-1] = vertices[2 * i + 1]
        segments.append(segment)
    return (segments, updates)


def _walk_contours(segments, partners, records, axis):
    remaining = set(range(len(segments)))
    polygons = []
    while remaining:
        first = min(remaining)
        endpoint = 2 * first
        start = endpoint
        chain = []
        while True:
            segment_no, reverse = divmod(endpoint, 2)
            if segment_no not in remaining:
                raise _SectionFailure("Topology walk revisited a segment")
            remaining.remove(segment_no)
            segment = segments[segment_no][::-1] if reverse else segments[segment_no]
            if chain and (not np.array_equal(chain[-1], segment[0])):
                raise _SectionFailure("Shared-node coordinates disagree")
            chain.extend(segment if not chain else segment[1:])
            endpoint = int(partners[endpoint ^ 1])
            if endpoint == start:
                break
        if not np.array_equal(chain[0], chain[-1]):
            raise _SectionFailure("Topology cycle has no shared closing node")
        polygon = np.array(chain[:-1])[:, [i for i in range(3) if i != axis]]
        if records[first]["direction"] < 0:
            polygon = polygon[::-1].copy()
        if len(polygon) >= 3:
            polygons.append(polygon)
    return polygons


class _SurfaceSectionEngine:
    def __init__(self, occ, scale=1, conditioning=1):
        source = _FaceSlabIndex(occ)
        low = source.lo.min(axis=0)
        high = source.hi.max(axis=0)
        self.centre = (low + high) / 2
        self.input_scale = float(scale)
        self.conditioning = conditioning
        self.factor = fine_detail_scale(tuple(low), tuple(high)) * conditioning
        self.scale = self.input_scale * self.factor
        transform = gp_Trsf()
        f = self.factor
        transform.SetValues(
            f, 0, 0, -f * self.centre[0], 0, f, 0, -f * self.centre[1], 0, 0, f, -f * self.centre[2]
        )
        measured = BRepBuilderAPI_Transform(occ, transform, True).Shape()
        self.prepared = _PreparedGeometry(measured)
        bounds = [
            _conservative_face_bounds(face, surface, uv)
            for face, surface, uv in zip(
                self.prepared.slabs.faces, self.prepared.surfaces, self.prepared.bounds
            )
        ]
        self.support_bounds = None if any((b is None for b in bounds)) else np.array(bounds)

    def boundary_empty(self, axis, position):
        if self.support_bounds is None:
            return False
        low = np.min(self.support_bounds[:, 0, :], axis=0)
        high = np.max(self.support_bounds[:, 1, :], axis=0)
        rounding = 64 * np.finfo(float).eps * max(np.linalg.norm(high - low), abs(position))
        if position > high[axis] + rounding or position < low[axis] - rounding:
            return True
        on_boundary = min(abs(position - low[axis]), abs(position - high[axis])) <= rounding
        flat_on_plane = (
            np.max(np.abs(self.support_bounds[:, :, axis] - position), axis=1) <= rounding
        ).any()
        return bool(on_boundary and (not flat_on_plane))

    def section(self, axis, position, budget=1e-09):
        measured_pos = (position * self.input_scale - self.centre[axis]) * self.factor
        measured_budget = budget * self.scale
        diagonal = np.linalg.norm(
            self.prepared.slabs.hi.max(axis=0) - self.prepared.slabs.lo.min(axis=0)
        )
        # DD-287: numerical residuals consume at most 1/4096 of the
        # chord budget, subject to a relative double-rounding floor.
        tol = max(64 * np.finfo(float).eps * diagonal, measured_budget / 4096)
        if self.boundary_empty(axis, measured_pos):
            return ([], [{"support_extremum_empty": True}], [])
        if self.support_bounds is not None:
            spread = self.support_bounds[:, 1, axis] - self.support_bounds[:, 0, axis]
            offset = np.abs(self.support_bounds[:, 0, axis] - measured_pos)
            if ((spread <= tol) & (offset <= tol)).any():
                raise _CoplanarSection(
                    "Coplanar face requires the existing face-region section route"
                )
        _, diagnostics, records = _trimmed_segments(
            self.prepared.occ,
            axis,
            measured_pos,
            tol=tol,
            budget=measured_budget,
            prepared=self.prepared,
        )
        partners, edges = _endpoint_partners(records)
        segments, updates = _resolve_nodes(
            self.prepared, records, partners, edges, axis, measured_budget, tol
        )
        polygons = _walk_contours(segments, partners, records, axis)
        uv = [i for i in range(3) if i != axis]
        return (
            [(p / self.factor + self.centre[uv]) / self.input_scale for p in polygons],
            diagnostics,
            updates,
        )
