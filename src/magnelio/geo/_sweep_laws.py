"""Arc-length section laws for pipe construction (DD-275)."""

from __future__ import annotations

import math

import numpy as np


def _location_law(spine, frame, binormal, tangent):
    from OCC.Core.BRepFill import BRepFill_Edge3DLaw
    from OCC.Core.GeomFill import (
        GeomFill_ConstantBiNormal,
        GeomFill_CorrectedFrenet,
        GeomFill_CurveAndTrihedron,
        GeomFill_Fixed,
        GeomFill_Frenet,
    )
    from OCC.Core.gp import gp_Dir, gp_Vec

    if frame == "fixed":
        reference = gp_Vec(1, 0, 0) if abs(tangent.X()) < 0.9 else gp_Vec(0, 1, 0)
        reference.Subtract(gp_Vec(tangent).Multiplied(reference.Dot(gp_Vec(tangent))))
        reference.Normalize()
        trihedron = GeomFill_Fixed(gp_Vec(tangent), reference)
    elif frame == "fixed_binormal":
        trihedron = GeomFill_ConstantBiNormal(gp_Dir(*binormal))
    else:
        trihedron = GeomFill_Frenet() if frame == "frenet" else GeomFill_CorrectedFrenet()
    law = BRepFill_Edge3DLaw(spine, GeomFill_CurveAndTrihedron(trihedron))
    law.TransformInCompatibleLaw(1e-8)
    return law


def _arc_length(curve, parameter, tolerance):
    from OCC.Core.GeomAbs import GeomAbs_C1
    from OCC.Core.gp import gp_Pnt, gp_Vec
    from OCC.Core.TColStd import TColStd_Array1OfReal
    from scipy.integrate import quad

    intervals = TColStd_Array1OfReal(1, curve.NbIntervals(GeomAbs_C1) + 1)
    curve.Intervals(intervals, GeomAbs_C1)
    knots = [
        intervals.Value(i)
        for i in range(2, intervals.Upper())
        if curve.FirstParameter() < intervals.Value(i) < parameter
    ]
    point, derivative = gp_Pnt(), gp_Vec()

    def speed(value):
        curve.D1(value, point, derivative)
        return derivative.Magnitude()

    return quad(
        speed,
        curve.FirstParameter(),
        parameter,
        points=knots,
        epsabs=tolerance,
        epsrel=1e-12,
        limit=200,
    )[0]


def _pose(law, distance, curves, cumulative, tolerance):
    from OCC.Core.gp import gp_Mat, gp_Vec
    from scipy.optimize import brentq

    index = min(int(np.searchsorted(cumulative, distance, side="right")), len(curves))
    curve = curves[index - 1]
    offset = distance - cumulative[index - 1]
    if offset <= 0:
        parameter = curve.FirstParameter()
    elif distance >= cumulative[-1]:
        parameter = curve.LastParameter()
    else:
        parameter = brentq(
            lambda value: _arc_length(curve, value, tolerance) - offset,
            curve.FirstParameter(),
            curve.LastParameter(),
            xtol=1e-14,
            rtol=1e-14,
        )
    matrix, point = gp_Mat(), gp_Vec()
    if not law.law(index).D0(parameter, matrix, point):
        raise RuntimeError("swept(): the section frame could not be evaluated.")
    return (
        np.array([[matrix.Value(i, j) for j in range(1, 4)] for i in range(1, 4)]),
        np.array(point.XYZ().Coord()),
    )


def _boundary_face(wire, normal):
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
    from OCC.Core.Geom import Geom_Curve
    from OCC.Core.GeomAbs import GeomAbs_Ellipse
    from OCC.Core.TopAbs import TopAbs_REVERSED
    from OCC.Core.TopoDS import topods

    from magnelio.geo._occ_backend import _wire_edges, extract_face_wire, make_wire_face
    from magnelio.geo.topology import _normal

    # Rebuild from 3-D curves: source wires can carry pcurves belonging to the
    # holed source face. Neither those pcurves nor shared vertices are offset.
    maker = BRepBuilderAPI_MakeWire()
    for edge in _wire_edges(wire):
        curve = BRepAdaptor_Curve(edge)
        geometry = Geom_Curve.DownCast(curve.Curve().Curve().Transformed(curve.Trsf()))
        parameters = [curve.FirstParameter(), curve.LastParameter()]
        if curve.GetType() == GeomAbs_Ellipse:
            # The offset builder splits elliptic curves at quadrant boundaries.
            # Give the initial section those same exact analytic edges so that
            # representational splitting is not mistaken for boundary collapse.
            step = math.pi / 2
            parameters = (
                [parameters[0]]
                + [
                    k * step
                    for k in range(
                        math.floor(parameters[0] / step) + 1, math.ceil(parameters[1] / step)
                    )
                ]
                + [parameters[1]]
            )
        spans = list(zip(parameters[:-1], parameters[1:]))
        if edge.Orientation() == TopAbs_REVERSED:
            spans.reverse()
        for start, end in spans:
            fresh = BRepBuilderAPI_MakeEdge(geometry, start, end).Edge()
            if edge.Orientation() == TopAbs_REVERSED:
                fresh.Reverse()
            maker.Add(fresh)
    if not maker.IsDone():
        raise RuntimeError("swept(): the draft boundary could not be copied.")
    face = make_wire_face(maker.Wire())
    if np.dot(_normal(face), normal) < 0:
        face = make_wire_face(topods.Wire(maker.Wire().Reversed()))
    canonical = BRepBuilderAPI_MakeWire()
    for edge in _wire_edges(extract_face_wire(face)):
        canonical.Add(edge)
    return make_wire_face(canonical.Wire())


def _offset_boundary(face, distance):
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeOffset
    from OCC.Core.GeomAbs import GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_Intersection
    from OCC.Core.ShapeFix import ShapeFix_Edge
    from OCC.Core.TopoDS import topods

    from magnelio.geo._occ_backend import (
        _shape_wires,
        _wire_edges,
        extract_face_wire,
    )
    from magnelio.geo._occ_backend import bounding_box as occ_bbox

    if distance == 0:
        return extract_face_wire(face)
    bounds = occ_bbox(face)
    diagonal = math.dist(*bounds)
    center = tuple((a + b) / 2 for a, b in zip(*bounds))
    scale = 2.0 ** round(math.log2(128 / diagonal))
    # Normalize each boundary, including tiny offset holes. Negative circular
    # offsets below unit size crash the installed kernel without normalization.
    scaled = topods.Face(_transform(face, np.eye(3) * scale, -scale * np.array(center)))
    edges = _wire_edges(extract_face_wire(scaled))
    if len(edges) == 1:
        curve = BRepAdaptor_Curve(edges[0])
        if curve.GetType() == GeomAbs_Circle and curve.Circle().Radius() + distance * scale <= 1e-7:
            raise ValueError("swept(draft_deg=...): a section boundary closes along the path.")
    if all(BRepAdaptor_Curve(edge).GetType() == GeomAbs_Ellipse for edge in edges):
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
        from OCC.Core.Geom import Geom_OffsetCurve, Geom_TrimmedCurve
        from OCC.Core.GeomAbs import GeomAbs_C1
        from OCC.Core.GeomConvert import GeomConvert_ApproxCurve
        from OCC.Core.gp import gp_Dir
        from OCC.Core.TopAbs import TopAbs_REVERSED

        from magnelio.geo.topology import _normal

        builder = BRepBuilderAPI_MakeWire()
        for edge in edges:
            curve = BRepAdaptor_Curve(edge)
            ellipse = curve.Ellipse()
            offset = distance * scale
            if offset <= -(ellipse.MinorRadius() ** 2) / ellipse.MajorRadius():
                raise ValueError("swept(draft_deg=...): a section offset develops a cusp.")
            sign = -1 if edge.Orientation() == TopAbs_REVERSED else 1
            geometry = curve.Curve().Curve()
            offset_curve = Geom_OffsetCurve(geometry, sign * offset, gp_Dir(*_normal(scaled)))
            trimmed = Geom_TrimmedCurve(offset_curve, curve.FirstParameter(), curve.LastParameter())
            approximation = GeomConvert_ApproxCurve(trimmed, 1e-7, GeomAbs_C1, 16, 14)
            if not approximation.HasResult():
                raise RuntimeError("swept(): an elliptic offset could not be represented.")
            fresh = BRepBuilderAPI_MakeEdge(approximation.Curve()).Edge()
            if sign < 0:
                fresh.Reverse()
            builder.Add(fresh)
        wires = [builder.Wire()]
    else:
        maker = BRepOffsetAPI_MakeOffset(scaled, GeomAbs_Intersection)
        maker.Perform(distance * scale)
        if not maker.IsDone():
            raise ValueError("swept(draft_deg=...): the section offset could not be constructed.")
        wires = _shape_wires(maker.Shape())
    if len(wires) != 1 or len(_wire_edges(wires[0])) != len(edges):
        raise ValueError("swept(draft_deg=...): draft changes the section boundary topology.")
    from magnelio.geo.topology import _normal

    # Offset curves carry pcurves of the temporary source face. Rebuild their
    # independent 3-D edges before covering them with the section plane.
    boundary = _boundary_face(wires[0], _normal(scaled))
    result = extract_face_wire(boundary)
    # Record the pcurve/3-D parameter agreement on these newly constructed
    # edges. This updates edge tolerances, without changing the 3-D outline.
    parameterization = ShapeFix_Edge()
    for edge in _wire_edges(result):
        parameterization.FixSameParameter(edge, boundary, 1e-7)
    result = topods.Wire(_transform(result, np.eye(3) / scale, np.array(center)))
    boundary = _boundary_face(result, _normal(face))
    result = extract_face_wire(boundary)
    for edge in _wire_edges(result):
        parameterization.FixSameParameter(edge, boundary, 1e-7)
    return result


def _section_boundaries(face):
    from magnelio.geo._occ_backend import _shape_wires, extract_face_wire

    outer = extract_face_wire(face)
    return [outer] + [wire for wire in _shape_wires(face) if not wire.IsSame(outer)]


def drafted_boundaries(face, distance):
    from magnelio.geo._occ_backend import make_profile_face
    from magnelio.geo.topology import _normal

    normal = _normal(face)
    wires = [
        _offset_boundary(_boundary_face(wire, normal), distance if index == 0 else -distance)
        for index, wire in enumerate(_section_boundaries(face))
    ]
    if distance:
        # This is the existing Profile construction contract, not the cancelled
        # sweep self-intersection diagnostic. Holes may not escape or collapse.
        make_profile_face(wires[0], wires[1:])
    return wires


def draft_radius(profile, distance):
    from magnelio.geo._occ_backend import bounding_box as occ_bbox
    from magnelio.geo._occ_backend import make_face_with_holes

    wires = drafted_boundaries(profile, distance)
    return math.dist(*occ_bbox(make_face_with_holes(wires[0], wires[1:])))


def _transform(wire, rotation, translation):
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.gp import gp_Trsf

    transform = gp_Trsf()
    transform.SetValues(*[v for row, shift in zip(rotation, translation) for v in (*row, shift)])
    return BRepBuilderAPI_Transform(wire, transform, True).Shape()


def _wire_points(wire):
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve

    from magnelio.geo._occ_backend import _wire_edges

    for edge in _wire_edges(wire):
        curve = BRepAdaptor_Curve(edge)
        for fraction in np.linspace(0, 1, 9):
            yield curve.Value(
                curve.FirstParameter() + fraction * (curve.LastParameter() - curve.FirstParameter())
            )


def _section_error(expected, actual, tolerance):
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape

    for source, target in ((expected, actual), (actual, expected)):
        for point in _wire_points(source):
            check = BRepExtrema_DistShapeShape(BRepBuilderAPI_MakeVertex(point).Vertex(), target)
            if not check.IsDone() or check.Value() > tolerance:
                return True
    return False


def make_law_sweep(profile, spine, frame, binormal, tangent, twist_deg, draft_deg, tolerance):
    from OCC.Core.Approx import Approx_IsoParametric
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_Copy,
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeWire,
    )
    from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
    from OCC.Core.BRepTools import breptools
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods

    from magnelio.geo._occ_backend import _history_result, boolean_difference, boolean_union

    law = _location_law(spine, frame, binormal, tangent)
    curves = [law.law(i).GetCurve() for i in range(1, law.NbLaw() + 1)]
    arc_tolerance = min(tolerance / 32, 1e-8)
    cumulative = np.concatenate(
        (
            [0.0],
            np.cumsum(
                [_arc_length(curve, curve.LastParameter(), arc_tolerance) for curve in curves]
            ),
        )
    )
    length = cumulative[-1]

    def pose(distance):
        return _pose(law, distance, curves, cumulative, arc_tolerance)

    initial, anchor = pose(0)
    normal = np.array(tangent.XYZ().Coord())
    skew = np.array(
        [[0, -normal[2], normal[1]], [normal[2], 0, -normal[0]], [-normal[1], normal[0], 0]]
    )
    faces = [_boundary_face(wire, normal) for wire in _section_boundaries(profile)]
    cache = {}

    def section(fraction):
        if fraction in cache:
            return cache[fraction]
        s = fraction * length
        matrix, point = pose(s)
        transport = matrix @ initial.T
        angle = math.radians(twist_deg) * fraction
        spin = np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * skew @ skew
        rotation = transport @ spin
        translation = point - rotation @ anchor
        distance = s * math.tan(math.radians(draft_deg))
        wires = [
            _offset_boundary(face, distance if i == 0 else -distance)
            for i, face in enumerate(faces)
        ]
        if distance:
            from magnelio.geo._occ_backend import make_profile_face

            make_profile_face(wires[0], wires[1:])
        placed = [_transform(wire, rotation, translation) for wire in wires]
        cache[fraction] = placed
        return cache[fraction]

    closed = np.linalg.norm(pose(length)[1] - anchor) <= length * 1e-10
    if closed and (
        draft_deg
        or any(_section_error(start, end, tolerance) for start, end in zip(section(0), section(1)))
    ):
        raise ValueError("swept(): a closed path requires matching start and end sections.")

    def configure(boundary, sections):
        loft = BRepOffsetAPI_ThruSections(True, False, max(1e-7, min(tolerance / 64, 1e-6)))
        # Angular correspondence is part of the law, not a kernel heuristic.
        # Equal section parameters represent equal arc-length stations.
        loft.CheckCompatibility(False)
        loft.SetParType(Approx_IsoParametric)
        for wires in sections:
            loft.AddWire(topods.Wire(BRepBuilderAPI_Copy(wires[boundary], True, False).Shape()))
        loft.Build()
        if not loft.IsDone():
            raise RuntimeError("swept(): twist/draft section fitting failed.")
        return loft

    def interpolated_section(loft, fraction):
        wire = BRepBuilderAPI_MakeWire()
        explorer = TopExp_Explorer(loft.Shape(), TopAbs_FACE)
        while explorer.More():
            face = topods.Face(explorer.Current())
            explorer.Next()
            if face.IsSame(loft.FirstShape()) or face.IsSame(loft.LastShape()):
                continue
            u0, u1, v0, v1 = breptools.UVBounds(face)
            surface = BRep_Tool.Surface(face)
            curve = surface.VIso(v0 + fraction * (v1 - v0))
            wire.Add(BRepBuilderAPI_MakeEdge(curve, u0, u1).Edge())
        if not wire.IsDone():
            raise RuntimeError("swept(): the fitted section could not be inspected.")
        return wire.Wire()

    if tolerance < 1e-7:
        raise RuntimeError("swept(): the requested fitting tolerance is below CAD resolution.")
    # Native edge laws must meet continuously. Sharp corners need an explicit
    # joint construction, rather than an inferred smoothing rule.
    from OCC.Core.gp import gp_Mat, gp_Vec

    for index in range(1, len(curves)):
        left, right = gp_Mat(), gp_Mat()
        law.law(index).D0(curves[index - 1].LastParameter(), left, gp_Vec())
        law.law(index + 1).D0(curves[index].FirstParameter(), right, gp_Vec())
        if (
            max(abs(left.Value(i, j) - right.Value(i, j)) for i in range(1, 4) for j in range(1, 4))
            > 1e-6
        ):
            raise ValueError("swept(): twist/draft requires tangent-connected path edges.")

    # Fit edge spans separately: a line/arc joint is tangent-continuous but
    # its curvature jumps. A global C2 fit would smooth that physical joint.
    accepted = [[] for _ in faces]
    for start, end in zip(cumulative[:-1] / length, cumulative[1:] / length):
        count = max(9, 1 + math.ceil(abs(twist_deg) * (end - start) / 15))
        for _attempt in range(10):
            if count > 4097:
                raise RuntimeError("swept(): twist/draft fitting requires too many sections.")
            sections = [
                section(float(start + fraction * (end - start)))
                for fraction in np.linspace(0, 1, count)
            ]
            lofts = [configure(boundary, sections) for boundary in range(len(faces))]
            refine = False
            # This is a sampled fit, not a global Hausdorff bound or a
            # self-intersection diagnostic.
            for fraction in (i / (4 * (count - 1)) for i in range(1, 4 * (count - 1)) if i % 4):
                expected = section(float(start + fraction * (end - start)))
                for boundary, loft in enumerate(lofts):
                    actual = interpolated_section(loft, fraction)
                    if _section_error(expected[boundary], actual, tolerance / 2):
                        refine = True
                        break
                if refine:
                    break
            if not refine:
                break
            count = 2 * count - 1
        else:
            raise RuntimeError(
                "swept(): twist/draft fitting did not reach the requested tolerance."
            )
        for group, loft in zip(accepted, lofts):
            group.append(loft)
    solids = []
    for group in accepted:
        if len(group) > 1:
            solids.append(boolean_union([_history_result(loft) for loft in group]))
            continue
        loft = group[0]
        if not closed:
            solids.append(_history_result(loft))
            continue
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing
        from OCC.Core.BRepLib import breplib
        from OCC.Core.TopAbs import TopAbs_SHELL

        sewing = BRepBuilderAPI_Sewing(tolerance)
        explorer = TopExp_Explorer(loft.Shape(), TopAbs_FACE)
        while explorer.More():
            face = explorer.Current()
            explorer.Next()
            if not face.IsSame(loft.FirstShape()) and not face.IsSame(loft.LastShape()):
                sewing.Add(face)
        sewing.Perform()
        shell = sewing.SewedShape()
        if shell.ShapeType() != TopAbs_SHELL or not shell.Closed():
            raise RuntimeError("swept(): the periodic section seam could not be closed.")
        solid = BRepBuilderAPI_MakeSolid(topods.Shell(shell)).Solid()
        breplib.OrientClosedSolid(solid)
        solids.append(solid)
    solid = solids[0]
    for hole in solids[1:]:
        solid = boolean_difference(solid, hole)
    return solid
