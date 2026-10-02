"""Projection of independent curves onto bounded sheets and faces."""

from __future__ import annotations

import math

from magnelio.geo._axes import normalize_axis
from magnelio.geo._scaling import box_diagonal, choose_scale, union_boxes
from magnelio.geo._validate import point3, positive

_PARTIAL = "projected_onto(): only part of the curve reaches the bounded target; use clip=True."


def _target_sheet(target):
    from magnelio.geo._sheet import Sheet
    from magnelio.geo.topology import FaceRef

    if isinstance(target, FaceRef):
        return target.detached()
    if isinstance(target, Sheet):
        return target
    raise TypeError(
        f"projected_onto() target must be a Sheet or FaceRef; got {type(target).__name__}."
    )


def _point_box(point):
    return point, point


def _projection_wires(source_shape, target_shape, scale, *, direction, perspective_source):
    from OCC.Core.BRepProj import BRepProj_Projection
    from OCC.Core.gp import gp_Dir, gp_Pnt

    if perspective_source is not None:
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
        from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape

        eye = gp_Pnt(*(value * scale for value in perspective_source))
        distance = BRepExtrema_DistShapeShape(BRepBuilderAPI_MakeVertex(eye).Vertex(), source_shape)
        if distance.IsDone() and distance.Value() <= 1e-8:
            raise ValueError("projected_onto(): perspective source lies on the curve.")
    argument = (
        gp_Dir(*direction)
        if direction is not None
        else gp_Pnt(*(value * scale for value in perspective_source))
    )
    operation = BRepProj_Projection(source_shape, target_shape, argument)
    wires = []
    while operation.More():
        wires.append(operation.Current())
        operation.Next()
    if not operation.IsDone() and wires:
        raise RuntimeError("projected_onto(): ray projection failed in the CAD kernel.")
    return wires


def _split_source_at_face(source_shape, target_shape):
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section, BRepAlgoAPI_Splitter
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire
    from OCC.Core.BRepTools import BRepTools_WireExplorer
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
    from OCC.Core.TopExp import TopExp_Explorer, topexp
    from OCC.Core.TopoDS import topods

    from magnelio.geo._occ_backend import _shape_list, keep_operands_intact

    section = BRepAlgoAPI_Section(source_shape, target_shape)
    keep_operands_intact(section)
    section.Build()
    if not section.IsDone():
        raise RuntimeError("projected_onto(): source-target intersection failed.")
    edges = TopExp_Explorer(section.Shape(), TopAbs_EDGE)
    if edges.More():
        return [source_shape]
    vertices = TopExp_Explorer(section.Shape(), TopAbs_VERTEX)
    contacts = []
    while vertices.More():
        contacts.append(BRep_Tool.Pnt(topods.Vertex(vertices.Current())))
        vertices.Next()
    if not contacts:
        return [source_shape]
    splitter = BRepAlgoAPI_Splitter()
    splitter.SetArguments(_shape_list((source_shape,)))
    splitter.SetTools(_shape_list((target_shape,)))
    keep_operands_intact(splitter)
    splitter.Build()
    if not splitter.IsDone():
        raise RuntimeError("projected_onto(): source curve could not be split at the target.")
    pieces = []
    run = BRepBuilderAPI_MakeWire()
    explorer = BRepTools_WireExplorer(topods.Wire(splitter.Shape()))
    while explorer.More():
        edge = topods.Edge(explorer.Current())
        run.Add(edge)
        if not run.IsDone():
            raise RuntimeError("projected_onto(): a split source curve became disconnected.")
        endpoint = BRep_Tool.Pnt(topexp.LastVertex(edge, True))
        if any(endpoint.Distance(contact) <= 1e-7 for contact in contacts):
            pieces.append(run.Wire())
            run = BRepBuilderAPI_MakeWire()
        explorer.Next()
    if run.IsDone():
        pieces.append(run.Wire())
    return pieces


def _coincident_wires(source_shape, target_shape, *, length_tolerance=1e-7):
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.TopAbs import TopAbs_EDGE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods
    from OCC.Core.TopTools import TopTools_HSequenceOfShape

    from magnelio.geo._occ_backend import keep_operands_intact
    from magnelio.geo._occ_compat import connect_edges_to_wires

    common = BRepAlgoAPI_Common(source_shape, target_shape)
    keep_operands_intact(common)
    common.Build()
    if not common.IsDone():
        raise RuntimeError("projected_onto(): coincidence with the target could not be checked.")
    edges = TopTools_HSequenceOfShape()
    explorer = TopExp_Explorer(common.Shape(), TopAbs_EDGE)
    while explorer.More():
        edges.Append(topods.Edge(explorer.Current()))
        explorer.Next()
    if edges.Length() == 0:
        return [], False
    wires = connect_edges_to_wires(edges, 1e-7)
    source_properties, common_properties = GProp_GProps(), GProp_GProps()
    brepgprop.LinearProperties(source_shape, source_properties)
    brepgprop.LinearProperties(common.Shape(), common_properties)
    full = abs(source_properties.Mass() - common_properties.Mass()) <= length_tolerance
    return [topods.Wire(wires.Value(i)) for i in range(1, wires.Length() + 1)], full


def _trim_seed_parameters(common_wires, source_shape):
    from OCC.Core.BRepAdaptor import BRepAdaptor_CompCurve
    from scipy.optimize import minimize_scalar

    source = BRepAdaptor_CompCurve(source_shape)
    first, last = source.FirstParameter(), source.LastParameter()
    grid = [first + (last - first) * index / 128 for index in range(129)]
    source_points = [source.Value(parameter) for parameter in grid]
    seeds = []
    for wire in common_wires:
        curve = BRepAdaptor_CompCurve(wire)
        for end in (curve.FirstParameter(), curve.LastParameter()):
            point = curve.Value(end)
            if min(point.Distance(source_points[0]), point.Distance(source_points[-1])) <= 1e-7:
                continue
            ranked = sorted(
                range(len(grid)), key=lambda index: point.Distance(source_points[index])
            )
            for index in ranked[:3]:
                left, right = grid[max(0, index - 1)], grid[min(128, index + 1)]
                if left == right:
                    continue
                optimum = minimize_scalar(
                    lambda parameter: source.Value(parameter).SquareDistance(point),
                    bounds=(left, right),
                    method="bounded",
                    options={"xatol": 1e-12 * (last - first)},
                )
                if optimum.success:
                    seeds.append(optimum.x)
    seeds = sorted(set(seeds))
    seeds.extend((a + b) / 2 for a, b in zip(seeds, seeds[1:]))
    return tuple(sorted(set(seeds)))


def _closest_wires(source_shape, target_shape, tolerance, extra_parameters=()):
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepAdaptor import BRepAdaptor_CompCurve
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeVertex,
        BRepBuilderAPI_MakeWire,
    )
    from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape, BRepExtrema_IsOnEdge
    from OCC.Core.BRepLib import breplib
    from OCC.Core.GCE2d import GCE2d_MakeSegment
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve
    from OCC.Core.GeomLProp import GeomLProp_SLProps
    from OCC.Core.gp import gp_Pnt2d
    from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
    from OCC.Core.TopAbs import TopAbs_EDGE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods

    surface = BRep_Tool.Surface(target_shape)
    analyzer = ShapeAnalysis_Surface(surface)
    source = BRepAdaptor_CompCurve(source_shape)
    start, stop = source.FirstParameter(), source.LastParameter()
    boundaries = []
    explorer = TopExp_Explorer(target_shape, TopAbs_EDGE)
    while explorer.More():
        edge = topods.Edge(explorer.Current())
        curve, first, last = BRep_Tool.Curve(edge)
        if curve is not None:
            boundaries.append((edge, curve, first, last))
        explorer.Next()

    def nearest(parameter):
        point = source.Value(parameter)
        extrema = BRepExtrema_DistShapeShape(
            BRepBuilderAPI_MakeVertex(point).Vertex(), target_shape
        )
        if not extrema.IsDone() or extrema.NbSolution() == 0:
            raise RuntimeError("projected_onto(): nearest point could not be found on the target.")
        projected = extrema.PointOnShape2(1)
        distance = point.Distance(projected)
        edge = None
        edge_parameter = None
        for boundary_edge, curve, first, last in boundaries:
            candidate = GeomAPI_ProjectPointOnCurve(point, curve, first, last)
            if candidate.NbPoints() == 0:
                continue
            better = candidate.LowerDistance() < distance - 1e-12
            same_point = (
                candidate.LowerDistance() <= distance + 1e-12
                and candidate.NearestPoint().Distance(projected) <= 1e-8
            )
            if better or (same_point and edge is None):
                projected = candidate.NearestPoint()
                distance = candidate.LowerDistance()
                edge = boundary_edge
                edge_parameter = candidate.LowerDistanceParameter()
        uv = analyzer.ValueOfUV(projected, 1e-8)
        properties = GeomLProp_SLProps(surface, uv.X(), uv.Y(), 1, 1e-10)
        if min(properties.D1U().Magnitude(), properties.D1V().Magnitude()) <= 1e-10:
            raise ValueError("projected_onto(): the target is singular at a closest point.")
        if edge is None and extrema.SupportTypeShape2(1) == BRepExtrema_IsOnEdge:
            edge = topods.Edge(extrema.SupportOnShape2(1))
            edge_parameter = extrema.ParOnEdgeS2(1)
        return uv.X(), uv.Y(), projected, edge, edge_parameter

    def unwrap(left, right):
        u, v, point, edge, edge_parameter = right
        if surface.IsUPeriodic():
            period = surface.UPeriod()
            u += round((left[0] - u) / period) * period
        if surface.IsVPeriodic():
            period = surface.VPeriod()
            v += round((left[1] - v) / period) * period
        return u, v, point, edge, edge_parameter

    def refined(a, left, b, right, depth):
        right = unwrap(left, right)
        middle = nearest((a + b) / 2)
        middle = unwrap(left, middle)
        if (
            left[3] is not None
            and right[3] is not None
            and middle[3] is not None
            and left[3].IsSame(right[3])
            and left[3].IsSame(middle[3])
        ):
            boundary_curve, _, _ = BRep_Tool.Curve(left[3])
            first, last, mid = left[4], right[4], middle[4]
            if boundary_curve.IsPeriodic():
                period = boundary_curve.Period()
                last += round((first - last) / period) * period
                mid += round((first - mid) / period) * period
            if min(first, last) - 1e-10 <= mid <= max(first, last) + 1e-10:
                return [(b, right)]
        uv_middle = gp_Pnt2d((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
        on_surface = surface.Value(uv_middle.X(), uv_middle.Y())
        error = on_surface.Distance(middle[2])
        if error <= tolerance:
            return [(b, right)]
        if depth == 0:
            raise ValueError(
                "projected_onto(): closest points are discontinuous or singular on the target."
            )
        return refined(a, left, (a + b) / 2, middle, depth - 1) + refined(
            (a + b) / 2, middle, b, right, depth - 1
        )

    parameters = sorted(
        {start + (stop - start) * index / 128 for index in range(129)} | set(extra_parameters)
    )
    samples = [(parameters[0], nearest(parameters[0]))]
    for parameter in parameters[1:]:
        previous, left = samples[-1]
        right = nearest(parameter)
        samples.extend(refined(previous, left, parameter, right, 12))
    wire_builder = BRepBuilderAPI_MakeWire()
    previous = samples[0][1]
    for _, current in samples[1:]:
        if previous[2].Distance(current[2]) <= 1e-8:
            previous = current
            continue
        if previous[3] is not None and current[3] is not None and previous[3].IsSame(current[3]):
            boundary_curve, _, _ = BRep_Tool.Curve(previous[3])
            a, b = previous[4], current[4]
            if boundary_curve.IsPeriodic():
                b += round((a - b) / boundary_curve.Period()) * boundary_curve.Period()
            if b < a:
                edge = BRepBuilderAPI_MakeEdge(boundary_curve, b, a).Edge()
                edge.Reverse()
            else:
                edge = BRepBuilderAPI_MakeEdge(boundary_curve, a, b).Edge()
        else:
            segment = GCE2d_MakeSegment(
                gp_Pnt2d(previous[0], previous[1]), gp_Pnt2d(current[0], current[1])
            ).Value()
            edge = BRepBuilderAPI_MakeEdge(segment, surface).Edge()
            if not breplib.BuildCurves3d(edge):
                raise RuntimeError(
                    "projected_onto(): nearest trace could not be built on the target."
                )
        wire_builder.Add(edge)
        if not wire_builder.IsDone():
            raise RuntimeError("projected_onto(): nearest trace is disconnected on the target.")
        previous = current
    return [wire_builder.Wire()] if wire_builder.IsDone() else []


def _sample_points(wire):
    from OCC.Core.BRepAdaptor import BRepAdaptor_CompCurve

    curve = BRepAdaptor_CompCurve(wire)
    start, stop = curve.FirstParameter(), curve.LastParameter()
    count = 129
    return [curve.Value(start + (stop - start) * i / (count - 1)) for i in range(count)]


def _matches(point, wires, tolerance):
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape

    vertex = BRepBuilderAPI_MakeVertex(point).Vertex()
    matches = []
    for index, wire in enumerate(wires):
        distance = BRepExtrema_DistShapeShape(vertex, wire)
        if distance.IsDone() and distance.Value() <= tolerance:
            matches.append(index)
    return matches


def _ray_hits(intersector, source_point, *, direction, perspective_source, scale):
    from OCC.Core.gp import gp_Dir, gp_Lin, gp_Pnt

    if direction is None:
        origin = gp_Pnt(*(v * scale for v in perspective_source))
        ray = tuple(source_point.Coord()[i] - origin.Coord()[i] for i in range(3))
        length = math.sqrt(sum(v * v for v in ray))
        if length <= 1e-12:
            raise ValueError("projected_onto(): perspective source lies on the curve.")
        unit = tuple(v / length for v in ray)
    else:
        origin = source_point
        unit = direction
    intersector.Perform(gp_Lin(origin, gp_Dir(*unit)), 0.0, 1e100)
    if not intersector.IsDone():
        raise RuntimeError("projected_onto(): ray intersection failed in the CAD kernel.")
    lower = length if direction is None else 0.0
    hits = []
    for i in range(1, intersector.NbPnt() + 1):
        parameter = intersector.WParameter(i)
        if parameter >= lower - 1e-8:
            hits.append((parameter, intersector.Pnt(i)))
    return sorted(hits, key=lambda pair: pair[0])


def _select_ray_wires(
    wires, source_shape, target_shape, scale, *, direction, perspective_source, clip, all_hits
):
    from OCC.Core.IntCurvesFace import IntCurvesFace_ShapeIntersector

    intersector = IntCurvesFace_ShapeIntersector()
    intersector.Load(target_shape, 1e-8)
    selected = set()
    missed = False
    matched = False
    for point in _sample_points(source_shape):
        hits = _ray_hits(
            intersector,
            point,
            direction=direction,
            perspective_source=perspective_source,
            scale=scale,
        )
        if not hits:
            missed = True
            continue
        matched = True
        for _, hit in hits if all_hits else hits[:1]:
            selected.update(_matches(hit, wires, 1e-7))
    for index, wire in enumerate(wires):
        status = _wire_forward_status(
            wire,
            source_shape,
            intersector,
            direction=direction,
            perspective_source=perspective_source,
            scale=scale,
            all_hits=all_hits,
        )
        if status is True:
            selected.add(index)
            matched = True
        elif status is False:
            selected.discard(index)
    if not matched:
        return []
    if missed and not clip:
        raise ValueError(_PARTIAL)
    if not selected:
        return []
    chosen = [wire for index, wire in enumerate(wires) if index in selected]
    if not clip and _has_clipped_endpoint(
        chosen,
        source_shape,
        target_shape,
        intersector,
        direction=direction,
        perspective_source=perspective_source,
        scale=scale,
    ):
        raise ValueError(_PARTIAL)
    return chosen


def _wire_forward_status(
    wire, source_shape, intersector, *, direction, perspective_source, scale, all_hits
):
    from OCC.Core.BRepAdaptor import BRepAdaptor_CompCurve
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
    from OCC.Core.gp import gp_Pnt

    from magnelio.geo._occ_backend import bounding_box

    adaptor = BRepAdaptor_CompCurve(wire)
    source_box = bounding_box(source_shape)
    statuses = []
    for fraction in (0.25, 0.5, 0.75):
        point = adaptor.Value(
            adaptor.FirstParameter()
            + fraction * (adaptor.LastParameter() - adaptor.FirstParameter())
        )
        if direction is None:
            eye = gp_Pnt(*(value * scale for value in perspective_source))
            ray = tuple(point.Coord()[i] - eye.Coord()[i] for i in range(3))
            length = math.sqrt(sum(value * value for value in ray))
            if length <= 1e-8:
                continue
            unit = tuple(value / length for value in ray)
        else:
            unit = direction
        span = (
            max(
                1.0,
                math.dist(source_box[0], source_box[1]),
                math.dist(source_box[0], point.Coord()),
                math.dist(source_box[1], point.Coord()),
            )
            * 4
        )
        start = gp_Pnt(*(point.Coord()[i] - span * unit[i] for i in range(3)))
        end = gp_Pnt(*(point.Coord()[i] + span * unit[i] for i in range(3)))
        line = BRepBuilderAPI_MakeEdge(start, end).Edge()
        extrema = BRepExtrema_DistShapeShape(line, source_shape)
        if not extrema.IsDone() or extrema.Value() > 1e-7:
            continue
        status = False
        for index in range(1, extrema.NbSolution() + 1):
            source_point = extrema.PointOnShape2(index)
            hits = _ray_hits(
                intersector,
                source_point,
                direction=direction,
                perspective_source=perspective_source,
                scale=scale,
            )
            if any(point.Distance(hit) <= 1e-7 for _, hit in (hits if all_hits else hits[:1])):
                status = True
                break
        statuses.append(status)
    if len(set(statuses)) > 1:
        raise ValueError("projected_onto(): forward branch changes within one projected curve.")
    return statuses[0] if statuses else None


def _has_clipped_endpoint(
    wires, source_shape, target_shape, intersector, *, direction, perspective_source, scale
):
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepAdaptor import BRepAdaptor_CompCurve
    from OCC.Core.TopAbs import TopAbs_EDGE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods

    boundary = []
    explorer = TopExp_Explorer(target_shape, TopAbs_EDGE)
    while explorer.More():
        edge = topods.Edge(explorer.Current())
        if not BRep_Tool.IsClosed(edge, target_shape):
            boundary.append(edge)
        explorer.Next()
    source = BRepAdaptor_CompCurve(source_shape)
    expected = []
    for parameter in (source.FirstParameter(), source.LastParameter()):
        expected.extend(
            point
            for _, point in _ray_hits(
                intersector,
                source.Value(parameter),
                direction=direction,
                perspective_source=perspective_source,
                scale=scale,
            )
        )
    for wire in wires:
        curve = BRepAdaptor_CompCurve(wire)
        for parameter in (curve.FirstParameter(), curve.LastParameter()):
            point = curve.Value(parameter)
            if any(point.Distance(end) <= 1e-7 for end in expected):
                continue
            if _matches(point, boundary, 1e-7):
                return True
    return False


def _curve_from_wire(wire, scale):
    from magnelio.geo._occ_backend import bounding_box
    from magnelio.geo.curves import Curve
    from magnelio.geo.partition import _wire_ends
    from magnelio.geo.topology import _rescale_copy

    return Curve(
        _build=lambda target_scale: _rescale_copy(wire, scale, target_scale),
        _bounds=bounding_box(wire, scale=scale),
        _ends=_wire_ends(wire, scale),
    )


def project_curve(
    source,
    target,
    *,
    direction=None,
    perspective_source=None,
    closest=False,
    clip=False,
    all_hits=False,
    tolerance=None,
):
    """Implement :meth:`Curve.projected_onto`."""
    if type(closest) is not bool or type(clip) is not bool or type(all_hits) is not bool:
        raise TypeError("projected_onto() closest, clip and all_hits must be bool values.")
    if sum((direction is not None, perspective_source is not None, closest)) != 1:
        raise ValueError(
            "projected_onto() needs exactly one of direction=, perspective_source= or closest=True."
        )
    if closest and all_hits:
        raise ValueError("projected_onto() all_hits applies only to ray projection.")
    if closest and clip:
        raise ValueError("projected_onto() clip applies only to ray projection.")
    if tolerance is not None:
        if not closest:
            raise ValueError("projected_onto() tolerance applies only with closest=True.")
        tolerance = positive(tolerance, "projected_onto(tolerance)")
    if direction is not None:
        direction = normalize_axis(direction, "projected_onto(direction)")
    if perspective_source is not None:
        perspective_source = point3(perspective_source, "projected_onto(perspective_source)")
    target = _target_sheet(target)
    bounds = union_boxes((source._analytic_bbox(), target._analytic_bbox()))
    if perspective_source is not None:
        bounds = union_boxes((bounds, _point_box(perspective_source)))
    scale = choose_scale(*bounds)
    if closest:
        if tolerance is None:
            tolerance = max(1e-8 / scale, 1e-6 * box_diagonal(source._analytic_bbox()))
        tolerance *= scale
    source_shape = source._occ_shape(scale)
    target_shape = target._occ_shape(scale)
    if closest:
        extra = ()
        for _ in range(3):
            wires = _closest_wires(source_shape, target_shape, tolerance, extra)
            if not wires:
                raise ValueError("projected_onto(): the closest-point image has no curve.")
            clipped = [
                _coincident_wires(wire, target_shape, length_tolerance=min(1e-9, tolerance / 10))
                for wire in wires
            ]
            if all(full for _, full in clipped):
                return tuple(_curve_from_wire(wire, scale) for wire in wires)
            new = _trim_seed_parameters(
                [member for members, _ in clipped for member in members], source_shape
            )
            if not new or set(new).issubset(extra):
                break
            extra = tuple(sorted(set(extra) | set(new)))
        raise ValueError("projected_onto(): closest-point trace crosses a target boundary or hole.")
    result = []
    for piece in _split_source_at_face(source_shape, target_shape):
        wires = _projection_wires(
            piece,
            target_shape,
            scale,
            direction=direction,
            perspective_source=perspective_source,
        )
        if not wires:
            coincident, full = _coincident_wires(piece, target_shape)
            if coincident:
                if all_hits:
                    raise ValueError(
                        "projected_onto(): a ray lies in the target and has infinitely many hits."
                    )
                if not full and not clip:
                    raise ValueError(_PARTIAL)
                result.extend(coincident)
                continue
            if clip:
                continue
            raise ValueError("projected_onto(): the curve has no projection on the target.")
        chosen = _select_ray_wires(
            wires,
            piece,
            target_shape,
            scale,
            direction=direction,
            perspective_source=perspective_source,
            clip=clip,
            all_hits=all_hits,
        )
        if not chosen:
            coincident, full = _coincident_wires(piece, target_shape)
            if coincident:
                if all_hits:
                    raise ValueError(
                        "projected_onto(): a ray lies in the target and has infinitely many hits."
                    )
                if not full and not clip:
                    raise ValueError(_PARTIAL)
                result.extend(coincident)
                continue
            if clip:
                continue
            raise ValueError("projected_onto(): the curve has no forward hit on the target.")
        result.extend(chosen)
    return tuple(_curve_from_wire(wire, scale) for wire in result)
