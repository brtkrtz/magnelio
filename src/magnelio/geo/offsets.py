"""Independent planar and normal offsets of CAD geometry."""

from __future__ import annotations

import math

import numpy as np

from magnelio.geo._axes import normalize_axis
from magnelio.geo._scaling import box_diagonal, choose_scale, pad_box
from magnelio.geo._validate import finite


def _distance(value):
    value = finite(value, "offset distance")
    return value


def _scale_and_center(bounds):
    diagonal = box_diagonal(bounds)
    if diagonal <= 0:
        raise ValueError("offset(): the geometry has no spatial extent.")
    base = choose_scale(*bounds)
    factor = 2.0 ** round(math.log2(128.0 / (diagonal * base)))
    return base * factor, np.mean(np.asarray(bounds, dtype=float), axis=0)


def _move(shape, translation):
    from magnelio.geo._sweep_laws import _transform

    return _transform(shape, np.eye(3), translation)


def _offset_wire(wire, distance, normal, *, closed):
    from OCC.Core.BRepAdaptor import BRepAdaptor_CompCurve
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeOffset
    from OCC.Core.GeomAbs import GeomAbs_Arc
    from OCC.Core.gp import gp_Dir, gp_Pln, gp_Pnt, gp_Vec
    from OCC.Core.TopoDS import topods

    from magnelio.geo._occ_backend import _shape_wires, make_wire_face
    from magnelio.geo.topology import _normal

    if closed:
        face = make_wire_face(wire)
        # A counterclockwise boundary has its region on the left. The
        # kernel's positive face offset expands that region to the right.
        sign = -1.0 if np.dot(_normal(face), normal) > 0 else 1.0
        maker = BRepOffsetAPI_MakeOffset(face, GeomAbs_Arc)
        maker.Perform(sign * distance)
    else:
        curve = BRepAdaptor_CompCurve(wire)
        point, tangent = gp_Pnt(), gp_Vec()
        curve.D1(curve.FirstParameter(), point, tangent)
        if tangent.Magnitude() <= 1e-12:
            raise ValueError("offset(): the Curve has no starting tangent.")
        tangent.Normalize()
        lateral = gp_Vec(*normal).Crossed(tangent)
        expected = point.Translated(lateral.Multiplied(distance))
        end, end_tangent = gp_Pnt(), gp_Vec()
        curve.D1(curve.LastParameter(), end, end_tangent)
        if end_tangent.Magnitude() <= 1e-12:
            raise ValueError("offset(): the Curve has no ending tangent.")
        end_tangent.Normalize()
        end_lateral = gp_Vec(*normal).Crossed(end_tangent)
        expected_end = end.Translated(end_lateral.Multiplied(distance))
        plane = BRepBuilderAPI_MakeFace(gp_Pln(point, gp_Dir(*normal))).Face()
        candidates = []
        for reversed_input in (False, True):
            trial = BRepOffsetAPI_MakeOffset()
            trial.Init(plane, GeomAbs_Arc, True)
            argument = topods.Wire(wire.Reversed()) if reversed_input else wire
            trial.AddWire(argument)
            trial.Perform(abs(distance))
            built = trial.Shape() if trial.IsDone() else None
            if built is None or built.IsNull():
                continue
            pieces = _shape_wires(built)
            for piece in pieces:
                component = BRepAdaptor_CompCurve(piece)
                parameter = (
                    component.LastParameter() if reversed_input else component.FirstParameter()
                )
                final_parameter = (
                    component.FirstParameter() if reversed_input else component.LastParameter()
                )
                mismatch = max(
                    component.Value(parameter).Distance(expected),
                    component.Value(final_parameter).Distance(expected_end),
                )
                candidates.append((mismatch, reversed_input, pieces))
        if not candidates:
            raise ValueError("offset(): the planar offset could not be constructed.")
        mismatch, reversed_input, pieces = min(candidates, key=lambda candidate: candidate[0])
        if mismatch > max(1e-6, 1e-6 * abs(distance)):
            raise ValueError("offset(): the CAD kernel could not keep the requested side.")
        return tuple(topods.Wire(piece.Reversed()) if reversed_input else piece for piece in pieces)
    if not maker.IsDone():
        raise ValueError("offset(): the planar offset could not be constructed.")
    result = maker.Shape()
    return () if result.IsNull() else tuple(_shape_wires(result))


def _circle_radius(wire):
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.GeomAbs import GeomAbs_Circle

    from magnelio.geo._occ_backend import _wire_edges

    edges = _wire_edges(wire)
    if len(edges) != 1:
        return None
    curve = BRepAdaptor_Curve(edges[0])
    return curve.Circle().Radius() if curve.GetType() == GeomAbs_Circle else None


def _ellipse_minor_radius(wire):
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.GeomAbs import GeomAbs_Ellipse

    from magnelio.geo._occ_backend import _wire_edges

    edges = _wire_edges(wire)
    if len(edges) != 1:
        return None
    curve = BRepAdaptor_Curve(edges[0])
    return curve.Ellipse().MinorRadius() if curve.GetType() == GeomAbs_Ellipse else None


def offset_curve(source, distance, *, normal):
    """Offset a directed planar curve to its left or right."""
    from OCC.Core.BRepCheck import BRepCheck_Analyzer

    from magnelio.geo._occ_backend import _wire_is_straight, wire_plane_normal
    from magnelio.geo.projection import _curve_from_wire

    distance = _distance(distance)
    normal = normalize_axis(normal, "offset normal")
    bounds = source._analytic_bbox()
    scale, center = _scale_and_center(bounds)
    wire = source._occ_shape(scale)
    plane_normal = wire_plane_normal(wire)
    if plane_normal is None:
        if not _wire_is_straight(wire):
            raise ValueError("offset(): the Curve must lie in the specified plane.")
        from OCC.Core.BRepAdaptor import BRepAdaptor_CompCurve
        from OCC.Core.gp import gp_Pnt, gp_Vec

        adaptor = BRepAdaptor_CompCurve(wire)
        tangent = gp_Vec()
        adaptor.D1(adaptor.FirstParameter(), gp_Pnt(), tangent)
        tangent.Normalize()
        if abs(np.dot((tangent.X(), tangent.Y(), tangent.Z()), normal)) > 1e-8:
            raise ValueError("offset(): the normal must be perpendicular to the Curve.")
    elif abs(np.dot(plane_normal, normal)) < 1 - 1e-8:
        raise ValueError("offset(): the normal must be perpendicular to the Curve plane.")
    if distance == 0:
        return (_curve_from_wire(wire, scale),)
    if source.is_closed and (radius := _circle_radius(wire)) is not None:
        from magnelio.geo._occ_backend import make_wire_face
        from magnelio.geo.topology import _normal

        side = np.dot(_normal(make_wire_face(wire)), normal)
        if distance * scale * side >= radius:
            return ()
    if source.is_closed:
        from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
        from OCC.Core.GeomAbs import GeomAbs_Ellipse

        from magnelio.geo._occ_backend import _wire_edges, make_wire_face
        from magnelio.geo.topology import _normal

        edges = _wire_edges(wire)
        if len(edges) == 1:
            adaptor = BRepAdaptor_Curve(edges[0])
            if adaptor.GetType() == GeomAbs_Ellipse:
                ellipse = adaptor.Ellipse()
                curvature_radius = ellipse.MinorRadius() ** 2 / ellipse.MajorRadius()
                side = np.dot(_normal(make_wire_face(wire)), normal)
                if distance * scale * side >= curvature_radius:
                    raise ValueError("offset(): the Curve offset develops a curvature cusp.")
    origin = center * scale
    centered = _move(wire, -origin)
    output = []
    for piece in _offset_wire(centered, distance * scale, normal, closed=source.is_closed):
        if not BRepCheck_Analyzer(piece).IsValid():
            raise ValueError("offset(): the Curve offset has invalid or crossing branches.")
        output.append(_curve_from_wire(_move(piece, origin), scale))
    return tuple(output)


def _profile_boundary(wire, distance, scale, center, *, hole=False):
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.GeomAbs import GeomAbs_Ellipse

    from magnelio.geo._occ_backend import _wire_edges, make_wire_face
    from magnelio.geo.topology import _normal

    origin = center * scale
    centered = _move(wire, -origin)
    face = make_wire_face(centered)
    # For a closed region, positive distance always grows its interior.
    edges = _wire_edges(centered)
    if all(BRepAdaptor_Curve(edge).GetType() == GeomAbs_Ellipse for edge in edges):
        from magnelio.geo._sweep_laws import _boundary_face, _offset_boundary

        ellipse = BRepAdaptor_Curve(edges[0]).Ellipse()
        inward = distance * scale if hole else -distance * scale
        cusp = ellipse.MinorRadius() ** 2 / ellipse.MajorRadius()
        if inward <= -cusp:
            side = -distance if hole else distance
            pieces = _offset_wire(centered, side * scale, _normal(face), closed=True)
        else:
            try:
                pieces = (_offset_boundary(_boundary_face(centered, _normal(face)), inward),)
            except ValueError as error:
                raise ValueError(str(error).replace("swept(draft_deg=...)", "offset()")) from error
    else:
        pieces = _offset_wire(centered, distance * scale, _normal(face), closed=True)
    output = []
    for piece in pieces:
        if not BRepCheck_Analyzer(piece).IsValid():
            raise ValueError("offset(): a profile boundary folds or intersects itself.")
        output.append(_move(piece, origin))
    return output


def offset_profile(source, distance):
    """Dilate or erode a planar material region, retaining all components."""
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCC.Core.BRepCheck import BRepCheck_Analyzer

    from magnelio.geo._occ_backend import (
        _shape_wires,
        bounding_box,
        extract_face_wire,
        keep_operands_intact,
        make_wire_face,
    )
    from magnelio.geo._sheet import Profile
    from magnelio.geo.partition import _pieces
    from magnelio.geo.topology import _detached_class

    distance = _distance(distance)
    bounds = source._analytic_bbox()
    scale, center = _scale_and_center(bounds)
    face = source._occ_shape(scale)
    if distance == 0:
        return (_detached_class(Profile)(face, scale, bounds, source.material),)
    outer = extract_face_wire(face)
    holes = [wire for wire in _shape_wires(face) if not wire.IsSame(outer)]
    outer_radius = _circle_radius(outer) or _ellipse_minor_radius(outer)
    outer_wires = (
        []
        if outer_radius is not None and -distance * scale >= outer_radius
        else _profile_boundary(outer, -distance, scale, center)
    )
    hole_wires = [
        wire
        for hole in holes
        if not (
            (radius := _circle_radius(hole) or _ellipse_minor_radius(hole)) is not None
            and distance * scale >= radius
        )
        for wire in _profile_boundary(hole, -distance, scale, center, hole=True)
    ]
    regions = [make_wire_face(wire) for wire in outer_wires]
    for hole in hole_wires:
        tool = make_wire_face(hole)
        next_regions = []
        for region in regions:
            cut = BRepAlgoAPI_Cut(region, tool)
            keep_operands_intact(cut)
            cut.Build()
            if not cut.IsDone():
                raise ValueError("offset(): profile boundaries could not be resolved.")
            next_regions.extend(_pieces(cut.Shape(), "face"))
        regions = next_regions
    result = []
    for region in regions:
        if not BRepCheck_Analyzer(region).IsValid():
            raise ValueError("offset(): the resulting profile is invalid.")
        result.append(
            _detached_class(Profile)(
                region,
                scale,
                bounding_box(region, scale=scale),
                source.material,
            )
        )
    return tuple(result)


def offset_sheet(source, distance, *, tolerance=None):
    """Move a bounded curved sheet along its oriented normal."""
    from OCC.Core.BOPAlgo import BOPAlgo_CheckerSI
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeOffsetShape
    from OCC.Core.TopTools import TopTools_ListOfShape

    from magnelio.geo._occ_backend import bounding_box
    from magnelio.geo.partition import _pieces
    from magnelio.geo.surfaces import Surface
    from magnelio.geo.topology import _detached_class

    distance = _distance(distance)
    bounds = source._analytic_bbox()
    scale = choose_scale(*bounds)
    extent = box_diagonal(bounds)
    if tolerance is None:
        tolerance = max(extent * 1e-6, 1e-7 / scale)
    else:
        tolerance = finite(tolerance, "offset tolerance")
        if tolerance <= 0:
            raise ValueError("offset tolerance must be positive.")
    face = source._occ_shape(scale)
    if distance == 0:
        return (_detached_class(Surface)(face, scale, bounds, source.material),)
    maker = BRepOffsetAPI_MakeOffsetShape()
    maker.PerformBySimple(face, distance * scale)
    if not maker.IsDone() or maker.Shape().IsNull():
        raise ValueError("offset(): the sheet normal offset could not be constructed.")
    pieces = _pieces(maker.Shape(), "face")
    if len(pieces) != 1 or not BRepCheck_Analyzer(pieces[0]).IsValid():
        raise ValueError("offset(): the sheet normal offset is invalid or disconnected.")
    result = pieces[0]
    _check_sheet_offset(face, result, distance * scale, tolerance * scale)
    candidates = TopTools_ListOfShape()
    candidates.Append(result)
    checker = BOPAlgo_CheckerSI()
    checker.SetArguments(candidates)
    checker.Perform()
    if checker.HasErrors():
        raise ValueError("offset(): the sheet offset intersects itself.")
    return (
        _detached_class(Surface)(
            result,
            scale,
            pad_box(bounding_box(result, scale=scale), tolerance),
            source.material,
        ),
    )


def _check_sheet_offset(source, result, distance, tolerance):
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.gp import gp_Pnt, gp_Vec

    original = BRepAdaptor_Surface(source)
    shifted = BRepAdaptor_Surface(result)
    bounds = (
        original.FirstUParameter(),
        original.LastUParameter(),
        original.FirstVParameter(),
        original.LastVParameter(),
    )
    other_bounds = (
        shifted.FirstUParameter(),
        shifted.LastUParameter(),
        shifted.FirstVParameter(),
        shifted.LastVParameter(),
    )
    if any(abs(a - b) > 1e-7 * max(1.0, abs(a)) for a, b in zip(bounds, other_bounds)):
        raise ValueError("offset(): the sheet parameterization changed unexpectedly.")
    sign = -1 if source.Orientation() != result.Orientation() else 1
    for u in np.linspace(bounds[0], bounds[1], 9):
        for v in np.linspace(bounds[2], bounds[3], 9):
            p, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
            q, qu, qv = gp_Pnt(), gp_Vec(), gp_Vec()
            original.D1(float(u), float(v), p, du, dv)
            shifted.D1(float(u), float(v), q, qu, qv)
            n = du.Crossed(dv)
            m = qu.Crossed(qv)
            if n.Magnitude() <= 1e-12 or m.Magnitude() <= 1e-12:
                raise ValueError("offset(): the sheet has a singular normal.")
            if sign * n.Dot(m) <= 0:
                raise ValueError("offset(): the sheet offset folds across a curvature singularity.")
            n.Normalize()
            if source.Orientation() != 0:
                n.Reverse()
            expected = p.Translated(n.Multiplied(distance))
            if expected.Distance(q) > tolerance:
                raise ValueError("offset(): the sheet offset exceeds the geometric tolerance.")
