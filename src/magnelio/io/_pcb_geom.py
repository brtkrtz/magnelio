"""Board records into kernel geometry: 2D first, one prism per layer.

The expensive and fragile way to build a board is to extrude every pad
and every track into its own 35 µm slab and fuse the slabs.  Boolean
operations on thousands of solids that thin produce slivers, and slivers
are where a geometry kernel loses faces.

So nothing is extruded until a layer is finished.  Every drawn object
becomes a *face* in the z = 0 plane, all the faces of one layer are
merged there — where the operands are coplanar by construction, which
is the case Booleans handle best — and the finished area is extruded
once.  A hole drilled through the board is a circle cut from each
layer's face before it is extruded, and the via that fills it is a
cylinder built to the same circle: the copper barrel and the layers it
joins meet on coincident faces, with no overlap to resolve.

Nothing in this module knows about the stackup or about materials; it
turns records into shapes.  Coordinates arrive in meters and are
multiplied by the construction scale on the way in.
"""

from __future__ import annotations

import math

from magnelio.geo._occ_backend import (
    boolean_difference_many,
    boolean_intersection,
    boolean_union,
    make_extrude,
    make_face_with_holes,
    unify_same_domain,
)
from magnelio.io._gerber import (
    ArcSegment,
    ArcStroke,
    Circle,
    Flash,
    GerberLayer,
    LineSegment,
    MacroAperture,
    MacroCenterLine,
    MacroCircle,
    MacroOutline,
    MacroPolygon,
    MacroVectorLine,
    Obround,
    Rect,
    Region,
    RegularPolygon,
    Stroke,
)

# Points sampled along an arc when a loop is reduced to a polygon for
# area and containment tests.  Only the topology of the nesting is
# decided from it, never the geometry that is built.
_ARC_SAMPLES = 16


# ─────────────────────────────────────────────────────────────────────
# wires
# ─────────────────────────────────────────────────────────────────────


def _point(xy):
    from OCC.Core.gp import gp_Pnt  # noqa: PLC0415

    return gp_Pnt(xy[0], xy[1], 0.0)


def _line_edge(start, end):
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge  # noqa: PLC0415

    return BRepBuilderAPI_MakeEdge(_point(start), _point(end)).Edge()


def _sweep(start_angle: float, end_angle: float, clockwise: bool) -> float:
    """Signed angle swept from *start_angle* to *end_angle* [rad].

    A start that coincides with the end is a full turn, not a zero one:
    that is how the format writes a complete circle.
    """
    turn = 2.0 * math.pi
    if clockwise:
        swept = -((start_angle - end_angle) % turn)
        return -turn if swept == 0.0 else swept
    swept = (end_angle - start_angle) % turn
    return turn if swept == 0.0 else swept


def _angle(center, point) -> float:
    return math.atan2(point[1] - center[1], point[0] - center[0])


def _on_circle(center, radius: float, angle: float):
    return (center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle))


def _arc_edge(start, end, center, clockwise: bool):
    """A circular edge from *start* to *end* around *center*.

    Built through three points rather than from an angular range, so
    the edge ends exactly on the coordinates the file gave: a track and
    the arc continuing it have to meet, and the file's own radius at
    the two ends may differ by a rounding step.
    """
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge  # noqa: PLC0415
    from OCC.Core.GC import GC_MakeArcOfCircle  # noqa: PLC0415

    start_angle = _angle(center, start)
    end_angle = _angle(center, end)
    radius = 0.5 * (math.dist(center, start) + math.dist(center, end))
    middle = _on_circle(
        center, radius, start_angle + 0.5 * _sweep(start_angle, end_angle, clockwise)
    )
    arc = GC_MakeArcOfCircle(_point(start), _point(middle), _point(end))
    if not arc.IsDone():
        raise ValueError("Could not build a circular arc from the coordinates in the file.")
    return BRepBuilderAPI_MakeEdge(arc.Value()).Edge()


def _wire(edges):
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire  # noqa: PLC0415

    builder = BRepBuilderAPI_MakeWire()
    for edge in edges:
        builder.Add(edge)
    if not builder.IsDone():
        raise ValueError("Could not chain the outline into a closed wire.")
    return builder.Wire()


def _circle_wire(center, radius: float):
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge  # noqa: PLC0415
    from OCC.Core.gp import gp_Ax2, gp_Circ, gp_Dir  # noqa: PLC0415

    axis = gp_Ax2(_point(center), gp_Dir(0.0, 0.0, 1.0))
    return _wire([BRepBuilderAPI_MakeEdge(gp_Circ(axis, radius)).Edge()])


def _polygon_wire(points):
    edges = [
        _line_edge(points[i], points[(i + 1) % len(points)])
        for i in range(len(points))
        if math.dist(points[i], points[(i + 1) % len(points)]) > 0.0
    ]
    if len(edges) < 3:
        raise ValueError("A closed outline needs at least three distinct corners.")
    return _wire(edges)


def _rectangle_wire(center, width: float, height: float):
    half_x, half_y = 0.5 * width, 0.5 * height
    return _polygon_wire(
        [
            (center[0] - half_x, center[1] - half_y),
            (center[0] + half_x, center[1] - half_y),
            (center[0] + half_x, center[1] + half_y),
            (center[0] - half_x, center[1] + half_y),
        ]
    )


def _capsule_wire(start, end, radius: float):
    """A straight track of radius *radius* with semicircular ends."""
    length = math.dist(start, end)
    if length == 0.0:
        return _circle_wire(start, radius)
    along = ((end[0] - start[0]) / length, (end[1] - start[1]) / length)
    across = (-along[1], along[0])

    def offset(point, sideways, forward=0.0):
        return (
            point[0] + sideways * across[0] + forward * along[0],
            point[1] + sideways * across[1] + forward * along[1],
        )

    left_start, left_end = offset(start, radius), offset(end, radius)
    right_end, right_start = offset(end, -radius), offset(start, -radius)
    return _wire(
        [
            _line_edge(left_start, left_end),
            _arc_edge_through(left_end, right_end, offset(end, 0.0, radius)),
            _line_edge(right_end, right_start),
            _arc_edge_through(right_start, left_start, offset(start, 0.0, -radius)),
        ]
    )


def _arc_edge_through(start, end, middle):
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge  # noqa: PLC0415
    from OCC.Core.GC import GC_MakeArcOfCircle  # noqa: PLC0415

    arc = GC_MakeArcOfCircle(_point(start), _point(middle), _point(end))
    if not arc.IsDone():
        raise ValueError("Could not build the rounded end of a track.")
    return BRepBuilderAPI_MakeEdge(arc.Value()).Edge()


def _arc_track_face(stroke: ArcStroke, radius: float):
    """The area swept by a round aperture running along a circular arc."""
    center = stroke.center
    mean = 0.5 * (math.dist(center, stroke.start) + math.dist(center, stroke.end))
    if mean <= radius:
        raise ValueError(
            f"An arc of radius {mean:g} is drawn with a track {2 * radius:g} "
            f"wide, so the track covers its own centre. Such an arc has no "
            f"well-defined outline; re-export the layer with the arc drawn "
            f"as a filled region."
        )
    start_angle = _angle(center, stroke.start)
    end_angle = _angle(center, stroke.end)
    swept = _sweep(start_angle, end_angle, stroke.clockwise)
    outer, inner = mean + radius, mean - radius

    if abs(abs(swept) - 2.0 * math.pi) < 1e-12:
        return make_face_with_holes(_circle_wire(center, outer), [_circle_wire(center, inner)])

    def cap(angle: float, at_end: bool):
        """Mid-point of the track's semicircular end at *angle*.

        The cap bulges the way the track was heading — forwards at the
        far end, backwards at the near one — so the two arcs close the
        outline instead of folding it back over itself.
        """
        heading = 0.5 * math.pi if not stroke.clockwise else -0.5 * math.pi
        outward = angle + heading + (0.0 if at_end else math.pi)
        return _on_circle(_on_circle(center, mean, angle), radius, outward)

    return make_face_with_holes(
        _wire(
            [
                _arc_edge(
                    _on_circle(center, outer, start_angle),
                    _on_circle(center, outer, end_angle),
                    center,
                    stroke.clockwise,
                ),
                _arc_edge_through(
                    _on_circle(center, outer, end_angle),
                    _on_circle(center, inner, end_angle),
                    cap(end_angle, at_end=True),
                ),
                _arc_edge(
                    _on_circle(center, inner, end_angle),
                    _on_circle(center, inner, start_angle),
                    center,
                    not stroke.clockwise,
                ),
                _arc_edge_through(
                    _on_circle(center, inner, start_angle),
                    _on_circle(center, outer, start_angle),
                    cap(start_angle, at_end=False),
                ),
            ]
        ),
        [],
    )


# ─────────────────────────────────────────────────────────────────────
# transforms
# ─────────────────────────────────────────────────────────────────────


def _transformed(shape, transform):
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform  # noqa: PLC0415

    return BRepBuilderAPI_Transform(shape, transform, True).Shape()


def _translated(shape, dx: float, dy: float, dz: float = 0.0):
    from OCC.Core.gp import gp_Trsf, gp_Vec  # noqa: PLC0415

    transform = gp_Trsf()
    transform.SetTranslation(gp_Vec(dx, dy, dz))
    return _transformed(shape, transform)


def _rotated_about_origin(shape, degrees: float):
    if degrees % 360.0 == 0.0:
        return shape
    from OCC.Core.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf  # noqa: PLC0415

    transform = gp_Trsf()
    transform.SetRotation(
        gp_Ax1(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(0.0, 0.0, 1.0)), math.radians(degrees)
    )
    return _transformed(shape, transform)


def _compound(shapes):
    """Collect shapes side by side, without a Boolean."""
    from OCC.Core.BRep import BRep_Builder  # noqa: PLC0415
    from OCC.Core.TopoDS import TopoDS_Compound  # noqa: PLC0415

    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    for shape in shapes:
        builder.Add(compound, shape)
    return compound


# ─────────────────────────────────────────────────────────────────────
# apertures
# ─────────────────────────────────────────────────────────────────────


def _regular_polygon_wire(center, diameter: float, vertices: int, rotation: float):
    radius = 0.5 * diameter
    start = math.radians(rotation)
    return _polygon_wire(
        [
            _on_circle(center, radius, start + index * 2.0 * math.pi / vertices)
            for index in range(vertices)
        ]
    )


def _macro_face(primitive):
    """One evaluated macro primitive as a face at the macro origin."""
    if isinstance(primitive, MacroCircle):
        face = make_face_with_holes(_circle_wire(primitive.center, 0.5 * primitive.diameter), [])
    elif isinstance(primitive, MacroVectorLine):
        # A vector line has square ends: it is a rectangle, not a track.
        length = math.dist(primitive.start, primitive.end)
        if length == 0.0:
            raise ValueError("A macro vector line of zero length draws nothing.")
        along = (
            (primitive.end[0] - primitive.start[0]) / length,
            (primitive.end[1] - primitive.start[1]) / length,
        )
        across = (-along[1] * 0.5 * primitive.width, along[0] * 0.5 * primitive.width)
        face = make_face_with_holes(
            _polygon_wire(
                [
                    (primitive.start[0] + across[0], primitive.start[1] + across[1]),
                    (primitive.end[0] + across[0], primitive.end[1] + across[1]),
                    (primitive.end[0] - across[0], primitive.end[1] - across[1]),
                    (primitive.start[0] - across[0], primitive.start[1] - across[1]),
                ]
            ),
            [],
        )
    elif isinstance(primitive, MacroCenterLine):
        face = make_face_with_holes(
            _rectangle_wire(primitive.center, primitive.width, primitive.height), []
        )
    elif isinstance(primitive, MacroOutline):
        points = list(primitive.points)
        if len(points) > 1 and math.dist(points[0], points[-1]) == 0.0:
            points.pop()
        face = make_face_with_holes(_polygon_wire(points), [])
    elif isinstance(primitive, MacroPolygon):
        face = make_face_with_holes(
            _regular_polygon_wire(primitive.center, primitive.diameter, primitive.vertices, 0.0), []
        )
    else:  # pragma: no cover - the reader emits no other primitive
        raise ValueError(f"Unknown macro primitive {type(primitive).__name__}.")
    return _rotated_about_origin(face, primitive.rotation)


def aperture_shape(aperture):
    """The area an aperture covers, as a face at the origin."""
    if isinstance(aperture, Circle):
        outer = _circle_wire((0.0, 0.0), 0.5 * aperture.diameter)
    elif isinstance(aperture, Rect):
        outer = _rectangle_wire((0.0, 0.0), aperture.width, aperture.height)
    elif isinstance(aperture, Obround):
        radius = 0.5 * min(aperture.width, aperture.height)
        reach = 0.5 * abs(aperture.width - aperture.height)
        if aperture.width >= aperture.height:
            outer = _capsule_wire((-reach, 0.0), (reach, 0.0), radius)
        else:
            outer = _capsule_wire((0.0, -reach), (0.0, reach), radius)
    elif isinstance(aperture, RegularPolygon):
        outer = _regular_polygon_wire(
            (0.0, 0.0), aperture.diameter, aperture.vertices, aperture.rotation
        )
    elif isinstance(aperture, MacroAperture):
        return _macro_shape(aperture)
    else:  # pragma: no cover - the reader emits no other aperture
        raise ValueError(f"Unknown aperture {type(aperture).__name__}.")

    holes = []
    if aperture.hole:
        holes.append(_circle_wire((0.0, 0.0), 0.5 * aperture.hole))
    return make_face_with_holes(outer, holes)


def _macro_shape(aperture: MacroAperture):
    """A macro aperture, its primitives added and subtracted in order."""
    shape = None
    for primitive in aperture.primitives:
        face = _macro_face(primitive)
        if shape is None:
            if not primitive.exposure:
                continue  # nothing to cut from yet
            shape = face
        elif primitive.exposure:
            shape = boolean_union([shape, face])
        else:
            shape = boolean_difference_many(shape, [face])
    if shape is None:
        raise ValueError(f"Aperture macro {aperture.name!r} exposes no area.")
    return unify_same_domain(shape)


# ─────────────────────────────────────────────────────────────────────
# drawn objects
# ─────────────────────────────────────────────────────────────────────


def _scaled(point, scale: float):
    return (point[0] * scale, point[1] * scale)


def object_shape(obj, scale: float, apertures: dict):
    """The area one drawn object covers, in construction units.

    *apertures* caches the face of each aperture: a board flashes the
    same pad thousands of times, and building it once and translating
    it is both faster and exactly repeatable.
    """
    if isinstance(obj, Flash):
        key = id(obj.aperture)
        face = apertures.get(key)
        if face is None:
            face = apertures[key] = aperture_shape(_scale_aperture(obj.aperture, scale))
        at = _scaled(obj.at, scale)
        return _translated(face, at[0], at[1])
    if isinstance(obj, Stroke):
        if obj.width <= 0.0:
            raise ValueError("A track of zero width covers no copper.")
        return make_face_with_holes(
            _capsule_wire(
                _scaled(obj.start, scale), _scaled(obj.end, scale), 0.5 * obj.width * scale
            ),
            [],
        )
    if isinstance(obj, ArcStroke):
        if obj.width <= 0.0:
            raise ValueError("A track of zero width covers no copper.")
        scaled = ArcStroke(
            start=_scaled(obj.start, scale),
            end=_scaled(obj.end, scale),
            center=_scaled(obj.center, scale),
            clockwise=obj.clockwise,
            width=obj.width * scale,
        )
        return _arc_track_face(scaled, 0.5 * scaled.width)
    if isinstance(obj, Region):
        return _compound(_region_faces(obj.contours, scale))
    raise ValueError(f"Unknown drawn object {type(obj).__name__}.")  # pragma: no cover


def _scale_aperture(aperture, scale: float):
    """The same aperture with every length in construction units."""
    if isinstance(aperture, Circle):
        return Circle(aperture.diameter * scale, _times(aperture.hole, scale))
    if isinstance(aperture, Rect):
        return Rect(aperture.width * scale, aperture.height * scale, _times(aperture.hole, scale))
    if isinstance(aperture, Obround):
        return Obround(
            aperture.width * scale, aperture.height * scale, _times(aperture.hole, scale)
        )
    if isinstance(aperture, RegularPolygon):
        return RegularPolygon(
            aperture.diameter * scale,
            aperture.vertices,
            aperture.rotation,
            _times(aperture.hole, scale),
        )
    return MacroAperture(
        aperture.name, tuple(_scale_primitive(p, scale) for p in aperture.primitives)
    )


def _times(value, scale: float):
    return None if value is None else value * scale


def _scale_primitive(primitive, scale: float):
    if isinstance(primitive, MacroCircle):
        return MacroCircle(
            primitive.exposure,
            primitive.diameter * scale,
            _scaled(primitive.center, scale),
            primitive.rotation,
        )
    if isinstance(primitive, MacroVectorLine):
        return MacroVectorLine(
            primitive.exposure,
            primitive.width * scale,
            _scaled(primitive.start, scale),
            _scaled(primitive.end, scale),
            primitive.rotation,
        )
    if isinstance(primitive, MacroCenterLine):
        return MacroCenterLine(
            primitive.exposure,
            primitive.width * scale,
            primitive.height * scale,
            _scaled(primitive.center, scale),
            primitive.rotation,
        )
    if isinstance(primitive, MacroOutline):
        return MacroOutline(
            primitive.exposure,
            tuple(_scaled(p, scale) for p in primitive.points),
            primitive.rotation,
        )
    return MacroPolygon(
        primitive.exposure,
        primitive.vertices,
        _scaled(primitive.center, scale),
        primitive.diameter * scale,
        primitive.rotation,
    )


# ─────────────────────────────────────────────────────────────────────
# contours: nesting and faces
# ─────────────────────────────────────────────────────────────────────


def _polyline(contour, scale: float) -> list[tuple[float, float]]:
    """A polygon following *contour*, for area and containment only."""
    points: list[tuple[float, float]] = []
    for segment in contour:
        points.append(_scaled(segment.start, scale))
        if isinstance(segment, ArcSegment):
            center = _scaled(segment.center, scale)
            start = _scaled(segment.start, scale)
            end = _scaled(segment.end, scale)
            start_angle = _angle(center, start)
            swept = _sweep(start_angle, _angle(center, end), segment.clockwise)
            radius = 0.5 * (math.dist(center, start) + math.dist(center, end))
            for index in range(1, _ARC_SAMPLES):
                points.append(
                    _on_circle(center, radius, start_angle + swept * index / _ARC_SAMPLES)
                )
    return points


def _signed_area(points) -> float:
    total = 0.0
    for index, point in enumerate(points):
        other = points[(index + 1) % len(points)]
        total += point[0] * other[1] - other[0] * point[1]
    return 0.5 * total


def _contains(polygon, point) -> bool:
    """Ray casting: is *point* inside *polygon*?"""
    inside = False
    x, y = point
    for index, corner in enumerate(polygon):
        other = polygon[(index + 1) % len(polygon)]
        if (corner[1] > y) != (other[1] > y):
            crossing = corner[0] + (y - corner[1]) * (other[0] - corner[0]) / (other[1] - corner[1])
            if crossing > x:
                inside = not inside
    return inside


def _nest(polygons) -> list[tuple[int, list[int]]]:
    """Group contours into ``(outer, holes)`` by containment.

    A contour inside an odd number of others bounds a hole; one inside
    an even number bounds an area of its own.  This is the reading every
    viewer applies, and the only one under which a region drawn with a
    nested contour produces the board the designer saw.
    """
    depth = [0] * len(polygons)
    parent: list[int | None] = [None] * len(polygons)
    for index, polygon in enumerate(polygons):
        probe = polygon[0]
        best_area = math.inf
        for other, candidate in enumerate(polygons):
            if other == index or not _contains(candidate, probe):
                continue
            depth[index] += 1
            area = abs(_signed_area(candidate))
            if area < best_area:
                best_area, parent[index] = area, other

    groups: list[tuple[int, list[int]]] = []
    for index in range(len(polygons)):
        if depth[index] % 2 == 0:
            holes = [
                other
                for other in range(len(polygons))
                if parent[other] == index and depth[other] % 2 == 1
            ]
            groups.append((index, holes))
    return groups


def _contour_wire(contour, scale: float):
    edges = []
    for segment in contour:
        start, end = _scaled(segment.start, scale), _scaled(segment.end, scale)
        if math.dist(start, end) == 0.0 and isinstance(segment, LineSegment):
            continue
        if isinstance(segment, ArcSegment):
            edges.append(_arc_edge(start, end, _scaled(segment.center, scale), segment.clockwise))
        else:
            edges.append(_line_edge(start, end))
    if not edges:
        raise ValueError("A contour without any segment of non-zero length.")
    return _wire(edges)


def _region_faces(contours, scale: float) -> list:
    """The faces one filled region covers."""
    polygons = [_polyline(contour, scale) for contour in contours]
    wires = [_contour_wire(contour, scale) for contour in contours]
    return [
        make_face_with_holes(wires[outer], [wires[hole] for hole in holes])
        for outer, holes in _nest(polygons)
    ]


# ─────────────────────────────────────────────────────────────────────
# merging a layer
# ─────────────────────────────────────────────────────────────────────


def _bbox(shape) -> tuple[float, float, float, float]:
    from OCC.Core.Bnd import Bnd_Box  # noqa: PLC0415
    from OCC.Core.BRepBndLib import brepbndlib  # noqa: PLC0415

    box = Bnd_Box()
    brepbndlib.Add(shape, box, False)  # geometry only, no triangulation
    x_min, y_min, _, x_max, y_max, _ = box.Get()
    return (x_min, y_min, x_max, y_max)


def _clusters(shapes, tolerance: float) -> list[list[int]]:
    """Group shapes whose bounding boxes touch, by a sweep over x.

    Most of a copper layer is pads that touch nothing: putting them
    through a Boolean would cost time and gain no area.  Only shapes
    that could possibly meet are merged, and the sweep is what finds
    them without testing every pair.
    """
    boxes = [_bbox(shape) for shape in shapes]
    parent = list(range(len(shapes)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(a: int, b: int) -> None:
        left, right = root(a), root(b)
        if left != right:
            parent[right] = left

    order = sorted(range(len(shapes)), key=lambda index: boxes[index][0])
    active: list[int] = []
    for index in order:
        x_min, y_min, x_max, y_max = boxes[index]
        active = [other for other in active if boxes[other][2] >= x_min - tolerance]
        for other in active:
            _, other_y_min, _, other_y_max = boxes[other]
            if other_y_max >= y_min - tolerance and other_y_min <= y_max + tolerance:
                join(index, other)
        active.append(index)

    groups: dict[int, list[int]] = {}
    for index in range(len(shapes)):
        groups.setdefault(root(index), []).append(index)
    return list(groups.values())


def merge_faces(shapes: list, tolerance: float):
    """Fuse coplanar faces into as few faces as they really are."""
    if not shapes:
        return None
    if len(shapes) == 1:
        return shapes[0]
    merged = []
    for group in _clusters(shapes, tolerance):
        if len(group) == 1:
            merged.append(shapes[group[0]])
        else:
            merged.append(unify_same_domain(boolean_union([shapes[index] for index in group])))
    return merged[0] if len(merged) == 1 else _compound(merged)


def layer_shape(layer: GerberLayer, scale: float):
    """The copper of one Gerber layer, as faces in the z = 0 plane.

    Polarity is folded here rather than at the end: a clear object
    removes what was drawn *before* it and nothing that comes after, so
    the file's order decides the result and the objects are worked
    through in runs of equal polarity.
    """
    tolerance = max(layer.resolution * scale, 0.0)
    apertures: dict = {}
    result = None
    batch: list = []
    batch_dark = True

    def flush(shape):
        merged = merge_faces(batch, tolerance)
        batch.clear()
        if merged is None:
            return shape
        if shape is None:
            return merged if batch_dark else None
        if batch_dark:
            return boolean_union([shape, merged])
        return boolean_difference_many(shape, [merged])

    for dark, obj in layer.objects:
        if dark != batch_dark and batch:
            result = flush(result)
        batch_dark = dark
        batch.append(object_shape(obj, scale, apertures))
    result = flush(result)
    return result


# ─────────────────────────────────────────────────────────────────────
# the board outline
# ─────────────────────────────────────────────────────────────────────


def _outline_segments(layer: GerberLayer) -> list:
    """The centre lines the profile layer draws, as contour segments."""
    segments: list = []
    for dark, obj in layer.objects:
        if not dark:
            continue
        if isinstance(obj, Stroke):
            segments.append(LineSegment(start=obj.start, end=obj.end))
        elif isinstance(obj, ArcStroke):
            segments.append(
                ArcSegment(start=obj.start, end=obj.end, center=obj.center, clockwise=obj.clockwise)
            )
        elif isinstance(obj, Flash):
            raise ValueError(
                "The board outline layer flashes an aperture. An outline is "
                "a closed line; re-export it with the outline drawn as lines "
                "and arcs."
            )
    return segments


def _chain(segments, tolerance: float) -> list[list]:
    """Chain loose segments into closed loops through their endpoints.

    A wire builder would take whichever branch it met first at a node
    where three segments join and report success; here a node that is
    not met by exactly two segments is an error, because an outline
    that branches does not describe one board.
    """
    nodes: list[tuple[float, float]] = []
    incident: list[list[tuple[int, bool]]] = []

    def node_of(point) -> int:
        for index, known in enumerate(nodes):
            if math.dist(known, point) <= tolerance:
                return index
        nodes.append(point)
        incident.append([])
        return len(nodes) - 1

    ends = []
    for index, segment in enumerate(segments):
        start, end = node_of(segment.start), node_of(segment.end)
        if start == end and math.dist(segment.start, segment.end) <= tolerance:
            if isinstance(segment, ArcSegment):
                ends.append((start, end))  # a full circle: its own loop
                incident[start].append((index, True))
                incident[end].append((index, False))
                continue
            raise ValueError("The board outline contains a segment of zero length.")
        ends.append((start, end))
        incident[start].append((index, True))
        incident[end].append((index, False))

    for index, touching in enumerate(incident):
        if len(touching) != 2:
            where = nodes[index]
            raise ValueError(
                f"The board outline is not a set of closed loops: "
                f"{len(touching)} segment(s) meet at ({where[0]:.6g}, "
                f"{where[1]:.6g}). An outline must close and must not branch."
            )

    loops: list[list] = []
    used = [False] * len(segments)
    for start_index in range(len(segments)):
        if used[start_index]:
            continue
        loop: list = []
        index, forward = start_index, True
        while not used[index]:
            used[index] = True
            segment = segments[index]
            loop.append(segment if forward else _reversed(segment))
            node = ends[index][1] if forward else ends[index][0]
            following = [pair for pair in incident[node] if pair[0] != index]
            if not following:
                break
            index, forward = following[0]
        loops.append(loop)
    return loops


def _reversed(segment):
    if isinstance(segment, ArcSegment):
        return ArcSegment(
            start=segment.end,
            end=segment.start,
            center=segment.center,
            clockwise=not segment.clockwise,
        )
    return LineSegment(start=segment.end, end=segment.start)


def outline_faces(layer: GerberLayer, scale: float) -> list:
    """The board area the profile layer encloses, as faces.

    A profile is drawn as a line, not as an area, so the area has to be
    recovered: the segments are chained into closed loops, and a loop
    inside another is a cut-out rather than a second board.
    """
    regions = [obj for dark, obj in layer.objects if dark and isinstance(obj, Region)]
    if regions:
        faces = []
        for region in regions:
            faces.extend(_region_faces(region.contours, scale))
        return faces

    segments = _outline_segments(layer)
    if not segments:
        raise ValueError(
            "The board outline layer draws nothing. The profile file has to carry the board edge."
        )
    loops = _chain(segments, max(layer.resolution, 0.0) * 2.0)
    polygons = [_polyline(loop, scale) for loop in loops]
    wires = [_contour_wire(loop, scale) for loop in loops]
    return [
        make_face_with_holes(wires[outer], [wires[hole] for hole in holes])
        for outer, holes in _nest(polygons)
    ]


# ─────────────────────────────────────────────────────────────────────
# from faces to solids
# ─────────────────────────────────────────────────────────────────────


def circle_face(center, diameter: float, scale: float):
    """A drilled hole, as the face it removes from a layer."""
    return make_face_with_holes(_circle_wire(_scaled(center, scale), 0.5 * diameter * scale), [])


def slot_face(start, end, diameter: float, scale: float):
    """A routed slot, as the face it removes from a layer."""
    return make_face_with_holes(
        _capsule_wire(_scaled(start, scale), _scaled(end, scale), 0.5 * diameter * scale), []
    )


def clip(shape, boundary):
    """Keep only the part of *shape* inside *boundary*."""
    return boolean_intersection(shape, boundary)


def cut(shape, tools: list):
    """Remove every shape in *tools* from *shape*."""
    return boolean_difference_many(shape, tools) if tools else shape


def _faces_of(shape) -> list:
    """Every face inside *shape*, in kernel order."""
    from OCC.Core.TopAbs import TopAbs_FACE  # noqa: PLC0415
    from OCC.Core.TopExp import TopExp_Explorer  # noqa: PLC0415
    from OCC.Core.TopoDS import topods  # noqa: PLC0415

    faces = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)  # pyright: ignore[reportArgumentType]
    while explorer.More():
        faces.append(topods.Face(explorer.Current()))
        explorer.Next()
    return faces


def extrude(face_shape, z_bottom: float, z_top: float, scale: float):
    """Raise a face from ``z_bottom`` to ``z_top``.

    A layer of a thousand isolated pads is a compound of a thousand
    faces, and handing that compound to the kernel's prism builder in
    one piece costs superlinearly in the number of faces — measured
    2.9 s against 0.11 s for 3600 pads extruded one at a time, for the
    same solids.  Each face is therefore raised on its own.
    """
    thickness = (z_top - z_bottom) * scale
    if thickness <= 0.0:
        raise ValueError("A layer needs a positive thickness to be extruded.")
    lifted = _translated(face_shape, 0.0, 0.0, z_bottom * scale)
    faces = _faces_of(lifted)
    if not faces:
        raise ValueError("There is no area here to raise into a solid.")
    if len(faces) == 1:
        return make_extrude(faces[0], (0.0, 0.0, thickness))
    return _compound([make_extrude(face, (0.0, 0.0, thickness)) for face in faces])
