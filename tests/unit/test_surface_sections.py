"""Independent near-tangency and bounded-face material-area references."""

from __future__ import annotations

import numpy as np
import pytest
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepTools import breptools
from OCC.Core.Geom import Geom_BezierCurve, Geom_Circle, Geom_Plane, Geom_TrimmedCurve
from OCC.Core.Geom2d import Geom2d_Curve
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pln, gp_Pnt, gp_Vec, gp_Vec2d
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_INTERNAL
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from scipy.integrate import quad

from magnelio import geo
from magnelio.geo._occ_backend import _PlanarSectionEngine, cross_section_polygons
from magnelio.geo._polygon_clip import polygon_area
from magnelio.geo._section_policy import section_policy
from magnelio.geo._surface_sections import (
    _endpoint_partners,
    _PreparedGeometry,
    _refine_trace,
    _strict_inside,
    _SurfaceRouter,
    _SurfaceSectionEngine,
    _tessellate,
    _walk_contours,
)

R, BORE, HEIGHT, DEFLECTION = 2.3e-3, 0.7e-3, 10e-3, 2.5e-6


@pytest.fixture(autouse=True)
def enable_robust_sections():
    with section_policy(True):
        yield


def drilled(axis, offset=0):
    origin = (offset, 0, -HEIGHT / 2) if axis == "x" else (0, offset, -HEIGHT / 2)
    cylinder = geo.Cylinder(origin=origin, axis="z", radius=R, height=HEIGHT, material="pec")
    start = (offset - 2 * R, 0, 0) if axis == "x" else (0, offset - 2 * R, 0)
    bore = geo.Cylinder(origin=start, axis=axis, radius=BORE, height=4 * R, material="pec")

    def square(z, side):
        return geo.Profile.polygon(
            [
                (u, v, z)
                for u, v in (
                    (-side / 2, -side / 2),
                    (side / 2, -side / 2),
                    (side / 2, side / 2),
                    (-side / 2, side / 2),
                )
            ]
        )

    loft = geo.Loft(square(0.003, 1.2 * R), square(0.009, 0.6 * R), blend="ruled", material="pec")
    return (cylinder - bore) + loft


def area(polygons):
    return abs(sum(polygon_area(p - p[0]) for p in polygons))


def exact_drilled(d):
    if d <= 0:
        return 0.0
    half_width = np.sqrt(d * (2 * R - d))
    return quad(
        lambda x: HEIGHT - 2 * np.sqrt(BORE**2 - x * x),
        -half_width,
        half_width,
        epsabs=1e-20,
        epsrel=1e-12,
    )[0]


@pytest.fixture(scope="module", params=[("x", 0), ("y", 0), ("x", 0.0021), ("y", 0.0021)])
def drilled_case(request):
    axis, offset = request.param
    shape = drilled(axis, offset)
    occ = shape._occ_shape(1)
    return axis, offset, occ, _PlanarSectionEngine(occ, deflection=DEFLECTION)


@pytest.mark.parametrize("d", [1e-6, 1e-7, 1e-9, 0, -1e-9])
def test_trimmed_cylinder_retains_both_regions(drilled_case, d):
    axis, offset, occ, engine = drilled_case
    pos = offset + R - d
    expected = exact_drilled(d)
    direct = cross_section_polygons(occ, axis, pos, deflection=DEFLECTION)
    fast = engine.section("xyz".index(axis), pos)
    assert fast is not None
    for polygons in (direct, fast):
        if expected:
            assert len(polygons) == 2
            assert area(polygons) == pytest.approx(expected, rel=1e-4, abs=0)
        else:
            assert polygons == []


def test_compiled_batch_does_not_delegate_a_resolved_sliver():
    shape = geo.Cylinder(origin=(0, 0, 0), axis="z", height=HEIGHT, radius=R, material="pec")
    engine = _PlanarSectionEngine(shape._occ_shape(1), deflection=DEFLECTION)
    positions = [R - 1e-9, 0, R - 1e-7, R, R + 1e-9]
    scalar = [engine.section(0, pos) for pos in positions]
    batch = engine.sections(0, positions)
    packed = engine.sections_packed(0, positions)
    for i, (reference, result) in enumerate(zip(scalar, batch)):
        assert (result is None) == (reference is None)
        if result is not None:
            assert area(result) == pytest.approx(area(reference), rel=1e-12, abs=1e-20)
            assert area(packed.polygons(i)) == pytest.approx(area(reference), rel=1e-12, abs=1e-20)


def test_regular_planes_leave_the_conditioned_representation_unprepared():
    body = drilled("y")
    router = _SurfaceRouter(body._occ_shape(1), 1, DEFLECTION)
    assert not router.sensitive(1, 0.001)
    assert router.engine is None
    assert router.sensitive(1, R - 1e-9)
    router.section(1, R - 1e-9)
    assert router.engine is not None


def test_freeform_true_tangency_is_not_a_parameter_pole():
    surface = geo.Surface.parametric(
        lambda u, v: (u, v, (u * u + v * v) / 0.02),
        u=(-0.003, 0.003),
        v=(-0.003, 0.003),
        samples=(9, 9),
    )
    face = topods.Face(TopExp_Explorer(surface._occ_shape(1), TopAbs_FACE).Current())
    cad = BRep_Tool.Surface(face)
    a, b, c, d = breptools.UVBounds(face)
    point, du, dv, duu, dvv, duv = gp_Pnt(), gp_Vec(), gp_Vec(), gp_Vec(), gp_Vec(), gp_Vec()
    cad.D2((a + b) / 2, (c + d) / 2, point, du, dv, duu, dvv, duv)
    projected_jacobian = abs(du.X() * dv.Y() - du.Y() * dv.X())
    hessian = np.array(((duu.Z(), duv.Z()), (duv.Z(), dvv.Z())))
    body = surface.extruded(vector=(0, 0, -0.002), material="pec")
    router = _SurfaceRouter(body._occ_shape(1), 1, 1e-8)
    delta = 1e-9
    pos = point.Z() - 0.002 + delta
    assert router.sensitive(2, pos)
    polygons = router.section(2, pos)
    assert polygons is not None
    expected = 2 * np.pi * delta * projected_jacobian / np.sqrt(np.linalg.det(hessian))
    assert area(polygons) == pytest.approx(expected, rel=5e-4)


def test_smooth_polar_parameter_pole_does_not_trigger_geometric_grazing():
    def paraboloid(r, phi):
        x, y = 0.05 + r * np.cos(phi), r * np.sin(phi)
        return x, y, (x * x + y * y) / 0.48

    body = geo.Surface.parametric(
        paraboloid, u=(0, 0.06), v=(0, 2 * np.pi), samples=(16, 32)
    ).extruded(vector=(0, 0, -0.02), material="pec")
    router = _SurfaceRouter(body._occ_shape(1), 1, 1e-4)
    assert not router.sensitive(1, 0)
    assert not router.sensitive(2, 0.05**2 / 0.48)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_coplanar_material_faces_retain_their_area(axis):
    block = geo.Brick(origin=(0, 0, 0), size=(0.001, 0.001, 0.001), material="pec")
    for pos in (0, 0.001):
        result = cross_section_polygons(
            block._occ_shape(1), "xyz"[axis], pos, deflection=DEFLECTION, exact_at_faces=True
        )
        assert area(result) == pytest.approx(1e-6, rel=1e-12)


def test_disabling_the_guard_reproduces_the_original_defect():
    body = drilled("y")
    occ = body._occ_shape(1)
    with section_policy(False):
        old = _PlanarSectionEngine(occ, deflection=DEFLECTION).section(1, R - 1e-7)
        assert cross_section_polygons(occ, "y", R - 1e-7, deflection=DEFLECTION) == []
    assert area(old) < 0.8 * exact_drilled(1e-7)
    corrected = _PlanarSectionEngine(occ, deflection=DEFLECTION).section(1, R - 1e-7)
    assert area(corrected) == pytest.approx(exact_drilled(1e-7), rel=1e-4)


@pytest.mark.parametrize("enabled", [False, True])
def test_spawn_worker_obeys_selected_cut_route(enabled):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from magnelio.geo import _occ_backend as backend

    body = drilled("y")
    blob = breptools.WriteToString(body._occ_shape(1))
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=backend._section_worker_init,
        initargs=([(0, blob)], enabled),
    ) as pool:
        polygons = pool.submit(
            backend._section_worker, ("y", R - 1e-7, 0, DEFLECTION, 1.0, 1e-7, "")
        ).result(timeout=60)
    if enabled:
        assert area(polygons) == pytest.approx(exact_drilled(1e-7), rel=1e-4)
    else:
        assert polygons == []


@pytest.mark.parametrize("shift", [-1e-9, 0, 1e-9])
def test_torus_internal_tangent_preserves_touching_regions(shift):
    major, minor = 0.0035, 0.0012
    x = major - minor + shift
    torus = geo.Torus(
        center=(0, 0, 0), axis="z", major_radius=major, minor_radius=minor, material="pec"
    )
    ymax = np.sqrt((major + minor) ** 2 - x * x)

    def thickness(y):
        radius = np.hypot(x, y)
        return 2 * np.sqrt(max(minor * minor - (radius - major) ** 2, 0))

    expected = 2 * quad(thickness, 0, ymax, epsabs=1e-16, epsrel=1e-9)[0]
    result = cross_section_polygons(torus._occ_shape(1), "x", x, deflection=DEFLECTION)
    perimeter = sum(np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1).sum() for p in result)
    assert abs(area(result) - expected) <= DEFLECTION * 0.1 * perimeter


@pytest.fixture
def imprinted_union():
    radius = 1e-3
    block = geo.Brick(
        origin=(0, -2 * radius, 0), size=(4 * radius, 4 * radius, radius), material="pec"
    )
    pipe = geo.Cylinder(
        origin=(3 * radius, 0, 0), axis="x", height=3 * radius, radius=radius, material="pec"
    )
    return block + pipe, radius


@pytest.mark.parametrize("cached", [False, True])
def test_internal_imprint_preserves_face_membership(imprinted_union, cached):
    body, _ = imprinted_union
    shape = body._occ_shape(1)
    assert BRepCheck_Analyzer(shape).IsValid()
    prepared = _PreparedGeometry(shape)
    faces = []
    for index, face in enumerate(prepared.slabs.faces):
        explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while explorer.More():
            if explorer.Current().Orientation() == TopAbs_INTERNAL:
                faces.append(index)
                break
            explorer.Next()
    assert faces, "The tangent union must retain an embedded CAD edge"
    for index in faces:
        face = prepared.slabs.faces[index]
        u0, u1, v0, v1 = prepared.bounds[index]
        u = u0 + 0.875 * (u1 - u0)
        borders = prepared.borders[index] if cached else None
        assert not _strict_inside(face, u, v0 - 0.1 * (v1 - v0), 1e-12, borders)
        for fraction in (0.25, 0.75):
            assert _strict_inside(face, u, v0 + fraction * (v1 - v0), 1e-12, borders)


@pytest.mark.parametrize("angle", [0, 45, 67.5])
def test_tangent_union_section_matches_independent_area(imprinted_union, angle):
    body, radius = imprinted_union
    shape = body.rotated(axis="z", angle_deg=angle)._occ_shape(1)
    assert BRepCheck_Analyzer(shape).IsValid()
    cosine = np.cos(np.deg2rad(angle))
    tangent = np.tan(np.deg2rad(angle))

    def material_height(y):
        x = 3.5 * radius + tangent * y
        in_block = 0 < x < 4 * radius
        in_pipe = 3 * radius < x < 6 * radius and abs(y) < radius
        if in_pipe:
            half = np.sqrt(max(radius**2 - y**2, 0))
            return radius + half if in_block else 2 * half
        return radius if in_block else 0

    points = [-radius, 0, radius]
    if tangent:
        points.extend((x - 3.5 * radius) / tangent for x in (0, 3 * radius, 4 * radius, 6 * radius))
    expected = (
        quad(
            material_height,
            -2 * radius,
            2 * radius,
            points=sorted(p for p in points if -2 * radius < p < 2 * radius),
            epsabs=1e-16,
            epsrel=1e-10,
        )[0]
        / cosine
    )
    polygons, _, _ = _SurfaceSectionEngine(shape).section(0, 3.5 * radius * cosine, budget=1e-8)
    assert sum(polygon_area(p - p[0]) for p in polygons) == pytest.approx(expected, rel=5e-6)


@pytest.mark.parametrize("cached", [False, True])
def test_vertex_tolerance_gap_uses_an_unambiguous_ray(cached):
    face = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(), gp_Dir(0, 0, 1)), 0.0, 1.0, 0.0, 1.0).Face()
    gap = 1e-10
    explorer = TopExp_Explorer(face, TopAbs_EDGE)
    shifted = False
    while explorer.More():
        edge = topods.Edge(explorer.Current())
        explorer.Next()
        pcurve, first, last = BRep_Tool.CurveOnSurface(edge, face)
        midpoint = pcurve.Value((first + last) / 2)
        if abs(midpoint.Y() - 1) < 1e-12:
            assert gap < BRep_Tool.Tolerance(edge)
            measured = Geom2d_Curve.DownCast(pcurve.Copy())
            measured.Translate(gp_Vec2d(gap, 0))
            BRep_Builder().UpdateEdge(edge, measured, face, BRep_Tool.Tolerance(edge))
            shifted = True
            break
    assert shifted and BRepCheck_Analyzer(face).IsValid()
    borders = _PreparedGeometry(face).borders[0] if cached else None
    assert _strict_inside(face, gap / 2, 0.5, 1e-12, borders)
    assert not _strict_inside(face, gap / 2, -0.1, 1e-12, borders)
    assert not _strict_inside(face, gap / 2, 1.1, 1e-12, borders)


def test_dense_distinct_trace_samples_keep_their_parameterization():
    poles = TColgp_Array1OfPnt(1, 9)
    for index in range(1, 9):
        poles.SetValue(index, gp_Pnt(0, 0, 0))
    poles.SetValue(9, gp_Pnt(0, 1, 0))
    curve = Geom_BezierCurve(poles)
    surface = Geom_Plane(gp_Pln(gp_Pnt(), gp_Dir(0, 0, 1)))
    tolerance = 1e-6
    refined, pcurve = _refine_trace(curve, surface, None, 0, 1, 0, 0, tolerance, 4096 * tolerance)
    for parameter in np.linspace(0, 1, 65):
        point = refined.Value(float(parameter))
        uv = pcurve.Value(float(parameter))
        on_surface = surface.Value(uv.X(), uv.Y())
        assert point.Distance(on_surface) <= tolerance
        assert abs(point.X()) <= tolerance
        assert abs(point.Z()) <= tolerance
    assert refined.Value(0).Distance(gp_Pnt(0, 0, 0)) <= tolerance
    assert refined.Value(1).Distance(gp_Pnt(0, 1, 0)) <= tolerance


@pytest.mark.parametrize("height", [-0.001 + 4e-6, 0, 0.001 - 4e-6])
def test_swept_torus_sector_retains_wrapped_minor_region(height):
    radius, major, angle = 0.001, 0.02, 22.5
    pipe = geo.Cylinder(axis="x", origin=(0, 0, 0), height=0.003, radius=radius, material="pec")
    face = pipe.face(near=(0.003, 0, 0), normal=(1, 0, 0))
    path = geo.Path.from_face(face, up="z").turn_right(radius=major, angle_deg=angle).curve()
    shape = face.swept(path)._occ_shape(1)
    assert BRepCheck_Analyzer(shape).IsValid()
    polygons, _, _ = _SurfaceSectionEngine(shape).section(2, height, budget=1e-8)
    expected = 2 * np.deg2rad(angle) * major * np.sqrt(radius**2 - height**2)
    assert sum(polygon_area(p - p[0]) for p in polygons) == pytest.approx(expected, rel=5e-6)


@pytest.mark.parametrize("near_end", [False, True])
def test_swept_torus_sector_preserves_partial_major_domain(near_end):
    radius, major, angle = 0.001, 0.02, np.deg2rad(22.5)
    start = 0.003
    pipe = geo.Cylinder(axis="x", origin=(0, 0, 0), height=start, radius=radius, material="pec")
    face = pipe.face(near=(start, 0, 0), normal=(1, 0, 0))
    path = geo.Path.from_face(face, up="z").turn_right(radius=major, angle_deg=22.5).curve()
    shape = face.swept(path)._occ_shape(1)
    offset = (major + radius) * np.sin(angle) - 4e-6 if near_end else major * np.sin(angle) * 0.8

    def width(z):
        minor = np.sqrt(max(0, radius**2 - z**2))
        outer = np.sqrt(max(0, (major + minor) ** 2 - offset**2))
        inner = np.sqrt(max(0, (major - minor) ** 2 - offset**2))
        return max(0, outer - max(inner, offset / np.tan(angle)))

    expected = 2 * quad(width, 0, radius, epsabs=1e-14, epsrel=1e-9, limit=200)[0]
    polygons, _, _ = _SurfaceSectionEngine(shape).section(0, start + offset, budget=1e-8)
    assert area(polygons) == pytest.approx(expected, rel=5e-5)


def test_fillet_measurement_precision_retry_preserves_exact_cut():
    radius, fillet, height = 0.059, 0.0005, 0.0115
    body = geo.Cylinder(axis="z", height=height, radius=radius, inner_radius=0.046, material="pec")
    body = body.filleted(near=(0, radius, 0), radius=fillet).rotated(axis="z", angle_deg=22.5)
    shape = body._occ_shape(1)
    position = 0.05431825928689609
    half_width = np.sqrt(radius**2 - position**2)
    inner_fillet = radius - fillet

    def material_height(y):
        rho = np.hypot(position, y)
        lower = (
            0
            if rho <= inner_fillet
            else fillet - np.sqrt(max(fillet**2 - (rho - inner_fillet) ** 2, 0))
        )
        return height - lower

    expected = 2 * quad(material_height, 0, half_width, epsabs=1e-14, epsrel=1e-10)[0]
    router = _SurfaceRouter(shape, 1, 4e-6)
    polygons = router.section(0, position)
    assert polygons is not None
    assert sum(polygon_area(p - p[0]) for p in polygons) == pytest.approx(expected, rel=1e-4)
    repeat = router.section(0, position)
    assert sum(polygon_area(p - p[0]) for p in repeat) == pytest.approx(expected, rel=1e-4)


def test_refined_parts_retain_support_curve_cycle():
    circle = Geom_Circle(gp_Ax2(gp_Pnt(), gp_Dir(0, 0, 1)), 1.0)
    intervals = [(0.0, np.pi), (np.pi, 2 * np.pi)]
    records = []
    for index, (lo, hi) in enumerate(intervals):
        refined_part = Geom_TrimmedCurve(circle, lo, hi)
        assert not refined_part.IsClosed()
        records.append(
            {
                "curve": refined_part,
                "params": [lo, hi],
                "face": 0,
                "line": 1,
                "edge_tags": [[], [(0, 0.0)]] if index == 0 else [[(0, 0.0)], []],
                "direction": 1.0,
                "axis": 2,
                "support_closed": True,
                "support_range": (0.0, 2 * np.pi),
            }
        )
    partners, _ = _endpoint_partners(records)
    segments = [_tessellate(record["curve"], *record["params"], 1e-5) for record in records]
    for endpoint, partner in enumerate(partners):
        if partner > endpoint:
            shared = segments[endpoint // 2][0 if endpoint % 2 == 0 else -1].copy()
            segments[partner // 2][0 if partner % 2 == 0 else -1] = shared
    polygons = _walk_contours(segments, partners, records, 2)
    assert len(polygons) == 1
    assert polygon_area(polygons[0] - polygons[0][0]) == pytest.approx(np.pi, rel=1e-5)
