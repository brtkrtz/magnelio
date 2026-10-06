"""Independent near-tangency and bounded-face material-area references."""

from __future__ import annotations

import numpy as np
import pytest
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepTools import breptools
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from scipy.integrate import quad

from magnelio import geo
from magnelio.geo._occ_backend import _PlanarSectionEngine, cross_section_polygons
from magnelio.geo._polygon_clip import polygon_area
from magnelio.geo._surface_sections import _SurfaceRouter

R, BORE, HEIGHT, DEFLECTION = 2.3e-3, 0.7e-3, 10e-3, 2.5e-6


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


def test_disabling_the_guard_reproduces_the_original_defect(monkeypatch):
    body = drilled("y")
    occ = body._occ_shape(1)
    monkeypatch.setenv("MAGNELIO_SURFACE_SECTIONS", "0")
    old = _PlanarSectionEngine(occ, deflection=DEFLECTION).section(1, R - 1e-7)
    assert area(old) < 0.8 * exact_drilled(1e-7)
    assert cross_section_polygons(occ, "y", R - 1e-7, deflection=DEFLECTION) == []
    monkeypatch.delenv("MAGNELIO_SURFACE_SECTIONS")
    corrected = _PlanarSectionEngine(occ, deflection=DEFLECTION).section(1, R - 1e-7)
    assert area(corrected) == pytest.approx(exact_drilled(1e-7), rel=1e-4)


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
