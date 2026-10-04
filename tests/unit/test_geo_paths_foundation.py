"""Relative routing, transported poses and coax continuations (DD-275, WP5)."""

import math
import re
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from magnelio import geo

pytest.importorskip("OCC.Core.BRepAdaptor")


def assert_pose(path, point, tangent, up):
    assert path.current == pytest.approx(point, abs=1e-10)
    assert path.tangent == pytest.approx(tangent, abs=1e-9)
    assert path.up == pytest.approx(up, abs=1e-9)
    assert np.dot(path.tangent, path.up) == pytest.approx(0, abs=1e-12)


def test_pose_projects_up_and_branches_immutably():
    start = geo.Path.from_pose([1, 2, 3], (2, 0, 0), (1, 0, 2))
    assert_pose(start, (1, 2, 3), (1, 0, 0), (0, 0, 1))
    assert_pose(start.forward(2), (3, 2, 3), (1, 0, 0), (0, 0, 1))
    assert_pose(start.turn_left(radius=2, angle_deg=90), (3, 4, 3), (0, 1, 0), (0, 0, 1))
    assert start.current == (1, 2, 3)
    with pytest.raises(FrozenInstanceError):
        start.start = (0, 0, 0)
    with pytest.raises(FrozenInstanceError):
        start._up_direction = (0, 1, 0)


@pytest.mark.parametrize("scale", [1e-9, 1e-3, 1, 1e3])
@pytest.mark.parametrize("angle", [22.5, 90, 180, 270])
@pytest.mark.parametrize("side,sign", [("turn_left", 1), ("turn_right", -1)])
def test_exact_relative_circles_at_model_scales(scale, angle, side, sign):
    path = getattr(geo.Path.from_pose((0, 0, 0), "x", "z"), side)(radius=scale, angle_deg=angle)
    theta = math.radians(angle)
    assert np.asarray(path.current) / scale == pytest.approx(
        (math.sin(theta), sign * (1 - math.cos(theta)), 0), abs=1e-12
    )
    assert path.tangent == pytest.approx((math.cos(theta), sign * math.sin(theta), 0), abs=1e-12)
    assert path.up == pytest.approx((0, 0, 1))
    assert path.curve().length == pytest.approx(scale * theta, rel=1e-9)


def test_spatial_turn_to_transports_roll_and_subsequent_left():
    start = geo.Path.from_pose((0, 0, 0), "x", "z")
    bent = start.turn_to("z", radius=2)
    assert_pose(bent, (2, 0, 2), (0, 0, 1), (-1, 0, 0))
    assert_pose(bent.turn_left(radius=1, angle_deg=90), (2, 1, 3), (0, 1, 0), (-1, 0, 0))
    assert start.turn_to((5, 0, 0), radius=2) is start
    with pytest.raises(ValueError, match="bend plane"):
        start.turn_to((-1, 0, 0), radius=2)


def test_oblique_turn_to_is_an_exact_tangent_circle():
    target = np.array((1, 2, 3)) / math.sqrt(14)
    start = geo.Path.from_pose((1, -2, 4), "x", (0, 1, 1))
    path = start.turn_to(target, radius=3)
    assert path.tangent == pytest.approx(target)
    assert path.curve().length == pytest.approx(3 * math.acos(target[0]), rel=1e-10)
    assert np.dot(path.up, target) == pytest.approx(0, abs=1e-12)


def test_plane_intersections_signed_offsets_and_noop():
    start = geo.Path.from_pose((1, 2, 3), (1, 1, 0), "z")
    end = start.straight_to_plane((2, 2, 0), 7 / math.sqrt(2))
    assert_pose(end, (3, 4, 3), (1 / math.sqrt(2), 1 / math.sqrt(2), 0), (0, 0, 1))
    assert start.straight_to_plane("z", 3) is start
    assert geo.Path.from_pose((0, 0, 0), (-1, 0, 0), "z").straight_to_plane("x", -2).current == (
        -2,
        0,
        0,
    )
    with pytest.raises(ValueError, match="parallel"):
        start.straight_to_plane("z", 4)
    with pytest.raises(ValueError, match="behind"):
        start.straight_to_plane("x", 0)


@pytest.mark.parametrize("mirror", [False, True])
def test_from_placed_named_face_and_exact_hollow_bend_to_plane(mirror):
    body = geo.Cylinder(axis="x", radius=0.2, inner_radius=0.1, height=1, material="pec")
    body = body.tagged_face("port", normal="x").rotated("z", 22.5)
    if mirror:
        body = body.mirrored("y")
    face = body.face("port")
    path = geo.Path.from_face(face, up="z")
    assert path.current == pytest.approx(face.centroid)
    assert path.tangent == pytest.approx(face.normal)
    bend = (
        path.turn_right(radius=2, angle_deg=22.5)
        if not mirror
        else path.turn_left(radius=2, angle_deg=22.5)
    )
    end = bend.straight_to_plane("x", 5)
    assert end.current[0] == pytest.approx(5)
    assert end.tangent == pytest.approx((1, 0, 0), abs=1e-12)
    spine = end.curve()
    solid = face.swept(spine)
    assert solid.volume() == pytest.approx(face.area * spine.length, rel=1e-9)
    cap = solid.face(normal="x")
    assert len(cap.edges) == 2
    assert cap.centroid == pytest.approx(end.current)
    assert solid.material is body.material
    assert body.face("port").centroid == pytest.approx(path.current)


def test_absolute_line_updates_pose_and_unposed_forward():
    start = geo.Path((0, 0, 0))
    assert start.tangent is None and start.up is None
    end = start.line_to((0, 3, 0)).forward(2)
    assert end.current == (0, 5, 0) and end.tangent == (0, 1, 0)
    assert end.up is None
    with pytest.raises(ValueError, match="pose"):
        end.turn_left(radius=1, angle_deg=90)
    posed = geo.Path.from_pose((0, 0, 0), "x", "z").line_to((0, 0, 2))
    assert_pose(posed, (0, 0, 2), (0, 0, 1), (-1, 0, 0))


def test_absolute_major_circle_transports_actual_traversal_and_roll():
    path = geo.Path.from_pose((1, 0, 0), "y", (1, 0, 1)).arc_to(
        (0, -1, 0), center=(0, 0, 0), normal="z"
    )
    d = 1 / math.sqrt(2)
    assert_pose(path, (0, -1, 0), (1, 0, 0), (0, -d, d))
    assert path.curve().length == pytest.approx(1.5 * math.pi)
    assert path.forward(2).current == pytest.approx((2, -1, 0))


@pytest.mark.parametrize("axes", [(2, 1), (1, 2)])
def test_absolute_ellipse_updates_tangent_and_up(axes):
    a, b = axes
    path = geo.Path.from_pose((a, 0, 0), "y", "z").ellipse_to(
        (0, b, 0), center=(0, 0, 0), semi_axes=axes, major_axis="x", normal="z"
    )
    assert_pose(path, (0, b, 0), (-1, 0, 0), (0, 0, 1))
    assert path.forward(1).current == pytest.approx((-1, b, 0))


def test_spatial_spline_pose_matches_cad_and_is_rotation_covariant():
    from OCC.Core.BRepAdaptor import BRepAdaptor_CompCurve
    from OCC.Core.gp import gp_Pnt, gp_Vec

    points = [(0, 0, 0), (1, 0.3, 0.2), (2, -0.2, 0.6), (3, 0.4, 1)]
    path = geo.Path.from_pose(points[0], "x", "z").spline_to(*points[1:])
    comp = BRepAdaptor_CompCurve(path.curve()._occ_shape())
    p, v = gp_Pnt(), gp_Vec()
    comp.D1(comp.LastParameter(), p, v)
    expected = np.array((v.X(), v.Y(), v.Z())) / v.Magnitude()
    assert path.tangent == pytest.approx(expected)
    assert np.dot(path.tangent, path.up) == pytest.approx(0, abs=1e-12)
    rotation = np.asarray(geo.Rotation((1, 2, 3), 57).matrix)[:3, :3]
    moved_points = [rotation @ p for p in points]
    moved = geo.Path.from_pose(
        moved_points[0], rotation @ (1, 0, 0), rotation @ (0, 0, 1)
    ).spline_to(*moved_points[1:])
    assert moved.tangent == pytest.approx(rotation @ path.tangent, abs=1e-9)
    assert moved.up == pytest.approx(rotation @ path.up, abs=1e-9)
    assert np.asarray(path.forward(1).current) == pytest.approx(np.asarray(path.current) + expected)


@pytest.mark.parametrize(
    "tangent,up", [("x", "x"), ((0, 0, 0), "z"), ("x", (0, 0, 0)), ("x", (0, math.nan, 1))]
)
def test_invalid_pose_directions_fail_eagerly(tangent, up):
    with pytest.raises(ValueError):
        geo.Path.from_pose((0, 0, 0), tangent, up)


@pytest.mark.parametrize("magnitude", [1e-300, 1e300])
def test_finite_direction_magnitudes_do_not_overflow_or_underflow(magnitude):
    path = geo.Path.from_pose((0, 0, 0), (magnitude, 0, 0), (magnitude, 0, magnitude))
    assert_pose(path, (0, 0, 0), (1, 0, 0), (0, 0, 1))
    bend = path.turn_to((0, magnitude, 0), radius=1)
    assert_pose(bend, (1, 1, 0), (0, 1, 0), (0, 0, 1))
    end = bend.straight_to_plane((0, magnitude, 0), 2)
    assert_pose(end, (1, 2, 0), (0, 1, 0), (0, 0, 1))


@pytest.mark.parametrize(
    "verb,args,kwargs",
    [
        ("forward", (0,), {}),
        ("forward", (-1,), {}),
        ("forward", (math.inf,), {}),
        ("turn_left", (), {"radius": 0, "angle_deg": 90}),
        ("turn_right", (), {"radius": 1, "angle_deg": 360}),
        ("turn_left", (), {"radius": 1, "angle_deg": -90}),
        ("turn_right", (), {"radius": 1, "angle_deg": math.nan}),
        ("turn_to", ((0, 0, 0),), {"radius": 1}),
        ("straight_to_plane", ("x", math.inf), {}),
    ],
)
def test_invalid_relative_arguments(verb, args, kwargs):
    path = geo.Path.from_pose((0, 0, 0), "x", "z")
    with pytest.raises(ValueError):
        getattr(path, verb)(*args, **kwargs)


def test_from_face_rejects_unowned_or_curved_inputs():
    with pytest.raises(TypeError, match="FaceRef"):
        geo.Path.from_face(geo.Profile.circle((0, 0, 0), 1), up="z")
    with pytest.raises(ValueError, match="planar"):
        geo.Path.from_face(geo.Sphere().face(surface_type="sphere"), up="z")
    with pytest.raises(ValueError, match="parallel"):
        geo.Path.from_face(geo.Brick().face(normal="z"), up="z")


def test_planar_spline_inflection_retains_up_and_composes_absolute_relative():
    path = geo.Path.from_pose((0, 0, 0), "x", "z").spline_to(
        (1, 1, 0), (2, 0, 0), (3, -1, 0), (4, 0, 0)
    )
    assert path.up == pytest.approx((0, 0, 1), abs=1e-12)
    straight = path.turn_to("x", radius=1).forward(2)
    end = straight.line_to((9, 0, 0))
    delta = np.array((9, 0, 0)) - straight.current
    assert_pose(end, (9, 0, 0), delta / np.linalg.norm(delta), (0, 0, 1))


def test_spatial_spline_roll_matches_independent_discrete_parallel_transport():
    from OCC.Core.BRepAdaptor import BRepAdaptor_CompCurve
    from OCC.Core.gp import gp_Pnt, gp_Vec

    path = geo.Path.from_pose((0, 0, 0), "x", "z").spline_to(
        (1, 0.7, 0.2), (2, -0.3, 1), (3, 0.4, 2), (4, 1, 1)
    )
    comp = BRepAdaptor_CompCurve(path.curve()._occ_shape())
    tangent = np.array((1.0, 0.0, 0.0))
    up = np.array((0.0, 0.0, 1.0))
    for q in np.linspace(comp.FirstParameter(), comp.LastParameter(), 8193):
        p, v = gp_Pnt(), gp_Vec()
        comp.D1(q, p, v)
        target = np.array((v.X(), v.Y(), v.Z())) / v.Magnitude()
        axis = np.cross(tangent, target)
        sine, cosine = np.linalg.norm(axis), np.dot(tangent, target)
        if sine > 1e-14:
            axis /= sine
            up = up * cosine + np.cross(axis, up) * sine + axis * np.dot(axis, up) * (1 - cosine)
        tangent = target
    assert np.asarray(path.up) == pytest.approx(up, abs=2e-7)


def test_pose_reversal_and_closed_route_preserve_branch():
    start = geo.Path.from_pose((0, 0, 0), "x", "z").forward(1)
    reversed_path = start.line_to((0, 0, 0))
    assert_pose(reversed_path, (0, 0, 0), (-1, 0, 0), (0, 0, 1))
    loop = geo.Path.from_pose((0, 0, 0), "x", "z").turn_left(radius=1, angle_deg=180)
    loop = loop.turn_left(radius=1, angle_deg=180)
    assert loop.closed().length == pytest.approx(2 * math.pi)
    assert start.current == (1, 0, 0)


def test_plane_roundoff_and_large_translation_do_not_add_degenerate_edges():
    start = geo.Path.from_pose((2, 4, -1), (1, 1, 1), (0, 0, 1))
    position = np.dot(np.array((1, 2, 3)) / math.sqrt(14), start.current)
    assert start.straight_to_plane((1, 2, 3), np.nextafter(position, math.inf)) is start
    origin = (1e5, 0, 0)
    end = geo.Path.from_pose(origin, "x", "z").straight_to_plane("x", origin[0] + 0.01)
    assert end.current[0] == origin[0] + 0.01
    assert end.curve().length == pytest.approx(0.01, rel=1e-8)


def test_routed_named_face_and_swept_body_round_trip_through_project(tmp_path):
    import magnelio as mio
    from magnelio.io.project import ProjectStore
    from magnelio.mesh import GridLines

    body = geo.Cylinder(axis="x", radius=2e-3, inner_radius=1e-3, height=6e-3, material="pec")
    body = body.tagged_face("port", normal="x").rotated("z", 22.5)
    route = geo.Path.from_face(body.face("port"), up="z").turn_right(radius=8e-3, angle_deg=22.5)
    route = route.straight_to_plane("x", 20e-3)
    extension = body.face("port").swept(route.curve()).tagged_face("outlet", normal="x")
    model = mio.GeometryModel()
    model.add(geo.Group(body, extension))
    grid = GridLines(
        x=np.linspace(0, 0.02, 5), y=np.linspace(-0.005, 0.01, 5), z=np.linspace(-0.005, 0.005, 5)
    )
    ProjectStore.create(tmp_path / "route", mio.Mesh.from_grid(grid), geometry=model)
    restored_body, restored_extension = mio.open_project(tmp_path / "route").geometry
    restored_route = geo.Path.from_face(restored_body.face("port"), up="z").turn_right(
        radius=8e-3, angle_deg=22.5
    )
    restored_route = restored_route.straight_to_plane("x", 20e-3)
    assert_pose(restored_route, route.current, route.tangent, route.up)
    assert restored_extension.volume() == pytest.approx(extension.volume(), rel=1e-9)
    assert len(restored_extension.face("outlet").edges) == 2
    assert restored_extension.face("outlet").centroid == pytest.approx(route.current)


def test_path_prose_recipes_execute():
    root = Path(__file__).resolve().parents[2]
    for source in ["docs/methods/geometry.md", "docs/migration-geometry.md"]:
        blocks = re.findall(r"```python\n(.*?)```", (root / source).read_text(), re.S)
        for block in blocks:
            if "Path.from_" in block:
                exec(compile(block, source, "exec"), {})
