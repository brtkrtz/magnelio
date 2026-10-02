"""Bounded curve projection and its public result contract."""

import math
import re
from pathlib import Path

import pytest

from magnelio import geo

pytest.importorskip("OCC.Core.BRepProj")


def _holed_target():
    outer = geo.Curve.polyline([(-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0), (-2, -2, 0)])
    return geo.Profile.from_wires(outer, holes=[geo.Curve.circle((0, 0, 0), 0.5)])


def test_parallel_projection_uses_forward_rays_and_preserves_exact_line():
    target = geo.Profile.rectangle((0, 0, 0), (2, 2))
    source = geo.Curve.line((-0.5, 0, 1), (0.5, 0, 1))
    (result,) = source.projected_onto(target, direction=(0, 0, -1))
    assert result is not source
    assert isinstance(result, geo.Curve)
    assert result.length == pytest.approx(1)
    assert result.bounding_box()[0] == pytest.approx((-0.5, 0, 0))
    assert result.bounding_box()[1] == pytest.approx((0.5, 0, 0))
    with pytest.raises(ValueError, match="no forward hit"):
        source.projected_onto(target, direction=(0, 0, 1))
    assert source.projected_onto(target, direction=(0, 0, 1), clip=True) == ()


def test_perspective_source_doubles_circle_radius_and_retains_closure():
    source = geo.Curve.circle((0, 0, 1), 0.5)
    target = geo.Profile.rectangle((0, 0, 0), (4, 4))
    (projected,) = source.projected_onto(target, perspective_source=(0, 0, 2))
    assert projected.is_closed
    assert projected.length == pytest.approx(2 * math.pi, rel=1e-9)
    with pytest.raises(ValueError, match="no forward hit"):
        source.projected_onto(target, perspective_source=(0, 0, -2))


def test_hole_and_outer_boundary_require_explicit_clipping():
    target = _holed_target()
    source = geo.Curve.line((-1, 0, 1), (1, 0, 1))
    with pytest.raises(ValueError, match="clip=True"):
        source.projected_onto(target, direction=(0, 0, -1))
    pieces = source.projected_onto(target, direction=(0, 0, -1), clip=True)
    assert len(pieces) == 2
    assert sorted(piece.length for piece in pieces) == pytest.approx([0.5, 0.5])
    assert all(piece is not source for piece in pieces)
    outer = geo.Curve.line((-3, 1, 1), (3, 1, 1))
    with pytest.raises(ValueError, match="clip=True"):
        outer.projected_onto(target, direction=(0, 0, -1))
    (clipped,) = outer.projected_onto(target, direction=(0, 0, -1), clip=True)
    assert clipped.length == pytest.approx(4)


def test_narrow_hole_between_ray_samples_still_requires_clipping():
    outer = geo.Curve.polyline([(-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0), (-2, -2, 0)])
    target = geo.Profile.from_wires(outer, holes=[geo.Curve.circle((0.005, 0, 0), 0.001)])
    source = geo.Curve.line((-1, 0, 1), (1, 0, 1))
    with pytest.raises(ValueError, match="clip=True"):
        source.projected_onto(target, direction=(0, 0, -1))
    pieces = source.projected_onto(target, direction=(0, 0, -1), clip=True)
    assert sorted(piece.length for piece in pieces) == pytest.approx([0.994, 1.004])


def test_narrow_hole_between_closest_samples_is_not_crossed():
    outer = geo.Curve.polyline([(-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0), (-2, -2, 0)])
    target = geo.Profile.from_wires(outer, holes=[geo.Curve.circle((0.005, 0, 0), 0.001)])
    source = geo.Curve.line((-1, 0, 1), (1, 0, 1))
    with pytest.raises(ValueError, match="discontinuous or singular"):
        source.projected_onto(target, closest=True)


def test_closest_trace_bends_along_short_arc_of_narrow_hole():
    outer = geo.Curve.polyline([(-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0), (-2, -2, 0)])
    radius, offset = 0.001, 0.0008
    target = geo.Profile.from_wires(outer, holes=[geo.Curve.circle((0.005, 0, 0), radius)])
    source = geo.Curve.line((-1, offset, 1), (1, offset, 1))
    (trace,) = source.projected_onto(target, closest=True)
    expected = 2 - 2 * math.sqrt(radius**2 - offset**2) + 2 * radius * math.acos(offset / radius)
    assert trace.length == pytest.approx(expected, abs=5e-8)


def test_small_target_between_ray_samples_is_retained_when_clipped():
    target = geo.Profile.rectangle((0.005, 0, 0), (0.002, 2))
    source = geo.Curve.line((-1, 0, 1), (1, 0, 1))
    with pytest.raises(ValueError, match="clip=True"):
        source.projected_onto(target, direction=(0, 0, -1))
    (piece,) = source.projected_onto(target, direction=(0, 0, -1), clip=True)
    assert piece.length == pytest.approx(0.002)


def test_first_forward_branch_is_default_and_all_hits_is_explicit():
    housing = geo.Cylinder(radius=1, height=2)
    wall = housing.face(near=(1, 0, 1))
    source = geo.Curve.line((-2, -0.3, 1), (-2, 0.3, 1))
    (front,) = source.projected_onto(wall, direction="x")
    assert front.bounding_box()[1][0] < 0
    both = source.projected_onto(wall, direction="x", all_hits=True)
    assert len(both) == 2
    assert sorted(
        (c.bounding_box()[0][0] + c.bounding_box()[1][0]) / 2 for c in both
    ) == pytest.approx([-0.9769696, 0.9769696], rel=1e-6)
    assert wall.owner is housing


def test_first_hit_switches_when_source_crosses_front_wall():
    wall = geo.Cylinder(radius=1, height=2).face(near=(1, 0, 1))
    source = geo.Curve.line((-2, -0.3, 1), (0, 0.3, 1))
    parts = source.projected_onto(wall, direction="x")
    assert len(parts) == 2
    assert parts[0].bounding_box()[1][0] < 0
    assert parts[1].bounding_box()[0][0] > 0
    all_branches = source.projected_onto(wall, direction="x", all_hits=True)
    assert len(all_branches) == 3
    beyond = geo.Curve.line((-2, -0.3, 1), (2, 0.3, 1))
    with pytest.raises(ValueError, match="clip=True"):
        beyond.projected_onto(wall, direction="x")
    assert len(beyond.projected_onto(wall, direction="x", clip=True)) == 2


def test_closest_point_uses_bounded_face_and_its_edge():
    target = geo.Profile.rectangle((0, 0, 0), (2, 2))
    source = geo.Curve.line((-2, 2, 1), (2, 2, 1))
    (edge_trace,) = source.projected_onto(target, closest=True)
    assert edge_trace.length == pytest.approx(2)
    assert edge_trace.bounding_box()[0] == pytest.approx((-1, 1, 0))
    assert edge_trace.bounding_box()[1] == pytest.approx((1, 1, 0))


def test_closest_point_on_cylinder_is_on_curved_wall_and_crosses_seam():
    wall = geo.Cylinder(radius=1, height=2).face(near=(1, 0, 1))
    source = geo.Curve.circle((0, 0, 1), 2)
    (trace,) = source.projected_onto(wall, closest=True)
    assert trace.is_closed
    assert trace.length == pytest.approx(2 * math.pi, rel=1e-7)
    assert trace.bounding_box()[0][:2] == pytest.approx((-1, -1), abs=1e-6)
    assert trace.bounding_box()[1][:2] == pytest.approx((1, 1), abs=1e-6)


def test_curved_standalone_sheet_accepts_ray_projection():
    sheet = geo.Surface.parametric(
        lambda u, v: (u, v, 0.1 * u * u),
        u=(-1, 1),
        v=(-1, 1),
        samples=(9, 9),
    )
    source = geo.Curve.line((-0.5, 0, 1), (0.5, 0, 1))
    (trace,) = source.projected_onto(sheet, direction=(0, 0, -1))
    assert trace.length > source.length
    assert trace.bounding_box()[1][2] == pytest.approx(0.025, abs=1e-6)


def test_closest_point_discontinuity_at_hole_is_reported():
    with pytest.raises(ValueError, match="discontinuous or singular"):
        geo.Curve.line((-1, 0, 1), (1, 0, 1)).projected_onto(_holed_target(), closest=True)


def test_closest_point_follows_exact_hole_rim():
    source = geo.Curve.line((0.1, -0.2, 1), (0.1, 0.2, 1))
    (rim,) = source.projected_onto(_holed_target(), closest=True)
    assert rim.length == pytest.approx(math.atan(2), rel=1e-10)


def test_tangent_trace_and_no_hit_are_distinct():
    wall = geo.Cylinder(radius=1, height=2).face(near=(1, 0, 1))
    tangent = geo.Curve.line((1, -2, 0.5), (1, -2, 1.5))
    (trace,) = tangent.projected_onto(wall, direction="y")
    assert trace.length == pytest.approx(1)
    missed = geo.Curve.line((1.00001, -2, 0.5), (1.00001, -2, 1.5))
    with pytest.raises(ValueError, match="no projection"):
        missed.projected_onto(wall, direction="y")


def test_curve_on_target_with_tangent_rays_uses_zero_distance_hit():
    target = geo.Profile.rectangle((0, 0, 0), (2, 2))
    on_face = geo.Curve.line((-0.5, 0, 0), (0.5, 0, 0))
    (trace,) = on_face.projected_onto(target, direction="x")
    assert trace.length == pytest.approx(on_face.length)
    with pytest.raises(ValueError, match="infinitely many"):
        on_face.projected_onto(target, direction="x", all_hits=True)
    partial = geo.Curve.line((-2, 0, 0), (2, 0, 0))
    with pytest.raises(ValueError, match="clip=True"):
        partial.projected_onto(target, direction="x")
    (clipped,) = partial.projected_onto(target, direction="x", clip=True)
    assert clipped.length == pytest.approx(2)


def test_closest_point_at_target_pole_reports_singularity():
    sphere_face = geo.Sphere(radius=1).face(near=(0, 0, 1))
    crossing = geo.Curve.line((-0.2, 0, 2), (0.2, 0, 2))
    with pytest.raises(ValueError, match="singular"):
        crossing.projected_onto(sphere_face, closest=True)


@pytest.mark.parametrize("size", [1e-9, 1, 1e3])
def test_scale_covariance(size):
    target = geo.Profile.rectangle((0, 0, 0), (2 * size, 2 * size))
    source = geo.Curve.line((-0.5 * size, 0, size), (0.5 * size, 0, size))
    (trace,) = source.projected_onto(target, direction=(0, 0, -1))
    assert trace.length / size == pytest.approx(1)
    (nearest,) = source.projected_onto(target, closest=True)
    assert nearest.length / size == pytest.approx(1)


def test_rigid_covariance_of_target_source_and_world_direction():
    target = geo.Profile.rectangle((0, 0, 0), (2, 2))
    source = geo.Curve.line((-0.5, 0, 1), (0.5, 0, 1))
    placed_target = target.rotated("y", 30).translated((1, 2, 3))
    placed_source = source.rotated("y", 30).translated((1, 2, 3))
    (trace,) = placed_source.projected_onto(placed_target, direction=(-0.5, 0, -math.sqrt(3) / 2))
    assert trace.length == pytest.approx(1)


def test_project_store_replays_body_built_from_projected_curve(tmp_path):
    import numpy as np

    import magnelio as mio
    from magnelio.io.project import ProjectStore
    from magnelio.mesh import GridLines

    target = geo.Profile.rectangle((0, 0, 0), (2, 2))
    source = geo.Curve.line((-0.5, 0, 1), (0.5, 0, 1))
    (trace,) = source.projected_onto(target, direction=(0, 0, -1))
    conductor = geo.Profile.circle((-0.5, 0, 0), 0.05, normal="x").swept(trace, material="pec")
    lines = np.linspace(-1, 1, 3)
    grid = GridLines(x=lines, y=lines, z=lines)
    ProjectStore.create(tmp_path / "projected", mio.Mesh.from_grid(grid), geometry=[conductor])
    (restored,) = mio.open_project(tmp_path / "projected").geometry
    assert restored.volume() == pytest.approx(conductor.volume())


def test_invalid_policy_and_target_fail_at_call():
    source = geo.Curve.line((0, 0, 1), (1, 0, 1))
    target = geo.Profile.rectangle((0, 0, 0), (2, 2))
    with pytest.raises(ValueError, match="exactly one"):
        source.projected_onto(target)
    with pytest.raises(ValueError, match="exactly one"):
        source.projected_onto(target, direction=(0, 0, -1), closest=True)
    with pytest.raises(TypeError, match="Sheet or FaceRef"):
        source.projected_onto(geo.Cylinder(), direction=(0, 0, -1))
    with pytest.raises(ValueError, match="all_hits"):
        source.projected_onto(target, closest=True, all_hits=True)
    with pytest.raises(ValueError, match="clip applies"):
        source.projected_onto(target, closest=True, clip=True)
    with pytest.raises(ValueError, match="tolerance applies"):
        source.projected_onto(target, direction=(0, 0, -1), tolerance=1e-6)
    with pytest.raises(ValueError, match="positive"):
        source.projected_onto(target, closest=True, tolerance=0)


def test_methods_recipe_and_public_tutorial_execute():
    import matplotlib

    matplotlib.use("Agg")
    root = Path(__file__).resolve().parents[2]
    source = (root / "docs/methods/geometry.md").read_text()
    prose = source.split("## Project curves onto bounded faces", 1)[1].split("\n## ", 1)[0]
    namespace = {}
    for block in re.findall(r"```python\n(.*?)```", prose, re.S):
        exec(block, namespace)
    assert namespace["wire"].curve.length > namespace["sketch"].length
