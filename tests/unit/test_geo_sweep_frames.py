"""Explicit sweep transport, independent cap/volume gates (DD-275)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from magnelio import geo
from magnelio.geo._topology_store import from_recipe, to_recipe

pytest.importorskip("OCC.Core.BRepOffsetAPI")

FRAMES = ("corrected_frenet", "frenet", "fixed", "fixed_binormal")


def options(frame):
    return {"frame": frame, **({"binormal": "z"} if frame == "fixed_binormal" else {})}


@pytest.mark.parametrize("frame", FRAMES)
@pytest.mark.parametrize("scale", [1e-9, 1e-3, 1, 1e3])
def test_straight_annular_sweep_preserves_material_and_holes(frame, scale):
    body = geo.Cylinder(
        axis="y", radius=2 * scale, inner_radius=scale, height=scale, material="pec"
    ).tag_face("end", normal="y")
    face = body.face("end")
    spine = geo.Curve.line(face.centroid, (0, 4 * scale, 0))
    result = face.swept(spine, **options(frame))
    assert result.material is body.material
    assert result.volume() == pytest.approx(9 * math.pi * scale**3, rel=1e-8)
    assert len(result.face(normal="y").edges) == 2
    assert face.centroid == pytest.approx((0, scale, 0))
    assert body.face("end").area == pytest.approx(3 * math.pi * scale**2)
    assert face.swept(spine, material="air", **options(frame)).material.name == "air"


@pytest.mark.parametrize("frame", FRAMES)
def test_circular_asymmetric_sweep_roll_and_independent_volume(frame):
    radius, angle = 8, math.pi / 4
    profile = geo.Profile.rectangle((0, 0, 0), (1, 0.3), normal="y", x_direction="x")
    profile = profile.rotated("y", 31).translated((radius, 0, 0))
    spine = geo.Curve.arc(
        (radius, 0, 0),
        (radius * math.cos(angle / 2), radius * math.sin(angle / 2), 0),
        (radius * math.cos(angle), radius * math.sin(angle), 0),
    )
    result = profile.swept(spine, **options(frame))
    expected_volume = profile.area * (
        radius * math.sin(angle) if frame == "fixed" else spine.length
    )
    assert result.volume() == pytest.approx(expected_volume, rel=1e-7)
    rotation = np.eye(3) if frame == "fixed" else np.asarray(geo.Rotation("z", 45).matrix)[:3, :3]
    normal = rotation @ np.array((0, 1, 0))
    cap = result.face(normal=normal)
    start = profile.extruded((0, -1, 0)).face(normal="y")
    end = np.array((radius * math.cos(angle), radius * math.sin(angle), 0))
    expected = (np.asarray([v.point for v in start.vertices]) - (radius, 0, 0)) @ rotation.T + end
    for vertex in cap.vertices:
        assert np.min(np.linalg.norm(expected - vertex.point, axis=1)) < 1e-7


@pytest.mark.parametrize("frame", FRAMES)
def test_offset_holes_share_one_spine_station(frame):
    profile = geo.Profile.from_wires(
        geo.Curve.circle((8, 0, 0), 2, normal="y"),
        [
            geo.Curve.circle((8.6, 0, 0.4), 0.2, normal="y"),
            geo.Curve.circle((7.4, 0, -0.4), 0.3, normal="y"),
        ],
    )
    angle = math.pi / 4
    spine = geo.Curve.arc(
        (8, 0, 0),
        (8 * math.cos(angle / 2), 8 * math.sin(angle / 2), 0),
        (8 * math.cos(angle), 8 * math.sin(angle), 0),
    )
    result = profile.swept(spine, **options(frame))
    normal = (0, 1, 0) if frame == "fixed" else (-math.sin(angle), math.cos(angle), 0)
    assert len(result.face(normal=normal).edges) == 3
    assert result.face(normal=normal).area == pytest.approx(profile.area, rel=1e-8)


@pytest.mark.parametrize("frame", FRAMES)
def test_named_result_project_recipe_preserves_sweep_transport(frame):
    profile = geo.Profile.rectangle((0, 0, 0), (1, 0.3), normal="y")
    curve = geo.Curve.arc(
        (8, 0, 0),
        (8 * math.cos(math.pi / 8), 8 * math.sin(math.pi / 8), 0),
        (8 / 2**0.5, 8 / 2**0.5, 0),
    )
    result = profile.swept(curve, material="pec", **options(frame)).tag_face(
        "start", normal=(0, -1, 0)
    )
    restored = from_recipe(json.loads(json.dumps(to_recipe(result))))
    assert restored.volume() == pytest.approx(result.volume(), rel=1e-9)
    assert restored.face("start").area == pytest.approx(profile.area)
    assert np.asarray(restored.bounding_box()) == pytest.approx(np.asarray(result.bounding_box()))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"frame": "unknown"}, "frame"),
        ({"frame": "fixed_binormal"}, "requires binormal"),
        ({"binormal": "z"}, "requires frame"),
        ({"frame": "fixed_binormal", "binormal": (0, 0, 0)}, "zero vector"),
    ],
)
def test_mode_arguments_fail_at_call(kwargs, message):
    profile = geo.Profile.circle((0, 0, 0), 1)
    with pytest.raises(ValueError, match=message):
        profile.swept(geo.Curve.line((0, 0, 0), (0, 0, 1)), **kwargs)


def test_parallel_binormal_is_rejected_before_kernel_build():
    with pytest.raises(ValueError, match="parallel"):
        geo.Profile.circle((0, 0, 0), 1).swept(
            geo.Curve.line((0, 0, 0), (0, 0, 1)), frame="fixed_binormal", binormal="z"
        )


def test_resolvable_near_parallel_binormal_is_not_rejected_by_a_dot_roundoff_test():
    result = geo.Profile.rectangle((0, 0, 0), (1, 2)).swept(
        geo.Curve.line((0, 0, 0), (0, 0, 3)),
        frame="fixed_binormal",
        binormal=(1e-8, 0, 1),
    )
    assert result.volume() == pytest.approx(6)


@pytest.mark.parametrize("magnitude", [1e-300, 1e300])
def test_binormal_magnitude_does_not_change_direction(magnitude):
    profile = geo.Profile.rectangle((0, 0, 0), (1, 2))
    curve = geo.Curve.line((0, 0, 0), (0, 0, 3))
    result = profile.swept(curve, frame="fixed_binormal", binormal=(magnitude, 0, 0))
    assert result.volume() == pytest.approx(6)


@pytest.mark.parametrize("frame", FRAMES)
def test_frame_roll_is_covariant_under_spatial_placement_and_reflection(frame):
    placement = geo.Rotation((1, 2, 3), 37) @ geo.Mirror("x")
    profile = geo.Profile.rectangle((0, 0, 0), (1, 0.3), normal="y", x_direction="x")
    curve = geo.Curve.arc(
        (8, 0, 0),
        (8 * math.cos(math.pi / 8), 8 * math.sin(math.pi / 8), 0),
        (8 / 2**0.5, 8 / 2**0.5, 0),
    )
    first = placement @ profile.swept(curve, **options(frame))
    direction = np.asarray(placement.matrix)[:3, :3] @ (0, 0, 1)
    settings = {"frame": frame, **({"binormal": direction} if frame == "fixed_binormal" else {})}
    second = (placement @ profile).swept(placement @ curve, **settings)
    assert first.volume() == pytest.approx(second.volume(), rel=1e-8)
    assert np.asarray(first.bounding_box()) == pytest.approx(
        np.asarray(second.bounding_box()), abs=1e-7
    )


def test_spatial_fixed_and_binormal_caps_follow_their_declared_world_constraints():
    from OCC.Core.BRepAdaptor import BRepAdaptor_CompCurve
    from OCC.Core.gp import gp_Pnt, gp_Vec

    curve = geo.Curve.spline([(0, 0, 0), (0, 0, 1), (1, 0, 2), (1, 1, 3), (0, 2, 4)])
    adaptor = BRepAdaptor_CompCurve(curve._occ_shape())
    tangent = gp_Vec()
    adaptor.D1(adaptor.FirstParameter(), gp_Pnt(), tangent)
    normal = np.array(tangent.XYZ().Coord()) / tangent.Magnitude()
    profile = geo.Profile.rectangle((0, 0, 0), (0.2, 0.1))
    fixed = profile.swept(curve, frame="fixed")
    assert fixed.volume() == pytest.approx(profile.area * np.dot(normal, (0, 2, 4)), rel=1e-7)
    assert fixed.face(near=(0, 2, 4), surface_type="plane").normal == pytest.approx(normal)
    binormal = profile.swept(curve, frame="fixed_binormal", binormal="x")
    cap = binormal.face(near=(0, 2, 4), surface_type="plane")
    assert cap.normal[0] == pytest.approx(normal[0], abs=1e-8)
    frenet = profile.swept(curve, frame="frenet")
    assert frenet.volume() == pytest.approx(profile.area * curve.length, rel=1e-4)


def test_methods_recipe_executes():
    source = Path(__file__).resolve().parents[2] / "docs/methods/geometry.md"
    section = source.read_text().split("## Sweep orientation modes", 1)[1].split("\n## ", 1)[0]
    namespace = {}
    for block in section.split("```python\n")[1:]:
        exec(block.split("```", 1)[0], namespace)
    assert namespace["kept_parallel"].volume() == pytest.approx(namespace["expected_volume"])
