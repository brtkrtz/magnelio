"""Independent geometry gates for constant sweep laws (DD-275)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from magnelio import geo
from magnelio.geo._topology_store import from_recipe, to_recipe

pytest.importorskip("OCC.Core.BRepOffsetAPI")

FRAMES = ("corrected_frenet", "frenet", "fixed", "fixed_binormal")


def settings(frame):
    return {"frame": frame, **({"binormal": "x"} if frame == "fixed_binormal" else {})}


def section_distance(body, point, normal, expected):
    """Cut the final BREP, independently of the section-fitting machinery."""
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
    from OCC.Core.gp import gp_Dir, gp_Pln, gp_Pnt

    cut = BRepAlgoAPI_Section(body._occ_shape(), gp_Pln(gp_Pnt(*point), gp_Dir(*normal)))
    cut.Build()
    assert cut.IsDone()
    for vertex in expected:
        distance = BRepExtrema_DistShapeShape(
            BRepBuilderAPI_MakeVertex(gp_Pnt(*vertex)).Vertex(), cut.Shape()
        )
        assert distance.IsDone()
        assert distance.Value() < 3e-6


def rectangle_points(z, angle_deg, offset=0):
    angle = math.radians(angle_deg)
    matrix = np.array(((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))))
    points = np.array(
        [(x, y) for x in (-1 - offset, 1 + offset) for y in (-0.5 - offset, 0.5 + offset)]
    )
    return np.column_stack((points @ matrix.T, np.full(4, z)))


@pytest.mark.parametrize("frame", FRAMES)
def test_zero_laws_preserve_existing_backend_exactly(frame):
    profile = geo.Profile.rectangle((0, 0, 0), (2, 1))
    spine = geo.Curve.line((0, 0, 0), (0, 0, 4))
    plain = profile.swept(spine, **settings(frame))
    explicit = profile.swept(spine, twist_deg=0, draft_deg=0, tolerance=1e-8, **settings(frame))
    assert explicit.volume() == plain.volume()
    assert np.asarray(explicit.bounding_box()) == pytest.approx(np.asarray(plain.bounding_box()))


@pytest.mark.parametrize("frame", FRAMES)
@pytest.mark.parametrize("twist", [-120, 90, 450])
def test_twist_is_total_right_hand_roll_and_preserves_straight_volume(frame, twist):
    profile = geo.Profile.rectangle((0, 0, 0), (2, 1))
    body = profile.swept(geo.Curve.line((0, 0, 0), (0, 0, 4)), twist_deg=twist, **settings(frame))
    assert body.volume() == pytest.approx(8, rel=2e-6)
    for z in (0.37, 1.31, 3.17):
        section_distance(body, (0, 0, z), (0, 0, 1), rectangle_points(z, twist * z / 4))
    expected = rectangle_points(4, twist)
    for vertex in body.face(normal="z").vertices:
        assert np.min(np.linalg.norm(expected - vertex.point, axis=1)) < 1e-7


@pytest.mark.parametrize("scale", [1e-9, 1e-3, 1, 1e3])
@pytest.mark.parametrize("draft", [-1, 1])
def test_annular_draft_expands_material_and_preserves_offset_hole(scale, draft):
    radius, hole_radius, length = 2, 0.3, 4
    profile = geo.Profile.from_wires(
        geo.Curve.circle((0, 0, 0), radius * scale),
        [geo.Curve.circle((0.6 * scale, 0, 0), hole_radius * scale)],
        material="pec",
    )
    original_area = profile.area
    body = profile.swept(
        geo.Curve.line((0, 0, 0), (0, 0, length * scale)), twist_deg=90, draft_deg=draft
    )
    slope = math.tan(math.radians(draft))
    volume = math.pi * (
        (radius**2 - hole_radius**2) * length + (radius + hole_radius) * slope * length**2
    )
    assert body.volume() / scale**3 == pytest.approx(volume, rel=2e-6)
    cap = body.face(normal="z")
    area = math.pi * ((radius + slope * length) ** 2 - (hole_radius - slope * length) ** 2)
    assert cap.area / scale**2 == pytest.approx(area, rel=2e-7)
    assert len(cap.edges) == 2
    assert body.material is profile.material
    assert profile.area == original_area


@pytest.mark.parametrize("draft", [-2, 2])
def test_rectangular_draft_is_normal_offset_not_scaling(draft):
    body = geo.Profile.rectangle((0, 0, 0), (2, 1)).swept(
        geo.Curve.line((0, 0, 0), (0, 0, 4)), twist_deg=75, draft_deg=draft
    )
    slope = math.tan(math.radians(draft))
    expected = 8 + 3 * slope * 16 + 4 / 3 * slope**2 * 64
    assert body.volume() == pytest.approx(expected, rel=2e-6)
    for z in (0.37, 1.31, 3.17):
        section_distance(body, (0, 0, z), (0, 0, 1), rectangle_points(z, 75 * z / 4, slope * z))


def test_offset_bore_follows_twist_at_an_independent_middle_cut():
    profile = geo.Profile.from_wires(
        geo.Curve.circle((0, 0, 0), 2), [geo.Curve.circle((0.6, 0, 0), 0.3)]
    )
    body = profile.swept(geo.Curve.line((0, 0, 0), (0, 0, 4)), twist_deg=90, draft_deg=1)
    centroid_shift = 0.3**2 * 0.6 / (2**2 - 0.3**2)
    delta = 2 * math.tan(math.radians(1))
    angle = math.pi / 4
    rotation = np.array(((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))))
    expected = []
    for center, radius in ((centroid_shift, 2 + delta), (0.6 + centroid_shift, 0.3 - delta)):
        for theta in np.linspace(0, 2 * math.pi, 9)[:-1]:
            xy = rotation @ (center + radius * math.cos(theta), radius * math.sin(theta))
            expected.append((*xy, 2))
    section_distance(body, (0, 0, 2), (0, 0, 1), expected)


def test_curved_non_circular_boundary_matches_independent_parallel_area_law():
    from scipy.special import ellipe

    profile = geo.Profile.from_wires(
        geo.Curve.ellipse((0, 0, 0), (2, 1), major_axis="x"),
        [geo.Curve.circle((0, 0, 0), 0.2)],
    )
    body = profile.swept(geo.Curve.line((0, 0, 0), (0, 0, 4)), draft_deg=1)
    perimeter = 8 * ellipe(0.75)
    slope = math.tan(math.radians(1))
    expected = profile.area * 4 + (perimeter + 0.4 * math.pi) * slope * 16 / 2
    assert body.volume() == pytest.approx(expected, rel=2e-6)


def test_fitted_twisted_drafted_solid_is_consumed_by_the_mesher():
    from magnelio.mesh import Mesh, MeshControl

    body = geo.Profile.rectangle((0, 0, 0), (2e-3, 1e-3), material="pec").swept(
        geo.Curve.line((0, 0, 0), (0, 0, 4e-3)), twist_deg=90, draft_deg=1
    )
    model = geo.GeometryModel()
    model.add(body)
    mesh = Mesh.from_geometry(
        model, MeshControl(min_nodes_per_wavelength=4, max_cell_size=0.4e-3), f_max=10e9
    )
    pec_ids = [i for i, material in mesh.material_library.items() if material.name == "PEC"]
    assert pec_ids and np.any(mesh.material_id == pec_ids[0])


def test_unattainable_fit_and_excessive_roll_fail_with_construction_errors():
    profile = geo.Profile.rectangle((0, 0, 0), (2, 1))
    path = geo.Curve.line((0, 0, 0), (0, 0, 4))
    with pytest.raises(RuntimeError, match="CAD resolution"):
        profile.swept(path, twist_deg=90, tolerance=1e-20).volume()
    with pytest.raises(RuntimeError, match="too many sections"):
        profile.swept(path, twist_deg=1e6).volume()


def test_sharp_route_has_no_implicit_corner_joint():
    profile = geo.Profile.rectangle((0, 0, 0), (0.2, 0.1))
    with pytest.raises(ValueError, match="tangent-connected"):
        profile.swept(geo.Curve.polyline([(0, 0, 0), (0, 0, 2), (2, 0, 2)]), twist_deg=90).volume()


def test_multi_edge_arc_uses_total_arc_length_not_edge_parameter():
    radius = 4
    spine = (
        geo.Path.from_pose((0, 0, 0), "z", "y")
        .forward(2)
        .turn_right(radius=radius, angle_deg=90)
        .curve()
    )
    profile = geo.Profile.rectangle((0, 0, 0), (2, 1))
    body = profile.swept(spine, twist_deg=90)
    expected_length = 2 + radius * math.pi / 2
    section_distance(body, (0, 0, 1), (0, 0, 1), rectangle_points(1, 90 / expected_length))
    assert body.volume() == pytest.approx(2 * expected_length, rel=2e-6)


@pytest.mark.parametrize("frame", FRAMES)
def test_spatial_spline_roll_preserves_frame_plane_constraint(frame):
    curve = geo.Curve.spline([(0, 0, 0), (0, 0, 1), (1, 0, 2), (1, 1, 3), (0, 2, 4)])
    profile = geo.Profile.rectangle((0, 0, 0), (0.2, 0.1))
    plain = profile.swept(curve, **settings(frame))
    twisted = profile.swept(curve, twist_deg=60, **settings(frame))
    cap = twisted.face(near=(0, 2, 4), surface_type="plane")
    reference = plain.face(near=(0, 2, 4), surface_type="plane")
    assert cap.normal == pytest.approx(reference.normal, abs=1e-7)
    assert cap.area == pytest.approx(profile.area, rel=2e-6)


def test_reflection_reverses_twist_handedness_and_preserves_draft():
    placement = geo.Translation((3, 4, 5)) @ geo.Rotation((1, 2, 3), 37) @ geo.Mirror("x")
    profile = geo.Profile.rectangle((0, 0, 0), (2, 1))
    spine = geo.Curve.line((0, 0, 0), (0, 0, 4))
    first = placement @ profile.swept(spine, twist_deg=75, draft_deg=1)
    second = (placement @ profile).swept(placement @ spine, twist_deg=-75, draft_deg=1)
    assert first.volume() == pytest.approx(second.volume(), rel=1e-7)
    cap_normal = np.asarray(placement.matrix)[:3, :3] @ np.array((0, 0, 1))
    vertices = np.asarray([v.point for v in first.face(normal=cap_normal).vertices])
    for vertex in second.face(normal=cap_normal).vertices:
        assert np.min(np.linalg.norm(vertices - vertex.point, axis=1)) < 2e-6


def test_face_ref_sheet_and_named_recipe_keep_ownership_material_and_laws():
    owner = geo.Cylinder(
        axis="z", radius=2, inner_radius=0.5, height=1, material="pec"
    ).tagged_face("end", normal="z")
    face = owner.face("end")
    spine = geo.Curve.line(face.centroid, (0, 0, 5))
    result = face.swept(spine, twist_deg=90, draft_deg=1).tagged_face("outlet", normal="z")
    restored = from_recipe(json.loads(json.dumps(to_recipe(result))))
    assert restored.volume() == pytest.approx(result.volume(), rel=1e-8)
    assert restored.face("outlet").area == pytest.approx(result.face("outlet").area)
    assert owner.face("end").area == pytest.approx(3.75 * math.pi)
    assert result.material is owner.material
    assert face.swept(spine, draft_deg=1, material="air").material.name == "air"
    sheet = geo.Surface.parametric(lambda u, v: (u, v, 0), u=(-1, 1), v=(-0.5, 0.5), samples=(4, 4))
    assert sheet.swept(
        geo.Curve.line((0, 0, 0), (0, 0, 4)), twist_deg=90
    ).volume() == pytest.approx(8, rel=2e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"twist_deg": math.nan},
        {"twist_deg": math.inf},
        {"draft_deg": 90},
        {"draft_deg": -90},
        {"draft_deg": math.nan},
        {"tolerance": 0},
        {"tolerance": math.inf},
    ],
)
def test_bad_law_arguments_fail_at_call(kwargs):
    with pytest.raises(ValueError):
        geo.Profile.circle((0, 0, 0), 1).swept(geo.Curve.line((0, 0, 0), (0, 0, 4)), **kwargs)


def test_draft_closure_fails_without_deleting_holes():
    profile = geo.Profile.from_wires(
        geo.Curve.circle((0, 0, 0), 2), [geo.Curve.circle((0, 0, 0), 0.1)]
    )
    with pytest.raises(ValueError, match="closes|topology"):
        profile.swept(geo.Curve.line((0, 0, 0), (0, 0, 4)), draft_deg=5).volume()
    assert len(profile.boundary()) == 2


def test_closed_route_rejects_nonperiodic_laws_and_sews_periodic_seam():
    profile = geo.Profile.rectangle((0, 0, 0), (0.4, 0.2))
    spine = geo.Curve.circle((0, 0, 0), 4)
    for law in ({"draft_deg": 1}, {"twist_deg": 90}):
        with pytest.raises(ValueError, match="closed path"):
            profile.swept(spine, **law).volume()
    result = profile.swept(spine, twist_deg=360, tolerance=1e-5)
    assert result.volume() == pytest.approx(profile.area * spine.length, rel=2e-5)
    from OCC.Core.BRepCheck import BRepCheck_Analyzer

    assert BRepCheck_Analyzer(result._occ_shape()).IsValid()


def test_methods_law_recipe_executes():
    source = Path(__file__).resolve().parents[2] / "docs/methods/geometry.md"
    section = source.read_text().split("## Sweep twist and draft", 1)[1].split("\n## ", 1)[0]
    namespace = {}
    for block in section.split("```python\n")[1:]:
        exec(block.split("```", 1)[0], namespace)
    assert namespace["twisted"].volume() == pytest.approx(8e-9, rel=2e-6)
