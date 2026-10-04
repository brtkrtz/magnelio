"""Uniform geometry operation contracts (DD-275, WP4)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from magnelio import geo
from magnelio.geo._topology_store import from_recipe, to_recipe

pytest.importorskip("OCC.Core.BRepPrimAPI")


def annulus(z=0, scale=1, material="pec"):
    return geo.Profile.from_wires(
        geo.Curve.circle((0, 0, z * scale), 2 * scale),
        [geo.Curve.circle((0, 0, z * scale), scale)],
        material=material,
    )


@pytest.mark.parametrize("scale", [1e-9, 1e-3, 1, 1e3])
def test_face_ref_extrusion_and_sweep_keep_holes_and_material(scale):
    body = geo.Cylinder(radius=2 * scale, inner_radius=scale, height=scale, material="pec")
    face = body.face(normal="z")
    extrusion = face.extruded((0, 0, 3 * scale))
    sweep = face.swept(geo.Curve.line(face.centroid, (0, 0, 4 * scale)))
    for result in (extrusion, sweep):
        assert isinstance(result, geo.Solid)
        assert result.material is body.material
        assert result.volume() == pytest.approx(9 * math.pi * scale**3, rel=1e-9)
        assert len(result.face(normal="z").edges) == 2
    assert face.extruded((0, 0, scale), material="air").material.name == "air"


def test_face_based_circular_coax_sweep_matches_area_times_spine_length():
    body = geo.Cylinder(
        origin=(8, 0, 0), axis="y", radius=2, inner_radius=1, height=-1, material="pec"
    )
    face = body.face(normal="y")
    curve = geo.Curve.arc((8, 0, 0), (8 / 2**0.5, 8 / 2**0.5, 0), (0, 8, 0))
    result = face.swept(curve)
    assert result.volume() == pytest.approx(face.area * curve.length, rel=1e-9)
    assert len(result.face(normal=(-1, 0, 0)).edges) == 2
    assert body.swept(curve, face_near=(9.5, 0, 0)).volume() == pytest.approx(result.volume())


def test_shortest_sweep_rotation_retains_asymmetric_profile_roll_after_placement():
    placement = geo.Translation((2, 4, -1)) @ geo.Rotation((1, 2, 3), 47) @ geo.Rotation("z", 31)
    body = placement @ geo.Brick(origin=(-2, -0.5, -1), size=(4, 1, 1), material="pec")
    normal = np.asarray(placement.matrix)[:3, :3] @ np.array((0, 0, 1))
    face = body.face(normal=normal)
    end = np.asarray(face.centroid) + 3 * normal
    result = face.swept(geo.Curve.line(face.centroid, end))
    cap = result.face(normal=normal)
    expected = np.asarray([v.point for v in face.vertices]) + 3 * normal
    for vertex in cap.vertices:
        assert np.min(np.linalg.norm(expected - vertex.point, axis=1)) < 1e-9
    assert result.volume() == pytest.approx(face.area * 3)


@pytest.mark.parametrize("mirror", [False, True])
def test_sweep_and_thicken_follow_outward_normal_including_reflection(mirror):
    body = geo.Brick(origin=(-2, -0.5, -1), size=(4, 1, 1), material="pec")
    if mirror:
        body = body.mirrored("z")
    face = body.face(normal=(0, 0, -1))
    result = face.thickened(2)
    assert result.face(normal=(0, 0, -1)).centroid[2] == pytest.approx(face.centroid[2] - 2)
    assert result.volume() == pytest.approx(8)
    sweep = face.swept(
        geo.Curve.line(face.centroid, tuple(c + 2 * n for c, n in zip(face.centroid, face.normal)))
    )
    assert sweep.volume() == pytest.approx(result.volume())
    assert np.array(sweep.bounding_box()) == pytest.approx(np.array(result.bounding_box()))


def test_revolve_face_and_profile_use_the_same_orientation_holes_and_material():
    profile = annulus().rotated("x", 90).translated((8, 0, 0))
    body = profile.extruded((0, -1, 0))
    face = body.face(normal="y")
    assert face.revolved("z").volume() == pytest.approx(profile.revolved("z").volume(), rel=1e-9)
    assert face.revolved("z", material="air").material.name == "air"


@pytest.mark.parametrize("blend", ["spline", "ruled", "tangent"])
def test_face_loft_preserves_annular_holes_and_material(blend):
    start = geo.Cylinder(radius=2, inner_radius=1, height=1, material="pec").face(normal="z")
    end = geo.Cylinder(origin=(0, 0, 4), radius=2, inner_radius=1, height=1, material="air").face(
        normal=(0, 0, -1)
    )
    result = start.lofted(end, blend=blend)
    assert result.volume() == pytest.approx(9 * math.pi, rel=1e-7)
    assert result.material is start.owner.material
    assert len(result.face(normal="z").edges) == 2
    assert start.lofted(end, material="air", blend=blend).material.name == "air"


@pytest.mark.parametrize("blend", ["spline", "ruled"])
def test_loft_constructor_and_verb_mix_profiles_planar_sheets_and_faces(blend):
    start = annulus(0)
    end = geo.Cylinder(origin=(0, 0, 3), radius=2, inner_radius=1, height=1, material="air").face(
        normal=(0, 0, -1)
    )
    assert geo.Loft(start, end, blend=blend).volume() == pytest.approx(9 * math.pi)
    assert start.lofted(end, blend=blend).volume() == pytest.approx(9 * math.pi)
    sheet = geo.Surface.parametric(
        lambda u, v: (u, v, 0), u=(0, 2), v=(0, 1), samples=(4, 4), material="pec"
    )
    moved = sheet.translated((0, 0, 3))
    assert geo.Loft(sheet, moved, blend=blend).volume() == pytest.approx(6)
    assert sheet.swept(geo.Curve.line((0, 0, 0), (0, 0, 3))).volume() == pytest.approx(6)


@pytest.mark.parametrize("verb", ["extruded", "revolved", "swept", "lofted", "thickened"])
def test_uniform_materialless_profile_operations_produce_boolean_tools(verb):
    profile = geo.Profile.rectangle((3, 0, 0), (1, 1))
    args = {
        "extruded": ((0, 0, 1),),
        "revolved": ("y",),
        "swept": (geo.Curve.line((0, 0, 0), (0, 0, 1)),),
        "lofted": (profile.translated((0, 0, 1)),),
        "thickened": (1,),
    }[verb]
    result = getattr(profile, verb)(*args)
    assert isinstance(result, geo.Solid)
    assert result.material is None
    assert result.volume() > 0
    from magnelio import GeometryModel

    with pytest.raises(ValueError, match="carries no material"):
        GeometryModel().add(result)


@pytest.mark.parametrize("verb", ["swept", "revolved", "lofted"])
def test_curved_profile_category_is_rejected_at_call(verb):
    face = geo.Sphere(material="pec").face(surface_type="sphere")
    args = {
        "swept": (geo.Curve.line((0, 0, 0), (0, 0, 2)),),
        "revolved": ("z",),
        "lofted": (annulus(4),),
    }[verb]
    with pytest.raises(ValueError, match="planar"):
        getattr(face, verb)(*args)


@pytest.mark.parametrize("verb", ["extruded", "revolved", "swept", "thickened", "lofted"])
def test_curve_profile_category_errors_are_eager(verb):
    curve = geo.Curve.line((0, 0, 0), (1, 0, 0))
    args = {
        "extruded": ((0, 0, 1),),
        "revolved": ("z",),
        "swept": (curve,),
        "thickened": (1,),
        "lofted": (annulus(2),),
    }[verb]
    with pytest.raises(TypeError, match="Profile.*Sheet.*FaceRef"):
        getattr(curve, verb)(*args)


@pytest.mark.parametrize("verb,size", [("filleted", "radius"), ("chamfered", "distance")])
def test_edge_and_face_reference_selection_matches_point_convenience(verb, size):
    body = geo.Brick(material="pec")
    face = body.face(normal="z")
    edge = body.edge(near=(0, 0, 0.5))
    single = getattr(body, verb)(edges=edge, **{size: 0.1})
    old = getattr(body, verb)(near=(0, 0, 0.5), **{size: 0.1})
    assert single.volume() == pytest.approx(old.volume())
    for refs in (face, body.faces(normal="z"), [face]):
        modified = getattr(body, verb)(faces=refs, **{size: 0.1})
        old = getattr(body, verb)(face_near=(0.5, 0.5, 1), **{size: 0.1})
        assert modified.volume() == pytest.approx(old.volume())
    assert getattr(body, verb)(edges=face.edges, **{size: 0.1}).volume() == pytest.approx(
        old.volume()
    )


@pytest.mark.parametrize(
    "verb,kwargs",
    [
        ("filleted", {"radius": 0.1}),
        ("chamfered", {"distance": 0.1}),
        ("shelled", {"thickness": 0.1}),
    ],
)
def test_wrong_owner_references_are_rejected_even_for_identical_geometry(verb, kwargs):
    body = geo.Brick(material="pec")
    other = body.translated((0, 0, 0))
    selection = {"openings" if verb == "shelled" else "faces": other.face(normal="z")}
    with pytest.raises(ValueError, match="exact Solid owner"):
        getattr(body, verb)(**selection, **kwargs)


@pytest.mark.parametrize(
    "selection",
    [lambda b: b.face(normal="z"), lambda b: b.faces(normal="z"), lambda b: [b.face(normal="z")]],
)
def test_shell_openings_accept_refs_and_sets(selection):
    body = geo.Brick(material="pec")
    shell = body.shelled(0.1, openings=selection(body))
    assert shell.volume() == pytest.approx(1 - 0.8**2 * 0.9)
    assert shell.material is body.material


@pytest.mark.parametrize("scale", [1e-9, 1, 1e3])
@pytest.mark.parametrize("operation", ["filleted", "chamfered", "shelled", "extruded"])
def test_selected_operations_and_named_history_work_at_different_build_scales(scale, operation):
    body = geo.Brick(size=(scale, scale, scale), material="pec").tagged_face("side", normal="x")
    if operation == "shelled":
        body = geo.Brick(size=(scale, scale, scale), material="pec").tagged_faces(
            "side", normal="x"
        )
        result = body.shelled(0.1 * scale, openings=body.face(normal="z"))
    elif operation == "extruded":
        body = geo.Brick(size=(scale, scale, scale), material="pec").tagged_face("cap", normal="z")
        result = body.extruded((0, 0, scale), face_near=(scale / 2, scale / 2, scale))
    else:
        result = getattr(body, operation)(
            edges=body.edge(near=(0, 0, scale / 2)),
            **{"radius" if operation == "filleted" else "distance": scale * 0.1},
        )
    reference_volume = result.volume()
    for build_scale in (0.25, 1, 128):
        assert result.volume(scale=build_scale) == pytest.approx(reference_volume, rel=1e-8)
    restored = from_recipe(json.loads(json.dumps(to_recipe(result))))
    assert restored.volume() == pytest.approx(reference_volume, rel=1e-8)
    label = "cap" if operation == "extruded" else "side"
    if operation == "shelled":
        assert len(restored.faces(label)) == len(result.faces(label))
    else:
        assert restored.face(label).area == pytest.approx(result.face(label).area)


def test_connected_edge_selection_replays_exact_origin_without_topology_indices():
    body = geo.Brick(material="pec").tagged_face("side", normal="x")
    cap = body.face(normal="z")
    result = body.filleted(edges=cap.edges, radius=0.1)
    recipe = json.loads(json.dumps(to_recipe(result)))
    restored = from_recipe(recipe)
    assert restored.volume() == pytest.approx(result.volume())
    assert restored.face("side").area == pytest.approx(result.face("side").area)


@pytest.mark.parametrize("blend", ["spline", "ruled", "tangent"])
def test_loft_hole_count_and_tangent_tension_are_validated_at_call(blend):
    a = annulus()
    with pytest.raises(ValueError, match="same number of holes"):
        a.lofted(geo.Profile.circle((0, 0, 3), 2), blend=blend)
    body_a = a.extruded((0, 0, 1))
    body_b = geo.Profile.circle((0, 0, 4), 2).extruded((0, 0, 1))
    with pytest.raises(ValueError, match="same number of holes"):
        body_a.lofted(other=body_b, face_near=(0, 1.5, 1), other_face_near=(0, 0, 4), blend=blend)
    with pytest.raises(ValueError, match="finite"):
        a.lofted(annulus(3), blend="tangent", tension=float("nan"))


def test_wp4_methods_recipes_execute():
    import re

    root = Path(__file__).parents[2]
    text = (root / "docs/methods/geometry.md").read_text()
    section = text.split("## Uniform profile operations", 1)[1].split("## Placement", 1)[0]
    namespace = {}
    for block in re.findall(r"```python\n(.*?)```", section, re.S):
        exec(block, namespace)
    assert namespace["extension"].volume() > 0
    assert namespace["housing"].volume() > 0


def test_wp4_upgrade_recipe_executes():
    import re

    text = (Path(__file__).parents[2] / "docs/migration-geometry.md").read_text()
    section = text.split("## Uniform operations", 1)[1].split("\n## ", 1)[0]
    namespace = {}
    for block in re.findall(r"```python\n(.*?)```", section, re.S):
        exec(block, namespace)
    assert namespace["extension"].volume() == pytest.approx(36e-9)
    assert namespace["housing"].volume() > 0


@pytest.mark.parametrize(
    "verb,kwargs", [("filleted", {"radius": 0.1}), ("chamfered", {"distance": 0.1})]
)
@pytest.mark.parametrize("selection", ["empty", "wrong_kind", "conflicting", "tied"])
def test_edge_selection_failures_are_eager(verb, kwargs, selection):
    body = geo.Brick(material="pec")
    modes = {
        "empty": {"edges": []},
        "wrong_kind": {"edges": body.face(normal="z")},
        "conflicting": {"edges": body.edge(near=(0, 0, 0.5)), "faces": body.face(normal="z")},
        "tied": {"near": (0, 0, 0)},
    }
    expected = TypeError if selection == "wrong_kind" else ValueError
    with pytest.raises(expected):
        getattr(body, verb)(**modes[selection], **kwargs)


def test_duplicate_selected_edges_and_openings_are_deduplicated():
    body = geo.Brick(material="pec")
    edge = body.edge(near=(0, 0, 0.5))
    assert body.filleted(edges=[edge, edge], radius=0.1).volume() == pytest.approx(
        body.filleted(edges=edge, radius=0.1).volume()
    )
    cap = body.face(normal="z")
    assert body.shelled(0.1, openings=[cap, cap]).volume() == pytest.approx(
        body.shelled(0.1, openings=cap).volume()
    )
    with pytest.raises(ValueError, match="non-empty"):
        body.filleted(near=[], radius=0.1)


def test_flat_spline_sheet_supports_symmetric_thickening():
    sheet = geo.Surface.parametric(
        lambda u, v: (u, v, 0), u=(0, 2), v=(0, 1), samples=(4, 4), material="pec"
    )
    slab = sheet.thickened(0.2, direction="symmetric")
    assert slab.volume() == pytest.approx(0.4)
    assert np.array(slab.bounding_box()) == pytest.approx(
        np.array(((0, 0, -0.1), (2, 1, 0.1))), abs=2e-7
    )


def test_tangent_bend_keeps_both_hole_boundaries_under_spatial_placement():
    from OCC.Core.BRepCheck import BRepCheck_Analyzer

    start = geo.Cylinder(radius=2e-3, inner_radius=1e-3, height=4e-3, material="pec").face(
        normal=(0, 0, -1)
    )
    end = geo.Cylinder(
        origin=(0, 20e-3, -12e-3),
        axis="y",
        radius=2e-3,
        inner_radius=1e-3,
        height=4e-3,
        material="pec",
    ).face(normal=(0, -1, 0))
    placement = geo.Translation((3e-3, -2e-3, 4e-3)) @ geo.Rotation((1, 2, 3), 37)
    a = placement @ start.owner
    b = placement @ end.owner
    rotation = np.array(placement.matrix)[:3, :3]
    face_a = a.face(normal=rotation @ start.normal)
    face_b = b.face(normal=rotation @ end.normal)
    result = face_a.lofted(face_b, blend="tangent")
    assert result.volume() > 0
    assert BRepCheck_Analyzer(result._occ_shape()).IsValid()
    assert (
        len(
            result.face(near=face_a.centroid, normal=-(rotation @ start.normal))
            .detached()
            .boundary()
        )
        == 2
    )
    assert (
        len(
            result.face(near=face_b.centroid, normal=-(rotation @ end.normal)).detached().boundary()
        )
        == 2
    )


def test_tangent_loft_with_several_offset_holes_retains_correspondence():
    outer = geo.Curve.polyline([(-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0), (-2, -2, 0)])
    profile = geo.Profile.from_wires(
        outer,
        [geo.Curve.circle((-0.8, 0, 0), 0.3), geo.Curve.circle((0.8, 0.2, 0), 0.4)],
        material="pec",
    )
    a = profile.extruded((0, 0, -1)).face(normal="z")
    b = profile.translated((0, 0, 3)).extruded((0, 0, 1)).face(normal=(0, 0, -1))
    transition = a.lofted(b, blend="tangent")
    assert transition.volume() == pytest.approx(profile.area * 3, rel=1e-7)
    assert len(transition.face(normal="z").edges) == 6


def test_reference_codec_refuses_changed_connected_snapshot():
    from copy import deepcopy

    body = geo.Brick(material="pec").tagged_face("side", normal="x")
    result = body.filleted(edges=body.face(normal="z").edges, radius=0.1)
    recipe = deepcopy(to_recipe(result))
    for node in recipe["nodes"]:
        if node["operation"] == "_FilletedShape":
            node["arguments"]["_edges"][0]["reference"]["member_brep"] = "broken"
    with pytest.raises(geo.TopologyEvolutionError, match="exact snapshot"):
        from_recipe(recipe)


def test_sheet_and_face_loft_share_the_same_keyword_grammar():
    import inspect

    assert inspect.signature(geo.Sheet.lofted) == inspect.signature(geo.FaceRef.lofted)
    profile = annulus()
    end = annulus(3)
    assert profile.lofted(other=end).volume() == pytest.approx(9 * math.pi)


def test_tangent_volume_measurement_is_preserved_after_placement_and_tagging():
    a = geo.Cylinder(radius=2, inner_radius=1, height=1, material="pec").face(normal="z")
    b = geo.Cylinder(origin=(0, 0, 4), radius=2, inner_radius=1, height=1, material="pec").face(
        normal=(0, 0, -1)
    )
    result = a.lofted(b, blend="tangent")
    placed = result.rotated((1, 2, 3), 37).translated((3, -2, 4))
    assert placed.volume() == pytest.approx(9 * math.pi, rel=1e-7)
    tagged = result.tagged_face("cap", normal="z")
    restored = from_recipe(json.loads(json.dumps(to_recipe(tagged))))
    assert restored.volume() == pytest.approx(result.volume(), rel=1e-9)
    assert "DetachedProfile" in repr(a.detached())
