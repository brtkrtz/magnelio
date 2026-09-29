"""Owned topology, selection identity and construction histories (DD-275)."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

import magnelio as mio
from magnelio import geo
from magnelio.geo._topology_history import names
from magnelio.geo._topology_store import from_recipe, to_recipe
from magnelio.geo.topology import _scale
from magnelio.io.project import ProjectStore
from magnelio.mesh import GridLines


@pytest.mark.parametrize("size", [1e-9, 1e-6, 1e-3, 1, 1e3])
def test_measurements_connectivity_and_scale(size):
    b = geo.Brick(size=(size, 2 * size, 3 * size), material="pec")
    f = b.face(normal="z", surface_type="plane")
    assert f.owner is b
    assert f.area == pytest.approx(2 * size**2)
    assert f.centroid == pytest.approx((0.5 * size, size, 3 * size))
    assert f.normal == pytest.approx((0, 0, 1))
    assert f.is_planar
    assert len(f.edges) == 4
    assert len(f.vertices) == 4
    assert sum(e.length for e in f.edges) == pytest.approx(6 * size)
    assert all(e.owner is b for e in f.edges)
    e = b.edge(near=(0.5 * size, 0, 0), curve_type="line")
    assert e.length == pytest.approx(size)
    assert np.linalg.norm(np.subtract(e.end, e.start)) == pytest.approx(size)
    assert len(e.vertices) == 2
    assert b.vertex(near=(0, 0, 0)).point == pytest.approx((0, 0, 0))
    assert len(b.faces()) == 6
    assert len(b.edges()) == 12


@pytest.mark.parametrize(
    "selector,kwargs,count",
    [
        ("face", {"near": (0, 0.5, 0)}, 2),
        ("face", {"near": (0.5, 0.5, 0.5)}, 6),
        ("edge", {"near": (0, 0, 0)}, 3),
        ("vertex", {"near": (0.5, 0.5, 0.5)}, 8),
    ],
)
def test_ties_refuse_kernel_order(selector, kwargs, count):
    b = geo.Brick()
    with pytest.raises(geo.AmbiguousTopologyError, match=f"{count} equally"):
        getattr(b, selector)(**kwargs)
    if selector != "vertex":
        assert len(getattr(b, selector + "s")(**kwargs)) == count
    assert b.face(near=(0, 0.5, 0), normal=(0, 0, -1)).normal == (0, 0, -1)


def test_distance_is_to_complete_trimmed_face_not_centroid():
    b = geo.Brick(size=(100, 1, 1))
    assert b.face(near=(1, 0.5, 1.001)).normal == (0, 0, 1)
    tube = geo.Cylinder(radius=2, inner_radius=1, height=3)
    with pytest.raises(geo.AmbiguousTopologyError):
        tube.face(near=(0, 0, 1.5), surface_type="plane")
    assert tube.face(near=(0, 0, 0), surface_type="cylinder").area == pytest.approx(6 * math.pi)


@pytest.mark.parametrize(
    "verb,kwargs,error",
    [
        ("face", {}, ValueError),
        ("face", {"name": 0}, ValueError),
        ("face", {"name": ""}, ValueError),
        ("face", {"name": "missing"}, geo.TopologySelectionError),
        ("face", {"name": "missing", "normal": "z"}, ValueError),
        ("face", {"surface_type": "sphere"}, geo.TopologySelectionError),
        ("face", {"surface_type": "flat"}, ValueError),
        ("edge", {"curve_type": "straight"}, ValueError),
        ("face", {"normal": (0, 0, 0)}, ValueError),
        ("vertex", {"near": (math.nan, 0, 0)}, ValueError),
    ],
)
def test_selector_argument_errors(verb, kwargs, error):
    with pytest.raises(error):
        getattr(geo.Brick(), verb)(**kwargs)


def test_references_are_read_only_owner_views_not_geometry():
    b = geo.Brick(material="pec")
    f = b.face(normal="z")
    assert isinstance(f, geo.TopologyRef)
    assert not isinstance(f, geo.Shape)
    with pytest.raises(FrozenInstanceError):
        f.owner = geo.Brick()
    with pytest.raises(AttributeError):
        f.area = 0
    assert not hasattr(f, "translated")
    assert hasattr(f, "extruded")  # WP4 consumes refs without independent placement.
    with pytest.raises(TypeError):
        mio.GeometryModel().add(f)
    with pytest.raises(TypeError):
        geo.Union(b, f)
    with pytest.raises(TypeError):
        geo.Translation((1, 0, 0)) @ f
    assert not hasattr(geo.Profile, "tag_face")


def test_curved_normals_and_detachment():
    b = geo.Cylinder(radius=2, height=3, material="pec")
    f = b.face(surface_type="cylinder")
    assert not f.is_planar
    assert f.area == pytest.approx(12 * math.pi)
    with pytest.raises(ValueError, match="normal_at"):
        _ = f.normal
    assert f.normal_at((2, 0, 1)) == pytest.approx((1, 0, 0))
    assert f.normal_at((0, -2, 1)) == pytest.approx((0, -1, 0))
    with pytest.raises(ValueError, match="on the selected face"):
        f.normal_at((0, 0, 1))
    assert isinstance(f.detached(), geo.Surface)
    assert f.detached().material is b.material
    picked = b.face(near=(2.1, 0, 1), normal="x", surface_type="cylinder")
    assert picked.area == pytest.approx(f.area)
    with pytest.raises(geo.TopologySelectionError):
        b.face(normal="x", surface_type="cylinder")
    moved = b.translated((7, 0, 0)).mirrored("x")
    assert moved.face(surface_type="cylinder").normal_at((-9, 0, 1)) == pytest.approx((-1, 0, 0))


def test_detached_profile_keeps_holes_placement_and_material():
    b = geo.Cylinder(origin=(3, 4, 5), radius=2, inner_radius=1, height=3, material="pec")
    f = b.face(near=(4.5, 4, 8), normal="z")
    profile = f.detached()
    assert isinstance(profile, geo.Profile)
    assert profile.material is b.material
    assert profile.area == pytest.approx(3 * math.pi)
    assert len(profile.boundary()) == 2
    assert profile.translated((0, 0, 1)).area == pytest.approx(f.area)
    assert profile.extruded((0, 0, 2)).volume() == pytest.approx(6 * math.pi)
    perimeter = [e.as_curve() for e in f.edges]
    assert sum(c.length for c in perimeter) == pytest.approx(6 * math.pi)
    assert all(isinstance(c.rotated("y", 90), geo.Curve) for c in perimeter)
    assert np.array(profile.bounding_box()) == pytest.approx(np.array(f.bounding_box()))


def tagged_box(size=1):
    return (
        geo.Brick(size=(size, size, size), material="pec", name="body")
        .tag_face("port", normal="z")
        .tag_edge("rim", near=(size / 2, 0, size))
        .tag_vertex("corner", near=(0, 0, size))
        .tag_faces("walls", surface_type="plane")
        .tag_edges("outline", curve_type="line")
    )


def test_registration_is_immutable_unique_and_kind_specific():
    b = geo.Brick()
    t = b.tag_face("port", normal="z").tag_edge("port", near=(0.5, 0, 1))
    with pytest.raises(geo.TopologySelectionError):
        b.face("port")
    assert t.face("port").owner is t
    assert t.edge("port").owner is t
    with pytest.raises(ValueError, match="already exists"):
        t.tag_face("port", normal="x")
    with pytest.raises(ValueError, match="non-empty"):
        b.tag_face(" ", normal="z")
    with pytest.raises(geo.TopologySelectionError, match="singular"):
        t.faces("port")
    s = b.tag_faces("all")
    with pytest.raises(geo.TopologySelectionError, match="plural"):
        s.face("all")
    assert len(s.faces("all")) == 6


@pytest.mark.parametrize("size", [1e-9, 1e-3, 1, 1e3])
def test_all_names_follow_composed_affine_placements_and_reflections(size):
    b = tagged_box(size)
    transform = (
        geo.Translation((7 * size, 0, 0)) @ geo.Scale(-2) @ geo.Mirror("x") @ geo.Rotation("y", 35)
    )
    moved = transform @ b
    assert moved.material is b.material
    assert moved.name == "body"
    assert moved.face("port").centroid == pytest.approx(transform.point(b.face("port").centroid))
    assert moved.vertex("corner").point == pytest.approx(transform.point(b.vertex("corner").point))
    assert moved.face("port").area == pytest.approx(4 * size**2)
    assert moved.edge("rim").length == pytest.approx(2 * size)
    linear = np.array(transform.matrix)[:3, :3]
    expected_normal = linear @ np.array((0, 0, 1)) / 2
    assert moved.face("port").normal == pytest.approx(expected_normal)
    assert len(moved.faces("walls")) == 6
    assert len(moved.edges("outline")) == 12
    # A different enclosing model scale rebuilds origins at that scale.
    scale = _scale(moved) * 8
    assert len(names(moved, scale)) == 5
    assert len(moved.faces(normal=expected_normal)) == 1
    assert b.face("port").normal == (0, 0, 1)
    added = moved.tag_face("side", near=transform.point((size, size / 2, size / 2)))
    assert added.scaled(0.5).face("side").area == pytest.approx(size**2)


def test_tagged_arrays_groups_and_conflicting_union_names():
    b = geo.Brick(material="pec").tag_face("port", normal="z")
    group = b.translated((2, 0, 0), repeat=3, copy=True, group=True)
    assert [s.face("port").centroid[0] for s in group.members()] == [0.5, 2.5, 4.5, 6.5]
    assert all(s.face("port").owner is s for s in group.members())
    with pytest.raises(geo.TopologyEvolutionError, match="conflicting"):
        b.translated((2, 0, 0), repeat=2, unite=True)


@pytest.mark.parametrize(
    "kind,selectors",
    [
        ("face", {"normal": "z"}),
        ("edge", {"near": (1, 0, 2)}),
    ],
)
def test_boolean_split_requires_deliberate_set_and_never_retargets(kind, selectors):
    b = geo.Brick(size=(2, 2, 2), material="pec")
    slit = geo.Brick(origin=(0.9, -1, -1), size=(0.2, 4, 4))
    singular = getattr(b, "tag_" + kind)("pick", **selectors)
    with pytest.raises(geo.TopologyEvolutionError, match="split into 2"):
        _ = singular - slit
    deliberate = getattr(b, "tag_" + kind + "s")("pick", **selectors)
    split = deliberate - slit
    assert len(getattr(split, kind + "s")("pick")) == 2
    assert sum(
        ref.area if kind == "face" else ref.length for ref in getattr(split, kind + "s")("pick")
    ) == pytest.approx(3.6 if kind == "face" else 1.8)


@pytest.mark.parametrize(
    "kind,selectors",
    [
        ("face", {"normal": "z"}),
        ("edge", {"near": (0.5, 0, 1)}),
        ("vertex", {"near": (0, 0, 1)}),
        ("faces", {"normal": "z"}),
    ],
)
def test_deleted_names_fail_at_construction_call(kind, selectors):
    b = getattr(geo.Brick(), "tag_" + kind)("pick", **selectors)
    with pytest.raises(geo.TopologyEvolutionError, match="deleted|provable"):
        _ = b - geo.Brick(size=(2, 2, 2))


@pytest.mark.parametrize(
    "operation",
    [
        lambda b: b + geo.Brick(origin=(2, 0, 0)),
        lambda b: b - geo.Brick(origin=(2, 0, 0)),
        lambda b: b & geo.Brick(origin=(-1, -1, -1), size=(3, 3, 3)),
        lambda b: b.chamfered(near=(0, 0, 0.5), distance=0.1),
        lambda b: b.filleted(near=(0, 0, 0.5), radius=0.1),
        lambda b: b.extruded((0, 0, 1), face_near=(0.5, 0.5, 1)),
    ],
)
def test_provable_face_successor_and_store_replay(operation):
    b = geo.Brick(material="pec").tag_face("port", normal="z")
    r = operation(b)
    expected_normal = (0, 0, -1) if type(r).__name__ == "_ExtrudedFaceShape" else (0, 0, 1)
    assert r.face("port").normal == pytest.approx(expected_normal)
    restored = from_recipe(json.loads(json.dumps(to_recipe(r))))
    assert restored.face("port").centroid == pytest.approx(r.face("port").centroid)
    assert restored.face("port").area == pytest.approx(r.face("port").area)
    assert restored.volume() == pytest.approx(r.volume())


def test_shell_reports_offset_face_split_and_replays_deliberate_set():
    b = geo.Brick(material="pec")
    with pytest.raises(geo.TopologyEvolutionError, match="split"):
        b.tag_face("cap", normal="z").shelled(thickness=0.1, opening_face_near=(0.5, 0.5, 0))
    shell = b.tag_faces("cap", normal="z").shelled(thickness=0.1, opening_face_near=(0.5, 0.5, 0))
    assert len(shell.faces("cap")) == 2
    restored = from_recipe(json.loads(json.dumps(to_recipe(shell))))
    assert len(restored.faces("cap")) == 2
    assert sorted(f.area for f in restored.faces("cap")) == pytest.approx(
        sorted(f.area for f in shell.faces("cap"))
    )


def test_fillet_and_chamfer_unique_edge_successors_and_vertex_deletion():
    b = geo.Brick().tag_edge("rim", near=(0.5, 0, 1))
    for r in [
        b.filleted(near=(0, 0, 0.5), radius=0.1),
        b.chamfered(near=(0, 0, 0.5), distance=0.1),
    ]:
        assert r.edge("rim").length == pytest.approx(0.9)
    corner = geo.Brick().tag_vertex("corner", near=(0, 0, 1))
    with pytest.raises(geo.TopologyEvolutionError):
        corner.filleted(near=(0, 0, 0.5), radius=0.1)


def test_unprovable_loft_face_successor_is_explicit():
    a = geo.Brick(material="pec").tag_face("port", normal="z")
    b = geo.Brick(origin=(0, 0, 3), material="pec")
    with pytest.raises(geo.TopologyEvolutionError, match="provable"):
        a.lofted((0.5, 0.5, 1), b, (0.5, 0.5, 3), blend="ruled")


@pytest.mark.parametrize("size", [1e-9, 1e-3, 1, 1e3])
def test_project_round_trip_preserves_semantic_origin_history_and_sets(tmp_path, size):
    b = tagged_box(size).rotated("y", 22.5).mirrored("x").translated((4 * size, 0, 0))
    slit = geo.Brick(origin=(100 * size, 100 * size, 100 * size), size=(size, size, size))
    b = b - slit
    grid = GridLines(
        x=np.linspace(0, size, 3), y=np.linspace(0, size, 3), z=np.linspace(0, size, 3)
    )
    ProjectStore.create(tmp_path / "p", mio.Mesh.from_grid(grid), geometry=[b])
    (loaded,) = mio.open_project(tmp_path / "p").geometry
    assert loaded.face("port").centroid == pytest.approx(b.face("port").centroid)
    assert loaded.face("port").normal == pytest.approx(b.face("port").normal)
    assert loaded.edge("rim").length == pytest.approx(size)
    assert loaded.vertex("corner").point == pytest.approx(b.vertex("corner").point)
    assert len(loaded.edges("outline")) == 12
    assert len(loaded.faces("walls")) == 6
    assert loaded.material == b.material
    metadata = json.loads((tmp_path / "p/geometry.json").read_text())
    recipe = metadata["topology_recipes"][0]
    assert any(node["operation"] == "tag" for node in recipe["nodes"])
    assert any(node["operation"] == "transform" for node in recipe["nodes"])
    assert "topology_index" not in json.dumps(recipe)
    assert loaded.mirrored("y").face("port").area == pytest.approx(size**2)
    recipe["expected"][0]["count"] = 2
    with pytest.raises(geo.TopologyEvolutionError, match="cardinality"):
        from_recipe(recipe)


def test_split_set_replay_and_origin_cardinality_validation():
    b = geo.Brick(size=(2, 2, 2)).tag_faces("top", normal="z")
    r = b - geo.Brick(origin=(0.9, -1, -1), size=(0.2, 4, 4))
    recipe = json.loads(json.dumps(to_recipe(r)))
    assert len(from_recipe(recipe).faces("top")) == 2
    tag_node = next(n for n in recipe["nodes"] if n["operation"] == "tag")
    tag_node["registration"][4] = 2
    with pytest.raises(geo.TopologyEvolutionError, match="cardinality"):
        from_recipe(recipe)


def test_inventory_cached_per_owner_and_scale_with_hundreds_of_faces(monkeypatch):
    import magnelio.geo.topology as topology

    b = geo.Union(*(geo.Brick(origin=(3 * i, 0, 0)) for i in range(100)))
    count = 0
    original = topology._index

    def counted(shape, kind):
        nonlocal count
        count += 1
        return original(shape, kind)

    monkeypatch.setattr(topology, "_index", counted)
    assert len(b.faces()) == 600
    assert len(b.faces()) == 600
    assert len(b.edges()) == 1200
    assert len(b.edges()) == 1200
    assert count == 2
    other = b.translated((0, 1, 0))
    assert len(other.faces()) == 600
    assert count == 3
    topology._inventory(b, _scale(b) * 2, "face")
    assert count == 4


def test_topology_methods_documentation_recipes_execute():
    import re

    text = (Path(__file__).resolve().parents[2] / "docs/methods/geometry.md").read_text()
    text = text.split("## Owned topology", 1)[1].split("## Uniform profile operations", 1)[0]
    blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    assert blocks
    namespace = {}
    for snippet in blocks:
        exec(compile(snippet, "topology methods recipe", "exec"), namespace)
    assert namespace["extension"].volume() == pytest.approx(3 * math.pi * 1e-9)
    assert len(namespace["split"].faces("cap")) == 2


def test_nearest_selection_screens_cached_bounds_before_exact_distances(monkeypatch):
    import magnelio.geo.topology as topology

    b = geo.Union(*(geo.Brick(origin=(3 * i, 0, 0)) for i in range(100)))
    calls = 0
    original = topology._distance

    def measured(shape, point, scale):
        nonlocal calls
        calls += 1
        return original(shape, point, scale)

    monkeypatch.setattr(topology, "_distance", measured)
    for _ in range(2):
        assert b.face(near=(297.5, 0.5, 1.01)).normal == pytest.approx((0, 0, 1))
    assert calls == 2
    assert len(b._topology_bounds) == 1


def test_untagged_lazy_tool_histories_do_not_delete_base_names():
    b = geo.Brick(material="pec").tag_face("port", normal="z")
    tool = geo.Brick(origin=(3, 0, 0)).filleted(near=(3, 0, 0.5), radius=0.1)
    result = b + tool
    assert result.face("port").area == pytest.approx(1)
    restored = from_recipe(json.loads(json.dumps(to_recipe(result))))
    assert restored.face("port").area == pytest.approx(1)


def test_closed_shell_history_retains_both_original_and_offset_branches():
    b = geo.Brick(material="pec")
    with pytest.raises(geo.TopologyEvolutionError, match="split"):
        b.tag_face("cap", normal="z").shelled(thickness=0.1)
    shell = b.tag_faces("cap", normal="z").shelled(thickness=0.1)
    assert len(shell.faces("cap")) == 2
    assert sorted(f.area for f in shell.faces("cap")) == pytest.approx([0.64, 1])
    restored = from_recipe(json.loads(json.dumps(to_recipe(shell))))
    assert len(restored.faces("cap")) == 2


def test_planarity_and_detachment_use_geometry_not_only_analytic_type():
    sheet = geo.Surface.parametric(lambda u, v: (u, v, 0), u=(0, 1), v=(0, 1), samples=(4, 4))
    b = sheet.extruded((0, 0, 1), material="pec")
    f = b.face(surface_type="bspline", normal="z")
    assert f.is_planar
    assert f.normal == pytest.approx((0, 0, 1))
    profile = f.detached()
    assert isinstance(profile, geo.Profile)
    assert profile.area == pytest.approx(1)
    assert len(profile.boundary()) == 1
    spine = geo.Curve.line(f.centroid, (f.centroid[0], f.centroid[1], 3))
    assert profile.swept(spine).volume() == pytest.approx(2)
    moved = b.tag_face("port", surface_type="bspline", normal="z").mirrored("x")
    assert moved.face("port").normal == pytest.approx((0, 0, 1))
