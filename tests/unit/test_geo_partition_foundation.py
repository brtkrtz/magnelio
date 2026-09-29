"""Geometry contracts for independently owned partitions and sections."""

import json
import math
import re
import runpy
from pathlib import Path

import pytest

from magnelio import geo
from magnelio.geo._topology_store import from_recipe, to_recipe

pytest.importorskip("OCC.Core.BRepAlgoAPI")


def _box():
    return geo.Brick(origin=(0, 0, 0), size=(2, 2, 2), material="pec")


def test_plane_partition_conserves_volume_and_returns_independent_solids():
    source = _box()
    pieces = source.partition(normal="x", position=0.5)
    assert len(pieces) == 2
    assert all(isinstance(piece, geo.Solid) and piece is not source for piece in pieces)
    assert all(piece.material is source.material for piece in pieces)
    assert sorted(piece.volume() for piece in pieces) == pytest.approx([2, 6])
    assert source.volume() == pytest.approx(8)


def test_oblique_world_plane_has_exact_conserved_volume_and_section_area():
    source = _box()
    normal = (1, 1, 0)
    position = math.sqrt(2)
    pieces = source.partition(normal=normal, position=position)
    assert sorted(piece.volume() for piece in pieces) == pytest.approx([4, 4])
    section = source.section(normal=normal, position=position)
    assert len(section) == 1
    assert section[0].is_closed
    assert section[0].length == pytest.approx(4 + 4 * math.sqrt(2))
    filled = source.section(normal=normal, position=position, filled=True)
    assert len(filled) == 1
    assert filled[0].area == pytest.approx(4 * math.sqrt(2))


def test_geometric_cutter_partitions_inside_and_outside_in_one_evaluation():
    source = _box()
    cutter = geo.Brick(origin=(0.5, -1, -1), size=(1, 4, 4))
    pieces = source.partition(cutter)
    assert sorted(piece.volume() for piece in pieces) == pytest.approx([2, 2, 4])
    assert sum(piece.volume() for piece in pieces) == pytest.approx(source.volume())
    assert sorted(curve.length for curve in source.section(cutter)) == pytest.approx([8, 8])


def test_spatial_sheet_cutter_preserves_oblique_hexagonal_intersection():
    source = _box()
    cutter = geo.Profile.rectangle((1, 1, 1), (5, 5), normal=(1, 1, 1))
    parts = source.partition(cutter)
    assert sorted(part.volume() for part in parts) == pytest.approx([4, 4])
    (trace,) = source.section(cutter)
    assert trace.is_closed
    assert trace.length == pytest.approx(6 * math.sqrt(2))


def test_partition_covaries_with_rigid_placement():
    angle = math.radians(31)
    placed = _box().rotated("z", 31)
    normal = (math.cos(angle), math.sin(angle), 0)
    parts = placed.partition(normal=normal, position=1)
    assert sorted(part.volume() for part in parts) == pytest.approx([4, 4])
    (trace,) = placed.section(normal=normal, position=1)
    assert trace.length == pytest.approx(8)


def test_annular_section_preserves_two_curves_and_one_filled_profile_with_hole():
    tube = geo.Cylinder(origin=(0, 0, 0), radius=2, inner_radius=1, height=4, material="pec")
    curves = tube.section(normal="z", position=2)
    assert len(curves) == 2
    assert all(curve.is_closed for curve in curves)
    assert sorted(curve.length for curve in curves) == pytest.approx([2 * math.pi, 4 * math.pi])
    filled = tube.section(normal="z", position=2, filled=True)
    assert len(filled) == 1
    assert isinstance(filled[0], geo.Profile)
    assert filled[0].area == pytest.approx(3 * math.pi)
    assert len(filled[0].boundary()) == 2
    assert filled[0].material is tube.material


def test_disconnected_regions_and_filled_sections_are_all_returned():
    source = geo.Union(
        geo.Brick(origin=(0, 0, 0), size=(1, 1, 1), material="pec"),
        geo.Brick(origin=(3, 0, 0), size=(1, 1, 1)),
    )
    parts = source.partition(normal="z", position=0.5)
    assert len(parts) == 4
    assert sorted(part.volume() for part in parts) == pytest.approx([0.5] * 4)
    filled = source.section(normal="z", position=0.5, filled=True)
    assert len(filled) == 2
    assert sorted(profile.area for profile in filled) == pytest.approx([1, 1])


def test_sheet_partition_retains_dimension_and_curved_sheet_section_is_open():
    profile = geo.Profile.rectangle((0, 0, 0), (2, 2), material="pec")
    parts = profile.partition(normal="x", position=0)
    assert len(parts) == 2
    assert all(isinstance(part, geo.Profile) for part in parts)
    assert sum(part.area for part in parts) == pytest.approx(profile.area)
    assert profile.area == pytest.approx(4)
    solid_cutter = geo.Brick(origin=(0, -2, -1), size=(2, 4, 2), material="air")
    by_solid = profile.partition(solid_cutter)
    assert sorted(part.area for part in by_solid) == pytest.approx([2, 2])
    assert all(part.material is profile.material for part in by_solid)
    curved = geo.Surface.parametric(
        lambda u, v: (u, v, 0.25 * u * u), u=(0, 2), v=(0, 2), samples=(8, 8)
    )
    curved_parts = curved.partition(normal="x", position=1)
    assert len(curved_parts) == 2
    assert all(isinstance(part, geo.Surface) for part in curved_parts)
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps

    def area(sheet):
        props = GProp_GProps()
        brepgprop.SurfaceProperties(sheet._occ_shape(), props, 1e-10)
        return props.Mass()

    assert sum(area(part) for part in curved_parts) == pytest.approx(area(curved), rel=1e-10)
    (trace,) = curved.section(normal="x", position=1)
    assert not trace.is_closed
    assert trace.length == pytest.approx(2)


def test_repeated_partition_retains_material_and_total_volume():
    first = _box().partition(normal="x", position=1)
    quarters = tuple(part for half in first for part in half.partition(normal="y", position=1))
    assert len(quarters) == 4
    assert sorted(part.volume() for part in quarters) == pytest.approx([2] * 4)
    assert all(part.material is first[0].material for part in quarters)


def test_no_cut_tangency_and_coincident_boundary_semantics():
    source = _box()
    for position in (-1, 0, 2, 3):
        (whole,) = source.partition(normal="x", position=position)
        assert whole is not source
        assert whole.volume() == pytest.approx(source.volume())
    assert source.section(normal="x", position=3) == ()
    with pytest.raises(ValueError, match="share a face"):
        source.section(normal="x", position=0)
    sphere = geo.Sphere(center=(0, 0, 0), radius=1)
    assert len(sphere.partition(normal="z", position=1)) == 1
    assert sphere.section(normal="z", position=1) == ()


@pytest.mark.parametrize("length", [1e-9, 1, 1e3])
def test_coincident_plane_is_detected_at_model_scale(length):
    source = geo.Brick(origin=(0, 0, 0), size=(2 * length,) * 3)
    with pytest.raises(ValueError, match="share a face"):
        source.section(normal="x", position=0)


def test_empty_and_ineligible_requests_fail_at_call():
    source = _box()
    for kwargs in ({}, {"normal": "x"}, {"position": 0}):
        with pytest.raises(ValueError):
            source.partition(**kwargs)
    with pytest.raises(ValueError, match="not both"):
        source.section(source, normal="x", position=1)
    with pytest.raises(TypeError, match="cutter"):
        source.partition(geo.Curve.line((0, 0, 0), (1, 0, 0)))
    with pytest.raises(TypeError, match="filled section"):
        geo.Profile.rectangle((0, 0, 0), (2, 2)).section(normal="x", position=0, filled=True)
    with pytest.raises(TypeError, match="filled section"):
        source.section(geo.Profile.rectangle((0, 0, 1), (2, 2)), filled=True)


def test_named_successor_replays_without_indices_and_split_name_fails():
    source = _box().tag_face("port", near=(2, 1, 1))
    pieces = source.partition(normal="x", position=1)
    assert len(pieces) == 2
    named = [part for part in pieces if hasattr(part, "_topology_inputs")]
    assert len(named) == 1
    piece = named[0]
    assert piece.face("port").area == pytest.approx(4)
    restored = from_recipe(json.loads(json.dumps(to_recipe(piece))))
    assert restored.face("port").centroid == pytest.approx(piece.face("port").centroid)
    assert restored.volume() == pytest.approx(piece.volume())
    split_name = _box().tag_face("side", near=(1, 0, 1))
    with pytest.raises(geo.TopologyEvolutionError, match="split"):
        split_name.partition(normal="x", position=1)


def test_deliberate_face_set_retains_split_successors_on_both_regions():
    source = _box().tag_faces("wall", normal="y")
    pieces = source.partition(normal="x", position=1)
    assert len(pieces) == 2
    assert [len(piece.faces("wall")) for piece in pieces] == [1, 1]
    assert sorted(next(iter(piece.faces("wall"))).area for piece in pieces) == pytest.approx([2, 2])
    restored = from_recipe(json.loads(json.dumps(to_recipe(pieces[0]))))
    assert len(restored.faces("wall")) == 1
    assert next(iter(restored.faces("wall"))).area == pytest.approx(2)


def test_project_round_trip_of_named_partition_region(tmp_path):
    import numpy as np

    import magnelio as mio
    from magnelio.io.project import ProjectStore
    from magnelio.mesh import GridLines

    source = _box().tag_face("port", near=(2, 1, 1))
    piece = next(
        part
        for part in source.partition(normal="x", position=1)
        if hasattr(part, "_topology_inputs")
    )
    lines = np.linspace(0, 2, 3)
    grid = GridLines(x=lines, y=lines, z=lines)
    ProjectStore.create(tmp_path / "partition", mio.Mesh.from_grid(grid), geometry=[piece])
    (loaded,) = mio.open_project(tmp_path / "partition").geometry
    assert loaded.face("port").area == pytest.approx(4)
    assert loaded.volume() == pytest.approx(4)


@pytest.mark.parametrize("length", [1e-9, 1e-3, 1, 1e3])
def test_scale_covariance(length):
    source = geo.Brick(origin=(0, 0, 0), size=(2 * length,) * 3, material="pec")
    pieces = source.partition(normal="x", position=length)
    assert sorted(part.volume() / length**3 for part in pieces) == pytest.approx([4, 4])
    (profile,) = source.section(normal="x", position=length, filled=True)
    assert profile.area / length**2 == pytest.approx(4)


def test_methods_recipe_and_public_tutorial_execute():
    import matplotlib

    matplotlib.use("Agg")
    root = Path(__file__).resolve().parents[2]
    source = (root / "docs/methods/geometry.md").read_text()
    prose = source.split("## Partition and section", 1)[1].split("\n## ", 1)[0]
    namespace = {}
    for block in re.findall(r"```python\n(.*?)```", prose, re.S):
        exec(block, namespace)
    assert namespace["ring"].area == pytest.approx(3 * math.pi * 1e-6)
    runpy.run_path(str(root / "examples/tutorials/plot_24_partition_section.py"))
