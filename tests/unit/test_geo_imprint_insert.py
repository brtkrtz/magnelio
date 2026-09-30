"""Directed imprints and explicit material-overlap precedence."""

import json
import math
import re
import runpy
from pathlib import Path

import pytest

from magnelio import GeometryModel, geo
from magnelio.geo._occ_backend import check_pairwise_overlaps
from magnelio.geo._scaling import model_scale
from magnelio.geo._topology_store import from_recipe, to_recipe

pytest.importorskip("OCC.Core.BRepFeat")


def _box(origin=(0, 0, 0), size=(2, 2, 2), material="pec", name=None):
    return geo.Brick(origin=origin, size=size, material=material, name=name)


def test_imprint_is_directed_and_preserves_both_volumes_and_materials():
    receiver = _box()
    cutter = _box((1, 1, -1), (2, 2, 2), "air")
    result = receiver.imprint(cutter)
    assert result is not receiver
    assert isinstance(result, geo.Solid)
    assert result.material is receiver.material
    assert result.volume() == pytest.approx(receiver.volume())
    assert len(result.faces()) > len(receiver.faces())
    assert len(cutter.faces()) == 6
    assert cutter.volume() == pytest.approx(8)


def test_imprint_accepts_sheet_and_no_intersection():
    receiver = _box()
    sheet = geo.Profile.rectangle((1, 1, 1), (4, 4))
    result = receiver.imprint(sheet)
    assert result.volume() == pytest.approx(8)
    assert len(result.faces()) > 6
    far = receiver.imprint(_box((4, 0, 0), (1, 1, 1), None))
    assert far is not receiver
    assert far.volume() == pytest.approx(8)
    assert len(far.faces()) == 6


def test_imprint_named_receiver_history_and_exact_replay():
    receiver = _box().tag_face("cap", near=(1, 1, 2))
    cutter = _box((1, 1, -1), (2, 2, 2), None)
    result = receiver.imprint(cutter)
    assert result.face("cap").area == pytest.approx(4)
    assert receiver.face("cap").area == pytest.approx(4)
    restored = from_recipe(json.loads(json.dumps(to_recipe(result))))
    assert restored.face("cap").area == pytest.approx(4)
    assert restored.volume() == pytest.approx(8)


def test_imprint_singular_split_fails_but_deliberate_set_survives():
    cutter = _box((1, 1, -1), (2, 2, 2), None)
    receiver = _box().tag_face("side", near=(2, 1, 1))
    with pytest.raises(geo.TopologyEvolutionError, match="imprint.*split"):
        receiver.imprint(cutter)
    chosen = _box().tag_faces("sides", normal="x")
    result = chosen.imprint(cutter)
    assert len(result.faces("sides")) > 1


def test_insert_trims_host_and_retains_physical_insert():
    host = _box()
    dielectric = _box((0.5, 0.5, 0.5), (1, 1, 1), "air")
    assembly = geo.insert(host, dielectric, priorities=(0, 1))
    regions = tuple(assembly.members())
    assert [region.volume() for region in regions] == pytest.approx([7, 1])
    assert [region.material for region in regions] == [host.material, dielectric.material]
    assert host.volume() == pytest.approx(8)
    assert dielectric.volume() == pytest.approx(1)
    assert check_pairwise_overlaps(list(regions), scale=model_scale(regions)) == []
    model = GeometryModel().add(assembly)
    model.validate()
    assert len(model.shapes) == 2


def test_insert_material_regions_reach_the_mesh():
    import numpy as np

    from magnelio import Material, Mesh, MeshControl

    host = _box(size=(2e-3,) * 3)
    dielectric = _box(
        (0.5e-3,) * 3,
        (1e-3,) * 3,
        Material(name="dielectric", epsilon=(2.5,) * 3),
    )
    assembly = geo.insert(host, dielectric, priorities=(0, 1))
    mesh = Mesh.from_geometry(
        GeometryModel().add(assembly),
        MeshControl(min_nodes_per_wavelength=4, max_cell_size=0.5e-3),
        f_max=10e9,
    )
    ids = {material.name: index for index, material in mesh.material_library.items()}
    assert np.any(mesh.material_id == ids["PEC"])
    assert np.any(mesh.material_id == ids["dielectric"])
    reversed_mesh = Mesh.from_geometry(
        GeometryModel().add(tuple(reversed(tuple(assembly.members())))),
        MeshControl(min_nodes_per_wavelength=4, max_cell_size=0.5e-3),
        f_max=10e9,
    )
    for axis in "xyz":
        assert np.array_equal(getattr(mesh.grid, axis), getattr(reversed_mesh.grid, axis))
    material_names = np.vectorize(lambda item: mesh.material_library[int(item)].name)(
        mesh.material_id
    )
    reversed_names = np.vectorize(lambda item: reversed_mesh.material_library[int(item)].name)(
        reversed_mesh.material_id
    )
    assert np.array_equal(material_names, reversed_names)


def test_reverse_precedence_trims_other_operand():
    host = _box()
    insert = _box((1, 1, 1), (2, 2, 2), "air")
    regions = tuple(geo.insert(host, insert, priorities=(2, -3)).members())
    assert [region.volume() for region in regions] == pytest.approx([8, 7])
    assert check_pairwise_overlaps(list(regions)) == []


def test_three_body_precedence_is_independent_of_model_add_order():
    bodies = (
        _box((0, 0, 0), (3, 3, 3), "pec", "host"),
        _box((1, 1, 1), (2, 2, 2), "air", "middle"),
        _box((0, 0, 0), (2, 2, 2), "vacuum", "winner"),
    )
    assembly = geo.insert(*bodies, priorities=(0, 1, 2))
    by_name = {region.name: region for region in assembly.members()}
    assert {name: region.volume() for name, region in by_name.items()} == pytest.approx(
        {"host": 12, "middle": 7, "winner": 8}
    )
    reverse = geo.insert(*reversed(bodies), priorities=(2, 1, 0))
    reverse_by_name = {region.name: region for region in reverse.members()}
    assert {name: region.volume() for name, region in reverse_by_name.items()} == pytest.approx(
        {"host": 12, "middle": 7, "winner": 8}
    )
    for group in (assembly, reverse):
        GeometryModel().add(group).validate()
        assert check_pairwise_overlaps(list(group.members())) == []


def test_void_removes_material_from_all_regions_and_has_no_result():
    host = _box()
    dielectric = _box((0.5, 0.5, 0.5), (1, 1, 1), "air")
    void = _box((0.75, 0.75, 0.75), (0.5, 0.5, 0.5), None)
    regions = tuple(geo.insert(host, dielectric, priorities=(0, 1), voids=(void,)).members())
    assert [region.volume() for region in regions] == pytest.approx([7, 0.875])
    assert len(regions) == 2
    assert check_pairwise_overlaps(list(regions)) == []
    GeometryModel().add(regions).validate()


def test_same_material_overlap_is_trimmed_and_contact_stays_separate():
    first = _box()
    overlapping = _box((1, 0, 0), (2, 2, 2))
    regions = tuple(geo.insert(first, overlapping, priorities=(0, 1)).members())
    assert [region.volume() for region in regions] == pytest.approx([4, 8])
    touching = _box((2, 0, 0), (1, 2, 2), "air")
    regions = tuple(geo.insert(first, touching, priorities=(0, 1)).members())
    assert len(regions) == 2
    assert regions[0].volume() == pytest.approx(8)
    assert regions[1].volume() == pytest.approx(4)
    tangent = geo.Sphere(center=(3, 1, 1), radius=1, material="air")
    regions = tuple(geo.insert(first, tangent, priorities=(0, 1)).members())
    assert len(regions) == 2
    assert regions[0].volume() == pytest.approx(8)


def test_representable_small_overlap_is_not_ignored_by_material_precedence():
    host = _box(size=(1, 1, 1))
    width = 5e-5
    insert = _box((1 - width,) * 3, (1, 1, 1), "air")
    host_region, winner = geo.insert(host, insert, priorities=(0, 1)).members()
    assert host.volume() - host_region.volume() == pytest.approx(width**3, rel=2e-3)
    assert winner.volume() == pytest.approx(1)
    assert check_pairwise_overlaps([host_region, winner], tolerance=0.0) == []


def test_equal_rank_overlap_fails_but_disjoint_equal_ranks_are_valid():
    first = _box()
    overlapping = _box((1, 0, 0), (2, 2, 2), "air")
    with pytest.raises(ValueError, match="equal priority"):
        geo.insert(first, overlapping, priorities=(1, 1))
    far = _box((4, 0, 0), (1, 1, 1), "air")
    assert len(tuple(geo.insert(first, far, priorities=(1, 1)).members())) == 2


def test_full_consumption_omits_unnamed_loser_and_preserves_winner():
    small = _box((0.5, 0.5, 0.5), (1, 1, 1), "air")
    large = _box()
    (winner,) = geo.insert(small, large, priorities=(0, 1)).members()
    assert winner is large
    assert winner.volume() == pytest.approx(8)
    void = _box((0, 0, 0), (2, 2, 2), None)
    assert tuple(geo.insert(large, priorities=(0,), voids=(void,)).members()) == ()


def test_insert_preserves_only_its_source_names_and_replays():
    host = _box().tag_face("outer", near=(0, 1, 1))
    inserted = _box((0.5, 0.5, 0.5), (1, 1, 1), "air").tag_face("insert_cap", near=(1, 1, 1.5))
    host_result, inserted_result = geo.insert(host, inserted, priorities=(0, 1)).members()
    assert host_result.face("outer").area == pytest.approx(4)
    with pytest.raises(geo.TopologySelectionError):
        host_result.face("insert_cap")
    assert inserted_result.face("insert_cap").area == pytest.approx(1)
    restored = from_recipe(json.loads(json.dumps(to_recipe(host_result))))
    assert restored.face("outer").area == pytest.approx(4)
    assert restored.volume() == pytest.approx(7)


def test_insert_named_deleted_face_fails_eagerly():
    host = _box().tag_face("inside", near=(0, 1, 1))
    winning = _box((-1, -1, -1), (2, 4, 4), "air")
    with pytest.raises(geo.TopologyEvolutionError, match="insert.*deleted"):
        geo.insert(host, winning, priorities=(0, 1))
    split_host = _box().tag_face("wall", near=(0, 1, 1))
    slot = _box((-1, 0.5, -1), (2, 1, 4), "air")
    with pytest.raises(geo.TopologyEvolutionError, match="insert.*split"):
        geo.insert(split_host, slot, priorities=(0, 1))


def test_project_round_trip_of_named_imprinted_insert_region(tmp_path):
    import numpy as np

    import magnelio as mio
    from magnelio.io.project import ProjectStore
    from magnelio.mesh import GridLines

    host = _box().tag_face("cap", near=(1, 1, 2))
    tool = _box((1, 1, -1), (2, 2, 2), None)
    imprinted = host.imprint(tool)
    dielectric = _box((0.5, 0.5, 0.5), (1, 1, 1), "air")
    assembly = geo.insert(imprinted, dielectric, priorities=(0, 1))
    lines = np.linspace(0, 2, 3)
    grid = GridLines(x=lines, y=lines, z=lines)
    ProjectStore.create(
        tmp_path / "insert", mio.Mesh.from_grid(grid), geometry=list(assembly.members())
    )
    loaded_host, loaded_dielectric = mio.open_project(tmp_path / "insert").geometry
    assert loaded_host.face("cap").area == pytest.approx(4)
    assert loaded_host.volume() == pytest.approx(7)
    assert loaded_dielectric.volume() == pytest.approx(1)


@pytest.mark.parametrize("length", [1e-9, 1, 1e3])
def test_imprint_and_insert_are_scale_covariant(length):
    source = _box(size=(2 * length,) * 3)
    cutter = _box((length, length, -length), (2 * length,) * 3, "air")
    imprinted = source.imprint(cutter)
    assert imprinted.volume() / length**3 == pytest.approx(8)
    assert len(imprinted.faces()) > 6
    regions = tuple(geo.insert(source, cutter, priorities=(0, 1)).members())
    assert [region.volume() / length**3 for region in regions] == pytest.approx([7, 8])
    assert check_pairwise_overlaps(list(regions), scale=model_scale(regions)) == []


@pytest.mark.parametrize("length", [1e-9, 1e3])
def test_named_imprint_and_insert_replay_at_model_scale(length):
    receiver = _box(size=(2 * length,) * 3).tag_face("cap", near=(length, length, 2 * length))
    cutter = _box((length, length, -length), (2 * length,) * 3, "air")
    imprinted = receiver.imprint(cutter)
    restored_imprint = from_recipe(json.loads(json.dumps(to_recipe(imprinted))))
    assert restored_imprint.face("cap").area / length**2 == pytest.approx(4)
    host_region = next(geo.insert(imprinted, cutter, priorities=(0, 1)).members())
    restored_host = from_recipe(json.loads(json.dumps(to_recipe(host_region))))
    assert restored_host.face("cap").area / length**2 == pytest.approx(4)
    assert restored_host.volume() / length**3 == pytest.approx(7)


def test_wrong_categories_and_material_roles_fail_at_call():
    body = _box()
    sheet = geo.Profile.rectangle((0, 0, 0), (1, 1))
    with pytest.raises(TypeError, match="cutter"):
        body.imprint(geo.Curve.line((0, 0, 0), (1, 0, 0)))
    with pytest.raises(ValueError, match="at least one"):
        geo.insert(priorities=())
    with pytest.raises(TypeError, match="body"):
        geo.insert(sheet, priorities=(0,))
    with pytest.raises(ValueError, match="needs a material"):
        geo.insert(_box(material=None), priorities=(0,))
    with pytest.raises(ValueError, match="must not carry a material"):
        geo.insert(body, priorities=(0,), voids=(body,))
    with pytest.raises(ValueError, match="one integer priority"):
        geo.insert(body, priorities=(True,))


def test_methods_recipe_and_tutorial_execute():
    import matplotlib

    matplotlib.use("Agg")
    root = Path(__file__).resolve().parents[2]
    source = (root / "docs/methods/geometry.md").read_text()
    prose = source.split("## Imprint and insert", 1)[1].split("\n## ", 1)[0]
    namespace = {}
    for block in re.findall(r"```python\n(.*?)```", prose, re.S):
        exec(block, namespace)
    assert math.isclose(sum(x.volume() for x in namespace["assembly"].members()), 8e-9)
    runpy.run_path(str(root / "examples/tutorials/plot_25_imprint_insert.py"))
