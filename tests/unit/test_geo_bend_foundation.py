"""Freeform neutral-surface bends of existing bodies and sheets."""

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

import magnelio as mio
from magnelio import geo
from magnelio.geo._topology_store import from_recipe, to_recipe
from magnelio.io.project import ProjectStore
from magnelio.mesh import GridLines

pytest.importorskip("OCC.Core.BRepFeat")


def _cylindrical(scale=1, *, angle=math.pi / 2, width=0.4):
    length = angle * scale
    half = width * scale / 2
    target = geo.Surface.parametric(
        lambda u, v: (
            scale * np.sin(u / scale),
            v,
            scale * (1 - np.cos(u / scale)),
        ),
        u=(0, length),
        v=(-half, half),
        samples=(65, 9),
    )
    bend = geo.Bend(
        target,
        origin=(0, 0, 0),
        along="x",
        across="y",
        u=(0, length),
        v=(-half, half),
        max_strain=0.01,
    )
    return bend, length


def _block(length, scale=1, *, before=0, after=0, material="pec"):
    return geo.Brick(
        origin=(-before, -0.2 * scale, -0.05 * scale),
        size=(length + before + after, 0.4 * scale, 0.1 * scale),
        material=material,
    )


@pytest.mark.parametrize("scale", [1e-6, 1, 1e3])
def test_cylindrical_bend_volume_bounds_and_scale(scale):
    bend, length = _cylindrical(scale)
    source = _block(length, scale)
    result = bend @ source
    expected = 0.4 * 0.1 * math.pi / 2 * scale**3
    assert isinstance(result, geo.Solid)
    assert result is not source
    assert result.material == source.material
    assert result.volume() == pytest.approx(expected, rel=1e-5)
    assert result.bounding_box()[1] == pytest.approx(
        (1.05 * scale, 0.2 * scale, scale), abs=1e-6 * scale
    )
    assert source.volume() == pytest.approx(length * 0.4 * 0.1 * scale**2)


def test_finite_bend_preserves_preceding_body_and_carries_rigid_end():
    bend, length = _cylindrical()
    source = _block(length, before=0.5, after=0.5)
    result = bend @ source
    expected = 0.4 * 0.1 * (length + 1)
    assert result.volume() == pytest.approx(expected, rel=1e-5)
    low, high = result.bounding_box()
    assert low[0] == pytest.approx(-0.5)
    assert low[2] == pytest.approx(-0.05, abs=1e-6)
    assert high[2] == pytest.approx(1.5, abs=1e-6)


def test_common_mapping_stretches_outer_layer_and_compresses_inner_layer():
    bend, length = _cylindrical()
    outer = geo.Brick(origin=(0, -0.2, -0.1), size=(length, 0.4, 0.1), material="pec")
    inner = geo.Brick(origin=(0, -0.2, 0), size=(length, 0.4, 0.1), material="air")
    group = bend @ geo.Group(outer, inner, name="layers")
    exterior, interior = tuple(group.members())
    angle = math.pi / 2
    assert group.name == "layers"
    assert exterior.material == outer.material
    assert interior.material == inner.material
    assert exterior.volume() == pytest.approx(0.4 * angle * (1.1**2 - 1**2) / 2, rel=1e-5)
    assert interior.volume() == pytest.approx(0.4 * angle * (1**2 - 0.9**2) / 2, rel=1e-5)
    assert exterior.volume() > interior.volume()


def test_bent_material_region_reaches_the_mesh():
    bend, length = _cylindrical(1e-2)
    body = bend @ _block(length, 1e-2)
    mesh = mio.Mesh.from_geometry(
        geo.GeometryModel().add(body),
        mio.MeshControl(min_nodes_per_wavelength=4, max_cell_size=0.5e-3),
        f_max=10e9,
    )
    pec_ids = [index for index, material in mesh.material_library.items() if material.name == "PEC"]
    assert pec_ids and np.any(mesh.material_id == pec_ids[0])


def test_opening_survives_bend_and_source_is_unchanged():
    bend, length = _cylindrical()
    source = _block(length) - geo.Cylinder(origin=(length / 2, 0, -0.1), radius=0.05, height=0.2)
    result = bend @ source
    assert result.volume() == pytest.approx(source.volume(), rel=1e-5)
    assert len(result.faces()) >= len(source.faces())
    assert result.bounding_box()[1][2] == pytest.approx(1, abs=1e-6)


def test_imported_curved_body_uses_the_same_freeform_map():
    bend, length = _cylindrical()
    source = _block(length) - geo.Cylinder(origin=(length / 2, 0, -0.1), radius=0.05, height=0.2)
    imported = geo.ImportedSolid(source._occ_shape(1), material="pec", name="imported housing")
    result = bend @ imported
    assert result.material == imported.material
    assert result.name == imported.name
    assert result.volume() == pytest.approx((bend @ source).volume(), rel=1e-5)


def test_curved_neutral_surface_accepts_bounded_double_curvature():
    target = geo.Surface.parametric(
        lambda u, v: (
            u,
            v,
            0.02 * np.sin(np.pi * u) ** 2 * np.cos(np.pi * v / 0.6),
        ),
        u=(0, 1),
        v=(-0.3, 0.3),
        samples=(65, 33),
    )
    bend = geo.Bend(
        target,
        origin=(0, 0, 0),
        along="x",
        across="y",
        u=(0, 1),
        v=(-0.3, 0.3),
        max_strain=0.1,
        tolerance=1e-4,
    )
    source = geo.Brick(origin=(0, -0.25, -0.02), size=(1, 0.5, 0.04), material="pec")
    result = bend @ source
    assert result.volume() > source.volume()
    assert result.volume() < 1.01 * source.volume()
    assert result.bounding_box()[1][2] > source.bounding_box()[1][2]


def test_sheet_bends_into_independent_curved_sheet():
    bend, length = _cylindrical()
    source = geo.Profile.rectangle((length / 2, 0, 0), (length, 0.3), normal="z", x_direction="x")
    result = bend @ source
    assert isinstance(result, geo.Sheet)
    assert not isinstance(result, geo.Profile)
    assert result.bounding_box()[1][2] == pytest.approx(1, abs=1e-6)
    assert source.bounding_box()[1][2] == 0
    assert result.thickened(0.02, material="pec").volume() > 0


def test_sheet_crossing_interval_is_thickened_before_bending():
    bend, length = _cylindrical()
    source = geo.Profile.rectangle(
        (length / 2, 0, 0), (length + 1, 0.3), normal="z", x_direction="x"
    )
    bent_sheet = bend @ source
    with pytest.raises(ValueError, match="Thicken the source Sheet first"):
        bent_sheet.thickened(0.02, material="pec")
    assert (bend @ source.thickened(0.02, material="pec")).volume() > 0


def test_oblique_placement_preserves_volume_and_world_frame():
    bend, length = _cylindrical()
    placement = geo.Translation((2, 3, 4)) @ geo.Rotation((1, 1, 1), 37)
    rotation = np.asarray(placement.matrix)[:3, :3]
    placed_bend = geo.Bend(
        placement @ bend.target,
        origin=placement.point((0, 0, 0)),
        along=rotation @ (1, 0, 0),
        across=rotation @ (0, 1, 0),
        u=bend.u,
        v=bend.v,
        max_strain=bend.max_strain,
    )
    straight = placement @ _block(length)
    assert (placed_bend @ straight).volume() == pytest.approx(
        (bend @ _block(length)).volume(), rel=1e-5
    )


def test_named_end_face_survives_and_replays_through_project(tmp_path):
    bend, length = _cylindrical()
    source = _block(length).tagged_face("end", normal="x")
    result = bend @ source
    assert result.face("end").area == pytest.approx(0.04)
    restored = from_recipe(json.loads(json.dumps(to_recipe(result))))
    assert restored.face("end").area == pytest.approx(0.04)
    grid = GridLines(x=np.linspace(0, 2, 3), y=np.linspace(0, 1, 3), z=np.linspace(0, 2, 3))
    ProjectStore.create(tmp_path / "bend", mio.Mesh.from_grid(grid), geometry=[result])
    (loaded,) = mio.open_project(tmp_path / "bend").geometry
    assert loaded.face("end").area == pytest.approx(0.04)
    assert loaded.volume() == pytest.approx(result.volume())


def test_singular_face_name_fails_when_interval_seam_splits_it():
    bend, length = _cylindrical()
    source = _block(length, before=0.5, after=0.5).tagged_face("side", near=(0.5, 0.2, 0))
    with pytest.raises(geo.TopologyEvolutionError, match="split"):
        bend @ source
    deliberate = _block(length, before=0.5, after=0.5).tagged_faces("side", near=(0.5, 0.2, 0))
    assert len((bend @ deliberate).faces("side")) == 3


def test_maximum_neutral_strain_is_enforced():
    target = geo.Surface.parametric(
        lambda u, v: (u, v, 0.02 * np.sin(np.pi * u) ** 2 * np.cos(np.pi * v / 0.6)),
        u=(0, 1),
        v=(-0.3, 0.3),
        samples=(65, 33),
    )
    strict = geo.Bend(
        target,
        origin=(0, 0, 0),
        along="x",
        across="y",
        u=(0, 1),
        v=(-0.3, 0.3),
        max_strain=0.001,
        tolerance=1e-4,
    )
    with pytest.raises(ValueError, match="strain"):
        (strict @ geo.Brick(origin=(0, -0.25, -0.02), size=(1, 0.5, 0.04))).volume()


def test_normal_extension_rejects_folded_thickness():
    bend, length = _cylindrical(width=4)
    source = geo.Brick(origin=(0, -0.1, 0), size=(length, 0.2, 1.2))
    with pytest.raises(ValueError, match="folds"):
        (bend @ source).volume()


def test_target_requires_complete_transverse_coverage():
    bend, length = _cylindrical(width=0.2)
    with pytest.raises(ValueError, match="transverse extent"):
        (bend @ _block(length)).volume()


def test_invalid_input_and_unattainable_tolerance_report_errors():
    bend, length = _cylindrical()
    with pytest.raises(ValueError, match="perpendicular"):
        geo.Bend(
            bend.target, origin=(0, 0, 0), along="x", across="x", u=bend.u, v=bend.v, max_strain=0.1
        )
    with pytest.raises(TypeError, match="Solid, Sheet or Group"):
        bend @ geo.Curve.line((0, 0, 0), (1, 0, 0))
    strict = geo.Bend(
        bend.target,
        origin=(0, 0, 0),
        along="x",
        across="y",
        u=bend.u,
        v=bend.v,
        max_strain=0.01,
        tolerance=1e-12,
    )
    with pytest.raises(ValueError, match="tolerance"):
        (strict @ _block(length)).volume()


def test_methods_recipe_and_public_tutorial_execute():
    import matplotlib

    matplotlib.use("Agg")
    root = Path(__file__).resolve().parents[2]
    text = (root / "docs/methods/geometry.md").read_text()
    section = text.split("## Bend existing bodies and sheets", 1)[1].split("\n## ", 1)[0]
    namespace = {}
    for block in re.findall(r"```python\n(.*?)```", section, re.S):
        exec(block, namespace)
    assert namespace["curved_layer"].volume() > namespace["layer"].volume()
