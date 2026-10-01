"""Reusable placement of mixed, named components (WP6.10)."""

import json
import re
from pathlib import Path

import pytest

from magnelio import geo
from magnelio.geo._topology_store import from_recipe, to_recipe


def _component():
    shell = (
        geo.Brick(size=(2, 1, 1), material="pec", name="shell")
        .tag_face("contact", normal="z")
        .tag_faces("sides", surface_type="plane")
    )
    core = geo.Brick(origin=(0.5, 0.25, 0), size=(1, 0.5, 1), material="air", name="core")
    outline = geo.Curve.line((0, 0, 0), (2, 0, 0), name="datum")
    aperture = geo.Profile.rectangle(center=(1, 0.5, 0), size=(2, 1), name="aperture")
    return geo.Group(geo.Group(shell, core, name="layers"), outline, aperture, name="component")


def test_reusable_placements_keep_mixed_members_materials_names_and_owners():
    source = _component()
    placements = (
        geo.Transform.identity(),
        geo.Translation((5, 0, 0)) @ geo.Rotation("z", 90),
        geo.Translation((10, 0, 0)) @ geo.Mirror("z") @ geo.Scale(2),
    )
    copies = [placement @ source for placement in placements]
    source_members = list(source.members())
    original_contact = source_members[0].face("contact")

    for scale, placement, copy in zip((1, 1, 2), placements, copies):
        assert copy is not source and copy.name == "component"
        assert copy.shapes[0].name == "layers"
        members = list(copy.members())
        assert [member.name for member in members] == ["shell", "core", "datum", "aperture"]
        assert [getattr(member, "material", None) for member in members] == [
            getattr(member, "material", None) for member in source_members
        ]
        assert members[0] is not source_members[0]
        assert members[0].volume() == pytest.approx(2 * scale**3)
        assert members[0].face("contact").area == pytest.approx(2 * scale**2)
        assert members[0].face("contact").owner is members[0]
        assert members[0].face("contact").centroid == pytest.approx(
            placement.point(original_contact.centroid)
        )
        assert members[0].face("contact").normal == pytest.approx((0, 0, -1 if scale == 2 else 1))
        assert len(members[0].faces("sides")) == 6
        assert all(face.owner is members[0] for face in members[0].faces("sides"))
        assert members[2].length == pytest.approx(2 * scale)
        assert members[3].area == pytest.approx(2 * scale**2)
        restored = from_recipe(json.loads(json.dumps(to_recipe(members[0]))))
        assert restored.face("contact").centroid == pytest.approx(
            members[0].face("contact").centroid
        )
        assert restored.material == members[0].material
        assert len(restored.faces("sides")) == 6

    assert original_contact.owner is source_members[0]
    assert original_contact.centroid == pytest.approx((1, 0.5, 1))
    assert source_members[0].volume() == pytest.approx(2)
    assert copies[1].shapes[0].shapes[0].face("contact").owner is not copies[2].shapes[0].shapes[0]


def test_nonuniform_and_sheared_placement_are_rejected():
    for linear in (((2, 0, 0), (0, 1, 0), (0, 0, 1)), ((1, 0.2, 0), (0, 1, 0), (0, 0, 1))):
        matrix = tuple((*row, 0) for row in linear) + ((0, 0, 0, 1),)
        with pytest.raises(ValueError, match="uniform scale|perpendicular"):
            geo.Transform(matrix)


def test_documented_component_recipe_executes():
    guide = (Path(__file__).resolve().parents[2] / "docs/methods/geometry.md").read_text()
    (recipe,) = [
        block
        for block in re.findall(r"```python\n(.*?)```", guide, re.DOTALL)
        if block.startswith("shell = geo.Brick(size=(2e-3")
    ]
    namespace = {"geo": geo}
    exec(compile(recipe, "component placement documentation", "exec"), namespace)
    assert len(namespace["copies"]) == len(namespace["contacts"]) == 3
    assert all(
        contact.owner is next(copy.members())
        for contact, copy in zip(namespace["contacts"], namespace["copies"])
    )
