"""Dimensional geometry and affine value algebra (geometry foundation WP1)."""

import math
from dataclasses import FrozenInstanceError

import pytest

from magnelio import geo
from magnelio.materials import Material


def _assert_box(actual, expected):
    for actual_corner, expected_corner in zip(actual, expected):
        assert actual_corner == pytest.approx(expected_corner, abs=1e-12)


def test_public_dimensional_hierarchy_and_group_boundary():
    assert issubclass(geo.Curve, geo.Shape)
    assert issubclass(geo.Profile, geo.Sheet)
    assert issubclass(geo.Surface, geo.Sheet)
    assert issubclass(geo.Solid, geo.Shape)
    assert issubclass(geo.Brick, geo.Solid)
    assert issubclass(geo.Union, geo.Solid)
    assert not issubclass(geo.Group, geo.Shape)


def test_transform_values_are_immutable_and_validate_similarity_matrices():
    placement = geo.Translation((1.0, 2.0, 3.0))
    with pytest.raises(FrozenInstanceError):
        placement.matrix = geo.Transform.identity().matrix
    with pytest.raises(FrozenInstanceError):
        placement.label = "mutable metadata"
    with pytest.raises(ValueError, match="uniform scale"):
        geo.Transform(
            (
                (2.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        )


def test_composition_uses_column_vectors_and_applies_rightmost_first():
    placement = geo.Translation((10.0, 0.0, 0.0)) @ geo.Rotation("z", 90.0)
    assert placement.point((2.0, 0.0, 0.0)) == pytest.approx((10.0, 2.0, 0.0))
    assert (geo.Transform.identity() @ placement).matrix == placement.matrix


def test_named_methods_and_transform_values_share_the_backend():
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    body = geo.Brick(origin=(1.0, 0.0, 0.0), size=(1.0, 2.0, 3.0))

    named = body.rotated("z", 90.0).translated((10.0, 0.0, 0.0))
    algebraic = geo.Translation((10.0, 0.0, 0.0)) @ geo.Rotation("z", 90.0) @ body

    _assert_box(named.bounding_box(), algebraic.bounding_box())
    assert named.volume() == pytest.approx(algebraic.volume())


def test_every_standalone_category_is_transformable_and_preserved():
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    transform = geo.Translation((2.0, 0.0, 0.0))
    curve = geo.Curve.polyline(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), name="route")
    profile = geo.Profile.polygon(
        [(u, v, 0.0) for u, v in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))], name="section"
    )
    surface = geo.Surface(
        (((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), ((1.0, 0.0, 0.2), (1.0, 1.0, 0.2))),
        name="dish",
    )
    solid = geo.Brick(name="body")

    placed_curve = transform @ curve
    placed_profile = transform @ profile
    placed_surface = transform @ surface
    placed_solid = transform @ solid

    assert isinstance(placed_curve, geo.Curve)
    assert isinstance(placed_profile, geo.Profile)
    assert isinstance(placed_surface, geo.Surface)
    assert not isinstance(placed_surface, geo.Profile)
    assert isinstance(placed_solid, geo.Solid)
    assert [placed_curve.name, placed_profile.name, placed_surface.name, placed_solid.name] == [
        "route",
        "section",
        "dish",
        "body",
    ]


def test_reflection_has_the_expected_orientation_and_positive_solid_volume():
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    body = geo.Brick(origin=(1.0, 2.0, 3.0), size=(2.0, 3.0, 4.0))
    image = geo.Mirror((1.0, 0.0, 0.0), position=0.5) @ body

    _assert_box(image.bounding_box(), ((-2.0, 2.0, 3.0), (0.0, 5.0, 7.0)))
    assert image.volume() == pytest.approx(body.volume())


def test_transform_preserves_material_name_and_group_structure():
    material_a = Material.pec()
    material_b = Material.air()
    group = geo.Group(
        geo.Brick(material=material_a, name="metal"),
        geo.Sphere(material=material_b, name="dielectric"),
        name="assembly",
    )

    placed = geo.Scale(2.0) @ group

    assert placed.name == "assembly"
    assert not isinstance(placed, geo.Shape)
    members = list(placed.members())
    assert [member.name for member in members] == ["metal", "dielectric"]
    assert [member.material for member in members] == [material_a, material_b]


@pytest.mark.parametrize("operation", [geo.Union, geo.Intersection, geo.Difference])
def test_boolean_constructors_reject_non_solids_by_category(operation):
    solid = geo.Brick()
    profile = geo.Profile.polygon([(u, v, 0.0) for u, v in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))])

    with pytest.raises(TypeError, match="accepts Solid operands"):
        operation(solid, profile)


def test_solid_operators_reject_curves_and_geometry_cannot_right_apply_transform():
    solid = geo.Brick()
    curve = geo.Curve.polyline(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))

    with pytest.raises(TypeError, match="accepts Solid operands"):
        solid + curve
    with pytest.raises(TypeError, match="unsupported operand"):
        solid @ geo.Translation((1.0, 0.0, 0.0))


def test_requested_eightfold_rotation_includes_original_and_fuses_solids():
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    body = geo.Sphere(center=(2, 0, 0), radius=0.1, material="pec", name="post")
    result = body.rotated(axis="z", angle_deg=45, repeat=7, copy=True, unite=True)
    assert isinstance(result, geo.Union)
    assert len(result.shapes) == 8 and result.shapes[0] is body
    for i, member in enumerate(result.shapes):
        lo, hi = member.bounding_box()
        center = tuple((a + b) / 2 for a, b in zip(lo, hi))
        angle = math.radians(45 * i)
        assert center == pytest.approx((2 * math.cos(angle), 2 * math.sin(angle), 0), abs=1e-9)
        assert member.name == "post" and member.material is body.material
    assert result.volume() == pytest.approx(8 * body.volume(), rel=1e-10)
    assert body.center == (2, 0, 0)


@pytest.mark.parametrize("mode,result_type", [("group", geo.Group), ("unite", geo.Union)])
@pytest.mark.parametrize("method,args", [("translated", ((1, 0, 0),)), ("rotated", ("z", 45))])
def test_explicit_aggregation_is_honoured_for_one_copy(mode, result_type, method, args):
    result = getattr(geo.Brick(), method)(*args, **{mode: True})
    assert isinstance(result, result_type)
    assert len(result.shapes) == 1


def test_repeated_translation_uses_original_placement_and_preserves_profile_holes():
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    profile = (
        geo.Profile.from_wires(
            geo.Curve.circle((0, 0, 0), 2), [geo.Curve.circle((0, 0, 0), 1)], name="ring"
        )
        .rotated("y", 30)
        .translated((10, 0, 0))
    )
    copies = profile.translated((0, 0, 5), repeat=3, copy=True)
    assert copies[0] is profile and len(copies) == 4
    for i, member in enumerate(copies):
        assert isinstance(member, geo.Profile) and member.name == "ring"
        assert member.area == pytest.approx(3 * math.pi)
        assert len(member.boundary()) == 2
        _assert_box(
            member.bounding_box(),
            tuple(
                tuple(x + (5 * i if j == 2 else 0) for j, x in enumerate(p))
                for p in profile.bounding_box()
            ),
        )


@pytest.mark.parametrize("method,args", [("translated", ((1, 0, 0),)), ("rotated", ("z", 45))])
@pytest.mark.parametrize(
    "repeat,error", [(0, ValueError), (-1, ValueError), (2.5, TypeError), (True, TypeError)]
)
def test_invalid_repetition_count_rejected(method, args, repeat, error):
    with pytest.raises(error, match="repeat"):
        getattr(geo.Brick(), method)(*args, repeat=repeat)


@pytest.mark.parametrize(
    "method,args", [("translated", ((1, 0, 0),)), ("rotated", ("z", 45)), ("mirrored", ("x",))]
)
def test_aggregation_conflicts_and_non_solid_fusion_rejected(method, args):
    curve = geo.Curve.line((0, 0, 0), (1, 0, 0))
    with pytest.raises(ValueError, match="either unite=True or group=True"):
        getattr(curve, method)(*args, copy=True, unite=True, group=True)
    with pytest.raises(TypeError, match="Solid"):
        getattr(curve, method)(*args, copy=True, unite=True)
    grouped = getattr(curve, method)(*args, copy=True, group=True)
    assert isinstance(grouped, geo.Group)
    assert all(isinstance(s, geo.Curve) for s in grouped.shapes)


@pytest.mark.parametrize("mode", ["unite", "group"])
def test_mirror_aggregation_requires_original(mode):
    with pytest.raises(ValueError, match="requires copy=True"):
        geo.Brick().mirrored("x", **{mode: True})


@pytest.mark.parametrize(
    "method,args",
    [("translated", ((1, 2, 3),)), ("rotated", ("z", 45, (1, 2, 3))), ("mirrored", ("x", 1))],
)
def test_repeated_nested_assemblies_keep_materials_and_member_placement(method, args):
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    assembly = geo.Group(
        geo.Group(geo.Brick(material="pec", name="metal"), name="inner"),
        geo.Sphere(center=(2, 0, 0), radius=0.25, material="air", name="dielectric"),
        name="component",
    )
    options = {} if method == "mirrored" else {"repeat": 2}
    result = getattr(assembly, method)(*args, copy=True, group=True, **options)
    assert result.shapes[0] is assembly
    assert len(result.shapes) == (2 if method == "mirrored" else 3)
    for i, placed in enumerate(result.shapes[1:], 1):
        assert placed.name == "component" and placed.shapes[0].name == "inner"
        members = list(placed.members())
        originals = list(assembly.members())
        assert [m.material for m in members] == [m.material for m in originals]
        if method == "translated":
            transform = geo.Translation(tuple(i * x for x in args[0]))
        elif method == "rotated":
            transform = geo.Rotation(args[0], i * args[1], args[2])
        else:
            transform = geo.Mirror(*args)
        for member, original in zip(members, originals):
            _assert_box(member.bounding_box(), (transform @ original).bounding_box())
    with pytest.raises(TypeError, match="Solid"):
        getattr(assembly, method)(*args, copy=True, unite=True)


def test_documented_array_recipe_executes_and_has_eight_elements():
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    import re
    from pathlib import Path

    guide = (Path(__file__).resolve().parents[2] / "docs/methods/geometry.md").read_text()
    (recipe,) = [
        block
        for block in re.findall(r"```python\n(.*?)```", guide, re.DOTALL)
        if block.startswith("post = geo.Cylinder")
    ]
    namespace = {"geo": geo}
    exec(compile(recipe, "array documentation", "exec"), namespace)
    assert namespace["posts"].volume() == pytest.approx(8 * namespace["post"].volume(), rel=1e-10)
    assert len(namespace["fence"].shapes) == 8
    assert namespace["fence"].shapes[0] is namespace["via"]
    assert len(namespace["pair"].shapes) == 2
