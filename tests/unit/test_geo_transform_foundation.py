"""Dimensional geometry and affine value algebra (geometry foundation WP1)."""

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
    profile = geo.Face("z", ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), name="section")
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
    profile = geo.Face("z", ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))

    with pytest.raises(TypeError, match="accepts Solid operands"):
        operation(solid, profile)


def test_solid_operators_reject_curves_and_geometry_cannot_right_apply_transform():
    solid = geo.Brick()
    curve = geo.Curve.polyline(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))

    with pytest.raises(TypeError, match="accepts Solid operands"):
        solid + curve
    with pytest.raises(TypeError, match="unsupported operand"):
        solid @ geo.Translation((1.0, 0.0, 0.0))
