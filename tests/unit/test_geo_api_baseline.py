"""Characterize the public geometry API before its foundation migration."""

from __future__ import annotations

import inspect

import pytest

import magnelio.geo as geo
from magnelio.materials.material import Material

EXPECTED_EXPORTS = [
    "Shape",
    "Brick",
    "Sphere",
    "Cylinder",
    "Cone",
    "Torus",
    "Face",
    "Surface",
    "Curve",
    "Path",
    "Union",
    "Intersection",
    "Difference",
    "Loft",
    "Group",
    "ThinWire",
    "ImportedSolid",
    "GeometryOverlapError",
]

EXPECTED_CONSTRUCTOR_SIGNATURES = {
    "Shape": (),
    "Brick": (
        ("material", "POSITIONAL_OR_KEYWORD", "None"),
        ("name", "POSITIONAL_OR_KEYWORD", "None"),
        ("origin", "POSITIONAL_OR_KEYWORD", "(0.0, 0.0, 0.0)"),
        ("size", "POSITIONAL_OR_KEYWORD", "(1.0, 1.0, 1.0)"),
    ),
    "Sphere": (
        ("material", "POSITIONAL_OR_KEYWORD", "None"),
        ("name", "POSITIONAL_OR_KEYWORD", "None"),
        ("center", "POSITIONAL_OR_KEYWORD", "(0.0, 0.0, 0.0)"),
        ("radius", "POSITIONAL_OR_KEYWORD", "1.0"),
    ),
    "Cylinder": (
        ("material", "POSITIONAL_OR_KEYWORD", "None"),
        ("name", "POSITIONAL_OR_KEYWORD", "None"),
        ("origin", "POSITIONAL_OR_KEYWORD", "(0.0, 0.0, 0.0)"),
        ("radius", "POSITIONAL_OR_KEYWORD", "1.0"),
        ("height", "POSITIONAL_OR_KEYWORD", "1.0"),
        ("axis", "POSITIONAL_OR_KEYWORD", "'z'"),
        ("inner_radius", "POSITIONAL_OR_KEYWORD", "0.0"),
        ("angle_deg", "POSITIONAL_OR_KEYWORD", "None"),
    ),
    "Cone": (
        ("material", "POSITIONAL_OR_KEYWORD", "None"),
        ("name", "POSITIONAL_OR_KEYWORD", "None"),
        ("origin", "POSITIONAL_OR_KEYWORD", "(0.0, 0.0, 0.0)"),
        ("bottom_radius", "POSITIONAL_OR_KEYWORD", "1.0"),
        ("top_radius", "POSITIONAL_OR_KEYWORD", "0.0"),
        ("height", "POSITIONAL_OR_KEYWORD", "1.0"),
        ("axis", "POSITIONAL_OR_KEYWORD", "'z'"),
    ),
    "Torus": (
        ("material", "POSITIONAL_OR_KEYWORD", "None"),
        ("name", "POSITIONAL_OR_KEYWORD", "None"),
        ("center", "POSITIONAL_OR_KEYWORD", "(0.0, 0.0, 0.0)"),
        ("major_radius", "POSITIONAL_OR_KEYWORD", "1.0"),
        ("minor_radius", "POSITIONAL_OR_KEYWORD", "0.25"),
        ("axis", "POSITIONAL_OR_KEYWORD", "'z'"),
    ),
    "Face": (
        ("normal", "POSITIONAL_OR_KEYWORD", "<required>"),
        ("points", "POSITIONAL_OR_KEYWORD", "<required>"),
        ("position", "POSITIONAL_OR_KEYWORD", "0.0"),
        ("material", "POSITIONAL_OR_KEYWORD", "None"),
        ("name", "POSITIONAL_OR_KEYWORD", "None"),
    ),
    "Surface": (
        ("points", "POSITIONAL_OR_KEYWORD", "<required>"),
        ("material", "POSITIONAL_OR_KEYWORD", "None"),
        ("name", "POSITIONAL_OR_KEYWORD", "None"),
    ),
    "Curve": (
        ("_build", "POSITIONAL_OR_KEYWORD", "<required>"),
        ("name", "POSITIONAL_OR_KEYWORD", "None"),
        ("_bounds", "POSITIONAL_OR_KEYWORD", "None"),
        ("_ends", "POSITIONAL_OR_KEYWORD", "None"),
        ("_segments", "POSITIONAL_OR_KEYWORD", "<factory>"),
    ),
    "Path": (
        ("start", "POSITIONAL_OR_KEYWORD", "<required>"),
        ("_segments", "POSITIONAL_OR_KEYWORD", "<factory>"),
    ),
    "Union": (
        ("shapes", "VAR_POSITIONAL", "<required>"),
        ("material", "KEYWORD_ONLY", "None"),
        ("name", "KEYWORD_ONLY", "None"),
    ),
    "Intersection": (
        ("shape_a", "POSITIONAL_OR_KEYWORD", "<required>"),
        ("shape_b", "POSITIONAL_OR_KEYWORD", "<required>"),
        ("material", "POSITIONAL_OR_KEYWORD", "None"),
        ("name", "POSITIONAL_OR_KEYWORD", "None"),
    ),
    "Difference": (
        ("base", "POSITIONAL_OR_KEYWORD", "<required>"),
        ("tools", "VAR_POSITIONAL", "<required>"),
        ("material", "KEYWORD_ONLY", "None"),
        ("name", "KEYWORD_ONLY", "None"),
    ),
    "Loft": (
        ("sections", "VAR_POSITIONAL", "<required>"),
        ("blend", "KEYWORD_ONLY", "'spline'"),
        ("material", "KEYWORD_ONLY", "None"),
        ("name", "KEYWORD_ONLY", "None"),
    ),
    "Group": (
        ("shapes", "VAR_POSITIONAL", "<required>"),
        ("name", "KEYWORD_ONLY", "None"),
    ),
    "ThinWire": (
        ("curve", "POSITIONAL_OR_KEYWORD", "<required>"),
        ("radius", "POSITIONAL_OR_KEYWORD", "<required>"),
        ("name", "POSITIONAL_OR_KEYWORD", "None"),
    ),
    "ImportedSolid": (
        ("shape", "POSITIONAL_OR_KEYWORD", "<required>"),
        ("material", "POSITIONAL_OR_KEYWORD", "None"),
        ("name", "POSITIONAL_OR_KEYWORD", "None"),
        ("color", "POSITIONAL_OR_KEYWORD", "None"),
    ),
}

EXPECTED_SHAPE_VERB_SIGNATURES = {
    "translated": "(self, vector, *, repeat=1, copy=False, unite=False, group=False)",
    "rotated": (
        "(self, axis, angle_deg, origin=(0.0, 0.0, 0.0), *, repeat=1, "
        "copy=False, unite=False, group=False)"
    ),
    "scaled": "(self, factor, center=(0.0, 0.0, 0.0))",
    "mirrored": "(self, normal, position=0.0, *, copy=False, unite=False, group=False)",
    "chamfered": "(self, *, near=None, face_near=None, edges=None, distance)",
    "filleted": "(self, *, near=None, face_near=None, edges=None, radius)",
    "extruded": "(self, vector, *, face_near=None, material=None)",
    "revolved": "(self, axis, angle_deg=360.0, *, origin=(0.0, 0.0, 0.0), material=None)",
    "swept": "(self, spine, *, material=None)",
    "shelled": "(self, thickness, *, opening_face_near=None)",
    "thickened": "(self, thickness, *, direction='forward', material=None)",
    "lofted": (
        "(self, face_near, other, other_face_near, *, material=None, blend='spline', tension=None)"
    ),
}


def _air():
    return Material.air()


def _assert_box(actual, expected):
    for actual_corner, expected_corner in zip(actual, expected):
        assert actual_corner == pytest.approx(expected_corner)


def test_curated_exports_are_pinned_before_the_breaking_migration():
    assert geo.__all__ == EXPECTED_EXPORTS


@pytest.mark.parametrize(("name", "signature"), EXPECTED_CONSTRUCTOR_SIGNATURES.items())
def test_constructor_signatures_are_pinned(name, signature):
    parameters = inspect.signature(getattr(geo, name)).parameters.values()
    observed = tuple(
        (
            parameter.name,
            parameter.kind.name,
            (
                "<required>"
                if parameter.default is inspect.Signature.empty
                else repr(parameter.default)
            ),
        )
        for parameter in parameters
    )
    assert observed == signature


@pytest.mark.parametrize(("name", "signature"), EXPECTED_SHAPE_VERB_SIGNATURES.items())
def test_common_shape_verb_signatures_are_pinned(name, signature):
    assert str(inspect.signature(getattr(geo.Shape, name))) == signature


def test_current_dimensional_categories_are_implicit():
    assert issubclass(geo.Brick, geo.Shape)
    assert issubclass(geo.Face, geo.Shape)
    assert issubclass(geo.Surface, geo.Shape)
    assert issubclass(geo.Group, geo.Shape)
    assert not issubclass(geo.Curve, geo.Shape)
    assert not issubclass(geo.Path, geo.Shape)
    assert not issubclass(geo.ThinWire, geo.Shape)
    assert not hasattr(geo.Curve.polyline([(0, 0, 0), (1, 0, 0)]), "translated")


def test_axis_normal_face_and_profile_extrusion_numerics():
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    profile = geo.Face(
        normal="z",
        points=((0.0, 0.0), (2.0, 0.0), (2.0, 3.0), (0.0, 3.0)),
        position=4.0,
        material=_air(),
    )

    _assert_box(profile.bounding_box(), ((0.0, 0.0, 4.0), (2.0, 3.0, 4.0)))
    assert profile.volume() == pytest.approx(0.0, abs=1e-14)
    assert profile.extruded((0.0, 0.0, 5.0)).volume() == pytest.approx(30.0, rel=1e-12)


def test_named_transform_methods_chain_left_to_right():
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    body = geo.Brick(size=(1.0, 2.0, 3.0), material=_air())

    placed = body.translated((2.0, 0.0, 0.0)).rotated("z", 90.0)

    _assert_box(placed.bounding_box(), ((-2.0, 2.0, 0.0), (0.0, 3.0, 3.0)))


def test_repeated_transforms_currently_change_result_category():
    body = geo.Brick(material=_air())

    assert isinstance(body.translated((1.0, 0.0, 0.0)), geo.Shape)
    assert isinstance(body.translated((1.0, 0.0, 0.0), repeat=2), list)
    assert isinstance(body.translated((1.0, 0.0, 0.0), repeat=2, copy=True, unite=True), geo.Union)
    assert isinstance(body.translated((1.0, 0.0, 0.0), repeat=2, copy=True, group=True), geo.Group)


def test_face_selection_is_a_loose_point_consumed_by_the_operation():
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    body = geo.Brick(size=(1.0, 1.0, 1.0), material=_air())

    extended = body.extruded((0.0, 0.0, 1.0), face_near=(0.5, 0.5, 1.0))

    # Solid-face extrusion returns the new prism, not a union with the input.
    assert extended.volume() == pytest.approx(1.0, rel=1e-12)
    assert not hasattr(body, "face")


def test_equidistant_nearest_face_currently_uses_kernel_order():
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    from magnelio.geo._occ_backend import find_nearest_face

    body = geo.Brick(size=(1.0, 1.0, 1.0), material=_air())

    # The centre is equidistant from all six faces.  The old selector returns
    # one of them instead of reporting an ambiguity.
    assert find_nearest_face(body._occ_shape(), (0.5, 0.5, 0.5)) is not None
