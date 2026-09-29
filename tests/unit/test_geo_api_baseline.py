"""Characterize the geometry API boundary as the foundation migrates."""

from __future__ import annotations

import inspect

import pytest

import magnelio.geo as geo
from magnelio.materials.material import Material

EXPECTED_EXPORTS = [
    "Shape",
    "Curve",
    "Sheet",
    "Profile",
    "Surface",
    "Solid",
    "TopologyRef",
    "VertexRef",
    "EdgeRef",
    "FaceRef",
    "EdgeSetRef",
    "FaceSetRef",
    "TopologySelectionError",
    "AmbiguousTopologyError",
    "TopologyEvolutionError",
    "Transform",
    "Translation",
    "Rotation",
    "Mirror",
    "Scale",
    "Brick",
    "Sphere",
    "Cylinder",
    "Cone",
    "Torus",
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
        "(self, axis, angle_deg, origin=(0.0, 0.0, 0.0), *, "
        "repeat=1, copy=False, unite=False, group=False)"
    ),
    "scaled": "(self, factor, center=(0.0, 0.0, 0.0))",
    "mirrored": "(self, normal, position=0.0, *, copy=False, unite=False, group=False)",
    "chamfered": "(self, *, near=None, face_near=None, edges=None, faces=None, distance)",
    "filleted": "(self, *, near=None, face_near=None, edges=None, faces=None, radius)",
    "extruded": "(self, vector, *, face_near=None, material=None)",
    "revolved": "(self, axis, angle_deg=360.0, *, origin=(0.0, 0.0, 0.0), material=None)",
    "swept": "(self, spine, *, face_near=None, material=None)",
    "shelled": "(self, thickness, *, opening_face_near=None, openings=None)",
    "thickened": "(self, thickness, *, direction='forward', material=None)",
    "lofted": (
        "(self, face_near, other=None, other_face_near=None, *, "
        "material=None, blend='spline', tension=None)"
    ),
}


def _air():
    return Material.air()


def _assert_box(actual, expected):
    for actual_corner, expected_corner in zip(actual, expected):
        assert actual_corner == pytest.approx(expected_corner)


def test_curated_exports_are_pinned_during_the_breaking_migration():
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


def test_wp1_dimensional_categories_are_explicit():
    assert issubclass(geo.Curve, geo.Shape)
    assert issubclass(geo.Profile, geo.Sheet)
    assert issubclass(geo.Sheet, geo.Shape)
    assert issubclass(geo.Solid, geo.Shape)
    assert issubclass(geo.Brick, geo.Solid)
    assert not hasattr(geo, "Face")
    assert not hasattr(geo.Curve, "covered")
    assert issubclass(geo.Surface, geo.Sheet)
    assert not issubclass(geo.Group, geo.Shape)
    assert not issubclass(geo.Path, geo.Shape)
    assert not issubclass(geo.ThinWire, geo.Shape)
    assert hasattr(geo.Curve.polyline([(0, 0, 0), (1, 0, 0)]), "translated")


def test_axis_normal_face_and_profile_extrusion_numerics():
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    profile = geo.Profile.polygon(
        [(u, v, 4.0) for u, v in ((0.0, 0.0), (2.0, 0.0), (2.0, 3.0), (0.0, 3.0))], material=_air()
    )

    _assert_box(profile.bounding_box(), ((0.0, 0.0, 4.0), (2.0, 3.0, 4.0)))
    assert profile.volume() == pytest.approx(0.0, abs=1e-14)
    assert profile.extruded((0.0, 0.0, 5.0)).volume() == pytest.approx(30.0, rel=1e-12)


def test_named_transform_methods_chain_left_to_right():
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    body = geo.Brick(size=(1.0, 2.0, 3.0), material=_air())

    placed = body.translated((2.0, 0.0, 0.0)).rotated("z", 90.0)

    _assert_box(placed.bounding_box(), ((-2.0, 2.0, 0.0), (0.0, 3.0, 3.0)))


def test_transform_defaults_preserve_category_and_arrays_are_explicit():
    body = geo.Brick(material=_air())

    assert isinstance(body.translated((1.0, 0.0, 0.0)), geo.Solid)
    copies = body.translated((1.0, 0.0, 0.0), repeat=2)
    assert isinstance(copies, list) and len(copies) == 2
    assert all(isinstance(s, geo.Solid) for s in copies)


def test_point_convenience_selects_the_same_owned_face_for_extrusion():
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    body = geo.Brick(size=(1.0, 1.0, 1.0), material=_air())

    extended = body.extruded((0.0, 0.0, 1.0), face_near=(0.5, 0.5, 1.0))

    # Solid-face extrusion returns the new prism, not a union with the input.
    assert extended.volume() == pytest.approx(1.0, rel=1e-12)
    assert isinstance(body.face(near=(0.5, 0.5, 1.0)), geo.FaceRef)


def test_equidistant_operation_selection_raises_instead_of_using_kernel_order():
    body = geo.Brick(size=(1, 1, 1), material=_air())
    with pytest.raises(geo.AmbiguousTopologyError):
        body.extruded((0, 0, 1), face_near=(0.5, 0.5, 0.5))
