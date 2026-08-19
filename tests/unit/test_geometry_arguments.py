"""Argument checking across the geometry API.

The point of every test here is *when* the error arrives: at the call
that got the argument wrong, not later in the bounding box, the CAD
kernel or the mesher.  So none of them build an OCC shape — a check that
only fires once the kernel runs has already missed its moment.
"""

import pytest

from magnelio.geo import (
    Brick,
    Cone,
    Curve,
    Cylinder,
    Difference,
    Face,
    GeometryModel,
    Group,
    Intersection,
    Path,
    Sphere,
    ThinWire,
    Torus,
    Union,
)
from magnelio.materials.material import Material

PEC = Material.pec()
AIR = Material.air()


def _brick(**kw):
    return Brick(origin=(0, 0, 0), size=(1, 1, 1), material=PEC, **kw)


# ── points and vectors ───────────────────────────────────────────────────────


class TestPointArguments:
    """A coordinate triple given as something else."""

    def test_scalar_center_names_the_field_and_the_fix(self):
        with pytest.raises(TypeError) as excinfo:
            Sphere(center=95e-3, radius=100e-3, material=PEC)
        message = str(excinfo.value)
        assert "Sphere.center" in message
        assert "(x, y, z)" in message
        assert "single number is not three coordinates" in message

    @pytest.mark.parametrize(
        "make",
        [
            lambda p: Sphere(center=p, radius=1.0),
            lambda p: Brick(origin=p, size=(1, 1, 1)),
            lambda p: Cylinder(origin=p, radius=1.0, height=1.0),
            lambda p: Cone(origin=p, height=1.0),
            lambda p: Torus(center=p),
        ],
        ids=["sphere", "brick", "cylinder", "cone", "torus"],
    )
    def test_every_primitive_rejects_a_two_component_point(self, make):
        with pytest.raises(ValueError, match="2 coordinates"):
            make((1.0, 2.0))

    def test_non_numeric_coordinate_is_a_type_error(self):
        with pytest.raises(TypeError):
            Sphere(center=(0, 0, "top"), radius=1.0)

    def test_a_string_is_not_a_point(self):
        with pytest.raises(TypeError, match="string"):
            Sphere(center="xyz", radius=1.0)

    def test_nan_coordinate_is_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            Sphere(center=(0, 0, float("nan")), radius=1.0)

    def test_points_are_normalised_to_float_tuples(self):
        pytest.importorskip("numpy")
        import numpy as np

        s = Sphere(center=np.array([1, 2, 3]), radius=1)
        assert s.center == (1.0, 2.0, 3.0)
        assert isinstance(s.center, tuple)
        assert all(isinstance(c, float) for c in s.center)

    def test_translation_vector_is_checked_at_the_call(self):
        with pytest.raises(TypeError, match=r"translated\(vector\)"):
            _brick().translated(1e-3)

    def test_rotation_origin_is_checked_at_the_call(self):
        with pytest.raises(ValueError, match=r"rotated\(origin\)"):
            _brick().rotated("z", 90.0, origin=(0, 0))


# ── sizes, radii and extents ─────────────────────────────────────────────────


class TestExtentArguments:
    """Dimensions that cannot describe a solid."""

    @pytest.mark.parametrize("radius", [0.0, -1e-3])
    def test_sphere_radius_must_be_positive(self, radius):
        with pytest.raises(ValueError, match="Sphere.radius must be positive"):
            Sphere(center=(0, 0, 0), radius=radius)

    def test_brick_size_must_be_positive_and_points_at_from_corners(self):
        with pytest.raises(ValueError) as excinfo:
            Brick(origin=(0, 0, 0), size=(1e-3, -2e-3, 1e-3))
        assert "from_corners" in str(excinfo.value)

    def test_zero_height_cylinder_is_rejected(self):
        with pytest.raises(ValueError, match="Cylinder.height must not be zero"):
            Cylinder(radius=1e-3, height=0.0)

    def test_negative_height_stays_legal(self):
        # A negative height extrudes along -axis; only zero is degenerate.
        assert Cylinder(radius=1e-3, height=-2e-3).height == -2e-3

    def test_cylinder_radius_must_be_positive(self):
        with pytest.raises(ValueError, match="Cylinder.radius must be positive"):
            Cylinder(radius=0.0, height=1e-3)

    def test_cone_pointed_at_both_ends_is_not_a_solid(self):
        with pytest.raises(ValueError, match="non-zero radius at one end"):
            Cone(bottom_radius=0.0, top_radius=0.0, height=1e-3)

    def test_cone_keeps_a_zero_radius_tip(self):
        assert Cone(bottom_radius=1e-3, top_radius=0.0, height=1e-3).top_radius == 0.0

    def test_torus_tube_may_not_reach_the_axis(self):
        with pytest.raises(ValueError, match="intersect itself"):
            Torus(major_radius=1e-3, minor_radius=2e-3)

    def test_scaling_by_zero_is_rejected(self):
        with pytest.raises(ValueError, match=r"scaled\(factor\) must not be zero"):
            _brick().scaled(0.0)

    def test_repeat_must_be_a_whole_number(self):
        with pytest.raises(TypeError, match="repeat"):
            _brick().translated((1e-3, 0, 0), repeat=2.5)

    def test_repeat_below_one_is_rejected(self):
        with pytest.raises(ValueError, match="repeat"):
            _brick().translated((1e-3, 0, 0), repeat=0)


# ── axes ─────────────────────────────────────────────────────────────────────


class TestAxisArguments:
    """An axis that names no direction, caught in the constructor."""

    @pytest.mark.parametrize(
        "make, field",
        [
            (lambda a: Cylinder(radius=1.0, height=1.0, axis=a), "Cylinder.axis"),
            (lambda a: Cone(height=1.0, axis=a), "Cone.axis"),
            (lambda a: Torus(axis=a), "Torus.axis"),
        ],
        ids=["cylinder", "cone", "torus"],
    )
    def test_bad_axis_letter_names_the_field(self, make, field):
        with pytest.raises(ValueError, match=field):
            make("q")

    def test_zero_axis_vector_is_rejected(self):
        with pytest.raises(ValueError, match="non-zero"):
            Cylinder(radius=1.0, height=1.0, axis=(0, 0, 0))


# ── operands ─────────────────────────────────────────────────────────────────


class TestOperandArguments:
    """Boolean operands and model members that are not geometry."""

    def test_a_list_of_shapes_is_not_an_operand(self):
        with pytest.raises(TypeError) as excinfo:
            Union([_brick(), Sphere(radius=1.0, material=PEC)])
        assert "Union(a, b)" in str(excinfo.value)

    def test_union_of_nothing_is_rejected(self):
        with pytest.raises(ValueError, match="at least 1 operand"):
            Union()

    def test_difference_needs_a_tool(self):
        with pytest.raises(ValueError, match="at least 2 operands"):
            Difference(_brick())

    def test_intersection_rejects_a_non_shape(self):
        with pytest.raises(TypeError, match="geometry object"):
            Intersection(_brick(), None)

    def test_group_member_must_be_geometry(self):
        with pytest.raises(TypeError, match="Group member 1"):
            Group(_brick(), "the other one")

    def test_model_rejects_a_non_shape(self):
        with pytest.raises(TypeError, match="geometry object"):
            GeometryModel(background=AIR).add("a brick")

    def test_group_still_takes_nested_groups(self):
        inner = Group(_brick())
        assert len(list(Group(inner, _brick()).members())) == 2


# ── profiles and curves ──────────────────────────────────────────────────────


class TestProfileArguments:
    """Point sequences for faces, curves and paths."""

    def test_face_points_must_be_in_plane_pairs(self):
        with pytest.raises(ValueError, match="3 coordinates"):
            Face(normal="z", points=[(0, 0, 0), (1, 0, 0), (1, 1, 0)])

    def test_face_needs_three_points(self):
        with pytest.raises(ValueError, match="at least 3 points"):
            Face(normal="z", points=[(0, 0), (1, 0)])

    def test_flat_coordinate_list_is_named_as_such(self):
        with pytest.raises(TypeError, match="flat list of coordinates"):
            Curve.polyline([0, 0, 0, 1, 0, 0])

    def test_polyline_points_must_be_three_dimensional(self):
        with pytest.raises(ValueError, match="point 1 has 2 coordinates"):
            Curve.polyline([(0, 0, 0), (1, 0)])

    def test_helix_radius_must_be_positive(self):
        with pytest.raises(ValueError, match=r"Curve.helix\(radius\)"):
            Curve.helix(radius=0.0, pitch=1e-3, turns=2)

    def test_path_start_must_be_a_point(self):
        with pytest.raises(TypeError, match=r"Path\(start\)"):
            Path(0.0)

    def test_path_segment_point_is_checked_at_the_call(self):
        with pytest.raises(ValueError, match=r"line_to\(point\)"):
            Path((0, 0, 0)).line_to((1e-3, 0))

    def test_thin_wire_radius_must_be_positive(self):
        with pytest.raises(ValueError, match="ThinWire.radius"):
            ThinWire(curve=Curve.polyline([(0, 0, 0), (1e-3, 0, 0)]), radius=0.0)


# ── verbs ────────────────────────────────────────────────────────────────────


class TestVerbArguments:
    """Verb arguments that the kernel would only reject much later."""

    def test_chamfer_without_a_selector_is_rejected(self):
        with pytest.raises(ValueError, match="exactly one of near=, face_near= or edges="):
            _brick().chamfered(distance=1e-4)

    def test_chamfer_with_two_selectors_is_rejected(self):
        with pytest.raises(ValueError, match="exactly one of"):
            _brick().chamfered(edges="all", face_near=(0, 0, 1), distance=1e-4)

    def test_fillet_radius_must_be_positive(self):
        with pytest.raises(ValueError, match=r"filleted\(radius\)"):
            _brick().filleted(edges="all", radius=-1e-4)

    def test_chamfer_takes_an_asymmetric_pair(self):
        assert _brick().chamfered(edges="all", distance=(1e-4, 2e-4)) is not None

    def test_edges_takes_only_all(self):
        with pytest.raises(ValueError, match="takes only 'all'"):
            _brick().chamfered(edges="every", distance=1e-4)

    def test_zero_extrusion_vector_is_rejected(self):
        sheet = Face(normal="z", points=[(0, 0), (1e-3, 0), (1e-3, 1e-3)])
        with pytest.raises(ValueError, match="zero vector"):
            sheet.extruded((0, 0, 0), material=PEC)

    def test_sweep_spine_must_be_a_curve(self):
        sheet = Face(normal="z", points=[(0, 0), (1e-3, 0), (1e-3, 1e-3)])
        with pytest.raises(TypeError, match="needs a Curve as its spine"):
            sheet.swept([(0, 0, 0), (0, 0, 1e-3)], material=PEC)

    def test_revolve_beyond_a_full_turn_is_rejected(self):
        sheet = Face(normal="x", points=[(0, 0), (1e-3, 0), (1e-3, 1e-3)])
        with pytest.raises(ValueError, match="at most a full turn"):
            sheet.revolved("z", 720.0, material=PEC)


class TestModelArguments:
    """The model's own arguments."""

    def test_background_must_be_a_material(self):
        with pytest.raises(TypeError, match=r"GeometryModel\(background=\.\.\.\)"):
            GeometryModel(background="air")

    def test_from_corners_names_its_own_argument(self):
        with pytest.raises(ValueError, match=r"Brick.from_corners\(p2\)"):
            Brick.from_corners((0, 0, 0), (1e-3, 1e-3))
