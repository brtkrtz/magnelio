"""Shape base class: CSG operators, chainable verbs, axis/height ergonomics."""

import math

import pytest

import magnelio.geo as geo
from magnelio.geo import (
    Brick,
    Cone,
    Cylinder,
    Difference,
    Face,
    Group,
    Intersection,
    Shape,
    Sphere,
    Torus,
    Union,
)
from magnelio.geo._axes import normalize_axis
from magnelio.geo.transforms import mirror, rotate, translate
from magnelio.materials.material import Material


def _occ():
    """Skip test if pythonocc-core is not installed."""
    return pytest.importorskip("OCC.Core.BRepPrimAPI")


PEC = Material.pec()
AIR = Material.air()

VERBS = (
    "translated",
    "rotated",
    "scaled",
    "mirrored",
    "chamfered",
    "filleted",
    "extruded",
    "revolved",
    "swept",
    "lofted",
)


def _brick(**kw):
    return Brick(material=PEC, origin=(0, 0, 0), size=(1, 1, 1), **kw)


# ── documented surface ───────────────────────────────────────────────────────


class TestDocumentedSurface:
    """Guards the API reference: the verbs are only discoverable there if
    Shape is exported and carries their full descriptions itself.  Their
    implementations live in private modules that no doc page renders."""

    def test_shape_is_exported(self):
        assert "Shape" in geo.__all__
        assert geo.Shape is Shape

    # Curve and ThinWire are deliberately absent: they are 1D objects
    # (a sweep spine, a sub-cell wire), not solids, so the Boolean
    # operators and the verbs do not apply to them.
    @pytest.mark.parametrize("cls", [Brick, Sphere, Cylinder, Cone, Torus, Face])
    def test_every_geometry_class_is_a_shape(self, cls):
        assert issubclass(cls, Shape)

    @pytest.mark.parametrize("cls", [Union, Intersection, Difference, Group])
    def test_every_boolean_result_is_a_shape(self, cls):
        assert issubclass(cls, Shape)

    @pytest.mark.parametrize("verb", VERBS)
    def test_verb_carries_its_own_documentation(self, verb):
        doc = getattr(Shape, verb).__doc__
        assert doc is not None, f"Shape.{verb} has no docstring"
        # A pointer to the implementation is what the docs used to show:
        # a one-liner whose target no page renders.  Demand the real text.
        assert "Parameters" in doc, f"Shape.{verb} documents no parameters"
        assert "Returns" in doc, f"Shape.{verb} documents no return value"

    @pytest.mark.parametrize("op", ["__add__", "__sub__", "__and__"])
    def test_operators_are_documented(self, op):
        assert getattr(Shape, op).__doc__


# ── CSG operators ────────────────────────────────────────────────────────────


class TestCSGOperators:
    def test_add_is_union(self):
        a, b = _brick(), Sphere(material=PEC, center=(1, 0, 0), radius=0.5)
        u = a + b
        assert isinstance(u, Union)
        assert u.shapes == (a, b)
        assert u.material is a.material

    def test_sub_is_difference(self):
        a, b = _brick(), Sphere(material=PEC, center=(0.5, 0.5, 0.5), radius=0.2)
        d = a - b
        assert isinstance(d, Difference)
        assert d.base is a
        assert d.tools == (b,)

    def test_and_is_intersection(self):
        a, b = _brick(), Sphere(material=PEC, center=(0, 0, 0), radius=0.8)
        i = a & b
        assert isinstance(i, Intersection)
        assert i.shape_a is a and i.shape_b is b

    def test_operators_chain(self):
        a, b, c = (
            _brick(),
            Sphere(material=PEC, radius=0.5),
            Sphere(material=PEC, center=(1, 1, 1), radius=0.3),
        )
        d = a + b - c
        assert isinstance(d, Difference)
        assert isinstance(d.base, Union)

    def test_operator_on_boolean_result(self):
        a, b = _brick(), Sphere(material=PEC, radius=0.4)
        u = Union(a, b)
        d = u - Sphere(material=PEC, center=(1, 0, 0), radius=0.2)
        assert isinstance(d, Difference)

    def test_non_shape_operand_raises_type_error(self):
        with pytest.raises(TypeError):
            _brick() + 3

    def test_group_operand_keeps_descriptive_error(self):
        g = Group(_brick())
        with pytest.raises(TypeError, match="Group"):
            _brick() + g


# ── chainable verbs ──────────────────────────────────────────────────────────


class TestVerbMethods:
    def test_translated_equals_free_function(self):
        _occ()
        b = _brick()
        via_method = b.translated((0.5, 0.0, 0.0))
        via_function = translate(b, (0.5, 0.0, 0.0))
        assert via_method.bounding_box() == via_function.bounding_box()

    def test_chaining_translate_rotate(self):
        _occ()
        b = _brick()
        chained = b.translated((1.0, 0.0, 0.0)).rotated("z", 90.0)
        stacked = rotate(translate(b, (1.0, 0.0, 0.0)), "z", 90.0)
        lo1, hi1 = chained.bounding_box()
        lo2, hi2 = stacked.bounding_box()
        assert lo1 == pytest.approx(lo2)
        assert hi1 == pytest.approx(hi2)

    def test_chain_then_subtract(self):
        _occ()
        outer = _brick()
        hole = Cylinder(material=AIR, origin=(0.5, 0.5, -1), radius=0.2, height=3.0).translated(
            (0.0, 0.0, 0.0)
        )
        d = outer - hole
        assert isinstance(d, Difference)
        lo, hi = d.bounding_box()
        assert hi[2] <= 1.0 + 1e-9

    def test_group_distributes_translated(self):
        _occ()
        g = Group(_brick(), Sphere(material=AIR, center=(2, 0, 0), radius=0.5))
        moved = g.translated((0.0, 0.0, 3.0))
        assert isinstance(moved, Group)
        lo, hi = moved.bounding_box()
        assert lo[2] == pytest.approx(3.0 - 0.5)

    def test_wrapper_result_keeps_verbs(self):
        _occ()
        b = _brick().rotated((0, 0, 1), 45.0).scaled(2.0)
        assert hasattr(b, "translated")
        lo, hi = b.bounding_box()
        assert hi[0] - lo[0] == pytest.approx(2.0 * math.sqrt(2.0), rel=1e-6)

    def test_filleted_method(self):
        _occ()
        f = _brick().filleted(edges="all", radius=0.1)
        lo, hi = f.bounding_box()
        assert hi[0] - lo[0] == pytest.approx(1.0, rel=1e-6)


# ── plane mirror ─────────────────────────────────────────────────────────────


def _volume(shape, scale=1.0):
    """Solid volume of a shape [m^3], via OCC mass properties."""
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps

    props = GProp_GProps()
    brepgprop.VolumeProperties(shape._occ_shape(scale), props)
    return props.Mass() / scale**3


# Chiral on purpose: offset from every plane through the origin, so a
# mirror image is distinguishable from any rotation of the original.
def _chiral():
    return Brick(material=PEC, origin=(1.0, 2.0, 0.5), size=(3.0, 1.0, 2.0))


class TestMirror:
    def test_mirror_reflects_the_normal_axis_only(self):
        _occ()
        m = _chiral().mirrored(normal="x")
        lo, hi = m.bounding_box()
        assert (lo[0], hi[0]) == pytest.approx((-4.0, -1.0))
        assert (lo[1], hi[1]) == pytest.approx((2.0, 3.0))
        assert (lo[2], hi[2]) == pytest.approx((0.5, 2.5))

    def test_mirror_is_not_a_rotation(self):
        _occ()
        b = _chiral()
        # Rotating 180 deg about y also flips z; the mirror must not.
        mirrored = b.mirrored(normal="x")
        rotated = b.rotated("y", 180.0)
        assert mirrored.bounding_box()[0][2] == pytest.approx(0.5)
        assert rotated.bounding_box()[0][2] == pytest.approx(-2.5)

    def test_mirror_preserves_volume_and_validity(self):
        _occ()
        from OCC.Core.BRepCheck import BRepCheck_Analyzer

        b = _chiral()
        m = b.mirrored(normal="x")
        # A reflection has determinant -1; a shape whose orientation was
        # not reversed would come back inside-out (negative volume).
        assert _volume(m) == pytest.approx(_volume(b))
        assert _volume(m) > 0.0
        assert BRepCheck_Analyzer(m._occ_shape(1.0)).IsValid()

    def test_position_offsets_the_plane(self):
        _occ()
        m = _chiral().mirrored(normal="x", position=5.0)
        lo, hi = m.bounding_box()
        assert (lo[0], hi[0]) == pytest.approx((6.0, 9.0))

    def test_mirroring_twice_is_the_identity(self):
        _occ()
        b = _chiral()
        back = b.mirrored(normal="y", position=-3.0).mirrored(normal="y", position=-3.0)
        assert back.bounding_box()[0] == pytest.approx(b.bounding_box()[0])
        assert back.bounding_box()[1] == pytest.approx(b.bounding_box()[1])

    def test_oblique_normal(self):
        _occ()
        # Plane x + y = 0 swaps and negates x and y.
        m = _chiral().mirrored(normal=(1.0, 1.0, 0.0))
        lo, hi = m.bounding_box()
        assert (lo[0], hi[0]) == pytest.approx((-3.0, -2.0))
        assert (lo[1], hi[1]) == pytest.approx((-4.0, -1.0))

    def test_analytic_bbox_matches_occ(self):
        _occ()
        m = _chiral().mirrored(normal="z", position=1.25)
        lo_occ, hi_occ = m.bounding_box()
        lo_an, hi_an = m._analytic_bbox()
        assert lo_an == pytest.approx(lo_occ)
        assert hi_an == pytest.approx(hi_occ)

    def test_copy_returns_original_and_image(self):
        _occ()
        b = _chiral()
        pair = b.mirrored(normal="x", copy=True)
        assert isinstance(pair, list) and len(pair) == 2
        assert pair[0] is b

    def test_copy_unite_builds_the_symmetric_whole(self):
        _occ()
        b = _chiral()
        whole = b.mirrored(normal="x", copy=True, unite=True)
        assert isinstance(whole, Union)
        # Disjoint halves (the brick starts at x = 1), so volumes add.
        assert _volume(whole) == pytest.approx(2.0 * _volume(b))
        lo, hi = whole.bounding_box()
        assert (lo[0], hi[0]) == pytest.approx((-4.0, 4.0))

    def test_unite_without_copy_raises(self):
        with pytest.raises(ValueError, match="copy=True"):
            _chiral().mirrored(normal="x", unite=True)

    def test_group_distributes_and_keeps_materials(self):
        _occ()
        g = Group(_chiral(), Sphere(material=AIR, center=(2, 0, 0), radius=0.5))
        m = g.mirrored(normal="x")
        assert isinstance(m, Group)
        assert [s.material for s in m.members()] == [PEC, AIR]
        assert m.bounding_box()[1][0] == pytest.approx(-1.0)

    def test_method_equals_free_function(self):
        _occ()
        b = _chiral()
        assert b.mirrored(normal="y", position=0.5).bounding_box() == (
            mirror(b, normal="y", position=0.5).bounding_box()
        )

    def test_material_is_inherited(self):
        assert _chiral().mirrored(normal="x").material is PEC

    def test_mirrored_result_keeps_verbs(self):
        _occ()
        m = _chiral().mirrored(normal="x").translated((10.0, 0.0, 0.0))
        assert m.bounding_box()[0][0] == pytest.approx(6.0)

    def test_invalid_normal_raises(self):
        with pytest.raises(ValueError, match=r"mirrored\(normal\)"):
            _chiral().mirrored(normal="q")


# ── axis ergonomics ──────────────────────────────────────────────────────────


class TestAxisNormalization:
    def test_letters(self):
        assert normalize_axis("x") == (1.0, 0.0, 0.0)
        assert normalize_axis("Z") == (0.0, 0.0, 1.0)

    def test_vector_normalised(self):
        d = normalize_axis((0.0, 2.0, 0.0))
        assert d == pytest.approx((0.0, 1.0, 0.0))

    def test_invalid_letter_raises(self):
        with pytest.raises(ValueError, match="axis"):
            normalize_axis("q")

    def test_zero_vector_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            normalize_axis((0.0, 0.0, 0.0))

    def test_cylinder_vector_axis_matches_letter(self):
        _occ()
        by_letter = Cylinder(material=PEC, origin=(0, 0, 0), radius=0.3, height=2.0, axis="y")
        by_vector = Cylinder(
            material=PEC, origin=(0, 0, 0), radius=0.3, height=2.0, axis=(0.0, 5.0, 0.0)
        )
        lo_l, hi_l = by_letter.bounding_box()
        lo_v, hi_v = by_vector.bounding_box()
        assert lo_l == pytest.approx(lo_v)
        assert hi_l == pytest.approx(hi_v)

    def test_cylinder_oblique_axis(self):
        _occ()
        c = Cylinder(material=PEC, origin=(0, 0, 0), radius=0.1, height=1.0, axis=(1.0, 0.0, 1.0))
        lo, hi = c.bounding_box()
        assert hi[0] == pytest.approx(1.0 / math.sqrt(2.0) + 0.1 / math.sqrt(2.0), rel=1e-3)


class TestNegativeHeight:
    def test_cylinder_negative_height_extrudes_down(self):
        _occ()
        up = Cylinder(material=PEC, origin=(0, 0, 0), radius=0.3, height=2.0)
        down = Cylinder(material=PEC, origin=(0, 0, 0), radius=0.3, height=-2.0)
        lo_u, hi_u = up.bounding_box()
        lo_d, hi_d = down.bounding_box()
        assert hi_u[2] == pytest.approx(2.0)
        assert lo_d[2] == pytest.approx(-2.0)
        assert hi_d[2] == pytest.approx(0.0, abs=1e-9)

    def test_cone_negative_height(self):
        _occ()
        c = Cone(
            material=PEC, origin=(0, 0, 0), bottom_radius=0.5, top_radius=0.0, height=-1.0, axis="z"
        )
        lo, hi = c.bounding_box()
        assert lo[2] == pytest.approx(-1.0)
