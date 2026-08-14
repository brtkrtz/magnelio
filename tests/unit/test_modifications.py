"""Tests for chamfer, fillet, extrude, and loft shape modifications."""

import pytest

from magnelio.geo import (
    Brick,
    Cylinder,
    Difference,
    GeometryModel,
    Union,
)
from magnelio.geo.modifications import chamfer, extrude, fillet, loft
from magnelio.geo.transforms import translate
from magnelio.materials.material import Material


@pytest.fixture
def brick():
    return Brick(origin=(0, 0, 0), size=(1, 1, 1), material=Material.pec())


def _volume(shape):
    """Compute volume of a CSG shape via OCC."""
    from OCC.Core.BRepGProp import brepgprop  # noqa: PLC0415
    from OCC.Core.GProp import GProp_GProps  # noqa: PLC0415

    props = GProp_GProps()
    brepgprop.VolumeProperties(shape._occ_shape(), props)
    return abs(props.Mass())


# ── Edge selection ──────────────────────────────────────────────────────────


class TestEdgeSelection:
    def test_get_all_edges_brick(self, brick):
        from magnelio.geo._occ_backend import get_all_edges

        edges = get_all_edges(brick._occ_shape())
        assert len(edges) == 12

    def test_find_nearest_edge(self, brick):
        from magnelio.geo._occ_backend import find_nearest_edge

        edge = find_nearest_edge(brick._occ_shape(), (1.0, 0.5, 0.0))
        assert edge is not None

    def test_find_edges_on_nearest_face_top(self, brick):
        from magnelio.geo._occ_backend import find_edges_on_nearest_face

        edges = find_edges_on_nearest_face(brick._occ_shape(), (0.5, 0.5, 1.0))
        assert len(edges) == 4

    def test_find_edges_on_nearest_face_side(self, brick):
        from magnelio.geo._occ_backend import find_edges_on_nearest_face

        edges = find_edges_on_nearest_face(brick._occ_shape(), (1.0, 0.5, 0.5))
        assert len(edges) == 4

    def test_resolve_edges_no_mode_raises(self, brick):
        from magnelio.geo._occ_backend import resolve_edges

        with pytest.raises(ValueError, match="Exactly one"):
            resolve_edges(brick._occ_shape())

    def test_resolve_edges_multiple_modes_raises(self, brick):
        from magnelio.geo._occ_backend import resolve_edges

        with pytest.raises(ValueError, match="Exactly one"):
            resolve_edges(
                brick._occ_shape(),
                near=(0.5, 0.5, 0),
                face_near=(0.5, 0.5, 1),
            )

    def test_resolve_edges_invalid_string(self, brick):
        from magnelio.geo._occ_backend import resolve_edges

        with pytest.raises(ValueError, match="must be 'all'"):
            resolve_edges(brick._occ_shape(), edges="top")

    def test_resolve_edges_all(self, brick):
        from magnelio.geo._occ_backend import resolve_edges

        edges = resolve_edges(brick._occ_shape(), edges="all")
        assert len(edges) == 12

    def test_resolve_edges_near_single(self, brick):
        from magnelio.geo._occ_backend import resolve_edges

        edges = resolve_edges(brick._occ_shape(), near=(1.0, 0.5, 0.0))
        assert len(edges) == 1

    def test_resolve_edges_near_list(self, brick):
        from magnelio.geo._occ_backend import resolve_edges

        edges = resolve_edges(
            brick._occ_shape(),
            near=[(1.0, 0.5, 0.0), (0.0, 0.5, 0.0)],
        )
        assert len(edges) == 2


# ── Chamfer ─────────────────────────────────────────────────────────────────


class TestChamfer:
    def test_chamfer_single_edge(self, brick):
        c = chamfer(brick, near=(1.0, 0.5, 0.0), distance=0.1)
        assert c.material == brick.material
        vol = _volume(c)
        assert 0.9 < vol < 1.0

    def test_chamfer_face(self, brick):
        c = chamfer(brick, face_near=(0.5, 0.5, 1.0), distance=0.1)
        vol = _volume(c)
        assert vol < 1.0

    def test_chamfer_all_edges(self, brick):
        c = chamfer(brick, edges="all", distance=0.1)
        vol_all = _volume(c)
        c_one = chamfer(brick, near=(1.0, 0.5, 0.0), distance=0.1)
        vol_one = _volume(c_one)
        assert vol_all < vol_one

    def test_chamfer_asymmetric(self, brick):
        c = chamfer(brick, near=(1.0, 0.5, 0.0), distance=(0.1, 0.2))
        vol = _volume(c)
        assert vol < 1.0

    def test_chamfer_multiple_near(self, brick):
        c = chamfer(
            brick,
            near=[(1.0, 0.5, 0.0), (0.0, 0.5, 0.0)],
            distance=0.1,
        )
        vol = _volume(c)
        c_one = chamfer(brick, near=(1.0, 0.5, 0.0), distance=0.1)
        vol_one = _volume(c_one)
        assert vol < vol_one

    def test_chamfer_bounding_box(self, brick):
        c = chamfer(brick, edges="all", distance=0.1)
        lo, hi = c.bounding_box()
        assert lo[0] >= -1e-10
        assert hi[0] <= 1.0 + 1e-10

    def test_chamfer_invalid_edges_value(self, brick):
        with pytest.raises(ValueError):
            chamfer(brick, edges="invalid", distance=0.1)._occ_shape()


# ── Fillet ──────────────────────────────────────────────────────────────────


class TestFillet:
    def test_fillet_single_edge(self, brick):
        f = fillet(brick, near=(1.0, 0.5, 0.0), radius=0.1)
        assert f.material == brick.material
        vol = _volume(f)
        assert vol < 1.0

    def test_fillet_all_edges(self, brick):
        f = fillet(brick, edges="all", radius=0.1)
        vol = _volume(f)
        assert vol < 1.0

    def test_fillet_face(self, brick):
        f = fillet(brick, face_near=(0.5, 0.5, 1.0), radius=0.1)
        vol = _volume(f)
        assert vol < 1.0


# ── Integration ─────────────────────────────────────────────────────────────


class TestIntegration:
    def test_chamfer_on_difference(self):
        mat = Material.pec()
        outer = Brick(origin=(0, 0, 0), size=(2, 2, 2), material=mat)
        inner = Brick(
            origin=(0.5, 0.5, 0.5),
            size=(1, 1, 1),
            material=Material.air(),
        )
        diff = Difference(outer, inner)
        c = chamfer(diff, near=(2.0, 1.0, 0.0), distance=0.1)
        vol = _volume(c)
        assert vol > 0

    def test_chamfer_then_translate(self, brick):
        c = chamfer(brick, near=(1.0, 0.5, 0.0), distance=0.1)
        t = translate(c, (5, 0, 0))
        lo, hi = t.bounding_box()
        assert lo[0] >= 4.9

    def test_fillet_in_geometry_model(self, brick):
        f = fillet(brick, edges="all", radius=0.1)
        model = GeometryModel()
        model.add(f)
        assert len(model) == 1
        lo, hi = model.bounding_box()
        assert lo[0] >= -1e-10


# ── Extrude ─────────────────────────────────────────────────────────────────


class TestExtrude:
    def test_extrude_brick_top_face(self, brick):
        ext = extrude(brick, face_near=(0.5, 0.5, 1.0), vector=(0, 0, 0.5))
        vol = _volume(ext)
        # Extruded piece: 1×1 face × 0.5 height = 0.5
        assert abs(vol - 0.5) < 0.01

    def test_extrude_brick_side_face(self, brick):
        ext = extrude(brick, face_near=(1.0, 0.5, 0.5), vector=(1, 0, 0))
        vol = _volume(ext)
        # Extruded piece: 1×1 face × 1.0 length = 1.0
        assert abs(vol - 1.0) < 0.01

    def test_extrude_cylinder_top(self):
        import math

        mat = Material.pec()
        cyl = Cylinder(origin=(0, 0, 0), radius=1.0, height=2.0, material=mat)
        ext = extrude(cyl, face_near=(0, 0, 2.0), vector=(0, 0, 1.0))
        vol = _volume(ext)
        expected = math.pi * 1.0**2 * 1.0
        assert abs(vol - expected) < 0.1

    def test_extrude_material_default(self, brick):
        ext = extrude(brick, face_near=(0.5, 0.5, 1.0), vector=(0, 0, 1))
        assert ext.material == brick.material

    def test_extrude_material_override(self, brick):
        air = Material.air()
        ext = extrude(brick, face_near=(0.5, 0.5, 1.0), vector=(0, 0, 1), material=air)
        assert ext.material == air

    def test_extrude_bounding_box(self, brick):
        ext = extrude(brick, face_near=(0.5, 0.5, 1.0), vector=(0, 0, 2.0))
        lo, hi = ext.bounding_box()
        assert lo[2] >= 1.0 - 1e-10
        assert hi[2] <= 3.0 + 1e-10

    def test_extrude_then_translate(self, brick):
        ext = extrude(brick, face_near=(0.5, 0.5, 1.0), vector=(0, 0, 1))
        moved = translate(ext, (10, 0, 0))
        lo, hi = moved.bounding_box()
        assert lo[0] >= 9.9

    def test_extrude_in_geometry_model(self, brick):
        ext = extrude(brick, face_near=(0.5, 0.5, 1.0), vector=(0, 0, 1))
        model = GeometryModel()
        model.add(ext)
        assert len(model) == 1

    def test_extrude_union_with_original(self, brick):
        ext = extrude(brick, face_near=(0.5, 0.5, 1.0), vector=(0, 0, 1))
        combined = Union(brick, ext)
        vol = _volume(combined)
        assert abs(vol - 2.0) < 0.01


# ── Loft ────────────────────────────────────────────────────────────────────


class TestLoft:
    def test_loft_brick_to_brick(self):
        mat = Material.pec()
        a = Brick(origin=(0, 0, 0), size=(2, 2, 1), material=mat)
        b = Brick(origin=(0.25, 0.25, 3), size=(1.5, 1.5, 1), material=mat)
        transition = loft(a, (1, 1, 1), b, (1, 1, 3), material=mat)
        vol = _volume(transition)
        assert vol > 0

    def test_loft_cylinder_to_brick(self):
        mat = Material.pec()
        cyl = Cylinder(origin=(0, 0, 0), radius=1.0, height=2.0, material=mat)
        rect = Brick(origin=(-1, -1, 4), size=(2, 2, 1), material=mat)
        transition = loft(cyl, (0, 0, 2), rect, (0, 0, 4), material=mat)
        vol = _volume(transition)
        assert vol > 0
        lo, hi = transition.bounding_box()
        assert lo[2] >= 2.0 - 0.1
        assert hi[2] <= 4.0 + 0.1

    def test_loft_blend_modes(self):
        mat = Material.pec()
        a = Brick(origin=(0, 0, 0), size=(2, 2, 1), material=mat)
        b = Brick(origin=(0.25, 0.25, 3), size=(1.5, 1.5, 1), material=mat)
        smooth = loft(a, (1, 1, 1), b, (1, 1, 3), material=mat, blend="spline")
        ruled = loft(a, (1, 1, 1), b, (1, 1, 3), material=mat, blend="ruled")
        # Both should produce valid solids
        assert _volume(smooth) > 0
        assert _volume(ruled) > 0

    def test_loft_material_default(self):
        mat_a = Material.pec()
        mat_b = Material.air()
        a = Brick(origin=(0, 0, 0), size=(1, 1, 1), material=mat_a)
        b = Brick(origin=(0, 0, 3), size=(1, 1, 1), material=mat_b)
        transition = loft(a, (0.5, 0.5, 1), b, (0.5, 0.5, 3))
        assert transition.material == mat_a

    def test_loft_material_override(self):
        mat = Material.pec()
        air = Material.air()
        a = Brick(origin=(0, 0, 0), size=(1, 1, 1), material=mat)
        b = Brick(origin=(0, 0, 3), size=(1, 1, 1), material=mat)
        transition = loft(a, (0.5, 0.5, 1), b, (0.5, 0.5, 3), material=air)
        assert transition.material == air

    def test_loft_in_geometry_model(self):
        mat = Material.pec()
        cyl = Cylinder(origin=(0, 0, 0), radius=1.0, height=2.0, material=mat)
        rect = Brick(origin=(-1, -1, 4), size=(2, 2, 1), material=mat)
        transition = loft(cyl, (0, 0, 2), rect, (0, 0, 4), material=mat)
        model = GeometryModel()
        model.add(transition)
        assert len(model) == 1


class TestTangentBlend:
    """blend='tangent' leaves both faces along their own normal.

    The two bricks face different directions on purpose: A ends on a
    zmin face (outward normal -z), B on a ymin face (outward normal -y),
    so a straight loft between them has to run diagonally and creases at
    both joints, while a tangent blend has to turn a corner.
    """

    # A's end face: z = 0, spanning y in [0, 4] mm.
    A_FACE = (0.0, 2e-3, 0.0)
    # B's end face: y = 20 mm, spanning z in [-14, -10] mm.
    B_FACE = (0.0, 20e-3, -12e-3)

    def _pair(self):
        mat = Material.pec()
        a = Brick(origin=(-2e-3, 0.0, 0.0), size=(4e-3, 4e-3, 10e-3), material=mat)
        b = Brick(origin=(-2e-3, 20e-3, -14e-3), size=(4e-3, 10e-3, 4e-3), material=mat)
        return a, b

    def _blend(self, **kwargs):
        a, b = self._pair()
        return loft(a, self.A_FACE, b, self.B_FACE, material=Material.pec(), **kwargs)

    def test_leaves_the_start_face_along_its_normal(self):
        """The sideways drift off the start face grows as the square of depth.

        That exponent *is* the right angle: a blend leaving squarely has
        no first-order sideways motion left, so its drift is pure
        curvature, O(d**2).  Any residual tilt would contribute an O(d)
        term and pull the measured exponent towards 1.  Doubling the
        depth therefore has to roughly quadruple the drift, not double it.
        """
        from magnelio.geo import Intersection

        blend = self._blend(blend="tangent")

        def drift(depth):
            slab = Brick(
                origin=(-5e-3, -5e-3, -depth), size=(10e-3, 30e-3, depth), material=Material.pec()
            )
            _, hi = Intersection(blend, slab).bounding_box()
            return hi[1] - 4e-3  # the start face ends at y = 4 mm

        near, mid, far = drift(0.5e-3), drift(1e-3), drift(2e-3)
        assert 0.0 < near < 0.05e-3  # 0.5 mm along a 21.6 mm span
        assert mid / near == pytest.approx(4.0, abs=0.6)
        assert far / mid == pytest.approx(4.0, abs=0.6)

    def test_reaches_both_faces(self):
        blend = self._blend(blend="tangent")
        lo, hi = blend.bounding_box()
        assert hi[2] == pytest.approx(0.0, abs=1e-6)  # touches A at z = 0
        assert hi[1] == pytest.approx(20e-3, abs=1e-6)  # touches B at y = 20 mm

    def test_tension_controls_the_reach(self):
        volumes = [self._blend(blend="tangent", tension=t).volume() for t in (0.2, 1 / 3, 0.6)]
        assert volumes[0] < volumes[1] < volumes[2]

    def test_tension_pair_acts_per_end(self):
        """An asymmetric pair is not the same solid as its mirror."""
        start_stiff = self._blend(blend="tangent", tension=(0.6, 0.2))
        end_stiff = self._blend(blend="tangent", tension=(0.2, 0.6))
        assert start_stiff.volume() != pytest.approx(end_stiff.volume(), rel=1e-6)
        assert start_stiff.volume() > 0.0
        assert end_stiff.volume() > 0.0

    def test_analytic_box_contains_a_stiff_blend(self):
        """The padded box has to keep up with the bow a big tension adds."""
        blend = self._blend(blend="tangent", tension=0.9)
        lo, hi = blend._analytic_bbox()
        occ_lo, occ_hi = blend.bounding_box()
        gap = 1e-7
        assert all(lo[i] <= occ_lo[i] + gap for i in range(3))
        assert all(hi[i] >= occ_hi[i] - gap for i in range(3))

    def test_unknown_blend_is_rejected(self):
        with pytest.raises(ValueError, match="blend must be one of"):
            self._blend(blend="smooth")

    def test_tension_without_tangent_is_rejected(self):
        with pytest.raises(ValueError, match="no meaning for blend"):
            self._blend(blend="spline", tension=0.5)

    def test_negative_tension_is_rejected(self):
        with pytest.raises(ValueError, match="tension must be positive"):
            self._blend(blend="tangent", tension=-0.2)

    def test_n_ary_loft_points_at_the_verb_for_tangent(self):
        from magnelio.geo import Face, Loft

        def square(half, z):
            return Face(
                normal="z",
                points=[(-half, -half), (half, -half), (half, half), (-half, half)],
                position=z,
            )

        with pytest.raises(ValueError, match="use Shape.lofted"):
            Loft(square(5e-3, 0.0), square(2e-3, 10e-3), blend="tangent")


# ── Shell / thicken ─────────────────────────────────────────────────────────


class TestShelled:
    """shelled() hollows a solid, keeping its outer surface."""

    A, T = 20e-3, 2e-3

    def _box(self):
        return Brick(origin=(0, 0, 0), size=(self.A, self.A, self.A), material=Material.pec())

    def test_sealed_void(self):
        a, t = self.A, self.T
        shelled = self._box().shelled(thickness=t)
        assert _volume(shelled) == pytest.approx(a**3 - (a - 2 * t) ** 3, rel=1e-9)

    def test_one_opening(self):
        a, t = self.A, self.T
        shelled = self._box().shelled(thickness=t, opening_face_near=(a / 2, a / 2, a))
        assert _volume(shelled) == pytest.approx(a**3 - (a - 2 * t) ** 2 * (a - t), rel=1e-9)

    def test_two_openings_make_a_tube(self):
        a, t = self.A, self.T
        shelled = self._box().shelled(
            thickness=t, opening_face_near=[(a / 2, a / 2, 0), (a / 2, a / 2, a)]
        )
        assert _volume(shelled) == pytest.approx(a**3 - (a - 2 * t) ** 2 * a, rel=1e-9)

    def test_curved_walls(self):
        import math

        r, h, t = 10e-3, 30e-3, 1e-3
        cyl = Cylinder(radius=r, height=h, material=Material.pec())
        shelled = cyl.shelled(thickness=t, opening_face_near=(0, 0, h))
        expected = math.pi * r * r * h - math.pi * (r - t) ** 2 * (h - t)
        assert _volume(shelled) == pytest.approx(expected, rel=1e-9)

    def test_openings_deduplicate(self):
        a, t = self.A, self.T
        one = self._box().shelled(thickness=t, opening_face_near=(a / 2, a / 2, a))
        twice = self._box().shelled(
            thickness=t, opening_face_near=[(a / 2, a / 2, a), (a / 2, a / 2, a * 0.99)]
        )
        assert _volume(twice) == pytest.approx(_volume(one), rel=1e-12)

    def test_material_and_box_are_kept(self):
        shelled = self._box().shelled(thickness=self.T)
        assert shelled.material == self._box().material
        assert shelled._analytic_bbox() == self._box()._analytic_bbox()

    def test_wall_that_fills_the_body_is_reported(self):
        a = self.A
        with pytest.raises(RuntimeError, match="leaves no cavity"):
            _volume(self._box().shelled(thickness=0.75 * a, opening_face_near=(a / 2, a / 2, a)))

    def test_sealed_wall_too_thick_is_reported(self):
        with pytest.raises(RuntimeError, match="Hollowing the solid failed"):
            _volume(self._box().shelled(thickness=0.75 * self.A))

    def test_sheet_is_sent_to_thickened(self):
        from magnelio.geo import Face

        sheet = Face(normal="z", points=[(0, 0), (1e-3, 0), (1e-3, 1e-3)])
        with pytest.raises(TypeError, match="thickened"):
            sheet.shelled(thickness=1e-4)

    def test_thickness_must_be_positive(self):
        with pytest.raises(ValueError, match="must be positive"):
            self._box().shelled(thickness=0.0)


class TestThickened:
    """thickened() grows a planar sheet into a slab."""

    W, H, T = 10e-3, 4e-3, 35e-6

    def _sheet(self, material=None):
        from magnelio.geo import Face

        return Face(
            normal="z",
            points=[(0, 0), (self.W, 0), (self.W, self.H), (0, self.H)],
            position=1e-3,
            material=material,
        )

    def test_volume_is_area_times_thickness(self):
        slab = self._sheet().thickened(thickness=self.T, material=Material.pec())
        assert _volume(slab) == pytest.approx(self.W * self.H * self.T, rel=1e-9)

    @pytest.mark.parametrize(
        ("direction", "lo", "hi"),
        [
            ("forward", 1e-3, 1e-3 + T),
            ("backward", 1e-3 - T, 1e-3),
            ("symmetric", 1e-3 - T / 2, 1e-3 + T / 2),
        ],
    )
    def test_direction_places_the_slab(self, direction, lo, hi):
        slab = self._sheet().thickened(
            thickness=self.T, direction=direction, material=Material.pec()
        )
        box_lo, box_hi = slab.bounding_box()
        assert box_lo[2] == pytest.approx(lo, abs=1e-12)
        assert box_hi[2] == pytest.approx(hi, abs=1e-12)

    def test_covered_curve_can_be_thickened(self):
        from magnelio.geo import Path

        outline = (
            Path((0, 0, 0))
            .line_to((self.W, 0, 0))
            .line_to((self.W, self.H, 0))
            .line_to((0, self.H, 0))
            .closed()
        )
        slab = outline.covered().thickened(thickness=self.T, material=Material.pec())
        assert _volume(slab) == pytest.approx(self.W * self.H * self.T, rel=1e-9)

    def test_sheet_material_is_inherited(self):
        mat = Material.pec()
        assert self._sheet(mat).thickened(thickness=self.T).material == mat

    def test_construction_profile_needs_a_material(self):
        with pytest.raises(ValueError, match="requires an explicit material"):
            self._sheet().thickened(thickness=self.T)

    def test_solid_is_sent_to_shelled(self):
        solid = Brick(origin=(0, 0, 0), size=(1e-3, 1e-3, 1e-3), material=Material.pec())
        with pytest.raises(TypeError, match="shelled"):
            solid.thickened(thickness=1e-4, material=Material.pec())

    def test_unknown_direction_rejected(self):
        with pytest.raises(ValueError, match="forward"):
            self._sheet().thickened(thickness=self.T, direction="up", material=Material.pec())


# ── Loft over free sections ─────────────────────────────────────────────────


class TestLoftSections:
    """Loft() interpolates a series of profiles."""

    def _square(self, half, offset):
        from magnelio.geo import Face

        return Face(
            normal="z",
            points=[(-half, -half), (half, -half), (half, half), (-half, half)],
            position=offset,
        )

    def test_two_sections_match_the_frustum(self):
        import math

        from magnelio.geo import Loft

        a1, a2, h = (10e-3) ** 2, (4e-3) ** 2, 10e-3
        solid = Loft(
            self._square(5e-3, 0.0), self._square(2e-3, h), blend="ruled", material=Material.pec()
        )
        expected = h / 3 * (a1 + a2 + math.sqrt(a1 * a2))
        assert _volume(solid) == pytest.approx(expected, rel=1e-9)

    def test_three_sections_stack_two_frusta(self):
        import math

        from magnelio.geo import Loft

        h = 5e-3
        solid = Loft(
            self._square(5e-3, 0.0),
            self._square(2e-3, h),
            self._square(4e-3, 2 * h),
            blend="ruled",
            material=Material.pec(),
        )

        def frustum(a, b):
            return h / 3 * (a**2 + b**2 + math.sqrt(a**2 * b**2))

        expected = frustum(10e-3, 4e-3) + frustum(4e-3, 8e-3)
        assert _volume(solid) == pytest.approx(expected, rel=1e-9)

    def test_matches_the_lofted_verb_on_the_same_profiles(self):
        """The n-ary form reproduces the two-solid verb it generalises."""
        from magnelio.geo import Loft

        a = Brick(origin=(-5e-3, -5e-3, 0), size=(10e-3, 10e-3, 1e-3), material=Material.pec())
        b = Brick(origin=(-2e-3, -2e-3, 10e-3), size=(4e-3, 4e-3, 1e-3), material=Material.pec())
        verb = loft(a, (0, 0, 1e-3), b, (0, 0, 10e-3), blend="ruled")
        n_ary = Loft(
            self._square(5e-3, 1e-3),
            self._square(2e-3, 10e-3),
            blend="ruled",
            material=Material.pec(),
        )
        assert _volume(n_ary) == pytest.approx(_volume(verb), rel=1e-9)

    def test_closed_curve_sections_accepted(self):
        from magnelio.geo import Loft, Path

        def ring(half, z):
            return (
                Path((-half, -half, z))
                .line_to((half, -half, z))
                .line_to((half, half, z))
                .line_to((-half, half, z))
                .closed()
            )

        solid = Loft(ring(5e-3, 0.0), ring(2e-3, 10e-3), blend="ruled", material=Material.pec())
        assert _volume(solid) > 0.0

    def test_open_curve_section_rejected(self):
        from magnelio.geo import Curve, Loft

        with pytest.raises(ValueError, match="open curve"):
            Loft(self._square(5e-3, 0.0), Curve.polyline([(0, 0, 0), (1e-3, 0, 0)]))

    def test_solid_section_points_at_the_verb(self):
        from magnelio.geo import Loft

        with pytest.raises(TypeError, match="lofted"):
            Loft(self._square(5e-3, 0.0), Brick())

    def test_needs_two_sections(self):
        from magnelio.geo import Loft

        with pytest.raises(ValueError, match="at least 2 cross-sections"):
            Loft(self._square(5e-3, 0.0))

    def test_ruled_box_contains_the_solid(self):
        from magnelio.geo import Loft

        solid = Loft(
            self._square(5e-3, 0.0),
            self._square(2e-3, 10e-3),
            blend="ruled",
            material=Material.pec(),
        )
        lo, hi = solid._analytic_bbox()
        occ_lo, occ_hi = solid.bounding_box()
        # A lofted surface is tolerant geometry, so the kernel pads its
        # box by Precision::Confusion(); the analytic box bounds the
        # true solid, not that padding.
        gap = 1e-7
        assert all(lo[i] <= occ_lo[i] + gap for i in range(3))
        assert all(hi[i] >= occ_hi[i] - gap for i in range(3))

    def test_construction_loft_is_rejected_by_the_model(self):
        from magnelio.geo import Loft

        solid = Loft(self._square(5e-3, 0.0), self._square(2e-3, 10e-3), blend="ruled")
        model = GeometryModel()
        with pytest.raises(ValueError, match="carries no material"):
            model.add(solid)
