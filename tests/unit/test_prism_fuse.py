"""The planar fuse route: prisms meet in their plane, the rest in space.

``boolean_union`` fuses operands that are prisms along one axis over the
same interval through their caps and raises them once; everything else
meets the general fuser only inside clusters of interfering bounding
boxes.  The point set must equal the general fuse, the seams must be
gone, and nothing may be lost on the way.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio import geo
from magnelio.geo import _occ_backend as backend
from magnelio.geo import _prism_fuse
from magnelio.geo._prism_fuse import cluster_boxes, fuse_faces_tree, prism_candidates
from magnelio.materials import Material

MM = 1e-3


def _volume(shape) -> float:
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps

    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    return props.Mass()


def _count(shape, kind) -> int:
    from OCC.Core.TopExp import TopExp_Explorer

    explorer = TopExp_Explorer(shape, kind)
    n = 0
    while explorer.More():
        n += 1
        explorer.Next()
    return n


def _faces(shape) -> int:
    from OCC.Core.TopAbs import TopAbs_FACE

    return _count(shape, TopAbs_FACE)


def _solids(shape) -> int:
    from OCC.Core.TopAbs import TopAbs_SOLID

    return _count(shape, TopAbs_SOLID)


def _general_fuse(shapes):
    """The kernel's plain N-ary fuse — the reference point set."""
    occ = backend._require_occ()
    return backend._run_bop(occ["Fuse"], shapes[:1], shapes[1:])


def _occ(*shapes):
    return [s._occ_shape(1.0) for s in shapes]


class TestClusterBoxes:
    def test_touching_and_disjoint(self):
        boxes = np.array(
            [
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 0.5, 2.0, 1.5],  # shares an edge with the first
                [5.0, 5.0, 6.0, 6.0],  # alone
            ]
        )
        groups = sorted(sorted(g) for g in cluster_boxes(boxes, 0.0))
        assert groups == [[0, 1], [2]]

    def test_transitive_chain(self):
        boxes = np.array([[i, 0.0, i + 1.2, 1.0] for i in range(6)])
        assert len(cluster_boxes(boxes, 0.0)) == 1

    def test_tolerance_bridges_a_gap(self):
        boxes = np.array([[0.0, 0.0, 1.0, 1.0], [1.05, 0.0, 2.0, 1.0]])
        assert len(cluster_boxes(boxes, 0.0)) == 2
        assert len(cluster_boxes(boxes, 0.1)) == 1

    def test_three_dimensional_rows(self):
        boxes = np.array([[0, 0, 0, 1, 1, 1], [0.5, 0.5, 2.0, 1.5, 1.5, 3.0]], dtype=float)
        assert len(cluster_boxes(boxes, 0.0)) == 2  # separated in the third axis only


class TestPrismCandidates:
    def test_brick_is_a_prism_on_all_axes(self):
        (shape,) = _occ(geo.Brick(origin=(0, 0, 1 * MM), size=(3 * MM, 2 * MM, 0.5 * MM)))
        cand = prism_candidates(shape)
        assert set(cand) == {0, 1, 2}
        low, high, caps = cand[2]
        assert (low, high) == pytest.approx((1 * MM, 1.5 * MM))
        assert len(caps) == 1

    def test_cylinder_only_along_its_axis(self):
        (shape,) = _occ(geo.Cylinder(origin=(0, 0, 0), radius=MM, height=2 * MM, axis="x"))
        assert set(prism_candidates(shape)) == {0}

    def test_sphere_is_no_prism(self):
        (shape,) = _occ(geo.Sphere(center=(0, 0, 0), radius=MM))
        assert prism_candidates(shape) == {}

    def test_bored_brick_is_a_prism_along_the_bore(self):
        brick = geo.Brick(origin=(0, 0, 0), size=(4 * MM, 3 * MM, 3 * MM))
        bore = geo.Cylinder(
            origin=(-MM, 1.5 * MM, 1.5 * MM), radius=0.5 * MM, height=6 * MM, axis="x"
        )
        (shape,) = _occ(geo.Difference(brick, bore))
        assert set(prism_candidates(shape)) == {0}

    def test_stepped_solid_is_no_prism(self):
        step = geo.Union(
            geo.Brick(origin=(0, 0, 0), size=(2 * MM, MM, MM)),
            geo.Brick(origin=(MM, 0, 0), size=(MM, MM, 2 * MM)),
        )
        (shape,) = _occ(step)
        assert 2 not in prism_candidates(shape)


def _area(shape) -> float:
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps

    props = GProp_GProps()
    brepgprop.SurfaceProperties(shape, props)
    return props.Mass()


def _bottom_caps(shapes) -> list:
    from OCC.Core.TopAbs import TopAbs_FORWARD

    caps = []
    for shape in _occ(*shapes):
        caps.extend(face.Oriented(TopAbs_FORWARD) for face in prism_candidates(shape)[2][2])
    return caps


def _fuse_faces(faces):
    return backend.unify_same_domain(_general_fuse(faces))


def _comb(n: int) -> list:
    """A spine with *n* overlapping teeth — one connected copper network."""
    teeth = [
        geo.Brick(origin=(k * MM, 0, 0), size=(0.6 * MM, 3 * MM, 0.1 * MM), material="pec")
        for k in range(n)
    ]
    spine = geo.Brick(origin=(-0.5 * MM, -0.2 * MM, 0), size=((n + 1) * MM, MM, 0.1 * MM))
    return [spine, *teeth]


class TestFuseFacesTree:
    def test_matches_the_flat_fuse(self):
        caps = _bottom_caps(_comb(40))  # more than one leaf
        tree = fuse_faces_tree(caps, _fuse_faces)
        flat = _fuse_faces(caps)
        assert _faces(tree) == _faces(flat) == 1
        assert _area(tree) == pytest.approx(_area(flat), rel=1e-12)
        expected = (40 * 0.6 * 3 + 41 * 1 - 40 * 0.6 * 0.8) * MM**2  # teeth, spine, overlaps
        assert _area(tree) == pytest.approx(expected, rel=1e-9)

    def test_small_sets_take_one_call(self):
        caps = _bottom_caps(_comb(5))
        calls = []

        def counting(faces):
            calls.append(len(faces))
            return _fuse_faces(faces)

        fuse_faces_tree(caps, counting)
        assert calls == [6]
        (face,) = caps[:1]
        assert fuse_faces_tree([face], counting) is face
        assert calls == [6]

    def test_bisects_by_the_axis_of_largest_spread(self):
        caps = _bottom_caps(_comb(9))
        calls = []

        def counting(faces):
            calls.append(len(faces))
            return _fuse_faces(faces)

        tree = fuse_faces_tree(caps, counting, leaf=3)
        # 10 caps → 5 + 5 → (2 + 3) + (2 + 3): four leaves, three pair fuses
        assert sorted(calls) == [2, 2, 2, 2, 2, 3, 3]
        assert _faces(tree) == 1
        assert _area(tree) == pytest.approx(_area(_fuse_faces(caps)), rel=1e-12)

    def test_union_takes_the_tree(self, monkeypatch):
        monkeypatch.setattr(_prism_fuse, "_FUSE_TREE_LEAF", 4)
        parts = _comb(12)
        fused = geo.Union(*parts)._occ_shape(1.0)
        assert _solids(fused) == 1
        assert _volume(fused) == pytest.approx(_volume(_general_fuse(_occ(*parts))), rel=1e-9)


class TestUnionPointSet:
    def test_overlapping_strips_lose_their_seams(self):
        a = geo.Brick(origin=(0, 0, 0), size=(5 * MM, MM, 0.1 * MM), material="pec")
        b = geo.Brick(origin=(4 * MM, 0, 0), size=(MM, 4 * MM, 0.1 * MM), material="pec")
        fused = geo.Union(a, b)._occ_shape(1.0)
        assert _volume(fused) == pytest.approx((5 * 1 + 1 * 4 - 1 * 1) * MM**2 * 0.1 * MM)
        assert _volume(fused) == pytest.approx(_volume(_general_fuse(_occ(a, b))))
        assert _solids(fused) == 1
        assert _faces(fused) == 8  # two caps and the six walls of an L

    def test_touching_strips_become_one_solid(self):
        a = geo.Brick(origin=(0, 0, 0), size=(MM, MM, 0.1 * MM))
        b = geo.Brick(origin=(MM, 0, 0), size=(MM, MM, 0.1 * MM))
        fused = geo.Union(a, b)._occ_shape(1.0)
        assert _volume(fused) == pytest.approx(2 * MM**2 * 0.1 * MM)
        assert _solids(fused) == 1
        assert _faces(fused) == 6

    def test_round_pad_keeps_its_arc(self):
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.GeomAbs import GeomAbs_Cylinder
        from OCC.Core.TopAbs import TopAbs_FACE
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopoDS import topods

        strip = geo.Brick(origin=(0, -0.2 * MM, 0), size=(3 * MM, 0.4 * MM, 0.05 * MM))
        pad = geo.Cylinder(origin=(3 * MM, 0, 0), radius=MM, height=0.05 * MM)
        fused = geo.Union(strip, pad)._occ_shape(1.0)
        assert _volume(fused) == pytest.approx(_volume(_general_fuse(_occ(strip, pad))))
        kinds = set()
        explorer = TopExp_Explorer(fused, TopAbs_FACE)
        while explorer.More():
            kinds.add(BRepAdaptor_Surface(topods.Face(explorer.Current())).GetType())
            explorer.Next()
        assert GeomAbs_Cylinder in kinds

    def test_two_levels_meet_in_space(self):
        lower = [
            geo.Brick(origin=(0, 0, 0), size=(3 * MM, MM, MM)),
            geo.Brick(origin=(2 * MM, 0, 0), size=(3 * MM, MM, MM)),
        ]
        upper = geo.Brick(origin=(MM, 0, MM), size=(MM, MM, MM))  # sits on the lower pair
        fused = geo.Union(*lower, upper)._occ_shape(1.0)
        assert _volume(fused) == pytest.approx((5 + 1) * MM**3)
        assert _solids(fused) == 1

    def test_non_prism_operand_goes_through_the_general_fuser(self):
        bricks = [
            geo.Brick(origin=(0, 0, 0), size=(2 * MM, MM, MM)),
            geo.Brick(origin=(MM, 0, 0), size=(2 * MM, MM, MM)),
        ]
        ball = geo.Sphere(center=(1.5 * MM, 0.5 * MM, MM), radius=0.6 * MM)
        fused = geo.Union(*bricks, ball)._occ_shape(1.0)
        reference = _general_fuse(_occ(*bricks, ball))
        assert _volume(fused) == pytest.approx(_volume(reference), rel=1e-9)
        assert _solids(fused) == 1

    def test_disjoint_operands_are_kept_side_by_side(self, monkeypatch):
        calls = []
        original = backend._run_bop
        monkeypatch.setattr(backend, "_run_bop", lambda *a: calls.append(1) or original(*a))
        a = geo.Brick(origin=(0, 0, 0), size=(MM, MM, MM))
        b = geo.Brick(origin=(5 * MM, 0, 0), size=(MM, MM, MM))
        fused = geo.Union(a, b)._occ_shape(1.0)
        assert calls == []
        assert _solids(fused) == 2
        assert _volume(fused) == pytest.approx(2 * MM**3)

    def test_compound_operand_keeps_its_untouched_solid(self):
        """One solid of a compound meets a strip, the other meets nothing."""
        pair = geo.Union(
            geo.Brick(origin=(0, 0, 0), size=(MM, MM, 0.1 * MM)),
            geo.Brick(origin=(5 * MM, 0, 0), size=(MM, MM, 0.1 * MM)),
        )
        bridge = geo.Brick(origin=(0.5 * MM, 0, 0), size=(MM, MM, 0.1 * MM))
        fused = geo.Union(pair, bridge)._occ_shape(1.0)
        assert _volume(fused) == pytest.approx((1.5 + 1) * MM**2 * 0.1 * MM)
        assert _solids(fused) == 2

    def test_vertical_plates_fuse_along_x(self):
        a = geo.Brick(origin=(0, 0, 0), size=(0.1 * MM, 3 * MM, 3 * MM))
        b = geo.Brick(origin=(0, 2 * MM, 2 * MM), size=(0.1 * MM, 3 * MM, 3 * MM))
        fused = geo.Union(a, b)._occ_shape(1.0)
        assert _volume(fused) == pytest.approx((9 + 9 - 1) * MM**2 * 0.1 * MM)
        assert _faces(fused) == 10  # two caps and the eight walls of the outline

    def test_difference_tools_take_the_same_route(self):
        air = geo.Brick(origin=(0, 0, 0), size=(10 * MM, 10 * MM, 2 * MM), material="air")
        strips = [
            geo.Brick(origin=(i * MM, 0, 0), size=(1.5 * MM, 5 * MM, 0.1 * MM), material="pec")
            for i in range(6)
        ]
        cut = geo.Difference(air, *strips)._occ_shape(1.0)
        assert _volume(cut) == pytest.approx((200 - 6.5 * 5 * 0.1) * MM**3)


def _pairwise_reference(shapes_with_material, library, scale=1.0):
    """The former loop: accumulated higher coverage, subtracted pairwise."""
    effective = None
    higher = None
    for shape_obj, mat_id in reversed(shapes_with_material):
        occ_shape = shape_obj._occ_shape(scale)
        if library[mat_id].is_pec:
            contribution = (
                occ_shape if higher is None else backend.boolean_difference(occ_shape, higher)
            )
            effective = (
                contribution if effective is None else _general_fuse([effective, contribution])
            )
        higher = occ_shape if higher is None else _general_fuse([higher, occ_shape])
    return effective


class TestEffectivePecSolid:
    PEC = Material(name="pec", is_pec=True)
    AIR = Material(name="air")

    def _library(self):
        return {0: self.AIR, 1: self.PEC}

    def test_later_air_overrides_earlier_metal(self):
        metal = geo.Brick(origin=(0, 0, 0), size=(2 * MM, MM, MM))
        air = geo.Brick(origin=(MM, 0, 0), size=(2 * MM, MM, MM))
        solid = backend.build_effective_pec_solid([(metal, 1), (air, 0)], self._library())
        assert _volume(solid) == pytest.approx(MM**3)

    def test_later_metal_wins_over_earlier_air(self):
        air = geo.Brick(origin=(0, 0, 0), size=(3 * MM, MM, MM))
        metal = geo.Brick(origin=(MM, 0, 0), size=(MM, MM, MM))
        solid = backend.build_effective_pec_solid([(air, 0), (metal, 1)], self._library())
        assert _volume(solid) == pytest.approx(MM**3)

    def test_fully_covered_metal_contributes_nothing(self):
        metal = geo.Brick(origin=(MM, 0, 0), size=(MM, MM, MM))
        air = geo.Brick(origin=(0, 0, 0), size=(3 * MM, MM, MM))
        solid = backend.build_effective_pec_solid([(metal, 1), (air, 0)], self._library())
        assert solid is None or _volume(solid) == pytest.approx(0.0, abs=1e-30)

    def test_no_metal_gives_none(self):
        air = geo.Brick(origin=(0, 0, 0), size=(MM, MM, MM))
        assert backend.build_effective_pec_solid([(air, 0)], self._library()) is None
        assert backend.build_effective_pec_solid([], self._library()) is None

    def test_matches_the_pairwise_reference_on_a_row(self):
        """A row of posts under a ribbon, an air cut through it, a metal lid on top."""
        shapes = []
        for i in range(8):
            shapes.append((geo.Brick(origin=(i * MM, 0, 0), size=(0.6 * MM, 0.6 * MM, 2 * MM)), 1))
        shapes.append(
            (geo.Brick(origin=(0, 0.2 * MM, 2 * MM), size=(8 * MM, 0.2 * MM, 0.1 * MM)), 1)
        )
        shapes.append((geo.Brick(origin=(3 * MM, 0, 0), size=(1.5 * MM, MM, 3 * MM)), 0))
        shapes.append((geo.Brick(origin=(2 * MM, 0, 1.5 * MM), size=(2 * MM, MM, MM)), 1))
        library = self._library()
        new = backend.build_effective_pec_solid(shapes, library)
        reference = _pairwise_reference(shapes, library)
        assert _volume(new) == pytest.approx(_volume(reference), rel=1e-12)

    def test_distant_shapes_are_not_cut_against_each_other(self, monkeypatch):
        calls = []
        original = backend.boolean_difference_many
        monkeypatch.setattr(
            backend, "boolean_difference_many", lambda b, t: calls.append(len(t)) or original(b, t)
        )
        near = geo.Brick(origin=(0, 0, 0), size=(MM, MM, MM))
        far = geo.Brick(origin=(9 * MM, 0, 0), size=(MM, MM, MM))
        lid = geo.Brick(origin=(9 * MM, 0, MM), size=(MM, MM, MM))
        backend.build_effective_pec_solid([(near, 1), (far, 1), (lid, 0)], self._library())
        assert calls == [1]  # only `far` reaches the lid; `near` is cut against nothing


def test_effective_pec_solid_cuts_only_with_non_pec_tools(monkeypatch):
    """A higher PEC shape is in the union anyway; only non-PEC shapes are tools."""
    from magnelio.geo import _occ_backend as ob

    pec, air = Material.pec(), Material.air()
    base = geo.Brick(origin=(0, 0, 0), size=(4 * MM, 4 * MM, 4 * MM), material=pec)
    metal = geo.Brick(origin=(1 * MM, 1 * MM, 1 * MM), size=(2 * MM, 2 * MM, 5 * MM), material=pec)
    pocket = geo.Brick(origin=(2 * MM, 2 * MM, 2 * MM), size=(3 * MM, 3 * MM, 3 * MM), material=air)
    tools_seen = []
    original = ob.boolean_difference_many

    def counting(shape, tools):
        tools_seen.append(len(tools))
        return original(shape, tools)

    monkeypatch.setattr(ob, "boolean_difference_many", counting)
    solid = ob.build_effective_pec_solid([(base, 1), (metal, 1), (pocket, 2)], {1: pec, 2: air})
    assert tools_seen == [1, 1]  # the pocket only, for each PEC shape
    expected = 63 * MM**3  # 64 + 8 above the base, minus the pocket (8 + 3 - 2)
    assert abs(_volume(solid) - expected) <= 1e-9 * expected


class TestContainedPecContributions:
    """A PEC shape inside a box-shaped PEC contribution is not fused."""

    PEC = Material(name="pec", is_pec=True)
    AIR = Material(name="air")
    SUBSTRATE = Material(name="alumina")

    def _library(self):
        return {0: self.AIR, 1: self.PEC, 2: self.SUBSTRATE}

    @staticmethod
    def _union_operands(monkeypatch):
        seen = []
        original = backend.boolean_union

        def counting(shapes):
            seen.append(len(shapes))
            return original(shapes)

        monkeypatch.setattr(backend, "boolean_union", counting)
        return seen

    def test_metal_of_a_pec_housing_is_held_by_the_brick(self, monkeypatch):
        """Substrate touching from below, air body cut by the metal: nothing to fuse."""
        housing = geo.Brick(origin=(0, 0, 0), size=(10 * MM, 10 * MM, 10 * MM))
        substrate = geo.Brick(origin=(0, 0, 0), size=(10 * MM, 10 * MM, 2 * MM))
        air = geo.Brick(origin=(0, 0, 2 * MM), size=(10 * MM, 10 * MM, 8 * MM))
        metal = geo.Brick(origin=(4 * MM, 4 * MM, 2 * MM), size=(2 * MM, 2 * MM, MM))
        shapes = [(housing, 1), (substrate, 2), (geo.Difference(air, metal), 0), (metal, 1)]
        seen = self._union_operands(monkeypatch)
        solid = backend.build_effective_pec_solid(shapes, self._library())
        assert seen == [1]  # the housing's own cut holds the metal already
        assert _volume(solid) == pytest.approx(4 * MM**3, rel=1e-12)
        reference = _pairwise_reference(shapes, self._library())
        assert _volume(solid) == pytest.approx(_volume(reference), rel=1e-12)

    def test_metal_over_a_lower_air_pocket_is_fused(self, monkeypatch):
        """The pocket is cut from the housing but not from the metal above it."""
        housing = geo.Brick(origin=(0, 0, 0), size=(10 * MM, 10 * MM, 10 * MM))
        pocket = geo.Brick(origin=(4 * MM, 4 * MM, 4 * MM), size=(2 * MM, 2 * MM, 2 * MM))
        metal = geo.Brick(origin=(4 * MM, 4 * MM, 5 * MM), size=(2 * MM, 2 * MM, 2 * MM))
        seen = self._union_operands(monkeypatch)
        solid = backend.build_effective_pec_solid(
            [(housing, 1), (pocket, 0), (metal, 1)], self._library()
        )
        assert seen == [2]
        assert _volume(solid) == pytest.approx((1000 - 8 + 4) * MM**3, rel=1e-12)

    def test_a_touching_shape_between_does_not_block(self, monkeypatch):
        housing = geo.Brick(origin=(0, 0, 0), size=(10 * MM, 10 * MM, 10 * MM))
        substrate = geo.Brick(origin=(0, 0, 0), size=(10 * MM, 10 * MM, 2 * MM))
        metal = geo.Brick(origin=(4 * MM, 4 * MM, 2 * MM), size=(2 * MM, 2 * MM, MM))
        seen = self._union_operands(monkeypatch)
        solid = backend.build_effective_pec_solid(
            [(housing, 1), (substrate, 2), (metal, 1)], self._library()
        )
        assert seen == [1]
        assert _volume(solid) == pytest.approx(800 * MM**3, rel=1e-12)

    def test_metal_under_a_higher_pec_box_is_neither_cut_nor_fused(self, monkeypatch):
        metal = geo.Brick(origin=(4 * MM, 4 * MM, 4 * MM), size=(2 * MM, 2 * MM, 2 * MM))
        pocket = geo.Brick(origin=(3 * MM, 3 * MM, 3 * MM), size=(4 * MM, 4 * MM, 4 * MM))
        lid = geo.Brick(origin=(0, 0, 0), size=(10 * MM, 10 * MM, 10 * MM))
        cuts = []
        original = backend.boolean_difference_many
        monkeypatch.setattr(
            backend, "boolean_difference_many", lambda b, t: cuts.append(len(t)) or original(b, t)
        )
        seen = self._union_operands(monkeypatch)
        solid = backend.build_effective_pec_solid(
            [(metal, 1), (pocket, 0), (lid, 1)], self._library()
        )
        assert seen == [1] and cuts == []
        assert _volume(solid) == pytest.approx(1000 * MM**3, rel=1e-12)

    def test_a_holder_that_is_not_a_box_does_not_hold(self, monkeypatch):
        block = geo.Brick(origin=(0, 0, 0), size=(10 * MM, 10 * MM, 10 * MM))
        corner = geo.Brick(origin=(0, 0, 0), size=(5 * MM, 5 * MM, 10 * MM))
        metal = geo.Brick(origin=(MM, MM, MM), size=(2 * MM, 2 * MM, 2 * MM))
        seen = self._union_operands(monkeypatch)
        solid = backend.build_effective_pec_solid(
            [(geo.Difference(block, corner), 1), (metal, 1)], self._library()
        )
        assert seen == [2]
        assert _volume(solid) == pytest.approx((1000 - 250 + 8) * MM**3, rel=1e-12)

    def test_an_overlapping_shape_between_blocks(self, monkeypatch):
        housing = geo.Brick(origin=(0, 0, 0), size=(10 * MM, 10 * MM, 10 * MM))
        air = geo.Brick(origin=(0, 0, 2 * MM), size=(8 * MM, 8 * MM, 8 * MM))
        metal = geo.Brick(origin=(4 * MM, 4 * MM, 2 * MM), size=(2 * MM, 2 * MM, MM))
        seen = self._union_operands(monkeypatch)
        solid = backend.build_effective_pec_solid(
            [(housing, 1), (air, 0), (metal, 1)], self._library()
        )
        assert seen == [2]
        assert _volume(solid) == pytest.approx((1000 - 512 + 4) * MM**3, rel=1e-12)

    def test_the_housing_is_cut_by_the_air_box_not_the_air_body(self, monkeypatch):
        """Brick − (air − metal) = (brick − air) ∪ metal: the air body's faces stay out of it."""
        housing = geo.Brick(origin=(0, 0, 0), size=(10 * MM, 10 * MM, 10 * MM))
        substrate = geo.Brick(origin=(0, 0, 0), size=(10 * MM, 10 * MM, 2 * MM))
        air = geo.Brick(origin=(MM, MM, 2 * MM), size=(8 * MM, 8 * MM, 7 * MM))
        metal = geo.Brick(origin=(4 * MM, 4 * MM, 2 * MM), size=(2 * MM, 2 * MM, MM))
        air_body = geo.Difference(air, metal)
        shapes = [(housing, 1), (substrate, 2), (air_body, 0), (metal, 1)]
        tools_seen = []
        original = backend.boolean_difference_many
        monkeypatch.setattr(
            backend,
            "boolean_difference_many",
            lambda b, t: tools_seen.append([id(x) for x in t]) or original(b, t),
        )
        seen = self._union_operands(monkeypatch)
        solid = backend.build_effective_pec_solid(shapes, self._library())
        assert tools_seen == [[id(substrate._occ_shape()), id(air._occ_shape())]]
        assert seen == [2]  # the housing shell and the metal
        expected = (1000 - 200 - 8 * 8 * 7 + 4) * MM**3
        assert _volume(solid) == pytest.approx(expected, rel=1e-12)
        reference = _pairwise_reference(shapes, self._library())
        assert _volume(solid) == pytest.approx(_volume(reference), rel=1e-12)

    def test_substituted_tools_are_cut_by_a_pocket_that_overlaps_them(self, monkeypatch):
        housing = geo.Brick(origin=(0, 0, 0), size=(10 * MM, 10 * MM, 10 * MM))
        air = geo.Brick(origin=(MM, MM, MM), size=(8 * MM, 8 * MM, 8 * MM))
        metal = geo.Brick(origin=(4 * MM, 4 * MM, 4 * MM), size=(2 * MM, 2 * MM, 2 * MM))
        pocket = geo.Brick(origin=(5 * MM, 5 * MM, 5 * MM), size=(2 * MM, 2 * MM, 2 * MM))
        shapes = [(housing, 1), (geo.Difference(air, metal), 0), (metal, 1), (pocket, 0)]
        solid = backend.build_effective_pec_solid(shapes, self._library())
        assert _volume(solid) == pytest.approx((1000 - 512 + 8 - 1) * MM**3, rel=1e-12)
        reference = _pairwise_reference(shapes, self._library())
        assert _volume(solid) == pytest.approx(_volume(reference), rel=1e-12)

    def test_tools_beyond_the_box_are_not_substituted(self, monkeypatch):
        housing = geo.Brick(origin=(0, 0, 0), size=(10 * MM, 10 * MM, 10 * MM))
        air = geo.Brick(origin=(0, 0, 0), size=(10 * MM, 10 * MM, 10 * MM))
        metal = geo.Brick(origin=(4 * MM, 4 * MM, 8 * MM), size=(2 * MM, 2 * MM, 4 * MM))
        tools_seen = []
        original = backend.boolean_difference_many
        monkeypatch.setattr(
            backend,
            "boolean_difference_many",
            lambda b, t: tools_seen.append([id(x) for x in t]) or original(b, t),
        )
        air_body = geo.Difference(air, metal)
        solid = backend.build_effective_pec_solid(
            [(housing, 1), (air_body, 0), (metal, 1)], self._library()
        )
        assert tools_seen == [[id(air_body._occ_shape())]]
        assert _volume(solid) == pytest.approx(16 * MM**3, rel=1e-12)
