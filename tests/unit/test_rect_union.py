"""The in-house union of rectilinear coplanar faces.

The union of straight, axis-aligned caps is computed on the compressed
grid of their vertex coordinates; its point set must equal the kernel
fuse, corner contacts must come out as the kernel resolves them (two
faces; two hole wires sharing a vertex), and anything that is not
rectilinear must be declined so that the kernel's tree takes it.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio import geo
from magnelio.geo import _occ_backend as backend
from magnelio.geo import _rect_union
from magnelio.geo._prism_fuse import _LEVEL_TOLERANCE, fuse_faces_tree, prism_candidates
from magnelio.geo._rect_union import (
    faces_from_pieces,
    fuse_rectilinear_faces,
    rectilinear_rings,
    rectilinear_union,
)

MM = 1e-3
TOL = _LEVEL_TOLERANCE


def _area(shape) -> float:
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps

    props = GProp_GProps()
    brepgprop.SurfaceProperties(shape, props)
    return props.Mass()


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


def _wires(shape) -> int:
    from OCC.Core.TopAbs import TopAbs_WIRE

    return _count(shape, TopAbs_WIRE)


def _valid(shape) -> bool:
    from OCC.Core.BRepCheck import BRepCheck_Analyzer

    return BRepCheck_Analyzer(shape).IsValid()


def _caps(*shapes, axis: int = 2):
    """Bottom caps of prisms along *axis*, oriented forward."""
    from OCC.Core.TopAbs import TopAbs_FORWARD

    faces = []
    for shape in shapes:
        caps = prism_candidates(shape._occ_shape(1.0))[axis][2]
        faces.extend(f.Oriented(TopAbs_FORWARD) for f in caps)
    return faces


def _kernel_fuse_faces(faces):
    occ = backend._require_occ()
    return backend.unify_same_domain(backend._run_bop(occ["Fuse"], faces[:1], faces[1:]))


def _rect(u0, v0, u1, v1) -> np.ndarray:
    return np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]], dtype=float)


class TestRectilinearUnion:
    def test_overlapping_rectangles_give_one_ring(self):
        pieces = rectilinear_union([_rect(0, 0, 5, 1), _rect(4, 0, 5, 4)], TOL)
        assert len(pieces) == 1
        outer, holes = pieces[0]
        assert holes == []
        assert len(outer) == 6  # an L
        assert _rect_union._signed_area(outer) == pytest.approx(5 + 4 - 1)
        assert {tuple(p) for p in outer} <= {(0, 0), (5, 0), (5, 4), (4, 4), (4, 1), (0, 1)}

    def test_vertices_are_input_coordinates(self):
        x = 0.1 + 0.2  # not exactly 0.3
        pieces = rectilinear_union([_rect(0, 0, x, 1), _rect(x - 0.05, 0.5, 1, 0.7)], TOL)
        coordinates = {c for outer, _ in pieces for c in outer.ravel()}
        assert coordinates <= {0.0, x, 1.0, x - 0.05, 0.5, 0.7}

    def test_frame_has_a_hole(self):
        strips = [_rect(0, 0, 3, 1), _rect(0, 2, 3, 3), _rect(0, 0, 1, 3), _rect(2, 0, 3, 3)]
        pieces = rectilinear_union(strips, TOL)
        assert len(pieces) == 1
        outer, holes = pieces[0]
        assert len(outer) == 4
        assert len(holes) == 1
        assert _rect_union._signed_area(holes[0]) == pytest.approx(-1.0)

    def test_corner_contact_gives_two_pieces(self):
        pieces = rectilinear_union([_rect(0, 0, 1, 1), _rect(1, 1, 2, 2)], TOL)
        assert len(pieces) == 2
        assert all(len(outer) == 4 and not holes for outer, holes in pieces)

    def test_two_holes_meeting_at_a_corner_stay_two_rings(self):
        plate = _rect(0, 0, 4, 4)
        hole_a = _rect(1, 1, 2, 2)[::-1]
        hole_b = _rect(2, 2, 3, 3)[::-1]
        pieces = rectilinear_union([plate, hole_a, hole_b], TOL)
        assert len(pieces) == 1
        outer, holes = pieces[0]
        assert len(outer) == 4
        assert sorted(len(h) for h in holes) == [4, 4]
        assert sum(_rect_union._signed_area(h) for h in holes) == pytest.approx(-2.0)

    def test_hole_touching_the_outline_at_a_corner(self):
        plate = _rect(0, 0, 3, 3)
        hole = _rect(2, 2, 3, 3)[::-1]  # shares the corner (3, 3) with the outline
        pieces = rectilinear_union([plate, hole], TOL)
        # A square with a corner notch: one ring of six vertices, no hole.
        assert len(pieces) == 1
        outer, holes = pieces[0]
        assert holes == []
        assert len(outer) == 6

    def test_hole_of_one_face_covered_by_another(self):
        frame = [_rect(0, 0, 3, 3), _rect(1, 1, 2, 2)[::-1]]
        pieces = rectilinear_union([*frame, _rect(0.5, 0.5, 2.5, 2.5)], TOL)
        assert len(pieces) == 1
        outer, holes = pieces[0]
        assert holes == []
        assert _rect_union._signed_area(outer) == pytest.approx(9.0)

    def test_disjoint_rectangles_are_separate_pieces(self):
        pieces = rectilinear_union([_rect(0, 0, 1, 1), _rect(5, 5, 6, 7)], TOL)
        assert sorted(_rect_union._signed_area(o) for o, _ in pieces) == pytest.approx([1.0, 2.0])

    def test_coordinates_within_tolerance_merge(self):
        pieces = rectilinear_union([_rect(0, 0, 1, 1), _rect(1 + 0.5 * TOL, 0, 2, 1)], TOL)
        assert len(pieces) == 1
        assert len(pieces[0][0]) == 4

    def test_matches_the_kernel_on_a_random_layout(self):
        rng = np.random.default_rng(7)
        rects = []
        for _ in range(60):
            u0, v0 = rng.uniform(0, 10, 2)
            du, dv = rng.uniform(0.3, 3.0, 2)
            rects.append(_rect(u0, v0, u0 + du, v0 + dv))
        pieces = rectilinear_union(rects, TOL)
        area = sum(
            _rect_union._signed_area(o) + sum(_rect_union._signed_area(h) for h in holes)
            for o, holes in pieces
        )
        faces = faces_from_pieces(pieces, 2, 0.0)
        assert all(_valid(f) for f in faces)
        bricks = [
            geo.Brick(
                origin=(r[0, 0], r[0, 1], 0.0), size=(r[2, 0] - r[0, 0], r[2, 1] - r[0, 1], 1.0)
            )
            for r in rects
        ]
        kernel = _kernel_fuse_faces(_caps(*bricks))
        assert area == pytest.approx(_area(kernel), rel=1e-12)
        assert len(faces) == _faces(kernel)


class TestRectilinearRings:
    def test_bricks_are_rectilinear(self):
        a = geo.Brick(origin=(0, 0, 0), size=(MM, MM, 0.1 * MM))
        b = geo.Brick(origin=(MM / 2, 0, 0), size=(MM, 2 * MM, 0.1 * MM))
        axis, level, rings = rectilinear_rings(_caps(a, b), TOL)
        assert axis == 2
        assert level == 0.0
        assert len(rings) == 2
        assert all(_rect_union._signed_area(r) > 0 for r in rings)

    def test_rotated_brick_is_declined(self):
        a = geo.Brick(origin=(0, 0, 0), size=(MM, MM, 0.1 * MM))
        b = geo.Brick(origin=(0, 0, 0), size=(MM, MM, 0.1 * MM)).rotated(axis="z", angle_deg=30.0)
        assert rectilinear_rings(_caps(a, b), TOL) is None

    def test_arc_is_declined(self):
        a = geo.Brick(origin=(0, 0, 0), size=(MM, MM, 0.1 * MM))
        pad = geo.Cylinder(origin=(MM, 0, 0), radius=MM / 2, height=0.1 * MM)
        assert rectilinear_rings(_caps(a, pad), TOL) is None

    def test_declined_cluster_takes_the_tree(self):
        a = geo.Brick(origin=(0, 0, 0), size=(MM, MM, 0.1 * MM))
        pad = geo.Cylinder(origin=(MM, 0, 0), radius=MM / 2, height=0.1 * MM)
        faces = _caps(a, pad)
        assert fuse_rectilinear_faces(faces, TOL) is None
        tree = fuse_faces_tree(faces, _kernel_fuse_faces)
        assert _area(tree) == pytest.approx(_area(_kernel_fuse_faces(faces)))


class TestFuseRectilinearFaces:
    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_matches_the_kernel_on_every_axis(self, axis):
        size_a = [3 * MM, 3 * MM, 3 * MM]
        size_b = [2 * MM, 2 * MM, 2 * MM]
        size_a[axis] = size_b[axis] = 0.1 * MM
        origin_b = [MM, 2 * MM, 2 * MM]
        origin_b[axis] = 0.0
        a = geo.Brick(origin=(0, 0, 0), size=tuple(size_a))
        b = geo.Brick(origin=tuple(origin_b), size=tuple(size_b))
        faces = _caps(a, b, axis=axis)
        fused = fuse_rectilinear_faces(faces, TOL)
        assert fused is not None
        assert _valid(fused)
        assert _faces(fused) == 1
        assert _area(fused) == pytest.approx(_area(_kernel_fuse_faces(faces)), rel=1e-12)

    def test_frame_face_carries_its_hole(self):
        strips = [
            geo.Brick(origin=(0, 0, 0), size=(3 * MM, MM, 0.1 * MM)),
            geo.Brick(origin=(0, 2 * MM, 0), size=(3 * MM, MM, 0.1 * MM)),
            geo.Brick(origin=(0, 0, 0), size=(MM, 3 * MM, 0.1 * MM)),
            geo.Brick(origin=(2 * MM, 0, 0), size=(MM, 3 * MM, 0.1 * MM)),
        ]
        fused = fuse_rectilinear_faces(_caps(*strips), TOL)
        assert _valid(fused)
        assert _faces(fused) == 1
        assert _wires(fused) == 2
        assert _area(fused) == pytest.approx(8 * MM**2)

    def test_corner_contact_gives_two_faces_like_the_kernel(self):
        a = geo.Brick(origin=(0, 0, 0), size=(MM, MM, 0.1 * MM))
        b = geo.Brick(origin=(MM, MM, 0), size=(MM, MM, 0.1 * MM))
        faces = _caps(a, b)
        fused = fuse_rectilinear_faces(faces, TOL)
        assert _valid(fused)
        assert _faces(fused) == 2 == _faces(_kernel_fuse_faces(faces))

    def test_union_of_bricks_takes_the_route(self, monkeypatch):
        calls = []
        real = _rect_union.fuse_rectilinear_faces

        def spy(faces, tolerance):
            calls.append(len(faces))
            return real(faces, tolerance)

        monkeypatch.setattr(_rect_union, "fuse_rectilinear_faces", spy)
        a = geo.Brick(origin=(0, 0, 0), size=(5 * MM, MM, 0.1 * MM), material="pec")
        b = geo.Brick(origin=(4 * MM, 0, 0), size=(MM, 4 * MM, 0.1 * MM), material="pec")
        c = geo.Brick(origin=(0, 3 * MM, 0), size=(5 * MM, MM, 0.1 * MM), material="pec")
        fused = geo.Union(a, b, c)._occ_shape(1.0)
        assert calls == [3]
        assert _valid(fused)
        assert _volume(fused) == pytest.approx((5 + 4 - 1 + 5 - 1) * MM**2 * 0.1 * MM, rel=1e-12)
        assert _faces(fused) == 10  # two caps and the eight walls of a C
