"""The planar section engine's stitched contours equal the kernel's.

The engine pairs the plane's edge crossings along every candidate face
in one vectorised pass; the kernel Boolean is the reference, contour by
contour.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio.geo import Brick, Difference, Union
from magnelio.geo import _occ_backend as ob
from magnelio.geo._polygon_clip import polygon_area
from magnelio.materials import Material

pytest.importorskip("OCC")

AIR = Material.from_isotropic(name="air", epsilon=1.0)
PEC = Material.pec()
DEFLECTION = 1e-6
NUDGE = 1e-5


def _pocketed_comb():
    """A slab with teeth (a union) and pockets (a difference): planes cross
    many faces with many crossings each."""
    body = Brick(origin=(0, 0, 0), size=(6e-3, 3e-3, 1e-3), material=AIR)
    teeth = [
        Brick(
            origin=(0.2e-3 + 0.7e-3 * i, 2.9e-3, 0.1e-3),
            size=(0.3e-3, 0.9e-3, 0.7e-3),
            material=AIR,
        )
        for i in range(8)
    ]
    pockets = [
        Brick(
            origin=(0.3e-3 + 0.5e-3 * i, 0.5e-3 + 0.3e-3 * (i % 3), 0.6e-3),
            size=(0.25e-3, 1.2e-3, 0.4e-3),
            material=PEC,
        )
        for i in range(11)
    ]
    return Difference(Union(body, *teeth), *pockets)


def _signature(polys):
    """Order-free description: per contour the vertex set and |area|."""
    out = []
    for p in polys:
        verts = sorted((round(float(u), 12), round(float(v), 12)) for u, v in p)
        out.append((round(abs(polygon_area(p)), 18), tuple(verts)))
    return sorted(out)


class TestPlanarSectionEngine:
    def test_stitched_contours_equal_the_kernel_sections(self):
        occ = _pocketed_comb()._occ_shape(1.0)
        engine = ob._PlanarSectionEngine(occ, scale=1.0, deflection=DEFLECTION)
        assert engine.enabled and not engine.facetted
        rng = np.random.default_rng(2)
        answered = 0
        for axis, letter, lo, hi in (
            (0, "x", 0.05e-3, 5.95e-3),
            (1, "y", 0.05e-3, 3.75e-3),
            (2, "z", 0.05e-3, 0.95e-3),
        ):
            for pos in rng.uniform(lo, hi, 40):
                fast = engine.section(axis, float(pos))
                if fast is None:
                    continue
                answered += 1
                kernel = ob.cross_section_polygons(
                    occ, letter, float(pos), deflection=DEFLECTION, scale=1.0, nudge=NUDGE
                )
                assert _signature(fast) == _signature(kernel), (letter, pos)
        assert answered >= 100

    def test_planes_through_vertices_or_faces_are_declined(self):
        occ = _pocketed_comb()._occ_shape(1.0)
        engine = ob._PlanarSectionEngine(occ, scale=1.0, deflection=DEFLECTION)
        assert engine.section(2, 0.6e-3) is None  # in the pockets' floor
        assert engine.section(0, 0.3e-3) is None  # through pocket walls
        assert engine.section(0, 0.7e-3) is not None

    def test_face_tangents_are_computed_once_per_axis(self):
        occ = _pocketed_comb()._occ_shape(1.0)
        engine = ob._PlanarSectionEngine(occ, scale=1.0, deflection=DEFLECTION)
        tangent, ok = engine._face_tangents(0)
        assert tangent.shape == (engine.face_count, 3)
        # Only the faces normal to x have no trace direction in an
        # x-plane (they are coplanar with it or never meet it).
        np.testing.assert_array_equal(ok, np.abs(engine._f_n[:, 0]) < 0.5)
        assert engine._face_tangents(0)[0] is tangent
        np.testing.assert_allclose(tangent[:, 0], 0.0)


# ── Cylindrical faces and circular edges ───────────────────────────────────

PITCH, RADIUS, HEIGHT = 2e-3, 0.5e-3, 4e-3


def _post_row():
    """Three PEC posts standing on the floor of an air box: the air body
    carries the holes (rim circles on the floor, top caps inside), the
    posts their seams and caps."""
    from magnelio.geo import Cylinder

    box = Brick(origin=(0.0, -3e-3, 0.0), size=(4 * PITCH, 6e-3, 6e-3), material=AIR)
    post = Cylinder(origin=(PITCH, 0.0, 0.0), radius=RADIUS, height=HEIGHT, axis="z", material=PEC)
    posts = post.translated((PITCH, 0.0, 0.0), repeat=2, copy=True, unite=True)
    return Difference(box, posts), posts


def _assert_same_contours(fast, kernel, atol):
    """Same contours within the kernel's tolerance: one-to-one by
    centroid, equal |area| to rounding, every vertex of either within
    *atol* of the other's polyline.  (The engine's vertices are exact;
    the kernel's may sit a vertex tolerance off — a plane 14 µm from a
    post's seam gets the cap chord extended to the diameter ends.)"""
    assert len(fast) == len(kernel)

    def key(p):
        c = p.mean(axis=0)
        return (round(float(c[0]), 7), round(float(c[1]), 7))

    for pa, pb in zip(sorted(fast, key=key), sorted(kernel, key=key), strict=True):
        area_a, area_b = abs(polygon_area(pa)), abs(polygon_area(pb))
        assert abs(area_a - area_b) <= 1e-12 * area_b + atol * atol
        assert _vertex_deviation([pa], [pb]) <= atol
        assert _vertex_deviation([pb], [pa]) <= atol


def _vertex_deviation(polys_a, polys_b):
    """Largest distance of a vertex of *polys_a* from the polylines of *polys_b*."""
    worst = 0.0
    for pa in polys_a:
        for q in pa:
            best = np.inf
            for pb in polys_b:
                p0, p1 = pb, np.roll(pb, -1, axis=0)
                d = p1 - p0
                tt = np.clip(
                    np.einsum("ij,ij->i", q - p0, d) / np.einsum("ij,ij->i", d, d), 0.0, 1.0
                )
                best = min(best, float(np.linalg.norm(q - (p0 + tt[:, None] * d), axis=1).min()))
            worst = max(worst, best)
    return worst


class TestCylinderSectionEngine:
    def test_post_row_matches_the_kernel_to_rounding(self):
        rng = np.random.default_rng(7)
        for body in _post_row():
            occ = body._occ_shape(1.0)
            engine = ob._PlanarSectionEngine(occ, scale=1.0, deflection=DEFLECTION)
            assert engine.enabled and engine._has_cylinders and engine._has_circles
            answered = 0
            for axis, letter, lo, hi in (
                (0, "x", 0.05e-3, 4 * PITCH - 0.05e-3),
                (1, "y", -0.49e-3, 0.49e-3),
                (2, "z", 0.05e-3, HEIGHT - 0.05e-3),
            ):
                for pos in rng.uniform(lo, hi, 25):
                    fast = engine.section(axis, float(pos))
                    if fast is None:
                        continue
                    answered += 1
                    kernel = ob.cross_section_polygons(
                        occ, letter, float(pos), deflection=DEFLECTION, scale=1.0, nudge=NUDGE
                    )
                    _assert_same_contours(fast, kernel, atol=3e-7)
            assert answered >= 60

    def test_oblique_cylinder_is_sectioned_along_its_conic(self):
        from magnelio.geo import Cylinder

        body = Cylinder(
            origin=(0.0, 0.0, 0.0),
            radius=1e-3,
            height=5e-3,
            axis=(1.0, 0.5, 2.0),
            material=PEC,
        )
        occ = body._occ_shape(1.0)
        engine = ob._PlanarSectionEngine(occ, scale=1.0, deflection=DEFLECTION)
        assert engine.enabled and engine._has_cylinders
        axis_dir = np.array((1.0, 0.5, 2.0)) / np.linalg.norm((1.0, 0.5, 2.0))
        rng = np.random.default_rng(3)
        answered = 0
        for axis, letter in ((0, "x"), (1, "y"), (2, "z")):
            for pos in rng.uniform(0.2e-3, 1.6e-3, 12):
                fast = engine.section(axis, float(pos))
                if fast is None:
                    continue
                answered += 1
                kernel = ob.cross_section_polygons(
                    occ, letter, float(pos), deflection=DEFLECTION, scale=1.0, nudge=NUDGE
                )
                # The ellipse arcs are tessellated by the same rule but
                # not at the same parameters: equal area within the
                # chord budget, every vertex within it of the other
                # polyline — and the engine's vertices lie *on* the
                # cylinder.
                area_fast = sum(abs(polygon_area(p)) for p in fast)
                area_kernel = sum(abs(polygon_area(p)) for p in kernel)
                perimeter = sum(
                    float(np.linalg.norm(np.diff(np.vstack((p, p[:1])), axis=0), axis=1).sum())
                    for p in kernel
                )
                assert abs(area_fast - area_kernel) <= DEFLECTION * perimeter
                assert _vertex_deviation(fast, kernel) <= DEFLECTION
                assert _vertex_deviation(kernel, fast) <= DEFLECTION
                u_idx, v_idx = engine._UV[axis]
                for p in fast:
                    xyz = np.zeros((len(p), 3))
                    xyz[:, axis] = pos
                    xyz[:, u_idx] = p[:, 0]
                    xyz[:, v_idx] = p[:, 1]
                    radial = xyz - np.outer(xyz @ axis_dir, axis_dir)
                    on_surface = np.abs(np.linalg.norm(radial, axis=1) - 1e-3) <= 1e-12
                    # Cap-rim crossings and seam crossings are on the
                    # surface too; only the caps' straight-edge-free
                    # interiors are off it, and those carry no vertex.
                    assert on_surface.all()
        assert answered >= 20

    def test_partial_cylinder_face_with_a_seam(self):
        """A half cylinder (its seam inside the face, generatrices as
        boundaries) is a rectangle of the (u, v) domain and answered."""
        from magnelio.geo import Cylinder

        body = Difference(
            Cylinder(origin=(0.0, 0.0, 0.0), radius=1e-3, height=2e-3, axis="z", material=PEC),
            Brick(origin=(-2e-3, -2e-3, -1e-3), size=(2e-3, 4e-3, 4e-3), material=PEC),
        )
        occ = body._occ_shape(1.0)
        engine = ob._PlanarSectionEngine(occ, scale=1.0, deflection=DEFLECTION)
        assert engine.enabled and engine._has_cylinders
        answered = 0
        for axis, letter, positions in (
            (2, "z", (0.3e-3, 1.1e-3, 1.7e-3)),
            (1, "y", (-0.7e-3, 0.2e-3, 0.9e-3)),
            (0, "x", (0.2e-3, 0.6e-3, 0.95e-3)),
        ):
            for pos in positions:
                fast = engine.section(axis, pos)
                if fast is None:
                    continue
                answered += 1
                kernel = ob.cross_section_polygons(
                    occ, letter, pos, deflection=DEFLECTION, scale=1.0, nudge=NUDGE
                )
                _assert_same_contours(fast, kernel, atol=3e-7)
        assert answered >= 7

    def test_non_parameter_boundaries_keep_the_kernel(self):
        """A cylindrical face bounded by a curve that is not one of its
        parameter lines (a spherical pocket) is delegated."""
        from magnelio.geo import Cylinder, Sphere

        body = Difference(
            Cylinder(origin=(0.0, 0.0, 0.0), radius=1e-3, height=2e-3, axis="z", material=PEC),
            Sphere(center=(1e-3, 0.0, 1e-3), radius=0.5e-3, material=PEC),
        )
        occ = body._occ_shape(1.0)
        engine = ob._PlanarSectionEngine(occ, scale=1.0, deflection=DEFLECTION)
        assert engine.enabled
        assert not engine._f_cyl_ok.all() and engine._f_cyl_ok.any() is not None
        assert engine.section(2, 1.0e-3) is None
        assert engine.section(1, 0.1e-3) is None

    def test_degenerate_planes_are_declined(self):
        air, posts = _post_row()
        for body in (air, posts):
            engine = ob._PlanarSectionEngine(body._occ_shape(1.0), scale=1.0, deflection=DEFLECTION)
            assert engine.section(1, 0.0) is None  # the seams lie in y = 0
            assert engine.section(1, RADIUS) is None  # tangent to every post
            assert engine.section(2, HEIGHT) is None  # in the top caps
            assert engine.section(2, 0.5 * HEIGHT) is not None
            assert engine.section(0, PITCH) is not None

    def test_toggle_keeps_cylinders_on_the_kernel(self, monkeypatch):
        monkeypatch.setenv("MAGNELIO_CYLINDER_SECTIONS", "0")
        _, posts = _post_row()
        engine = ob._PlanarSectionEngine(posts._occ_shape(1.0), scale=1.0, deflection=DEFLECTION)
        assert engine.enabled and not engine._has_cylinders
        assert engine.section(2, 0.5 * HEIGHT) is None
        assert engine.section(0, PITCH) is None
