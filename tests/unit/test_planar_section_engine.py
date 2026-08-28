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
