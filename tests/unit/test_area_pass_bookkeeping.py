"""Bookkeeping of the area passes: plane groups by one sort, one engine per shape.

The ε, σ and µ passes call ``compute_face_material_areas`` on the same
shapes; the grouping of faces by plane must equal the dict a loop
would build (plane order, face order, float keys), and the planar
section engine of a shape must be built once and reused.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio.geo import Brick
from magnelio.geo import _occ_backend as ob
from magnelio.materials import Material

pytest.importorskip("OCC")

MM = 1e-3


def _reference_groups(face_axes, plane_pos):
    groups: dict[tuple[int, float], list[int]] = {}
    for fi in range(len(face_axes)):
        groups.setdefault((int(face_axes[fi]), float(plane_pos[fi])), []).append(fi)
    return groups


class TestFacesByPlane:
    def test_matches_the_loop_on_a_random_face_set(self):
        rng = np.random.default_rng(3)
        n = 5_000
        axes = rng.integers(0, 3, n)
        pos = rng.choice(np.linspace(-1.0, 1.0, 37), n)
        groups = ob._faces_by_plane(axes, pos)
        reference = _reference_groups(axes, pos)
        assert list(groups) == list(reference)
        for key, faces in reference.items():
            assert groups[key].tolist() == faces

    def test_negative_zero_is_the_same_plane(self):
        axes = np.array([2, 2, 1, 2])
        pos = np.array([0.0, -0.0, 0.0, 0.5])
        groups = ob._faces_by_plane(axes, pos)
        assert list(groups) == [(2, 0.0), (1, 0.0), (2, 0.5)]
        assert groups[(2, 0.0)].tolist() == [0, 1]

    def test_empty(self):
        assert ob._faces_by_plane(np.zeros(0, dtype=int), np.zeros(0)) == {}


class TestSectionEngineCache:
    def test_engine_is_built_once_per_shape(self, monkeypatch):
        builds = []
        original = ob._PlanarSectionEngine.__init__

        def counting(self, shape, scale=1.0, deflection=None):
            builds.append(deflection)
            original(self, shape, scale=scale, deflection=deflection)

        monkeypatch.setattr(ob._PlanarSectionEngine, "__init__", counting)
        brick = Brick(origin=(0, 0, 0), size=(2 * MM, 2 * MM, 2 * MM))
        first = ob._section_engine(brick, 1.0, 1e-4)
        assert ob._section_engine(brick, 1.0, 1e-4) is first
        assert ob._section_engine(brick, 1.0, 5e-5) is not first
        assert builds == [1e-4, 5e-5]

    def test_repeated_passes_share_the_engine(self, monkeypatch):
        builds = []
        original = ob._PlanarSectionEngine.__init__

        def counting(self, shape, scale=1.0, deflection=None):
            builds.append(1)
            original(self, shape, scale=scale, deflection=deflection)

        monkeypatch.setattr(ob._PlanarSectionEngine, "__init__", counting)
        brick = Brick(origin=(0, 0, 0), size=(2 * MM, 2 * MM, 2 * MM), material="pec")
        library = {0: Material.vacuum(), 1: Material.pec()}
        specs = np.array(
            [[MM, 0.0, 0.0, MM, MM], [MM, MM, 0.0, 2 * MM, MM], [0.5 * MM, 0.0, 0.0, MM, MM]]
        )
        axes = np.array([2, 2, 0])
        cache: dict = {}
        first = ob.compute_face_material_areas(
            [(brick, 1)], library, specs, axes, prop="epsilon", section_cache=cache
        )
        second = ob.compute_face_material_areas(
            [(brick, 1)], library, specs, axes, prop="mu", section_cache=cache
        )
        assert len(builds) == 1
        assert np.all(first == 0.0)  # PEC excludes its area from the ε average
        assert np.all(second == 1.0)  # and counts as µ_r = 1 in the µ average
