"""Tests for mesh.conformal — boundary detection and PEC surface extraction.

DD-051: the per-edge / per-face material data structures and the
unified classifier moved to :mod:`magnelio.geo._subcell` and are
covered by ``tests/unit/test_subcell_pipeline.py``.  This file retains
the boundary-cell detector and PEC-surface extractor coverage that is
unchanged by DD-051.
"""

import numpy as np

from magnelio.materials.material import Material
from magnelio.mesh._conformal import (
    PECSurfaceData,
    detect_boundary_cells,
    extract_pec_surface,
)
from magnelio.mesh.grid import GridLines

# ---------------------------------------------------------------------------
# detect_boundary_cells
# ---------------------------------------------------------------------------


class TestDetectBoundaryCells:
    def test_uniform_material_no_boundaries(self):
        mat_id = np.zeros((5, 5, 5), dtype=np.int32)
        boundary = detect_boundary_cells(mat_id)
        assert not boundary.any()

    def test_two_materials_x_interface(self):
        """Interface at x-index 2: cells 0-1 are mat 0, cells 2-4 are mat 1."""
        mat_id = np.zeros((5, 4, 3), dtype=np.int32)
        mat_id[2:, :, :] = 1
        boundary = detect_boundary_cells(mat_id)
        assert boundary[1, :, :].all()
        assert boundary[2, :, :].all()
        assert not boundary[0, :, :].any()
        assert not boundary[3, :, :].any()
        assert not boundary[4, :, :].any()

    def test_two_materials_y_interface(self):
        mat_id = np.zeros((3, 6, 3), dtype=np.int32)
        mat_id[:, 3:, :] = 1
        boundary = detect_boundary_cells(mat_id)
        assert boundary[:, 2, :].all()
        assert boundary[:, 3, :].all()
        assert not boundary[:, 0, :].any()
        assert not boundary[:, 5, :].any()

    def test_two_materials_z_interface(self):
        mat_id = np.zeros((3, 3, 8), dtype=np.int32)
        mat_id[:, :, 4:] = 1
        boundary = detect_boundary_cells(mat_id)
        assert boundary[:, :, 3].all()
        assert boundary[:, :, 4].all()
        assert not boundary[:, :, 0].any()
        assert not boundary[:, :, 7].any()

    def test_single_embedded_cell(self):
        mat_id = np.zeros((5, 5, 5), dtype=np.int32)
        mat_id[2, 2, 2] = 1
        boundary = detect_boundary_cells(mat_id)
        assert boundary[2, 2, 2]
        assert boundary[1, 2, 2]
        assert boundary[3, 2, 2]
        assert boundary[2, 1, 2]
        assert boundary[2, 3, 2]
        assert boundary[2, 2, 1]
        assert boundary[2, 2, 3]
        assert not boundary[0, 0, 0]
        assert not boundary[4, 4, 4]

    def test_checkerboard_all_boundaries(self):
        mat_id = np.zeros((4, 4, 4), dtype=np.int32)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    mat_id[i, j, k] = (i + j + k) % 2
        boundary = detect_boundary_cells(mat_id)
        assert boundary.all()


# ---------------------------------------------------------------------------
# extract_pec_surface
# ---------------------------------------------------------------------------


class TestExtractPecSurface:
    @staticmethod
    def _make_grid(nx, ny, nz, step=1e-3):
        return GridLines(
            x=np.linspace(0, nx * step, nx + 1),
            y=np.linspace(0, ny * step, ny + 1),
            z=np.linspace(0, nz * step, nz + 1),
        )

    def test_no_pec_empty_surface(self):
        grid = self._make_grid(4, 4, 4)
        mat_id = np.zeros((4, 4, 4), dtype=np.int32)
        mat_lib = {0: Material.air()}
        surf = extract_pec_surface(grid, mat_id, mat_lib)
        assert len(surf.face_indices) == 0

    def test_all_pec_no_boundary(self):
        grid = self._make_grid(3, 3, 3)
        pec = Material(name="pec", is_pec=True)
        mat_id = np.ones((3, 3, 3), dtype=np.int32)
        mat_lib = {0: Material.air(), 1: pec}
        surf = extract_pec_surface(grid, mat_id, mat_lib)
        assert len(surf.face_indices) == 0

    def test_pec_slab_x(self):
        grid = self._make_grid(4, 2, 2)
        pec = Material(name="pec", is_pec=True)
        mat_id = np.zeros((4, 2, 2), dtype=np.int32)
        mat_id[:2, :, :] = 1
        mat_lib = {0: Material.air(), 1: pec}

        surf = extract_pec_surface(grid, mat_id, mat_lib)
        assert len(surf.face_indices) > 0
        assert np.all(surf.face_components == 0)
        assert np.all(surf.outward_normals[:, 0] == 1.0)
        assert np.all(surf.outward_normals[:, 1] == 0.0)
        assert np.all(surf.outward_normals[:, 2] == 0.0)
        assert len(surf.face_indices) == 4

    def test_surface_areas_correct(self):
        grid = GridLines(
            x=np.array([0.0, 1e-3, 3e-3, 6e-3]),
            y=np.array([0.0, 2e-3, 5e-3]),
            z=np.array([0.0, 4e-3]),
        )
        pec = Material(name="pec", is_pec=True)
        mat_id = np.zeros((3, 2, 1), dtype=np.int32)
        mat_id[0, :, :] = 1
        mat_lib = {0: Material.air(), 1: pec}

        surf = extract_pec_surface(grid, mat_id, mat_lib)
        expected_areas = {8e-6, 12e-6}
        actual_areas = set(np.round(surf.surface_areas, 10))
        assert actual_areas == expected_areas

    def test_pec_cube_in_center_six_faces(self):
        grid = self._make_grid(3, 3, 3)
        pec = Material(name="pec", is_pec=True)
        mat_id = np.zeros((3, 3, 3), dtype=np.int32)
        mat_id[1, 1, 1] = 1
        mat_lib = {0: Material.air(), 1: pec}

        surf = extract_pec_surface(grid, mat_id, mat_lib)
        assert len(surf.face_indices) == 6
        unique, counts = np.unique(surf.face_components, return_counts=True)
        assert set(unique) == {0, 1, 2}
        assert all(c == 2 for c in counts)


# ---------------------------------------------------------------------------
# PECSurfaceData smoke
# ---------------------------------------------------------------------------


class TestPECSurfaceData:
    def test_pec_surface_data_empty(self):
        ps = PECSurfaceData(
            face_indices=np.empty(0, dtype=int),
            face_components=np.empty(0, dtype=int),
            outward_normals=np.empty((0, 3)),
            surface_areas=np.empty(0),
        )
        assert len(ps.face_indices) == 0
