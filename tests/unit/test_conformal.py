"""Tests for mesh.conformal — boundary detection and PEC surface extraction.

DD-051: the per-edge / per-face material data structures and the
unified classifier moved to :mod:`magnelio.geo._subcell` and are
covered by ``tests/unit/test_subcell_pipeline.py``.  This file retains
the boundary-cell detector and PEC-surface extractor coverage that is
unchanged by DD-051.
"""

import numpy as np

from magnelio.mesh._conformal import (
    detect_boundary_cells,
)

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
