"""Tests for magnelio.geo._subcell — DD-051 unified sub-cell classifier.

Covers the four-category EdgeMaterialData classification and its mu
sibling FaceMaterialData.  Replaces the legacy classify_edges /
ConformalData / DeyMittraData test surface in test_conformal.py.
"""

import numpy as np

from magnelio.geo._subcell import (
    EdgeMaterialData,
    FaceMaterialData,
    compute_subcell_data,
    compute_subcell_data_mu,
)
from magnelio.geo.primitives import Brick
from magnelio.materials.material import Material
from magnelio.mesh.grid import GridLines


def _make_uniform_grid(nx, ny, nz, h):
    x = np.linspace(0, nx * h, nx + 1)
    y = np.linspace(0, ny * h, ny + 1)
    z = np.linspace(0, nz * h, nz + 1)
    return GridLines(x, y, z)


# ---------------------------------------------------------------------------
# Dataclass smoke tests
# ---------------------------------------------------------------------------


class TestEdgeMaterialData:
    def test_creation_and_field_shapes(self):
        n = 100
        em = EdgeMaterialData(
            category=np.zeros(n, dtype=np.int8),
            eps_avg=np.full(n, np.nan),
            sigma_avg=np.full(n, np.nan),
            A_free=np.full(n, np.nan),
            L_free=np.full(n, np.nan),
            f_A=np.full(n, np.nan),
            pec_mask=np.zeros((3, n), dtype=bool),
            enlarged_cell_donor=np.full(n, -1, dtype=np.int64),
            enlarged_cell_area=np.zeros(n),
        )
        assert em.category.shape == (n,)
        assert em.category.dtype == np.int8
        assert em.eps_avg.shape == (n,)
        assert em.A_free.shape == (n,)
        assert em.L_free.shape == (n,)
        assert em.pec_mask.shape == (3, n)
        assert em.enlarged_cell_donor.dtype == np.int64
        assert np.all(em.enlarged_cell_donor == -1)

    def test_nan_means_bulk_or_pec(self):
        em = EdgeMaterialData(
            category=np.array([0, 1, 2, 3], dtype=np.int8),
            eps_avg=np.array([np.nan, 4.0, 1.0, np.nan]),
            sigma_avg=np.full(4, np.nan),
            A_free=np.array([np.nan, 1e-6, 5e-7, np.nan]),
            L_free=np.array([np.nan, 1e-3, 5e-4, np.nan]),
            f_A=np.array([np.nan, 1.0, 0.5, np.nan]),
            pec_mask=np.zeros((3, 4), dtype=bool),
            enlarged_cell_donor=np.full(4, -1, dtype=np.int64),
            enlarged_cell_area=np.zeros(4),
        )
        assert np.isnan(em.eps_avg[0])  # cat 0 bulk
        assert em.eps_avg[1] == 4.0  # cat 1 dielectric
        assert em.eps_avg[2] == 1.0  # cat 2 curved-PEC
        assert np.isnan(em.eps_avg[3])  # cat 3 interior PEC


class TestFaceMaterialData:
    def test_creation(self):
        n = 60
        fm = FaceMaterialData(
            category=np.zeros(n, dtype=np.int8),
            mu_avg=np.full(n, np.nan),
            A_face_free=np.full(n, np.nan),
            L_dual_free=np.full(n, np.nan),
        )
        assert fm.mu_avg.shape == (n,)
        assert fm.category.shape == (n,)
        assert fm.A_face_free.shape == (n,)
        assert fm.L_dual_free.shape == (n,)
        assert np.all(np.isnan(fm.mu_avg))


# ---------------------------------------------------------------------------
# compute_subcell_data — four-category classification
# ---------------------------------------------------------------------------


class TestComputeSubcellData:
    def test_uniform_air_only_cat0(self):
        """All-air domain: every edge category 0, no boundary data."""
        grid = _make_uniform_grid(4, 4, 4, 0.01)
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        mat_id = np.zeros((Nx, Ny, Nz), dtype=np.int32)
        mat_lib = {0: Material("air")}

        em = compute_subcell_data(grid, mat_id, mat_lib, [])

        assert isinstance(em, EdgeMaterialData)
        assert np.all(em.category == 0), "all interior bulk"
        assert np.all(np.isnan(em.eps_avg))
        assert np.all(np.isnan(em.A_free))
        assert np.all(np.isnan(em.L_free))
        assert np.all(em.enlarged_cell_donor == -1)
        assert not em.pec_mask.any()

    def test_full_pec_domain_cat3(self):
        """Domain entirely filled with PEC: every edge masked."""
        grid = _make_uniform_grid(3, 3, 3, 0.01)
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        mat_id = np.ones((Nx, Ny, Nz), dtype=np.int32)
        mat_lib = {
            0: Material("air"),
            1: Material("PEC", is_pec=True),
        }

        em = compute_subcell_data(grid, mat_id, mat_lib, [])

        n_Ex = Nx * (Ny + 1) * (Nz + 1)
        n_Ey = (Nx + 1) * Ny * (Nz + 1)
        n_Ez = (Nx + 1) * (Ny + 1) * Nz
        assert em.pec_mask[0, :n_Ex].all()
        assert em.pec_mask[1, :n_Ey].all()
        assert em.pec_mask[2, :n_Ez].all()

    def test_two_dielectrics_boundary_edges_cat1(self):
        """Air/dielectric boundary: boundary edges land in category 1."""
        grid = _make_uniform_grid(4, 4, 4, 0.01)
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        mat_id = np.zeros((Nx, Ny, Nz), dtype=np.int32)
        mat_id[2:, :, :] = 1
        mat_lib = {
            0: Material("air"),
            1: Material("diel", epsilon=(4.0, 4.0, 4.0)),
        }

        air_box = Brick(
            origin=(0.0, 0.0, 0.0),
            size=(0.02, 0.04, 0.04),
            material=mat_lib[0],
        )
        diel_box = Brick(
            origin=(0.02, 0.0, 0.0),
            size=(0.02, 0.04, 0.04),
            material=mat_lib[1],
        )
        shapes = [(air_box, 0), (diel_box, 1)]

        em = compute_subcell_data(grid, mat_id, mat_lib, shapes)

        assert not em.pec_mask.any(), "no PEC in dielectric-only setup"
        # Some edges must land in cat 1; values must be in [1, 4]
        cat1 = em.category == 1
        assert cat1.any(), "boundary edges should be classified"
        vals = em.eps_avg[cat1]
        assert np.all(vals >= 1.0 - 1e-6)
        assert np.all(vals <= 4.0 + 1e-6)
        # Cat 1 has A_free = A_dual, L_free = L_primal (no PEC):
        # both must be finite and positive.
        assert np.all(em.A_free[cat1] > 0)
        assert np.all(em.L_free[cat1] > 0)

    def test_air_pec_block_unmask_threshold(self):
        """Air/PEC boundary: with eta>0, boundary edges land in cat 2."""
        grid = _make_uniform_grid(4, 4, 4, 0.01)
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        mat_id = np.zeros((Nx, Ny, Nz), dtype=np.int32)
        mat_id[2:, :, :] = 1
        mat_lib = {
            0: Material("air"),
            1: Material("PEC", is_pec=True),
        }

        air_box = Brick(
            origin=(0.0, 0.0, 0.0),
            size=(0.02, 0.04, 0.04),
            material=mat_lib[0],
        )
        pec_box = Brick(
            origin=(0.02, 0.0, 0.0),
            size=(0.02, 0.04, 0.04),
            material=mat_lib[1],
        )
        shapes = [(air_box, 0), (pec_box, 1)]

        # No pec_solid → cat 2 path is disabled, every PEC-adjacent edge
        # falls back to cat 3.
        em_no_dm = compute_subcell_data(
            grid,
            mat_id,
            mat_lib,
            shapes,
            pec_solid=None,
            eta=0.0,
        )
        assert (em_no_dm.category == 2).sum() == 0
        # Some edges must be cat 3 (interior PEC) since the half-domain
        # is fully PEC.
        assert (em_no_dm.category == 3).any()


class TestComputeSubcellDataMu:
    def test_no_nontrivial_mu_returns_all_nan(self):
        grid = _make_uniform_grid(3, 3, 3, 0.01)
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        mat_id = np.zeros((Nx, Ny, Nz), dtype=np.int32)
        mat_lib = {0: Material.air()}
        fm = compute_subcell_data_mu(grid, mat_id, mat_lib, [])
        n_Hx = (Nx + 1) * Ny * Nz
        n_Hy = Nx * (Ny + 1) * Nz
        n_Hz = Nx * Ny * (Nz + 1)
        assert fm.mu_avg.shape == (n_Hx + n_Hy + n_Hz,)
        assert np.all(np.isnan(fm.mu_avg))
