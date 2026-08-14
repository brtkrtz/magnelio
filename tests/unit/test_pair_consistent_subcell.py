"""Unit tests for the DD-053 pair-consistent conformal correction.

Covers the three mechanisms on a small conformal coax
(z-translation-invariant, curved PEC contours at both conductors):

* ``EdgeMaterialData.f_A`` — the conformal free-area fraction of the
  E dual face, populated exactly on category-1/2 edges;
* the tangential-surface re-masking rule (classifier step 6b) — no
  unmasked E edge may connect two nodes of the same masked-edge
  component (fixed-point property);
* the LC-consistent M_mu coupling — the co-located pair product
  ``M_eps * M_mu`` is exactly ``eps0*mu0 * eps_r * dz * dz~`` on every
  interior transversal pair with free partners.

Plus the bulk guard: a homogeneous dielectric geometry without PEC
keeps the plain staircase ``M_mu`` (the coupling is a strict no-op).
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio import Material, Mesh, MeshControl
from magnelio._operators.material_matrices import (
    EPS0,
    MU0,
    _build_avg_d,
    _build_geom_H,
    build_M_eps,
    build_M_mu,
)
from magnelio.geo import Brick, Cylinder, Difference, GeometryModel
from magnelio.geo._subcell import (
    _edge_endpoint_nodes,
    _masked_component_labels,
)

D_I, D_A, EPS_R, LENGTH, F_MAX = 0.41e-3, 5.0e-3, 9.0, 2.4e-3, 10.0e9


@pytest.fixture(scope="module")
def coax_mesh() -> Mesh:
    pec = Material.pec()
    diel = Material.from_isotropic(name="dielectric", epsilon=EPS_R)
    out_cyl = Cylinder(origin=(0, 0, 0), radius=D_A / 2, height=LENGTH, axis="z", material=diel)
    in_cyl = Cylinder(origin=(0, 0, 0), radius=D_I / 2, height=LENGTH, axis="z", material=pec)
    model = GeometryModel(background=pec)
    model.add(Difference(out_cyl, in_cyl))
    model.add(in_cyl)
    control = MeshControl(
        min_nodes_per_wavelength=8,
        min_cells_per_feature=3,
        growth_factor=1.4,
        conformal=True,
        max_cell_size=0.4e-3,
        min_cell_size=50e-6,
        min_feature_gap=20e-6,
    )
    return Mesh.from_geometry(model, control, f_max=F_MAX)


class TestFreeAreaFraction:
    def test_f_A_populated_on_conformal_categories(self, coax_mesh):
        em = coax_mesh.edge_material
        conf = np.isin(em.category, (1, 2))
        assert conf.any()
        vals = em.f_A[conf]
        assert np.all(np.isfinite(vals))
        assert np.all((vals > 0.0) & (vals <= 1.0))
        # Curved contour: some dual faces must be genuinely cut.
        assert (vals < 0.999).any()

    def test_f_A_nan_on_bulk_and_interior_pec(self, coax_mesh):
        em = coax_mesh.edge_material
        assert np.all(np.isnan(em.f_A[np.isin(em.category, (0, 3))]))


class TestTangentialSurfaceRemask:
    def test_no_unmasked_edge_within_one_conductor(self, coax_mesh):
        """Fixed point of the step-6b rule on the final mesh mask."""
        grid = coax_mesh.grid
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        n_Ex = Nx * (Ny + 1) * (Nz + 1)
        n_Ey = (Nx + 1) * Ny * (Nz + 1)
        n_Ez = (Nx + 1) * (Ny + 1) * Nz
        pec = coax_mesh.pec_mask_edges
        masked = np.concatenate(
            [
                pec[0, :n_Ex],
                pec[1, :n_Ey],
                pec[2, :n_Ez],
            ]
        )
        node_a, node_b = _edge_endpoint_nodes(grid)
        labels = _masked_component_labels(
            node_a,
            node_b,
            masked,
            (Nx + 1) * (Ny + 1) * (Nz + 1),
        )
        tangential = ~masked & (labels[node_a] == labels[node_b])
        assert not tangential.any()

    def test_donors_are_never_masked(self, coax_mesh):
        em = coax_mesh.edge_material
        donors = em.enlarged_cell_donor[em.enlarged_cell_donor >= 0]
        if donors.size == 0:
            pytest.skip("no enlarged-cell donors on this mesh")
        grid = coax_mesh.grid
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        n_Ex = Nx * (Ny + 1) * (Nz + 1)
        n_Ey = (Nx + 1) * Ny * (Nz + 1)
        n_Ez = (Nx + 1) * (Ny + 1) * Nz
        pec = coax_mesh.pec_mask_edges
        masked = np.concatenate(
            [
                pec[0, :n_Ex],
                pec[1, :n_Ey],
                pec[2, :n_Ez],
            ]
        )
        assert not masked[donors].any()


class TestPairConsistentCoupling:
    def test_interior_transversal_pairs_exact(self, coax_mesh):
        """Every interior pair with free partners satisfies the identity."""
        mesh = coax_mesh
        grid = mesh.grid
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        assert Nz >= 5, "fixture must have interior z slabs"
        m_eps = build_M_eps(mesh)
        m_mu = build_M_mu(mesh)
        dz_avg = _build_avg_d(grid.dz, Nz)
        n_Hx = (Nx + 1) * Ny * Nz
        pec = mesh.pec_mask_edges

        k = Nz // 2  # interior slab
        checked = 0
        # Hy faces (i, j, k) paired with Ex edges (i, j, k) / (i, j, k+1).
        for i in range(Nx):
            for j in range(Ny + 1):
                e1 = (i * (Ny + 1) + j) * (Nz + 1) + k
                e2 = e1 + 1
                if pec[0, e1] or pec[0, e2]:
                    continue
                f = n_Hx + (i * (Ny + 1) + j) * Nz + k
                prod = m_eps[e1] * m_mu[f]
                ref = EPS0 * MU0 * EPS_R * grid.dz[k] * dz_avg[k]
                assert prod == pytest.approx(ref, rel=1e-9), (i, j, k)
                checked += 1
        assert checked > 10

    def test_homogeneous_dielectric_is_noop(self):
        """No PEC anywhere: the coupling must keep the staircase M_mu."""
        diel = Material.from_isotropic(name="diel", epsilon=4.0)
        model = GeometryModel()
        model.add(Brick(origin=(0, 0, 0), size=(4e-3, 3e-3, 5e-3), material=diel))
        control = MeshControl(conformal=True, max_cell_size=1e-3)
        mesh = Mesh.from_geometry(model, control, f_max=10e9)
        m_mu = build_M_mu(mesh)
        expected = MU0 * _build_geom_H(mesh.grid)
        np.testing.assert_allclose(m_mu, expected, rtol=1e-12)

    def test_magnetic_bulk_is_noop(self):
        """μ_r = 2 bulk with a uniform ladder: the LC pair target must
        carry the face's μ̄ (session-126 fix — the missing factor
        HALVED M_mu on every μ_r = 2 uniform-ladder face; invisible on
        μ_r = 1 fixtures, caught by the WP-C5 rotated μ-slab
        reference).  The no-op property is the sharp form: the pair
        value must reproduce the μ_r-bearing staircase exactly."""
        mag = Material(name="mag", epsilon=(2.0,) * 3, mu=(2.0,) * 3)
        model = GeometryModel()
        model.add(Brick(origin=(0, 0, 0), size=(4e-3, 3e-3, 5e-3), material=mag))
        control = MeshControl(conformal=True, max_cell_size=1e-3)
        mesh = Mesh.from_geometry(model, control, f_max=10e9)
        m_mu = build_M_mu(mesh)
        expected = 2.0 * MU0 * _build_geom_H(mesh.grid)
        np.testing.assert_allclose(m_mu, expected, rtol=1e-12)
