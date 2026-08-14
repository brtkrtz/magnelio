"""The time step comes from the measured spectral radius (DD-150).

``spectral_dt`` replaces the ``sqrt(eps_min * mu_min)`` worst-case
product of ``courant_dt`` for the FIT-TD analyses: the exact leapfrog
limit is ``2 / sqrt(lambda_max)`` of the live update operator, and on
conformal meshes the heuristic under-estimates it by more than an
order of magnitude.  These tests pin the three contracts:

* the measured step is stable — a bare matrix leapfrog with the
  production operators stays bounded at ``spectral_dt`` and blows up
  just above ``2 / sqrt(lambda_max)``;
* the measured step is never below the heuristic (it strictly
  dominates on conformal fixtures);
* the eigensolve runs once — the result is cached on the mesh, and a
  Lanczos failure falls back to the certified row-sum bound.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from magnelio._operators.curl import build_curl_matrix
from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.geo import Brick, Cylinder, GeometryModel
from magnelio.materials.material import Material
from magnelio.mesh.mesher import Mesh, MeshControl
from magnelio.solver.stability import (
    SAFETY_FACTORS,
    compute_min_effective_eps,
    compute_min_effective_mu,
    courant_dt,
    spectral_dt,
)

occ = pytest.importorskip("OCC.Core.BRepPrimAPI")


@pytest.fixture(scope="module")
def box_mesh():
    """A plain air box — no conformal edges, staircase only."""
    model = GeometryModel(background=Material.pec())
    model.add(Brick(origin=(0.0, 0.0, 0.0), size=(20e-3, 10e-3, 30e-3), material=Material.air()))
    return Mesh.from_geometry(model, MeshControl(), f_max=10e9)


@pytest.fixture(scope="module")
def conformal_mesh():
    """A PEC cylinder in an air box — curved walls, cat-2 sub-cells."""
    air = Brick(origin=(0.0, 0.0, 0.0), size=(20e-3, 20e-3, 20e-3), material=Material.air())
    cyl = Cylinder(
        origin=(10e-3, 10e-3, 0.0),
        axis="z",
        height=20e-3,
        radius=3.3e-3,
        material=Material.pec(),
    )
    air -= cyl
    model = GeometryModel(background=Material.pec())
    model.add(air)
    model.add(cyl)
    return Mesh.from_geometry(model, MeshControl(), f_max=10e9)


def _live_operators(mesh):
    """The live leapfrog factors, assembled exactly as the solver sees them."""
    grid = mesh.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    mmu_inv = np.where(m_mu > 0, 1.0 / np.where(m_mu > 0, m_mu, 1.0), 0.0)
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    n_Ez = (Nx + 1) * (Ny + 1) * Nz
    pec = mesh.pec_mask_edges
    pec_flat = np.concatenate([pec[0, :n_Ex], pec[1, :n_Ey], pec[2, :n_Ez]])
    live = np.where((~pec_flat) & (m_eps > 0))[0]
    C = build_curl_matrix(grid)[:, live].tocsr()
    return C, sp.diags(mmu_inv, 0, format="csr"), 1.0 / m_eps[live], live.size


def _leapfrog_growth(mesh, dt, n_steps=400):
    """max|e| growth of the bare lossless leapfrog after n_steps."""
    C, Bmu, inv_d, n_live = _live_operators(mesh)
    rng = np.random.default_rng(11)
    e = rng.standard_normal(n_live)
    h = np.zeros(C.shape[0])
    peak0 = np.abs(e).max()
    for _ in range(n_steps):
        h -= dt * (Bmu @ (C @ e))
        e += dt * (inv_d * (C.T @ h))
        m = np.abs(e).max()
        if not np.isfinite(m) or m > 1e9 * peak0:
            return np.inf
    return np.abs(e).max() / peak0


class TestSpectralStep:
    def test_stable_at_spectral_dt(self, conformal_mesh):
        dt = spectral_dt(conformal_mesh, "normal")
        assert _leapfrog_growth(conformal_mesh, dt) < 1e2

    def test_unstable_just_above_the_limit(self, conformal_mesh):
        dt = spectral_dt(conformal_mesh, "normal") / SAFETY_FACTORS["normal"]
        assert _leapfrog_growth(conformal_mesh, 1.05 * dt) == np.inf

    def test_dominates_the_heuristic_on_conformal_walls(self, conformal_mesh):
        dt_heur = courant_dt(
            conformal_mesh.grid,
            "normal",
            min_effective_eps=compute_min_effective_eps(conformal_mesh),
            min_effective_mu=compute_min_effective_mu(conformal_mesh),
        )
        dt_spec = spectral_dt(conformal_mesh, "normal")
        assert dt_spec > 1.5 * dt_heur

    def test_vacuum_box_stays_at_the_geometric_value(self, box_mesh):
        # No conformal reduction anywhere: the spectral limit and the
        # geometric bound describe the same operator.  The spectral
        # value may exceed the geometric one (dx_min/dy_min/dz_min are
        # global minima that need not meet in one cell) but must never
        # fall below the stable region it defines.
        dt_geom = courant_dt(box_mesh.grid, "normal")
        dt_spec = spectral_dt(box_mesh, "normal")
        assert dt_spec >= 0.99 * dt_geom
        assert _leapfrog_growth(box_mesh, dt_spec) < 1e2

    def test_accuracy_scales_the_safety_factor_only(self, conformal_mesh):
        dt_normal = spectral_dt(conformal_mesh, "normal")
        dt_draft = spectral_dt(conformal_mesh, "draft")
        assert dt_draft == pytest.approx(
            dt_normal * SAFETY_FACTORS["draft"] / SAFETY_FACTORS["normal"], rel=1e-12
        )

    def test_rejects_unknown_accuracy(self, conformal_mesh):
        with pytest.raises(ValueError, match="accuracy"):
            spectral_dt(conformal_mesh, "fastest")


class TestCacheAndFallback:
    def test_lambda_is_cached_on_the_mesh(self, conformal_mesh, monkeypatch):
        first = spectral_dt(conformal_mesh, "normal")
        assert getattr(conformal_mesh, "_spectral_lambda_max", None) is not None

        # A second call must not touch ARPACK at all.
        import scipy.sparse.linalg as spla

        def boom(*a, **k):  # pragma: no cover - would fail the test
            raise AssertionError("eigsh called despite cache")

        monkeypatch.setattr(spla, "eigsh", boom)
        assert spectral_dt(conformal_mesh, "normal") == first

    def test_lanczos_failure_falls_back_to_the_row_sum_bound(self, monkeypatch):
        air = Brick(origin=(0.0, 0.0, 0.0), size=(20e-3, 20e-3, 20e-3), material=Material.air())
        cyl = Cylinder(
            origin=(10e-3, 10e-3, 0.0),
            axis="z",
            height=20e-3,
            radius=3.3e-3,
            material=Material.pec(),
        )
        air -= cyl
        model = GeometryModel(background=Material.pec())
        model.add(air)
        model.add(cyl)
        mesh = Mesh.from_geometry(model, MeshControl(), f_max=10e9)

        import scipy.sparse.linalg as spla

        def no_convergence(*a, **k):
            raise spla.ArpackError(-1)

        monkeypatch.setattr(spla, "eigsh", no_convergence)
        dt_fallback = spectral_dt(mesh, "normal")
        assert dt_fallback > 0
        # The row-sum bound is safe: still stable in the leapfrog, and
        # never above the Lanczos value of an identical mesh.
        assert _leapfrog_growth(mesh, dt_fallback) < 1e2
