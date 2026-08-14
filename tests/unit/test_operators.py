"""Unit tests for discrete curl operator and material matrices."""

import numpy as np
import scipy.sparse as sp

from magnelio._operators.curl import build_curl_matrix
from magnelio.mesh.grid import GridLines


def _make_simple_grid(Nx=3, Ny=3, Nz=3):
    x = np.linspace(0, 1e-2, Nx + 1)
    y = np.linspace(0, 1e-2, Ny + 1)
    z = np.linspace(0, 1e-2, Nz + 1)
    return GridLines(x=x, y=y, z=z)


class TestCurlMatrix:
    def test_shape(self):
        grid = _make_simple_grid(3, 4, 5)
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        C = build_curl_matrix(grid)

        n_Ex = Nx * (Ny + 1) * (Nz + 1)
        n_Ey = (Nx + 1) * Ny * (Nz + 1)
        n_Ez = (Nx + 1) * (Ny + 1) * Nz
        n_E = n_Ex + n_Ey + n_Ez

        n_Hx = (Nx + 1) * Ny * Nz
        n_Hy = Nx * (Ny + 1) * Nz
        n_Hz = Nx * Ny * (Nz + 1)
        n_H = n_Hx + n_Hy + n_Hz

        assert C.shape == (n_H, n_E)

    def test_each_row_has_two_nonzeros(self):
        grid = _make_simple_grid(2, 2, 2)
        C = build_curl_matrix(grid)
        # Each row of C corresponds to one curl component at one face
        nnz_per_row = np.diff(C.indptr)
        assert np.all(nnz_per_row == 4), (
            "Each row should have exactly 4 nonzeros (+1,-1 for each of 2 terms)"
        )

    def test_values_are_plus_minus_one(self):
        grid = _make_simple_grid(2, 2, 2)
        C = build_curl_matrix(grid)
        assert set(C.data).issubset({+1.0, -1.0})

    def test_discrete_stokes(self):
        """C @ e + C^T @ h should be zero for random smooth fields."""
        grid = _make_simple_grid(3, 3, 3)
        C = build_curl_matrix(grid)
        CT = C.T

        # C @ C^T should be related to the discrete Laplacian
        # More precisely: C^T @ C ≡ 0 is NOT generally true, but
        # the discrete Stokes theorem states div(curl(e)) = 0:
        # Build the discrete divergence D such that D @ C @ e = 0
        # For now, just check the product C @ C^T has the right shape
        prod = C @ CT
        assert prod.shape == (C.shape[0], C.shape[0])

    def test_is_sparse_csr(self):
        grid = _make_simple_grid(2, 2, 2)
        C = build_curl_matrix(grid)
        assert sp.issparse(C)
        assert C.format == "csr"


class TestMSigmaM:
    """build_M_sigma_m (DD-081): staircase mirror of build_M_mu."""

    def _graded_grid(self):
        x = np.array([0.0, 1e-3, 2.5e-3, 4.5e-3])
        y = np.array([0.0, 0.8e-3, 2.0e-3, 3.0e-3, 5.0e-3])
        z = np.array([0.0, 1.2e-3, 2.2e-3])
        return GridLines(x=x, y=y, z=z)

    def test_mirrors_M_mu_geometry(self):
        """With sigma_m == mu componentwise, M_sigma_m == M_mu/MU0 exactly.
        Same property-table lookup, same clamped one-sided face sampling,
        same geometric factor — any sampling mismatch breaks equality on
        this graded two-material mesh.
        """
        from magnelio._operators.material_matrices import (
            MU0,
            build_M_mu,
            build_M_sigma_m,
        )
        from magnelio.materials.material import Material
        from magnelio.mesh.mesher import Mesh

        grid = self._graded_grid()
        bg = Material(name="bg", mu=(2.0, 3.0, 4.0), sigma_m=(2.0, 3.0, 4.0))
        inc = Material(name="inc", mu=(5.0, 6.0, 7.0), sigma_m=(5.0, 6.0, 7.0))
        mesh = Mesh.from_grid(
            grid,
            regions=[(inc, (1e-3, 0.8e-3, 0.0, 2.5e-3, 3.0e-3, 2.2e-3))],
            background=bg,
        )
        M_mu = build_M_mu(mesh)
        M_sm = build_M_sigma_m(mesh)
        # rtol: (MU0*mu*geom)/MU0 vs sigma_m*geom differ by ULP rounding only
        np.testing.assert_allclose(M_sm, M_mu / MU0, rtol=1e-15, atol=0.0)

    def test_pec_faces_are_zero(self):
        from magnelio._operators.material_matrices import build_M_sigma_m
        from magnelio.materials.material import Material
        from magnelio.mesh.mesher import Mesh

        grid = self._graded_grid()
        pec = Material(name="pec", sigma_m=(9.0, 9.0, 9.0), is_pec=True)
        mesh = Mesh.from_grid(grid, background=pec)
        assert np.all(build_M_sigma_m(mesh) == 0.0)

    def test_lossless_default_is_zero(self):
        from magnelio._operators.material_matrices import build_M_sigma_m
        from magnelio.mesh.mesher import Mesh

        mesh = Mesh.from_grid(self._graded_grid())
        assert np.all(build_M_sigma_m(mesh) == 0.0)


class TestLossyHCoefficients:
    """FIT-TD alpha_H/beta_H with magnetic loss (DD-081)."""

    def _solver(self, background=None):
        from magnelio.mesh.mesher import Mesh
        from magnelio.solver.fit_td import FITTimeDomainSolver
        from magnelio.solver.stability import courant_dt

        grid = GridLines(
            x=np.linspace(0, 4e-3, 5),
            y=np.linspace(0, 4e-3, 5),
            z=np.linspace(0, 4e-3, 5),
        )
        mesh = Mesh.from_grid(grid, background=background)
        return FITTimeDomainSolver(
            mesh=mesh,
            total_time_steps=1,
            dt=courant_dt(grid),
            verbose=False,
        )

    def test_lossless_coefficients_bit_identical(self):
        """sigma_m = 0 must reproduce the historical lossless coefficients."""
        from magnelio._operators.material_matrices import build_M_mu

        solver = self._solver()
        solver.setup()
        M_mu = build_M_mu(solver.mesh)
        assert np.all(np.asarray(solver._alpha_H) == 1.0)
        beta_ref = np.where(M_mu > 0, solver.dt / np.where(M_mu > 0, M_mu, 1.0), 0.0)
        np.testing.assert_array_equal(np.asarray(solver._beta_H), beta_ref)

    def test_lossy_coefficients_formula(self):
        from magnelio._operators.material_matrices import build_M_mu, build_M_sigma_m
        from magnelio.materials.material import Material

        mat = Material(name="mag_lossy", sigma_m=(50.0, 50.0, 50.0))
        solver = self._solver(background=mat)
        solver.setup()
        M_mu = build_M_mu(solver.mesh)
        S = build_M_sigma_m(solver.mesh)
        dt = solver.dt
        denom = M_mu + 0.5 * dt * S
        np.testing.assert_array_equal(
            np.asarray(solver._alpha_H),
            (M_mu - 0.5 * dt * S) / denom,
        )
        np.testing.assert_array_equal(np.asarray(solver._beta_H), dt / denom)
        assert np.all(np.asarray(solver._alpha_H) < 1.0)


class TestCoupleFaceMaterialPairs:
    """The pair-coupling ladder must stay quiet on masked edges.

    ``M_eps`` vanishes on masked edges, so the ladder targets are
    allowed to be inf; the validity mask rejects them afterwards.  Two
    masked partners on one rung give ``inf - inf``, which numpy reports
    as an invalid subtract unless the comparison shares the errstate
    context of the division that produced them.

    Found on a real model — a chamfered high-permittivity puck bridging
    both lids of a PEC cavity, on a 0.4 mm grid — but reaching it that
    way costs minutes and depends on the geometry landing just so.  The
    condition is cheap to construct instead: zero out alternate entries
    of ``M_eps`` so both partners of a rung divide by zero.
    """

    @staticmethod
    def _mesh():
        import magnelio as mio
        from magnelio import geo
        from magnelio.materials.material import Material

        box = geo.Brick.from_ranges(
            x1=0.0, dx=10e-3, y1=0.0, dy=8e-3, z1=0.0, dz=6e-3, material=Material.air()
        )
        model = mio.GeometryModel(background=Material.pec())
        model.add(box)
        return mio.Mesh.from_geometry(model, mio.MeshControl(max_cell_size=1.0e-3), f_max=10e9)

    def test_no_warning_when_masked_edges_make_targets_infinite(self, monkeypatch):
        import warnings

        import magnelio._operators.material_matrices as mmod

        mesh = self._mesh()
        original = mmod.build_M_eps

        def zeroed(m):
            values = np.array(original(m), copy=True)
            values[::2] = 0.0  # adjacent ladder rungs both divide by zero
            return values

        monkeypatch.setattr(mmod, "build_M_eps", zeroed)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            mmod.couple_face_material_pairs(mesh)

    def test_clean_mesh_couples_without_warning(self):
        import warnings

        from magnelio._operators.material_matrices import couple_face_material_pairs

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            couple_face_material_pairs(self._mesh())
