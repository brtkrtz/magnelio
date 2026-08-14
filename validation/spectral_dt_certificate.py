"""DD-150 certificate: the spectral time step is sharp and stable.

Reproduces the DD-150 measurement on a public fixture: a PEC cylinder
through an air-filled PEC box — curved conformal walls whose cat-2
sub-cells drive the ``sqrt(eps_min * mu_min)`` heuristic far below the
geometric Courant limit while the true spectral limit stays near it.

Certifies three claims:

1. waste — ``spectral_dt`` exceeds the heuristic step by a large
   factor on conformal meshes (the whole point of DD-150);
2. sharpness — a bare matrix leapfrog with the production operators
   is stable at ``0.999 * dt_crit`` and blows up at ``1.02 * dt_crit``
   (``dt_crit = 2 / sqrt(lambda_max)``, no safety factor);
3. safety of the fallback — the row-sum (Gershgorin) bound never
   exceeds the Lanczos value.

Run from ``magnelio/``:

    mamba run --no-capture-output -n mio python validation/spectral_dt_certificate.py
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from magnelio._operators.curl import build_curl_matrix
from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.geo import Brick, Cylinder, GeometryModel
from magnelio.materials.material import Material
from magnelio.mesh.mesher import Mesh, MeshControl
from magnelio.solver.stability import (
    compute_min_effective_eps,
    compute_min_effective_mu,
    courant_dt,
    spectral_dt,
)


def build_mesh():
    air = Brick(origin=(0.0, 0.0, 0.0), size=(40e-3, 40e-3, 30e-3), material=Material.air())
    cyl = Cylinder(
        origin=(20e-3, 20e-3, 0.0),
        axis="z",
        height=30e-3,
        radius=6.1e-3,
        material=Material.pec(),
    )
    air -= cyl
    model = GeometryModel(background=Material.pec())
    model.add(air)
    model.add(cyl)
    return Mesh.from_geometry(model, MeshControl(), f_max=10e9)


def live_operators(mesh):
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
    return C, sp.diags(mmu_inv, 0, format="csr"), m_eps[live]


def leapfrog_growth(C, Bmu, d, dt, n_steps=4000):
    rng = np.random.default_rng(7)
    e = rng.standard_normal(len(d)) * 1e-6
    h = np.zeros(C.shape[0])
    inv_d = 1.0 / d
    peak0 = np.abs(e).max()
    for step in range(n_steps):
        h -= dt * (Bmu @ (C @ e))
        e += dt * (inv_d * (C.T @ h))
        if step % 100 == 0:
            m = np.abs(e).max()
            if not np.isfinite(m) or m > 1e12 * peak0:
                return np.inf
    return np.abs(e).max() / peak0


def main():
    mesh = build_mesh()
    n = mesh.Nx * mesh.Ny * mesh.Nz
    print(f"mesh: {mesh.Nx} x {mesh.Ny} x {mesh.Nz} = {n} cells")

    dt_geom = courant_dt(mesh.grid, "normal")
    dt_heur = courant_dt(
        mesh.grid,
        "normal",
        min_effective_eps=compute_min_effective_eps(mesh),
        min_effective_mu=compute_min_effective_mu(mesh),
    )
    dt_spec = spectral_dt(mesh, "normal")
    print(f"dt geometric  = {dt_geom:.4e} s")
    print(f"dt heuristic  = {dt_heur:.4e} s")
    print(f"dt spectral   = {dt_spec:.4e} s")
    waste = dt_spec / dt_heur
    print(f"claim 1 (waste): spectral / heuristic = {waste:.1f}")
    assert waste > 3.0, "conformal fixture no longer shows the heuristic gap"

    C, Bmu, d = live_operators(mesh)
    Dm12 = sp.diags(1.0 / np.sqrt(d), 0, format="csr")
    S = (Dm12 @ (C.T @ Bmu @ C) @ Dm12).tocsr()
    lam = eigsh(S, k=1, which="LA", return_eigenvectors=False, tol=1e-10)[0]
    dt_crit = 2.0 / np.sqrt(lam)
    print(f"lambda_max = {lam:.6e}  ->  dt_crit = {dt_crit:.4e} s")

    g_stable = leapfrog_growth(C, Bmu, d, 0.999 * dt_crit)
    g_unstable = leapfrog_growth(C, Bmu, d, 1.02 * dt_crit)
    print(
        f"claim 2 (sharpness): growth at 0.999 dt_crit = {g_stable:.3e}, "
        f"at 1.02 dt_crit = {g_unstable}"
    )
    assert np.isfinite(g_stable) and g_stable < 1e2, "stable side of the limit failed"
    assert g_unstable == np.inf, "unstable side of the limit failed"

    S_abs = S.copy()
    S_abs.data = np.abs(S_abs.data)
    lam_gersh = float(S_abs.sum(axis=1).max())
    print(f"claim 3 (fallback): lambda Gershgorin / Lanczos = {lam_gersh / lam:.3f}")
    assert lam_gersh >= lam, "row-sum bound fell below the spectral radius"

    print("certificate PASSED")


if __name__ == "__main__":
    main()
