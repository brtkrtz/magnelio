"""Pilot diagnostic for the transversal-bbox conformal pathology.

Session-58 follow-up.  After DD-048 the ``PortSpecCoax`` produces
operator-consistent modes via ``solve_tem_laplace``, but on the
*clean* notebook geometry (``GeometryModel(background=pec)``, bbox =
D_a × D_a × L) the FIT-TD run still shows |S|² ≈ 1.34 and
``z_line_num`` divergent under refinement (Z = 63 → 73 Ω at 1× / 2×).
The 1.2 × D_a-Brick variant (the old "crutch" geometry) is stable
(Z = 47.9 Ω, |S|² ≈ 1.05).  This pilot isolates *what changes
between the two setups* in M_eps and the PEC mask.

Hypothesis.  At the four NESW tangential points where the dielectric
cylinder touches the bbox wall, the conformal area-weighting sees only
half the cell neighbourhood (no cells beyond the bbox).  This is
structurally analogous to Bug 5 (session 57), which lived at the
*longitudinal* port-plane boundary; the new pathology lives at the
*transversal* lateral-bbox boundary.  ``flatten_port_plane_mass``
fixes only the longitudinal slab; the transversal slab is untouched.

The pilot reads M_eps along the four bbox-tangent edge slabs (e.g.
along ``y_max`` for Ex- and Ez-edges) and compares against the first
interior slab one cell inward.  A large discrepancy at the cardinal
tangent points but not in the corners (or vice versa) localises the
bug.
"""

from __future__ import annotations

import numpy as np

from magnelio import Material, Mesh, MeshControl
from magnelio._operators.material_matrices import build_M_eps
from magnelio.geo import Brick, Cylinder, Difference, GeometryModel

D_i = 0.41e-3
D_a = 5.0e-3
EPS_R = 9.0
L = 10.0e-3
F_MAX = 10.0e9


def _build_clean_model():
    """Clean: GeometryModel(background=pec), bbox = D_a × D_a × L."""
    pec = Material.pec()
    diel = Material(name="dielectric", epsilon=(EPS_R,) * 3)
    out_cyl = Cylinder(origin=(0, 0, 0), radius=D_a / 2, height=L, axis="z", material=diel)
    in_cyl = Cylinder(origin=(0, 0, 0), radius=D_i / 2, height=L, axis="z", material=pec)
    model = GeometryModel(background=pec)
    model.add(Difference(out_cyl, in_cyl))
    model.add(in_cyl)
    return model


def _build_brick_model(scale: float = 1.2):
    """Crutch: PEC bbox brick of size (scale·D_a) × (scale·D_a) × L."""
    s = scale * D_a
    pec = Material.pec()
    diel = Material(name="dielectric", epsilon=(EPS_R,) * 3)
    bbox = Brick(origin=(-s / 2, -s / 2, 0.0), size=(s, s, L), material=pec)
    out_cyl = Cylinder(origin=(0, 0, 0), radius=D_a / 2, height=L, axis="z", material=diel)
    in_cyl = Cylinder(origin=(0, 0, 0), radius=D_i / 2, height=L, axis="z", material=pec)
    model = GeometryModel()
    model.add(Difference(bbox, out_cyl))
    model.add(Difference(out_cyl, in_cyl))
    model.add(in_cyl)
    return model


def _control(conformal: bool) -> MeshControl:
    return MeshControl(
        min_nodes_per_wavelength=8,
        min_cells_per_feature=3,
        growth_factor=1.4,
        conformal=conformal,
        max_cell_size=0.4e-3,
        min_cell_size=50e-6,
        min_feature_gap=20e-6,
    )


def _examine(label: str, model, conformal: bool) -> None:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    mesh = Mesh.from_geometry(model, _control(conformal), f_max=F_MAX)
    print(f"  Mesh: {mesh.Nx} x {mesh.Ny} x {mesh.Nz}")
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    m_eps = build_M_eps(mesh)

    # Layout offsets (must match FieldState).
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    Ex = m_eps[:n_Ex].reshape((Nx, Ny + 1, Nz + 1))

    # PEC mask
    pec_Ex = mesh.pec_mask_edges[0, :n_Ex].reshape((Nx, Ny + 1, Nz + 1))

    # Examine z-mid plane (k = Nz // 2) along the y_max bbox-wall row.
    k_mid = Nz // 2

    # Ex along y_max (j = Ny):
    ex_wall = Ex[:, Ny, k_mid]
    pec_wall = pec_Ex[:, Ny, k_mid]
    # Compare with first interior slab (j = Ny - 1):
    ex_inner = Ex[:, Ny - 1, k_mid]
    pec_inner = pec_Ex[:, Ny - 1, k_mid]

    # Cardinal tangent index in x: x_mid = (x_n[i] + x_n[i+1])/2 closest to 0
    x_n = mesh.grid.x
    x_c = 0.5 * (x_n[:-1] + x_n[1:])
    i_cardinal = int(np.argmin(np.abs(x_c)))

    print(f"  z_mid = {0.5 * (mesh.grid.z[k_mid] + mesh.grid.z[k_mid + 1]) * 1e3:.3f} mm")
    print(f"  cardinal tangent x_mid index = {i_cardinal} (x_c = {x_c[i_cardinal] * 1e3:+.3f} mm)")
    print()
    print(f"  Ex along y_max (j=Ny={Ny}) at k_mid={k_mid}:")
    print(
        f"  {'i':>3} | {'x_c [mm]':>9} | {'M_eps wall':>11} | "
        f"{'M_eps inner':>12} | {'wall/inner':>10} | "
        f"{'pec_wall':>8} | {'pec_inner':>9}"
    )
    for i in range(Nx):
        ratio = ex_wall[i] / ex_inner[i] if ex_inner[i] > 0 else float("nan")
        marker = " <- tangent" if i == i_cardinal else ""
        print(
            f"  {i:>3} | {x_c[i] * 1e3:>+9.3f} | {ex_wall[i]:>11.4e} | "
            f"{ex_inner[i]:>12.4e} | {ratio:>10.4f} | "
            f"{int(pec_wall[i]):>8d} | {int(pec_inner[i]):>9d}"
            f"{marker}"
        )


if __name__ == "__main__":
    _examine(
        "Pilot 1 — clean (background=pec, bbox=D_a×D_a), conformal=True",
        _build_clean_model(),
        conformal=True,
    )
    _examine(
        "Pilot 2 — crutch (1.2×D_a brick), conformal=True",
        _build_brick_model(scale=1.2),
        conformal=True,
    )
    _examine("Pilot 3 — clean, conformal=False (control)", _build_clean_model(), conformal=False)
