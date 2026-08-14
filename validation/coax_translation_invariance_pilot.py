"""Translation-invariance check on the clean coax geometry.

Hypothesis check.  At the four NESW tangential bbox-wall points the
clean ``D_a × D_a`` setup shows M_eps asymmetry between the
y_max-slab and the y_max-1-slab (pilot 1).  But the *port-plane* mode
solver only sees the 2D (x, y)-cross-section at z=0; if the FIT mesh
is translation-invariant along z, every interior z-slab has the same
2D M_eps slice as the port plane.  ``flatten_port_plane_mass`` then
brings the port slab in line with the interior — and the wave should
propagate consistently.

This pilot tests whether the M_eps slice at each z-cell is identical
to the slice at any other z-cell (modulo the port-plane flatten).
A z-variation in M_eps would mean the volume operator does *not*
share the eigenmode of the (port-flattened) 2D slice — the new bug.

Expected: bare ``Mesh.from_geometry(conformal=True)`` may show
z-variation at the bbox-wall edges because the conformal area
weighting at the *first* and *last* z-slabs sees only half a cell
neighbourhood (Bug 5 longitudinal pattern), but the *interior*
z-slabs (k = 1..Nz-2) should be byte-identical.
"""

from __future__ import annotations

import numpy as np

from magnelio import Material, Mesh, MeshControl
from magnelio._operators.material_matrices import build_M_eps
from magnelio.geo import Cylinder, Difference, GeometryModel

D_i = 0.41e-3
D_a = 5.0e-3
EPS_R = 9.0
L = 10.0e-3
F_MAX = 10.0e9


def _model():
    pec = Material.pec()
    diel = Material(name="dielectric", epsilon=(EPS_R,) * 3)
    out_cyl = Cylinder(origin=(0, 0, 0), radius=D_a / 2, height=L, axis="z", material=diel)
    in_cyl = Cylinder(origin=(0, 0, 0), radius=D_i / 2, height=L, axis="z", material=pec)
    model = GeometryModel(background=pec)
    model.add(Difference(out_cyl, in_cyl))
    model.add(in_cyl)
    return model


def main() -> None:
    mesh = Mesh.from_geometry(
        _model(),
        MeshControl(
            min_nodes_per_wavelength=8,
            min_cells_per_feature=3,
            growth_factor=1.4,
            conformal=True,
            max_cell_size=0.4e-3,
            min_cell_size=50e-6,
            min_feature_gap=20e-6,
        ),
        f_max=F_MAX,
    )
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    print(f"Mesh: {Nx} x {Ny} x {Nz}")

    m_eps = build_M_eps(mesh)
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    Ex = m_eps[:n_Ex].reshape((Nx, Ny + 1, Nz + 1))

    # For Ex, the (Ny+1, Nz+1) layout has nodes along z; Ex is a
    # primal edge along x, located at (x_mid, y_node, z_node).  For
    # *interior* z-nodes (k = 1..Nz-1) the Ex value should be the same
    # since the geometry is translation-invariant along z.

    print("\nEx M_eps (j, k=k_test) variation along z at the y_max wall (j=Ny):")
    j = Ny  # bbox wall
    print(f"  j = {j} (y_max)")
    print(f"  {'i':>3} | {'x_c [mm]':>9} |", end="")
    for k in (0, 1, 2, Nz // 2, Nz - 2, Nz - 1, Nz):
        print(f" k={k:>2} (z={mesh.grid.z[k] * 1e3:5.2f} mm)", end="")
    print()
    x_c = 0.5 * (mesh.grid.x[:-1] + mesh.grid.x[1:])
    for i in range(Nx):
        print(f"  {i:>3} | {x_c[i] * 1e3:>+9.3f} |", end="")
        for k in (0, 1, 2, Nz // 2, Nz - 2, Nz - 1, Nz):
            print(f"   {Ex[i, j, k]:.3e}", end="")
        print()

    print("\nz-translation-invariance assertion (interior k = 1..Nz-1):")
    interior_min = 1
    interior_max = Nz  # exclusive — so we go up to Nz-1
    interior_z_idx = list(range(interior_min, interior_max))
    Ex_interior = Ex[:, :, interior_z_idx]  # (Nx, Ny+1, n_interior)
    # Compare every interior k against k=interior_min:
    ref = Ex_interior[:, :, 0:1]
    diff = np.abs(Ex_interior - ref)
    rel = np.where(np.abs(ref) > 0, diff / np.abs(ref), diff)
    max_rel = float(np.max(rel))
    print(f"  max |Ex(z=k) - Ex(z=k_ref)| / |Ex(z=k_ref)| over interior k: {max_rel:.4e}")
    if max_rel < 1e-9:
        print("  -> z-translation-invariant: M_eps slice is identical across all interior z (good)")
    else:
        print("  -> NOT z-translation-invariant: a z-dependent bug is present")
        # Locate worst offender
        flat_idx = int(np.argmax(rel))
        i, j_w, k_w = np.unravel_index(flat_idx, rel.shape)
        k_w_actual = interior_z_idx[k_w]
        print(
            f"  worst at i={i}, j={j_w}, k={k_w_actual}: "
            f"Ex={Ex[i, j_w, k_w_actual]:.4e}  ref={ref[i, j_w, 0]:.4e}"
        )


if __name__ == "__main__":
    main()
