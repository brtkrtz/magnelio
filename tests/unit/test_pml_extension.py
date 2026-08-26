"""DD-198 step 0: the sub-cell classification continues into the PML.

The mesher grows an absorbing face outward and continues the cell
materials (step 3b); the conformal classifier, working against the
B-rep solids that end at the nominal bounding box, used to read the
extension as free space and dropped the PEC mask of a conductor
touching the face there (KB-029).  The extension slabs must now carry
the first interior slab's data.
"""

from __future__ import annotations

import numpy as np
import pytest

import magnelio as mio
from magnelio import geo

A, B, WALL, L = 20e-3, 10e-3, 3e-3, 30e-3


def _tube_mesh(conformal=True, xmin="CPML"):
    faces = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
    bcs = {f: ("CPML" if f == "xmin" else "PEC") for f in faces}
    bcs["xmin"] = xmin
    tube = geo.Brick(
        origin=(0.0, -A / 2 - WALL, -B / 2 - WALL),
        size=(L, A + 2 * WALL, B + 2 * WALL),
        material="pec",
    )
    bore = geo.Brick(origin=(-1e-3, -A / 2, -B / 2), size=(L + 2e-3, A, B), material="air")
    model = mio.GeometryModel(boundary_conditions=bcs)
    model.add(geo.Difference(tube, bore))
    model.add(
        geo.Difference(
            geo.Brick(origin=(0.0, -0.025, -0.02), size=(L, 0.05, 0.04), material="air"), tube
        )
    )
    return mio.Mesh.from_geometry(
        model, mio.MeshControl(max_cell_size=2e-3, conformal=conformal), f_max=12e9
    )


def _ey_slabs(mesh):
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    return mesh.pec_mask_edges[1, :n_Ey].reshape(Nx + 1, Ny, Nz + 1)


@pytest.mark.parametrize("conformal", [True, False])
def test_pml_slabs_carry_the_interior_conductor_mask(conformal):
    mesh = _tube_mesh(conformal=conformal)
    n = mesh.pml_cells["xmin"]
    ey = _ey_slabs(mesh)
    interior = ey[n + 2]
    assert interior.sum() > 0
    for i in range(n + 1):
        np.testing.assert_array_equal(ey[i], interior)


def test_categories_and_fractions_follow_the_interior_slab():
    mesh = _tube_mesh(conformal=True)
    n = mesh.pml_cells["xmin"]
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    cat = mesh.edge_material.category
    cat_ey = cat[n_Ex : n_Ex + n_Ey].reshape(Nx + 1, Ny, Nz + 1)
    cat_ex = cat[:n_Ex].reshape(Nx, Ny + 1, Nz + 1)
    for i in range(n + 1):
        np.testing.assert_array_equal(cat_ey[i], cat_ey[n + 1])
    for i in range(n):
        np.testing.assert_array_equal(cat_ex[i], cat_ex[n])
    # No donor link points across the copied slabs.
    donor = mesh.edge_material.enlarged_cell_donor
    assert np.all(donor[donor >= 0] < donor.size)


def test_pec_face_meshes_are_untouched():
    mesh = _tube_mesh(conformal=True, xmin="PEC")
    assert not mesh.pml_cells
