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


# ---------------------------------------------------------------------------
# Thin sheets are painted after step 3d — they need the extension too
# ---------------------------------------------------------------------------

H_SUB, W_STRIP, T_MET = 0.787e-3, 2.4e-3, 35e-6


def _microstrip_to_wall_model(xmin="CPML", frame=False):
    """A 35 um strip on a substrate, running from the ``xmin`` face inward.

    With ``frame`` a PEC frame surrounds the strip at the face, so a
    port window whose ring lies on conductor (DD-198) can sit there.
    """
    faces = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
    bcs = {f: ("CPML" if f == "xmax" else "PEC") for f in faces}
    bcs["xmin"] = xmin
    sub = mio.Material.from_isotropic(name="sub", epsilon=2.2)
    y0, y1, z1 = -8e-3, 8e-3, 8e-3
    strip = geo.Brick(
        origin=(0.0, -W_STRIP / 2, H_SUB), size=(12e-3, W_STRIP, T_MET), material="pec"
    )
    model = mio.GeometryModel(boundary_conditions=bcs)
    substrate = geo.Brick(origin=(0.0, y0, 0.0), size=(20e-3, y1 - y0, H_SUB), material=sub)
    air = geo.Brick(origin=(0.0, y0, H_SUB), size=(20e-3, y1 - y0, z1 - H_SUB), material="air")
    if frame:
        shield = geo.Difference(
            geo.Brick(origin=(0.0, -6.5e-3, 0.0), size=(4e-3, 13e-3, 6.5e-3), material="pec"),
            geo.Brick(origin=(-1e-3, -6e-3, -1e-3), size=(6e-3, 12e-3, 7e-3), material="air"),
            material="pec",
        )
        model.add(geo.Difference(substrate, shield))
        model.add(geo.Difference(air, strip, shield))
        model.add(strip)
        model.add(shield)
    else:
        model.add(substrate)
        model.add(geo.Difference(air, strip))
        model.add(strip)
    return model


def _microstrip_mesh(xmin="CPML"):
    return mio.Mesh.from_geometry(
        _microstrip_to_wall_model(xmin),
        mio.MeshControl(min_nodes_per_wavelength=20, min_cell_size=0.25e-3),
        f_max=12e9,
    )


def test_pml_slabs_carry_a_thin_sheet_touching_the_face():
    mesh = _microstrip_mesh()
    n = mesh.pml_cells["xmin"]
    ey = _ey_slabs(mesh)
    interior = ey[n + 2]
    k = int(np.argmin(np.abs(mesh.grid.z - H_SUB)))
    assert interior[:, k].sum() > 0, "the strip's y-edges must be masked on the sheet plane"
    for i in range(n + 1):
        np.testing.assert_array_equal(ey[i], interior)


def test_microstrip_window_in_an_absorbing_face_resolves_as_a_line_mode():
    from magnelio import ports

    model = _microstrip_to_wall_model(frame=True)
    model.add_port(
        ports.PortWaveguide(
            name="in", plane="xmin", corners=((None, -6e-3, 0.0), (None, 6e-3, 6e-3))
        )
    )
    mesh = mio.Mesh.from_geometry(
        model, mio.MeshControl(min_nodes_per_wavelength=20, min_cell_size=0.25e-3), f_max=12e9
    )
    report = mio.AnalysisScatteringTD(mesh=mesh, verbose=False).solve_ports()["in"]
    mode = report.modes[0]
    assert mode.mode_type.name == "TEM"
    assert 30.0 < float(mode.z_line) < 60.0
