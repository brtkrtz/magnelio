"""Thin PEC sheets rasterise their true footprint, not their bbox.

The WP-M2 sheet rasterisation painted the detected sheet's bounding-box
rectangle onto the grid plane — exact for a straight strip, but a
silent short across the whole span for any non-rectangular
metallization (an L-shape filled its corner, a ring became a solid
plane).  The footprint-exact path classifies each candidate edge
midpoint against the source shape's OCC solid instead.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("OCC.Core.BRepPrimAPI")

from magnelio.geo import Brick, Difference, GeometryModel, Union
from magnelio.materials.material import Material
from magnelio.mesh.mesher import Mesh, MeshControl

H_SUB = 0.5e-3
T_MET = 17e-6
ARM = 0.6e-3  # metal arm width
SPAN = 3.0e-3  # L-shape leg length


def _l_shape_mesh():
    """Air box with an L-shaped 17 um PEC sheet on a thin substrate."""
    pec = Material.pec()
    air = Material.from_isotropic(name="air", epsilon=1.0)
    sub = Material.from_isotropic(name="sub", epsilon=3.0)

    leg_x = Brick(origin=(0.0, H_SUB, 0.0), size=(SPAN, T_MET, ARM), material=pec)
    leg_z = Brick(origin=(0.0, H_SUB, 0.0), size=(ARM, T_MET, SPAN), material=pec)
    metal = Union(leg_x, leg_z, material=pec)

    model = GeometryModel(
        boundary_conditions={f: "PEC" for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    )
    model.add(Brick(origin=(0.0, 0.0, 0.0), size=(SPAN, H_SUB, SPAN), material=sub))
    model.add(
        Difference(
            Brick(origin=(0.0, H_SUB, 0.0), size=(SPAN, 2.0e-3, SPAN), material=air),
            metal,
        )
    )
    model.add(metal)
    return Mesh.from_geometry(
        model,
        MeshControl(min_nodes_per_wavelength=20, min_cell_size=100e-6),
        f_max=10e9,
    )


def test_l_shape_sheet_leaves_the_bbox_corner_open():
    mesh = _l_shape_mesh()
    g = mesh.grid
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    jy = int(np.argmin(np.abs(g.y - H_SUB)))
    assert abs(g.y[jy] - H_SUB) < 1e-9  # sheet plane is a grid line

    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    ex = mesh.pec_mask_edges[0, :n_Ex].reshape(Nx, Ny + 1, Nz + 1)
    x_c = 0.5 * (g.x[:-1] + g.x[1:])

    # A point deep inside the x-leg: marked.
    i_leg = int(np.argmin(np.abs(x_c - 0.8 * SPAN)))
    k_leg = int(np.argmin(np.abs(g.z - ARM / 2)))
    assert ex[i_leg, jy, k_leg], "edge inside the metal leg must be PEC"

    # The far bbox corner (large x AND large z) carries no metal: the
    # bbox fill marked it, the footprint rasterisation must not.
    k_far = int(np.argmin(np.abs(g.z - 0.8 * SPAN)))
    assert not ex[i_leg, jy, k_far], (
        "edge in the empty bbox corner of the L-shape was marked PEC — "
        "bbox-fill regression in the thin-sheet rasterisation"
    )
