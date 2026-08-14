"""DD-099 WP-B1.3: the unregistered-wall warning.

An interior conductor shell thinner than one cell carries near-equal
PEC coverage on opposite cell faces; its wall vector ``ΔA_face_pec``
cancels and the loss surface silently books ~nothing.  Mesh
consolidation warns once per scene.  Threshold rationale and the
zero-false-positive census: BOUNDARY_WALL_PLAN WP-B0/B1 and the
internal dossier ``investigations/boundary_wall/MEASUREMENTS.md``
(suite floor ~0.49,
must-fire signal 0.0055, threshold 0.1).
"""

import warnings

import numpy as np
import pytest

pytest.importorskip("OCC.Core.BRepPrimAPI")

from magnelio import Material, MeshControl
from magnelio.geo import Brick, Cylinder, Difference, GeometryModel
from magnelio.mesh._surfaces import detect_unregistered_walls
from magnelio.mesh.mesher import Mesh


def _thin_shell_mesh():
    """30 µm cylindrical PEC shell in 150 µm cells — sub-cell, curved,
    so neither the planar thin-sheet pipeline nor the feature anchors
    resolve it (floor 100 µm)."""
    pec, air = Material.pec(), Material.air()
    r, t, h, dom = 1.0e-3, 30e-6, 2.0e-3, 3.0e-3
    box = Brick(origin=(-dom / 2, -dom / 2, -0.5e-3), size=(dom, dom, h + 1.0e-3), material=air)
    outer = Cylinder(origin=(0, 0, 0), radius=r + t / 2, height=h, axis="z", material=pec)
    inner = Cylinder(origin=(0, 0, 0), radius=r - t / 2, height=h, axis="z", material=air)
    model = GeometryModel(background=air)
    model.add(Difference(box, outer))
    model.add(Difference(outer, inner))
    model.add(inner)
    control = MeshControl(
        min_nodes_per_wavelength=8,
        min_cells_per_feature=0,
        growth_factor=1.4,
        conformal=True,
        max_cell_size=0.15e-3,
        min_cell_size=100e-6,
        min_feature_gap=20e-6,
    )
    return Mesh.from_geometry(model, control, f_max=10e9)


def _ladder_coax_mesh():
    """The DD-053 pair-consistency fixture — conformal, curved,
    bbox-tangent outer conductor: every wall is registered, so the
    warning must stay silent."""
    pec = Material.pec()
    diel = Material.from_isotropic(name="dielectric", epsilon=9.0)
    out_cyl = Cylinder(origin=(0, 0, 0), radius=2.5e-3, height=2.4e-3, axis="z", material=diel)
    in_cyl = Cylinder(origin=(0, 0, 0), radius=0.205e-3, height=2.4e-3, axis="z", material=pec)
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
    return Mesh.from_geometry(model, control, f_max=10e9)


def test_fires_on_subcell_shell():
    with pytest.warns(UserWarning, match="conductor shell"):
        mesh = _thin_shell_mesh()
    cells, ratios = detect_unregistered_walls(mesh)
    assert cells.shape[0] > 0
    assert ratios.min() < 0.1
    # the flagged cells sit on the shell radius
    grid = mesh.grid
    xc = 0.5 * (grid.x[:-1] + grid.x[1:])
    yc = 0.5 * (grid.y[:-1] + grid.y[1:])
    r = np.hypot(xc[cells[:, 0]], yc[cells[:, 1]])
    assert 0.9e-3 < r.min() and r.max() < 1.1e-3


def test_silent_on_registered_conformal_scene():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mesh = _ladder_coax_mesh()
    assert not [w for w in caught if "conductor shell" in str(w.message)]
    cells, _ = detect_unregistered_walls(mesh)
    assert cells.shape[0] == 0
