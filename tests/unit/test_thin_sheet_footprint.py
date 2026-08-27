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


def _l_shape_model():
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
    return model


def _l_shape_mesh():
    return Mesh.from_geometry(
        _l_shape_model(),
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


# ---------------------------------------------------------------------------
# The section path against the classifier path, and the semantics it keeps
# ---------------------------------------------------------------------------

from magnelio.geo import Cylinder  # noqa: E402
from magnelio.mesh import _conformal  # noqa: E402

R_OUT = 1.0e-3
R_IN = 0.6e-3


def _ring_model():
    """A 17 um PEC annulus on a substrate, sheet normal y."""
    pec = Material.pec()
    air = Material.from_isotropic(name="air", epsilon=1.0)
    sub = Material.from_isotropic(name="sub", epsilon=3.0)
    c = (SPAN / 2, H_SUB, SPAN / 2)
    ring = Difference(
        Cylinder(origin=c, radius=R_OUT, height=T_MET, axis="y", material=pec),
        Cylinder(origin=c, radius=R_IN, height=T_MET, axis="y", material=pec),
        material=pec,
    )
    model = GeometryModel(
        boundary_conditions={f: "PEC" for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    )
    model.add(Brick(origin=(0.0, 0.0, 0.0), size=(SPAN, H_SUB, SPAN), material=sub))
    model.add(
        Difference(
            Brick(origin=(0.0, H_SUB, 0.0), size=(SPAN, 2.0e-3, SPAN), material=air),
            ring,
        )
    )
    model.add(ring)
    return model


def _mesh(model, **control):
    return Mesh.from_geometry(
        model,
        MeshControl(min_nodes_per_wavelength=20, min_cell_size=100e-6, **control),
        f_max=10e9,
    )


def _classifier_mask(model, monkeypatch, **control):
    """pec_mask_edges of the same build with the classifier path forced."""
    monkeypatch.setattr(
        _conformal, "rasterize_thin_sheet_footprint", _conformal._rasterize_by_classifier
    )
    return _mesh(model, **control).pec_mask_edges.copy()


def _ex_on_sheet(mesh):
    g = mesh.grid
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    jy = int(np.argmin(np.abs(g.y - H_SUB)))
    assert abs(g.y[jy] - H_SUB) < 1e-9
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    ex = mesh.pec_mask_edges[0, :n_Ex].reshape(Nx, Ny + 1, Nz + 1)
    return ex[:, jy, :], 0.5 * (g.x[:-1] + g.x[1:]), g.z


def test_section_path_matches_the_classifier_on_the_l_shape(monkeypatch):
    section = _l_shape_mesh().pec_mask_edges.copy()
    classified = _classifier_mask(_l_shape_model(), monkeypatch)
    assert section.any()
    np.testing.assert_array_equal(section, classified)


def test_ring_sheet_keeps_its_bore_open_and_agrees_with_the_classifier(monkeypatch):
    model = _ring_model()
    mesh = _mesh(model)
    ex, x_c, z = _ex_on_sheet(mesh)
    cx, cz = SPAN / 2, SPAN / 2
    i_c = int(np.argmin(np.abs(x_c - cx)))
    k_c = int(np.argmin(np.abs(z - cz)))
    assert not ex[i_c, k_c], "edge in the bore of the ring was marked PEC (even-odd broken)"
    k_ring = int(np.argmin(np.abs(z - (cz + 0.5 * (R_IN + R_OUT)))))
    assert ex[i_c, k_ring], "edge on the annulus must be PEC"
    # Outside the rim but inside the domain (the z-wall is at SPAN).
    assert not ex[i_c, int(np.argmin(np.abs(z - (cz + R_OUT + 0.25e-3))))]

    classified = _classifier_mask(model, monkeypatch)
    section = mesh.pec_mask_edges
    n_sheet = int(classified.sum())
    n_diff = int((section != classified).sum())
    # Chords of the tessellated circle against the exact solid: only a
    # few edges along the rim may differ, never the bulk.
    assert n_diff <= 0.03 * n_sheet, (n_diff, n_sheet)


def test_strip_boundary_edges_are_metal_inclusive():
    pec = Material.pec()
    air = Material.from_isotropic(name="air", epsilon=1.0)
    sub = Material.from_isotropic(name="sub", epsilon=3.0)
    z0, z1 = 0.9e-3, 1.7e-3
    strip = Brick(origin=(0.0, H_SUB, z0), size=(SPAN, T_MET, z1 - z0), material=pec)
    model = GeometryModel(
        boundary_conditions={f: "PEC" for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    )
    model.add(Brick(origin=(0.0, 0.0, 0.0), size=(SPAN, H_SUB, SPAN), material=sub))
    model.add(
        Difference(Brick(origin=(0.0, H_SUB, 0.0), size=(SPAN, 2.0e-3, SPAN), material=air), strip)
    )
    model.add(strip)
    ex, x_c, z = _ex_on_sheet(_mesh(model))
    i = int(np.argmin(np.abs(x_c - SPAN / 2)))
    k0 = int(np.argmin(np.abs(z - z0)))
    k1 = int(np.argmin(np.abs(z - z1)))
    assert abs(z[k0] - z0) < 1e-9 and abs(z[k1] - z1) < 1e-9, "strip edges must be grid planes"
    assert ex[i, k0] and ex[i, k1], "x-edges on the strip's lateral boundary lines are metal"
    assert ex[i, (k0 + k1) // 2]
    assert not ex[i, k0 - 1] and not ex[i, k1 + 1], "one node outside the strip is open"
