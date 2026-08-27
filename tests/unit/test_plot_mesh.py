"""plot_mesh_section (DD-200): layers, styles, geometry overlay, wrapper."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.collections import LineCollection, QuadMesh  # noqa: E402

pytest.importorskip("OCC.Core.BRepPrimAPI")


@pytest.fixture(scope="module")
def coax():
    from magnelio import Material, Mesh, MeshControl
    from magnelio.geo import Cylinder, Difference, GeometryModel

    ptfe = Material.from_isotropic(name="ptfe", epsilon=2.1)
    model = GeometryModel(background="pec", boundary_conditions={"zmax": "CPML"})
    outer = Cylinder(origin=(0, 0, 0), radius=2.5e-3, height=6e-3, axis="z", material=ptfe)
    inner = Cylinder(origin=(0, 0, 0), radius=0.5e-3, height=6e-3, axis="z", material="pec")
    model.add(Difference(outer, inner))
    model.add(inner)
    mesh = Mesh.from_geometry(
        model, MeshControl(max_cell_size=0.5e-3, forced_planes={"x": [0.0]}), f_max=10e9
    )
    return model, mesh


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _line_collections(ax):
    return [c for c in ax.collections if isinstance(c, LineCollection)]


def test_layers_and_legend(coax):
    from magnelio.plots import plot_mesh_section

    model, mesh = coax
    fig, ax = plot_mesh_section(mesh, "z", 3e-3)
    assert any(isinstance(c, QuadMesh) for c in ax.collections)
    # one line collection per graded batch and per plane on each axis
    n_planes = len(mesh.planes.x) + len(mesh.planes.y)
    assert len(_line_collections(ax)) >= n_planes
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "material face" in labels and "forced plane" in labels and "graded fill" in labels
    assert ax.get_xlabel() == "x [mm]" and ax.get_ylabel() == "y [mm]"
    assert ax.get_title().startswith("Mesh section at z = 3.00 mm")


def test_fill_none_draws_lines_only(coax):
    from magnelio.plots import plot_mesh_section

    _model, mesh = coax
    _fig, ax = plot_mesh_section(mesh, "z", 3e-3, fill=None, legend=False)
    assert not any(isinstance(c, QuadMesh) for c in ax.collections)
    assert ax.get_legend() is None
    assert _line_collections(ax)


def test_absorber_cells_are_shaded_on_the_cut_showing_them(coax):
    from magnelio.plots import plot_mesh_section

    _model, mesh = coax
    _fig, ax = plot_mesh_section(mesh, "x", 0.0)
    # zmax carries CPML: a hatched span along the vertical (z) axis
    spans = [p for p in ax.patches if p.get_hatch()]
    assert len(spans) == 1
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "absorber cells" in labels


def test_geometry_overlay_is_outline_only(coax):
    from magnelio.plots import plot_mesh_section

    model, mesh = coax
    _fig, ax = plot_mesh_section(mesh, "z", 3e-3, geometry=model)
    outlines = [p for p in ax.patches if not p.get_hatch()]
    assert outlines
    assert all(p.get_facecolor()[3] == 0.0 for p in outlines)


def test_flip_swaps_axes(coax):
    from magnelio.plots import plot_mesh_section

    _model, mesh = coax
    _fig, ax = plot_mesh_section(mesh, "x", 0.0, flip=True, fill=None)
    assert ax.get_xlabel() == "z [mm]" and ax.get_ylabel() == "y [mm]"
    assert ax.get_xlim() == pytest.approx(
        (float(mesh.grid.z[0]) * 1e3, float(mesh.grid.z[-1]) * 1e3)
    )


def test_mesh_without_provenance_draws_graded_lines(coax):
    from magnelio.mesh.grid import GridLines
    from magnelio.mesh.mesher import Mesh
    from magnelio.plots import plot_mesh_section

    grid = GridLines(
        x=np.linspace(0, 1e-3, 4), y=np.linspace(0, 1e-3, 5), z=np.linspace(0, 1e-3, 3)
    )
    mesh = Mesh.from_grid(grid, [], background="air")
    _fig, ax = plot_mesh_section(mesh, "z", 0.5e-3)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["graded fill"]


def test_mesh_plot_section_wrapper_and_argument_checks(coax):
    from magnelio.plots import plot_mesh_section

    _model, mesh = coax
    fig, ax = mesh.plot_section("y", 0.0, legend=False)
    assert fig is ax.figure
    with pytest.raises(ValueError, match="normal must be"):
        plot_mesh_section(mesh, "w", 0.0)
    with pytest.raises(ValueError, match="fill must be"):
        plot_mesh_section(mesh, "z", 0.0, fill="eps")
