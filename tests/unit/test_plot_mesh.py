"""plot_mesh_section (DD-200): fills, edge layer, styles, geometry overlay, wrapper."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.collections import LineCollection, QuadMesh  # noqa: E402

pytest.importorskip("OCC.Core.BRepPrimAPI")

R_OUTER = 2.5e-3
R_PIN = 0.5e-3


@pytest.fixture(scope="module")
def coax():
    from magnelio import Material, Mesh, MeshControl
    from magnelio.geo import Cylinder, Difference, GeometryModel

    ptfe = Material.from_isotropic(name="ptfe", epsilon=2.1)
    model = GeometryModel(background="pec", boundary_conditions={"zmax": "CPML"})
    outer = Cylinder(origin=(0, 0, 0), radius=R_OUTER, height=6e-3, axis="z", material=ptfe)
    inner = Cylinder(origin=(0, 0, 0), radius=R_PIN, height=6e-3, axis="z", material="pec")
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


def _legend_labels(ax):
    return [t.get_text() for t in ax.get_legend().get_texts()]


def _node(grid, axis, value):
    nodes = np.asarray(getattr(grid, axis))
    idx = int(np.argmin(np.abs(nodes - value)))
    assert abs(nodes[idx] - value) < 1e-9, f"no node at {axis} = {value}"
    return idx


def _cell(grid, axis, value):
    nodes = np.asarray(getattr(grid, axis))
    return int(np.searchsorted(nodes, value, side="right") - 1)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_dual_bounds():
    from magnelio.post.plot_mesh import _dual_bounds

    np.testing.assert_allclose(_dual_bounds(np.array([0.0, 1.0, 3.0])), [0.0, 0.5, 2.0, 3.0])


def test_edge_and_face_indices_match_the_indexing_module(coax):
    from magnelio.mesh import indexing
    from magnelio.post.plot_mesh import _edge_index, _face_index

    _model, mesh = coax
    g = mesh.grid
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    n_Hx = (Nx + 1) * Ny * Nz
    n_Hy = Nx * (Ny + 1) * Nz
    i, j, k = 3, 4, 5
    local, flat = _edge_index(0, i, j, k, g)
    assert local == flat == indexing.edge_index_Ex(i, j, k, Nx, Ny, Nz)
    local, flat = _edge_index(1, i, j, k, g)
    assert local == indexing.edge_index_Ey(i, j, k, Nx, Ny, Nz) and flat == local + n_Ex
    local, flat = _edge_index(2, i, j, k, g)
    assert local == indexing.edge_index_Ez(i, j, k, Nx, Ny, Nz) and flat == local + n_Ex + n_Ey
    assert _face_index(0, i, j, k, g) == indexing.face_index_Hx(i, j, k, Nx, Ny, Nz)
    assert _face_index(1, i, j, k, g) == n_Hx + indexing.face_index_Hy(i, j, k, Nx, Ny, Nz)
    assert _face_index(2, i, j, k, g) == n_Hx + n_Hy + indexing.face_index_Hz(i, j, k, Nx, Ny, Nz)


# ---------------------------------------------------------------------------
# coverage fill: exact PEC area share per primal cell
# ---------------------------------------------------------------------------


def test_coverage_tiles_are_the_geometric_pec_share(coax):
    from magnelio.post.plot_mesh import _coverage_tiles

    _model, mesh = coax
    g = mesh.grid
    k_n = _node(g, "z", 3e-3)
    k = _cell(g, "z", 3e-3)
    fraction, rgb = _coverage_tiles(mesh, "z", k_n, k)
    assert fraction.shape == (g.Nx, g.Ny) and rgb.shape == (g.Nx, g.Ny, 3)
    # cell [0.25, 0.5] x [0.25, 0.5]: the pin's quarter circle covers ~31 %
    i, j = _cell(g, "x", 0.375e-3), _cell(g, "y", 0.375e-3)
    assert fraction[i, j] == pytest.approx(0.31, abs=0.02)
    # cell [0.25, 0.5] x [0, 0.25]: almost fully inside the pin
    j0 = _cell(g, "y", 0.125e-3)
    assert fraction[i, j0] == pytest.approx(0.91, abs=0.02)
    # cell straddling the outer rim is partly PEC background,
    # a cell well inside the annulus is free, the pin centre is full
    assert 0.0 < fraction[_cell(g, "x", -1.95e-3), _cell(g, "y", 1.58e-3)] < 1.0
    assert fraction[_cell(g, "x", 1.5e-3), _cell(g, "y", 0.0)] == 0.0
    assert fraction[_cell(g, "x", 0.1e-3), _cell(g, "y", 0.1e-3)] == 1.0
    # colour: a free ptfe cell is the ptfe tint, a full cell is PEC grey
    np.testing.assert_allclose(rgb[_cell(g, "x", 0.1e-3), _cell(g, "y", 0.1e-3)], 0.65)
    assert not np.allclose(rgb[_cell(g, "x", 1.5e-3), _cell(g, "y", 0.0)], 0.65)


def test_coverage_is_the_default_fill_and_draws_a_quadmesh(coax):
    from magnelio.plots import plot_mesh_section

    model, mesh = coax
    fig, ax = plot_mesh_section(mesh, "z", 3e-3, geometry=model)
    assert any(isinstance(c, QuadMesh) for c in ax.collections)
    assert len(fig.axes) == 1  # no colour bar
    assert ax.get_title().startswith("Mesh section at z = 3.00 mm (node plane 3.00 mm)")


# ---------------------------------------------------------------------------
# conformal fill: eps_avg of the dual faces of the normal edges
# ---------------------------------------------------------------------------


def test_conformal_tiles_follow_the_classifier(coax):
    from magnelio.post.plot_mesh import _conformal_tiles, _edge_index, _ijk

    _model, mesh = coax
    g = mesh.grid
    k = _cell(g, "z", 3e-3)
    tiles, eps_max = _conformal_tiles(mesh, "z", k)
    assert tiles.shape == (g.Nx + 1, g.Ny + 1)
    assert eps_max == pytest.approx(2.1)
    assert tiles.min() == 0.0 and tiles.max() == pytest.approx(2.1)
    em = mesh.edge_material
    i, j, kk = _ijk("z", np.arange(g.Nx + 1), np.arange(g.Ny + 1), k)
    local, flat = _edge_index(2, i, j, kk, g)
    cat = em.category[flat]
    # every masked or interior-PEC edge is a PEC tile
    assert np.all(tiles[(cat == 3) | mesh.pec_mask_edges[2][local]] == 0.0)
    # every dielectric-boundary / curved-PEC edge above the floor carries eps_avg
    averaged = ((cat == 1) | (cat == 2)) & ~mesh.pec_mask_edges[2][local] & (em.f_A[flat] > 0.01)
    assert averaged.any()
    np.testing.assert_allclose(tiles[averaged], em.eps_avg[flat][averaged])
    # a bulk edge deep in the ptfe carries the staircase value, one in the pin is PEC
    ix = int(np.argmin(np.abs(g.x - 1.4e-3)))  # a graded node inside the ptfe
    assert tiles[ix, _node(g, "y", 0.0)] == pytest.approx(2.1)
    assert tiles[_node(g, "x", 0.0), _node(g, "y", 0.0)] == 0.0


def test_conformal_fill_adds_a_colour_bar(coax):
    from magnelio.plots import plot_mesh_section

    _model, mesh = coax
    fig, ax = plot_mesh_section(mesh, "z", 3e-3, fill="conformal")
    assert any(isinstance(c, QuadMesh) for c in ax.collections)
    assert len(fig.axes) == 2
    assert "PEC" in fig.axes[1].get_ylabel()


def test_eps_colormap_endpoints():
    from magnelio.post.plot_mesh import _eps_colormap

    cmap, norm = _eps_colormap(2.1)
    assert norm.vmin == 0.0 and norm.vmax == pytest.approx(2.1)
    np.testing.assert_allclose(cmap(norm(0.0))[:3], 0.65)  # PEC grey
    np.testing.assert_allclose(cmap(norm(1.0))[:3], 1.0, atol=0.02)  # air white
    assert cmap(norm(2.1))[2] > cmap(norm(2.1))[0]  # dielectric tint is bluish
    cmap_air, norm_air = _eps_colormap(1.0)
    assert norm_air.vmax == 1.0


# ---------------------------------------------------------------------------
# material fill and lines only
# ---------------------------------------------------------------------------


def test_material_fill_and_legend(coax):
    from magnelio.plots import plot_mesh_section

    model, mesh = coax
    fig, ax = plot_mesh_section(mesh, "z", 3e-3, fill="material")
    assert any(isinstance(c, QuadMesh) for c in ax.collections)
    n_planes = len(mesh.planes.x) + len(mesh.planes.y)
    assert len(_line_collections(ax)) >= n_planes
    labels = _legend_labels(ax)
    assert "material face" in labels and "forced plane" in labels and "graded fill" in labels
    assert ax.get_xlabel() == "x [mm]" and ax.get_ylabel() == "y [mm]"
    assert ax.get_title() == "Mesh section at z = 3.00 mm"


def test_fill_none_draws_lines_only(coax):
    from magnelio.plots import plot_mesh_section

    _model, mesh = coax
    _fig, ax = plot_mesh_section(mesh, "z", 3e-3, fill=None, legend=False)
    assert not any(isinstance(c, QuadMesh) for c in ax.collections)
    assert ax.get_legend() is None
    assert _line_collections(ax)


# ---------------------------------------------------------------------------
# edge layer
# ---------------------------------------------------------------------------


def test_section_edges_classes(coax):
    from magnelio.post.plot_mesh import _section_edges

    _model, mesh = coax
    g = mesh.grid
    k_n = _node(g, "z", 3e-3)
    classes = _section_edges(mesh, "z", k_n)
    masked, partial, f_L, borrowed = (
        classes["masked"],
        classes["partial"],
        classes["f_L"],
        classes["borrowed"],
    )
    assert masked.shape[1:] == (2, 2) and partial.shape[1:] == (2, 2)
    assert len(masked) > 0 and len(partial) > 0 and len(borrowed) > 0
    assert np.all((f_L > 0.0) & (f_L < 1.0))
    # every segment is axis-aligned and spans exactly one cell
    for seg in np.concatenate([masked, partial]):
        d = seg[1] - seg[0]
        assert (d[0] == 0.0) != (d[1] == 0.0)
    # masked edges include the domain wall (PEC background) at x = -2.5 mm
    assert np.any(np.all(np.isclose(masked[:, :, 0], -R_OUTER), axis=1))

    # the pin surface: the edge from (0.25, 0) to (0.5, 0) mm ends on the pin
    # and is masked, its continuation (0.5 -> 0.75) is free
    def has(seg_set, a, b):
        return any(
            np.allclose(s[0], a, atol=1e-12) and np.allclose(s[1], b, atol=1e-12) for s in seg_set
        )

    assert has(masked, (0.25e-3, 0.0), (0.5e-3, 0.0))
    assert not has(masked, (0.5e-3, 0.0), (0.75e-3, 0.0))
    assert not has(partial, (0.5e-3, 0.0), (0.75e-3, 0.0))
    # borrowed edges are masked short edges: their midpoints lie on masked segments
    mids = masked.mean(axis=1)
    for b in borrowed:
        assert np.any(np.all(np.isclose(mids, b, atol=1e-12), axis=1))


def test_edge_layer_draws_and_labels(coax):
    from magnelio.plots import plot_mesh_section

    model, mesh = coax
    _fig, ax_plain = plot_mesh_section(mesh, "z", 3e-3, fill=None, legend=False)
    _fig, ax = plot_mesh_section(mesh, "z", 3e-3, fill=None, edges=True, geometry=model)
    assert len(_line_collections(ax)) >= len(_line_collections(ax_plain)) + 2
    labels = _legend_labels(ax)
    assert "PEC-masked edge" in labels
    assert "partially in PEC (0 < f_L < 1)" in labels
    assert "borrowed edge (enlarged cell)" in labels
    assert "(node plane 3.00 mm)" in ax.get_title()


def test_edge_layer_uses_the_nearest_node_plane(coax):
    from magnelio.plots import plot_mesh_section

    _model, mesh = coax
    g = mesh.grid
    z_between = 0.5 * (g.z[6] + g.z[7]) + 0.1 * (g.z[7] - g.z[6])
    _fig, ax = plot_mesh_section(mesh, "z", z_between, fill=None, edges=True, legend=False)
    assert f"(node plane {g.z[7] * 1e3:.2f} mm)" in ax.get_title()


# ---------------------------------------------------------------------------
# overlay, flip, absorber, errors, wrapper
# ---------------------------------------------------------------------------


def test_absorber_cells_are_shaded_on_the_cut_showing_them(coax):
    from magnelio.plots import plot_mesh_section

    _model, mesh = coax
    _fig, ax = plot_mesh_section(mesh, "x", 0.0)
    # zmax carries CPML: a hatched span along the vertical (z) axis
    spans = [p for p in ax.patches if p.get_hatch()]
    assert len(spans) == 1
    assert "absorber cells" in _legend_labels(ax)


def test_geometry_overlay_is_outline_only(coax):
    from magnelio.plots import plot_mesh_section

    model, mesh = coax
    _fig, ax = plot_mesh_section(mesh, "z", 3e-3, geometry=model)
    outlines = [p for p in ax.patches if not p.get_hatch()]
    assert outlines
    assert all(p.get_facecolor()[3] == 0.0 for p in outlines)


@pytest.mark.parametrize("fill", ["coverage", "conformal", "material", None])
def test_flip_swaps_axes(coax, fill):
    from magnelio.plots import plot_mesh_section

    _model, mesh = coax
    _fig, ax = plot_mesh_section(mesh, "x", 0.0, flip=True, fill=fill, edges=True, legend=False)
    assert ax.get_xlabel() == "z [mm]" and ax.get_ylabel() == "y [mm]"
    assert ax.get_xlim() == pytest.approx(
        (float(mesh.grid.z[0]) * 1e3, float(mesh.grid.z[-1]) * 1e3)
    )
    if fill is not None:
        qm = [c for c in ax.collections if isinstance(c, QuadMesh)][0]
        # the quadmesh spans the flipped extent: z horizontally, y vertically
        coords = qm.get_coordinates()
        assert coords[..., 0].max() == pytest.approx(float(mesh.grid.z[-1]) * 1e3)
        assert coords[..., 1].max() == pytest.approx(float(mesh.grid.y[-1]) * 1e3)


def test_mesh_without_subcell_data(coax):
    from magnelio.mesh.grid import GridLines
    from magnelio.mesh.mesher import Mesh
    from magnelio.plots import plot_mesh_section

    grid = GridLines(
        x=np.linspace(0, 1e-3, 4), y=np.linspace(0, 1e-3, 5), z=np.linspace(0, 1e-3, 3)
    )
    mesh = Mesh.from_grid(grid, [], background="air")
    _fig, ax = plot_mesh_section(mesh, "z", 0.5e-3, fill="material")
    assert _legend_labels(ax) == ["graded fill"]
    for kwargs in ({"fill": "coverage"}, {"fill": "conformal"}, {"fill": None, "edges": True}):
        with pytest.raises(ValueError, match="no sub-cell data"):
            plot_mesh_section(mesh, "z", 0.5e-3, **kwargs)


def test_mesh_plot_section_wrapper_and_argument_checks(coax):
    from magnelio.plots import plot_mesh_section

    _model, mesh = coax
    fig, ax = mesh.plot_section("y", 0.0, legend=False)
    assert fig is ax.figure
    with pytest.raises(ValueError, match="normal must be"):
        plot_mesh_section(mesh, "w", 0.0)
    with pytest.raises(ValueError, match="fill must be"):
        plot_mesh_section(mesh, "z", 0.0, fill="eps")
