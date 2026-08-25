"""Unit tests for the 3D geometry/mesh viewer (``magnelio.post.plot_3d``).

The scenes are built off-screen (``mode="none"``) and inspected through
the plotter's actors; one screenshot smoke test exercises the render
window.  The trame widget itself needs a kernel and a browser and is
covered by the notebook spike, not here.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
pytest.importorskip("OCC")

import magnelio as mio  # noqa: E402
from magnelio import circuit, geo, plots, ports  # noqa: E402
from magnelio.post import plot_3d  # noqa: E402

pv.OFF_SCREEN = True


@pytest.fixture(scope="module")
def coax():
    pe = mio.Material.from_isotropic(name="polyethylene", epsilon=2.25)
    outer = geo.Cylinder(origin=(0, 0, 0), radius=1.75e-3, height=10e-3, axis="z", material=pe)
    inner = geo.Cylinder(origin=(0, 0, 0), radius=0.5e-3, height=10e-3, axis="z", material="pec")
    model = mio.GeometryModel(background="pec")
    model.add(geo.Difference(outer, inner))
    model.add(inner)
    model.add_port(ports.PortWaveguide(name="p1", plane="zmin", n_modes=1))
    model.add_port(ports.PortWaveguide(name="p2", plane="zmax", n_modes=1))
    mesh = mio.Mesh.from_geometry(model, mio.MeshControl(max_cell_size=0.25e-3), f_max=20e9)
    return model, mesh


@pytest.fixture(scope="module")
def features_model():
    model = mio.GeometryModel(
        boundary_conditions=mio.BoundaryConditions(xmin="SymmetryPMC", zmax="CPML"),
    )
    model.add(geo.Brick(origin=(0, -5e-3, 0), size=(5e-3, 10e-3, 8e-3), material="air"))
    model.add(geo.Brick(origin=(0, -1e-3, 0), size=(2e-3, 2e-3, 1e-3), material="pec"))
    model.add(
        geo.ThinWire(
            geo.Curve.polyline([(1e-3, 0, 1e-3), (1e-3, 0, 6e-3)]),
            radius=0.05e-3,
            name="wire",
        )
    )
    model.add_port(ports.PortLumped(name="feed", start=(1e-3, 0, 0), end=(1e-3, 0, 1e-3), Z0=50.0))
    model.add_element(
        circuit.LumpedElement(
            name="load",
            start=(1e-3, 0, 6e-3),
            end=(1e-3, 0, 7e-3),
            element=circuit.SeriesRLC(R=100.0),
        )
    )
    return model


def _dataset(pl, name):
    return pl.renderer.actors[name].mapper.dataset


class TestScene:
    def test_returns_plotter_and_draws_bodies_in_mm(self, coax):
        model, _ = coax
        pl = model.plot(mode="none")
        assert isinstance(pl, pv.Plotter)
        names = set(pl.renderer.actors)
        assert {"shape_0", "shape_1", "domain", "port_0", "port_1"} <= names
        # Display units are millimetres: the 10 mm line spans z = 0..10.
        zlo, zhi = _dataset(pl, "shape_1").bounds[4:6]
        assert zlo == pytest.approx(0.0, abs=1e-6)
        assert zhi == pytest.approx(10.0, abs=1e-6)

    def test_scale_mm_false_keeps_metres(self, coax):
        model, _ = coax
        pl = model.plot(mode="none", scale_mm=False)
        assert _dataset(pl, "shape_1").bounds[5] == pytest.approx(10e-3, abs=1e-9)

    def test_bodies_are_watertight(self, coax):
        model, _ = coax
        pl = model.plot(mode="none")
        for name in ("shape_0", "shape_1"):
            assert _dataset(pl, name).n_open_edges == 0, name

    def test_cut_clips_every_body_and_caps(self, coax):
        model, _ = coax
        pl = model.plot(mode="none", cut=("y", 0.0))
        for name in ("shape_0", "shape_1"):
            ds = _dataset(pl, name)
            ylo, yhi = ds.bounds[2:4]
            assert yhi == pytest.approx(0.0, abs=1e-6), name
            assert ylo < -0.4, name
            # Capped: the clipped surface is still closed.
            assert ds.n_open_edges == 0, name

    def test_flip_keeps_the_other_half(self, coax):
        model, _ = coax
        pl = model.plot(mode="none", cut=("y", 0.0), flip=True)
        ylo, yhi = _dataset(pl, "shape_0").bounds[2:4]
        assert ylo == pytest.approx(0.0, abs=1e-6)
        assert yhi > 0.4

    def test_cut_outside_body_hides_or_keeps_it_whole(self, coax):
        model, _ = coax
        pl = model.plot(mode="none", cut=("z", 20e-3))
        # The plane is beyond the line: nothing removed on the kept side.
        assert pl.renderer.actors["shape_1"].GetVisibility()
        assert _dataset(pl, "shape_1").bounds[5] == pytest.approx(10.0, abs=1e-6)
        pl = model.plot(mode="none", cut=("z", -1e-3))
        assert not pl.renderer.actors["shape_1"].GetVisibility()

    def test_bad_cut_axis_rejected(self, coax):
        model, _ = coax
        with pytest.raises(ValueError, match="cut normal"):
            model.plot(mode="none", cut=("w", 0.0))

    def test_bad_mode_rejected(self, coax):
        model, _ = coax
        with pytest.raises(ValueError, match="mode must be one of"):
            model.plot(mode="opengl")


class TestGrid:
    def test_grid_faces_and_cut_sheet(self, coax):
        model, mesh = coax
        pl = model.plot(mesh=mesh, mode="none", cut=("y", 0.0))
        faces = _dataset(pl, "grid_faces")
        # Grid lines on the domain faces: exactly the surface cells.
        nx, ny, nz = mesh.Nx, mesh.Ny, mesh.Nz
        assert faces.n_cells == 2 * (nx * ny + ny * nz + nx * nz)
        sheet = _dataset(pl, "grid_cut")
        assert pl.renderer.actors["grid_cut"].GetVisibility()
        assert sheet.n_cells == nx * nz
        # The sheet sits a hair into the removed (+y) half.
        ylo, yhi = sheet.bounds[2:4]
        assert 0.0 < ylo == yhi < 1e-3
        assert sheet.cell_data["color"].dtype == np.uint8

    def test_sheet_keeps_direct_colours_after_cut_update(self, coax):
        # The sheet dataset is swapped on every cut change; the mapper
        # must keep colouring by the RGB array, not by material_id
        # through a lookup table (that rendered every cell dark grey).
        model, mesh = coax
        pl = model.plot(mesh=mesh, mode="none", cut=("y", 0.0))
        mapper = pl.renderer.actors["grid_cut"].mapper
        assert mapper.GetArrayName() == "color"
        assert mapper.GetScalarModeAsString() == "UseCellFieldData"
        # VTK's default mode already takes unsigned-char arrays as colours.
        assert mapper.GetColorModeAsString() in ("Default", "DirectScalars")

    def test_sheet_carries_the_exposed_layer(self, coax):
        model, mesh = coax
        g = plot_3d._grid_dataset(mesh, unit_scale=1e3)
        y = np.asarray(mesh.grid.y) * 1e3
        k = 3
        pos = 0.5 * (y[k] + y[k + 1])
        sheet = plot_3d._grid_slab(g, plot_3d._CutState("y", pos, flip=False))
        expected = np.asarray(mesh.material_id)[:, k, :].ravel(order="F")
        np.testing.assert_array_equal(sheet.cell_data["material_id"], expected)
        # Flipped: the layer on the +y side of the plane.
        sheet = plot_3d._grid_slab(g, plot_3d._CutState("y", y[k], flip=True))
        np.testing.assert_array_equal(sheet.cell_data["material_id"], expected)

    def test_no_cut_hides_sheet_and_show_grid_false_hides_faces(self, coax):
        model, mesh = coax
        pl = model.plot(mesh=mesh, mode="none", show_grid=False)
        assert not pl.renderer.actors["grid_faces"].GetVisibility()
        assert not pl.renderer.actors["grid_cut"].GetVisibility()

    def test_material_colours_follow_the_2d_palette(self, coax):
        _, mesh = coax
        from magnelio.post._colors import material_color

        g = plot_3d._grid_dataset(mesh, unit_scale=1.0)
        ids = np.asarray(g.cell_data["material_id"])
        colors = np.asarray(g.cell_data["color"])
        for mid in np.unique(ids):
            rgba = material_color(mesh.material_library[int(mid)])
            expected = np.round(np.asarray(rgba[:3]) * 255).astype(np.uint8)
            np.testing.assert_array_equal(colors[ids == mid][0], expected)


class TestOverlays:
    def test_features_are_drawn(self, features_model):
        pl = features_model.plot(mode="none")
        names = set(pl.renderer.actors)
        assert {"wire_0", "port_0", "element_0", "symmetry_xmin", "domain"} <= names
        assert "symmetry_xmax" not in names
        # The wire tube follows the declared polyline (z = 1..6 mm).
        zlo, zhi = _dataset(pl, "wire_0").bounds[4:6]
        assert zlo == pytest.approx(1.0, abs=0.1)
        assert zhi == pytest.approx(6.0, abs=0.1)

    def test_toggles(self, features_model):
        pl = features_model.plot(mode="none", show_wires=False, show_ports=False)
        names = set(pl.renderer.actors)
        assert not names & {"wire_0", "port_0", "element_0"}

    def test_face_port_window(self, coax):
        model, _ = coax
        pl = model.plot(mode="none")
        quad = _dataset(pl, "port_0")
        assert quad.n_cells == 1
        assert quad.bounds[4] == pytest.approx(0.0, abs=1e-6)  # zmin face
        assert quad.bounds[5] == pytest.approx(0.0, abs=1e-6)

    def test_empty_boolean_is_skipped_with_warning(self):
        a = geo.Brick(origin=(0, 0, 0), size=(1e-3, 1e-3, 1e-3), material="pec")
        b = geo.Brick(origin=(0, 0, 0), size=(1e-3, 1e-3, 1e-3), material="pec")
        model = mio.GeometryModel()
        model.add(geo.Difference(a, b))
        model.add(geo.Brick(origin=(2e-3, 0, 0), size=(1e-3, 1e-3, 1e-3), material="pec"))
        with pytest.warns(UserWarning, match="no volume"):
            pl = model.plot(mode="none")
        assert "shape_1" in pl.renderer.actors
        assert "shape_0" not in pl.renderer.actors


class TestEntryPoints:
    def test_plots_namespace_and_legacy_kwargs(self, coax):
        model, _ = coax
        pl = plots.show_geometry(
            model,
            mode="none",
            size=(320, 240),
            render_edges=True,
            edge_color="#ff0000",
            quality=2.0,
        )
        assert tuple(pl.window_size) == (320, 240)
        assert pl.renderer.actors["shape_1"].prop.show_edges

    def test_screenshot_renders(self, coax, tmp_path):
        model, mesh = coax
        pl = model.plot(mesh=mesh, mode="none", cut=("y", 0.0), size=(300, 200))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = pl.screenshot(tmp_path / "coax.png", return_img=True)
        assert img.shape[:2] == (200, 300)
        # Something was drawn: not a blank canvas.
        assert img.std() > 5.0
