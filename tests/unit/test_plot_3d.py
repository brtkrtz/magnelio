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
    def test_cut_sheet(self, coax):
        model, mesh = coax
        pl = model.plot(mesh=mesh, mode="none", cut=("y", 0.0))
        # No grid "cage" on the domain faces: the grid shows on the cut only.
        assert "grid_faces" not in pl.renderer.actors
        nx, nz = mesh.Nx, mesh.Nz
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

    def test_no_cut_or_show_grid_false_hides_sheet(self, coax):
        model, mesh = coax
        pl = model.plot(mesh=mesh, mode="none")
        assert not pl.renderer.actors["grid_cut"].GetVisibility()
        pl = model.plot(mesh=mesh, mode="none", cut=("y", 0.0), show_grid=False)
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

    def test_labels_are_polydata_text(self, features_model, coax):
        pl = features_model.plot(mode="none")
        labels = [n for n in pl.renderer.actors if n.startswith("label_")]
        assert len(labels) == 2  # feed, load
        for n in labels:
            assert isinstance(_dataset(pl, n), pv.PolyData)
            assert _dataset(pl, n).n_cells > 0
        pl = features_model.plot(mode="none", show_labels=False)
        assert not [n for n in pl.renderer.actors if n.startswith("label_")]
        # Face ports carry their name on the window, lying in its plane.
        model, _ = coax
        pl = model.plot(mode="none")
        labels = [n for n in pl.renderer.actors if n.startswith("label_")]
        assert len(labels) == 2
        for n in labels:
            b = _dataset(pl, n).bounds
            assert b[5] - b[4] < 0.05 * (b[1] - b[0])  # flat in z: the zmin/zmax faces
            assert min(abs(b[4]), abs(b[4] - 10.0)) < 0.05

    def test_cut_removes_features_in_the_removed_half(self, features_model):
        # Element at z = 6..7 mm, port at z = 0..1 mm: a cut at z = 4 mm
        # keeping the lower half removes the element and its label.
        pl = features_model.plot(mode="none", cut=("z", 4e-3))
        acts = pl.renderer.actors
        assert acts["port_0"].GetVisibility()
        assert not acts["element_0"].GetVisibility()
        labels = {n: acts[n].GetVisibility() for n in acts if n.startswith("label_")}
        assert sorted(labels.values()) == [0, 1]
        # The wire (z = 1..6 mm) is clipped, not hidden.
        assert acts["wire_0"].GetVisibility()
        assert _dataset(pl, "wire_0").bounds[5] == pytest.approx(4.0, abs=0.05)
        pl = features_model.plot(mode="none", cut=("z", 4e-3), flip=True)
        assert pl.renderer.actors["element_0"].GetVisibility()
        assert not pl.renderer.actors["port_0"].GetVisibility()

    def test_hidden_groups(self, features_model):
        scene = plot_3d._build_scene(
            features_model,
            mesh=None,
            cut=None,
            flip=False,
            show_ports=True,
            show_wires=True,
            show_grid=True,
            show_labels=True,
            size=None,
            render_edges=False,
            edge_color="#202020",
            quality=1.0,
            scale_mm=True,
            camera="iso",
            off_screen=True,
        )
        assert scene.groups_present() == [
            "solids",
            "ports",
            "elements",
            "wires",
            "labels",
            "symmetry",
            "domain",
        ]
        acts = scene.plotter.renderer.actors
        scene.hidden_groups = {"ports", "symmetry", "labels"}
        plot_3d._apply_cut(scene)
        assert not acts["port_0"].GetVisibility()
        assert not acts["symmetry_xmin"].GetVisibility()
        assert not any(acts[n].GetVisibility() for n in acts if n.startswith("label_"))
        assert acts["wire_0"].GetVisibility()
        assert acts["element_0"].GetVisibility()
        scene.hidden_groups = set()
        plot_3d._apply_cut(scene)
        assert acts["port_0"].GetVisibility()

    def test_every_actor_dataset_is_polydata(self, coax):
        # The in-browser renderer serialises polydata only; a
        # rectilinear grid in the scene left the widget blank.
        model, mesh = coax
        pl = model.plot(mesh=mesh, mode="none", cut=("y", 0.0))
        for name, actor in pl.renderer.actors.items():
            mapper = getattr(actor, "mapper", None)
            if mapper is None or getattr(mapper, "dataset", None) is None:
                continue
            assert isinstance(mapper.dataset, pv.PolyData), name

    def test_toggles(self, features_model):
        pl = features_model.plot(mode="none", show_wires=False, show_ports=False)
        names = set(pl.renderer.actors)
        assert not names & {"wire_0", "port_0", "element_0"}
        assert not [n for n in names if n.startswith("label_")]

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


class TestBrowserSerialisation:
    def test_scene_serialises_for_vtkjs(self, coax, caplog):
        """Every actor of a full scene passes trame's vtk.js serialiser.

        The serialiser logs ``!!!No serializer for <class>`` for anything
        it cannot ship to the browser; such an actor left the widget
        blank (a rectilinear-grid sheet did exactly that).
        """
        import logging

        serializers = pytest.importorskip("trame_vtk.modules.vtk.serializers")
        model, mesh = coax
        pl = model.plot(mesh=mesh, mode="none", cut=("y", 0.0), size=(300, 200))
        pl.render()
        serializers.initialize_serializers()
        ctx = serializers.SynchronizationContext()
        with caplog.at_level(logging.WARNING):
            scene = serializers.serialize(
                None, pl.ren_win, serializers.reference_id(pl.ren_win), ctx, 0
            )
        assert scene is not None
        missing = [r.getMessage() for r in caplog.records if "No serializer" in r.getMessage()]
        assert not missing, missing

        # Every visible actor made it into the browser scene.
        def types(node):
            if isinstance(node, dict):
                yield node.get("type")
                for v in node.values():
                    yield from types(v)
            elif isinstance(node, list):
                for v in node:
                    yield from types(v)

        found = [t for t in types(scene) if t]
        serialised_actors = sum(1 for t in found if t.endswith("Actor"))
        visible = [a for a in pl.renderer.actors.values() if a.GetVisibility()]
        assert serialised_actors >= len(visible) - 1  # the axes widget is no vtkActor
        datasets = {t for t in found if t.startswith("vtk") and "Data" in t or "Grid" in t}
        assert datasets == {"vtkPolyData"}, datasets


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
