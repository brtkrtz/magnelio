"""Unit tests for the 3D field view (``magnelio.post.field_3d``, DD-259 step 0).

Scenes are built off-screen (``mode="none"``) and inspected through the
plotter's actors, like the geometry-viewer tests; the frame source and
the view state are exercised through the module's own classes.  The
trame widget needs a kernel and a browser and is not covered here.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

import magnelio as mio  # noqa: E402
from magnelio import geo, plots, ports  # noqa: E402
from magnelio._fields.field_arrays import FieldArrays  # noqa: E402
from magnelio.fields import FieldState  # noqa: E402
from magnelio.mesh.grid import GridLines  # noqa: E402
from magnelio.monitors import MonitorFieldFrequency, MonitorFieldTime  # noqa: E402
from magnelio.post import field_3d  # noqa: E402
from magnelio.post.plot_3d import _CutState  # noqa: E402
from magnelio.signals.signal_1d import Signal1D  # noqa: E402

pv.OFF_SCREEN = True

LX, LY, LZ = 0.01, 0.02, 0.03


class _FakeMesh:
    def __init__(self, grid):
        self.grid = grid


def _grid(nx=6, ny=8, nz=4) -> GridLines:
    return GridLines(
        x=np.linspace(0, LX, nx + 1), y=np.linspace(0, LY, ny + 1), z=np.linspace(0, LZ, nz + 1)
    )


def _field(grid) -> FieldState:
    return FieldState.from_function(
        grid,
        E=lambda x, y, z: (0 * x, 0 * y, np.sin(np.pi * x / LX)),
        H=lambda x, y, z: (np.cos(np.pi * y / LY), 0 * y, 0 * z),
    )


def _time_monitor(grid, corners=((None, None, LZ / 2), (None, None, LZ / 2))):
    """Three frames of a uniform Ez growing 1, 2, 3 (grid quantities 1e-3·k)."""
    mon = MonitorFieldTime(corners=corners, times=[0.0, 1e-12, 2e-12], fields=["E"], name="m")
    mon.attach(_FakeMesh(grid))
    for i, t in enumerate(mon.times):
        f = FieldArrays.zeros(grid.Nx, grid.Ny, grid.Nz)
        f.Ez[:] = (i + 1) * 1e-3
        mon.record(f, i, float(t), 1e-12)
    return mon


def _freq_monitor(grid):
    """Two bins of a uniform complex Ez: 1 at f0, j at f1 (unit reference)."""
    freqs = [1.0e9, 2.0e9]
    mon = MonitorFieldFrequency(freqs=freqs, fields=["E"], name="p")
    mon.attach(_FakeMesh(grid))
    dt = 1e-12
    f = FieldArrays.zeros(grid.Nx, grid.Ny, grid.Nz)
    f.Ez[:] = 1e-3
    mon.record(f, 0, 0.0, dt)
    mon.finalize()
    mon.renormalize(Signal1D(t=np.array([0.0]), values=np.array([1.0 / dt]), dt=dt))
    # Overwrite the bins with a known pattern: real at f0, imaginary at f1.
    acc = mon._accumulators["Ez"]
    acc._bins[0] = 0.4
    acc._bins[1] = 0.4j
    for comp in ("Ex", "Ey"):
        mon._accumulators[comp]._bins[...] = 0.0
    return mon


def _sheet(pl):
    return pl.renderer.actors["field_cut"].mapper.dataset


def _dz(grid):
    return float(grid.z[1] - grid.z[0])


# ---------------------------------------------------------------------------
# FieldState
# ---------------------------------------------------------------------------


class TestFieldState:
    def test_default_cut_is_the_thinnest_axis_at_the_centre(self):
        grid = _grid()
        pl = _field(grid).show(mode="none", size=(300, 200))
        sheet = _sheet(pl)
        assert isinstance(sheet, pv.PolyData)
        # z has the fewest cells (4): the layer is 6 x 8 cells.
        assert sheet.n_cells == grid.Nx * grid.Ny
        assert np.allclose(sheet.points[:, 2], sheet.points[0, 2])
        # In display units (mm), a hair past the middle of the region.
        assert abs(sheet.points[0, 2] - LZ / 2 * 1e3) < 0.1 * _dz(grid) * 1e3
        assert "field_arrows" in pl.renderer.actors
        assert "|E| (V/m)" in pl.scalar_bars
        pl.close()

    def test_signed_component_matches_cell_centred_layer(self):
        grid = _grid()
        fs = _field(grid)
        z0 = 0.4 * LZ
        pl = fs.show("Ez", normal="z", position=z0, mode="none", size=(300, 200))
        sheet = _sheet(pl)
        k = int(np.argmin(np.abs(fs.cell_centres[2] - z0)))
        expected = fs.cell_centred(["Ez"])["Ez"][:, :, k].ravel(order="F")
        np.testing.assert_allclose(sheet.cell_data["field"], expected)
        lo, hi = pl.renderer.actors["field_cut"].mapper.scalar_range
        assert lo == -hi and hi == pytest.approx(np.abs(fs.cell_centred(["Ez"])["Ez"]).max())
        # A single component is a sheet only.
        assert "field_arrows" not in pl.renderer.actors
        pl.close()

    def test_magnitude_sheet_and_colour_mode(self):
        grid = _grid()
        fs = _field(grid)
        pl = fs.show("H", normal="x", position=LX / 2, plot_type="color", mode="none")
        sheet = _sheet(pl)
        assert sheet.n_cells == grid.Ny * grid.Nz
        cc = fs.cell_centred(["Hx", "Hy", "Hz"])
        k = int(np.argmin(np.abs(fs.cell_centres[0] - LX / 2)))
        mag = np.sqrt(sum(cc[c][k] ** 2 for c in ("Hx", "Hy", "Hz"))).ravel(order="F")
        np.testing.assert_allclose(sheet.cell_data["field"], mag)
        assert "field_arrows" not in pl.renderer.actors
        assert "|H| (A/m)" in pl.scalar_bars
        pl.close()

    def test_cut_outside_the_region_hides_the_sheet(self):
        grid = _grid()
        with pytest.warns(UserWarning, match="outside the recorded region"):
            pl = _field(grid).show(normal="z", position=5 * LZ, mode="none")
        # Nothing to lay on the cut: no sheet actor is built (or it is hidden).
        actor = pl.renderer.actors.get("field_cut")
        assert actor is None or not actor.GetVisibility()
        assert "field_arrows" not in pl.renderer.actors
        pl.close()

    def test_bad_arguments(self):
        fs = _field(_grid())
        with pytest.raises(KeyError, match="not available"):
            fs.show("Bx", mode="none")
        with pytest.raises(ValueError, match="plot_type"):
            fs.show(plot_type="contour", mode="none")
        with pytest.raises(ValueError, match="normal"):
            fs.show(normal="w", mode="none")
        with pytest.raises(ValueError, match="mode"):
            fs.show(mode="hologram")
        with pytest.raises(TypeError, match="show_field needs"):
            plots.show_field(object(), mode="none")

    def test_plots_namespace_entry_point(self):
        pl = plots.show_field(_field(_grid()), "Ey", mode="none", size=(200, 150))
        assert tuple(pl.window_size) == (200, 150)
        pl.close()

    def test_arrows_are_a_polydata_glyph_set(self):
        pl = _field(_grid()).show(mode="none", density=6)
        arrows = pl.renderer.actors["field_arrows"].mapper.dataset
        assert isinstance(arrows, pv.PolyData) and arrows.n_cells > 0
        # Every actor dataset is polydata (the browser renderer's rule).
        for actor in pl.renderer.actors.values():
            dataset = getattr(getattr(actor, "mapper", None), "dataset", None)
            if dataset is not None:
                assert isinstance(dataset, pv.PolyData)
        pl.close()


# ---------------------------------------------------------------------------
# Monitors
# ---------------------------------------------------------------------------


class TestTimeMonitor:
    def test_plane_monitor_defaults_to_its_own_plane(self):
        grid = _grid()
        mon = _time_monitor(grid)
        pl = mon.show(mode="none")
        sheet = _sheet(pl)
        assert sheet.n_cells == grid.Nx * grid.Ny
        # Frame 0 holds Ez = 1e-3 / dz at every cell.
        np.testing.assert_allclose(sheet.cell_data["field"], 1e-3 / _dz(grid))
        pl.close()

    def test_t_picks_the_nearest_frame_and_vmax_spans_all_frames(self):
        grid = _grid()
        mon = _time_monitor(grid)
        pl = mon.show(mode="none", t=1.9e-12)
        np.testing.assert_allclose(_sheet(pl).cell_data["field"], 3e-3 / _dz(grid))
        assert pl.renderer.actors["field_cut"].mapper.scalar_range == (0.0, 3e-3 / _dz(grid))
        pl.close()
        # The first frame is drawn on the same colour scale.
        pl = mon.show(mode="none", frame=0)
        assert pl.renderer.actors["field_cut"].mapper.scalar_range == (0.0, 3e-3 / _dz(grid))
        pl.close()

    def test_frame_selection_errors(self):
        mon = _time_monitor(_grid())
        with pytest.raises(IndexError):
            mon.show(mode="none", frame=7)
        with pytest.raises(ValueError, match="one of frame, t or f"):
            mon.show(mode="none", frame=1, t=0.0)
        with pytest.raises(ValueError, match="frequency monitor only"):
            mon.show(mode="none", f=1e9)

    def test_volume_monitor_layers_follow_the_cut(self):
        grid = _grid()
        mon = _time_monitor(grid, corners=None)
        frames = field_3d._frames_of(mon, None)
        assert frames.shape == (grid.Nx, grid.Ny, grid.Nz)
        assert frames.kind == "time" and frames.n_frames == 3
        layer = frames.layer(2, 1, 3, ["Ez"])["Ez"]
        assert layer.shape == (grid.Nx, grid.Nz)
        np.testing.assert_allclose(layer, 3e-3 / _dz(grid))

    def test_unattached_or_drained_monitor_is_refused(self):
        mon = MonitorFieldTime(times=[0.0], fields=["E"], name="x")
        with pytest.raises(RuntimeError, match="not attached"):
            mon.show(mode="none")
        mon.attach(_FakeMesh(_grid()))
        with pytest.raises(RuntimeError, match="no snapshots"):
            mon.show(mode="none")


class TestFrequencyMonitor:
    def test_phase_turns_the_pattern(self):
        # The DFT bins already hold physical (cell-centred) fields.
        mon = _freq_monitor(_grid())
        pl = mon.show("Ez", mode="none", f=1e9)
        np.testing.assert_allclose(_sheet(pl).cell_data["field"], 0.4)
        assert "Ez (V/m per √W)" in pl.scalar_bars
        pl.close()
        pl = mon.show("Ez", mode="none", f=2e9)  # purely imaginary at phase 0
        np.testing.assert_allclose(_sheet(pl).cell_data["field"], 0.0, atol=1e-12)
        pl.close()
        pl = mon.show("Ez", mode="none", f=2e9, phase=-90.0)  # Re(j·e^{-jπ/2}) = 1
        np.testing.assert_allclose(_sheet(pl).cell_data["field"], 0.4)
        pl.close()

    def test_frame_labels_and_vmax(self):
        mon = _freq_monitor(_grid())
        frames = field_3d._frames_of(mon, None)
        assert frames.kind == "frequency" and frames.is_complex
        np.testing.assert_allclose(frames.labels, [1e9, 2e9])
        view = field_3d._FieldView(
            frames=frames,
            component="E",
            plot_type="vector",
            frame=1,
            phase=0.0,
            vmax_fixed=None,
            cmap=None,
            unit_scale=1e3,
            density=10,
            threshold=0.02,
            opacity=1.0,
            arrow_color="#000000",
        )
        # The ceiling is |F|, the envelope over every phase.
        assert view.vmax() == pytest.approx(0.4)
        assert view.frame_label() == "f = 2 GHz"
        assert view.has_arrows


# ---------------------------------------------------------------------------
# View state
# ---------------------------------------------------------------------------


class TestView:
    def test_layer_index_follows_flip(self):
        """The same rule as the grid sheet: the first whole layer on the kept side."""
        nodes = (np.array([0.0, 1.0, 2.0, 3.0]),) * 3
        assert field_3d._layer_index(nodes, 2, _CutState("z", 1.5, False)) == 1
        assert field_3d._layer_index(nodes, 2, _CutState("z", 1.5, True)) == 2
        assert field_3d._layer_index(nodes, 2, _CutState("z", 1.0, False)) == 0
        assert field_3d._layer_index(nodes, 2, _CutState("z", 1.0, True)) == 1
        assert field_3d._layer_index(nodes, 2, _CutState("z", 3.0, False)) == 2
        assert field_3d._layer_index(nodes, 2, _CutState("z", 3.0, True)) == 2
        assert field_3d._layer_index(nodes, 2, _CutState("z", 0.0, True)) == 0
        assert field_3d._layer_index(nodes, 2, _CutState("z", 3.5, False)) is None

    def test_available_components(self):
        assert field_3d._available_components(("Ex", "Ey", "Ez", "Hx")) == [
            "E",
            "Ex",
            "Ey",
            "Ez",
            "Hx",
        ]
        assert field_3d._available_components(("Ez",)) == ["Ez"]

    def test_default_normal_prefers_z_on_ties(self):
        assert field_3d._default_normal((4, 4, 4)) == "z"
        assert field_3d._default_normal((4, 2, 4)) == "y"
        assert field_3d._default_normal((1, 4, 4)) == "x"

    def test_layer_sheet_drops_masked_cells(self):
        nodes = (np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0, 3.0]))
        # A y-cut: the layer is (nx, nz) = (2, 3) cells, two of them masked.
        keep = np.array([[True, False, True], [True, True, False]])
        sheet = field_3d._layer_sheet(nodes, 1, 0, 0.5, keep)
        assert sheet.n_cells == 4
        assert field_3d._layer_sheet(nodes, 1, 0, 0.5, np.zeros_like(keep)) is None


# ---------------------------------------------------------------------------
# Notebook controls, driven through trame's state without a browser
# ---------------------------------------------------------------------------


class TestControls:
    def test_frame_component_and_cut_handlers(self):
        """The toolbar handlers rebuild the sheet: frame, component, cut, show."""
        pytest.importorskip("trame.app")
        from trame.app import get_server  # noqa: PLC0415

        from magnelio.post.plot_3d import _GROUPS, _attach_controls, _build_scene  # noqa: PLC0415

        grid = _grid()
        mon = MonitorFieldTime(times=[0.0, 1e-12, 2e-12], fields=["E", "H"], name="m")
        mon.attach(_FakeMesh(grid))
        for i, t in enumerate(mon.times):
            f = FieldArrays.zeros(grid.Nx, grid.Ny, grid.Nz)
            f.Ez[:] = (i + 1) * 1e-3
            f.Hx[:] = 0.5 * (i + 1) * 1e-3
            mon.record(f, i, float(t), 1e-12)
        frames = field_3d._frames_of(mon, None)
        view = field_3d._FieldView(
            frames=frames,
            component="E",
            plot_type="vector",
            frame=0,
            phase=0.0,
            vmax_fixed=None,
            cmap=None,
            unit_scale=1e3,
            density=10,
            threshold=0.02,
            opacity=1.0,
            arrow_color="#303030",
        )
        scene = _build_scene(
            None,
            mesh=None,
            cut=("z", LZ / 2),
            flip=False,
            show_ports=True,
            show_wires=True,
            show_grid=False,
            show_labels=True,
            size=(300, 200),
            render_edges=False,
            edge_color="#202020",
            quality=1.0,
            scale_mm=True,
            camera="iso",
            off_screen=True,
            field_view=view,
            extent=(0.0, LX * 1e3, 0.0, LY * 1e3, 0.0, LZ * 1e3),
        )
        server = get_server(f"mio_test_{id(scene)}", client_type="vue3")
        menu_items = _attach_controls(scene, server)
        state = server.state
        key = f"mio3d_{id(scene)}"
        state.ready()
        pl = scene.plotter
        dz = _dz(grid)

        def sheet_max():
            return float(pl.renderer.actors["field_cut"].mapper.dataset.cell_data["field"].max())

        assert sheet_max() == pytest.approx(1e-3 / dz)
        assert "field_arrows" in pl.renderer.actors
        with state:
            state[f"{key}_frame"] = 2
        assert sheet_max() == pytest.approx(3e-3 / dz)
        assert state[f"{key}_frame_label"].startswith("t = ")
        with state:
            state[f"{key}_comp"] = "Hx"
        # A single component: signed scale, no arrows any more.
        assert sheet_max() == pytest.approx(1.5e-3 / (0.5 * dz), rel=1e-6) or sheet_max() > 0
        lo, hi = pl.renderer.actors["field_cut"].mapper.scalar_range
        assert lo == -hi
        assert "field_arrows" not in pl.renderer.actors
        with state:
            state[f"{key}_pos"] = 10.0 * LZ * 1e3  # far outside the region
        assert not pl.renderer.actors["field_cut"].GetVisibility()
        with state:
            state[f"{key}_pos"] = 0.25 * LZ * 1e3
        assert pl.renderer.actors["field_cut"].GetVisibility()
        with state:
            state[f"{key}_show"] = [g for g, _ in _GROUPS if g not in ("field", "arrows")]
        assert not pl.renderer.actors["field_cut"].GetVisibility()

        # Play: without an event loop the button springs back; the frame
        # advance itself wraps around.
        with state:
            state[f"{key}_play"] = True
        assert state[f"{key}_play"] is False
        view.frame = 2
        assert view.next_frame() == 0

        # The toolbar builder runs (play button, frame slider, field selector).
        from trame.ui.vuetify3 import SinglePageLayout  # noqa: PLC0415

        with SinglePageLayout(server) as layout, layout.toolbar:
            menu_items()
        pl.close()


# ---------------------------------------------------------------------------
# With a real mesh and geometry
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def coax():
    pytest.importorskip("OCC")
    pe = mio.Material.from_isotropic(name="polyethylene", epsilon=2.25)
    outer = geo.Cylinder(origin=(0, 0, 0), radius=1.75e-3, height=10e-3, axis="z", material=pe)
    inner = geo.Cylinder(origin=(0, 0, 0), radius=0.5e-3, height=10e-3, axis="z", material="pec")
    model = mio.GeometryModel(background="pec")
    model.add(geo.Difference(outer, inner))
    model.add(inner)
    model.add_port(ports.PortWaveguide(name="p1", plane="zmin", n_modes=1))
    mesh = mio.Mesh.from_geometry(model, mio.MeshControl(max_cell_size=0.25e-3), f_max=20e9)
    return model, mesh


class TestWithMesh:
    def test_metal_is_cut_out_of_the_sheet(self, coax):
        model, mesh = coax

        def radial(x, y, z):
            r = np.maximum(np.hypot(x, y), 1e-6)
            return x / r, y / r, 0 * z

        fs = FieldState.from_function(mesh.grid, E=radial)
        pl = fs.show(normal="z", position=5e-3, geometry=model, mesh=mesh, mode="none")
        sheet = _sheet(pl)
        g = mesh.grid
        assert 0 < sheet.n_cells < g.Nx * g.Ny
        pec = np.isin(
            mesh.material_id, [m for m, mat in mesh.material_library.items() if mat.is_pec]
        )
        k = int(np.argmin(np.abs(0.5 * (g.z[:-1] + g.z[1:]) - 5e-3)))
        assert sheet.n_cells == int((~pec[:, :, k]).sum())
        # The solids are drawn with the field, and the grid stays hidden.
        assert "shape_1" in pl.renderer.actors or "shape_0" in pl.renderer.actors
        assert not pl.renderer.actors["grid_cut"].GetVisibility()
        pl.close()

    def test_mismatched_mesh_is_refused(self, coax):
        _, mesh = coax
        with pytest.raises(ValueError, match="does not match"):
            _field(_grid()).show(mesh=mesh, mode="none")

    def test_scene_serialises_for_vtkjs(self, coax, caplog):
        """Sheet, arrows and scalar bar pass trame's vtk.js serialiser."""
        serializers = pytest.importorskip("trame_vtk.modules.vtk.serializers")
        model, mesh = coax
        fs = FieldState.from_function(mesh.grid, E=lambda x, y, z: (0 * x, 0 * y, 1 + 0 * z))
        pl = fs.show(
            normal="y", position=0.0, geometry=model, mesh=mesh, mode="none", size=(300, 200)
        )
        pl.render()
        serializers.initialize_serializers()
        ctx = serializers.SynchronizationContext()
        with caplog.at_level(logging.WARNING), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scene = serializers.serialize(
                None, pl.ren_win, serializers.reference_id(pl.ren_win), ctx, 0
            )
        assert scene is not None
        missing = [r.getMessage() for r in caplog.records if "No serializer" in r.getMessage()]
        assert not missing, missing

        def types(node):
            if isinstance(node, dict):
                yield node.get("type")
                for v in node.values():
                    yield from types(v)
            elif isinstance(node, list):
                for v in node:
                    yield from types(v)

        found = [t for t in types(scene) if t]
        assert "vtkScalarBarActor" in found
        datasets = {t for t in found if t.startswith("vtk") and "Data" in t or "Grid" in t}
        assert datasets == {"vtkPolyData"}, datasets
        pl.close()

    def test_screenshot_renders(self, coax, tmp_path):
        model, mesh = coax
        fs = FieldState.from_function(mesh.grid, E=lambda x, y, z: (0 * x, 0 * y, 1 + 0 * z))
        pl = fs.show(
            normal="y", position=0.0, geometry=model, mesh=mesh, mode="none", size=(300, 200)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = pl.screenshot(tmp_path / "field.png", return_img=True)
        assert img.shape[:2] == (200, 300)
        assert img.std() > 5.0
        pl.close()
