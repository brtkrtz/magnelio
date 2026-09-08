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
    # The solver calls record() after step n with e at t + dt (DD-259):
    # feed it so that the electric instant lands on each requested time.
    for i, t in enumerate(mon.times):
        f = FieldArrays.zeros(grid.Nx, grid.Ny, grid.Nz)
        f.Ez[:] = (i + 1) * 1e-3
        mon.record(f, i, float(t) - 1e-12, 1e-12)
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
    # The bins are grid quantities (E·dz), so 0.4 V/m is 0.4·dz per bin.
    dz = float(grid.z[1] - grid.z[0])
    acc = mon._accumulators["Ez"]
    acc._bins[0] = 0.4 * dz
    acc._bins[1] = 0.4j * dz
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
        # Re(j·e^{-jπ/2}) = 1: the phasors are those of the e^{-jwt}
        # convention, so the phase advances with time.
        pl = mon.show("Ez", mode="none", f=2e9, phase=90.0)
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
        # Fixed decimals from the bin spacing, so the readout keeps its width.
        assert view.frame_label() == "f = 2.0 GHz"
        assert view.frame_label(0) == "f = 1.0 GHz"
        assert view.label_chars() == len("f = 1.0 GHz")
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
            mon.record(f, i, float(t) - 1e-12, 1e-12)
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
        # The arrow actor stays (one identity for the browser) but is hidden.
        assert not pl.renderer.actors["field_arrows"].GetVisibility()
        assert "Hx (A/m)" in pl.scalar_bars and "|E| (V/m)" not in pl.scalar_bars
        # The sheet actor is the same object frame after frame.
        sheet_actor = pl.renderer.actors["field_cut"]
        with state:
            state[f"{key}_comp"] = "E"
        assert pl.renderer.actors["field_cut"] is sheet_actor
        assert pl.renderer.actors["field_arrows"].GetVisibility()
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


# ---------------------------------------------------------------------------
# The volume: arrows on a 3D lattice and isosurfaces (DD-261)
# ---------------------------------------------------------------------------


def _actor(pl, name):
    return pl.renderer.actors[name]


class TestVolume:
    def test_volume_loader_matches_cell_centred(self):
        grid = _grid()
        fs = _field(grid)
        frames = field_3d._frames_of(fs, None)
        vol = frames.volume(0, ["Ex", "Ey", "Ez"])
        cc = fs.cell_centred(["Ex", "Ey", "Ez"])
        for c in ("Ex", "Ey", "Ez"):
            np.testing.assert_array_equal(vol[c], cc[c])
        mon = _time_monitor(grid, corners=None)
        vol = field_3d._frames_of(mon, None).volume(2, ["Ez"])
        np.testing.assert_allclose(vol["Ez"], 3e-3 / _dz(grid))

    def test_lattice_is_even_and_resampling_is_trilinear(self):
        centres = (np.linspace(0.5, 9.5, 10), np.linspace(0.5, 4.5, 5), np.array([1.0]))
        raster = field_3d._lattice3(centres, 10)
        assert raster[0].size == 10 and raster[2].size == 1
        step = raster[0][1] - raster[0][0]
        assert raster[1][1] - raster[1][0] == pytest.approx(step, rel=1e-9)
        x, y, z = np.meshgrid(*centres, indexing="ij")
        linear = 2.0 * x - 3.0 * y + 0.5
        (out,), live = field_3d._resample3(centres, raster, [linear], None)
        rx, ry, rz = np.meshgrid(*raster, indexing="ij")
        assert live.all()
        np.testing.assert_allclose(out, 2.0 * rx - 3.0 * ry + 0.5, rtol=1e-12)
        # A raster point whose stencil is mostly inside metal is not live.
        valid = np.ones_like(linear, dtype=bool)
        valid[:5] = False
        (out,), live = field_3d._resample3(centres, raster, [linear], valid)
        assert not live[0, 0, 0] and live[-1, -1, 0]
        assert np.isnan(out[0, 0, 0])

    def test_volume_arrows_fill_the_kept_half(self):
        grid = _grid()
        pl = _field(grid).show(mode="none", volume="arrows", density=6, normal="z", position=LZ / 2)
        arrows = _actor(pl, "field_volume_arrows")
        assert arrows.GetVisibility()
        pd = arrows.mapper.dataset
        assert isinstance(pd, pv.PolyData) and pd.n_cells > 0
        # Clipped to the kept half (below the cut, flip=False); an arrow
        # may reach past its lattice point by at most one spacing.
        spacing = LZ * 1e3 / 4  # the lattice follows the longest axis
        assert pd.bounds[5] <= LZ * 1e3 / 2 + spacing * 1.01
        assert pd.bounds[4] >= -spacing
        # Coloured by magnitude on the sheet's scale; the cut arrows step aside.
        assert "mag" in pd.point_data
        lo, hi = arrows.mapper.scalar_range
        assert lo == 0.0 and hi == pytest.approx(float(_sheet(pl).cell_data["field"].max()))
        assert _actor(pl, "field_cut").GetVisibility()
        assert not _actor(pl, "field_arrows").GetVisibility()
        pl.close()

    def test_arrow_length_has_a_floor(self):
        """The shortest arrow is three tenths of the lattice spacing."""
        grid = _grid()
        fs = FieldState.from_function(
            grid, E=lambda x, y, z: (0 * x, 0 * y, 0.05 + 0.95 * (x / LX) ** 4)
        )
        pl = fs.show(mode="none", density=6, normal="z", position=LZ / 2, threshold=0.0)
        pd = _actor(pl, "field_arrows").mapper.dataset
        lengths = np.asarray(pd["len"], dtype=float)
        # The strongest cell sets the ceiling, so the longest arrow spans
        # one lattice spacing; the weakest field (5 %) would be invisible
        # without the floor.
        assert lengths.min() >= field_3d._ARROW_FLOOR * lengths.max() * (1.0 - 1e-6)
        assert lengths.min() < 0.5 * lengths.max()
        pl.close()

    def test_single_colour_arrows_on_request(self):
        pl = _field(_grid()).show(mode="none", density=6, arrow_color="#303030")
        pd = _actor(pl, "field_arrows").mapper.dataset
        assert _actor(pl, "field_arrows").mapper.scalar_visibility is False or (
            "mag" in pd.point_data
        )
        pl.close()

    def test_isosurface_of_a_standing_pattern(self):
        """|E| = |sin(πx/L)| at half the ceiling: two sheets at L/6 and 5L/6."""
        grid = _grid(nx=24)
        pl = _field(grid).show(
            mode="none", plot_type="color", volume="isosurface", normal="z", position=LZ / 2
        )
        iso = _actor(pl, "field_iso")
        assert iso.GetVisibility()
        pd = iso.mapper.dataset
        assert isinstance(pd, pv.PolyData) and pd.n_cells > 0
        x = pd.points[:, 0] / 1e3
        cell = LX / 24
        near_low = np.abs(x - LX / 6) < cell
        near_high = np.abs(x - 5 * LX / 6) < cell
        assert near_low.any() and near_high.any()
        assert (near_low | near_high).all()
        # Clipped to the kept half.
        assert pd.points[:, 2].max() <= LZ * 1e3 / 2 + 1e-9
        lo, hi = iso.mapper.scalar_range
        assert lo == 0.0 and hi == pytest.approx(float(_sheet(pl).cell_data["field"].max()))
        # Fixed levels in field units move the sheets.
        pl2 = _field(grid).show(mode="none", volume="isosurface", levels=[0.9])
        x2 = _actor(pl2, "field_iso").mapper.dataset.points[:, 0] / 1e3
        x_level = np.arcsin(0.9) / np.pi * LX
        assert (np.abs(x2 - x_level) < cell).any() and (np.abs(x2 - (LX - x_level)) < cell).any()
        pl.close()
        pl2.close()

    def test_no_cut_shows_the_whole_volume(self):
        from magnelio.post.plot_3d import _apply_cut, _build_scene  # noqa: PLC0415

        grid = _grid()
        frames = field_3d._frames_of(_field(grid), None)
        view = field_3d._FieldView(
            frames=frames,
            component="E",
            plot_type="vector",
            frame=0,
            phase=0.0,
            vmax_fixed=None,
            cmap=None,
            unit_scale=1e3,
            density=6,
            threshold=0.0,
            opacity=1.0,
            arrow_color=None,
            volume_start="both",
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
        pl = scene.plotter
        half = _actor(pl, "field_volume_arrows").mapper.dataset.n_points
        scene.cut = _CutState()
        _apply_cut(scene)
        whole = _actor(pl, "field_volume_arrows").mapper.dataset.n_points
        assert whole > half
        assert not _actor(pl, "field_cut").GetVisibility()
        assert _actor(pl, "field_iso").GetVisibility()
        assert _actor(pl, "field_iso").mapper.dataset.points[:, 2].max() > LZ * 1e3 / 2
        pl.close()

    def test_plane_monitor_offers_arrows_but_no_isosurface(self):
        grid = _grid()
        mon = MonitorFieldTime(
            corners=((None, None, LZ / 2), (None, None, LZ / 2)),
            times=[0.0],
            fields=["E"],
            name="m",
        )
        mon.attach(_FakeMesh(grid))
        f = FieldArrays.zeros(grid.Nx, grid.Ny, grid.Nz)
        f.Ez[:] = 1e-3
        f.Ex[:] = 0.5e-3
        mon.record(f, 0, -1e-12, 1e-12)
        with pytest.raises(ValueError, match="two cells"):
            mon.show(mode="none", volume="isosurface")
        pl = mon.show(mode="none", volume="arrows", density=5)
        assert _actor(pl, "field_volume_arrows").GetVisibility()
        pl.close()

    def test_bad_arguments(self):
        fs = _field(_grid())
        with pytest.raises(ValueError, match="volume must be"):
            fs.show(mode="none", volume="fog")
        with pytest.raises(ValueError, match="levels"):
            fs.show(mode="none", levels=[-1.0])
        with pytest.raises(ValueError, match="iso_level"):
            fs.show(mode="none", iso_level=1.5)


class TestVolumeControls:
    def test_show_menu_level_and_density(self):
        """Volume groups start hidden, the menu turns them on, the sliders reshape them."""
        pytest.importorskip("trame.app")
        from trame.app import get_server  # noqa: PLC0415

        from magnelio.post.plot_3d import _GROUPS, _attach_controls, _build_scene  # noqa: PLC0415

        grid = _grid(nx=24)
        frames = field_3d._frames_of(_field(grid), None)
        view = field_3d._FieldView(
            frames=frames,
            component="E",
            plot_type="vector",
            frame=0,
            phase=0.0,
            vmax_fixed=None,
            cmap=None,
            unit_scale=1e3,
            density=6,
            threshold=0.0,
            opacity=1.0,
            arrow_color=None,
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
        assert {"volume arrows", "isosurface"} <= set(scene.groups_present())
        assert {"volume arrows", "isosurface"} <= scene.hidden_groups
        pl = scene.plotter
        assert "field_volume_arrows" not in pl.renderer.actors
        assert "field_iso" not in pl.renderer.actors
        server = get_server(f"mio_test_vol_{id(scene)}", client_type="vue3")
        menu_items = _attach_controls(scene, server)
        state = server.state
        key = f"mio3d_{id(scene)}"
        state.ready()
        with state:
            state[f"{key}_show"] = [g for g, _ in _GROUPS]
        assert _actor(pl, "field_volume_arrows").GetVisibility()
        assert _actor(pl, "field_iso").GetVisibility()
        n_arrows = _actor(pl, "field_volume_arrows").mapper.dataset.n_points
        x_before = _actor(pl, "field_iso").mapper.dataset.points[:, 0].min()
        with state:
            state[f"{key}_level"] = 90
        x_after = _actor(pl, "field_iso").mapper.dataset.points[:, 0].min()
        assert x_after > x_before  # the sheets move inwards at a higher level
        with state:
            state[f"{key}_density"] = 12
        assert _actor(pl, "field_volume_arrows").mapper.dataset.n_points > n_arrows
        with state:
            state[f"{key}_show"] = [g for g, _ in _GROUPS if g not in ("volume arrows",)]
        assert not _actor(pl, "field_volume_arrows").GetVisibility()
        # A signed component: the isosurface carries both signs.
        with state:
            state[f"{key}_comp"] = "Hx"
        iso = _actor(pl, "field_iso").mapper.dataset
        assert iso["field"].min() < 0.0 < iso["field"].max()
        from trame.ui.vuetify3 import SinglePageLayout  # noqa: PLC0415

        with SinglePageLayout(server) as layout, layout.toolbar:
            menu_items()
        pl.close()


# ---------------------------------------------------------------------------
# DD-263: the v0.7.0 review
# ---------------------------------------------------------------------------


def _view_of(frames, **overrides) -> field_3d._FieldView:
    kwargs = dict(
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
        arrow_color=None,
    )
    kwargs.update(overrides)
    return field_3d._FieldView(**kwargs)


class TestLabels:
    def test_time_labels_share_one_width(self):
        # Frames 1 ps apart: three decimals in ns, every label as wide
        # as the widest, so the toolbar does not reflow while playing.
        view = _view_of(field_3d._frames_of(_time_monitor(_grid()), None))
        labels = [view.frame_label(i) for i in range(3)]
        assert labels == ["t = 0.000 ns", "t = 0.001 ns", "t = 0.002 ns"]
        assert view.label_chars() == len(labels[0])

    def test_one_frame_keeps_the_general_format(self):
        grid = _grid()
        mon = MonitorFieldTime(times=[1.234e-9], fields=["E"], name="m")
        mon.attach(_FakeMesh(grid))
        mon.record(FieldArrays.zeros(grid.Nx, grid.Ny, grid.Nz), 0, 1.234e-9 - 1e-12, 1e-12)
        view = _view_of(field_3d._frames_of(mon, None))
        assert view.frame_label() == "t = 1.234 ns"

    def test_a_single_field_has_no_label(self):
        view = _view_of(field_3d._frames_of(_field(_grid()), None))
        assert view.frame_label() == ""
        assert view.label_chars() == 0


class TestGlyphs:
    def test_arrow_and_cone_are_centred_on_the_sample_point(self):
        view = _view_of(field_3d._frames_of(_field(_grid()), None))
        for style in ("arrow", "cone"):
            view.glyph = style
            b = view._glyph_geometry().bounds
            assert b[0] == pytest.approx(-0.5, abs=1e-6)
            assert b[1] == pytest.approx(0.5, abs=1e-6)
            assert b[2] == pytest.approx(-b[3], abs=1e-6)

    def test_width_scales_the_radius_not_the_length(self):
        view = _view_of(field_3d._frames_of(_field(_grid()), None))
        thin = view._glyph_geometry().bounds
        view.glyph_width = 2.0
        thick = view._glyph_geometry().bounds
        assert thick[3] == pytest.approx(2.0 * thin[3], rel=1e-6)
        assert thick[1] == pytest.approx(thin[1], abs=1e-9)

    def test_show_takes_the_glyph_arguments(self):
        grid = _grid()
        pl = _field(grid).show(glyph="cone", glyph_width=1.5, mode="none", size=(300, 200))
        assert _actor(pl, "field_arrows").mapper.dataset.n_cells > 0
        pl.close()
        with pytest.raises(ValueError, match="glyph must be"):
            _field(grid).show(glyph="dart", mode="none")
        with pytest.raises(ValueError, match="glyph_width"):
            _field(grid).show(glyph_width=0.0, mode="none")


class TestLineAndPointMonitors:
    """A monitor of one cell along two or three axes still draws (the review's IndexError)."""

    def test_line_monitor_shows_sheet_and_arrows(self):
        grid = _grid()
        y0, z0 = 0.5 * LY, 0.5 * LZ
        mon = _time_monitor(grid, corners=((None, y0, z0), (None, y0, z0)))
        assert mon.recording.shape[1:] == (1, 1)
        pl = mon.show(mode="none", size=(300, 200))
        sheet = _sheet(pl)
        assert sheet.n_cells == grid.Nx
        assert _actor(pl, "field_arrows").mapper.dataset.n_points > 0
        pl.close()

    def test_point_monitor_shows_one_cell_and_one_arrow(self):
        grid = _grid()
        p = (0.4 * LX, 0.5 * LY, 0.5 * LZ)
        mon = _time_monitor(grid, corners=(p, p))
        assert mon.recording.shape == (1, 1, 1)
        pl = mon.show(mode="none", size=(300, 200))
        assert _sheet(pl).n_cells == 1
        arrows = _actor(pl, "field_arrows").mapper.dataset
        assert arrows.n_points > 0
        # One arrow, centred on the cell: its x extent straddles the centre.
        cx = 0.5 * (grid.x[2] + grid.x[3]) * 1e3
        assert arrows.bounds[0] < cx < arrows.bounds[1]
        pl.close()

    def test_resample_reads_a_one_cell_axis_as_constant(self):
        from magnelio.post.plot_field import _arrow_grid, _resample  # noqa: PLC0415

        xc = np.linspace(0.5, 5.5, 6)
        yc = np.array([2.0])
        xs, ys = _arrow_grid(xc, yc, 4)
        assert ys.size == 1  # not two coincident raster points
        (a,), live = _resample(xc, yc, xs, ys, [np.arange(6.0)[:, None]], None)
        assert live.all()
        np.testing.assert_allclose(a[:, 0], xs - 0.5)


class TestPhasePlay:
    def test_a_single_bin_gets_a_phase_play_button(self):
        pytest.importorskip("trame.app")
        from trame.app import get_server  # noqa: PLC0415
        from trame.ui.vuetify3 import SinglePageLayout  # noqa: PLC0415

        from magnelio.post.plot_3d import _attach_controls, _build_scene  # noqa: PLC0415

        grid = _grid()
        mon = MonitorFieldFrequency(freqs=[1.0e9], fields=["E"], name="p")
        mon.attach(_FakeMesh(grid))
        f = FieldArrays.zeros(grid.Nx, grid.Ny, grid.Nz)
        f.Ez[:] = 1e-3
        mon.record(f, 0, 0.0, 1e-12)
        mon.finalize()
        mon.renormalize(Signal1D(t=np.array([0.0]), values=np.array([1e12]), dt=1e-12))
        frames = field_3d._frames_of(mon, None)
        assert frames.n_frames == 1 and frames.is_complex
        view = _view_of(frames)
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
        server = get_server(f"mio_test_phase_{id(scene)}", client_type="vue3")
        menu_items = _attach_controls(scene, server)
        state = server.state
        key = f"mio3d_{id(scene)}"
        state.ready()
        # No event loop here: the button springs back, like the frame play.
        with state:
            state[f"{key}_play_phase"] = True
        assert state[f"{key}_play_phase"] is False
        with SinglePageLayout(server) as layout, layout.toolbar:
            menu_items()
        html = layout.html
        assert "Play / pause the phase" in html
        assert "Play / pause the frames" not in html  # one bin: no frame slider
        assert "phase =" in html and "tabular-nums" in html
        scene.plotter.close()

    def test_phase_play_advances_the_phase_on_a_loop(self):
        pytest.importorskip("trame.app")
        import asyncio  # noqa: PLC0415

        from trame.app import get_server  # noqa: PLC0415

        from magnelio.post.plot_3d import _attach_controls, _build_scene  # noqa: PLC0415

        grid = _grid()
        frames = field_3d._frames_of(_freq_monitor(grid), None)
        view = _view_of(frames, fps=50.0)
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
        server = get_server(f"mio_test_phase_loop_{id(scene)}", client_type="vue3")
        _attach_controls(scene, server)
        state = server.state
        key = f"mio3d_{id(scene)}"
        state.ready()

        async def drive():
            with state:
                state[f"{key}_play"] = True  # the frame play runs first ...
            await asyncio.sleep(0.05)
            with state:
                state[f"{key}_play_phase"] = True  # ... and yields to the phase play
            assert state[f"{key}_play"] is False
            await asyncio.sleep(0.15)
            with state:
                state[f"{key}_play_phase"] = False
            return float(state[f"{key}_phase"])

        phase = asyncio.run(drive())
        assert phase > 0.0 and phase % field_3d._PHASE_STEP == pytest.approx(0.0)
        assert view.phase == phase
        assert view._phase_task is None and view._play_task is None
        scene.plotter.close()


class TestEigenmodes:
    @staticmethod
    def _result(mesh, complex_second=False):
        from magnelio.solver.eigenmode_result import EigenmodeResult  # noqa: PLC0415

        grid = mesh.grid
        first = FieldState.from_function(
            grid, E=lambda x, y, z: (0 * x, 0 * y, np.sin(np.pi * x / 3.5e-3) + 2.0)
        )
        second = FieldState.from_function(grid, E=lambda x, y, z: (0 * x, 0 * y, 0 * z + 1.0))
        modes = [first._raw, second._raw]
        if complex_second:
            raw = second._raw
            modes[1] = FieldArrays(
                **{c: np.asarray(getattr(raw, c)) * np.exp(0.7j) for c in field_3d._COMPONENTS}
            )
        return EigenmodeResult(frequencies=np.array([1.0e9, 2.5e9]), modes=modes, mesh=mesh)

    def test_modes_are_the_frames(self, coax):
        _model, mesh = coax
        result = self._result(mesh)
        frames = field_3d._frames_of(result, None)
        assert frames.kind == "mode" and frames.n_frames == 2
        assert frames.pec is not None  # the result's own mesh cuts the metal out
        view = _view_of(frames)
        assert view.frame_label(0) == "mode 0  1.0 GHz"
        assert view.frame_label(1) == "mode 1  2.5 GHz"
        assert view.unit == "a.u."
        assert view.bar_title == "|E| (a.u.)"
        # The second mode is uniform: the whole layer reads 1.
        view.frame = 1
        scalar, _ = view.values(2, 0)
        assert np.nanmax(scalar) == pytest.approx(1.0)

    def test_show_starts_at_the_requested_mode(self, coax):
        _model, mesh = coax
        result = self._result(mesh)
        pl = result.show(frame=1, mode="none", size=(300, 200))
        assert "|E| (a.u.)" in pl.scalar_bars
        assert float(_sheet(pl).cell_data["field"].max()) == pytest.approx(1.0)
        pl.close()
        with pytest.raises(IndexError):
            result.show(frame=5, mode="none")

    def test_complex_mode_is_turned_to_its_energy_maximum(self, coax):
        _model, mesh = coax
        result = self._result(mesh, complex_second=True)
        frames = field_3d._frames_of(result, None)
        assert frames.is_complex
        state = field_3d._mode_state(result, 1)
        # The arbitrary phase e^{0.7j} is taken out: the real part is
        # the whole field, the imaginary part nothing.
        ez = state.component("Ez")
        assert np.abs(ez.imag).max() < 1e-9 * np.abs(ez.real).max()

    def test_empty_result_is_refused(self, coax):
        from magnelio.solver.eigenmode_result import EigenmodeResult  # noqa: PLC0415

        _model, mesh = coax
        empty = EigenmodeResult(frequencies=np.zeros(0), modes=[], mesh=mesh)
        with pytest.raises(ValueError, match="no mode"):
            empty.show(mode="none")


@pytest.fixture(scope="module")
def half_box():
    """An air box across x = 0 with a magnetic symmetry plane there, and a metal block."""
    pytest.importorskip("OCC")
    model = mio.GeometryModel(
        background="pec", boundary_conditions=mio.BoundaryConditions(xmin="SymmetryPMC")
    )
    model.add(geo.Brick(origin=(-6e-3, 1e-3, 0), size=(12e-3, 3e-3, 3e-3), material="air"))
    model.add(geo.Brick(origin=(-6e-3, 0, 0), size=(12e-3, 1e-3, 3e-3), material="pec"))
    mesh = mio.Mesh.from_geometry(model, mio.MeshControl(max_cell_size=1e-3), f_max=20e9)
    return model, mesh


class TestMirror:
    def test_frames_span_the_whole_model_with_the_right_parities(self, half_box):
        _model, mesh = half_box
        grid = mesh.grid
        # The mesher kept x >= 0 and pulled the first grid line half a
        # cell in: the magnetic wall sits at x = 0, between the samples.
        h = float(grid.x[1] - grid.x[0])
        assert grid.x[0] == pytest.approx(0.5 * h)
        fs = FieldState.from_function(grid, E=lambda x, y, z: (x / 6e-3, 0 * y + 1.0, 0 * z))
        half = field_3d._frames_of(fs, mesh, mirror=False)
        full = field_3d._frames_of(fs, mesh, mirror=True)
        assert not half.mirrored and full.mirrored
        nx = half.shape[0]
        # A magnetic wall half a cell outside the grid: the mirrored
        # region has 2n + 1 cells, the middle one astride the wall.
        assert full.shape == (2 * nx + 1, half.shape[1], half.shape[2])
        assert full.nodes[0][0] == pytest.approx(-grid.x[-1], rel=1e-9)
        vol = full.volume(0, ["Ex", "Ey", "Ez"])
        # E across a magnetic wall: the normal component odd, the
        # tangential one even.
        np.testing.assert_allclose(vol["Ex"], -vol["Ex"][::-1], atol=1e-12)
        np.testing.assert_allclose(vol["Ey"], vol["Ey"][::-1], atol=1e-12)
        assert np.all(np.abs(vol["Ex"][nx]) < 1e-12)
        # The metal mask follows the field: the block lies at y < 1 mm
        # on both sides of the wall.
        assert full.pec is not None and full.pec.shape == full.shape
        np.testing.assert_array_equal(full.pec, full.pec[::-1])
        assert full.pec[:, 0, :].all() and not full.pec[:, -1, :].any()

    def test_the_default_is_mirrored_and_the_plane_sits_where_declared(self, half_box):
        model, mesh = half_box
        grid = mesh.grid
        fs = FieldState.from_function(grid, E=lambda x, y, z: (0 * x, 0 * y + 1.0, 0 * z))
        pl = fs.show(geometry=model, mesh=mesh, normal="z", mode="none", size=(300, 200))
        # The sheet spans both halves, in mm.
        assert _sheet(pl).bounds[0] < -5.0 and _sheet(pl).bounds[1] > 5.0
        # The symmetry plane is drawn at x = 0, not on the scene's edge.
        plane = _actor(pl, "symmetry_xmin").mapper.dataset.bounds
        assert plane[0] == pytest.approx(0.0, abs=1e-9) and plane[1] == pytest.approx(0.0, abs=1e-9)
        pl.close()
        pl = fs.show(mesh=mesh, mirror=False, normal="z", mode="none", size=(300, 200))
        assert _sheet(pl).bounds[0] >= -1e-9
        pl.close()

    def test_a_grid_asked_for_on_a_mirrored_field_says_it_is_not_drawn(self, half_box):
        """The mesh covers the modelled part only, so a mirrored field has no grid."""
        model, mesh = half_box
        fs = FieldState.from_function(mesh.grid, E=lambda x, y, z: (0 * x, 0 * y + 1.0, 0 * z))
        with pytest.warns(UserWarning, match="mirror=False"):
            pl = fs.show(
                geometry=model, mesh=mesh, normal="z", show_grid=True, mode="none", size=(300, 200)
            )
        assert "grid_cut" not in pl.renderer.actors
        pl.close()
        # Without the mirroring the grid is there, and nothing is said.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pl = fs.show(
                geometry=model,
                mesh=mesh,
                normal="z",
                show_grid=True,
                mirror=False,
                mode="none",
                size=(300, 200),
            )
        assert pl.renderer.actors["grid_cut"].GetVisibility()
        pl.close()

    def test_a_region_short_of_the_plane_is_not_mirrored(self, half_box):
        _model, mesh = half_box
        grid = mesh.grid
        x0 = float(grid.x[2])
        mon = MonitorFieldTime(
            corners=((x0, None, None), (None, None, None)), times=[0.0], fields=["E"], name="m"
        )
        mon.attach(mesh)
        mon.record(FieldArrays.zeros(grid.Nx, grid.Ny, grid.Nz), 0, -1e-12, 1e-12)
        frames = field_3d._frames_of(mon, mesh, mirror=True)
        assert not frames.mirrored
