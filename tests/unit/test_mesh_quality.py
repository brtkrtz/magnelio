"""Unit tests for the mesh quality checks.

The grading-undershoot check is the interesting one: it must fire when
the time-step-setting cell came out finer than its interval asked for,
and stay silent on every cell size the user actually requested.
"""

import warnings

import numpy as np
import pytest

from magnelio.geo import Difference, GeometryModel
from magnelio.geo.primitives import Brick
from magnelio.materials import Material
from magnelio.mesh import mesher
from magnelio.mesh._quality import check_grading_undershoot, check_quality
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh, MeshControl


def _uniform(nodes):
    """Grid-line dict with *nodes* on y and two coarse cells elsewhere."""
    coarse = [0.0, 1e-2, 2e-2]
    return {"x": coarse, "y": list(nodes), "z": coarse}


def _warnings_from(**kwargs):
    """Collect UserWarnings raised by check_grading_undershoot."""
    defaults = dict(
        axis_planes={"x": [0.0, 2e-2], "y": [0.0, 1e-2], "z": [0.0, 2e-2]},
        axis_anchors={"x": set(), "y": set(), "z": set()},
        h_fine_axis={"x": 1e-2, "y": 1e-3, "z": 1e-2},
        h_max=1e-2,
        control=MeshControl(),
    )
    defaults.update(kwargs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_grading_undershoot(**defaults)
    return [str(w.message) for w in caught if issubclass(w.category, UserWarning)]


class TestGradingUndershoot:
    def test_undershoot_is_reported(self):
        # h_fine = 1 mm, but the interval was cut into 0.5 mm cells.
        msgs = _warnings_from(grid_lines=_uniform(np.arange(0, 1.05e-2, 5e-4)))
        assert len(msgs) == 1
        assert "50% below" in msgs[0]
        assert "sets the time step" in msgs[0]

    def test_suggested_min_cell_size_is_the_requested_h_fine(self):
        msgs = _warnings_from(grid_lines=_uniform(np.arange(0, 1.05e-2, 5e-4)))
        assert "min_cell_size=0.001" in msgs[0]

    def test_cell_at_h_fine_is_silent(self):
        assert _warnings_from(grid_lines=_uniform(np.arange(0, 1.05e-2, 1e-3))) == []

    def test_small_undershoot_stays_below_threshold(self):
        # 0.9 mm against h_fine = 1 mm: 10 %, not worth a warning.
        nodes = [0.0, 9e-4] + list(np.arange(9e-4 + 1e-3, 1.05e-2, 1e-3))
        assert _warnings_from(grid_lines=_uniform(nodes)) == []

    def test_user_floor_is_not_an_undershoot(self):
        msgs = _warnings_from(
            grid_lines=_uniform(np.arange(0, 1.05e-2, 5e-4)),
            control=MeshControl(min_cell_size=5e-4),
        )
        assert msgs == []

    def test_anchor_pair_is_never_reported(self):
        nodes = [0.0, 5e-4] + list(np.arange(1e-3, 1.05e-2, 1e-3))
        msgs = _warnings_from(
            grid_lines=_uniform(nodes),
            axis_anchors={"x": set(), "y": {0.0, 5e-4}, "z": set()},
        )
        assert msgs == []

    def test_interval_shorter_than_h_fine_is_silent(self):
        # The interval [0, 0.3 mm] cannot hold a 1 mm cell; its single
        # cell is forced by the geometry, not by the cell count.
        nodes = [0.0, 3e-4] + list(np.arange(1.3e-3, 1.05e-2, 1e-3))
        msgs = _warnings_from(
            grid_lines=_uniform(nodes),
            axis_planes={"x": [0.0, 2e-2], "y": [0.0, 3e-4, 1e-2], "z": [0.0, 2e-2]},
        )
        assert msgs == []

    def test_wavelength_driven_fine_size_is_skipped(self):
        # h_fine == h_max: the cell count follows the wavelength
        # criterion, which is the user's accuracy choice.
        msgs = _warnings_from(
            grid_lines=_uniform(np.arange(0, 1.05e-2, 5e-4)),
            h_fine_axis={"x": 1e-2, "y": 1e-3, "z": 1e-2},
            h_max=1e-3,
        )
        assert msgs == []

    def test_only_the_globally_smallest_cell_is_examined(self):
        # y undershoots, but z holds an even smaller legitimate cell —
        # z bounds dt, so there is nothing to gain on y.
        lines = _uniform(np.arange(0, 1.05e-2, 5e-4))
        lines["z"] = [0.0, 1e-4, 2e-2]
        msgs = _warnings_from(
            grid_lines=lines,
            axis_planes={"x": [0.0, 2e-2], "y": [0.0, 1e-2], "z": [0.0, 1e-4, 2e-2]},
            h_fine_axis={"x": 1e-2, "y": 1e-3, "z": 1e-2},
        )
        assert msgs == []


class TestBufferUndershoot:
    """DD-107 follow-up: buffer cells that bound dt are reported.

    A wavelength-driven axis is normally skipped, but the domain-face
    buffer can force three sub-``h_max`` cells there — cells the user
    never asked for, which then set the global time step.
    """

    def _kwargs(self, **over):
        # y: buffered boundary interval [0, 4 mm] holding three 1.33 mm
        # buffer cells on an otherwise wavelength-driven (h = 10 mm) axis.
        third = 4e-3 / 3
        kw = dict(
            grid_lines={
                "x": [0.0, 1e-2, 2e-2],
                "y": [0.0, third, 2 * third, 4e-3, 1.2e-2, 2e-2],
                "z": [0.0, 1e-2, 2e-2],
            },
            axis_planes={"x": [0.0, 2e-2], "y": [0.0, 4e-3, 2e-2], "z": [0.0, 2e-2]},
            h_fine_axis={"x": 1e-2, "y": 1e-2, "z": 1e-2},
        )
        kw.update(over)
        return kw

    def test_buffer_cell_on_wavelength_axis_is_reported(self):
        msgs = _warnings_from(**self._kwargs(buffer_ends={"y": ("lo",)}))
        assert len(msgs) == 1
        assert "buffer at the ymin domain face" in msgs[0]
        assert "sets the time step" in msgs[0]
        assert "Declaring the analysis ports" in msgs[0]

    def test_declared_port_buffer_names_the_other_remedy(self):
        msgs = _warnings_from(**self._kwargs(buffer_ends={"y": ("lo",)}, ports_declared=True))
        assert len(msgs) == 1
        assert "required by the port" in msgs[0]

    def test_without_buffer_info_the_wavelength_axis_stays_skipped(self):
        assert _warnings_from(**self._kwargs()) == []

    def test_other_end_buffered_stays_silent(self):
        # The small cell sits at ymin; a buffer at ymax does not
        # explain it, so the wavelength skip applies.
        assert _warnings_from(**self._kwargs(buffer_ends={"y": ("hi",)})) == []

    def test_integer_rounding_in_buffered_interval_stays_silent(self):
        # Four 1 mm cells against max_cell_size = 1.5 mm: a plain fill
        # needs ceil(4/1.5) = 3 cells anyway, so the buffer forced
        # nothing — ordinary rounding, skipped as before.
        msgs = _warnings_from(
            **self._kwargs(
                grid_lines={
                    "x": [0.0, 1e-2, 2e-2],
                    "y": [0.0, 1e-3, 2e-3, 3e-3, 4e-3, 1.2e-2, 2e-2],
                    "z": [0.0, 1e-2, 2e-2],
                },
                buffer_ends={"y": ("lo",)},
                control=MeshControl(max_cell_size=1.5e-3),
            )
        )
        assert msgs == []

    def test_buffer_undershoot_reaches_the_production_path(self):
        # 60 x 52 x 60 mm, PEC bar leaving a 12 mm boundary interval on
        # y: the ymax buffer forces three 4 mm cells where the
        # wavelength (h_max ~ 10 mm) asked for nothing finer.
        air = Material.air()
        pec = Material.pec()
        body = Brick(material=air).from_corners((0, 0, 0), (60e-3, 40e-3, 60e-3), material=air)
        bar = Brick(material=pec).from_corners((0, 40e-3, 0), (60e-3, 52e-3, 60e-3), material=pec)
        model = GeometryModel()
        model.add((body, bar))
        with pytest.warns(UserWarning, match="buffer at the ymax domain face"):
            Mesh.from_geometry(
                model,
                MeshControl(min_nodes_per_wavelength=10, min_cells_per_feature=0),
                f_max=3e9,
            )


class TestCheckQuality:
    def _mesh_with(self, y_nodes):
        grid = GridLines(
            x=np.array([0.0, 1e-3, 2e-3]),
            y=np.array(y_nodes),
            z=np.array([0.0, 1e-3, 2e-3]),
        )
        return Mesh.from_grid(grid)

    def test_wide_cell_size_spread_is_not_a_warning(self):
        # A thin feature in a large domain: cells span 10 um to 1 mm,
        # a ratio of 100, reached at a smooth 1.3 per step.  Ordinary,
        # and it carries no defect.
        nodes, h = [0.0], 1e-5
        while h < 1e-3:
            nodes.append(nodes[-1] + h)
            h *= 1.3
        assert max(np.diff(nodes)) / min(np.diff(nodes)) > 10.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._mesh_with(nodes)
        assert [str(w.message) for w in caught if issubclass(w.category, UserWarning)] == []

    def test_steep_gradient_still_warns(self):
        mesh = self._mesh_with([0.0, 1e-4, 1e-3, 2e-3])
        with pytest.warns(UserWarning, match="growth factor"):
            check_quality(mesh)

    def test_zero_cell_size_raises(self):
        grid = GridLines(
            x=np.array([0.0, 1e-3]),
            y=np.array([0.0, 1e-3]),
            z=np.array([0.0, 1e-3]),
        )
        mesh = Mesh.from_grid(grid)
        object.__setattr__(mesh.grid, "x", np.array([0.0, 0.0]))
        with pytest.raises(ValueError, match="zero or negative"):
            check_quality(mesh)


class TestUndershootEndToEnd:
    """The reported geometry in miniature: a 1 mm wall next to a 2 mm gap."""

    def _model(self):
        air = Material.air()
        pec = Material.pec()
        outer = Brick(material=air).from_corners((0, 0, 0), (5e-3, 3e-3, 5e-3), material=air)
        inner = Brick(material=air).from_corners((0, 4e-3, 0), (5e-3, 1e-2, 5e-3), material=air)
        bar = Brick(material=pec).from_corners((0, 6e-3, 0), (2e-3, 8e-3, 5e-3), material=pec)
        model = GeometryModel(background=pec)
        model.add((outer, Difference(inner, bar), bar))
        return model

    def test_tolerance_keeps_the_gap_at_six_cells(self):
        # h_fine = 1 mm / 4 = 0.25 mm.  Six graded cells across the
        # 2 mm gap start at 0.2506 mm, over h_fine by 0.24 % — without
        # the slack that is rejected and seven cells of 0.1965 mm win.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mesh = Mesh.from_geometry(
                self._model(),
                MeshControl(min_nodes_per_wavelength=10),
                f_max=2e9,
            )
        assert not [w for w in caught if "interval asked for" in str(w.message)]
        y = mesh.grid.y
        gap = mesh.grid.dy[(y[:-1] >= 6e-3 - 1e-9) & (y[:-1] < 8e-3 - 1e-9)]
        assert len(gap) == 6
        assert min(gap) >= 2.5e-4 * (1.0 - 1e-9)

    def test_warning_reaches_the_production_path(self, monkeypatch):
        # Same geometry with the slack removed: one more cell per gap,
        # and the mesher says so.
        monkeypatch.setattr(mesher, "_H_FINE_TOL", 0.0)
        with pytest.warns(UserWarning, match="below the .* this interval asked for"):
            Mesh.from_geometry(
                self._model(),
                MeshControl(min_nodes_per_wavelength=10),
                f_max=2e9,
            )

    def test_min_cell_size_floors_every_cell(self):
        mesh = Mesh.from_geometry(
            self._model(),
            MeshControl(min_nodes_per_wavelength=10, min_cell_size=2.5e-4),
            f_max=2e9,
        )
        assert min(mesh.grid.dy) >= 2.5e-4 * (1.0 - 1e-9)
