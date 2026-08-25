"""DD-193: short intervals keep the fine-end cell at h_fine and relax the ratio."""

import numpy as np
import pytest

from magnelio.mesh.mesher import (
    MeshControl,
    _grade_symmetric_to_uniform,
    _grade_then_uniform,
    _ratio_for_exact_fill,
)


def _one_sided_total(h0, n, r):
    return h0 * n if abs(r - 1.0) < 1e-12 else h0 * (r**n - 1.0) / (r - 1.0)


class TestRatioForExactFill:
    def test_fills_exactly_with_ratio_below_g(self):
        r = _ratio_for_exact_fill(4e-3, 66.7e-6, 12, 1.3, symmetric=False)
        assert 1.0 < r < 1.3
        assert _one_sided_total(66.7e-6, 12, r) == pytest.approx(4e-3, rel=1e-9)

    def test_returns_g_when_g_cannot_fill(self):
        assert _ratio_for_exact_fill(1.0, 1e-3, 3, 1.3, symmetric=False) == 1.3

    def test_returns_one_when_uniform_overfills(self):
        assert _ratio_for_exact_fill(1e-3, 1e-3, 3, 1.3, symmetric=False) == 1.0

    def test_symmetric_series(self):
        r = _ratio_for_exact_fill(1.5e-3, 50e-6, 15, 1.3, symmetric=True)
        n_half = 7
        series = (r**n_half - 1.0) / (r - 1.0)
        assert 50e-6 * (2.0 * series + r**n_half) == pytest.approx(1.5e-3, rel=1e-9)
        assert 1.0 < r < 1.3


class TestShortIntervals:
    """The interval is too short for a full ramp to h_max (the DD-192 air
    slab above a thin trace) — the fine-end cell must not undershoot."""

    @pytest.mark.parametrize("h_fine", [66.7e-6, 50e-6, 33.3e-6, 25e-6, 71e-6])
    def test_one_sided_fine_end_is_h_fine(self, h_fine):
        nodes = np.asarray(
            _grade_then_uniform(p_fine=1e-3, p_coarse=5e-3, h_fine=h_fine, h_max=1.67e-3, g=1.3)
        )
        d = np.diff(nodes)
        # never below h_fine; at most the DD-105 5 % overshoot band
        assert h_fine * (1 - 1e-9) <= d[0] <= h_fine * 1.05
        assert (d[1:] / d[:-1]).max() <= 1.3 * (1 + 1e-9)
        assert nodes[-1] == pytest.approx(5e-3)

    def test_one_sided_descending_orientation(self):
        nodes = np.asarray(
            _grade_then_uniform(p_fine=5e-3, p_coarse=1e-3, h_fine=66.7e-6, h_max=1.67e-3, g=1.3)
        )
        d = np.diff(nodes)
        assert 66.7e-6 * (1 - 1e-9) <= d[-1] <= 66.7e-6 * 1.05
        assert (d[:-1] / d[1:]).max() <= 1.3 * (1 + 1e-9)

    @pytest.mark.parametrize("h_fine", [40e-6, 50e-6, 66.7e-6])
    def test_symmetric_fine_ends_are_h_fine(self, h_fine):
        nodes = np.asarray(
            _grade_symmetric_to_uniform(p0=0.0, p1=2e-3, h_fine=h_fine, h_max=1.5e-3, g=1.3)
        )
        d = np.diff(nodes)
        assert h_fine * (1 - 1e-9) <= d[0] <= h_fine * 1.05
        assert h_fine * (1 - 1e-9) <= d[-1] <= h_fine * 1.05
        assert (d[1:] / d[:-1]).max() <= 1.3 * (1 + 1e-9)
        assert (d[:-1] / d[1:]).max() <= 1.3 * (1 + 1e-9)

    def test_floor_still_wins(self):
        nodes = np.asarray(
            _grade_then_uniform(
                p_fine=1e-3, p_coarse=5e-3, h_fine=50e-6, h_max=1.67e-3, g=1.3, min_cell=80e-6
            )
        )
        assert np.diff(nodes).min() >= 80e-6 * (1 - 1e-9)

    def test_no_undershoot_warning_on_a_thin_trace_in_air(self):
        pytest.importorskip("OCC")
        import warnings

        from magnelio.geo import Brick, GeometryModel
        from magnelio.materials.material import Material
        from magnelio.mesh.mesher import Mesh
        from magnelio.ports import PortWaveguide

        fr4 = Material(name="FR4", epsilon=(4.3,) * 3)
        sub = Brick(origin=(-4e-3, 0, 0), size=(8e-3, 0.8e-3, 16e-3), material=fr4)
        air = Brick(origin=(-4e-3, 0.8e-3, 0), size=(8e-3, 4.2e-3, 16e-3), material="air")
        strip = Brick(origin=(-0.6e-3, 0.8e-3, 0), size=(1.2e-3, 0.2e-3, 16e-3), material="pec")
        m = GeometryModel(boundary_conditions={"xmin": "SymmetryPMC"})
        m.add(sub)
        m.add(air - strip)
        m.add(strip)
        m.add_port(PortWaveguide(name="port1", plane="zmin", n_modes=1))
        m.add_port(PortWaveguide(name="port2", plane="zmax", n_modes=1))
        for mnpw in (12, 16):
            ctrl = MeshControl(min_nodes_per_wavelength=mnpw, min_cells_per_feature=mnpw // 4)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                Mesh.from_geometry(m, ctrl, f_max=15e9)
            assert not [x for x in w if "below" in str(x.message)]
