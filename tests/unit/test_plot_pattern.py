"""DD-174: pattern plot functions and their delegation chain."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from magnelio.plots import plot_pattern_3d, plot_pattern_cut
from magnelio.post.far_field import FarFieldResult

THETA = np.linspace(0.0, np.pi, 19)
PHI = np.linspace(0.0, 2.0 * np.pi, 37)


def _dipole_result() -> FarFieldResult:
    # A synthetic sin(theta) pattern in the container's amplitude slot.
    e_theta = np.sin(THETA)[:, None] * np.ones_like(PHI)[None, :] + 0.0j
    return FarFieldResult(
        f=1e9,
        theta=THETA,
        phi=PHI,
        E_theta=e_theta,
        E_phi=np.zeros_like(e_theta),
    )


class TestPolarCut:
    def test_creates_polar_axes(self):
        angles = np.linspace(0, 2 * np.pi, 73)
        fig, ax = plot_pattern_cut(angles, 1.5 * np.sin(angles) ** 2)
        assert ax.name == "polar"
        plt.close(fig)

    def test_rejects_rectangular_axes(self):
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="polar"):
            plot_pattern_cut(np.linspace(0, np.pi, 10), np.ones(10), ax=ax)
        plt.close(fig)

    def test_db_floor_clips_and_sets_the_radial_limits(self):
        angles = np.linspace(0, 2 * np.pi, 73)
        values = 1.5 * np.sin(angles) ** 2  # exact nulls -> -inf unclipped
        fig, ax = plot_pattern_cut(angles, values, db=True, floor_db=-30.0)
        line = ax.get_lines()[0]
        r = line.get_ydata()
        assert np.min(r) == pytest.approx(-30.0)
        assert np.max(r) == pytest.approx(10 * np.log10(1.5))
        assert ax.get_rmin() == pytest.approx(-30.0)
        plt.close(fig)

    def test_linear_mode_plots_raw_values(self):
        angles = np.linspace(0, 2 * np.pi, 10)
        fig, ax = plot_pattern_cut(angles, np.full(10, 2.0), db=False)
        np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), 2.0)
        plt.close(fig)

    def test_caller_axes_are_reused(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        fig2, ax2 = plot_pattern_cut(np.linspace(0, np.pi, 10), np.ones(10), ax=ax)
        assert ax2 is ax and fig2 is fig
        plt.close(fig)


class TestSurface3D:
    def test_creates_3d_axes(self):
        values = np.ones((THETA.size, PHI.size))
        fig, ax = plot_pattern_3d(THETA, PHI, values)
        assert ax.name == "3d"
        plt.close(fig)

    def test_rejects_flat_axes(self):
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="3D"):
            plot_pattern_3d(THETA, PHI, np.ones((THETA.size, PHI.size)), ax=ax)
        plt.close(fig)

    def test_db_radius_puts_the_floor_at_the_origin(self):
        # A pattern that dips to the floor must reach radius zero.
        values = np.ones((THETA.size, PHI.size))
        values[0, :] = 1e-12
        fig, ax = plot_pattern_3d(THETA, PHI, values, db=True, floor_db=-40.0)
        # Radial extent: max radius = 0 dB - floor = 40.
        assert ax.get_xlim()[1] == pytest.approx(40.0 * 1.05)
        plt.close(fig)


class TestDelegation:
    def test_result_plot_cut(self):
        res = _dipole_result()
        fig, ax = res.plot_cut(plane="phi", angle=0.0, quantity="directivity")
        assert ax.name == "polar"
        n = res.theta.size
        trace = ax.get_lines()[0].get_ydata()
        assert trace.shape[0] == 2 * n - 1
        plt.close(fig)

    def test_result_plot_3d(self):
        res = _dipole_result()
        fig, ax = res.plot_3d(quantity="directivity", floor_db=-20.0)
        assert ax.name == "3d"
        plt.close(fig)

    def test_unknown_quantity_is_rejected(self):
        with pytest.raises(ValueError, match="unknown quantity"):
            _dipole_result().plot_cut(quantity="EIRP")
