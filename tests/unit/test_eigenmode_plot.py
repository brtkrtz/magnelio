"""Unit tests for EigenmodeResult.plot() slice rendering.

Uses a synthetic single-mode result on a uniform grid: the FieldState
carries FIT grid quantities (e = E·l), so a uniform physical field maps
to Ex = E0·dx on every x-edge and the cell-centre interpolation must
recover E0 exactly.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from magnelio._fields.field_arrays import FieldState
from magnelio.mesh.grid import GridLines
from magnelio.solver.eigenmode_result import EigenmodeResult


class _FakeMesh:
    def __init__(self, grid):
        self.grid = grid


E0 = 2.0


@pytest.fixture
def uniform_result():
    Nx, Ny, Nz = 4, 5, 6
    grid = GridLines(
        x=np.linspace(0, 0.01, Nx + 1),
        y=np.linspace(0, 0.02, Ny + 1),
        z=np.linspace(0, 0.03, Nz + 1),
    )
    dx = grid.x[1] - grid.x[0]
    mode = FieldState(
        Ex=np.full((Nx, Ny + 1, Nz + 1), E0 * dx),
        Ey=np.zeros((Nx + 1, Ny, Nz + 1)),
        Ez=np.zeros((Nx + 1, Ny + 1, Nz)),
        Hx=np.zeros((Nx + 1, Ny, Nz)),
        Hy=np.zeros((Nx, Ny + 1, Nz)),
        Hz=np.zeros((Nx, Ny, Nz + 1)),
    )
    return EigenmodeResult(
        frequencies=np.array([1.5e9]),
        modes=[mode],
        mesh=_FakeMesh(grid),
    )


class TestEigenmodePlot:
    def test_scalar_slice_recovers_field(self, uniform_result):
        fig, ax = uniform_result.plot(
            mode=0, component="Ex", normal="z", position=0.014, plot_type="color"
        )
        qm = [c for c in ax.collections if hasattr(c, "get_clim")][0]
        np.testing.assert_allclose(np.asarray(qm.get_array()), E0)
        assert "1.500 GHz" in ax.get_title()
        assert "z=" in ax.get_title()
        plt.close(fig)

    def test_magnitude_slice(self, uniform_result):
        fig, ax = uniform_result.plot(
            mode=0, component="E", normal="y", position=0.0, plot_type="color"
        )
        qm = [c for c in ax.collections if hasattr(c, "get_clim")][0]
        np.testing.assert_allclose(np.asarray(qm.get_array()), E0)
        plt.close(fig)

    def test_vector_slice_normal_dominated(self, uniform_result):
        from matplotlib.collections import PathCollection

        # x-normal slice: the uniform Ex field is purely out-of-plane,
        # so the plot must consist of ⊙ markers, not arrows
        fig, ax = uniform_result.plot(
            mode=0, component="E", normal="x", position=0.0, plot_type="vector"
        )
        scatters = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(scatters) > 0
        plt.close(fig)

    def test_requires_normal(self, uniform_result):
        with pytest.raises(ValueError, match="normal="):
            uniform_result.plot(mode=0, component="E")

    def test_mode_out_of_range(self, uniform_result):
        with pytest.raises(IndexError):
            uniform_result.plot(mode=3, component="E", normal="y")

    def test_invalid_component(self, uniform_result):
        with pytest.raises(KeyError):
            uniform_result.plot(mode=0, component="Q", normal="y")

    def test_vector_rejects_single_component(self, uniform_result):
        with pytest.raises(ValueError, match="component='E' or 'H'"):
            uniform_result.plot(mode=0, component="Ex", normal="y", plot_type="vector")
