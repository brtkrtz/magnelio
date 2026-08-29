"""Unit tests for the public field container ``magnelio.fields.FieldState`` (DD-224 Phase C)."""

import numpy as np
import pytest

from magnelio.fields import FieldState
from magnelio.mesh.grid import GridLines


@pytest.fixture
def grid():
    return GridLines(
        x=np.linspace(0, 10e-3, 6),
        y=np.array([0, 1e-3, 3e-3, 6e-3, 10e-3]),
        z=np.linspace(0, 4e-3, 4),
    )


def _linear(x, y, z):
    return (1.0 + x, 2.0 * y, 3.0 * z)


class TestConstruction:
    def test_zeros_shapes(self, grid):
        f = FieldState.zeros(grid)
        assert f.Ex.shape == (5, 5, 4)
        assert f.Ey.shape == (6, 4, 4)
        assert f.Ez.shape == (6, 5, 3)
        assert f.Hx.shape == (6, 4, 3)
        assert f.Hy.shape == (5, 5, 3)
        assert f.Hz.shape == (5, 4, 4)
        assert not f.is_complex

    def test_wrong_shape_rejected(self, grid):
        z = FieldState.zeros(grid)
        with pytest.raises(ValueError, match="Ey must have the Yee shape"):
            FieldState(grid, z.Ex, z.Ex, z.Ez, z.Hx, z.Hy, z.Hz)

    def test_grid_type_checked(self):
        with pytest.raises(TypeError, match="GridLines"):
            FieldState.zeros(object())

    def test_physical_round_trip(self, grid):
        rng = np.random.default_rng(3)
        z = FieldState.zeros(grid)
        comps = {
            c: rng.standard_normal(getattr(z, c).shape)
            for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        }
        f = FieldState(grid, **comps)
        for c, a in comps.items():
            np.testing.assert_allclose(f.component(c), a, rtol=1e-13)

    def test_from_function_samples_on_yee_positions(self, grid):
        f = FieldState.from_function(grid, E=_linear, H=_linear)
        for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            X, Y, Z = np.meshgrid(*f.positions(c), indexing="ij")
            expected = _linear(X, Y, Z)["xyz".index(c[1])]
            np.testing.assert_allclose(f.component(c), expected, rtol=1e-12)

    def test_from_function_complex(self, grid):
        f = FieldState.from_function(grid, E=lambda x, y, z: (1j + 0 * x, 0 * y, 0 * z))
        assert f.is_complex
        assert not f.real().is_complex
        np.testing.assert_allclose(f.real().Ex, 0.0)

    def test_unknown_component(self, grid):
        with pytest.raises(KeyError):
            FieldState.zeros(grid).component("Bx")


class TestPositions:
    def test_yee_offsets(self, grid):
        f = FieldState.zeros(grid)
        x, y, z = grid.x, grid.y, grid.z
        xc, yc, zc = 0.5 * (x[:-1] + x[1:]), 0.5 * (y[:-1] + y[1:]), 0.5 * (z[:-1] + z[1:])
        px, py, pz = f.positions("Ex")
        np.testing.assert_allclose(px, xc)
        np.testing.assert_allclose(py, y)
        np.testing.assert_allclose(pz, z)
        px, py, pz = f.positions("Hx")
        np.testing.assert_allclose(px, x)
        np.testing.assert_allclose(py, yc)
        np.testing.assert_allclose(pz, zc)
        np.testing.assert_allclose(f.cell_centres[1], yc)


class TestSampling:
    def test_at_reproduces_linear_field(self, grid):
        f = FieldState.from_function(grid, E=_linear, H=_linear)
        pts = np.array([[2.3e-3, 4.1e-3, 1.7e-3], [9.9e-3, 0.2e-3, 3.9e-3], [0.0, 0.0, 0.0]])
        E, H = f.at(pts)
        expected = np.stack(_linear(*pts.T), axis=-1)
        np.testing.assert_allclose(E, expected, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(H, expected, rtol=1e-10, atol=1e-12)

    def test_at_single_point_and_bounds(self, grid):
        f = FieldState.zeros(grid)
        E, H = f.at([1e-3, 1e-3, 1e-3])
        assert E.shape == (1, 3)
        with pytest.raises(ValueError, match="bounding box"):
            f.at([[20e-3, 0, 0]])

    def test_cell_centred_linear(self, grid):
        f = FieldState.from_function(grid, E=_linear, H=_linear)
        cc = f.cell_centred(["Ex", "Hz"])
        xc, yc, zc = f.cell_centres
        X, Y, Z = np.meshgrid(xc, yc, zc, indexing="ij")
        np.testing.assert_allclose(cc["Ex"], 1.0 + X, rtol=1e-12)
        np.testing.assert_allclose(cc["Hz"], 3.0 * Z, rtol=1e-12)

    def test_scaled(self, grid):
        f = FieldState.from_function(grid, E=_linear)
        np.testing.assert_allclose(f.scaled(2.0).Ez, 2.0 * f.Ez)


class TestPlot:
    def test_plot_vector_and_scalar(self, grid):
        import matplotlib

        matplotlib.use("Agg")
        f = FieldState.from_function(grid, E=_linear, H=_linear)
        fig, ax = f.plot("E", normal="z", position=2e-3)
        assert ax.get_title().startswith("E-field")
        fig, ax = f.plot("Hx", normal="y", plot_type="color", title="custom", unit="T")
        assert ax.get_title() == "custom"
        with pytest.raises(ValueError, match="Vector plots"):
            f.plot("Ex", normal="z")
