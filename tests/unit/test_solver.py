"""Unit tests for solver utilities (stability, Courant condition)."""

import numpy as np
import pytest

from magnelio.mesh.grid import GridLines
from magnelio.solver.stability import courant_dt, estimate_total_steps


def _grid(dx=1e-3, Nx=10, Ny=10, Nz=10):
    x = np.linspace(0, Nx * dx, Nx + 1)
    y = np.linspace(0, Ny * dx, Ny + 1)
    z = np.linspace(0, Nz * dx, Nz + 1)
    return GridLines(x=x, y=y, z=z)


class TestCourantDt:
    def test_normal_accuracy(self):
        grid = _grid(dx=1e-3)
        dt = courant_dt(grid, accuracy="normal")
        c0 = 299_792_458.0
        dt_max = 1.0 / (c0 * np.sqrt(3) / 1e-3)
        assert dt == pytest.approx(0.95 * dt_max, rel=1e-6)

    def test_draft_accuracy(self):
        grid = _grid(dx=1e-3)
        dt_draft = courant_dt(grid, accuracy="draft")
        dt_high = courant_dt(grid, accuracy="high")
        assert dt_draft < dt_high

    def test_invalid_accuracy(self):
        grid = _grid()
        with pytest.raises(ValueError, match="accuracy must be"):
            courant_dt(grid, accuracy="ultra")

    def test_dt_scales_with_cell_size(self):
        grid_coarse = _grid(dx=1e-2)
        grid_fine = _grid(dx=1e-3)
        dt_coarse = courant_dt(grid_coarse)
        dt_fine = courant_dt(grid_fine)
        assert dt_coarse > dt_fine


class TestEstimateTotalSteps:
    def test_basic(self):
        n = estimate_total_steps(f_max=1e9, dt=1e-12)
        # T_sim = 10 / 1e9 = 1e-8; steps = ceil(1e-8 / 1e-12) = 10000
        assert n == 10000

    def test_minimum_one(self):
        n = estimate_total_steps(f_max=1e12, dt=1e-12)
        assert n >= 1


class TestFieldState:
    def test_zeros_shapes(self):
        from magnelio._fields.field_arrays import FieldState

        Nx, Ny, Nz = 4, 5, 6
        f = FieldState.zeros(Nx, Ny, Nz)
        assert f.Ex.shape == (Nx, Ny + 1, Nz + 1)
        assert f.Ey.shape == (Nx + 1, Ny, Nz + 1)
        assert f.Ez.shape == (Nx + 1, Ny + 1, Nz)
        assert f.Hx.shape == (Nx + 1, Ny, Nz)
        assert f.Hy.shape == (Nx, Ny + 1, Nz)
        assert f.Hz.shape == (Nx, Ny, Nz + 1)

    def test_all_zeros_initially(self):
        from magnelio._fields.field_arrays import FieldState

        f = FieldState.zeros(3, 3, 3)
        for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            assert np.all(getattr(f, comp) == 0.0)


class TestEnergyStopSolver:
    """Tests for FITTimeDomainSolver energy_stop_db early termination (DD-019)."""

    def _make_solver(self, n_steps=500, energy_stop_db=None):
        from magnelio.mesh.grid import GridLines
        from magnelio.mesh.mesher import Mesh
        from magnelio.solver.fit_td import FITTimeDomainSolver
        from magnelio.solver.stability import courant_dt

        grid = GridLines(
            x=np.linspace(0, 5e-3, 6),
            y=np.linspace(0, 5e-3, 6),
            z=np.linspace(0, 5e-3, 6),
        )
        mesh = Mesh.from_grid(grid)
        dt = courant_dt(grid)
        return FITTimeDomainSolver(
            mesh=mesh,
            total_time_steps=n_steps,
            dt=dt,
            verbose=False,
            energy_stop_db=energy_stop_db,
        )

    def test_energy_stop_db_none_runs_all_steps(self):
        """Without energy_stop_db the solver runs exactly total_time_steps."""
        solver = self._make_solver(n_steps=50, energy_stop_db=None)
        solver.setup()
        solver.run()
        assert solver._actual_steps == 50

    def test_energy_stop_db_attribute_exists(self):
        """energy_stop_db parameter is accepted and stored."""
        solver = self._make_solver(energy_stop_db=25.0)
        assert solver.energy_stop_db == 25.0

    def test_peak_energy_set_after_run(self):
        """_peak_energy is set after run() (even for zero-field run)."""
        solver = self._make_solver(n_steps=100)
        solver.setup()
        solver.run()
        assert isinstance(solver._peak_energy, float)
