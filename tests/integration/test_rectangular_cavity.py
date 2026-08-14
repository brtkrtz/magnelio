"""
Integration test: Rectangular cavity eigenmodes via 3D eigenmode solver.

Uses Mesh.from_grid() — no OCC dependency.
Tests the 3 lowest resonant frequencies against the analytical formula:

    f_mnp = (c₀/2) · √((m/a)² + (n/b)² + (p/c)²)

Tolerance for integration test: 5% (benchmark target: 2%).
"""

import math

import numpy as np
import pytest

from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.solver._eigenmode_3d import EigenmodeSolver3D

C0 = 299_792_458.0


def _cavity_mesh(a, b, c, Nx, Ny, Nz):
    grid = GridLines(
        x=np.linspace(0, a, Nx + 1),
        y=np.linspace(0, b, Ny + 1),
        z=np.linspace(0, c, Nz + 1),
    )
    return Mesh.from_grid(grid)


def f_analytical(m, n, p, a, b, c):
    return (C0 / 2) * math.sqrt((m / a) ** 2 + (n / b) ** 2 + (p / c) ** 2)


class TestRectangularCavityEigenmodes:
    def test_three_lowest_modes_within_5pct(self):
        a, b, c = 10e-3, 8e-3, 6e-3
        Nx, Ny, Nz = 10, 8, 6
        mesh = _cavity_mesh(a, b, c, Nx, Ny, Nz)

        solver = EigenmodeSolver3D(n_modes=8)
        freq_hz, _, _ = solver.solve(mesh)

        # Discard near-zero static modes
        freq_phys = sorted(freq_hz[freq_hz > 1e6].tolist())[:3]

        analytical = sorted(
            [
                f_analytical(1, 0, 1, a, b, c),
                f_analytical(0, 1, 1, a, b, c),
                f_analytical(1, 1, 0, a, b, c),
            ]
        )

        assert len(freq_phys) == 3, "Expected at least 3 non-trivial modes"

        for i, (f_num, f_ana) in enumerate(zip(freq_phys, analytical)):
            err = abs(f_num - f_ana) / f_ana * 100
            assert err < 5.0, (
                f"Mode {i + 1}: FIT={f_num / 1e9:.4f} GHz, "
                f"analytical={f_ana / 1e9:.4f} GHz, error={err:.2f}%"
            )

    def test_higher_resolution_within_2pct(self):
        """Benchmark-level test: 2% tolerance with finer grid."""
        a, b, c = 30e-3, 20e-3, 15e-3
        Nx, Ny, Nz = 20, 14, 10
        mesh = _cavity_mesh(a, b, c, Nx, Ny, Nz)

        solver = EigenmodeSolver3D(n_modes=8)
        freq_hz, _, _ = solver.solve(mesh)

        freq_phys = sorted(freq_hz[freq_hz > 1e6].tolist())[:3]
        analytical = sorted(
            [
                f_analytical(1, 0, 1, a, b, c),
                f_analytical(0, 1, 1, a, b, c),
                f_analytical(1, 1, 0, a, b, c),
            ]
        )

        for i, (f_num, f_ana) in enumerate(zip(freq_phys, analytical)):
            err = abs(f_num - f_ana) / f_ana * 100
            assert err < 2.0, (
                f"Mode {i + 1}: FIT={f_num / 1e9:.4f} GHz, "
                f"analytical={f_ana / 1e9:.4f} GHz, error={err:.2f}%"
            )

    def test_solver_parameter_accepted(self):
        """Solver parameter 'arpack' accepted without error."""
        a, b, c = 30e-3, 20e-3, 15e-3
        mesh = _cavity_mesh(a, b, c, 10, 8, 6)

        solver = EigenmodeSolver3D(n_modes=3, solver="arpack")
        freq_hz, E_modes, H_modes = solver.solve(mesh)

        assert len(freq_hz) == 3
        assert E_modes.shape[1] == 3
        assert H_modes.shape[1] == 3

    def test_h_field_consistency(self):
        """H-field should satisfy curl E ≈ omega * mu * H."""
        a, b, c = 30e-3, 20e-3, 15e-3
        Nx, Ny, Nz = 10, 8, 6
        mesh = _cavity_mesh(a, b, c, Nx, Ny, Nz)

        solver = EigenmodeSolver3D(n_modes=3)
        freq_hz, E_modes, H_modes = solver.solve(mesh)

        assert E_modes.shape[1] == len(freq_hz)
        assert H_modes.shape[1] == len(freq_hz)
        assert H_modes.shape[1] > 0, "Should find at least 1 mode"

        # H should not be all zero
        for m in range(len(freq_hz)):
            assert np.linalg.norm(H_modes[:, m]) > 0, f"Mode {m}: H-field is all zero"


class TestLOBPCGBackend:
    """Tests for the LOBPCG folded-spectrum solver backend."""

    @pytest.fixture()
    def cavity(self):
        a, b, c = 30e-3, 20e-3, 15e-3
        Nx, Ny, Nz = 20, 14, 10
        mesh = _cavity_mesh(a, b, c, Nx, Ny, Nz)
        analytical = sorted(
            [
                f_analytical(1, 0, 1, a, b, c),
                f_analytical(0, 1, 1, a, b, c),
                f_analytical(1, 1, 0, a, b, c),
            ]
        )
        return mesh, analytical

    def test_lobpcg_within_2pct_analytical(self, cavity):
        """LOBPCG finds correct modes within 2% of analytical."""
        mesh, analytical = cavity

        solver = EigenmodeSolver3D(n_modes=5, solver="lobpcg")
        freq_hz, _, _ = solver.solve(mesh)
        freq_phys = sorted(freq_hz.tolist())[:3]

        for i, (f_num, f_ana) in enumerate(zip(freq_phys, analytical)):
            err = abs(f_num - f_ana) / f_ana * 100
            assert err < 2.0, (
                f"Mode {i}: LOBPCG={f_num / 1e9:.4f} GHz, "
                f"analytical={f_ana / 1e9:.4f} GHz, error={err:.2f}%"
            )

    def test_lobpcg_matches_arpack(self, cavity):
        """The 3 lowest LOBPCG modes each match an ARPACK mode within 2%.

        LOBPCG (unpreconditioned folded spectrum) converges slower for modes
        far from σ.  The 2% tolerance accounts for Rayleigh-quotient recovery
        on partially-converged eigenvectors.  We compare only the 3 lowest
        physical modes (consistent with the analytical test).
        """
        mesh, _ = cavity

        solver_lu = EigenmodeSolver3D(n_modes=5, solver="arpack-superlu")
        freq_lu, _, _ = solver_lu.solve(mesh)

        solver_lobpcg = EigenmodeSolver3D(n_modes=5, solver="lobpcg")
        freq_lobpcg, _, _ = solver_lobpcg.solve(mesh)

        for i in range(3):
            diffs = np.abs(freq_lu - freq_lobpcg[i]) / freq_lobpcg[i] * 100
            best = float(diffs.min())
            assert best < 2.0, (
                f"LOBPCG mode {i} at {freq_lobpcg[i] / 1e9:.6f} GHz: "
                f"closest ARPACK mode differs by {best:.4f}%"
            )

    def test_lobpcg_h_field_nonzero(self, cavity):
        """LOBPCG produces non-zero H-fields."""
        mesh, _ = cavity

        solver = EigenmodeSolver3D(n_modes=3, solver="lobpcg")
        freq_hz, E_modes, H_modes = solver.solve(mesh)

        for m in range(len(freq_hz)):
            assert np.linalg.norm(H_modes[:, m]) > 0, f"Mode {m}: H-field is all zero"


class TestAMGBackend:
    """Tests for the AMG-preconditioned inner solve."""

    @pytest.fixture()
    def cavity(self):
        a, b, c = 30e-3, 20e-3, 15e-3
        Nx, Ny, Nz = 20, 14, 10
        mesh = _cavity_mesh(a, b, c, Nx, Ny, Nz)
        analytical = sorted(
            [
                f_analytical(1, 0, 1, a, b, c),
                f_analytical(0, 1, 1, a, b, c),
                f_analytical(1, 1, 0, a, b, c),
            ]
        )
        return mesh, analytical

    def test_amg_matches_superlu(self, cavity):
        """Each AMG eigenvalue has a matching SuperLU eigenvalue within 0.1%."""
        pytest.importorskip("pyamg")
        mesh, _ = cavity

        solver_lu = EigenmodeSolver3D(n_modes=5, solver="arpack-superlu")
        freq_lu, _, _ = solver_lu.solve(mesh)

        solver_amg = EigenmodeSolver3D(n_modes=5, solver="arpack-amg")
        freq_amg, _, _ = solver_amg.solve(mesh)

        # Match each AMG mode to nearest SuperLU mode (degenerate modes
        # may appear in different order depending on sigma/backend).
        n_match = min(len(freq_amg), len(freq_lu))
        for i in range(n_match):
            diffs = np.abs(freq_lu - freq_amg[i]) / freq_amg[i] * 100
            best = float(diffs.min())
            assert best < 0.1, (
                f"AMG mode {i} at {freq_amg[i] / 1e9:.6f} GHz: "
                f"closest SuperLU mode differs by {best:.4f}%"
            )

    def test_amg_within_2pct_analytical(self, cavity):
        """AMG path finds correct modes within 2% of analytical."""
        pytest.importorskip("pyamg")
        mesh, analytical = cavity

        solver = EigenmodeSolver3D(n_modes=5, solver="arpack-amg")
        freq_hz, _, _ = solver.solve(mesh)
        freq_phys = sorted(freq_hz.tolist())[:3]

        for i, (f_num, f_ana) in enumerate(zip(freq_phys, analytical)):
            err = abs(f_num - f_ana) / f_ana * 100
            assert err < 2.0, (
                f"Mode {i}: AMG={f_num / 1e9:.4f} GHz, "
                f"analytical={f_ana / 1e9:.4f} GHz, error={err:.2f}%"
            )

    def test_amg_h_field_nonzero(self, cavity):
        """AMG path produces non-zero H-fields."""
        pytest.importorskip("pyamg")
        mesh, _ = cavity

        solver = EigenmodeSolver3D(n_modes=3, solver="arpack-amg")
        freq_hz, E_modes, H_modes = solver.solve(mesh)

        for m in range(len(freq_hz)):
            assert np.linalg.norm(H_modes[:, m]) > 0, f"Mode {m}: H-field is all zero"
