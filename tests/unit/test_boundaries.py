"""Unit tests for boundary conditions (PEC, PMC, CPML, Periodic)."""

import numpy as np
import pytest

from magnelio._fields.field_arrays import FieldState
from magnelio.mesh.grid import GridLines


def _grid(Nx=5, Ny=5, Nz=5):
    return GridLines(
        x=np.linspace(0, 1e-2, Nx + 1),
        y=np.linspace(0, 1e-2, Ny + 1),
        z=np.linspace(0, 1e-2, Nz + 1),
    )


def _fields(Nx=5, Ny=5, Nz=5):
    f = FieldState.zeros(Nx, Ny, Nz)
    f.Ex[:] = 1.0
    f.Ey[:] = 1.0
    f.Ez[:] = 1.0
    f.Hx[:] = 1.0
    f.Hy[:] = 1.0
    f.Hz[:] = 1.0
    return f


class TestPECBoundary:
    def test_xmin_zeros_tangential_E(self):
        from magnelio.boundaries.pec import PECBoundary

        bc = PECBoundary("xmin")
        fields = _fields()
        bc.apply(fields)
        np.testing.assert_array_equal(fields.Ey[0, :, :], 0.0)
        np.testing.assert_array_equal(fields.Ez[0, :, :], 0.0)
        # Normal component Ex should be unchanged
        assert np.all(fields.Ex != 0.0)

    def test_zmax_zeros_tangential_E(self):
        from magnelio.boundaries.pec import PECBoundary

        bc = PECBoundary("zmax")
        fields = _fields()
        bc.apply(fields)
        np.testing.assert_array_equal(fields.Ex[:, :, -1], 0.0)
        np.testing.assert_array_equal(fields.Ey[:, :, -1], 0.0)

    def test_invalid_face_raises(self):
        from magnelio.boundaries.pec import PECBoundary

        bc = PECBoundary("invalid")
        fields = _fields()
        with pytest.raises(ValueError, match="Unknown face"):
            bc.apply(fields)


class TestPMCBoundary:
    def test_natural_bc_leaves_fields_untouched(self):
        # PMC is the natural BC of the free FIT operators (magnetic
        # wall dy/2 outside the outermost grid line); the runtime
        # object is a face-coverage marker and must not touch fields.
        from magnelio.boundaries.pmc import PMCBoundary

        grid = _grid()
        bc = PMCBoundary("ymin", grid)
        fields = _fields()
        bc.apply_E(fields)
        bc.apply_H(fields)
        for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            np.testing.assert_array_equal(getattr(fields, comp), 1.0)

    def test_invalid_face_raises(self):
        from magnelio.boundaries.pmc import PMCBoundary

        grid = _grid()
        with pytest.raises(ValueError, match="Unknown face"):
            PMCBoundary("invalid", grid)


class TestCPMLBoundary:
    def test_initialization(self):
        from magnelio.boundaries.cpml import CPMLBoundary

        grid = _grid(Nx=20, Ny=20, Nz=20)
        bc = CPMLBoundary("zmax", grid, thickness_cells=8)
        assert not bc.is_initialized
        bc.initialize(dt=1e-12)
        assert bc.is_initialized
        assert bc._b is not None
        assert len(bc._b) == min(8, 20)

    def test_sigma_profile_monotonic(self):
        from magnelio.boundaries.cpml import CPMLBoundary

        grid = _grid(Nx=20, Ny=20, Nz=20)
        bc = CPMLBoundary("zmax", grid, thickness_cells=8)
        bc.initialize(dt=1e-12)
        # b should be in (0, 1] — absorption factor
        assert np.all(bc._b > 0)
        assert np.all(bc._b <= 1.0)

    def test_repr(self):
        from magnelio.boundaries.cpml import CPMLBoundary

        grid = _grid()
        bc = CPMLBoundary("xmin", grid)
        assert "xmin" in repr(bc)


class TestPeriodicBoundary:
    def test_invalid_axis(self):
        from magnelio.boundaries.periodic import PeriodicBoundary

        grid = _grid()
        with pytest.raises(ValueError, match="axis must be"):
            PeriodicBoundary("w", grid)

    def test_apply_E_x(self):
        from magnelio.boundaries.periodic import PeriodicBoundary

        grid = _grid()
        bc = PeriodicBoundary("x", grid)
        fields = _fields()
        fields.Ey[-2, :, :] = 5.0
        bc.apply_E(fields)
        np.testing.assert_array_equal(fields.Ey[0, :, :], 5.0)


class TestMaterializeBoundary:
    def test_all_types(self):
        from magnelio.boundaries.boundary_conditions import (
            materialize_boundary,
        )
        from magnelio.boundaries.cpml import CPMLBoundary
        from magnelio.boundaries.pec import PECBoundary
        from magnelio.boundaries.periodic import PeriodicBoundary
        from magnelio.boundaries.pmc import PMCBoundary

        grid = _grid()
        assert isinstance(
            materialize_boundary("ymin", "PEC", grid),
            PECBoundary,
        )
        assert isinstance(
            materialize_boundary("ymax", "PMC", grid),
            PMCBoundary,
        )
        cpml = materialize_boundary(
            "zmax",
            "CPML",
            grid,
            cpml_thickness_cells=4,
        )
        assert isinstance(cpml, CPMLBoundary)
        assert cpml.thickness_cells == 4
        assert isinstance(
            materialize_boundary("xmin", "Periodic", grid),
            PeriodicBoundary,
        )

    def test_unknown_type_raises(self):
        from magnelio.boundaries.boundary_conditions import (
            materialize_boundary,
        )

        grid = _grid()
        with pytest.raises(ValueError, match="Mur"):
            materialize_boundary("ymin", "Mur", grid)


class TestCPMLPortWindows:
    """DD-198: the absorber is switched off behind waveguide-port windows."""

    def _bc(self, face="xmin"):
        from magnelio.boundaries.cpml import CPMLBoundary

        grid = _grid(Nx=12, Ny=10, Nz=8)
        bc = CPMLBoundary(face, grid, thickness_cells=4)
        bc.initialize(dt=1e-12)
        return bc

    def test_no_windows_keeps_the_broadcast_coefficients(self):
        bc = self._bc()
        bc.set_port_windows([])
        assert bc._c_E1 is bc._c_3d and bc._ck_H2 is bc._ck_3d

    def test_footprint_zeroes_c_and_ck_over_the_full_depth(self):
        bc = self._bc("xmin")
        # Window: y nodes 3..6, z nodes 2..5 (inclusive node windows).
        bc.set_port_windows([{1: (3, 6), 2: (2, 5)}])
        c = np.asarray(bc._c_E1)  # psi_Ey: (n, Ny, Nz+1) — cell-sampled along y
        assert c.shape == (4, 10, 9)
        assert np.all(c[:, 3:6, 2:6] == 0.0)
        assert np.all(c[:, :3, :] == np.asarray(bc._c_3d)[:, :, :])
        assert np.all(c[:, 6:, :] == np.asarray(bc._c_3d)[:, :, :])
        ck = np.asarray(bc._ck_E2)  # psi_Ez: (n, Ny+1, Nz) — node-sampled along y
        assert np.all(ck[:, 3:7, 2:5] == 0.0)
        assert np.all(ck[:, :3, :] == np.asarray(bc._ck_3d))
        h1 = np.asarray(bc._c_H1)  # psi_Hy: (n, Ny+1, Nz)
        assert np.all(h1[:, 3:7, 2:5] == 0.0)
        assert np.all(h1[:, 7:, :] == np.asarray(bc._c_3d))

    def test_psi_stays_zero_inside_the_footprint(self):
        bc = self._bc("xmin")
        bc.set_port_windows([{1: (3, 6), 2: (2, 5)}])
        fields = _fields(12, 10, 8)
        rng = np.random.default_rng(0)
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            arr = getattr(fields, name)
            arr[...] = rng.standard_normal(arr.shape)
        n_E = fields.Ex.size + fields.Ey.size + fields.Ez.size
        n_H = fields.Hx.size + fields.Hy.size + fields.Hz.size
        for _ in range(3):
            bc.update_E(fields, np.full(n_E, 1e-3))
            bc.update_H(fields, np.full(n_H, 1e-3))
        assert np.all(bc._psi_Ey[:, 3:6, 2:6] == 0.0)
        assert np.all(bc._psi_Ez[:, 3:7, 2:5] == 0.0)
        assert np.all(bc._psi_Hy[:, 3:7, 2:5] == 0.0)
        assert np.all(bc._psi_Hz[:, 3:6, 2:6] == 0.0)
        assert np.any(bc._psi_Ey[:, :3, :] != 0.0)
