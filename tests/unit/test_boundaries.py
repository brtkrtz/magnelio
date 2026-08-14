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
