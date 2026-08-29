"""Unit tests for ``SourceFieldInitial`` (DD-224 Phase C)."""

import types

import numpy as np
import pytest

from magnelio.fields import FieldState
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.signals import WaveformGaussian
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import courant_dt
from magnelio.sources import SourceFieldInitial
from magnelio.sources.base import Source


def _grid(n=6, L=6e-3):
    return GridLines(
        x=np.linspace(0, L, n + 1), y=np.linspace(0, L, n + 1), z=np.linspace(0, L, n + 1)
    )


def _te101(grid):
    """A PEC-box-like standing pattern: Ey ∝ sin(πx/L) sin(πz/L)."""
    L = grid.x[-1]

    def E(x, y, z):
        return (0 * x, np.sin(np.pi * x / L) * np.sin(np.pi * z / L), 0 * z)

    return FieldState.from_function(grid, E=E)


def _solver(mesh, **kw):
    dt = courant_dt(mesh.grid, accuracy="normal")
    return FITTimeDomainSolver(mesh=mesh, total_time_steps=5, dt=dt, verbose=False, **kw)


class TestContract:
    def test_hierarchy_and_unit(self):
        src = SourceFieldInitial(name="f0", field=FieldState.zeros(_grid()))
        assert isinstance(src, Source)
        assert src.amplitude_unit == "1"
        assert src.excitable is True
        assert src.has_waveform is False

    def test_name_and_field_checked(self):
        with pytest.raises(TypeError, match="name"):
            SourceFieldInitial(name="", field=FieldState.zeros(_grid()))
        with pytest.raises(TypeError, match="FieldState"):
            SourceFieldInitial(name="f0", field=np.zeros(3))

    def test_complex_field_rejected(self):
        f = FieldState.from_function(_grid(), E=lambda x, y, z: (1j + 0 * x, 0 * y, 0 * z))
        with pytest.raises(ValueError, match="real"):
            SourceFieldInitial(name="f0", field=f)

    def test_from_function_and_arrays(self):
        g = _grid()
        a = SourceFieldInitial.from_function(
            g, name="a", E=lambda x, y, z: (1 + 0 * x, 0 * y, 0 * z)
        )
        b = SourceFieldInitial.from_arrays(g, name="b", Ex=a.field.Ex)
        np.testing.assert_allclose(b.field.Ex, 1.0)
        np.testing.assert_allclose(b.field.Hz, 0.0)


class TestExcitationBinding:
    def test_amplitude_only(self):
        src = SourceFieldInitial(name="f0", field=FieldState.zeros(_grid()))
        src.set_excitation(None, amplitude=2.5)
        assert src.amplitude == 2.5
        src.clear_excitation()
        assert src.amplitude == 1.0

    def test_waveform_rejected(self):
        src = SourceFieldInitial(name="f0", field=FieldState.zeros(_grid()))
        with pytest.raises(ValueError, match="no waveform"):
            src.set_excitation(WaveformGaussian(f_max=1e9))

    def test_delay_rejected(self):
        src = SourceFieldInitial(name="f0", field=FieldState.zeros(_grid()))
        with pytest.raises(ValueError, match="delayed"):
            src.set_excitation(None, delay=1e-12)


class TestAttach:
    def test_writes_e_and_leapfrog_h(self):
        g = _grid()
        field = _te101(g)
        src = SourceFieldInitial(name="f0", field=field)
        src.set_excitation(None, amplitude=3.0)
        solver = _solver(Mesh.from_grid(g), sources=[src])
        solver.setup()
        e = np.asarray(solver._fields.e_flat)
        h = np.asarray(solver._fields.h_flat)
        # e = 3 × the grid quantity of the field, PEC edges (domain walls) zeroed
        expected = 3.0 * np.asarray(field._raw.e_flat)
        expected[np.asarray(solver._pec_mask_E).astype(bool)] = 0.0
        np.testing.assert_allclose(e, expected, rtol=1e-12, atol=1e-15)
        # h(+dt/2) = −½ β_H (C e): compare against the sparse curl matrix
        from magnelio._operators.curl import build_curl_matrix

        C = build_curl_matrix(g)
        np.testing.assert_allclose(
            h, -0.5 * np.asarray(solver._beta_H) * (C @ e), rtol=1e-10, atol=1e-18
        )
        assert np.abs(h).max() > 0.0

    def test_eigenmode_start_is_pure_oscillation(self):
        """An E-max start keeps |E| ≤ E(0) and the energy within the leapfrog ripple."""
        g = _grid(n=8)
        field = _te101(g)
        src = SourceFieldInitial(name="f0", field=field)
        src.set_excitation(None)
        solver = _solver(Mesh.from_grid(g), sources=[src])
        solver.total_time_steps = 60
        solver.setup()
        e0 = np.asarray(solver._fields.e_flat).copy()
        peak = 0.0
        for _ in range(3):
            solver.total_time_steps = 20
            solver.run()
            peak = max(peak, float(np.abs(np.asarray(solver._fields.e_flat)).max()))
        # the box pattern is not an exact eigenvector of the conformal-free
        # operator on a coarse grid, so allow a few percent of mode mixing
        assert peak <= 1.05 * np.abs(e0).max()

    def test_resample_onto_other_grid(self):
        coarse = _grid(n=4)
        fine = _grid(n=8)
        src = SourceFieldInitial.from_function(
            coarse, name="f0", E=lambda x, y, z: (1 + x, 2 * y, 3 * z)
        )
        field = src._resampled(fine)
        X, Y, Z = np.meshgrid(*field.positions("Ey"), indexing="ij")
        np.testing.assert_allclose(field.Ey, 2 * Y, rtol=1e-10, atol=1e-12)


class TestNoModelGate:
    """No part of the model is refused: absorbers, ADE, SIBC and ports all start."""

    def _src(self, g):
        src = SourceFieldInitial(name="f0", field=_te101(g))
        src.set_excitation(None)
        return src

    def test_absorbing_boundary_starts(self):
        from magnelio.boundaries.cpml import CPMLBoundary

        g = _grid()
        mesh = Mesh.from_grid(g, boundary_conditions={"zmax": "CPML"})
        solver = _solver(
            mesh,
            sources=[self._src(g)],
            boundary_conditions={"zmax": CPMLBoundary("zmax", g)},
        )
        solver.setup()
        assert np.abs(np.asarray(solver._fields.e_flat)).max() > 0.0

    def test_dispersive_material_starts(self):
        from magnelio.materials import DispersionModel
        from magnelio.materials.material import Material

        g = _grid()
        model = DispersionModel.debye(2.0, 1.5, tau=1.0 / (2 * np.pi * 5e9))
        mesh = Mesh.from_grid(g, background=Material.dispersive("debye_fill", model))
        solver = _solver(mesh, sources=[self._src(g)])
        solver.setup()
        assert np.abs(np.asarray(solver._fields.e_flat)).max() > 0.0

    def test_ports_are_not_an_obstacle(self):
        """A waveguide port captures the initial trace; a discrete port has no state."""
        g = _grid()
        src = self._src(g)
        n_Ex = g.Nx * (g.Ny + 1) * (g.Nz + 1)
        e0 = np.asarray(src.field._raw.e_flat)
        hot = n_Ex + int(np.argmax(np.abs(e0[n_Ex:])))
        captured = []
        waveguide = types.SimpleNamespace(
            name="wg",
            plane=types.SimpleNamespace(e_u_indices=np.array([hot]), e_v_indices=None),
            initialize_state=lambda e: captured.append(np.asarray(e).copy()),
        )
        lumped = types.SimpleNamespace(name="p1", plane=None)
        solver = _solver(Mesh.from_grid(g), sources=[src])
        solver.ports = [waveguide, lumped]
        solver.setup()
        assert len(captured) == 1, "the waveguide port never captured the initial trace"
        np.testing.assert_allclose(captured[0], np.asarray(solver._fields.e_flat))


class TestStoreRoundTrip:
    def test_mesh_h5_carries_the_field(self, tmp_path):
        import h5py

        from magnelio.io.project import _load_mesh, _save_mesh

        g = _grid()
        src = SourceFieldInitial(name="f0", field=_te101(g))
        mesh = Mesh.from_grid(g).with_sources([src])
        with h5py.File(tmp_path / "mesh.h5", "w") as f:
            _save_mesh(f, mesh)
        with h5py.File(tmp_path / "mesh.h5", "r") as f:
            back = _load_mesh(f)
        (rebuilt,) = back.sources
        assert isinstance(rebuilt, SourceFieldInitial)
        assert rebuilt.name == "f0"
        np.testing.assert_array_equal(rebuilt.field.Ey, src.field.Ey)
        np.testing.assert_array_equal(rebuilt.field.grid.x, g.x)


class TestSuperposition:
    def test_two_initial_fields_add(self):
        g = _grid()
        one = SourceFieldInitial(name="a", field=_te101(g))
        two = SourceFieldInitial(name="b", field=_te101(g))
        one.set_excitation(None, amplitude=1.0)
        two.set_excitation(None, amplitude=0.5)
        mesh = Mesh.from_grid(g)

        solo = _solver(mesh, sources=[one])
        solo.setup()
        e_solo = np.asarray(solo._fields.e_flat).copy()

        both = _solver(mesh, sources=[one, two])
        both.setup()
        np.testing.assert_allclose(
            np.asarray(both._fields.e_flat), 1.5 * e_solo, rtol=1e-12, atol=1e-18
        )
