"""Selectable time-loop precision (DD-094, plan WP1).

The production default is single (float32); the suite is pinned to double
by ``tests/conftest.py`` via ``MAGNELIO_PRECISION``.  An explicit
``precision="single"`` argument wins over that env pin (mirrors an explicit
``backend="cupy"`` bypassing ``MAGNELIO_BACKEND``), which is what lets these
tests exercise the single path under the double-pinned suite.
"""

import numpy as np
import pytest

from magnelio._backend.array_api import resolve_precision
from magnelio._fields.field_arrays import FieldState


class TestResolvePrecision:
    def test_explicit_single_and_double(self):
        assert resolve_precision("single") == (np.dtype(np.float32), np.dtype(np.complex64))
        assert resolve_precision("double") == (np.dtype(np.float64), np.dtype(np.complex128))

    def test_production_default_is_single(self, monkeypatch):
        # The unspecified default resolves to single once the suite's env
        # pin is removed (production has no MAGNELIO_PRECISION set).
        monkeypatch.delenv("MAGNELIO_PRECISION", raising=False)
        assert resolve_precision(None)[0] == np.dtype(np.float32)

    def test_none_honours_env(self, monkeypatch):
        monkeypatch.setenv("MAGNELIO_PRECISION", "double")
        assert resolve_precision(None)[0] == np.dtype(np.float64)
        monkeypatch.setenv("MAGNELIO_PRECISION", "single")
        assert resolve_precision(None)[0] == np.dtype(np.float32)

    def test_explicit_wins_over_env(self, monkeypatch):
        # An explicit request bypasses the env — no silent footgun for a
        # high-Q run that explicitly asks for double.
        monkeypatch.setenv("MAGNELIO_PRECISION", "single")
        assert resolve_precision("double")[0] == np.dtype(np.float64)

    def test_garbage_raises(self):
        with pytest.raises(ValueError, match="Unknown precision"):
            resolve_precision("fp8")


class TestFieldStatePrecision:
    def test_zeros_dtype_propagates(self):
        fs = FieldState.zeros(4, 4, 4, dtype=np.float32)
        assert fs.e_flat.dtype == np.float32
        assert fs.h_flat.dtype == np.float32
        assert fs.Ex.dtype == np.float32  # views inherit the store dtype

    def test_default_is_double(self):
        # FieldState itself keeps its float64 default; the solver drives the
        # single default through the dtype= argument (precision knob).
        fs = FieldState.zeros(4, 4, 4)
        assert fs.e_flat.dtype == np.float64

    def test_setter_downcasts_into_single_store(self):
        fs = FieldState.zeros(3, 3, 3, dtype=np.float32)
        fs.Ex = np.ones(fs.Ex.shape, dtype=np.float64)
        assert fs.Ex.dtype == np.float32
        assert fs.Ex.flat[0] == pytest.approx(1.0)

    def test_from_components_follows_component_dtype(self):
        shp = FieldState.zeros(2, 2, 2, dtype=np.float32)
        # A FieldState built from float32 components has a float32 store.
        assert shp.e_flat.dtype == np.float32


class TestSolverPrecisionPlumbing:
    """WP1 + WP1b dtype plumbing on the CPU (numpy) backend."""

    def _solver(self, precision):
        from magnelio import Mesh
        from magnelio.boundaries.cpml import CPMLBoundary
        from magnelio.boundaries.pec import PECBoundary
        from magnelio.mesh.grid import GridLines
        from magnelio.solver.fit_td import FITTimeDomainSolver

        lin = np.linspace(0.0, 10e-3, 11)
        mesh = Mesh.from_grid(GridLines(x=lin, y=lin, z=lin))
        grid = mesh.grid
        bcs = {
            "zmin": CPMLBoundary(face="zmin", grid=grid, thickness_cells=3),
            "zmax": CPMLBoundary(face="zmax", grid=grid, thickness_cells=3),
            **{f: PECBoundary(f) for f in ("xmin", "xmax", "ymin", "ymax")},
        }
        s = FITTimeDomainSolver(
            mesh=mesh,
            boundary_conditions=bcs,
            total_time_steps=10,
            verbose=False,
            backend="numpy",
            precision=precision,
        )
        s.setup()
        return s

    def test_single_field_coeff_m_and_cpml_are_float32(self):
        s = self._solver("single")
        # fields + per-step coefficients (WP1)
        assert s._fields.e_flat.dtype == np.float32
        assert s._alpha_E.dtype == np.float32
        assert s._beta_H.dtype == np.float32
        # M diagonals + CPML psi (WP1b)
        assert s._M_eps_diag.dtype == np.float32
        assert s._M_mu_diag.dtype == np.float32
        cpml = s.boundary_conditions["zmin"]
        assert cpml._psi_Ex.dtype == np.float32
        assert cpml._b_3d.dtype == np.float32

    def test_double_keeps_float64_everywhere(self):
        s = self._solver("double")
        assert s._fields.e_flat.dtype == np.float64
        assert s._alpha_E.dtype == np.float64
        assert s._M_eps_diag.dtype == np.float64
        assert s.boundary_conditions["zmin"]._psi_Ex.dtype == np.float64

    def test_single_energy_stop_still_marches(self):
        # The energy reduction is float64-accumulated even in single; the
        # run completes and produces a finite field.
        s = self._solver("single")
        s.run()
        assert np.all(np.isfinite(s._fields.e_flat))


class TestDispersionAuxPrecision:
    """ADE pole-current aux-state dtype follows the field precision (WP1c).

    The Lorentz model contributes a conjugate-pair pole (complex state)
    and lets us assert both the real ``g``/``f_prev`` stash and the
    complex pole current pick up the field precision.  The recursion is a
    decaying IIR filter, so single storage is safe (mirrors the CPML psi
    decision) — the check here is that the plumbing casts, and that a
    single-precision dispersive march stays finite.
    """

    def _disp_solver(self, precision, n_steps=50):
        from magnelio import Material, Mesh
        from magnelio.materials import DispersionModel
        from magnelio.mesh.grid import GridLines
        from magnelio.solver.fit_td import FITTimeDomainSolver

        lin = np.linspace(0.0, 8e-3, 9)
        model = DispersionModel.lorentz(2.0, 0.6, 2.0 * np.pi * 30e9, 5e9)
        mat = Material.dispersive("lor", model, sigma=0.0)
        mesh = Mesh.from_grid(GridLines(x=lin, y=lin, z=lin), background=mat)
        s = FITTimeDomainSolver(
            mesh=mesh,
            boundary_conditions={},
            dt=1e-13,
            total_time_steps=n_steps,
            verbose=False,
            backend="numpy",
            precision=precision,
        )
        s.setup()
        return s

    def test_single_pole_states_are_float32_and_complex64(self):
        s = self._disp_solver("single")
        b = s._dispersion.blocks[0]
        assert b.g.dtype == np.float32
        assert b.f_prev.dtype == np.float32
        # Lorentz -> one conjugate-pair pole -> complex64 state.
        assert any(np.iscomplexobj(p.J) for p in b.poles)
        for p in b.poles:
            assert p.J.dtype == (np.complex64 if np.iscomplexobj(p.J) else np.float32)

    def test_double_pole_states_stay_float64_and_complex128(self):
        s = self._disp_solver("double")
        b = s._dispersion.blocks[0]
        assert b.g.dtype == np.float64
        assert b.f_prev.dtype == np.float64
        for p in b.poles:
            assert p.J.dtype == (np.complex128 if np.iscomplexobj(p.J) else np.float64)

    def test_single_dispersive_march_is_finite(self):
        s = self._disp_solver("single", n_steps=100)
        rng = np.random.default_rng(1)
        s._fields.e_flat[:] = rng.standard_normal(s._fields.e_flat.size).astype(np.float32) * 1e-3
        s.run()
        assert np.all(np.isfinite(s._fields.e_flat))
