"""Unit gates for the ADE dispersion operator (DD-084).

Covers the exact reduction paths (no dispersive material → bit-identical
coefficients; the Drude DC pole → the semi-implicit conductor), the
staircase edge-subset construction, the eps_inf CFL convention, and the
same-ops checkpoint rewind (WP-S6 pattern).
"""

from __future__ import annotations

import numpy as np
import pytest

import magnelio.solver._dispersion as solver_dispersion
from magnelio import Material, Mesh
from magnelio._operators.material_matrices import (
    EPS0,
    build_M_eps,
    build_M_sigma,
)
from magnelio.materials import DispersionModel
from magnelio.mesh.grid import GridLines
from magnelio.solver._dispersion import DispersionOperator
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import compute_min_effective_eps

# DD-103: the closure these fixtures always assumed.  A face
# with no BC used to evolve under the free curl operator —
# which IS the natural magnetic wall, hence "PMC".
_BC_OPEN = {
    "xmin": "PMC",
    "xmax": "PMC",
    "ymin": "PMC",
    "ymax": "PMC",
    "zmin": "PMC",
    "zmax": "PMC",
}

DT = 1e-12


def _grid(n=6):
    lin = np.linspace(0.0, n * 1e-3, n + 1)
    return GridLines(x=lin, y=lin, z=lin)


def _debye_mat(name="disp", eps_inf=2.0, delta_eps=1.0, tau=1e-11, sigma=0.0):
    return Material.dispersive(
        name,
        DispersionModel.debye(eps_inf, delta_eps, tau),
        sigma=sigma,
    )


def _solver(mesh, steps=None):
    return FITTimeDomainSolver(
        mesh=mesh,
        boundary_conditions={},
        dt=DT,
        total_time_steps=steps,
        verbose=False,
    )


class TestOperatorConstruction:
    def test_none_without_dispersive_material(self):
        mesh = Mesh.from_grid(_grid(), background=Material.air(), boundary_conditions=_BC_OPEN)
        assert DispersionOperator.from_mesh(mesh, DT) is None

    def test_edge_subset_single_cell(self):
        """One dispersive cell in a 2x2x2 grid: the clamped one-sided
        lookup owns exactly one edge per component — hand-checkable."""
        lin = np.linspace(0.0, 2e-3, 3)
        mesh = Mesh.from_grid(
            GridLines(x=lin, y=lin, z=lin),
            regions=[(_debye_mat(), (0.0, 0.0, 0.0, 1e-3, 1e-3, 1e-3))],
            boundary_conditions=_BC_OPEN,
        )
        op = DispersionOperator.from_mesh(mesh, DT)
        assert len(op.blocks) == 1
        idx = np.sort(op.blocks[0].idx)
        n_Ex = 2 * 3 * 3
        n_Ey = 3 * 2 * 3
        # Ex edge (0,0,0); Ey edge (0,0,0); Ez edge (0,0,0) in flat order.
        expected = np.array([0, n_Ex + 0, n_Ex + n_Ey + 0])
        np.testing.assert_array_equal(idx, expected)

    def test_coupling_matches_M_eps_geometry(self):
        """On a homogeneous dispersive fill, g must equal M_eps / eps_inf
        edge for edge (same geometry factor, same staircase sampling)."""
        mesh = Mesh.from_grid(
            _grid(), background=_debye_mat(eps_inf=2.0), boundary_conditions=_BC_OPEN
        )
        op = DispersionOperator.from_mesh(mesh, DT)
        (block,) = op.blocks
        M_eps = build_M_eps(mesh)
        np.testing.assert_allclose(
            block.g,
            M_eps[block.idx] / 2.0,
            rtol=1e-15,
        )

    def test_w_positive_for_passive_model(self):
        mesh = Mesh.from_grid(_grid(), background=_debye_mat(), boundary_conditions=_BC_OPEN)
        op = DispersionOperator.from_mesh(mesh, DT)
        assert (op.W[op.blocks[0].idx] > 0).all()


class TestCoefficientReduction:
    def test_no_dispersion_bit_identical(self):
        """Coefficient arrays without dispersive materials are the exact
        master expressions — no epsilon-perturbation from the new path."""
        mesh = Mesh.from_grid(
            _grid(),
            background=Material.from_isotropic("d", 2.0, sigma=0.05),
            boundary_conditions=_BC_OPEN,
        )
        s = _solver(mesh, steps=1)
        s.setup()
        assert s._dispersion is None
        M_eps, M_sigma = build_M_eps(mesh), build_M_sigma(mesh)
        denom = M_eps + 0.5 * DT * M_sigma
        np.testing.assert_array_equal(
            s._alpha_E,
            (M_eps - 0.5 * DT * M_sigma) / denom,
        )
        np.testing.assert_array_equal(s._beta_E, DT / denom)

    def test_drude_dc_pole_equals_conductor(self):
        """A pure DC pole (a=0, r=sigma/eps0) IS the semi-implicit
        conductor: with e^0 = 0 the trapezoidal state J^n = W e^n is
        exact by induction, so the marched field matches the
        sigma-material run to roundoff.  (Seeding H, not E — the ADE
        integrates dE/dt, so a nonzero e^0 would be a deliberate
        initial-polarisation offset, not a conductor.)"""
        sigma = 0.05

        def march(mat, n_steps=200):
            mesh = Mesh.from_grid(_grid(), background=mat, boundary_conditions=_BC_OPEN)
            s = _solver(mesh, steps=n_steps)
            s.setup()
            rng = np.random.default_rng(7)
            s._fields.h_flat[:] = rng.standard_normal(s._fields.h_flat.size)
            s.run()
            return s._fields.e_flat.copy(), s._fields.h_flat.copy()

        e_sig, h_sig = march(Material.from_isotropic("c", 2.0, sigma=sigma))
        dc = DispersionModel(
            eps_inf=2.0,
            poles=((complex(0.0), complex(sigma / EPS0)),),
            f_band=(1e8, 1e10),
        )
        e_ade, h_ade = march(Material.dispersive("d", dc))

        scale = np.abs(e_sig).max()
        assert np.abs(e_ade - e_sig).max() < 1e-10 * scale
        assert np.abs(h_ade - h_sig).max() < 1e-10 * np.abs(h_sig).max()


class TestCheckpointRewind:
    def test_same_ops_rewind_bit_exact(self):
        """WP-S6 pattern: march, checkpoint, march on, rewind, march the
        same ops again — bit-identical including the complex pole states
        (Lorentz pair) and a real-pole channel (Debye)."""
        model = DispersionModel.lorentz(
            2.0,
            0.5,
            2 * np.pi * 5e9,
            2 * np.pi * 5e8,
        )
        mat = Material.dispersive("mix", model, sigma=0.01)
        mesh = Mesh.from_grid(
            _grid(),
            regions=[
                (mat, (0.0, 0.0, 0.0, 3e-3, 6e-3, 6e-3)),
                (_debye_mat("real_poles"), (3e-3, 0.0, 0.0, 6e-3, 6e-3, 6e-3)),
            ],
            boundary_conditions=_BC_OPEN,
        )
        s = _solver(mesh, steps=40)
        s.setup()
        assert len(s._dispersion.blocks) == 2
        rng = np.random.default_rng(3)
        s._fields.e_flat[:] = rng.standard_normal(s._fields.e_flat.size)
        s.run()

        sd = s.state_dict()
        assert "dispersion" in sd
        s.total_time_steps = 80
        s.run()
        e_ref = s._fields.e_flat.copy()
        h_ref = s._fields.h_flat.copy()
        j_ref = {
            k: {kk: vv.copy() for kk, vv in v.items()}
            for k, v in s.state_dict()["dispersion"].items()
        }

        s.load_state_dict(sd)
        s.run()
        np.testing.assert_array_equal(s._fields.e_flat, e_ref)
        np.testing.assert_array_equal(s._fields.h_flat, h_ref)
        for k, v in s.state_dict()["dispersion"].items():
            for kk, vv in v.items():
                np.testing.assert_array_equal(vv, j_ref[k][kk])


class TestFusedADEKernels:
    """Gates for the Numba-fused ADE hooks (Workstream 2, session 111).

    The fused kernels replace the NumPy subset arithmetic on the CPU
    backend only; the NumPy branch remains the fallback (and the CuPy
    path).  Equality contract, verified here:

    - REAL poles (Debye, Drude, DS, overdamped Lorentz — every
      constitutive op is float64): fused vs NumPy **bit-identical**.
    - CONJUGATE-PAIR poles: the fused kernel equals the strict-IEEE
      scalar reference **bit-identically**; NumPy's own complex-multiply
      ufunc is FMA-contracted on SIMD builds (measured 1-ULP deviations
      from the scalar reference on this machine), so fused-vs-NumPy is
      gated at the rounding level instead — the difference is a
      different *rounding* of the same expression, machine-dependent on
      the NumPy side, not an algorithm change.
    """

    @staticmethod
    def _march(mat, fused, n_steps=150, seed=11):
        """Run with the fused path forced on/off.  The size threshold
        is zeroed so the small test subsets exercise the parallel
        kernel instead of falling back to the NumPy branch."""
        prev = solver_dispersion.HAS_NUMBA
        prev_thresh = solver_dispersion._PARALLEL_MIN_STATES
        solver_dispersion.HAS_NUMBA = fused and prev
        solver_dispersion._PARALLEL_MIN_STATES = 0
        try:
            mesh = Mesh.from_grid(
                _grid(8),
                background=Material.air(),
                regions=[(mat, (2e-3, 2e-3, 2e-3, 6e-3, 6e-3, 6e-3))],
                boundary_conditions=_BC_OPEN,
            )
            s = _solver(mesh, steps=n_steps)
            s.setup()
            rng = np.random.default_rng(seed)
            s._fields.e_flat[:] = rng.standard_normal(s._fields.e_flat.size) * 1e-3
            s.run()
            ops = [op for op in (s._dispersion, s._dispersion_mu) if op is not None]
            js = [p.J.copy() for op in ops for b in op.blocks for p in b.poles]
            return s._fields.e_flat.copy(), s._fields.h_flat.copy(), js
        finally:
            solver_dispersion.HAS_NUMBA = prev
            solver_dispersion._PARALLEL_MIN_STATES = prev_thresh

    @pytest.mark.skipif(not solver_dispersion.HAS_NUMBA, reason="numba not installed")
    def test_real_pole_eps_bit_identical(self):
        mat = _debye_mat(delta_eps=[0.5, 0.3], tau=[1e-11, 3e-11])
        e_n, h_n, j_n = self._march(mat, fused=False)
        e_f, h_f, j_f = self._march(mat, fused=True)
        np.testing.assert_array_equal(e_f, e_n)
        np.testing.assert_array_equal(h_f, h_n)
        for a, b in zip(j_f, j_n):
            np.testing.assert_array_equal(a, b)

    @pytest.mark.skipif(not solver_dispersion.HAS_NUMBA, reason="numba not installed")
    def test_real_pole_mu_bit_identical(self):
        mat = Material.dispersive_mu(
            "md",
            DispersionModel.debye(1.0, 0.5, 1e-11),
            epsilon=2.0,
        )
        e_n, h_n, j_n = self._march(mat, fused=False)
        e_f, h_f, j_f = self._march(mat, fused=True)
        np.testing.assert_array_equal(e_f, e_n)
        np.testing.assert_array_equal(h_f, h_n)
        for a, b in zip(j_f, j_n):
            np.testing.assert_array_equal(a, b)

    @pytest.mark.skipif(not solver_dispersion.HAS_NUMBA, reason="numba not installed")
    def test_conjugate_pair_strict_ieee_scalar_reference(self, monkeypatch):
        """The fused kernel's complex-pole arithmetic equals a pure-
        Python (strict IEEE, no FMA) implementation bit-identically."""
        monkeypatch.setattr(solver_dispersion, "_PARALLEL_MIN_STATES", 0)
        mat = Material.dispersive(
            "lor",
            DispersionModel.lorentz(2.0, 0.4, 2.0 * np.pi * 30e9, 5e9),
        )
        mesh = Mesh.from_grid(
            _grid(4),
            background=Material.air(),
            regions=[(mat, (1e-3, 1e-3, 1e-3, 3e-3, 3e-3, 3e-3))],
            boundary_conditions=_BC_OPEN,
        )
        op = DispersionOperator.from_mesh(mesh, DT)
        n_states = op.W.size
        rng = np.random.default_rng(5)
        beta = rng.random(n_states) * DT
        op.bind(beta, np)
        assert op.blocks[0].fused is not None

        b = op.blocks[0]
        (p,) = b.poles
        idx = np.asarray(b.idx)
        k, c = complex(p.k), complex(p.c)
        hw, opk = 0.5 * p.weight, 1.0 + k
        J_ref = [complex(0.0)] * idx.size
        f = rng.standard_normal(n_states) * 1e-3

        for _ in range(20):
            f_prev_ref = [float(f[j]) for j in idx]
            op.save_field(f)
            # Scalar reference of the update on a copy of f
            f_ref = f.copy()
            for i, j in enumerate(idx):
                jh = hw * (opk * J_ref[i]).real
                f_new = float(f_ref[j]) - float(b.beta[i]) * jh
                f_ref[j] = f_new
                gd = float(b.g[i]) * (f_new - f_prev_ref[i])
                J_ref[i] = J_ref[i] * k + c * gd
            op.update_field(f)
            np.testing.assert_array_equal(f, f_ref)
            np.testing.assert_array_equal(
                b.poles[0].J,
                np.array(J_ref, dtype=np.complex128),
            )
            f += rng.standard_normal(n_states) * 1e-4  # fresh "curl" input

    @pytest.mark.skipif(not solver_dispersion.HAS_NUMBA, reason="numba not installed")
    def test_mixed_model_matches_numpy_to_rounding(self):
        """Interleaved real + conjugate-pair poles (vector-fit shape):
        fused vs NumPy agree to rounding; bitwise equality is not
        required because NumPy's complex multiply is FMA-contracted
        (see class docstring)."""
        debye = DispersionModel.debye(2.0, [0.5, 0.3], [1e-11, 3e-11])
        lorentz = DispersionModel.lorentz(2.0, 0.4, 2.0 * np.pi * 30e9, 5e9)
        model = DispersionModel(
            eps_inf=2.0,
            poles=(debye.poles[0], lorentz.poles[0], debye.poles[1]),
            f_band=debye.f_band,
        )
        mat = Material.dispersive("mixed", model)
        e_n, h_n, j_n = self._march(mat, fused=False)
        e_f, h_f, j_f = self._march(mat, fused=True)
        scale = np.abs(e_n).max()
        assert np.abs(e_f - e_n).max() < 1e-13 * scale
        assert np.abs(h_f - h_n).max() < 1e-13 * np.abs(h_n).max()

    @pytest.mark.skipif(not solver_dispersion.HAS_NUMBA, reason="numba not installed")
    def test_small_blocks_stay_on_numpy_branch(self):
        """Blocks below _PARALLEL_MIN_STATES keep b.fused = None — the
        NumPy branch is faster there (the parallel win drowns in the
        per-region thread wake/join; a serial fused variant was
        measured and rejected, see the kernel comment)."""
        mat = _debye_mat()
        mesh = Mesh.from_grid(
            _grid(4),
            background=Material.air(),
            regions=[(mat, (1e-3, 1e-3, 1e-3, 3e-3, 3e-3, 3e-3))],
            boundary_conditions=_BC_OPEN,
        )
        op = DispersionOperator.from_mesh(mesh, DT)
        op.bind(np.full(op.W.size, DT), np)
        assert all(b.fused is None for b in op.blocks)

    @pytest.mark.skipif(not solver_dispersion.HAS_NUMBA, reason="numba not installed")
    def test_pole_state_views_stay_checkpoint_coherent(self, monkeypatch):
        """After bind(), each p.J is a row view into the fused stack —
        load_state_dict must reach the arrays the kernel reads."""
        monkeypatch.setattr(solver_dispersion, "_PARALLEL_MIN_STATES", 0)
        mat = _debye_mat(delta_eps=[0.5, 0.3], tau=[1e-11, 3e-11])
        mesh = Mesh.from_grid(
            _grid(4),
            background=Material.air(),
            regions=[(mat, (1e-3, 1e-3, 1e-3, 3e-3, 3e-3, 3e-3))],
            boundary_conditions=_BC_OPEN,
        )
        op = DispersionOperator.from_mesh(mesh, DT)
        op.bind(np.full(op.W.size, DT), np)
        b = op.blocks[0]
        Jr = b.fused[9]
        for q, p in enumerate(b.poles):
            assert np.shares_memory(p.J, Jr)
        state = {
            str(b.mat_id): {
                "J0": np.full(b.idx.size, 1.5),
                "J1": np.full(b.idx.size, -2.5),
            }
        }
        op.load_state_dict(state)
        np.testing.assert_array_equal(Jr[0], 1.5)
        np.testing.assert_array_equal(Jr[1], -2.5)


class TestCFLConvention:
    def test_eps_inf_drives_cfl(self):
        """The dispersive material's CFL input is its eps_inf (the
        high-frequency wave speed), identical to a static eps_inf fill —
        pole strength must not enter (DD-084)."""
        box = (0.0, 0.0, 0.0, 3e-3, 6e-3, 6e-3)
        m_disp = Mesh.from_grid(
            _grid(),
            regions=[(_debye_mat(eps_inf=2.0, delta_eps=50.0), box)],
            boundary_conditions=_BC_OPEN,
        )
        m_stat = Mesh.from_grid(
            _grid(),
            regions=[(Material.from_isotropic("s", epsilon=2.0), box)],
            boundary_conditions=_BC_OPEN,
        )
        assert compute_min_effective_eps(m_disp) == compute_min_effective_eps(m_stat)
