"""Unit gates for the TD-SIBC wall operator in the solver (WP-D4).

The load-bearing ones are TestGateANoOp (the SIBC_PLAN mandatory
sigma -> inf => PEC structural reduction: a zero impedance must leave
the run BIT-identical to the master PEC path) and
TestScalarReference (the operator + W folding must solve the coupled
implicit face/branch system EXACTLY — the reference solves that system
numerically per step, independent of the closed-form R_inst/k/q
algebra, so a wrong coefficient cannot hide).  Reference: internal
dossier ``investigations/sibc/DERIVATION.md`` §3/§5/§6.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio._operators.material_matrices import build_M_mu, build_M_sigma_m
from magnelio.boundaries.pec import PECBoundary
from magnelio.materials.material import Material
from magnelio.materials.surface_impedance import (
    SurfaceImpedanceFit,
    fit_wall_impedances,
)
from magnelio.mesh._surfaces import (
    SIBCSurface,
    enumerate_sibc_surfaces,
    resolve_wall_conductors,
)
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.solver._sibc import SIBCOperator, SIBCSpec
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import courant_dt

D = 1e-3
DT = 1e-12
_ALL_FACES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")


def _grid(n=8):
    lin = np.arange(n + 1) * D
    return GridLines(x=lin, y=lin, z=lin)


def _fit(c0=0.0, branches=(), sigma=5.8e7):
    """Hand-built ladder (bypasses the NNLS loop for exact coefficients)."""
    return SurfaceImpedanceFit(
        sigma=sigma,
        mu=1.0,
        roughness=None,
        f_lo=1e9,
        f_hi=1e11,
        c0=c0,
        branches=tuple(branches),
        rel_err_re=0.0,
        rel_err_cplx=0.0,
    )


def _brick_mesh(sigma=5.8e7):
    metal = Material.lossy_metal("cu", sigma=sigma)
    return Mesh.from_grid(
        _grid(),
        regions=[(metal, (2 * D, 2 * D, 1 * D, 6 * D, 5 * D, 3 * D))],
    )


def _cavity_spec(mesh, sigma, f_lo=1e9, f_hi=1e11):
    surfs = enumerate_sibc_surfaces(mesh, bc_pec_faces=_ALL_FACES)
    resolved = resolve_wall_conductors(mesh, surfs, sigma=sigma)
    fits = fit_wall_impedances(resolved, f_lo, f_hi)
    return SIBCSpec(surfaces=tuple(surfs), fits=fits)


def _cavity_solver(mesh, spec, steps, dt=None):
    return FITTimeDomainSolver(
        mesh=mesh,
        boundary_conditions={f: PECBoundary(f) for f in _ALL_FACES},
        dt=dt if dt is not None else DT,
        total_time_steps=steps,
        verbose=False,
        sibc=spec,
    )


class _EnergySampler:
    """Minimal monitor recording the instantaneous field energy."""

    def __init__(self, solver):
        self._s = solver
        self.energy: list[float] = []

    def record(self, fields, n, t, dt):
        e, h = fields.e_flat, fields.h_flat
        self.energy.append(
            0.5 * (float((self._s._M_eps_diag * e) @ e) + float((self._s._M_mu_diag * h) @ h))
        )


def _seed_te101(solver, mesh):
    """Seed the TE101-like Ey pattern of the empty box.

    A (near-)solenoidal single-mode seed: white noise would park a
    large share of the energy in curl-free E components that never
    decay in a source-free cavity and floor the rate measurement.
    """
    g = mesh.grid
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    a, c = g.x[-1] - g.x[0], g.z[-1] - g.z[0]
    ey = np.zeros((Nx + 1, Ny, Nz + 1))
    ey[1:-1, :, 1:-1] = (
        np.sin(np.pi * (g.x[1:-1] - g.x[0]) / a)[:, None, None]
        * np.sin(np.pi * (g.z[1:-1] - g.z[0]) / c)[None, None, :]
    )
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    solver._fields.e_flat[n_Ex : n_Ex + n_Ey] = ey.ravel()


class TestConstruction:
    def test_none_on_empty_spec(self):
        spec = SIBCSpec(surfaces=(), fits={})
        assert SIBCOperator.from_spec(spec, _grid(), DT) is None

    def test_none_on_zero_impedance(self):
        """Z identically zero (sigma -> inf) is the structural no-op of
        Gate A: no block, no operator."""
        mesh = _brick_mesh()
        surfs = enumerate_sibc_surfaces(mesh)
        spec = SIBCSpec(surfaces=tuple(surfs), fits={1: _fit()})
        assert SIBCOperator.from_spec(spec, mesh.grid, DT) is None

    def test_missing_fit_raises(self):
        mesh = _brick_mesh()
        surfs = enumerate_sibc_surfaces(mesh)
        spec = SIBCSpec(surfaces=tuple(surfs), fits={})
        with pytest.raises(KeyError, match="no surface-impedance fit"):
            SIBCOperator.from_spec(spec, mesh.grid, DT)

    def test_w_is_g_times_r_inst(self):
        """W accumulates g * (c0 + sum c_p/(1 + b_p dt/2)) on the booked
        states — the DERIVATION.md §3 closed form, hand-checked."""
        mesh = _brick_mesh()
        (surf,) = enumerate_sibc_surfaces(mesh)
        fit = _fit(c0=0.3, branches=((2e10, 0.05), (2e11, 0.12)))
        spec = SIBCSpec(surfaces=(surf,), fits={1: fit})
        op = SIBCOperator.from_spec(spec, mesh.grid, DT)
        r_inst = 0.3 + 0.05 / (1 + 2e10 * DT / 2) + 0.12 / (1 + 2e11 * DT / 2)
        expected = np.zeros(op.W.size)
        np.add.at(expected, surf.state_indices(mesh.grid), surf.g * r_inst)
        np.testing.assert_allclose(op.W, expected, rtol=1e-15)
        assert (op.W[surf.state_indices(mesh.grid)] > 0).all()

    def test_frozen_faces_excluded(self):
        mesh = _brick_mesh()
        (surf,) = enumerate_sibc_surfaces(mesh)
        fit = _fit(c0=0.3, branches=((2e10, 0.05),))
        spec = SIBCSpec(surfaces=(surf,), fits={1: fit})
        idx = surf.state_indices(mesh.grid)
        frozen = np.zeros(build_M_mu(mesh).size, dtype=bool)
        frozen[idx[:3]] = True
        op = SIBCOperator.from_spec(spec, mesh.grid, DT, frozen=frozen)
        (block,) = op.blocks
        assert not np.isin(np.asarray(block.idx), idx[:3]).any()
        assert (op.W[frozen] == 0.0).all()

    def test_pure_c0_fit_has_no_branch_state(self):
        """A branch-free ladder is entirely the W fold — no history."""
        mesh = _brick_mesh()
        (surf,) = enumerate_sibc_surfaces(mesh)
        spec = SIBCSpec(surfaces=(surf,), fits={1: _fit(c0=0.5)})
        op = SIBCOperator.from_spec(spec, mesh.grid, DT)
        assert op is not None
        assert op.blocks[0].branches == []
        assert op.state_dict() == {"1": {}}


class TestGateANoOp:
    def test_zero_impedance_bit_identical_to_pec_path(self):
        """Gate A (SIBC_PLAN / DERIVATION.md §6a): an SIBC run whose
        fitted Z is identically zero takes the structural no-op path and
        is BIT-identical to the master PEC run — coefficients and
        marched fields."""

        def march(spec):
            mesh = _brick_mesh()
            s = FITTimeDomainSolver(
                mesh=mesh,
                boundary_conditions={},
                dt=DT,
                total_time_steps=50,
                verbose=False,
                sibc=spec,
            )
            s.setup()
            rng = np.random.default_rng(11)
            s._fields.e_flat[:] = rng.standard_normal(s._fields.e_flat.size)
            s.run()
            return s

        surfs = enumerate_sibc_surfaces(_brick_mesh())
        spec = SIBCSpec(surfaces=tuple(surfs), fits={1: _fit()})
        s_ref = march(None)
        s_sibc = march(spec)
        assert s_sibc._sibc is None
        np.testing.assert_array_equal(s_sibc._alpha_H, s_ref._alpha_H)
        np.testing.assert_array_equal(s_sibc._beta_H, s_ref._beta_H)
        np.testing.assert_array_equal(
            s_sibc._fields.e_flat,
            s_ref._fields.e_flat,
        )
        np.testing.assert_array_equal(
            s_sibc._fields.h_flat,
            s_ref._fields.h_flat,
        )

    def test_setup_folds_w_into_alpha_beta(self):
        """With a live spec the solver's alpha_H/beta_H are the master
        expressions with M_sigma_m -> M_sigma_m + W, nothing else."""
        mesh = _brick_mesh()
        spec = _cavity_spec(mesh, sigma=5.8e7)
        s = FITTimeDomainSolver(
            mesh=mesh,
            boundary_conditions={},
            dt=DT,
            total_time_steps=1,
            verbose=False,
            sibc=spec,
        )
        s.setup()
        assert s._sibc is not None
        M_mu = build_M_mu(mesh)
        W = SIBCOperator.from_spec(
            spec,
            mesh.grid,
            DT,
            frozen=(M_mu <= 0),
        ).W
        M_sm = np.where(M_mu > 0, build_M_sigma_m(mesh), 0.0) + W
        denom = np.where(M_mu > 0, M_mu + 0.5 * DT * M_sm, 1.0)
        np.testing.assert_array_equal(
            s._alpha_H,
            np.where(M_mu > 0, (M_mu - 0.5 * DT * M_sm) / denom, 1.0),
        )
        np.testing.assert_array_equal(
            s._beta_H,
            np.where(M_mu > 0, DT / denom, 0.0),
        )


def _reference_march(M_mu, dt, terms, drive):
    """Exact per-step solve of the coupled implicit face/branch system.

    Solves the DERIVATION.md §3 equations for one face NUMERICALLY
    (unknowns ``[h+, u_1+, ...]``, trapezoidal branches, midpoint
    coupling) — independent of the closed-form R_inst/k_p/q_p algebra
    the operator implements.  ``terms``: ``[(G, fit), ...]`` acting on
    the face (two entries model a bimetal seam).
    """
    branches = [(G, b, c) for G, fit in terms for b, c in fit.branches]
    c0G = sum(G * fit.c0 for G, fit in terms)
    n_u = len(branches)
    h = 0.0
    u = np.zeros(n_u)
    out = np.empty(drive.size)
    for step, R in enumerate(drive):
        A = np.zeros((n_u + 1, n_u + 1))
        rhs = np.zeros(n_u + 1)
        gc_half = 0.5 * sum(G * c for G, b, c in branches)
        A[0, 0] = M_mu / dt + 0.5 * c0G + gc_half
        rhs[0] = (M_mu / dt - 0.5 * c0G - gc_half) * h + R
        for j, (G, b, c) in enumerate(branches):
            A[0, 1 + j] = -0.5 * G
            rhs[0] += 0.5 * G * u[j]
            A[1 + j, 1 + j] = 1.0 + b * dt / 2.0
            A[1 + j, 0] = -b * dt * c / 2.0
            rhs[1 + j] = (1.0 - b * dt / 2.0) * u[j] + b * dt * c / 2.0 * h
        x = np.linalg.solve(A, rhs)
        h, u = x[0], x[1:]
        out[step] = h
    return out


def _one_face_surface(grid, comp, flat_idx, g, tag):
    inv_l = 1.0 / D
    return SIBCSurface(
        tag=tag,
        comp=np.array([comp], dtype=np.uint8),
        flat_idx=np.array([flat_idx], dtype=np.int64),
        weight=np.array([g * D * D]),
        inv_l_dual=np.array([inv_l]),
        area_total=g * D * D,
    )


def _operator_march(op, n_states, face_state, M_mu, dt, drive):
    """The solver-path arithmetic: W-folded kernel + two-phase hooks."""
    denom = M_mu + 0.5 * dt * op.W
    alpha = (M_mu - 0.5 * dt * op.W) / denom
    beta = dt / denom
    op.bind(beta, np)
    h = np.zeros(n_states)
    rvec = np.zeros(n_states)
    out = np.empty(drive.size)
    for step, R in enumerate(drive):
        op.save_field(h)
        rvec[face_state] = R
        h = alpha * h + beta * rvec
        op.update_field(h)
        out[step] = h[face_state]
    return out


class TestScalarReference:
    """Operator + folding vs the numerically solved implicit system."""

    def test_single_face_two_branches(self):
        grid = GridLines(x=np.arange(3) * D, y=np.arange(3) * D, z=np.arange(3) * D)
        fit = _fit(c0=0.3, branches=((2e10, 0.05), (2e11, 0.12)))
        surf = _one_face_surface(grid, comp=2, flat_idx=0, g=1.0, tag=1)
        op = SIBCOperator.from_spec(
            SIBCSpec(surfaces=(surf,), fits={1: fit}),
            grid,
            DT,
        )
        face_state = int(surf.state_indices(grid)[0])
        M_mu = 1.26e-9
        rng = np.random.default_rng(5)
        drive = 1e-9 * rng.standard_normal(60)
        got = _operator_march(op, op.W.size, face_state, np.full(op.W.size, M_mu), DT, drive)
        ref = _reference_march(M_mu, DT, [(1.0, fit)], drive)
        # atol covers zero-crossing samples (field scale ~1e-12)
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-24)

    def test_bimetal_seam_shared_face(self):
        """One face booked by TWO tags (bimetal seam): the two-phase
        update must reproduce the jointly solved system exactly."""
        grid = GridLines(x=np.arange(3) * D, y=np.arange(3) * D, z=np.arange(3) * D)
        fit1 = _fit(c0=0.2, branches=((3e10, 0.04),))
        fit2 = _fit(c0=0.9, branches=((1e10, 0.30), (5e11, 0.08)), sigma=1.4e6)
        s1 = _one_face_surface(grid, comp=2, flat_idx=0, g=0.5, tag=1)
        s2 = _one_face_surface(grid, comp=2, flat_idx=0, g=0.5, tag=2)
        op = SIBCOperator.from_spec(
            SIBCSpec(surfaces=(s1, s2), fits={1: fit1, 2: fit2}),
            grid,
            DT,
        )
        assert len(op.blocks) == 2
        face_state = int(s1.state_indices(grid)[0])
        M_mu = 1.26e-9
        rng = np.random.default_rng(9)
        drive = 1e-9 * rng.standard_normal(60)
        got = _operator_march(op, op.W.size, face_state, np.full(op.W.size, M_mu), DT, drive)
        ref = _reference_march(
            M_mu,
            DT,
            [(0.5, fit1), (0.5, fit2)],
            drive,
        )
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-24)


class TestDissipation:
    def _decay(self, sigma, steps=4000, skip=500):
        """Ring the TE101-seeded cavity down; tail-fit the decay rate.

        The first ``skip`` samples are excluded: the sampled sine seed
        is not an exact discrete eigenvector, and its settling transient
        is sigma-INDEPENDENT — at high sigma it would swamp the tiny
        genuine decay (measured: slope −0.46 with the transient in the
        fit, −0.5000 without).
        """
        mesh = Mesh.from_grid(_grid())
        spec = _cavity_spec(mesh, sigma=sigma)
        dt = courant_dt(mesh.grid)
        s = _cavity_solver(mesh, spec, steps, dt=dt)
        s.setup()
        _seed_te101(s, mesh)
        sampler = _EnergySampler(s)
        s.monitors.append(sampler)
        s.run()
        energy = np.asarray(sampler.energy)
        tail = energy[skip:]
        n = np.arange(tail.size, dtype=float)
        rate = -np.polyfit(n * dt, np.log(tail), 1)[0]
        return rate, energy

    def test_energy_decays_and_stays_stable_at_cfl(self):
        """SIBC walls dissipate and the run is stable at the UNCHANGED
        lossless courant_dt (DERIVATION.md §5)."""
        rate, energy = self._decay(sigma=5.8e5, steps=1200, skip=200)
        assert np.isfinite(energy).all()
        assert energy[-1] < energy[0]
        assert rate > 0.0

    def test_rate_scales_as_inverse_sqrt_sigma(self):
        """Gate B physics in miniature: the damping rate must scale as
        R_s ~ sigma^(-1/2) across four decades of conductivity
        (measured at these settings: slope −0.5000, decade ratios
        9.99 / 10.01)."""
        sigmas = np.array([5.8e5, 5.8e7, 5.8e9])
        rates = np.array([self._decay(s)[0] for s in sigmas])
        assert (rates > 0).all()
        slope = np.polyfit(np.log(sigmas), np.log(rates), 1)[0]
        assert slope == pytest.approx(-0.5, abs=0.02)


class TestCheckpointResume:
    def test_same_ops_rewind_bit_exact(self):
        """WP-S6 pattern: checkpoint mid-run, march on, rewind, remarch
        — fields and branch states bit-exact (DD-070 gate extended)."""
        mesh = Mesh.from_grid(_grid())
        spec = _cavity_spec(mesh, sigma=5.8e6)
        s = _cavity_solver(mesh, spec, steps=40)
        s.setup()
        rng = np.random.default_rng(3)
        s._fields.e_flat[:] = rng.standard_normal(s._fields.e_flat.size)
        s.run()

        sd = s.state_dict()
        assert "sibc" in sd
        s.total_time_steps = 80
        s.run()
        h_ref = s._fields.h_flat.copy()
        e_ref = s._fields.e_flat.copy()
        u_ref = {
            k: {kk: vv.copy() for kk, vv in v.items()} for k, v in s.state_dict()["sibc"].items()
        }
        assert any(v for v in u_ref.values())  # branch states exist

        s.load_state_dict(sd)
        s.run()
        np.testing.assert_array_equal(s._fields.h_flat, h_ref)
        np.testing.assert_array_equal(s._fields.e_flat, e_ref)
        u_back = s.state_dict()["sibc"]
        for key, branches in u_ref.items():
            for bk, val in branches.items():
                np.testing.assert_array_equal(u_back[key][bk], val)


class TestSIBCPrecision:
    """Wall-impedance branch states follow the field precision (DD-094).

    ``|k| < 1`` on every Foster branch, so single storage carries no
    unbounded-accumulation hazard; the coefficients stay double.
    """

    def _solver(self, precision, steps=30):
        mesh = Mesh.from_grid(_grid())
        spec = _cavity_spec(mesh, sigma=5.8e6)
        s = FITTimeDomainSolver(
            mesh=mesh,
            boundary_conditions={f: PECBoundary(f) for f in _ALL_FACES},
            dt=DT,
            total_time_steps=steps,
            verbose=False,
            sibc=spec,
            backend="numpy",
            precision=precision,
        )
        s.setup()
        return s

    def test_single_branch_states_are_float32(self):
        s = self._solver("single")
        b = s._sibc.blocks[0]
        assert b.g.dtype == np.float32
        assert b.h_prev.dtype == np.float32
        assert all(br.u.dtype == np.float32 for br in b.branches)

    def test_double_branch_states_stay_float64(self):
        s = self._solver("double")
        b = s._sibc.blocks[0]
        assert b.g.dtype == np.float64
        assert all(br.u.dtype == np.float64 for br in b.branches)

    def test_single_sibc_march_is_finite(self):
        s = self._solver("single", steps=80)
        rng = np.random.default_rng(2)
        s._fields.e_flat[:] = rng.standard_normal(s._fields.e_flat.size).astype(np.float32)
        s.run()
        assert np.all(np.isfinite(s._fields.h_flat))
