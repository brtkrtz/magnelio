"""Unit tests for FIT-TD solver (leapfrog, PEC mask, energy conservation)."""

import numpy as np
import pytest

from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.boundaries.pec import PECBoundary
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import courant_dt

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


def _cavity_solver(Nx=6, Ny=6, Nz=6, n_steps=50):
    L = Nx * 1e-3
    grid = GridLines(
        x=np.linspace(0, L, Nx + 1),
        y=np.linspace(0, L, Ny + 1),
        z=np.linspace(0, L, Nz + 1),
    )
    mesh = Mesh.from_grid(grid, boundary_conditions=_BC_OPEN)
    dt = courant_dt(grid, "draft")
    bcs = {f: PECBoundary(f) for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    solver = FITTimeDomainSolver(
        mesh=mesh,
        boundary_conditions=bcs,
        total_time_steps=n_steps,
        dt=dt,
        verbose=False,
    )
    return solver, mesh


class TestFITTDSetup:
    def test_setup_allocates_fields(self):
        solver, _ = _cavity_solver()
        solver.setup()
        assert solver._fields is not None
        assert hasattr(solver, "_curl_bufs")
        assert solver._alpha_E is not None
        assert solver._beta_H is not None

    def test_beta_H_positive(self):
        """β_H = dt/M_mu must be strictly positive."""
        solver, _ = _cavity_solver()
        solver.setup()
        assert np.all(solver._beta_H > 0)

    def test_alpha_E_bounded(self):
        """α_E = (M_eps - 0.5dt·σ)/(M_eps + 0.5dt·σ) should be ≤ 1 (lossless: =1)."""
        solver, _ = _cavity_solver()
        solver.setup()
        assert np.all(solver._alpha_E <= 1.0 + 1e-10)

    def test_pec_mask_all_false_for_air(self):
        solver, _ = _cavity_solver()
        solver.setup()
        # Air-only mesh: no PEC cells → pec_mask_E all False
        assert not solver._pec_mask_E.any()


class TestFITTDLeapfrog:
    def test_initial_impulse_propagates(self):
        solver, _ = _cavity_solver(n_steps=10)
        solver.setup()
        solver._fields.Ex[3, 3, 3] = 1.0
        fields = solver.run()
        # Energy should have spread to Hz (via curl)
        assert np.abs(fields.Hz).max() > 0

    def test_zero_initial_fields_stay_zero(self):
        solver, _ = _cavity_solver(n_steps=20)
        solver.setup()
        fields = solver.run()
        for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            assert np.allclose(getattr(fields, comp), 0.0), f"{comp} should stay zero"

    def test_pec_boundary_enforced(self):
        """Tangential E on PEC faces must remain 0 at all times."""
        solver, _ = _cavity_solver(n_steps=30)
        solver.setup()
        solver._fields.Ex[3, 3, 3] = 1.0
        fields = solver.run()
        # xmin face: Ey[0,:,:] and Ez[0,:,:] must be zero
        np.testing.assert_allclose(fields.Ey[0, :, :], 0.0, atol=1e-12)
        np.testing.assert_allclose(fields.Ez[0, :, :], 0.0, atol=1e-12)

    def test_energy_conservation_lossless_cavity(self):
        """Time-averaged energy drift must be < 1% over 500 leapfrog steps.

        The staggered leapfrog stores E at integer steps and H at half-steps,
        so the instantaneous energy W = 0.5(M_eps·E² + M_mu·H²) oscillates
        at twice the cavity frequency.  The time-average over many cycles is
        the conserved quantity and should be stable to < 0.1%.
        """
        Nx = Ny = Nz = 6
        L = Nx * 1e-3
        grid = GridLines(
            x=np.linspace(0, L, Nx + 1),
            y=np.linspace(0, L, Ny + 1),
            z=np.linspace(0, L, Nz + 1),
        )
        mesh = Mesh.from_grid(grid, boundary_conditions=_BC_OPEN)
        M_eps = build_M_eps(mesh)
        M_mu = build_M_mu(mesh)
        dt = courant_dt(grid, "normal")
        bcs = {f: PECBoundary(f) for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}

        solver = FITTimeDomainSolver(
            mesh=mesh,
            boundary_conditions=bcs,
            total_time_steps=1,
            dt=dt,
            verbose=False,
        )
        solver.setup()
        solver._fields.Ex[3, 3, 3] = 1e3  # impulse

        def total_energy(f):
            e = np.concatenate([f.Ex.ravel(), f.Ey.ravel(), f.Ez.ravel()])
            h = np.concatenate([f.Hx.ravel(), f.Hy.ravel(), f.Hz.ravel()])
            return 0.5 * (np.dot(M_eps * e, e) + np.dot(M_mu * h, h))

        n_steps = 500
        energies = []
        for _ in range(n_steps):
            solver.run()
            energies.append(total_energy(solver._fields))

        energies = np.array(energies)
        half = n_steps // 2
        mean_first = energies[:half].mean()
        mean_second = energies[half:].mean()
        drift_pct = abs(mean_second - mean_first) / max(mean_first, 1e-30) * 100
        assert drift_pct < 1.0, (
            f"Time-averaged energy drift {drift_pct:.3f}% exceeds 1% "
            f"(1st-half mean={mean_first:.3e}, 2nd-half mean={mean_second:.3e})"
        )


class TestBboxFaceCoverageWarning:
    """The setup-time warning that flags uncovered bbox faces.

    Catches the foot-gun that produced the session-46 misdiagnosis:
    instantiating ``FITTimeDomainSolver`` directly with a port on
    X_MIN/X_MAX but no BCs on the lateral faces leaves the y/z bbox
    edges to evolve freely, breaking any waveguide-style simulation.
    """

    @staticmethod
    def _bare_solver(boundary_conditions=None, ports=None):
        """Minimal solver, only the inputs the warning logic looks at."""
        Nx = Ny = Nz = 4
        L = 4e-3
        grid = GridLines(
            x=np.linspace(0, L, Nx + 1),
            y=np.linspace(0, L, Ny + 1),
            z=np.linspace(0, L, Nz + 1),
        )
        mesh = Mesh.from_grid(grid, boundary_conditions=_BC_OPEN)
        dt = courant_dt(grid, "draft")
        return FITTimeDomainSolver(
            mesh=mesh,
            boundary_conditions=boundary_conditions or {},
            ports=ports or [],
            total_time_steps=2,
            dt=dt,
            verbose=False,
        )

    @staticmethod
    def _bbox_warnings_during_setup(solver):
        """Return all UserWarnings whose message mentions 'bbox face'."""
        import warnings

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            solver.setup()
        return [r for r in record if "bbox face" in str(r.message)]

    def test_zero_coverage_silent(self):
        """No BCs and no ports: silent (likely a solver-internals test)."""
        solver = self._bare_solver()
        warns = self._bbox_warnings_during_setup(solver)
        assert not warns, [str(r.message) for r in warns]

    def test_partial_coverage_warns(self):
        """Some BCs but missing faces: fires."""
        # Only x faces covered; y/z faces missing.
        solver = self._bare_solver(
            boundary_conditions={
                "xmin": PECBoundary("xmin"),
                "xmax": PECBoundary("xmax"),
            }
        )
        with pytest.warns(UserWarning, match=r"bbox face.*ymax.*ymin.*zmax.*zmin"):
            solver.setup()

    def test_full_pec_coverage_silent(self):
        """All six faces have a BC: silent."""
        bcs = {f: PECBoundary(f) for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
        solver = self._bare_solver(boundary_conditions=bcs)
        warns = self._bbox_warnings_during_setup(solver)
        assert not warns, [str(r.message) for r in warns]

    def test_modal_port_covers_face(self):
        """A port with plane.face = BoxFace.X_MIN counts as covering xmin."""
        from magnelio.ports._modal import BoxFace

        class _MinimalPlane:
            face = BoxFace.X_MIN

        class _MinimalOp:
            plane = _MinimalPlane()

        bcs = {f: PECBoundary(f) for f in ("xmax", "ymin", "ymax", "zmin", "zmax")}
        solver = self._bare_solver(
            boundary_conditions=bcs,
            ports=[_MinimalOp()],
        )
        warns = self._bbox_warnings_during_setup(solver)
        assert not warns, [str(r.message) for r in warns]


class TestFrozenPECEdges:
    """PEC edges are frozen degrees of freedom (session 111): exact
    alpha_E = beta_E = 0 replaces the former per-step e[pec_idx] = 0
    scatter (0.83 ms/step at 379k cells).  Landed after a bitwise
    old-vs-new gate on PEC-box, CPML and periodic marches."""

    @staticmethod
    def _pec_mesh(walls=()):
        from magnelio.materials.material import Material

        L = 8e-3
        lin = np.linspace(0.0, L, 9)
        mesh = Mesh.from_grid(
            GridLines(x=lin, y=lin, z=lin),
            regions=[(Material.pec(), (2e-3, 2e-3, 1e-3, 4e-3, 4e-3, 7e-3))],
            boundary_conditions=_BC_OPEN,
        )
        if walls:
            mesh = mesh.with_pec_boundaries(list(walls))
        return mesh

    def test_pec_coefficients_zeroed(self):
        mesh = self._pec_mesh()
        s = FITTimeDomainSolver(
            mesh=mesh, boundary_conditions={}, total_time_steps=1, verbose=False
        )
        s.setup()
        idx = s._pec_idx_E
        assert idx is not None and idx.size > 0
        assert np.all(s._alpha_E[idx] == 0.0)
        assert np.all(s._beta_E[idx] == 0.0)
        free = np.setdiff1d(np.arange(s._beta_E.size), idx)
        assert np.all(s._beta_E[free] > 0.0)

    def test_pec_edges_stay_exact_zero_through_march(self):
        """Even a deliberately corrupted initial state on PEC edges is
        wiped by the first kernel pass (alpha = 0) and stays +0.0."""
        walls = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
        mesh = self._pec_mesh(walls)
        bcs = {f: PECBoundary(f) for f in walls}
        s = FITTimeDomainSolver(
            mesh=mesh, boundary_conditions=bcs, total_time_steps=60, verbose=False
        )
        s.setup()
        rng = np.random.default_rng(2)
        s._fields.e_flat[:] = rng.standard_normal(s._fields.e_flat.size)
        s._fields.h_flat[:] = rng.standard_normal(s._fields.h_flat.size)
        s.run()
        e_pec = s._fields.e_flat[s._pec_idx_E]
        assert np.all(e_pec == 0.0)
        assert np.max(np.abs(s._fields.e_flat)) > 0.0  # field alive

    def test_reenforce_flag_only_for_unsafe_bcs(self):
        from magnelio.boundaries.cpml import CPMLBoundary
        from magnelio.boundaries.periodic import PeriodicBoundary

        walls4 = ("ymin", "ymax", "zmin", "zmax")
        mesh = self._pec_mesh(walls4)
        grid = mesh.grid
        safe = {
            "zmin": CPMLBoundary(face="zmin", grid=grid, thickness_cells=3),
            **{f: PECBoundary(f) for f in ("xmin", "xmax", "ymin", "ymax", "zmax")},
        }
        s = FITTimeDomainSolver(
            mesh=mesh, boundary_conditions=safe, total_time_steps=1, verbose=False
        )
        s.setup()
        assert s._pec_reenforce_after_bc is False
        unsafe = dict(safe)
        unsafe["xmin"] = PeriodicBoundary(axis="x", grid=grid)
        unsafe.pop("xmax")
        s2 = FITTimeDomainSolver(
            mesh=mesh, boundary_conditions=unsafe, total_time_steps=1, verbose=False
        )
        s2.setup()
        assert s2._pec_reenforce_after_bc is True
