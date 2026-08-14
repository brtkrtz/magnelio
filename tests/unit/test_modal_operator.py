"""Tests for magnelio.ports._modal.operator — modal Mur-1st-order absorber."""

from __future__ import annotations

import math

import numpy as np
import pytest

from magnelio._fields.field_arrays import FieldState
from magnelio._operators.material_matrices import (
    EPS0 as EPS0_MM,
)
from magnelio._operators.material_matrices import (
    MU0,
    build_M_eps,
    build_M_mu,
)
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.ports._modal import (
    BoxFace,
    CoaxAnalyticalModeSolver,
    ModeType,
    PortOperatorModal,
    PortPlane,
    PortSpecNumerical,
    RectWGAnalyticalModeSolver,
    build_modal_port,
    discretize_modes,
)
from magnelio.solver.stability import courant_dt

# DD-103: the closure these fixtures always assumed.  A face
# with no BC used to evolve under the free curl operator —
# which IS the natural magnetic wall, hence "PMC".
_BC_CLOSED = {
    "xmin": "PEC",
    "xmax": "PEC",
    "ymin": "PEC",
    "ymax": "PEC",
    "zmin": "PEC",
    "zmax": "PEC",
}

_BC_OPEN = {
    "xmin": "PMC",
    "xmax": "PMC",
    "ymin": "PMC",
    "ymax": "PMC",
    "zmin": "PMC",
    "zmax": "PMC",
}


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

WR90_A = 22.86e-3
WR90_B = 10.16e-3


def _wr90_numerical_mesh():
    """Uniform hollow WR-90 stub with consolidated PEC walls, for the
    numerical-path (KG-DTBC) selection tests."""
    grid = GridLines(
        x=np.linspace(0.0, 10e-3, 6),
        y=np.linspace(0.0, WR90_A, 15),
        z=np.linspace(0.0, WR90_B, 8),
    )
    mesh = Mesh.from_grid(grid, boundary_conditions=_BC_CLOSED)
    return mesh, courant_dt(grid, accuracy="normal")


def _wr90_setup(n_modes: int = 1, f_calc: float = 10e9):
    """Build a small WR-90 box with operator on X_MIN at given f_calc."""
    grid = GridLines(
        x=np.linspace(0.0, 30e-3, 6),
        y=np.linspace(0.0, WR90_A, 15),
        z=np.linspace(0.0, WR90_B, 8),
    )
    mesh = Mesh.from_grid(grid, boundary_conditions=_BC_OPEN)
    plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    modes = RectWGAnalyticalModeSolver(
        width_a=WR90_A,
        height_b=WR90_B,
    ).solve(n_modes=n_modes, f_calc=f_calc)
    discrete = discretize_modes(modes, plane, m_eps)
    dt = courant_dt(grid, accuracy="normal")
    op = PortOperatorModal(
        name="port1",
        plane=plane,
        discrete_modes=discrete,
        m_eps_flat=m_eps,
        m_mu_flat=m_mu,
        dt=dt,
        omega_calc=2 * math.pi * f_calc,
    )
    return mesh, plane, op, discrete


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


class TestModalPortOperatorConstruction:
    def test_attributes_populated(self):
        _, plane, op, _ = _wr90_setup()
        assert op.name == "port1"
        assert op.plane is plane
        assert op.n_modes == 1
        assert len(op.discrete_modes) == 1

    def test_rejects_zero_dt(self):
        _, plane, _, discrete = _wr90_setup()
        m_eps = build_M_eps(_wr90_setup()[0])
        m_mu = build_M_mu(_wr90_setup()[0])
        with pytest.raises(ValueError, match="dt"):
            PortOperatorModal(
                "x",
                plane,
                discrete,
                m_eps,
                m_mu,
                dt=0.0,
                omega_calc=2 * math.pi * 10e9,
            )

    def test_rejects_zero_omega_calc(self):
        _, plane, _, discrete = _wr90_setup()
        m_eps = build_M_eps(_wr90_setup()[0])
        m_mu = build_M_mu(_wr90_setup()[0])
        with pytest.raises(ValueError, match="omega_calc"):
            PortOperatorModal(
                "x",
                plane,
                discrete,
                m_eps,
                m_mu,
                dt=1e-12,
                omega_calc=0.0,
            )


# ----------------------------------------------------------------------
# Mur reflection coefficient r_m
# ----------------------------------------------------------------------


class TestMurCoefficient:
    def test_te10_above_cutoff_real_in_minus_one_one(self):
        """r_m = (v_p·dt − dx)/(v_p·dt + dx) with |r| < 1 for v_p·dt < dx."""
        _, _, op, _ = _wr90_setup(n_modes=1, f_calc=10e9)
        r = op.mur_r
        assert r.shape == (1,)
        # For TE10 above cutoff: v_p > c, dx > c·dt (Courant); so v_p·dt could
        # be greater than dx for very dispersive modes.  In any case |r| < 1
        # because v_p · dt is finite-positive and dx is finite-positive.
        # The CFL safety factor ensures this remains stable.
        assert np.all(np.abs(r) < 1.0)

    def test_mur_zero_at_courant_match(self):
        """r=0 when v_p·dt = dx (perfect 1D Mur condition)."""
        # Construct a setup where dt is chosen so that v_p·dt = dx exactly.
        # For TE10 in WR-90 at f_calc, β = √(ω² - ω_c²)/c, so v_p = ω/β.
        grid = GridLines(
            x=np.linspace(0.0, 30e-3, 6),  # dx ≈ 6 mm
            y=np.linspace(0.0, WR90_A, 15),
            z=np.linspace(0.0, WR90_B, 8),
        )
        mesh = Mesh.from_grid(grid, boundary_conditions=_BC_OPEN)
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        m_eps = build_M_eps(mesh)
        m_mu = build_M_mu(mesh)
        modes = RectWGAnalyticalModeSolver(
            width_a=WR90_A,
            height_b=WR90_B,
        ).solve(n_modes=1, f_calc=10e9)
        discrete = discretize_modes(modes, plane, m_eps)
        # Compute v_p at f_calc analytically
        omega = 2 * math.pi * 10e9
        gamma = modes[0].gamma(omega)
        beta = abs(gamma.imag)
        v_p = omega / beta
        dx = float(grid.dx[0])
        dt_match = dx / v_p
        op = PortOperatorModal(
            "x",
            plane,
            discrete,
            m_eps,
            m_mu,
            dt=dt_match,
            omega_calc=omega,
        )
        assert abs(op.mur_r[0]) < 1e-12, f"r should be 0; got {op.mur_r[0]}"


# ----------------------------------------------------------------------
# Projections — V at port and interior, I at port
# ----------------------------------------------------------------------


class TestModalProjections:
    def test_project_V_zero_on_zero_field(self):
        mesh, _, op, _ = _wr90_setup()
        e = np.zeros_like(build_M_eps(mesh))
        np.testing.assert_array_equal(op.project_V(e), 0.0)

    def test_project_V_interior_zero_on_zero_field(self):
        mesh, _, op, _ = _wr90_setup()
        e = np.zeros_like(build_M_eps(mesh))
        np.testing.assert_array_equal(op.project_V_interior(e), 0.0)

    def test_project_I_zero_on_zero_field(self):
        mesh, _, op, _ = _wr90_setup()
        h = np.zeros_like(build_M_mu(mesh))
        np.testing.assert_array_equal(op.project_I(h), 0.0)

    def test_project_V_recovers_unity_for_e_eq_eu(self):
        """If e_pp = ê_0, V[0] = 1 by M_eps orthonormality."""
        mesh, plane, op, discrete = _wr90_setup(n_modes=3, f_calc=15e9)
        e = np.zeros_like(build_M_eps(mesh))
        e[plane.e_u_indices] = discrete[0].e_u_profile
        e[plane.e_v_indices] = discrete[0].e_v_profile
        V = op.project_V(e)
        assert V[0] == pytest.approx(1.0, rel=1e-12)
        assert abs(V[1]) < 1e-10
        assert abs(V[2]) < 1e-10

    def test_project_V_interior_recovers_unity_for_e_eq_eu_at_interior(self):
        """If e at interior plane = ê_0, V_interior[0] = 1."""
        mesh, plane, op, discrete = _wr90_setup(n_modes=2, f_calc=15e9)
        e = np.zeros_like(build_M_eps(mesh))
        e[plane.e_u_indices_interior] = discrete[0].e_u_profile
        e[plane.e_v_indices_interior] = discrete[0].e_v_profile
        V = op.project_V_interior(e)
        # For a homogeneous mesh M_eps at interior == M_eps at port for
        # tangential edges, so orthonormality holds at interior too.
        assert V[0] == pytest.approx(1.0, rel=1e-12)
        assert abs(V[1]) < 1e-10


# ----------------------------------------------------------------------
# Mur absorption sanity — single-step and multi-step behaviour
# ----------------------------------------------------------------------


class TestMurAbsorption:
    def test_zero_field_stays_zero(self):
        mesh, _, op, _ = _wr90_setup()
        fields = FieldState.zeros(mesh.Nx, mesh.Ny, mesh.Nz)
        e_before = fields.e_flat.copy()
        op.update_e(fields, t=op._dt, dt=op._dt)
        np.testing.assert_array_equal(fields.e_flat, e_before)

    def test_step_zero_with_field_only_at_port_yields_correction(self):
        """At step 0, V_*_prev = 0; correction is r·V_int_naive − V_port_naive."""
        mesh, plane, op, discrete = _wr90_setup()
        fields = FieldState.zeros(mesh.Nx, mesh.Ny, mesh.Nz)
        e = fields.e_flat
        # Place mode-0 profile at port edges only (interior stays zero)
        e[plane.e_u_indices] = discrete[0].e_u_profile
        e[plane.e_v_indices] = discrete[0].e_v_profile
        # At step 0: V_int_prev = 0, V_port_prev = 0, V_int_new = 0
        # V_port_correct = 0 + r·(0 − 0) = 0.  diff = 1 − 0 = 1.
        # Subtract 1·ê_0 from port edges → port edges become zero.
        op.update_e(fields, t=op._dt, dt=op._dt)
        V_after = op.project_V(fields.e_flat)
        assert abs(V_after[0]) < 1e-10

    def test_multi_step_state_persists(self):
        """V_port_prev / V_interior_prev are saved across update_e calls."""
        mesh, plane, op, discrete = _wr90_setup()
        fields = FieldState.zeros(mesh.Nx, mesh.Ny, mesh.Nz)
        e = fields.e_flat
        # Plant a mode-0 profile at the interior plane.  After step 0,
        # the operator's _V_interior_prev should record that.
        e[plane.e_u_indices_interior] = discrete[0].e_u_profile
        e[plane.e_v_indices_interior] = discrete[0].e_v_profile
        op.update_e(fields, t=op._dt, dt=op._dt)
        # Internal V_interior_prev should now be ~1
        assert op._V_interior_prev[0] == pytest.approx(1.0, rel=1e-10)


# ----------------------------------------------------------------------
# Constructor signature change: legacy z_eff field is gone
# ----------------------------------------------------------------------


class TestNoLegacyZeff:
    def test_no_z_eff_attribute(self):
        _, _, op, _ = _wr90_setup()
        assert not hasattr(op, "z_eff")


# ----------------------------------------------------------------------
# V/I calibration — V_m / I_m must equal Z_modal for an analytical mode
# ----------------------------------------------------------------------


def _coax_setup(f_calc: float = 10e9):
    """Build a coax-shaped box with a TEM modal port operator on X_MIN.

    Round-coax cross-section in the (y, z) plane; outer/inner conductor
    confinement is irrelevant for this unit test (we synthesize the
    analytical FIT field explicitly), so we use a plain ``Mesh.from_grid``
    without PEC regions.
    """
    L_x = 30e-3
    L_yz = 3e-3
    r_i = 0.3e-3
    r_o = 1.0e-3
    yc = zc = L_yz / 2.0
    grid = GridLines(
        x=np.linspace(0.0, L_x, 11),
        y=np.linspace(0.0, L_yz, 13),
        z=np.linspace(0.0, L_yz, 13),
    )
    mesh = Mesh.from_grid(grid, boundary_conditions=_BC_OPEN)
    plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    modes = CoaxAnalyticalModeSolver(
        inner_radius=r_i,
        outer_radius=r_o,
        center=(yc, zc),
    ).solve(n_modes=1)
    discrete = discretize_modes(modes, plane, m_eps)
    dt = courant_dt(grid, accuracy="normal")
    op = PortOperatorModal(
        name="port1",
        plane=plane,
        discrete_modes=discrete,
        m_eps_flat=m_eps,
        m_mu_flat=m_mu,
        dt=dt,
        omega_calc=2 * math.pi * f_calc,
    )
    return mesh, plane, op, modes, m_eps, m_mu


class TestVICalibration:
    """The post-Gram-Schmidt h-profile rescale must restore V/I = Z_modal."""

    def test_v_over_i_equals_z_modal_for_analytical_tem(self):
        """Project synthetic FIT-discretised TEM field; V/I must equal Z_TEM."""
        mesh, plane, op, modes, m_eps, m_mu = _coax_setup(f_calc=10e9)

        # Sample the analytical TEM mode at edge midpoints (in V/m and A/m
        # for unit V_phys = 1; CoaxAnalyticalModeSolver normalises so
        # V_phys = ∫_a^b E_r dr = 1 and I_phys = V_phys / Z_TEM).
        E_u_at_u, _, _, H_v_at_u = modes[0].field_evaluator(
            plane.u_edge_uv[:, 0],
            plane.u_edge_uv[:, 1],
        )
        _, E_v_at_v, H_u_at_v, _ = modes[0].field_evaluator(
            plane.v_edge_uv[:, 0],
            plane.v_edge_uv[:, 1],
        )

        # FIT discrete-field ansatz at unit V_phys:
        #   e[u-edge] = E_u(midpoint) · L_primal_u
        #   e[v-edge] = E_v(midpoint) · L_primal_v
        e_test = np.zeros_like(m_eps)
        e_test[plane.e_u_indices] = E_u_at_u * plane.u_edge_lengths
        e_test[plane.e_v_indices] = E_v_at_v * plane.v_edge_lengths

        # For h, the FIT identity ``M_μ · L_dual = μ₀ · normal_dx · L_primal_other``
        # lets us derive the dual lengths without exposing them on the plane.
        h_test = np.zeros_like(m_mu)
        L_dual_for_h_v = MU0 * plane.normal_dx * plane.u_edge_lengths / m_mu[plane.h_v_indices]
        L_dual_for_h_u = MU0 * plane.normal_dx * plane.v_edge_lengths / m_mu[plane.h_u_indices]
        h_test[plane.h_v_indices] = H_v_at_u * L_dual_for_h_v
        h_test[plane.h_u_indices] = H_u_at_v * L_dual_for_h_u

        V_m = op.project_V(e_test)
        I_m = op.project_I(h_test)
        assert I_m[0] != 0.0, "I projection vanished — calibration broken"

        Z_modal = float(modes[0].z_modal(2 * math.pi * 10e9).real)
        ratio = float(V_m[0] / I_m[0])
        # For the analytical synthetic field the calibration is exact;
        # tolerate a tiny round-off slack.
        assert ratio == pytest.approx(Z_modal, rel=1e-9), (
            f"V/I = {ratio:.4f} Ω, expected Z_modal = {Z_modal:.4f} Ω "
            f"(rel err {abs(ratio - Z_modal) / Z_modal:.2e})"
        )

    def test_z_modal_attribute_unchanged_by_calibration(self):
        """Calibration only rescales h profile; mode.z_modal stays intact."""
        _, _, op, _, _, _ = _coax_setup(f_calc=10e9)
        Z_pre = op.discrete_modes[0].mode.z_modal(2 * math.pi * 10e9)
        # Z_modal lives on the underlying Mode object — unchanged by the
        # operator's per-mode h-profile rescale.
        ETA0 = math.sqrt(MU0 / 8.854187817e-12)
        Z_expected = ETA0 / (2 * math.pi) * math.log(1.0 / 0.3)
        assert abs(Z_pre.real - Z_expected) < 1e-6
        assert abs(Z_pre.imag) < 1e-12


class TestDTBCSelection:
    """Certified-uniform chains get the exact DTBC (TEM and, on the
    numerical path, TE/TM with the discrete Klein-Gordon mass)."""

    def test_analytical_tem_gets_dtbc_with_pair_product_courant(self):
        """r from the pair product equals c0*dt/dx on a vacuum mesh."""
        _, plane, op, _, _, _ = _coax_setup()
        assert op.termination_kinds == ["dtbc"]
        params = op.dtbc_line_params
        assert set(params.keys()) == {0}
        r, q, z0 = params[0]
        # Reference velocity from the *solver's* constants
        # (material_matrices.EPS0, CODATA 2014) — the pair product must
        # reproduce the actual discrete dynamics, not the mode-module
        # C0 (CODATA 2018, 2e-10 apart).
        r_expected = op._dt / (plane.normal_dx * math.sqrt(EPS0_MM * MU0))
        assert q == 0.0
        assert z0 is None
        assert abs(r - r_expected) / r_expected < 1e-12

    def test_analytical_te_mode_stays_on_mur(self):
        """Analytical-path modes carry the continuum cut-off, not the
        discrete eigenvalue — no certified q, so Mur it stays."""
        _, _, op, _ = _wr90_setup()
        assert op.termination_kinds == ["mur"]
        assert op.dtbc_line_params == {}

    def test_numerical_te_gets_kg_dtbc(self):
        """Numerical-path TE on a uniform chain: DTBC with
        q = omega_c*dt (discrete eigenvalue) and a positive z0."""
        mesh, dt = _wr90_numerical_mesh()
        m_eps = build_M_eps(mesh)
        m_mu = build_M_mu(mesh)
        spec = PortSpecNumerical(
            name="p1",
            plane=BoxFace.X_MIN,
            n_modes=1,
            mode_type=ModeType.TE,
        )
        op = build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=10e9)
        assert op.termination_kinds == ["dtbc"]
        r, q, z0 = op.dtbc_line_params[0]
        omega_c = op.discrete_modes[0].mode.omega_c
        assert q == pytest.approx(omega_c * dt, rel=1e-14)
        assert z0 is not None and z0 > 0.0
        assert 0.0 < r <= 1.0

    def test_numerical_tm_gets_kg_dtbc(self):
        mesh, dt = _wr90_numerical_mesh()
        m_eps = build_M_eps(mesh)
        m_mu = build_M_mu(mesh)
        spec = PortSpecNumerical(
            name="p1",
            plane=BoxFace.X_MIN,
            n_modes=1,
            mode_type=ModeType.TM,
        )
        op = build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=20e9)
        assert op.termination_kinds == ["dtbc"]
        r, q, z0 = op.dtbc_line_params[0]
        assert q == pytest.approx(
            op.discrete_modes[0].mode.omega_c * dt,
            rel=1e-14,
        )
        assert z0 is not None and z0 > 0.0

    def test_forced_mur_overrides_auto(self):
        mesh, plane, _, modes, m_eps, m_mu = _coax_setup()
        discrete = discretize_modes(modes, plane, m_eps)
        op = PortOperatorModal(
            "port1",
            plane,
            discrete,
            m_eps,
            m_mu,
            dt=courant_dt(mesh.grid, accuracy="normal"),
            omega_calc=2 * math.pi * 10e9,
            termination="mur",
        )
        assert op.termination_kinds == ["mur"]
        assert op.dtbc_line_params == {}

    def test_nonuniform_pair_product_falls_back_to_mur(self):
        """Breaking the pair identity on half the section rejects DTBC."""
        mesh, plane, _, modes, m_eps, m_mu = _coax_setup()
        m_mu_bad = m_mu.copy()
        half = plane.h_v_indices[: plane.h_v_indices.size // 2]
        m_mu_bad[half] *= 2.0
        discrete = discretize_modes(modes, plane, m_eps)
        op = PortOperatorModal(
            "port1",
            plane,
            discrete,
            m_eps,
            m_mu_bad,
            dt=courant_dt(mesh.grid, accuracy="normal"),
            omega_calc=2 * math.pi * 10e9,
        )
        assert op.termination_kinds == ["mur"]

    def test_rejects_unknown_termination(self):
        mesh, plane, _, modes, m_eps, m_mu = _coax_setup()
        discrete = discretize_modes(modes, plane, m_eps)
        with pytest.raises(ValueError, match="termination"):
            PortOperatorModal(
                "port1",
                plane,
                discrete,
                m_eps,
                m_mu,
                dt=courant_dt(mesh.grid, accuracy="normal"),
                omega_calc=2 * math.pi * 10e9,
                termination="pml",
            )
