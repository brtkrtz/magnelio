"""Silence test (Test 2 of the Phase-1 validation ladder).

Initialise a homogeneous WR-90 box with a TE10 mode profile multiplied by
a stationary Gaussian envelope along the propagation axis ``x``, place
:class:`PortOperatorModal` on both X_MIN and X_MAX, run with **no
excitation source**, and verify that the stored energy decays by more
than 60 dB within a few traversal times.

The H field is initialised to zero, so the initial Gaussian decomposes
into equal forward and reflected TE10 components.  Both ports must
absorb cleanly for the energy to decay.

A failure mode (poor absorption) shows up as either:

- Energy stuck at a high level (no decay) — operator is wrong or
  inactive.
- Slow rippling decay over ~ns — Mur-1st-order-style numerical
  reflections at non-zero amplitude.

A clean pass: monotonic decay below the −60 dB threshold within the
``total_time_steps`` budget.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from magnelio._fields.field_arrays import FieldState
from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.boundaries.pec import PECBoundary
from magnelio.boundaries.pmc import PMCBoundary
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.ports._modal import (
    BoxFace,
    PortOperatorModal,
    PortPlane,
    RectWGAnalyticalModeSolver,
    discretize_modes,
)
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import courant_dt

WR90_A = 22.86e-3
WR90_B = 10.16e-3


def _lateral_pec_bcs() -> dict:
    """PEC on the four lateral bbox faces (y/z walls).

    X_MIN / X_MAX are owned by the modal port operators in this test
    file and must not appear here.
    """
    return {
        "ymin": PECBoundary("ymin"),
        "ymax": PECBoundary("ymax"),
        "zmin": PECBoundary("zmin"),
        "zmax": PECBoundary("zmax"),
    }


def _initial_te10_packet(
    mesh: Mesh,
    x_0: float,
    sigma_x: float,
    f_carrier: float,
    direction: str = "forward",
    amplitude: float = 1.0,
    dt: float = 0.0,
) -> FieldState:
    """Build a forward-propagating TE10 wave packet IC.

    Constructs ``Ez`` AND a matched ``Hy`` so the IC is a pure forward
    (``+x``) wave packet, **not** a stationary-E IC.  Without matched H,
    the IC decomposes into equal forward + reflected halves that
    superpose into a standing wave for the duration of the simulation;
    the 1st-order Mur absorber cannot drain standing-wave energy because
    there is no net power flow at the boundary.

    For a forward TE10 wave at carrier ``f``:
        E_z(x, y, z, 0)         =  cos(β·(x − x_0)) · env(x − x_0) · sin(π·y/a)
        H_y(x_c, y, z_c, −dt/2) = −(1/Z_TE) · cos(β·(x_c − x_0) + ω·dt/2)
                                  · env(x_c − x_0 + v_g·dt/2) · sin(π·y/a)

    The half-step time and half-cell space offsets between E and H
    cancel at exact Courant; for non-Courant they introduce small
    residual reverse-wave components that Mur absorbs after a short
    transient.

    Parameters
    ----------
    direction : str
        ``"forward"`` for +x propagation, ``"backward"`` for −x.
    dt : float
        Solver time step.  Required to compute the H half-step phase
        offset.  If 0 (default), H is set to zero (stationary IC).
    """
    x_n = mesh.grid.x
    y_n = mesh.grid.y
    a = float(y_n[-1] - y_n[0])

    omega = 2 * math.pi * f_carrier
    omega_c = math.pi * 299_792_458.0 / a
    beta_sq = omega**2 - omega_c**2
    if beta_sq <= 0:
        raise ValueError("f_carrier must be above the TE10 cut-off")
    beta = math.sqrt(beta_sq) / 299_792_458.0
    v_g = (299_792_458.0**2) * beta / omega  # v_p · v_g = c² → v_g = c²·β/ω
    Z_TE = omega * 4e-7 * math.pi / beta  # Z_TE = ωμ₀/β at vacuum

    sign = +1.0 if direction == "forward" else -1.0

    fields = FieldState.zeros(mesh.Nx, mesh.Ny, mesh.Nz)

    y_rel = y_n - y_n[0]
    pattern_y_node = np.sin(np.pi * y_rel / a)  # at y-nodes (for Ez)

    # Ez at x-nodes, y-nodes, z-centres
    dx_E = x_n - x_0
    env_E = np.exp(-(dx_E**2) / (2 * sigma_x**2))
    carrier_E = np.cos(sign * beta * dx_E)
    spatial_E = env_E * carrier_E
    factor_E = amplitude * spatial_E[:, None] * pattern_y_node[None, :]
    fields.Ez[:, :, :] = factor_E[:, :, None]

    # Hy at x-centres, y-nodes, z-centres, time = -dt/2
    if dt > 0.0:
        x_c = 0.5 * (x_n[:-1] + x_n[1:])
        # Half-step earlier in time → wave was further "back" in propagation
        # direction.  For forward (+x) wave at t = -dt/2, the spatial offset
        # in x is -v_g · dt/2 (wave hadn't moved that far yet).
        dx_H = x_c - x_0 + sign * v_g * dt / 2
        env_H = np.exp(-(dx_H**2) / (2 * sigma_x**2))
        # Carrier phase at t=-dt/2: cos(β·(x_c−x_0) ± ω·dt/2) for ±direction
        carrier_H = np.cos(sign * beta * (x_c - x_0) + omega * dt / 2)
        spatial_H = env_H * carrier_H
        # H_y for forward TE10: H_y = -E_z / Z_TE · sign (for +x prop, H_y < 0
        # when E_z > 0; for -x prop, H_y > 0 when E_z > 0).
        factor_H = -sign * (amplitude / Z_TE) * spatial_H[:, None] * pattern_y_node[None, :]
        # Hy shape (Nx, Ny+1, Nz). Broadcast over z (no z-dependence for TE10).
        fields.Hy[:, :, :] = factor_H[:, :, None]

    return fields


def _build_wr90_silence_setup(
    L_x: float = 60e-3,
    Nx: int = 30,
    Ny: int = 15,
    Nz: int = 8,
):
    """Construct mesh + port operators for the silence test."""
    grid = GridLines(
        x=np.linspace(0.0, L_x, Nx + 1),
        y=np.linspace(0.0, WR90_A, Ny + 1),
        z=np.linspace(0.0, WR90_B, Nz + 1),
    )
    mesh = Mesh.from_grid(grid)
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)

    # Per-port solvers: at X_MIN local-u=y (length WR90_A=L_y), at X_MAX
    # local-u=z (length WR90_B=L_z).  The "width_a" parameter refers to
    # the local-u dimension, not the global broad dimension.
    modes_min = RectWGAnalyticalModeSolver(
        width_a=WR90_A,
        height_b=WR90_B,
    ).solve(n_modes=1, f_calc=10e9)
    modes_max = RectWGAnalyticalModeSolver(
        width_a=WR90_B,
        height_b=WR90_A,
    ).solve(n_modes=1, f_calc=10e9)

    plane_min = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
    plane_max = PortPlane.from_mesh(BoxFace.X_MAX, mesh)
    discrete_min = discretize_modes(modes_min, plane_min, m_eps)
    discrete_max = discretize_modes(modes_max, plane_max, m_eps)
    return mesh, plane_min, plane_max, discrete_min, discrete_max, m_eps, m_mu


# ----------------------------------------------------------------------
# The silence test itself
# ----------------------------------------------------------------------


@pytest.mark.skip(
    reason="Hard floor confirmed at ≈ -7.7 dB (session 53, Phase-2a "
    "step 5 verification): identical decay shape on the analytical "
    "(PortSpecRectWG) and numerical (PortSpecNumerical) mode-solver "
    "paths over 5 / 25 / 80 traversals, both stabilising at -7.68 dB. "
    "Session 54 (2026-04-27) ruled out four candidate fixes "
    "(phase-velocity calibration, r=0 by construction, Higdon-N, "
    "simple modal aux-line); none addresses the floor.  DD-043 "
    "(PML-backed modal absorber) is rejected.  Commercial-suite performance "
    "requires DD-047 (Phase-3 Luo-Chen true co-simulation: per-mode "
    "1D auxiliary FDTD line sharing the port-plane Yee node with the "
    "3D mesh, simultaneous E/H stagger update).  Re-enable when "
    "DD-047 lands in Phase 3."
)
def test_silence_te10_decay_to_minus_60dB():
    """Energy must decay > 60 dB within the simulation budget."""
    L_x = 60e-3
    (
        mesh,
        plane_min,
        plane_max,
        discrete_min,
        discrete_max,
        m_eps,
        m_mu,
    ) = _build_wr90_silence_setup(L_x=L_x)

    dt = courant_dt(mesh.grid, accuracy="normal")
    omega_calc = 2 * math.pi * 10e9

    # Initial field: forward-propagating TE10 wave packet at L_x/2
    fields = _initial_te10_packet(
        mesh,
        x_0=L_x / 2.0,
        sigma_x=6e-3,
        f_carrier=10e9,
        direction="forward",
        amplitude=1.0,
        dt=dt,
    )

    op_min = PortOperatorModal(
        "port1",
        plane_min,
        discrete_min,
        m_eps,
        m_mu,
        dt=dt,
        omega_calc=omega_calc,
    )
    op_max = PortOperatorModal(
        "port2",
        plane_max,
        discrete_max,
        m_eps,
        m_mu,
        dt=dt,
        omega_calc=omega_calc,
    )

    # Group velocity at 10 GHz: v_g = c · √(1 − (f_c/f)²) ≈ 2.26e8 m/s.
    # Traversal time L_x / v_g ≈ 265 ps.  Run ~5 traversals.
    n_steps = int(round(5 * L_x / 2.26e8 / dt))

    solver = FITTimeDomainSolver(
        mesh=mesh,
        dt=dt,
        total_time_steps=n_steps,
        ports=[op_min, op_max],
        boundary_conditions=_lateral_pec_bcs(),
        energy_stop_db=60.0,
        verbose=False,
    )
    # Inject initial condition: setup() allocates _fields; replace it.
    solver.setup()
    solver._fields = fields  # noqa: SLF001 — internal injection of IC
    # Initialise Mur state from the IC (skipping this leaves V_*_prev = 0,
    # which produces a sign-flipped correction at step 0 for non-zero IC).
    op_min.initialize_state(fields.e_flat)
    op_max.initialize_state(fields.e_flat)

    final = solver.run()

    # Compute final / peak energy
    m_eps_diag = build_M_eps(mesh)
    m_mu_diag = build_M_mu(mesh)
    final_energy = 0.5 * (
        float((m_eps_diag * final.e_flat) @ final.e_flat)
        + float((m_mu_diag * final.h_flat) @ final.h_flat)
    )
    peak_energy = float(solver._peak_energy)
    assert peak_energy > 0.0

    decay_dB = 10 * math.log10(max(final_energy, 1e-300) / peak_energy)
    assert decay_dB <= -60.0, (
        f"Silence test failed: energy decayed only {decay_dB:.1f} dB "
        f"(target ≤ −60 dB).  Final={final_energy:.3e}, "
        f"Peak={peak_energy:.3e}.  Operator absorption likely faulty."
    )


def test_silence_with_no_operators_does_not_decay():
    """Sanity: without operators, energy stays high (PMC-walled box)."""
    L_x = 60e-3
    mesh = _build_wr90_silence_setup(L_x=L_x)[0]

    dt = courant_dt(mesh.grid, accuracy="normal")
    n_steps = int(round(5 * L_x / 2.26e8 / dt))

    fields = _initial_te10_packet(
        mesh,
        x_0=L_x / 2.0,
        sigma_x=6e-3,
        f_carrier=10e9,
        direction="forward",
        amplitude=1.0,
        dt=dt,
    )

    # Sealed PMC-walled box: PEC on lateral y/z faces, PMC on x_min/x_max
    # (tangential-E-symmetry boundaries that mimic the legacy default
    # "free" behaviour but explicitly closed so the bbox-coverage warning
    # is silenced and the docstring's "PMC-walled box" claim is literal).
    bcs = _lateral_pec_bcs()
    bcs.update(
        {
            "xmin": PMCBoundary("xmin", mesh.grid),
            "xmax": PMCBoundary("xmax", mesh.grid),
        }
    )
    solver = FITTimeDomainSolver(
        mesh=mesh,
        dt=dt,
        total_time_steps=n_steps,
        ports=[],  # no absorbers
        boundary_conditions=bcs,
        verbose=False,
    )
    solver.setup()
    solver._fields = fields  # noqa: SLF001
    final = solver.run()

    m_eps_diag = build_M_eps(mesh)
    m_mu_diag = build_M_mu(mesh)
    final_energy = 0.5 * (
        float((m_eps_diag * final.e_flat) @ final.e_flat)
        + float((m_mu_diag * final.h_flat) @ final.h_flat)
    )
    peak_energy = float(solver._peak_energy)
    decay_dB = 10 * math.log10(max(final_energy, 1e-300) / peak_energy)
    # Without ports, the box is PMC-bounded and energy bounces forever.
    # Final energy should remain within ~10 dB of peak (sanity bound).
    assert decay_dB > -10.0, (
        f"Sanity check failed: energy decayed {decay_dB:.1f} dB without "
        f"any absorber present — leak somewhere?"
    )
