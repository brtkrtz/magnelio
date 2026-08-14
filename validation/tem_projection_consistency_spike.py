"""WP7.1 spike: TEM/QTEM launch-mismatch root cause + fix demonstration.

Findings (session 80), measured on a parallel-plate line (10 x 5 x
20 mm, PEC plates via BC, PMC sides) with uniform vs growth-1.4-graded
transversal y grid, and on its half-filled QTEM variant (eps_r = 4
lower half):

1.  The electrostatic 2D Laplace E profile *is* the exact transversal
    profile of the discrete 3D travelling wave: lifting it with
    e^(-j beta k dz) satisfies the free z-edge rows of C^T M_mu^-1 C
    to 1e-14 on both grids, and the per-edge t-dispersion ratio is
    exactly 1 (the per-edge product M_eps * M_mu = eps mu dz dz~ is
    grid-independent).  The Laplace metric is NOT the problem.

2.  The problem is the I-projection dual basis: the pointwise H
    convention (h = +/- e / eta per edge) is wrong in edge-voltage
    terms on non-uniform transversal grids — the discrete wave's dual
    voltages obey h_co = e_co * (dz sqrt(eps_eff) / c) / M_mu[face].
    Probing the *exact* discrete travelling wave through project_V /
    project_I / compute_s_parameters reproduces the TD floors
    bit-faithfully (uniform -66.6 vs TD -67.1 dB; graded -21.4 = TD
    -21.4 dB) and shows V/I = 223.1 Ohm against the calibrated
    z_line = 188.4 Ohm — (223-188)/(223+188) = -21.4 dB exactly.

3.  Fix (demonstrated here, production switch = WP7.2): h profile =
    travelling-wave voltage form + sign-aware calibration
    I(TW) = V(TW)/z_line.  Probe results:
      TEM  graded: -21.4 -> -60.1 dB   (uniform unchanged, -65.8)
      QTEM graded: -21.6 -> -56.1 dB   (uniform -60.5 / -58.7)
    The intrinsic QTEM limit (E_z != 0 on inhomogeneous cross
    sections) sits below -56 dB here and is not the bottleneck.

4.  Caveat for 7.2: the PortOperatorModal V/I calibration is not
    scale-invariant in the h profile (the profile enters both as
    projection weight and as calibration test field), so the profile
    form and the _calibrate_v_i formula must be switched together.
    numerical_2d (TE/TM) uses the same pointwise convention and needs
    the same treatment on graded transversal grids.

Historical note (WP7.2, session 81): the production switch landed —
``solve_tem_laplace`` / ``solve_qtem_laplace`` now build the
travelling-wave h profiles natively and ``_calibrate_v_i`` uses the
direct M-metric formula.  The "pointwise" tags below therefore now
exercise the *new* production profiles (the pointwise convention no
longer exists in production); the TD anchors measure the fixed floors
(graded −21.4 → −64 dB).  This file stays as the decision record.

Run:  python validation/tem_projection_consistency_spike.py
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from magnelio._operators.curl import build_curl_matrix, build_gradient_matrix
from magnelio._operators.material_matrices import (
    build_M_eps,
    build_M_mu,
    flatten_port_plane_mass,
    flatten_port_plane_pec_mask,
)
from magnelio.boundaries.pec import PECBoundary
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.ports import PortSignalRecorder
from magnelio.ports._modal import BoxFace, PortPlane
from magnelio.ports._modal.auto_conductors import (
    extract_conductor_groups_from_mesh,
)
from magnelio.ports._modal.curl_curl_2d import (
    build_2d_curl_curl,
    build_2d_gradient,
)
from magnelio.ports._modal.discrete import discretize_modes
from magnelio.ports._modal.operator import PortOperatorModal
from magnelio.ports._modal.port_report import PortOperatorReport
from magnelio.ports._modal.tem_laplace import (
    solve_qtem_laplace,
    solve_tem_laplace,
)
from magnelio.post import compute_s_parameters
from magnelio.signals.signal_1d import Signal1D
from magnelio.signals.waveforms import gaussian
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import courant_dt

A, B, L = 10e-3, 5e-3, 20e-3
F_MAX = 10e9
F0 = 5e9
EPS_SUB = 4.0
C0 = 299_792_458.0


def graded_axis(lo, hi, n_cells, growth):
    """Symmetric grading: fine at both ends, growth toward the centre."""
    half = n_cells // 2
    d = np.ones(half)
    for i in range(1, half):
        d[i] = d[i - 1] * growth
    d_all = np.concatenate([d, d[::-1]] if n_cells % 2 == 0 else [d, [d[-1] * growth], d[::-1]])
    d_all = d_all / d_all.sum() * (hi - lo)
    return lo + np.concatenate([[0.0], np.cumsum(d_all)])


def build_mesh(graded):
    x = np.linspace(-A / 2, A / 2, 9)
    y = graded_axis(-B / 2, B / 2, 12, 1.4) if graded else np.linspace(-B / 2, B / 2, 13)
    z = np.linspace(0.0, L, 41)
    mesh = Mesh.from_grid(GridLines(x=x, y=y, z=z))
    return mesh.with_boundary_conditions(
        {
            "ymin": "PEC",
            "ymax": "PEC",
            "xmin": "PMC",
            "xmax": "PMC",
            "zmin": "PMC",
            "zmax": "PMC",
        }
    )


def half_filled_M_eps(mesh):
    """QTEM variant: eps_r = EPS_SUB for edge midpoints at y < 0."""
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    y_n = mesh.grid.y
    M = build_M_eps(mesh)
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    n_Ez = (Nx + 1) * (Ny + 1) * Nz
    _, j, _ = np.meshgrid(np.arange(Nx), np.arange(Ny + 1), np.arange(Nz + 1), indexing="ij")
    M[np.arange(n_Ex)[(y_n[j] < -1e-12).ravel()]] *= EPS_SUB
    _, j, _ = np.meshgrid(np.arange(Nx + 1), np.arange(Ny), np.arange(Nz + 1), indexing="ij")
    ey_y = 0.5 * (y_n[j] + y_n[j + 1])
    M[n_Ex + np.arange(n_Ey)[(ey_y < -1e-12).ravel()]] *= EPS_SUB
    _, j, _ = np.meshgrid(np.arange(Nx + 1), np.arange(Ny + 1), np.arange(Nz), indexing="ij")
    M[n_Ex + n_Ey + np.arange(n_Ez)[(y_n[j] < -1e-12).ravel()]] *= EPS_SUB
    return M


def build_port(label, face, mesh, m_eps, m_mu, dt, *, qtem=False, m_eps_vac=None):
    """Hand-wired production-equivalent modal TEM/QTEM port."""
    m_eps_l = flatten_port_plane_mass(m_eps, mesh, face)
    object.__setattr__(
        mesh,
        "pec_mask_edges",
        flatten_port_plane_pec_mask(mesh.pec_mask_edges, mesh, face),
    )
    plane = PortPlane.from_mesh(face, mesh)
    groups = extract_conductor_groups_from_mesh(plane, mesh)
    c3 = build_curl_matrix(mesh.grid)
    g3 = build_gradient_matrix(mesh.grid)
    _, M2d, _ = build_2d_curl_curl(plane, mesh.grid, m_eps_l, m_mu, c3)
    g2d, _, _ = build_2d_gradient(plane, mesh.grid, g3)
    if qtem:
        m_eps_vac_l = flatten_port_plane_mass(m_eps_vac, mesh, face)
        _, M2dv, _ = build_2d_curl_curl(
            plane,
            mesh.grid,
            m_eps_vac_l,
            m_mu,
            c3,
        )
        modes = solve_qtem_laplace(plane, g2d, M2d, M2dv, groups, grid=mesh.grid, m_mu_flat=m_mu)
    else:
        modes = solve_tem_laplace(plane, g2d, M2d, groups, 1.0, grid=mesh.grid, m_mu_flat=m_mu)
    discrete = discretize_modes(modes, plane, m_eps_l)
    op = PortOperatorModal(
        label,
        plane,
        discrete,
        m_eps_l,
        m_mu,
        dt=dt,
        omega_calc=2 * math.pi * F_MAX,
        port_report=PortOperatorReport(z_line_num=modes[0].z_line),
    )
    return op, plane, modes[0]


def travelling_wave(mesh, plane, mode, m_mu, dt, eps_eff):
    """Lift the mode's E profile to the exact leapfrog travelling wave."""
    C = build_curl_matrix(mesh.grid)
    edges0 = np.concatenate([plane.e_u_indices, plane.e_v_indices])
    e_t = np.concatenate(
        [np.asarray(mode.discrete_e_u_profile), np.asarray(mode.discrete_e_v_profile)]
    )
    omega = 2 * math.pi * F0
    Om = 2 * math.sin(omega * dt / 2) / dt
    dz = mesh.grid.dz[0]
    beta = 2 / dz * math.asin(Om * dz * math.sqrt(eps_eff) / (2 * C0))
    zeta = np.exp(-1j * beta * dz)
    e_c = np.zeros(C.shape[1], dtype=complex)
    for k in range(mesh.Nz + 1):
        e_c[edges0 + k] = e_t * zeta**k
    h_c = (1.0 / m_mu) * np.asarray(-(C @ e_c) / (1j * Om))
    return e_c, h_c, omega


def apply_tw_h_profile(op, plane, e_c, h_c):
    """Swap the operator's h profile for the travelling-wave voltage form.

    Calibrated sign-aware so that I(TW) = V(TW) / z_line — the
    production switch (WP7.2) must adapt ``_calibrate_v_i``
    equivalently.
    """
    dm = op.discrete_modes[0]
    hu_c = h_c[plane.h_u_indices]
    hv_c = h_c[plane.h_v_indices]
    ref_phase = np.exp(-1j * np.angle(hv_c[np.argmax(np.abs(hv_c))]))
    hu_p = np.real(hu_c * ref_phase)
    hv_p = np.real(hv_c * ref_phase)
    V_c = complex(
        np.dot(op._me_u_port, dm.e_u_profile * e_c[plane.e_u_indices])
        + np.dot(op._me_v_port, dm.e_v_profile * e_c[plane.e_v_indices])
    )
    I_raw = complex(np.dot(op._mh_u, hu_p * hu_c) + np.dot(op._mh_v, hv_p * hv_c))
    scale = np.real((V_c / op.port_report.z_line_num) / I_raw)
    op.discrete_modes[0] = dataclasses.replace(
        dm,
        h_u_profile=hu_p * scale,
        h_v_profile=hv_p * scale,
    )


def probe(op, plane, e_c, h_c, omega, dt, tag):
    """Feed the exact travelling wave through the V/I measurement chain."""
    dm = op.discrete_modes[0]
    V_c = complex(
        np.dot(op._me_u_port, dm.e_u_profile * e_c[plane.e_u_indices])
        + np.dot(op._me_v_port, dm.e_v_profile * e_c[plane.e_v_indices])
    )
    I_c = complex(
        np.dot(op._mh_u, dm.h_u_profile * h_c[plane.h_u_indices])
        + np.dot(op._mh_v, dm.h_v_profile * h_c[plane.h_v_indices])
    )
    n_steps = 4096
    t = np.arange(n_steps) * dt
    ramp = 0.5 * (1 - np.cos(np.pi * np.minimum(t / (60 / F0), 1.0)))
    signals = {
        ("port1", 0): (
            Signal1D(t=t, values=np.real(V_c * np.exp(1j * omega * t)) * ramp, dt=dt, label="V"),
            # recorder convention: I sampled dt/2 before V (WP5.2)
            Signal1D(
                t=t,
                values=np.real(I_c * np.exp(1j * omega * (t - dt / 2))) * ramp,
                dt=dt,
                label="I",
            ),
        )
    }
    ref = Signal1D(t=t, values=np.real(np.exp(1j * omega * t)) * ramp, dt=dt, label="exc")
    S = compute_s_parameters(
        recorder_signals=signals,
        port_modes={"port1": [d.mode for d in op.discrete_modes]},
        excited=("port1", 0),
        reference_signal=ref,
        f_axis=np.array([F0]),
        port_normal_dx={"port1": plane.normal_dx},
    )
    s11 = 20 * np.log10(abs(S[("port1", 0)][0]) + 1e-300)
    print(
        f"  [{tag:24s}] probe |S11|({F0 / 1e9:.0f} GHz) = {s11:7.2f} dB   "
        f"V/I(TW) = {abs(V_c / I_c):8.3f} Ohm  "
        f"(z_line {op.port_report.z_line_num:.3f})"
    )


def td_anchor(graded):
    """Full TD run with today's production profiles (reference floors)."""
    mesh = build_mesh(graded)
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    dt = courant_dt(mesh.grid, accuracy="normal")
    op1, _, _ = build_port("port1", BoxFace.Z_MIN, mesh, m_eps, m_mu, dt)
    op2, _, _ = build_port("port2", BoxFace.Z_MAX, mesh, m_eps, m_mu, dt)
    op1.set_excitation(0, lambda t: float(gaussian(t, F_MAX)))
    n_steps = int(round((2.5 * (4.0 / F_MAX) + 5.0 * (L / C0)) / dt))
    recorder = PortSignalRecorder(dt=dt, ports=[op1, op2])
    solver = FITTimeDomainSolver(
        mesh=mesh,
        dt=dt,
        total_time_steps=n_steps,
        ports=[op1, op2],
        recorder=recorder,
        boundary_conditions={
            "ymin": PECBoundary("ymin"),
            "ymax": PECBoundary("ymax"),
            "xmin": "PMC",
            "xmax": "PMC",
            "zmin": "PMC",
            "zmax": "PMC",
        },
        verbose=False,
    )
    solver.run()
    signals = recorder.finalize()
    t = np.arange(recorder.n_steps_recorded) * dt
    ref = Signal1D(
        t=t,
        values=np.array([float(gaussian(float(tt), F_MAX)) for tt in t]),
        dt=dt,
        label="exc",
    )
    f_axis = np.linspace(F_MAX / 40, F_MAX, 81)
    S = compute_s_parameters(
        recorder_signals=signals,
        port_modes={
            "port1": [d.mode for d in op1.discrete_modes],
            "port2": [d.mode for d in op2.discrete_modes],
        },
        excited=("port1", 0),
        reference_signal=ref,
        f_axis=f_axis,
        port_normal_dx={"port1": op1.plane.normal_dx, "port2": op2.plane.normal_dx},
    )
    s11 = 20 * np.log10(np.abs(S[("port1", 0)]) + 1e-300)
    tag = "graded" if graded else "uniform"
    print(
        f"  [TD anchor, {tag:7s} pointwise] max|S11| = {s11.max():7.2f} dB"
        f"  (median {np.median(s11):7.2f})"
    )


def main():
    print("TD anchors (production profiles):")
    for graded in (False, True):
        td_anchor(graded)

    for qtem in (False, True):
        print(
            f"\n{'QTEM (half-filled, eps_r=4)' if qtem else 'TEM (vacuum)'}"
            f" — synthetic travelling-wave probe:"
        )
        for graded in (False, True):
            mesh = build_mesh(graded)
            m_eps = half_filled_M_eps(mesh) if qtem else build_M_eps(mesh)
            m_eps_vac = build_M_eps(mesh) if qtem else None
            m_mu = build_M_mu(mesh)
            dt = courant_dt(mesh.grid, accuracy="normal")
            if qtem:
                dt = dt / math.sqrt(EPS_SUB)
            op, plane, mode = build_port(
                "port1",
                BoxFace.Z_MIN,
                mesh,
                m_eps,
                m_mu,
                dt,
                qtem=qtem,
                m_eps_vac=m_eps_vac,
            )
            eps_eff = mode.epsilon_r if qtem else 1.0
            e_c, h_c, omega = travelling_wave(
                mesh,
                plane,
                mode,
                m_mu,
                dt,
                eps_eff,
            )
            g = "graded" if graded else "uniform"
            probe(op, plane, e_c, h_c, omega, dt, f"{g} / pointwise")
            apply_tw_h_profile(op, plane, e_c, h_c)
            probe(op, plane, e_c, h_c, omega, dt, f"{g} / tw-consistent")


if __name__ == "__main__":
    main()
