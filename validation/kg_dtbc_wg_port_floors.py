"""WP-R3 acceptance: TE/TM port floors with the Klein-Gordon DTBC.

Two measurement legs on the WP-R3 acceptance geometries of the
reflection-free plan (retired to git history, see DD-055; WR-90
TE10/TM11 uniform; conformal round
waveguide TE11/TM01):

1. **CW lock-in (the criterion measurement).**  Monochromatic drive
   through the *production* solver, operators and recorder; V/I
   extracted by two-quadrature least squares over a late steady-state
   window and decomposed with the production formulas (temporal
   ``e^{+j w dt/2}`` rotation, ``destagger_theta`` two-plane solve,
   ``dtbc_wave_impedance``).  This is the R1-gate methodology
   (dtbc_kernel_spike.py) carried end-to-end into 3D: it measures the
   true port reflection floor, free of finite-record truncation.
   Acceptance: |S11| < -100 dB from ``1.01 * f_c_hat`` upward.

2. **Broadband pulsed run (high-level overview).**  The standard
   S-parameter workflow.  NOTE: near a cut-off the pulsed measurement
   itself is *truncation-limited* — the band-edge resonance decays
   only algebraically (v_g -> 0; energy diffuses into the exact
   absorber ~ sqrt(t)), so the rectangular-window DFT of a run
   truncated at in-domain energy -70 dB leaks ~ -20..-40 dB into the
   near-edge S-parameters regardless of the absorber quality
   (measured: +10 dB per 10x run length; the R1 "pulse FFT ratios
   floor from finite-record leakage" lesson).  These numbers document
   the practical pulsed floor, not the port.

Results (session 86), CW lock-in |S11|:

    WR-90 TE10     1.01 f_c: -150.4 dB   band: -153 .. -166 dB
    WR-90 TM11     1.01 f_c: -137.3 dB   band: -154 .. -165 dB
    round WG TE11  1.05-1.5 f_c: -124 .. -132 dB   (conformal)
    round WG TM01  1.05/1.2 f_c: -124 / -129 dB    (conformal)

All 24+ dB below the acceptance line (WR-90: 37-66 dB), including
1.01 f_c_hat where the DD-047 Mur peak sat at -19 dB.  The pulsed
overview lands at max -19 .. -38 dB near the band edge —
truncation-limited exactly as predicted by the run-length scaling,
median -40 .. -89 dB.

Run:  python validation/kg_dtbc_wg_port_floors.py
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.special import erf

from magnelio import AnalysisScatteringTD, Material, Mesh, MeshControl
from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.geo import Brick, Cylinder, Difference, GeometryModel
from magnelio.mesh.grid import GridLines
from magnelio.ports import PortSpecNumerical
from magnelio.ports._modal import (
    BoxFace,
    ModeType,
    build_modal_port,
)
from magnelio.ports._modal.dtbc import destagger_theta, dtbc_wave_impedance
from magnelio.ports.recorder import PortSignalRecorder
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import (
    compute_min_effective_eps,
    compute_min_effective_mu,
    courant_dt,
)

# ----------------------------------------------------------------------
# Geometries
# ----------------------------------------------------------------------


def wr90_mesh():
    a, b, length = 22.86e-3, 10.16e-3, 40e-3
    grid = GridLines(
        x=np.linspace(0.0, a, 21),
        y=np.linspace(0.0, b, 10),
        z=np.linspace(0.0, length, 41),
    )
    mesh = Mesh.from_grid(grid).with_boundary_conditions(
        {
            "xmin": "PEC",
            "xmax": "PEC",
            "ymin": "PEC",
            "ymax": "PEC",
            "zmin": "PMC",
            "zmax": "PMC",
        }
    )
    return mesh, courant_dt(grid, "normal")


def round_wg_mesh():
    R, length, dz = 10.0e-3, 36.0e-3, 1.5e-3
    s_bbox = 2.4 * R
    pec = Material.pec()
    vacuum = Material.air()
    bbox = Brick(
        origin=(-s_bbox / 2, -s_bbox / 2, 0.0), size=(s_bbox, s_bbox, length), material=pec
    )
    inner = Cylinder(origin=(0.0, 0.0, 0.0), radius=R, height=length, axis="z", material=vacuum)
    model = GeometryModel()
    model.add(Difference(bbox, inner))
    model.add(inner)
    n_t_nodes = 23
    t_nodes = np.linspace(-s_bbox / 2, s_bbox / 2, n_t_nodes).tolist()
    control = MeshControl(
        min_nodes_per_wavelength=8,
        min_cells_per_feature=0,
        growth_factor=1.5,
        conformal=True,
        max_cell_size=4.0 * s_bbox / (n_t_nodes - 1),
        forced_planes={
            "x": t_nodes,
            "y": t_nodes,
            "z": np.linspace(0.0, length, round(length / dz) + 1).tolist(),
        },
    )
    mesh = Mesh.from_geometry(model, control, f_max=14.0e9)
    dt = courant_dt(
        mesh.grid,
        "normal",
        min_effective_eps=compute_min_effective_eps(mesh),
        min_effective_mu=compute_min_effective_mu(mesh),
    )
    return mesh, dt


# ----------------------------------------------------------------------
# Leg 1 — CW lock-in through the production solver
# ----------------------------------------------------------------------


def cw_lockin_floor(name, mesh, dt, mode_type, f_calc, ratios):
    """|S11| of port 1 by steady-state lock-in at spot frequencies.

    ``ratios`` are evaluation frequencies as multiples of the port's
    discrete cut-off ``f_c_hat``.
    """
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    nz = mesh.Nz
    print(f"  {name} (CW lock-in)")
    for ratio in ratios:
        op1 = build_modal_port(
            PortSpecNumerical(name="port1", plane=BoxFace.Z_MIN, n_modes=1, mode_type=mode_type),
            mesh,
            m_eps,
            m_mu,
            dt=dt,
            f_calc=f_calc,
        )
        op2 = build_modal_port(
            PortSpecNumerical(name="port2", plane=BoxFace.Z_MAX, n_modes=1, mode_type=mode_type),
            mesh,
            m_eps,
            m_mu,
            dt=dt,
            f_calc=f_calc,
        )
        assert op1.termination_kinds == ["dtbc"], op1.termination_kinds
        r, q, z0 = op1.dtbc_line_params[0]

        w_dt = ratio * q
        period = 2.0 * math.pi / w_dt
        sigma = max(6.0 / ((ratio - 1.0) * q), 8.0 * period)
        s_hat = math.sin(w_dt / 2.0)
        sin_b2 = math.sqrt(max(s_hat**2 - (q / 2.0) ** 2, 1e-30)) / r
        v_g = r * r * math.sin(2.0 * math.asin(min(sin_b2, 1.0))) / math.sin(w_dt)
        n_win = int(30 * period)
        n_meas0 = int(10.0 * sigma + 40.0 * period + 3.0 * nz / max(v_g, 1e-3))
        n_steps = n_meas0 + n_win + 2

        t0 = 5.0 * sigma * dt
        sig_t = sigma * dt
        w_phys = w_dt / dt

        def waveform(t: float) -> float:
            amp = 0.5 * (1.0 + float(erf((t - t0) / (math.sqrt(2.0) * sig_t))))
            return amp * math.sin(w_phys * t)

        op1.set_excitation(0, waveform)
        recorder = PortSignalRecorder(dt=dt, ports=[op1, op2])
        solver = FITTimeDomainSolver(
            mesh=mesh,
            boundary_conditions={},
            ports=[op1, op2],
            recorder=recorder,
            total_time_steps=n_steps,
            dt=dt,
            verbose=False,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*neither a BoundaryCondition.*",
            )
            solver.run()
        signals = recorder.finalize(n_steps_actual=n_steps)
        V_sig, I_sig = signals[("port1", 0)]

        n_grid = np.arange(n_meas0, n_meas0 + n_win)
        basis = np.column_stack([np.cos(w_dt * n_grid), np.sin(w_dt * n_grid)])
        cv, *_ = np.linalg.lstsq(basis, V_sig.values[n_grid], rcond=None)
        ci, *_ = np.linalg.lstsq(basis, I_sig.values[n_grid], rcond=None)
        res_fit = float(
            np.linalg.norm(V_sig.values[n_grid] - basis @ cv)
            / max(np.linalg.norm(V_sig.values[n_grid]), 1e-300)
        )
        V = cv[0] - 1j * cv[1]
        I = (ci[0] - 1j * ci[1]) * np.exp(1j * w_dt / 2.0)

        theta = destagger_theta(np.array([w_dt]), r, q)[0]
        Z = dtbc_wave_impedance(np.array([w_dt]), q, z0, mode_type.value)[0]
        sz = np.sqrt(Z)
        ep, em = np.exp(theta), np.exp(-theta)
        a = (V / sz * ep + sz * I) / (ep + em)
        b = (V / sz * em - sz * I) / (ep + em)
        s11_db = 20.0 * math.log10(max(abs(b / a), 1e-300))
        print(
            f"    f/f_c_hat {ratio:5.3f}   |S11| {s11_db:8.1f} dB"
            f"   ({n_steps} steps, fit-res {res_fit:.1e})"
        )


# ----------------------------------------------------------------------
# Leg 2 — broadband pulsed overview (high-level API)
# ----------------------------------------------------------------------


def pulsed_overview(name, analysis, f_max):
    report = analysis.solve_ports()["port1"]
    f_c_hat = report.cutoff_num
    f_axis = np.linspace(1.01 * f_c_hat, f_max, 161)
    res = analysis.run(f_axis=f_axis)
    s11 = res.db("port1", "port1")
    s21 = res.db("port2", "port1")
    print(
        f"  {name:14s} f_c_hat {f_c_hat / 1e9:7.4f} GHz"
        f"   max|S11| {np.nanmax(s11):7.1f} dB"
        f"   median {np.nanmedian(s11):7.1f} dB"
        f"   |S21| dev {np.nanmax(np.abs(s21)):8.5f} dB"
    )


def wr90_analysis(mode_type, f_max):
    mesh, _ = wr90_mesh()
    return AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions(
            {
                "xmin": "PEC",
                "xmax": "PEC",
                "ymin": "PEC",
                "ymax": "PEC",
                "zmin": "PEC",
                "zmax": "PEC",
            }
        ),
        ports=[
            PortSpecNumerical(name="port1", plane=BoxFace.Z_MIN, n_modes=1, mode_type=mode_type),
            PortSpecNumerical(name="port2", plane=BoxFace.Z_MAX, n_modes=1, mode_type=mode_type),
        ],
        f_max=f_max,
        verbose=False,
    )


def round_wg_analysis(mode_type, f_max):
    mesh, _ = round_wg_mesh()
    return AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions({}),
        ports=[
            PortSpecNumerical(name="port1", plane=BoxFace.Z_MIN, n_modes=1, mode_type=mode_type),
            PortSpecNumerical(name="port2", plane=BoxFace.Z_MAX, n_modes=1, mode_type=mode_type),
        ],
        f_max=f_max,
        verbose=False,
    )


def main() -> None:
    print(
        "WP-R3 acceptance — CW lock-in port floors (criterion: |S11| < -100 dB from 1.01*f_c_hat):"
    )
    mesh, dt = wr90_mesh()
    cw_lockin_floor("WR-90 TE10", mesh, dt, ModeType.TE, 10.0e9, (1.01, 1.02, 1.05, 1.2, 1.5, 1.8))
    mesh, dt = wr90_mesh()
    cw_lockin_floor("WR-90 TM11", mesh, dt, ModeType.TM, 20.0e9, (1.01, 1.02, 1.05, 1.2, 1.45))
    mesh, dt = round_wg_mesh()
    cw_lockin_floor("round WG TE11 (conformal)", mesh, dt, ModeType.TE, 13.0e9, (1.05, 1.2, 1.5))
    mesh, dt = round_wg_mesh()
    cw_lockin_floor("round WG TM01 (conformal)", mesh, dt, ModeType.TM, 13.0e9, (1.05, 1.2))

    print()
    print(
        "Broadband pulsed overview (truncation-limited near the band edge — see module docstring):"
    )
    pulsed_overview("WR-90 TE10", wr90_analysis(ModeType.TE, 12.4e9), 12.4e9)
    pulsed_overview("WR-90 TM11", wr90_analysis(ModeType.TM, 24.0e9), 24.0e9)
    pulsed_overview("round WG TE11", round_wg_analysis(ModeType.TE, 14.0e9), 14.0e9)
    pulsed_overview("round WG TM01", round_wg_analysis(ModeType.TM, 17.25e9), 17.25e9)


if __name__ == "__main__":
    main()
