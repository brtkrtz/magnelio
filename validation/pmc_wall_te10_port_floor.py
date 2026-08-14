"""WP-U0 acceptance: TD PMC wall consistency on the parallel plate.

Before WP-U0 stage 1 the TD ``PMCBoundary`` zeroed tangential H on the
first/last *cell-centre* layer, placing the magnetic wall ~dx/2 inside
the outermost grid line, while both mode solvers (2D port modes,
``EigenmodeSolver3D``) use the natural boundary — wall dx/2 *outside*.
Measured consequence (session 94, this fixture): the TD TE(1,0)
transmission edge sat at ~15.8 GHz ~ c/2(a-dx) while the port mode's
discrete cut-off is 14.26 GHz = c/2(a+dx); between the two cut-offs
the S-matrix was non-passive (|S21| up to +14.6 dB) and in-band |S11|
never beat -17 dB.

Stage 1 makes the TD update natural too (``PMCBoundary`` is a no-op
marker; the free operator + full boundary dual cell IS the magnetic
wall dx/2 outside).  Three measurement legs certify the fix:

1. **CW lock-in TE(1,0) port floor** (the criterion measurement,
   ``kg_dtbc_wg_port_floors.py`` methodology).  Acceptance:
   |S11| < -100 dB from ``1.05 * f_c_hat`` upward.

2. **Pulsed broadband passivity** through the former double-cut-off
   window [14.28, 15.78] GHz.  NOTE the near-edge bins of a pulsed
   record are truncation-limited (algebraic band-edge ring-down, see
   ``kg_dtbc_wg_port_floors.py``); the reference for "passive within
   the measurement class" is the all-PEC WR-90 TE10 pulsed overview
   run through the identical pipeline.

3. **TEM compatibility.**  The TEM plate mode carries no tangential H
   on the PMC walls, so removing the legacy zeroing must not change
   TEM results beyond the floating-point floor.  The same high-level TEM
   fixture is run twice — production (natural) PMC vs. a local shim
   reproducing the legacy cell-centre zeroing — and S11/S21 are
   compared bit-for-bit.

Geometry: air brick a x b x L = 10 x 5 x 20 mm; PMC on the x faces,
PEC plates on the y faces, waveguide ports on the z faces.
dx = 0.5 mm -> natural-wall TE(1,0) discrete cut-off f_c_hat =
14.2625 GHz (c/2(a+dx) = 14.276 GHz; c/2a = 14.990 GHz is the stage-2
target once the mesher places the wall ON the bbox face).

Results (2026-07-11, WP-U0 stage 1):

    CW lock-in |S11|: -156.6 dB at 1.05 f_c_hat, monotone to
                      -165.2 dB at 1.8 f_c_hat  (criterion: < -100 dB)
    pulsed passivity: max |S21| +0.055 dB / max power sum 1.0589 at
                      the 1.001 f_c_hat edge bin, decaying to 1.0045
                      at 1.05 f_c_hat — bit-for-bit the class of the
                      accepted all-PEC WR-90 reference (+0.046 dB /
                      1.0594 / 1.0042); the pre-fix window showed
                      |S21| up to +14.6 dB.  Pulsed |S11| above
                      1.05 f_c_hat: max -56.9 dB, median -79.5 dB.
    TEM S11/S21:      max |delta| = 1.7e-14 vs. the legacy zeroing —
                      the double-precision floor (the legacy zeroing
                      erased ~1e-16 Laplace-solver noise in the wall
                      Hy/Hz each step; the natural wall lets it
                      evolve).  No bit-exact claim is possible, the
                      TEM channel is unchanged at the FP floor.

Run:  python validation/pmc_wall_te10_port_floor.py
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import erf

from magnelio import (
    AnalysisScatteringTD,
    Mesh,
)
from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.boundaries import PECBoundary, PMCBoundary
from magnelio.mesh.grid import GridLines
from magnelio.ports import PortSpecNumerical
from magnelio.ports._modal import (
    BoxFace,
    ModeType,
    PortSpecMultiConductor,
    build_modal_port,
)
from magnelio.ports._modal.dtbc import destagger_theta, dtbc_wave_impedance
from magnelio.ports.recorder import PortSignalRecorder
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import courant_dt

# Plate width a (between the PMC walls), plate spacing b, length L.
A = 10.0e-3
B = 5.0e-3
LENGTH = 20.0e-3
C0 = 299_792_458.0


def parallel_plate_mesh():
    grid = GridLines(
        x=np.linspace(-A / 2, A / 2, 21),  # dx = 0.5 mm, PMC walls
        y=np.linspace(-B / 2, B / 2, 11),  # PEC plates
        z=np.linspace(-LENGTH / 2, LENGTH / 2, 41),
    )
    mesh = Mesh.from_grid(grid).with_boundary_conditions(
        {
            "ymin": "PEC",
            "ymax": "PEC",
            "xmin": "PMC",
            "xmax": "PMC",
            "zmin": "PMC",
            "zmax": "PMC",
        }
    )
    return mesh, courant_dt(grid, "normal")


def lateral_bcs(grid):
    return {
        "xmin": PMCBoundary("xmin", grid),
        "xmax": PMCBoundary("xmax", grid),
        "ymin": PECBoundary("ymin"),
        "ymax": PECBoundary("ymax"),
    }


# ----------------------------------------------------------------------
# Leg 1 — CW lock-in TE(1,0) floor through the production solver
# ----------------------------------------------------------------------


def cw_lockin_floor(ratios):
    mesh, dt = parallel_plate_mesh()
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    nz = mesh.Nz
    f_calc = 20.0e9
    print("  TE(1,0) on the PMC parallel plate (CW lock-in)")
    for ratio in ratios:
        op1 = build_modal_port(
            PortSpecNumerical(name="port1", plane=BoxFace.Z_MIN, n_modes=1, mode_type=ModeType.TE),
            mesh,
            m_eps,
            m_mu,
            dt=dt,
            f_calc=f_calc,
        )
        op2 = build_modal_port(
            PortSpecNumerical(name="port2", plane=BoxFace.Z_MAX, n_modes=1, mode_type=ModeType.TE),
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
            boundary_conditions=lateral_bcs(mesh.grid),
            ports=[op1, op2],
            recorder=recorder,
            total_time_steps=n_steps,
            dt=dt,
            verbose=False,
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
        Z = dtbc_wave_impedance(np.array([w_dt]), q, z0, ModeType.TE.value)[0]
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
# Leg 2 — pulsed passivity through the former double-cut-off window
# ----------------------------------------------------------------------


def pulsed_passivity(f_max=20.0e9):
    mesh, dt = parallel_plate_mesh()
    analysis = AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions(
            {
                "xmin": "PMC",
                "xmax": "PMC",
                "ymin": "PEC",
                "ymax": "PEC",
                "zmin": "PEC",
                "zmax": "PEC",
            }
        ),
        ports=[
            PortSpecNumerical(name="port1", plane=BoxFace.Z_MIN, n_modes=1, mode_type=ModeType.TE),
            PortSpecNumerical(name="port2", plane=BoxFace.Z_MAX, n_modes=1, mode_type=ModeType.TE),
        ],
        f_max=f_max,
        verbose=False,
    )
    report = analysis.solve_ports()["port1"]
    f_c_hat = report.cutoff_num
    f_axis = np.linspace(1.001 * f_c_hat, f_max, 161)
    res = analysis.run(f_axis=f_axis)
    s11 = res.S("port1", "port1")
    s21 = res.S("port2", "port1")
    power = np.abs(s11) ** 2 + np.abs(s21) ** 2

    # Former double-cut-off window: mode-solver wall (natural, a+dx)
    # vs. the legacy TD wall (a-dx).
    dx = A / 20
    f_lo, f_hi = C0 / (2 * (A + dx)), C0 / (2 * (A - dx))
    win = (f_axis >= f_lo) & (f_axis <= f_hi)
    s21_db = 20.0 * np.log10(np.abs(s21) + 1e-300)
    print(
        f"  f_c_hat {f_c_hat / 1e9:.4f} GHz; former window [{f_lo / 1e9:.2f}, {f_hi / 1e9:.2f}] GHz"
    )
    print(
        f"    window: max |S21| {np.max(s21_db[win]):+7.2f} dB   "
        f"max |S11|^2+|S21|^2 = {np.max(power[win]):.6f}"
    )
    print(
        f"    band:   max |S21| {np.max(s21_db):+7.2f} dB   "
        f"max |S11|^2+|S21|^2 = {np.max(power):.6f}"
    )
    band = f_axis >= 1.05 * f_c_hat
    s11_db = 20.0 * np.log10(np.abs(s11) + 1e-300)
    print(
        f"    |S11| above 1.05 f_c_hat: max {np.max(s11_db[band]):7.1f}"
        f" dB   median {np.median(s11_db[band]):7.1f} dB"
        f"   (pulsed, truncation-limited near the edge)"
    )


# ----------------------------------------------------------------------
# Leg 3 — TEM bit-compatibility (natural PMC vs. legacy zeroing)
# ----------------------------------------------------------------------


class _LegacyPMCBoundary:
    """The pre-WP-U0 PMC: zero tangential H on the cell-centre layer."""

    def __init__(self, face: str) -> None:
        self.face = face

    def apply_E(self, fields) -> None:
        return

    def apply_H(self, fields) -> None:
        if self.face == "xmin":
            fields.Hy[0, :, :] = 0.0
            fields.Hz[0, :, :] = 0.0
        elif self.face == "xmax":
            fields.Hy[-1, :, :] = 0.0
            fields.Hz[-1, :, :] = 0.0
        else:
            raise ValueError(self.face)


def _tem_run(legacy: bool):
    grid = GridLines(
        x=np.linspace(-A / 2, A / 2, 21),
        y=np.linspace(-B / 2, B / 2, 11),
        z=np.linspace(-LENGTH / 2, LENGTH / 2, 41),
    )
    bcs: dict = {"ymin": "PEC", "ymax": "PEC", "zmin": "PEC", "zmax": "PEC"}
    if legacy:
        bcs["xmin"] = _LegacyPMCBoundary("xmin")
        bcs["xmax"] = _LegacyPMCBoundary("xmax")
    else:
        bcs["xmin"] = PMCBoundary("xmin", grid)
        bcs["xmax"] = PMCBoundary("xmax", grid)
    analysis = AnalysisScatteringTD(
        mesh=Mesh.from_grid(grid, boundary_conditions=bcs),
        ports=[
            PortSpecMultiConductor(name="port1", plane=BoxFace.Z_MIN, n_modes=1),
            PortSpecMultiConductor(name="port2", plane=BoxFace.Z_MAX, n_modes=1),
        ],
        f_max=10.0e9,
        verbose=False,
    )
    f_axis = np.linspace(0.25e9, 10.0e9, 81)
    res = analysis.run(f_axis=f_axis, excited=["port1"])
    return res.S("port1", "port1"), res.S("port2", "port1")


def tem_bit_compatibility():
    s_new = _tem_run(legacy=False)
    s_old = _tem_run(legacy=True)
    worst = max(float(np.max(np.abs(np.asarray(n) - np.asarray(o)))) for n, o in zip(s_new, s_old))
    if worst == 0.0:
        print("  TEM S11/S21 bit-identical (natural PMC vs. legacy cell-centre zeroing)")
    elif worst < 1e-12:
        print(
            f"  TEM S11/S21 at the FP floor: max |delta| = "
            f"{worst:.3e} (natural PMC vs. legacy zeroing; the "
            f"zeroing erased ~1e-16 solver noise in the wall Hy/Hz)"
        )
    else:
        print(f"  TEM S11/S21 DIFFER: max |delta| = {worst:.3e}")


def main() -> None:
    print(
        "WP-U0 stage 1 acceptance — natural TD PMC wall "
        "(criterion: |S11| < -100 dB from 1.05*f_c_hat):"
    )
    cw_lockin_floor((1.05, 1.1, 1.2, 1.5, 1.8))
    print()
    print("Pulsed passivity through the former double-cut-off window:")
    pulsed_passivity()
    print()
    print("TEM bit-compatibility:")
    tem_bit_compatibility()


if __name__ == "__main__":
    main()
