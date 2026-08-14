"""WR-90 TE10 modal-port |S11| floor — DD-047 re-evaluation post-DD-048.

DD-047 (session 54) measured peak |S11| ≈ −19 dB / best-case ≈ −28 dB
on a straight, empty WR-90 with both ports modal and source-driven
across [8.2, 12.4] GHz; with a 2× finer mesh, −33 dB centre / −19 dB
edges.  That measurement was the trigger for DD-047's Phase-3
co-simulation plan (Luo-Chen).

Session 57 fixed five bugs in the modal-port pipeline (see
``commit bbd7a07``); session 58 (DD-048) split the modal-port
pipeline into a reference path (analytical) and an operator-consistent
path (numerical 2D mode solver on the 3D-mesh transversal slice).
This script re-measures the WR-90 / TE10 |S11| floor on the post-DD-048
codebase to determine whether the DD-047 Phase-3 trigger condition
still holds.

Session 68 (Level-A rework WP2.4): after WP1 identified the spatial
half-cell stagger of the I sampling plane as a −22 dB measurement
artefact for λ/20 meshes, each variant now evaluates
``compute_s_parameters`` twice — with the historical co-located a/b
decomposition and with the exact two-plane de-stagger
(``port_normal_dx``) — to separate the measurement artefact from the
genuine Mur-1st absorber floor.  Result: the de-stagger lifts the
in-band *median* to ≈ −28 dB on both meshes, but the peak at the
8.2 GHz band edge stays at −19 dB (β → 0 near cutoff, so the stagger
term vanishes exactly where the Mur floor peaks).  The DD-047 trigger
condition for TE/TM therefore still holds.

Setup mirrors the session-54 benchmark:
- WR-90 dimensions (a = 22.86 mm, b = 10.16 mm), L_x = 30 mm, vacuum.
- Both ports are ``PortSpecRectWG(n_modes=1)`` (TE10).
- Source: ``ExcitationSpec`` modulated Gaussian on [8.2, 12.4] GHz.
- Lateral PEC walls via ``PECBoundary`` on the four lateral bbox faces.
- Two mesh resolutions: baseline (24×11 transversal) and 2× refined.
"""

from __future__ import annotations

import math

import numpy as np

from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.boundaries.pec import PECBoundary
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.ports import PortSignalRecorder
from magnelio.ports._modal import (
    BoxFace,
    ExcitationSpec,
    PortSpecRectWG,
    build_modal_port,
)
from magnelio.post import compute_s_parameters
from magnelio.signals.signal_1d import Signal1D
from magnelio.signals.waveforms import modulated_gaussian
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import courant_dt

WR90_A = 22.86e-3
WR90_B = 10.16e-3
L_X = 30e-3
F_MIN = 8.2e9
F_MAX = 12.4e9
F_CALC = 10.0e9
C0 = 299_792_458.0


def run_variant(label: str, refine: int = 1) -> None:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    n_x = 31 * refine
    n_y = 24 * refine
    n_z = 11 * refine
    grid = GridLines(
        x=np.linspace(0.0, L_X, n_x),
        y=np.linspace(0.0, WR90_A, n_y),
        z=np.linspace(0.0, WR90_B, n_z),
    )
    mesh = Mesh.from_grid(grid)
    print(
        f"  Mesh: {mesh.Nx}x{mesh.Ny}x{mesh.Nz} = "
        f"{(mesh.Nx - 1) * (mesh.Ny - 1) * (mesh.Nz - 1)} cells (refine={refine}x)"
    )
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    dt = courant_dt(mesh.grid, accuracy="normal")

    excitation = ExcitationSpec(f_min=F_MIN, f_max=F_MAX, mode_index=0)
    spec_src = PortSpecRectWG(
        name="port1",
        plane=BoxFace.X_MIN,
        width_a=WR90_A,
        height_b=WR90_B,
        n_modes=1,
        excitation=excitation,
    )
    spec_load = PortSpecRectWG(
        name="port2",
        plane=BoxFace.X_MAX,
        width_a=WR90_A,
        height_b=WR90_B,
        n_modes=1,
    )
    op_src = build_modal_port(spec_src, mesh, m_eps, m_mu, dt=dt, f_calc=F_CALC)
    op_load = build_modal_port(spec_load, mesh, m_eps, m_mu, dt=dt, f_calc=F_CALC)
    print(f"  Path-(a) cutoff_ref = {op_src.port_report.cutoff_ref / 1e9:.4f} GHz")
    print(f"  Path-(b) cutoff_num = {op_src.port_report.cutoff_num / 1e9:.4f} GHz")

    f_c_te10 = C0 / (2.0 * WR90_A)
    v_g_calc = C0 * math.sqrt(max(0.0, 1.0 - (f_c_te10 / F_CALC) ** 2))
    bandwidth = F_MAX - F_MIN
    t0_pulse = 4.0 / bandwidth
    t_traversal = L_X / v_g_calc
    t_total = 2.5 * t0_pulse + 5.0 * t_traversal
    n_steps = int(round(t_total / dt))
    print(f"  Solver: dt={dt * 1e15:.2f} fs, n_steps={n_steps}")

    recorder = PortSignalRecorder(dt=dt, ports=[op_src, op_load])
    bcs = {
        "ymin": PECBoundary("ymin"),
        "ymax": PECBoundary("ymax"),
        "zmin": PECBoundary("zmin"),
        "zmax": PECBoundary("zmax"),
    }
    solver = FITTimeDomainSolver(
        mesh=mesh,
        dt=dt,
        total_time_steps=n_steps,
        ports=[op_src, op_load],
        recorder=recorder,
        boundary_conditions=bcs,
        verbose=False,
    )
    solver.run()

    signals = recorder.finalize()
    ref_t = np.arange(recorder.n_steps_recorded) * dt
    ref_values = np.array([float(modulated_gaussian(float(t), F_MAX, F_MIN)) for t in ref_t])
    ref_sig = Signal1D(t=ref_t, values=ref_values, dt=dt, label="exc")

    f_axis = np.linspace(F_MIN, F_MAX, 81)
    port_modes = {
        "port1": [dm.mode for dm in op_src.discrete_modes],
        "port2": [dm.mode for dm in op_load.discrete_modes],
    }
    normal_dx = {
        "port1": op_src.plane.normal_dx,
        "port2": op_load.plane.normal_dx,
    }
    for tag, kwargs in (
        ("co-located a/b (historical)", {}),
        ("de-staggered a/b (port_normal_dx)", {"port_normal_dx": normal_dx}),
    ):
        S = compute_s_parameters(
            recorder_signals=signals,
            port_modes=port_modes,
            excited=("port1", 0),
            reference_signal=ref_sig,
            f_axis=f_axis,
            **kwargs,
        )
        s11 = np.abs(S[("port1", 0)])
        s21 = np.abs(S[("port2", 0)])
        sum_sq = s11**2 + s21**2
        s11_db = 20.0 * np.log10(s11 + 1e-300)
        s21_db = 20.0 * np.log10(s21 + 1e-300)

        print(f"  Spectrum 8.2–12.4 GHz, 81 points — {tag}:")
        print(
            f"    max |S11|       = {np.max(s11_db):+.2f} dB  "
            f"(at {f_axis[np.argmax(s11_db)] / 1e9:.2f} GHz)"
        )
        print(f"    median |S11|    = {np.median(s11_db):+.2f} dB")
        print(
            f"    min  |S21|      = {np.min(s21_db):+.2f} dB  "
            f"(at {f_axis[np.argmin(s21_db)] / 1e9:.2f} GHz)"
        )
        print(f"    max  |S21|      = {np.max(s21_db):+.2f} dB")
        print(f"    max  |S|2 sum   = {np.max(sum_sq):.4f}")


if __name__ == "__main__":
    run_variant("WR-90 / TE10 — baseline mesh (31x24x11)", refine=1)
    run_variant("WR-90 / TE10 — 2x refined mesh (61x47x21)", refine=2)
