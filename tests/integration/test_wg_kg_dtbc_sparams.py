"""WP-R3 integration: Klein-Gordon DTBC port floors on hollow guides.

CW lock-in through the production solver/operators/recorder — the
certified measurement of the WP-R3 acceptance (pulsed band-edge
S-parameters are finite-record-truncation-limited on dispersive
lines; see ``validation/kg_dtbc_wg_port_floors.py``).  The
regression bound sits 20 dB below the −100 dB criterion, well above
the measured ~−150…−165 dB floors.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy.special import erf

from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.ports._modal import (
    BoxFace,
    ModeType,
    PortSpecNumerical,
    build_modal_port,
)
from magnelio.ports._modal.dtbc import destagger_theta, dtbc_wave_impedance
from magnelio.ports.recorder import PortSignalRecorder
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import courant_dt

WR90_A = 22.86e-3
WR90_B = 10.16e-3


def _wr90():
    grid = GridLines(
        x=np.linspace(0.0, WR90_A, 13),
        y=np.linspace(0.0, WR90_B, 7),
        z=np.linspace(0.0, 20e-3, 21),
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


def _cw_s11_db(mesh, dt, mode_type, f_calc, ratio):
    """|S11| at ``ratio * f_c_hat`` by production CW lock-in."""
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    ops = [
        build_modal_port(
            PortSpecNumerical(name=lbl, plane=face, n_modes=1, mode_type=mode_type),
            mesh,
            m_eps,
            m_mu,
            dt=dt,
            f_calc=f_calc,
        )
        for lbl, face in (("port1", BoxFace.Z_MIN), ("port2", BoxFace.Z_MAX))
    ]
    assert ops[0].termination_kinds == ["dtbc"]
    r, q, z0 = ops[0].dtbc_line_params[0]

    w_dt = ratio * q
    period = 2.0 * math.pi / w_dt
    sigma = max(6.0 / ((ratio - 1.0) * q), 8.0 * period)
    s_hat = math.sin(w_dt / 2.0)
    sin_b2 = math.sqrt(max(s_hat**2 - (q / 2.0) ** 2, 1e-30)) / r
    v_g = r * r * math.sin(2.0 * math.asin(min(sin_b2, 1.0))) / math.sin(w_dt)
    n_win = int(30 * period)
    n_meas0 = int(10.0 * sigma + 40.0 * period + 3.0 * mesh.Nz / max(v_g, 1e-3))
    n_steps = n_meas0 + n_win + 2

    t0 = 5.0 * sigma * dt
    sig_t = sigma * dt
    w_phys = w_dt / dt

    def waveform(t: float) -> float:
        amp = 0.5 * (1.0 + float(erf((t - t0) / (math.sqrt(2.0) * sig_t))))
        return amp * math.sin(w_phys * t)

    ops[0].set_excitation(0, waveform)
    recorder = PortSignalRecorder(dt=dt, ports=ops)
    solver = FITTimeDomainSolver(
        mesh=mesh,
        boundary_conditions={},
        ports=ops,
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
    V_sig, I_sig = recorder.finalize(n_steps_actual=n_steps)[("port1", 0)]

    n_grid = np.arange(n_meas0, n_meas0 + n_win)
    basis = np.column_stack([np.cos(w_dt * n_grid), np.sin(w_dt * n_grid)])
    cv, *_ = np.linalg.lstsq(basis, V_sig.values[n_grid], rcond=None)
    ci, *_ = np.linalg.lstsq(basis, I_sig.values[n_grid], rcond=None)
    V = cv[0] - 1j * cv[1]
    I = (ci[0] - 1j * ci[1]) * np.exp(1j * w_dt / 2.0)

    theta = destagger_theta(np.array([w_dt]), r, q)[0]
    Z = dtbc_wave_impedance(np.array([w_dt]), q, z0, mode_type.value)[0]
    sz = np.sqrt(Z)
    ep, em = np.exp(theta), np.exp(-theta)
    a = (V / sz * ep + sz * I) / (ep + em)
    b = (V / sz * em - sz * I) / (ep + em)
    return 20.0 * math.log10(max(abs(b / a), 1e-300))


class TestKGDTBCPortFloors:
    @pytest.mark.parametrize("ratio,bound_db", [(1.05, -120.0), (1.5, -130.0)])
    def test_te10_floor(self, ratio, bound_db):
        mesh, dt = _wr90()
        s11 = _cw_s11_db(mesh, dt, ModeType.TE, 10.0e9, ratio)
        assert s11 < bound_db, f"|S11| = {s11:.1f} dB at {ratio} f_c"

    def test_tm11_floor(self):
        mesh, dt = _wr90()
        s11 = _cw_s11_db(mesh, dt, ModeType.TM, 25.0e9, 1.05)
        assert s11 < -120.0, f"|S11| = {s11:.1f} dB"

    def test_te10_near_edge_floor(self):
        """The DD-047 Mur peak sat at −19 dB near cut-off; the exact
        DTBC must hold the acceptance criterion at 1.01 f_c_hat."""
        mesh, dt = _wr90()
        s11 = _cw_s11_db(mesh, dt, ModeType.TE, 10.0e9, 1.01)
        assert s11 < -110.0, f"|S11| = {s11:.1f} dB at 1.01 f_c"
