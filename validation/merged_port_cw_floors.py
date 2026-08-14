"""WP-U5 acceptance: CW port floors of the merged TEM (+) TE/TM ports.

The WP-U2 unified multi-mode port serves the ``n_modes`` lowest modes
of a conductor cross-section across families (TEM channels from the
Laplace solve, TE/TM channels from the curl-curl solve, one operator,
per-channel DTBC).  Acceptance (PORT_MODES_PLAN.md WP-U5): CW lock-in
|S11| < -100 dB above 1.05 f_c_hat on the certified chains — the
``kg_dtbc_wg_port_floors.py`` methodology, measured per channel of
the *merged* port:

* RG-58 coax, 3-mode port (TEM + TE11 degenerate pair): the TE11
  channel, and the TEM channel at spot frequencies.
* Shielded two-wire, 4-mode port (2 TEM + 2 TE): the first TE
  channel.

Pulsed S-parameters near the cut-offs stay truncation-limited as
documented (kg benchmark); no acceptance number is read off a pulsed
band edge.

Results (2026-07-11, WP-U2/U5; coax TE11 re-measured after the
DD-067 port-plane mu-flatten):

    coax TEM (channel 0):       -144.6 dB at both spot frequencies
    coax TE11 (channel 1):      -134.3 / -139.5 / -142.0 dB at
                                1.05/1.2/1.5 f_c_hat.  Before DD-067
                                this channel sat at -34.8/-42.1/
                                -49.2 dB: the conformal boundary-slab
                                Hz-M_mu deviated 36 % from the
                                interior feed (halved cell
                                neighbourhood on the bbox face), so
                                the 2D mode was solved against a
                                different transversal operator than
                                the volume propagates — invisible to
                                the transversal pair-product chain
                                certificate.  Fixed by
                                ``flatten_port_plane_mu`` (factory +
                                TD solver), guarded by the DD-067
                                slab-consistency certificate stage.
    two-wire TE#1 (channel 2):  -149.5 / -154.6 / -157.2 dB
                                (acceptance < -100 dB: 49 dB margin)
    two-wire TEM odd (ch 0):    -159.7 dB at both spot frequencies

Run:  python validation/merged_port_cw_floors.py
"""

from __future__ import annotations

import math

# Geometry fixtures shared with the WP-U1 certification benchmark.
import os
import sys
import warnings

import numpy as np
from scipy.special import erf

from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.ports._modal import (
    BoxFace,
    PortSpecMultiConductor,
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from curlcurl_conductor_certification import (  # noqa: E402
    EPS_R,
    coax_mesh,
    two_wire_mesh,
)


def _mesh_dt(mesh):
    return courant_dt(
        mesh.grid,
        "normal",
        min_effective_eps=compute_min_effective_eps(mesh),
        min_effective_mu=compute_min_effective_mu(mesh),
    )


def cw_channel_floor(name, mesh, spec, f_calc, channel, ratios):
    """|S11| of one channel of a merged port by CW lock-in."""
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    dt = _mesh_dt(mesh)
    nz = mesh.Nz
    print(f"  {name} (CW lock-in, channel {channel})")
    for ratio in ratios:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*degenerate.*")
            op1 = build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=f_calc)
            spec2 = type(spec)(**{**spec.__dict__, "name": "port2", "plane": BoxFace.Z_MAX})
            op2 = build_modal_port(spec2, mesh, m_eps, m_mu, dt=dt, f_calc=f_calc)
        assert op1.termination_kinds[channel] == "dtbc"
        r, q, z0 = op1.dtbc_line_params[channel]
        mode_type = op1.discrete_modes[channel].mode.mode_type.value

        if q > 0.0:
            w_dt = ratio * q
        else:
            # TEM channel: spot frequency as a fraction of the
            # first higher channel's cut-off instead.
            q_ref = min(qq for _, (rr, qq, _) in op1.dtbc_line_params.items() if qq > 0.0)
            w_dt = ratio * q_ref
        period = 2.0 * math.pi / w_dt
        if q > 0.0:
            sigma = max(6.0 / ((ratio - 1.0) * q), 8.0 * period)
            s_hat = math.sin(w_dt / 2.0)
            sin_b2 = math.sqrt(max(s_hat**2 - (q / 2.0) ** 2, 1e-30)) / r
            v_g = r * r * math.sin(2.0 * math.asin(min(sin_b2, 1.0))) / math.sin(w_dt)
        else:
            sigma = 8.0 * period
            v_g = r
        n_win = int(30 * period)
        n_meas0 = int(10.0 * sigma + 40.0 * period + 3.0 * nz / max(v_g, 1e-3))
        n_steps = n_meas0 + n_win + 2

        t0 = 5.0 * sigma * dt
        sig_t = sigma * dt
        w_phys = w_dt / dt

        def waveform(t: float) -> float:
            amp = 0.5 * (1.0 + float(erf((t - t0) / (math.sqrt(2.0) * sig_t))))
            return amp * math.sin(w_phys * t)

        op1.set_excitation(channel, waveform)
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
        V_sig, I_sig = signals[("port1", channel)]

        n_grid = np.arange(n_meas0, n_meas0 + n_win)
        basis = np.column_stack([np.cos(w_dt * n_grid), np.sin(w_dt * n_grid)])
        cv, *_ = np.linalg.lstsq(basis, V_sig.values[n_grid], rcond=None)
        ci, *_ = np.linalg.lstsq(basis, I_sig.values[n_grid], rcond=None)
        V = cv[0] - 1j * cv[1]
        I = (ci[0] - 1j * ci[1]) * np.exp(1j * w_dt / 2.0)

        theta = destagger_theta(np.array([w_dt]), r, q)[0]
        if q > 0.0:
            Z = dtbc_wave_impedance(np.array([w_dt]), q, z0, mode_type)[0]
        else:
            # TEM channel: the calibrated V/I is frequency-flat
            # (DD-054); the reference impedance is the mode's z_line.
            Z = op1.discrete_modes[channel].mode.z_line
        sz = np.sqrt(Z)
        ep, em = np.exp(theta), np.exp(-theta)
        a = (V / sz * ep + sz * I) / (ep + em)
        b = (V / sz * em - sz * I) / (ep + em)
        s11_db = 20.0 * math.log10(max(abs(b / a), 1e-300))
        f_ghz = w_dt / dt / (2.0 * math.pi * 1e9)
        print(
            f"    f {f_ghz:7.3f} GHz (ratio {ratio:5.3f})   "
            f"|S11| {s11_db:8.1f} dB   ({n_steps} steps)"
        )


def main() -> None:
    print("WP-U2/U5 — merged-port CW floors (criterion: |S11| < -100 dB from 1.05*f_c_hat):")
    mesh = coax_mesh(0.12e-3)
    spec = PortSpecMultiConductor(name="port1", plane=BoxFace.Z_MIN, epsilon_r=EPS_R, n_modes=3)
    cw_channel_floor(
        "coax TEM+TE11 merged port / TE11", mesh, spec, 45.0e9, channel=1, ratios=(1.05, 1.2, 1.5)
    )
    cw_channel_floor(
        "coax TEM+TE11 merged port / TEM", mesh, spec, 45.0e9, channel=0, ratios=(0.3, 0.7)
    )

    mesh = two_wire_mesh()
    spec = PortSpecMultiConductor(name="port1", plane=BoxFace.Z_MIN, epsilon_r=1.0, n_modes=4)
    cw_channel_floor(
        "two-wire 2TEM+2TE merged port / TE#1",
        mesh,
        spec,
        25.0e9,
        channel=2,
        ratios=(1.05, 1.2, 1.5),
    )
    cw_channel_floor(
        "two-wire 2TEM+2TE merged port / TEM odd", mesh, spec, 25.0e9, channel=0, ratios=(0.3, 0.7)
    )


if __name__ == "__main__":
    main()
