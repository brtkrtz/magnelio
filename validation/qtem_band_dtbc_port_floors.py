"""WP-R4b-impl acceptance: broadband band-subspace DTBC port floors.

The QTEM acceptance criterion (developer decision 2026-07-09) measured
from ONE pulsed run per port pair and excitation channel: build the
Galerkin band-subspace DTBC ports (``build_band_dtbc_port``, DD-057),
launch the flat-spectrum band pulse (``set_excitation_band``), record
V/I through the production solver, and decompose per frequency with
the WP-R4a true-mode machinery (``compute_band_s_parameters``) —
profile, zeta and discrete V/I response of the *true* mode per
S-parameter axis point, from a single broadband record.

Geometries: the WP-R4a acceptance set (imported from
``qtem_cw_dtbc_port_floors``).

* layered     — fundamental, axis 1.0-7.8 GHz (full band).
* layered2    — second hybrid family (two-channel dual-basis port,
                channel 1 excited), axis from 1.05 * f_c_hat
                (f_c_hat = 8.4465 GHz).  The 1.01 * f_c_hat point is
                anchored CW by WP-R4a (-176.3 dB): a pulsed record
                that close to the cut-on is finite-record limited
                (the WP-R3 measurement-methodology finding — the
                band-edge resonance decays algebraically), so the
                pulsed acceptance for higher families is evaluated
                from 1.05 * f_c_hat here.
* block       — fundamental, axis 2.1-6.2 GHz.
* microstrip  — fundamental, axis 1.0-7.8 GHz (production-sized
                cross-section; the dense a-priori certificate is
                skipped there, the subspace tail is reported
                instead).

For the small cross-sections the exact a-priori modal-reflection
certificate of the built boundary (``band_apriori_reflection``, the
WP-R4b gate formula) is evaluated on the tracked family points —
the measured pulsed floors must be compatible with it.

Method lessons carried in this benchmark (measured in session 90):

* The ghost source must track the family direction per frequency
  (fixed-profile injection excites the interface at the profile-
  drift level: -40 dB class |S11| at band edges).
* The excitation spectrum must fit inside the subspace band, and the
  spectral window must be Gaussian-edged (erfc product): a C^1
  window decays like t^-3 and is still active at the synthesis-
  window end, whose truncation kicks near-Nyquist grid modes the
  band boundary does not absorb.
* The record must cover the synthesis window plus ring-down.

Results (session 90), ONE pulsed run per case, |S21| = 0.00 dB
throughout, record end/peak 1e-11..1e-13:

    layered fundamental   1.0-7.8 GHz (18 pts):  -159.6 .. -231.3 dB
    layered 2nd family    1.05-1.28 * f_c_hat:   -166.7 .. -189.8 dB
    block                 2.1-6.2 GHz (12 pts):  -186.7 .. -202.8 dB
    microstrip            1.0-7.8 GHz (18 pts):  -171.1 .. -211.0 dB

All 59+ dB below the -100 dB criterion.  A-priori ceilings on the
family points -114 .. -125 dB (the rho-offset evaluation floor at
the cut-on grid points; the TD floors go deeper, as in the gate
spike).  Port builds 18-220 s (contour-QZ kernels dominate),
decomposition 0.3-262 s per axis (per-frequency sparse mode solves
on the production cross-section, the DD-056 cost class).

Run:  python validation/qtem_band_dtbc_port_floors.py
      [--case layered|layered2|block|microstrip|all] [--fast]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qtem_cw_dtbc_port_floors import (  # noqa: E402
    block_mesh,
    layered_mesh,
    microstrip_mesh,
)

from magnelio._operators.material_matrices import (  # noqa: E402
    build_M_eps,
    build_M_mu,
)
from magnelio.ports._modal import (  # noqa: E402
    BoxFace,
    PortSpecMultiConductor,
    build_band_dtbc_port,
)
from magnelio.ports._modal.band_dtbc import (  # noqa: E402
    band_apriori_reflection,
)
from magnelio.ports.recorder import PortSignalRecorder  # noqa: E402
from magnelio.post import (  # noqa: E402
    compute_band_s_parameters,
)
from magnelio.solver.fit_td import FITTimeDomainSolver  # noqa: E402


def db(x: float) -> float:
    return 20.0 * math.log10(max(x, 1e-300))


def apriori_ceiling(op, n_pts: int = 6) -> float:
    """Exact a-priori |Gamma| of the built boundary (dense; small N)."""
    bd = op.band_data
    worst = -1e9
    for fam in bd.families:
        step = max(fam.freqs.size // n_pts, 1)
        pts = [
            (float(fam.freqs[i]), complex(fam.zetas[i]), fam.traces[:, i])
            for i in range(0, fam.freqs.size, step)
        ]
        g = band_apriori_reflection(
            bd.chain_boundary,
            bd.exterior,
            bd.chain_boundary.dt,
            pts,
        )
        worst = max(worst, max(db(x) for x in g))
    return worst


def pulsed_case(
    name: str,
    mesh,
    dt: float,
    f_band: tuple[float, float],
    f_span: tuple[float, float],
    f_axis: np.ndarray,
    *,
    exc_channel: int = 0,
    n_grid: int = 25,
    n_syn: int = 8192,
    n_steps: int = 12288,
    n_kernel_init: int = 16384,
    apriori: bool = True,
) -> float:
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    ops = []
    t0 = time.perf_counter()
    for name, face in (("port1", BoxFace.Z_MIN), ("port2", BoxFace.Z_MAX)):
        ops.append(
            build_band_dtbc_port(
                PortSpecMultiConductor(name=name, plane=face, epsilon_r=None),
                mesh,
                m_eps,
                m_mu,
                dt=dt,
                f_band=f_band,
                n_grid=n_grid,
                n_kernel_init=n_kernel_init,
            )
        )
    t_build = time.perf_counter() - t0
    op1, op2 = ops
    bd = op1.band_data
    sv_tail = bd.singular_values[bd.p] if bd.p < bd.singular_values.size else 0.0
    print(
        f"  [{name}] ports built in {t_build:.0f} s: p = {bd.p}, "
        f"{len(bd.families)} families "
        f"(cut-ons "
        + ", ".join(f"{f.f_first / 1e9:.2f}" for f in bd.families)
        + f" GHz), sv tail {sv_tail:.1e}, "
        f"z_line {bd.z_line:.2f} Ohm"
    )
    if apriori:
        g_ap = apriori_ceiling(op1)
        print(f"  [{name}] a-priori max |Gamma| on family points: {g_ap:6.1f} dB")

    op1.set_excitation_band(exc_channel, f_span, n_syn=n_syn)
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
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*neither a BoundaryCondition.*",
        )
        solver.run()
    t_run = time.perf_counter() - t0
    signals = recorder.finalize(n_steps_actual=n_steps)
    v1 = signals[("port1", exc_channel)][0].values
    decay = float(np.abs(v1[-256:]).max() / np.abs(v1).max())
    print(f"  [{name}] 3D run {t_run:.0f} s ({n_steps} steps), record end/peak {decay:.1e}")

    t0 = time.perf_counter()
    S = compute_band_s_parameters(
        signals,
        [op1, op2],
        ("port1", exc_channel),
        f_axis,
    )
    t_dec = time.perf_counter() - t0
    worst = -1e9
    for k, f in enumerate(f_axis):
        s11 = abs(S[("port1", exc_channel)][k])
        s21 = abs(S[("port2", exc_channel)][k])
        s11_db = db(s11)
        worst = max(worst, s11_db)
        print(f"    f {f / 1e9:6.3f} GHz  |S11| {s11_db:8.1f} dB   |S21| {db(s21):7.2f} dB")
    print(
        f"  [{name}] max |S11| on the axis: {worst:6.1f} dB  "
        f"(criterion -100 dB; decomposition {t_dec:.1f} s "
        f"for {f_axis.size} points)"
    )
    return worst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--case", default="all", choices=["layered", "layered2", "block", "microstrip", "all"]
    )
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    cases = [args.case] if args.case != "all" else ["layered", "layered2", "block", "microstrip"]
    f_c2 = 8.4465e9  # layered second-family cut-on (WP-R4a)

    for name in cases:
        if name == "layered":
            nz = 24 if args.fast else 48
            mesh, dt = layered_mesh(nz=nz)
            n_ax = 6 if args.fast else 18
            pulsed_case(
                name,
                mesh,
                dt,
                (0.3e9, 9.6e9),
                (1.0e9, 7.8e9),
                np.linspace(1.0e9, 7.8e9, n_ax),
                n_grid=17 if args.fast else 25,
            )
        elif name == "layered2":
            nz = 24 if args.fast else 48
            mesh, dt = layered_mesh(nz=nz)
            ratios = [1.05, 1.07, 1.10, 1.20, 1.28]
            pulsed_case(
                name,
                mesh,
                dt,
                (0.3e9, 11.6e9),
                (1.05 * f_c2, 1.29 * f_c2),
                np.array([r * f_c2 for r in ratios]),
                exc_channel=1,
                n_grid=25 if args.fast else 49,
                n_syn=12288,
                n_steps=14336,
            )
        elif name == "block":
            nz = 24 if args.fast else 48
            mesh, dt = block_mesh(nz=nz)
            n_ax = 6 if args.fast else 12
            pulsed_case(
                name,
                mesh,
                dt,
                (0.8e9, 8.0e9),
                (2.1e9, 6.2e9),
                np.linspace(2.1e9, 6.2e9, n_ax),
                n_grid=17 if args.fast else 25,
            )
        elif name == "microstrip":
            # dt is ~5x smaller than on the 1 mm meshes, so the
            # synthesis window must grow with 1/dt for the same
            # physical roll-off duration (the compactness gate
            # enforces this).
            nz = 20 if args.fast else 40
            mesh, dt = microstrip_mesh(nz=nz)
            if args.fast:
                pulsed_case(
                    name,
                    mesh,
                    dt,
                    (0.25e9, 9.6e9),
                    (2.0e9, 7.0e9),
                    np.linspace(2.0e9, 7.0e9, 6),
                    n_grid=17,
                    n_syn=16384,
                    n_steps=20480,
                    n_kernel_init=32768,
                    apriori=False,
                )
            else:
                pulsed_case(
                    name,
                    mesh,
                    dt,
                    (0.25e9, 9.6e9),
                    (1.0e9, 7.8e9),
                    np.linspace(1.0e9, 7.8e9, 18),
                    n_grid=25,
                    n_syn=32768,
                    n_steps=36864,
                    n_kernel_init=65536,
                    apriori=False,
                )
        print()


if __name__ == "__main__":
    main()
