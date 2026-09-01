"""DD-238: the modal-Mur QTEM port floor, measured and decomposed.

DD-237 priced the shipped Mur-1st boundary of the modal port a priori --
pure arithmetic on the true discrete mode, no time stepping:

    Gamma(w) = (lambda~(w) - zeta(w)) / (1/zeta(w) - lambda~(w)),
    lambda~_Mur(z) = (1 + r z) / (z + r),   r = op.mur_r[m],  z = exp(i w)

and listed two "free gains" on top of it: fit ``r`` to the TRUE DISCRETE
phase advance, ``r_exact = sin((w - theta)/2) / sin((w + theta)/2)`` with
``theta = |angle(zeta)|``, then minimax it over the band (+3.8 + 2.2 dB
predicted on the production microstrip).  DD-237 demanded a measured run
before shipping either.  This certificate is that run, and it REFUTES the
item: fitting ``r`` alone makes every measured point WORSE.

What it measures.  ``|a2/b2|``, the one-port reflectometer reading at the
PASSIVE port under test, driven from the far end by an exact-DTBC
true-mode generator whose own floor is -145..-176 dB.  CW erf-ramped
sine, true-mode lock-in decomposition (``cw_wave_phasors`` ->
``cw_lockin_phasors`` -> ``cw_decompose``), never ``compute_s_parameters``:
the ordinary de-staggered a/b split caps at -40..-60 dB, above the effect.
``courant_dt``, ``f_calc = f_cw`` per point, one QTEM channel per port,
``termination_kinds`` asserted per point.

The port under test must be the PASSIVE one.  Driven, with a matched far
end, there is no returning wave at all and ``|b1/a1|`` prices the source's
profile instead of the boundary (block 4.2 GHz reads -71.5 dB shipped /
-159.5 dB true-mode that way, i.e. the boundary law is invisible).
``drive-side`` = ``|b1/a1|`` is reported as a second, independent read of
the same reflection, one transit later and propagating content only.

Variants of port 2 -- same mesh, same dt, same driver, same postprocessing:

  "mur"              ``build_modal_port`` -- THE SHIPPED PATH: quasi-static
                     Laplace mode profile, Mur-1st with ``r`` from the
                     continuum phase velocity.
  "mur_rexact"       shipped profile, ``r`` replaced by ``r_exact`` --
                     DD-237's proposed gain, at its best case (fitted at
                     the run frequency itself).
  "mur_true"         true discrete mode profile (``build_cw_true_mode_port``
                     with its DTBC terminations dropped), that port's own
                     continuum ``r`` -- not the shipped one, which belongs
                     to the quasi-static profile.
  "mur_true_rexact"  true profile AND ``r_exact`` -- both terms corrected.
  "dtbc"             ``build_cw_true_mode_port`` -- INSTRUMENT FLOOR.
  "mur_frac_<pct>"   shipped profile, ``r`` moved ``pct`` percent of the way
                     from the shipped value to ``r_exact`` (``--alpha``).

THE FINDING.  ``build_modal_port`` derives BOTH the Mur coefficient AND the
mode profile from the SAME frequency-flat quasi-static Laplace mode.  Both
are wrong by comparable amounts and measured 161-180 degrees out of phase,
so the shipped floor is the residual of their near-perfect destructive
interference, not a boundary defect.  Exactly, block 6.2 GHz, |Gamma|:

    boundary term alone   4.103e-3  (-47.74 dB)   a priori, shipped r
    profile  term alone   2.443e-3  (-52.25 dB)   measured "mur_rexact"
    antiphase residual    1.660e-3  (-55.60 dB)   == measured "mur", 3 digits

The certificate recomputes that reconciliation from its own run and prints
the inferred angle between the two terms: 180.0 deg on block (where the
magnitudes subtract exactly), 174.1 / 174.7 deg on the layered and
microstrip mid-band points, and 161.3 deg on the layered line at 7.8 GHz,
where the closure loosens by the same amount (|difference| -42.36 dB
against a measured -36.48 dB).

Pinned numbers (2026-09-01, HEAD 4097abe).  Shipped floor, |a2/b2| in dB,
worst at band top and monotone in f:

    microstrip  -87.54 (1.0) -74.67 (2.1) -62.69 (4.2) -56.07 (6.2) -52.26 (7.8 GHz)
    layered     -78.58 (1.0) -65.63 (2.1) -52.93 (4.2) -43.83 (6.2) -36.48 (7.8 GHz)
    block       -74.36 (2.1) -67.05 (3.2) -62.34 (4.2) -58.64 (5.2) -55.60 (6.2 GHz)

DD-237's a-priori worst-in-band said -38.9 / -27.2 / -47.7 dB: the scalar
symbol is an EXACT instrument for the boundary term (on a true discrete
profile, "mur_true" equals its own a-priori Gamma to 0.01 dB) but it prices
only ONE of the port's two error terms, so it is pessimistic by 7.5-14.9 dB
and is NOT a bound on the port floor.

The 2x2 decomposition plus the control (block, the cheapest fixture):

    variant            2.1 GHz    4.2 GHz    6.2 GHz
    mur                 -74.36     -62.34     -55.60   shipped
    mur_rexact          -71.64     -59.37     -52.25   boundary corrected
    mur_true            -78.60     -66.50     -59.65   profile corrected
    mur_true_rexact    -156.79    -157.32    -154.39   both -> instrument floor
    dtbc               -157.02    -159.73    -156.06   exact transparent boundary

Correcting either term alone destroys the cancellation; correcting both
reaches the instrument floor, so there is no third error term.

Partial step toward ``r_exact`` (``--alpha``), |a2/b2| in dB:

    fixture     f/GHz   alpha=0    0.25     0.50     0.75     1.00
    block        2.1    -74.36   -82.13   -89.02   -76.56   -71.64
    block        6.2    -55.60   -63.80   -67.76   -56.96   -52.25
    layered      1.0    -78.58   -90.36   -75.27      --    -66.73
    layered      4.2    -52.93   -56.80   -48.76      --    -41.04
    layered      7.8    -36.48   -37.21   -34.50      --    -28.85
    microstrip   4.2    -62.69   -55.87   -51.88      --    -47.01
    microstrip   7.8    -52.26   -47.10   -43.05      --    -37.95

alpha = 1 is worse at 8/8 points (2.7-3.4 dB block, 7.6-11.9 layered,
14.3-15.7 microstrip).  The optimum is INTERIOR and fixture- and
frequency-dependent -- about 0.50 on block, about 0.25 on layered and
drifting down with frequency, at most 0 on the production microstrip --
because it sits where the two magnitudes are equal, and the profile term is
unknown without a discrete mode solve per port per frequency, which IS the
DTBC/band path (DD-057).  The shipped ``r`` sits near a cancellation optimum
by accident, and any future improvement to the mode profile alone would make
the floor worse unless ``r`` moves in the same step.

Line length is not a factor: microstrip 4.2 GHz measures -62.69 dB on both
Nz=24 (nz=8) and Nz=120 (the shipped nz=40), 0.00 dB apart.

Relation to DD-047.  DD-047 already lists as a do-not-revive non-path
"phase-velocity calibration in the Mur formula: ~0.05 dB", measured on a
HOMOGENEOUS hollow WR-90 where ``r`` was already nearly right.  The
measurement here generalises that non-path rather than contradicting it: on
an inhomogeneous QTEM line ``r`` is genuinely wrong, but the profile is
wrong by a comparable amount with the opposite sign, so calibrating ``r``
alone is not merely worthless but harmful.  DD-047's own text stands
unchanged; the strengthening is recorded in DD-238.

Acceptance:
  1. every pinned point above reproduces within 0.5 dB (2.0 dB where the
     cancellation is deep, i.e. pinned below -80 dB);
  2. "mur_true_rexact" and "dtbc" stay below -140 dB, i.e. at least 90 dB of
     instrument headroom under every Mur number;
  3. the block 6.2 GHz reconciliation closes to within 0.05 dB of the
     measured shipped floor;
  4. lock-in residuals stay below 1e-5;
  5. no run point raised (a point that raised is skipped by (1), so it is
     counted separately on line "(1b)").

This certificate measures the one-port reflectometer reading only.  The
user-visible S11 of a production run is a different quantity: it passes
through the de-staggered a/b split and, in a two-port run, also carries the
drive port's soft-source error.  Nothing here confirms or contradicts the
-30 dB class figures of DD-064.

Derivation and the full sweep: internal record `investigations/qtem-midpath/`
(`DERIVATION.md` section 9.5, `probe_mur_floor_a..d.py`).

Cost: the block sweep above is 58 s for 25 points (5 frequencies x 5
variants); the layered sweep is of the same order, and one microstrip point
costs 22 s per variant.

Run:  python validation/qtem_mur_floor_decomposition.py
      [--case block|layered|microstrip|all] [--freqs 2.1,6.2]
      [--variants mur,mur_rexact,mur_true,mur_true_rexact,dtbc]
      [--alpha 25,50,75] [--nz N] [--fast] [--out rows.json]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
import warnings

import numpy as np
from scipy.special import erf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qtem_cw_dtbc_port_floors import (  # noqa: E402
    block_mesh,
    layered_mesh,
    microstrip_mesh,
)

from magnelio._operators.curl import build_curl_matrix  # noqa: E402
from magnelio._operators.material_matrices import (  # noqa: E402
    build_M_eps,
    build_M_mu,
    flatten_port_plane_mass,
    flatten_port_plane_mu,
    flatten_port_plane_pec_mask,
)
from magnelio.constants import C0  # noqa: E402
from magnelio.ports._modal import BoxFace, PortSpecMultiConductor  # noqa: E402
from magnelio.ports._modal.factory import (  # noqa: E402
    build_cw_true_mode_port,
    build_modal_port,
)
from magnelio.ports._modal.port_plane import PortPlane  # noqa: E402
from magnelio.ports._modal.zeta_pencil import (  # noqa: E402
    build_period_blocks,
    cw_decompose,
    cw_lockin_phasors,
    cw_wave_phasors,
    find_propagating_modes,
    make_channel,
)
from magnelio.ports.recorder import PortSignalRecorder  # noqa: E402
from magnelio.solver.fit_td import FITTimeDomainSolver  # noqa: E402

FACE_DRIVE = BoxFace.Z_MIN
FACE_TEST = BoxFace.Z_MAX

CASES = {
    "block": (block_mesh, (2.1e9, 3.2e9, 4.2e9, 5.2e9, 6.2e9)),
    "layered": (layered_mesh, (1.0e9, 2.1e9, 4.2e9, 6.2e9, 7.8e9)),
    "microstrip": (microstrip_mesh, (1.0e9, 2.1e9, 4.2e9, 6.2e9, 7.8e9)),
}

DEFAULT_VARIANTS = "mur,mur_rexact,mur_true,mur_true_rexact,dtbc"

# Pinned |a2/b2| in dB, keyed (case, GHz, variant).  Reproduced by the
# acceptance check below; never edited to match a run (DD-238).
PINNED = {
    ("block", 2.1): {
        "mur": -74.36,
        "mur_rexact": -71.64,
        "mur_true": -78.60,
        "mur_frac_25": -82.13,
        "mur_frac_50": -89.02,
        "mur_frac_75": -76.56,
    },
    ("block", 3.2): {"mur": -67.05, "mur_true": -71.26},
    ("block", 4.2): {"mur": -62.34, "mur_rexact": -59.37, "mur_true": -66.50},
    ("block", 5.2): {"mur": -58.64, "mur_true": -62.75},
    ("block", 6.2): {
        "mur": -55.60,
        "mur_rexact": -52.25,
        "mur_true": -59.65,
        "mur_frac_25": -63.80,
        "mur_frac_50": -67.76,
        "mur_frac_75": -56.96,
    },
    ("layered", 1.0): {
        "mur": -78.58,
        "mur_rexact": -66.73,
        "mur_true": -89.74,
        "mur_frac_25": -90.36,
        "mur_frac_50": -75.27,
    },
    ("layered", 2.1): {"mur": -65.63, "mur_true": -76.77},
    ("layered", 4.2): {
        "mur": -52.93,
        "mur_rexact": -41.04,
        "mur_true": -64.35,
        "mur_frac_25": -56.80,
        "mur_frac_50": -48.76,
    },
    ("layered", 6.2): {"mur": -43.83, "mur_true": -56.92},
    ("layered", 7.8): {
        "mur": -36.48,
        "mur_rexact": -28.85,
        "mur_true": -52.14,
        "mur_frac_25": -37.21,
        "mur_frac_50": -34.50,
    },
    ("microstrip", 1.0): {"mur": -87.54},
    ("microstrip", 2.1): {"mur": -74.67},
    ("microstrip", 4.2): {
        "mur": -62.69,
        "mur_rexact": -47.01,
        "mur_frac_25": -55.87,
        "mur_frac_50": -51.88,
    },
    ("microstrip", 6.2): {"mur": -56.07},
    ("microstrip", 7.8): {
        "mur": -52.26,
        "mur_rexact": -37.95,
        "mur_frac_25": -47.10,
        "mur_frac_50": -43.05,
    },
}

FLOOR_VARIANTS = ("dtbc", "mur_true_rexact")
FLOOR_DB = -140.0
RESIDUAL_MAX = 1e-5


def db(x: float) -> float:
    return 20.0 * math.log10(max(float(x), 1e-300))


def lin(x_db: float) -> float:
    return 10.0 ** (float(x_db) / 20.0)


# ----------------------------------------------------------------------
# Analysis chain (production-faithful)
# ----------------------------------------------------------------------


def make_chain(mesh, dt, face):
    """PeriodChain plus the flattened masses the port was built on.

    ``build_modal_port`` flattens the port-plane slab of ``m_eps`` /
    ``m_mu`` / the PEC mask into locals and never exposes them, so the
    analysis side must redo the identical flatten or its projection
    weights differ from the operator's.
    """
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    plane = PortPlane.from_mesh(face, mesh, window=None)
    m_eps_f = flatten_port_plane_mass(m_eps, mesh, face)
    m_mu_f = flatten_port_plane_mu(m_mu, mesh, face)
    mesh2 = dataclasses.replace(
        mesh,
        pec_mask_edges=flatten_port_plane_pec_mask(mesh.pec_mask_edges, mesh, face),
    )
    c_3d = build_curl_matrix(mesh.grid)
    chain = build_period_blocks(plane, mesh2, m_eps_f, m_mu_f, c_3d, dt)
    return chain, plane, m_eps_f, m_mu_f, c_3d


def analysis_channel(op, chain, plane, m_eps_f, m_mu_f, c_3d, w_dt, mode=0):
    """True discrete mode of the cross-section, projected as the port projects.

    The channel is picked out of the propagating family by overlap with the
    port's own stored mode profile in the ``W_t`` metric (the factory picks
    the fundamental the same way, with the DC Laplace profile as tracker).
    The phasors are synthesised through the port's stored, post-calibration
    profiles and dual projections, so the decomposition sees exactly what
    the recorder sees for a unit incident discrete wave.
    """
    dm = op.discrete_modes[mode]
    n_t = chain.n_t
    dz = float(plane.normal_dx)
    track = np.concatenate(
        [
            np.asarray(dm.e_u_profile)[chain.free_u],
            np.asarray(dm.e_v_profile)[chain.free_v],
        ]
    )
    eps_hint = max(float(dm.mode.epsilon_r), 1.0)
    theta0 = w_dt / chain.dt * math.sqrt(eps_hint) / C0 * dz
    zp, pp = find_propagating_modes(chain, w_dt, 1.3 * theta0)
    if zp.size == 0:
        raise ValueError("no propagating mode at this frequency")
    w_t = chain.w_period[:n_t]
    ov = np.abs(track @ (w_t[:, None] * pp[:n_t, :]))
    j = int(np.argmax(ov))
    ch = make_channel(complex(zp[j]), pp[:, j], chain, w_dt, dz)
    dual = op._dual_e_profiles[mode] if op._dual_e_profiles is not None else None
    proj_u, proj_v = dual if dual is not None else (dm.e_u_profile, dm.e_v_profile)
    ch = cw_wave_phasors(
        ch,
        chain,
        plane,
        m_eps_f,
        m_mu_f,
        c_3d,
        w_dt,
        h_u_prof=dm.h_u_profile,
        h_v_prof=dm.h_v_profile,
        proj_u=proj_u,
        proj_v=proj_v,
    )
    return ch, int(zp.size)


# ----------------------------------------------------------------------
# A priori boundary symbol (DD-237)
# ----------------------------------------------------------------------


def lam_mur(z: complex, r_mur: float) -> complex:
    return (1.0 + r_mur * z) / (z + r_mur)


def gamma_apriori(zeta: complex, r_mur: float, w_dt: float) -> float:
    z = complex(np.exp(1j * w_dt))
    lam = lam_mur(z, r_mur)
    return float(abs(lam - zeta) / abs(1.0 / zeta - lam))


def r_exact(zeta: complex, w_dt: float) -> float:
    """Mur coefficient that matches the TRUE discrete phase advance."""
    theta = abs(float(np.angle(zeta)))
    return math.sin((w_dt - theta) / 2.0) / math.sin((w_dt + theta) / 2.0)


# ----------------------------------------------------------------------
# One CW frequency point
# ----------------------------------------------------------------------


def run_point(name, mesh, dt, f, variant, verbose=True):
    """Drive on the exact true-mode DTBC port, measure the far end."""
    t_all0 = time.perf_counter()
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    spec1 = PortSpecMultiConductor(name="port1", plane=FACE_DRIVE, epsilon_r=None)
    spec2 = PortSpecMultiConductor(name="port2", plane=FACE_TEST, epsilon_r=None)

    op1 = build_cw_true_mode_port(spec1, mesh, m_eps, m_mu, dt=dt, f_cw=f)
    shipped_profile = variant.startswith("mur") and not variant.startswith("mur_true")
    if shipped_profile:
        op2 = build_modal_port(spec2, mesh, m_eps, m_mu, dt=dt, f_calc=f)
    else:
        op2 = build_cw_true_mode_port(spec2, mesh, m_eps, m_mu, dt=dt, f_cw=f)
        if variant.startswith("mur_true"):
            # The one internals touch in this file: identical to constructing
            # the operator with termination="mur", which
            # build_cw_true_mode_port does not expose.
            op2._dtbc = [None] * len(op2.discrete_modes)

    kinds = list(op2.termination_kinds)
    want = "dtbc" if variant == "dtbc" else "mur"
    if kinds[0] != want:
        raise RuntimeError(f"{name}/{variant}: port 2 channel 0 terminates on {kinds[0]!r}")
    if op1.termination_kinds[0] != "dtbc":
        raise RuntimeError(f"{name}: driven port is on {op1.termination_kinds[0]!r}, not dtbc")

    chain2, plane2, me2, mm2, c_3d = make_chain(mesh, dt, FACE_TEST)
    if not np.array_equal(plane2.e_u_indices, op2.plane.e_u_indices):
        raise RuntimeError(f"{name}: analysis plane does not match the port plane")
    w_dt = 2.0 * math.pi * f * dt
    ch2, n_prop = analysis_channel(op2, chain2, plane2, me2, mm2, c_3d, w_dt)

    r_nominal = float(op2.mur_r[0])
    r_disc = r_exact(ch2.zeta, w_dt)
    alpha = 0.0
    if variant.endswith("rexact"):
        alpha = 1.0
    elif variant.startswith("mur_frac"):
        alpha = float(variant.split("_")[-1]) / 100.0
    if alpha:
        op2._mur_r[0] = (1.0 - alpha) * r_nominal + alpha * r_disc

    # --- driver ---------------------------------------------------------
    # ch.r / ch.q are the CHAIN parameters (Courant number, Klein-Gordon
    # mass), NOT op.mur_r, which is the reflection coefficient.
    ch_drive = op1.cw_data.channels[0]
    period = 2.0 * math.pi / w_dt
    theta = abs(np.angle(ch_drive.zeta))
    v_g = ch_drive.r**2 * math.sin(theta) / math.sin(w_dt)
    nz = mesh.Nz
    gap = w_dt - 2.0 * math.asin(min(ch_drive.q / 2.0, 1.0))
    sigma = max(8.0 / max(gap, 1e-12), 6.0 * period)
    n_win = int(40 * period)
    n_meas0 = int(10.0 * sigma + 20.0 * period + 3.0 * nz / max(v_g, 1e-3))
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
    t_run0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*neither a BoundaryCondition.*")
        solver.run()
    t_run = time.perf_counter() - t_run0
    signals = recorder.finalize(n_steps_actual=n_steps)

    # Port under test: b2 arrives, a2 is sent back -> reflection = a2/b2.
    v2, i2 = signals[("port2", 0)]
    vp2, ip2, res2 = cw_lockin_phasors(v2.values, i2.values, w_dt, n_win)
    a2, b2 = cw_decompose(vp2, ip2, ch2)
    # Driven port witness: its own |b1/a1|, one transit later.
    v1, i1 = signals[("port1", 0)]
    vp1, ip1, res1 = cw_lockin_phasors(v1.values, i1.values, w_dt, n_win)
    a1, b1 = cw_decompose(vp1, ip1, op1.cw_data.channels[0])

    r_used = float(op2.mur_r[0])
    info = dict(
        case=name,
        variant=variant,
        f=f,
        termination=kinds,
        measured_db=db(abs(a2 / b2)),
        apriori_db=db(gamma_apriori(ch2.zeta, r_used, w_dt)),
        apriori_nominal_db=db(gamma_apriori(ch2.zeta, r_nominal, w_dt)),
        source_side_db=db(abs(b1 / a1)),
        lockin_residual=float(res2),
        lockin_residual_drive=float(res1),
        alpha=alpha,
        mur_r=r_used,
        mur_r_nominal=r_nominal,
        mur_r_exact=r_disc,
        zeta=[float(ch2.zeta.real), float(ch2.zeta.imag)],
        n_prop=n_prop,
        n_steps=int(n_steps),
        nz=int(nz),
        dt=float(dt),
        runtime_s=time.perf_counter() - t_all0,
        run_s=t_run,
    )
    if verbose:
        print(
            f"    {name:11s} {variant:16s} f {f / 1e9:5.2f} GHz  "
            f"|a2/b2| {info['measured_db']:8.2f} dB   "
            f"a-priori(nominal r) {info['apriori_nominal_db']:8.2f} dB   "
            f"drive-side {info['source_side_db']:8.2f} dB   "
            f"res {res2:.2e}   r {r_used:+.4f}   "
            f"({n_steps} steps, {info['run_s']:.1f} s)",
            flush=True,
        )
    return info


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


def reconcile(rows, name, f):
    """Boundary term, profile term, and their near-antiphase residual.

    ``G_bnd`` is DD-237's a-priori Gamma with the shipped ``r`` (the pure
    boundary error), ``G_prof`` the measured "mur_rexact" floor (the pure
    profile error, boundary corrected), and the shipped floor is their
    vector sum.  The inferred angle comes from the law of cosines; at
    180 deg the difference of the magnitudes closes exactly.
    """
    got = {
        r["variant"]: r for r in rows if r["case"] == name and r["f"] == f and "measured_db" in r
    }
    if "mur" not in got or "mur_rexact" not in got:
        return None
    a = lin(got["mur"]["apriori_nominal_db"])
    b = lin(got["mur_rexact"]["measured_db"])
    m = lin(got["mur"]["measured_db"])
    cos_psi = float(np.clip((m * m - a * a - b * b) / (2.0 * a * b), -1.0, 1.0))
    return dict(
        case=name,
        f=f,
        g_bnd=a,
        g_prof=b,
        measured=m,
        diff_db=db(abs(a - b)),
        measured_db=got["mur"]["measured_db"],
        closure_db=abs(db(abs(a - b)) - got["mur"]["measured_db"]),
        angle_deg=math.degrees(math.acos(cos_psi)),
    )


def acceptance(rows, recs):
    """Print the four acceptance lines of the module docstring."""
    print("\n  Acceptance:")
    worst_excess = -math.inf
    worst_abs = 0.0
    n_pin = 0
    for row in rows:
        if "measured_db" not in row:
            continue
        pin = PINNED.get((row["case"], round(row["f"] / 1e9, 2)), {}).get(row["variant"])
        if pin is None:
            continue
        n_pin += 1
        tol = 2.0 if pin < -80.0 else 0.5
        d = abs(row["measured_db"] - pin)
        worst_excess = max(worst_excess, d - tol)
        worst_abs = max(worst_abs, d)
        flag = "ok " if d <= tol else "OFF"
        print(
            f"    {flag} {row['case']:11s} {row['variant']:16s} "
            f"{row['f'] / 1e9:5.2f} GHz  measured {row['measured_db']:8.2f} dB  "
            f"pinned {pin:8.2f} dB  delta {row['measured_db'] - pin:+6.2f} dB (tol {tol:.1f})"
        )
    n_failed = sum(1 for r in rows if "failed" in r)
    floors = [r for r in rows if r.get("variant") in FLOOR_VARIANTS and "measured_db" in r]
    worst_floor = max((r["measured_db"] for r in floors), default=None)
    res = max((r["lockin_residual"] for r in rows if "lockin_residual" in r), default=0.0)

    if n_pin:
        print(
            f"\n    (1) pinned points: {n_pin} checked, worst |delta| {worst_abs:.2f} dB, "
            f"margin to tolerance {-worst_excess:+.2f} dB -> "
            f"{'ok' if worst_excess <= 0.0 else 'OFF'}"
        )
    else:
        print("\n    (1) pinned points: none in this selection")
    # A point that raised is skipped by (1) above, which would otherwise read
    # "ok" on a shrunken sample; say so on its own line.
    print(f"    (1b) run points that raised: {n_failed} -> {'ok' if n_failed == 0 else 'OFF'}")
    if worst_floor is None:
        print("    (2) instrument floor: not measured in this selection")
    else:
        print(
            f"    (2) instrument floor: worst {worst_floor:.2f} dB "
            f"(line {FLOOR_DB:.0f} dB) -> {'ok' if worst_floor <= FLOOR_DB else 'OFF'}"
        )
    blk = [r for r in recs if r["case"] == "block" and abs(r["f"] - 6.2e9) < 1e6]
    if blk:
        print(
            f"    (3) block 6.2 GHz reconciliation closes to "
            f"{blk[0]['closure_db']:.3f} dB (line 0.05 dB) -> "
            f"{'ok' if blk[0]['closure_db'] <= 0.05 else 'OFF'}"
        )
    else:
        print("    (3) block 6.2 GHz reconciliation: not in this selection")
    print(
        f"    (4) worst lock-in residual {res:.2e} "
        f"(line {RESIDUAL_MAX:.0e}) -> {'ok' if res <= RESIDUAL_MAX else 'OFF'}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="all", choices=[*CASES, "all"])
    ap.add_argument("--freqs", default="", help="comma-separated GHz, overrides the case band")
    ap.add_argument(
        "--variants",
        default=DEFAULT_VARIANTS,
        help="mur|mur_rexact|mur_true|mur_true_rexact|dtbc|mur_frac_<pct>",
    )
    ap.add_argument(
        "--alpha",
        default="",
        help="comma-separated percentages, appended as mur_frac_<pct> variants",
    )
    ap.add_argument("--fast", action="store_true", help="one frequency and two variants per case")
    ap.add_argument("--nz", type=int, default=0, help="override the fixture line length")
    ap.add_argument("--out", default="", help="write the rows as JSON")
    args = ap.parse_args()

    cases = list(CASES) if args.case == "all" else [args.case]
    variants = [v for v in args.variants.split(",") if v]
    variants += [f"mur_frac_{int(float(a))}" for a in args.alpha.split(",") if a.strip()]
    if args.fast:
        variants = ["mur", "dtbc"]

    print(
        "DD-238 — modal-Mur QTEM port floor, measured and decomposed (|a2/b2| at the passive port):"
    )
    rows, recs = [], []
    for name in cases:
        builder, band = CASES[name]
        kw = {"nz": args.nz} if args.nz else {}
        t0 = time.perf_counter()
        mesh, dt = builder(**kw)
        print(
            f"\n  {name}: mesh {mesh.Nx} x {mesh.Ny} x {mesh.Nz}  "
            f"dt {dt * 1e12:.4f} ps (courant_dt, the fixture's own)  "
            f"[{time.perf_counter() - t0:.1f} s]",
            flush=True,
        )
        if args.freqs:
            freqs = [float(x) * 1e9 for x in args.freqs.split(",")]
        else:
            freqs = [band[len(band) // 2]] if args.fast else list(band)
        for f in freqs:
            for variant in variants:
                try:
                    rows.append(run_point(name, mesh, dt, f, variant))
                except Exception as exc:  # noqa: BLE001
                    print(f"    {name:11s} {variant:16s} f {f / 1e9:5.2f} GHz  FAILED: {exc}")
                    rows.append(
                        dict(case=name, variant=variant, f=f, failed=f"{type(exc).__name__}: {exc}")
                    )
            rec = reconcile(rows, name, f)
            if rec is not None:
                recs.append(rec)
                print(
                    f"      decomposition: boundary {rec['g_bnd']:.3e} "
                    f"({db(rec['g_bnd']):.2f} dB)  profile {rec['g_prof']:.3e} "
                    f"({db(rec['g_prof']):.2f} dB)  |difference| "
                    f"{abs(rec['g_bnd'] - rec['g_prof']):.3e} "
                    f"({rec['diff_db']:.2f} dB)  vs shipped {rec['measured_db']:.2f} dB  "
                    f"[relative phase {rec['angle_deg']:.1f} deg]",
                    flush=True,
                )
            if args.out:
                with open(args.out, "w") as fh:
                    json.dump(rows, fh, indent=1)
    acceptance(rows, recs)
    if args.out:
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
