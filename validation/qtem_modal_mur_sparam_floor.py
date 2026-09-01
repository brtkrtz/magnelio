"""The user-visible modal-Mur S11 of a microstrip line, decomposed.

DD-238 measured the modal-Mur QTEM *port floor* -- ``|a2/b2|`` at the
passive port under an exact-DTBC true-mode generator -- at -52.3 dB
(microstrip), -36.5 dB (layered), -55.6 dB (block).  The number a user
reads is a different one and worse: tutorial 09's straight 50 ohm
microstrip prints ``|S11| max -32.9 dB``.  This certificate reproduces
that number through the public path and splits it into its terms.

THE VERDICT.  The far-port termination floor is NOT what caps the user
number.  Removing it entirely -- replacing the shipped Mur port with an
exactly transparent one and leaving everything else as measured --
moves the reported ``|S11|`` by at most 1.93 dB anywhere in the band.
What caps it are two terms of equal weight that are the same defect
twice, the port's frequency-flat quasi-static mode:

  * the **a/b split** in ``compute_s_parameters``, which values the
    recorded ``(V, I)`` with the quasi-static wave impedance and the
    quasi-static propagation constant, and
  * the **drive-port source**, which injects the quasi-static mode
    profile through a soft source into a line whose true discrete mode
    is a different shape.

Removing both buys +10.3 dB at the band top (+12.0 dB on the refined
mesh) and +24.0 dB at 5 GHz.  Further boundary work provably cannot
move the user-visible number; de-staggering can.

The fixture.  Tutorial 09's own geometry, mesh control and band --
1.2 mm trace, 0.2 mm thick, on 0.8 mm FR4 (eps_r 4.3) in an 8 x 5 mm
shield, 20 mm long, ``SymmetryPMC`` on ``xmin``,
``MeshControl(min_nodes_per_wavelength=25)``, ``f_max = 15 GHz``.  The
mesh is 14 x 23 x 52 = 16744 cells; nothing is reduced, so the
mesh-artefact question below is a generality check, not a caveat on the
headline number.

Stage 1, the number the user reads.  ``AnalysisScatteringTD.run`` ->
``compute_s_parameters``, public API only: ``|S11| max -32.88 dB`` at
15.00 GHz, ``|S21| min -0.000 dB`` -- tutorial 09's printed -32.9 dB to
0.02 dB.  Per frequency: -38.12 (5 GHz), -46.28 (10), -32.88 (15).

Stage 2, the control cube.  One CW erf-ramped sine per frequency, true
discrete-mode lock-in (``cw_wave_phasors`` -> ``cw_lockin_phasors`` ->
``cw_decompose``), the DD-238 instrument.  Mesh, ``dt``, driver sizing
and analysis channels are identical across the cube; only the port
build and the postprocessor change:

  "shipped"       both ports ``build_modal_port`` -- the production run.
  "far_exact"     drive port shipped, far port ``build_cw_true_mode_port``.
  "source_exact"  drive port exact, far port shipped.
  "both_exact"    both exact -- the instrument floor control.

Two postprocessors run on the *same* recorded phasors: the production
split (``spectral_power_waves``, the shared core of
``compute_s_parameters`` and ``destaggered_power_waves``) and the exact
true-mode split (``cw_decompose``).  That makes the postprocessing term
confound-free: it is the difference of two readings of one record.

  ``g_split``   production split minus exact split on the shipped run.
  ``g_source``  exact split on the run whose far end is transparent --
                everything the drive port does wrong on its own:
                injecting the quasi-static profile through a soft
                source, and re-absorbing the mismatch on its own Mur
                boundary.  The two are not separable from outside and
                are not separated here.
  ``g_far``     exact split on the run whose drive end is exact -- the
                far port's floor, read at the drive port one round trip
                later (and cross-read directly at the far port).

Measured, |Gamma| in dB (2026-09-01):

    Nz  f/GHz  reported   split   source  far floor  residual
    52   5.00   -38.12   -35.86   -44.83    -62.28    -98.20
    52  10.00   -46.62   -31.80   -31.79    -50.23    -74.25
    52  15.00   -33.72   -28.98   -30.17    -43.73    -63.49
    84  15.00   -34.12   -29.05   -30.41    -46.14    -63.19

The three terms are exhaustive: the residual sits 27.6-60.1 dB below
the reported value, and the "both_exact" control reads -164.7 to
-183.4 dB, at least 118 dB below every reported value.  Lock-in
residuals 1.0e-7 .. 6.0e-7.

The terms add as VECTORS, so a magnitude ranking is not the answer.
The decision-grade quantity is the counterfactual -- what the run would
have reported with one term removed and the others left as measured:

    Nz  f/GHz  reported   floor removed   split+source removed
    52   5.00   -38.12   -37.67 (-0.45)     -62.15 (+24.03)
    52  10.00   -46.62   -47.34 (+0.72)     -50.30  (+3.68)
    52  15.00   -33.72   -35.65 (+1.93)     -44.06 (+10.34)
    84  15.00   -34.12   -35.93 (+1.81)     -46.11 (+11.99)

At 5 GHz removing the far-port floor makes the *reported* number 0.45 dB
worse, because a term 24 dB down can only rotate the sum.

The 10 GHz row explains the dip in tutorial 09's own S11 curve: there
the split (-31.80 dB at -1.7 deg) and the source (-31.79 dB at
-172.6 deg) are equal in magnitude and 170.9 deg apart, so they very
nearly cancel and the reported value falls to -46.6 dB.  That null is a
cancellation between a postprocessing error and a source error, not
physics; it is also why the band-top point, not the band median, is the
one the verdict rests on.

Stage 3, what actually caps the a/b split.  The production split models
the port plane with exactly two numbers -- a wave impedance and a
half-cell de-stagger exponent.  Reading both back out of the true
discrete channel and substituting one at a time (a stub mode fed to the
production code path, no reimplementation) prices each:

    f/GHz  Z_ship  Z_true    dZ%    split   Z only  th only   control
     5.00  103.03  106.40  -3.17   -35.86   -35.86   -76.05   -322.88
    10.00  103.03  108.45  -4.99   -31.80   -31.81   -59.86   -322.73
    15.00  103.03  110.55  -6.80   -29.00   -29.02   -51.57   -316.53

("Z only" keeps the shipped impedance and corrects the exponent, "th
only" the reverse; "control" corrects both and must vanish, which is
what certifies that the two-parameter model of the split is complete.)
The split is the WAVE IMPEDANCE, to within 0.02 dB: ``z_modal`` is the
frequency-flat quasi-static value 103.0 ohm on the meshed half, and the
true discrete travelling wave carries 110.6 ohm at 15 GHz, 6.8 % away.
A relative impedance error ``e`` leaks ``|b/a| = |e|/2``, which is
-29 dB.  The a-priori price of the split on a unit incident true mode
-- pure arithmetic, no time stepping -- reproduces the measured split
term to 0.00-0.02 dB at every point.

**The campaign's cap formula is refuted as a cap.**  DD-055 records the
continuum de-stagger factor's grid-dispersion gap as
``(beta*dz/2)^3 (1-r^2)/6``, and the campaign has carried it since as
the reason a *reported* reflection cannot fall below -40..-60 dB.  That
expression is a phase gap; the reflection it leaks is
``gap / |2 cosh theta|``, and on this fixture it prices -108.9 / -90.3 /
-79.2 dB at 5 / 10 / 15 GHz -- 50 to 73 dB BELOW the split it is
supposed to cap.  It is not wrong, it is the wrong term: it prices the
*grid* part of the de-stagger exponent, which is 2.3-4.2 % of that
exponent's total error here, and the exponent in turn is 22.6 dB below
the impedance term.  The measured cap is a DISPERSION error, not a
discretisation error.

Which is why it does not refine away.  At 15 GHz, over a 4.2x change in
cell size (a priori, and confirmed by measurement at Nz 52 and 84):

    Nz  dz/mm    split   th only   grid formula
    25  0.800   -28.74   -43.91      -59.69
    38  0.526   -28.93   -48.50      -70.87
    52  0.385   -29.00   -51.57      -79.15
    73  0.274   -29.05   -54.74      -88.07
   104  0.192   -29.08   -57.93      -97.37

The grid formula falls 37.7 dB and the split moves 0.34 dB -- in the
wrong direction for an artefact, i.e. a coarser mesh does not
exaggerate it.  The refined measured run (Nz 84, 37800 cells) reports
``|S11| max -33.09 dB`` publicly and reproduces the whole attribution:
split -29.05, source -30.41, far floor -46.14, removing the floor worth
+1.81 dB.  The dominance conclusion is not a mesh artefact.

Relation to DD-238.  Nothing here contradicts it.  Its instrument
reading is this file's ``g_far``, measured here both ways -- read at
the drive port one round trip later and read directly at the far port
as ``|a2/b2|``.  The two agree to 0.9 dB at 15 GHz (-43.73 against
-42.83) and 1.3 dB on the refined mesh (-46.14 against -44.89); that
gap is the round-trip read's own accuracy and is immaterial to the
verdict, since either reading leaves the term 10-12 dB below the
reported value.  DD-238's conclusion -- do not calibrate the Mur
coefficient alone -- stands; this certificate adds that even a
*perfect* boundary would leave the tutorial's -32.9 dB essentially
where it is.

Relation to DD-064.  The -30 dB class acceptance is unaffected as a
number, but its attribution moves.  DD-064 records the class as "Mur
-25.8 dB worst" and tutorial 09 explains it to the reader as "the
residual of the port termination absorbing a dispersive quasi-TEM
wave".  The termination is 10-24 dB better than the number blamed on
it.  The -30 dB class is the price of describing a dispersive
quasi-TEM line with one frequency-flat quasi-static mode, paid twice --
once when the source injects that mode and once when the S-parameter
split values it -- and the two payments partly cancel, which is why
the reported curve dips to -46 dB mid-band.

Acceptance:
  1.  the public ``|S11| max`` reproduces the pin to 0.30 dB and
      tutorial 09's printed -32.9 dB to 0.50 dB;
  1b. every pinned term reproduces within 0.5 dB (2.0 dB below -80 dB);
  2.  the "both_exact" instrument floor stays below -110 dB;
  3.  the three-term residual stays at least 25 dB below the reported
      value at every point -- no fourth term of consequence;
  4.  the a-priori split price matches the measured split within
      0.30 dB;
  5.  the two-parameter control of the split stays below -200 dB, i.e.
      the split really has only those two parameters;
  6.  lock-in residuals stay below 1e-5;
  7.  THE VERDICT: removing the far-port floor moves the reported S11
      by at most 3.0 dB at every measured point;
  8.  and at the band top, where the user's max sits, removing the
      split and the source instead buys at least 8.0 dB;
  9.  no run point raised.

Cost: 376 s for the default selection (3 frequencies x 4 control runs
on the shipped mesh, the a-priori ladder, two public runs and the
refined confirmation) on one core.

Run:  python validation/qtem_modal_mur_sparam_floor.py
      [--freqs 5,10,15] [--npw 25] [--npw-ladder 12,18,25,35,50]
      [--refine-npw 40] [--configs shipped,far_exact,...] [--no-public]
      [--fast] [--out rows.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings

import numpy as np
from scipy.special import erf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qtem_mur_floor_decomposition import (  # noqa: E402
    analysis_channel,
    db,
    make_chain,
)

import magnelio as mio  # noqa: E402
from magnelio import geo  # noqa: E402
from magnelio import ports as pub_ports  # noqa: E402
from magnelio._operators.material_matrices import build_M_eps, build_M_mu  # noqa: E402
from magnelio.ports._modal import BoxFace, PortSpecMultiConductor  # noqa: E402
from magnelio.ports._modal.factory import (  # noqa: E402
    build_cw_true_mode_port,
    build_modal_port,
)
from magnelio.ports._modal.zeta_pencil import cw_decompose, cw_lockin_phasors  # noqa: E402
from magnelio.ports.recorder import PortSignalRecorder  # noqa: E402
from magnelio.post.modal_sparameters import spectral_power_waves  # noqa: E402
from magnelio.solver.fit_td import FITTimeDomainSolver  # noqa: E402
from magnelio.solver.stability import courant_dt  # noqa: E402

FACE_DRIVE = BoxFace.Z_MIN
FACE_TEST = BoxFace.Z_MAX

# Tutorial 09's own cross-section, verbatim.
H_SUB = 0.8e-3
W_STRIP = 1.2e-3
T_STRIP = 0.2e-3
W_BOX = 8.0e-3
H_BOX = 5.0e-3
LENGTH = 20.0e-3
EPS_R = 4.3
F_MAX = 15.0e9
NPW_SHIPPED = 25

# (source at z_min, termination at z_max) -- "modal" is the shipped
# build_modal_port, "true" the exact CW true-mode DTBC reference.
CONFIGS = {
    "shipped": ("modal", "modal"),
    "far_exact": ("modal", "true"),
    "source_exact": ("true", "modal"),
    "both_exact": ("true", "true"),
}

# Acceptance lines (see the module docstring).
TUTORIAL_PRINTED_DB = -32.9  # what tutorial 09 prints for |S11| max
PINNED_PUBLIC_MAX_DB = -32.88
FLOOR_DB = -110.0  # instrument floor: both ends exactly transparent
CLOSURE_DB = 25.0  # residual this far below the reported S11
SPLIT_CONTROL_DB = -200.0  # both split parameters corrected
FLOOR_GAIN_DB = 3.0  # dB the reported S11 moves if the far-port floor vanishes
SPLIT_SOURCE_GAIN_DB = 8.0  # dB it gains at the band top if split + source vanish
RESIDUAL_MAX = 1e-5

# Pinned |Gamma| in dB per (Nz, GHz).  Reproduced by acceptance line (1b);
# never edited to match a run (DD-238's rule).
PINNED_TERMS = {
    (52, 5.0): {
        "reported": -38.12,
        "split": -35.86,
        "source": -44.83,
        "far": -62.28,
        "gain_far": -0.45,
        "gain_split_source": 24.03,
    },
    (52, 10.0): {
        "reported": -46.62,
        "split": -31.80,
        "source": -31.79,
        "far": -50.23,
        "gain_far": 0.72,
        "gain_split_source": 3.68,
    },
    (52, 15.0): {
        "reported": -33.72,
        "split": -28.98,
        "source": -30.17,
        "far": -43.73,
        "gain_far": 1.93,
        "gain_split_source": 10.34,
    },
    (84, 15.0): {
        "reported": -34.12,
        "split": -29.05,
        "source": -30.41,
        "far": -46.14,
        "gain_far": 1.81,
        "gain_split_source": 11.99,
    },
}


# ----------------------------------------------------------------------
# Fixture -- tutorial 09, through the public API
# ----------------------------------------------------------------------


def tutorial09_model(with_ports=True, length=LENGTH):
    fr4 = mio.Material.from_isotropic(name="FR4", epsilon=EPS_R)
    model = mio.GeometryModel(boundary_conditions={"xmin": "SymmetryPMC"})
    model.add(geo.Brick(origin=(-W_BOX / 2, 0.0, 0.0), size=(W_BOX, H_SUB, length), material=fr4))
    air_cap = geo.Brick(
        origin=(-W_BOX / 2, H_SUB, 0.0), size=(W_BOX, H_BOX - H_SUB, length), material="air"
    )
    strip = geo.Brick(
        origin=(-W_STRIP / 2, H_SUB, 0.0), size=(W_STRIP, T_STRIP, length), material="pec"
    )
    model.add(geo.Difference(air_cap, strip))
    model.add(strip)
    if with_ports:
        model.add_port(pub_ports.PortWaveguide(name="port1", plane="zmin", n_modes=1))
        model.add_port(pub_ports.PortWaveguide(name="port2", plane="zmax", n_modes=1))
    return model


def tutorial09_mesh(npw=NPW_SHIPPED, length=LENGTH, f_max=F_MAX):
    """Tutorial 09's mesh; ``npw`` is its ``min_nodes_per_wavelength``."""
    model = tutorial09_model(length=length)
    mesh = mio.Mesh.from_geometry(model, mio.MeshControl(min_nodes_per_wavelength=npw), f_max=f_max)
    return mesh, courant_dt(mesh.grid, "normal")


# ----------------------------------------------------------------------
# Stage 1 -- the number the user reads
# ----------------------------------------------------------------------


def public_run(mesh, f_max=F_MAX):
    """Tutorial 09's own S-parameter run, public API only."""
    analysis = mio.AnalysisScatteringTD(mesh=mesh, verbose=False)
    t0 = time.perf_counter()
    result = analysis.run(excited=["port1"])
    s11 = np.asarray(result.S("port1", "port1"))
    s21 = np.asarray(result.S("port2", "port1"))
    f_axis = np.asarray(result.f_axis)
    j = int(np.argmax(np.abs(s11)))
    return dict(
        f_axis=f_axis,
        s11=s11,
        s21=s21,
        s11_max_db=db(abs(s11[j])),
        f_at_max=float(f_axis[j]),
        s21_min_db=db(abs(s21).min()),
        runtime_s=time.perf_counter() - t0,
    )


# ----------------------------------------------------------------------
# The two a/b splits
# ----------------------------------------------------------------------


class _ModeStub:
    """A ``Mode``-shaped stub with substituted ``(Z, gamma)``.

    ``spectral_power_waves`` -- the production split, shared by
    ``compute_s_parameters`` and ``destaggered_power_waves`` -- reads
    exactly ``z_modal``, ``gamma`` and ``mode_type`` off the mode.
    Feeding it a stub swaps one of the two quasi-static parameters
    for its true discrete counterpart without touching the code path.
    """

    def __init__(self, z, gamma, mode_type):
        self._z = complex(z)
        self._gamma = complex(gamma)
        self.mode_type = mode_type

    def z_modal(self, omega):
        return self._z

    def gamma(self, omega):
        return self._gamma


def production_split(v_ph, i_ph, mode, normal_dx, omega, dt):
    """``(a, b)`` exactly as the shipped S-parameter path forms them.

    One rfft bin of :func:`spectral_power_waves`; the lock-in phasors
    already carry the temporal Yee half-step rotation the production
    path applies to ``I``.
    """
    a, b = spectral_power_waves(
        np.array([complex(v_ph)]),
        np.array([complex(i_ph)]),
        np.array([float(omega)]),
        dt,
        mode,
        normal_dx=float(normal_dx),
        line_params=None,
    )
    return complex(a[0]), complex(b[0])


def channel_z_theta(channel):
    """The ``(Z, theta)`` the channel's exact phasors correspond to.

    The production split models the port plane as
    ``v_in = sqrt(Z)``, ``i_in = e^{-theta}/sqrt(Z)``,
    ``v_out = sqrt(Z)``, ``i_out = -e^{+theta}/sqrt(Z)``
    (wave impedance plus half-cell de-stagger).  Reading ``Z`` and
    ``theta`` back out of the true discrete channel gives the two
    numbers the shipped path *should* have used.
    """
    zi_in = channel.v_in / channel.i_in
    zi_out = channel.v_out / channel.i_out
    z_true = complex(np.sqrt(-zi_in * zi_out))
    theta_true = 0.5 * complex(np.log(-zi_in / zi_out))
    return z_true, theta_true


def split_terms(channel, mode, normal_dx, omega, dt):
    """A-priori price of the a/b split on a unit incident true mode.

    Feeds the channel's own exact incident phasors ``(v_in, i_in)``
    -- a pure forward-travelling discrete wave, reflection identically
    zero -- through four splits: the shipped one, the shipped one with
    only the wave impedance corrected, the shipped one with only the
    de-stagger exponent corrected, and both corrected (the control,
    which must read the instrument floor).
    """
    z_true, theta_true = channel_z_theta(channel)
    z_ship = complex(mode.z_modal(omega))
    theta_ship = 0.5 * float(normal_dx) * complex(mode.gamma(omega))
    gam_of = lambda th: 2.0 * th / float(normal_dx)  # noqa: E731

    def price(z, theta):
        a, b = production_split(
            channel.v_in,
            channel.i_in,
            _ModeStub(z, gam_of(theta), mode.mode_type),
            normal_dx,
            omega,
            dt,
        )
        return b / a

    return dict(
        z_shipped=z_ship,
        z_true=z_true,
        theta_shipped=theta_ship,
        theta_true=theta_true,
        g_split=price(z_ship, theta_ship),
        g_split_theta_only=price(z_true, theta_ship),
        g_split_z_only=price(z_ship, theta_true),
        g_split_control=price(z_true, theta_true),
    )


def destagger_formula_db(channel, theta_true):
    """The campaign's grid-dispersion cap ``(beta*dz/2)^3 (1-r^2)/6``.

    That expression is the *phase* gap between the continuum half-cell
    exponent and the exact discrete one at the same physical wave; a
    small exponent error ``delta`` leaks ``|b/a| = |delta| / |2 cosh
    theta|``, so the reflection it caps is half of it (to the same
    order).  Returned as ``(gap_rad, reflection_db)``.
    """
    beta_dz = 2.0 * abs(theta_true.imag)
    gap = (beta_dz / 2.0) ** 3 * (1.0 - channel.r**2) / 6.0
    return gap, db(gap / abs(2.0 * np.cosh(theta_true)))


# ----------------------------------------------------------------------
# One CW control point
# ----------------------------------------------------------------------


def _build_port(kind, spec, mesh, m_eps, m_mu, dt, f):
    if kind == "modal":
        return build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=f)
    return build_cw_true_mode_port(spec, mesh, m_eps, m_mu, dt=dt, f_cw=f)


def run_cw(mesh, dt, f, config, ref, chains, verbose=True):
    """One CW frequency point in one control configuration.

    ``ref`` holds the *shipped* modal port per face -- the source of
    the production split's ``(z_modal, gamma, normal_dx)`` whatever the
    run itself is terminated with; ``chains`` the per-face period
    blocks, both built once per mesh and frequency.
    """
    t_all0 = time.perf_counter()
    src_kind, term_kind = CONFIGS[config]
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    spec1 = PortSpecMultiConductor(name="port1", plane=FACE_DRIVE, epsilon_r=None)
    spec2 = PortSpecMultiConductor(name="port2", plane=FACE_TEST, epsilon_r=None)
    op1 = _build_port(src_kind, spec1, mesh, m_eps, m_mu, dt, f)
    op2 = _build_port(term_kind, spec2, mesh, m_eps, m_mu, dt, f)
    want = {"modal": "mur", "true": "dtbc"}
    for op, kind, label in ((op1, src_kind, "port1"), (op2, term_kind, "port2")):
        if op.termination_kinds[0] != want[kind]:
            raise RuntimeError(
                f"{label}: asked for {kind!r}, got termination {op.termination_kinds[0]!r}"
            )

    w_dt = 2.0 * math.pi * f * dt
    omega = w_dt / dt
    # The instrument channel must be synthesised through the projections
    # of the operator that actually recorded, not through the reference
    # port's: the two carry different mode profiles and duals.
    an = {}
    for key, op, face in (("port1", op1, FACE_DRIVE), ("port2", op2, FACE_TEST)):
        chain, plane, m_e, m_m, c_3d = chains[face]
        an[key], _ = analysis_channel(op, chain, plane, m_e, m_m, c_3d, w_dt)

    # --- driver: the DD-238 erf-ramped CW sine, same sizing law ---------
    ch_drive = an["port1"]
    period = 2.0 * math.pi / w_dt
    theta = abs(np.angle(ch_drive.zeta))
    v_g = ch_drive.r**2 * math.sin(theta) / math.sin(w_dt)
    gap = w_dt - 2.0 * math.asin(min(ch_drive.q / 2.0, 1.0))
    sigma = max(8.0 / max(gap, 1e-12), 6.0 * period)
    n_win = int(40 * period)
    n_steps = int(10.0 * sigma + 20.0 * period + 3.0 * mesh.Nz / max(v_g, 1e-3)) + n_win + 2
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

    out = dict(
        config=config,
        f=f,
        source=src_kind,
        termination=term_kind,
        n_steps=int(n_steps),
        nz=int(mesh.Nz),
        runtime_s=time.perf_counter() - t_all0,
        run_s=t_run,
    )
    for key, op, face in (("port1", op1, FACE_DRIVE), ("port2", op2, FACE_TEST)):
        v, i = signals[(key, 0)]
        v_ph, i_ph, res = cw_lockin_phasors(v.values, i.values, w_dt, n_win)
        a_i, b_i = cw_decompose(v_ph, i_ph, an[key])
        ref_op = ref[face]
        a_p, b_p = production_split(
            v_ph, i_ph, ref_op.discrete_modes[0].mode, ref_op.plane.normal_dx, omega, dt
        )
        out[key] = dict(
            gamma_instrument=complex(b_i / a_i),
            gamma_production=complex(b_p / a_p),
            incident_instrument=complex(a_i),
            reflected_instrument=complex(b_i),
            lockin_residual=float(res),
        )
    # Port 2 reads its reflection the other way round: b2 arrives, a2
    # is sent back (DD-238's |a2/b2|).
    p2 = out["port2"]
    p2["gamma_far_end"] = p2["incident_instrument"] / p2["reflected_instrument"]

    if verbose:
        print(
            f"    {config:13s} f {f / 1e9:5.2f} GHz  "
            f"S11 production {db(abs(out['port1']['gamma_production'])):8.2f} dB   "
            f"S11 instrument {db(abs(out['port1']['gamma_instrument'])):8.2f} dB   "
            f"far end |a2/b2| {db(abs(p2['gamma_far_end'])):8.2f} dB   "
            f"res {out['port1']['lockin_residual']:.1e}   "
            f"({n_steps} steps, {t_run:.0f} s)",
            flush=True,
        )
    return out


# ----------------------------------------------------------------------
# Attribution
# ----------------------------------------------------------------------


def attribute(cube, f):
    """Split the user-visible S11 into its three terms plus a residual.

    All four control runs share the mesh, ``dt``, driver and analysis
    channels; only the port build and the postprocessor change.

      ``g_user``   production split on the shipped run -- what
                   ``compute_s_parameters`` reports.
      ``g_split``  ``g_user`` minus the exact-instrument split of the
                   *same* recorded phasors: the postprocessing term
                   alone, confound-free by construction.
      ``g_source`` instrument reading of the shipped run whose far end
                   is exactly transparent: the drive port's own error.
      ``g_far``    instrument reading of the run driven by an exact
                   true-mode generator: the far port's floor, seen at
                   the drive port one round trip later.

    The terms are complex and add as vectors, so the decision-grade
    quantity is not a term's magnitude but the *counterfactual*: what
    the run would have reported with that term removed and the others
    left as measured.  Those are computed here from the measured
    vectors, not modelled.
    """
    got = {c: r for c, r in cube.items() if r is not None}
    if not {"shipped", "far_exact", "source_exact"} <= set(got):
        return None
    g_user = got["shipped"]["port1"]["gamma_production"]
    g_phys = got["shipped"]["port1"]["gamma_instrument"]
    g_split = g_user - g_phys
    g_source = got["far_exact"]["port1"]["gamma_instrument"]
    g_far = got["source_exact"]["port1"]["gamma_instrument"]
    g_floor = got.get("both_exact", {}).get("port1", {}).get("gamma_instrument")
    total = g_split + g_source + g_far
    residual = g_user - total
    terms = {"a/b split": g_split, "drive source": g_source, "far port floor": g_far}
    order = sorted(terms, key=lambda k: -abs(terms[k]))
    counterfactual = {
        "far port floor removed": g_user - g_far,
        "a/b split removed": g_user - g_split,
        "drive source removed": g_user - g_source,
        "split and source removed": g_far + residual,
    }
    return dict(
        f=f,
        g_user=g_user,
        g_phys=g_phys,
        g_split=g_split,
        g_source=g_source,
        g_far=g_far,
        g_far_direct=got["source_exact"]["port2"]["gamma_far_end"],
        g_floor=g_floor,
        g_sum=total,
        residual=residual,
        closure_db=db(abs(g_user)) - db(abs(residual)),
        closure_phys_db=db(abs(g_source + g_far - g_phys)),
        dominant=order[0],
        runner_up=order[1],
        margin_db=db(abs(terms[order[0]])) - db(abs(terms[order[1]])),
        floor_below_db=db(abs(g_user)) - db(abs(g_far)),
        counterfactual=counterfactual,
        gain_far_db=db(abs(g_user)) - db(abs(counterfactual["far port floor removed"])),
        gain_split_source_db=db(abs(g_user)) - db(abs(counterfactual["split and source removed"])),
        residuals=[got[c]["port1"]["lockin_residual"] for c in got],
    )


def print_attribution(rec):
    f = rec["f"] / 1e9
    print(f"\n    attribution at {f:.2f} GHz  (|Gamma| in dB, vector terms):")
    rows = [
        ("reported S11 (production split)", rec["g_user"]),
        ("  = a/b split term", rec["g_split"]),
        ("  + drive-port source term", rec["g_source"]),
        ("  + far-port floor term", rec["g_far"]),
        ("  + residual", rec["residual"]),
        ("vector sum of the three terms", rec["g_sum"]),
        ("physical (exact split, shipped run)", rec["g_phys"]),
        ("far-port floor, read at the far port", rec["g_far_direct"]),
    ]
    if rec["g_floor"] is not None:
        rows.append(("instrument floor (both ends exact)", rec["g_floor"]))
    for label, val in rows:
        print(
            f"      {label:38s} {db(abs(val)):8.2f} dB   "
            f"phase {math.degrees(np.angle(val)):+7.1f} deg"
        )
    print("      counterfactual reported S11, one term removed, the rest as measured:")
    for label, val in rec["counterfactual"].items():
        gain = db(abs(rec["g_user"])) - db(abs(val))
        print(f"        {label:30s} {db(abs(val)):8.2f} dB   ({gain:+6.2f} dB)")
    print(
        f"      largest term {rec['dominant']!r}, {rec['margin_db']:.2f} dB over "
        f"{rec['runner_up']!r}; the far-port floor sits {rec['floor_below_db']:.2f} dB "
        f"below the reported S11 and removing it buys {rec['gain_far_db']:+.2f} dB"
    )


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def apriori_row(mesh, dt, f, ref=None, chains=None):
    """The arithmetic price of the a/b split -- no time stepping."""
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    face = FACE_DRIVE
    if ref is None:
        spec = PortSpecMultiConductor(name="port1", plane=face, epsilon_r=None)
        op = build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=f)
    else:
        op = ref[face]
    if chains is None:
        chain, plane, m_e, m_m, c_3d = make_chain(mesh, dt, face)
    else:
        chain, plane, m_e, m_m, c_3d = chains[face]
    w_dt = 2.0 * math.pi * f * dt
    ch, _ = analysis_channel(op, chain, plane, m_e, m_m, c_3d, w_dt)
    st = split_terms(ch, op.discrete_modes[0].mode, plane.normal_dx, w_dt / dt, dt)
    gap, gap_db = destagger_formula_db(ch, st["theta_true"])
    st.update(
        f=f,
        nz=int(mesh.Nz),
        cells=int(mesh.Nx * mesh.Ny * mesh.Nz),
        dz=float(plane.normal_dx),
        beta_dz_shipped=float(2.0 * st["theta_shipped"].imag),
        beta_dz_true=float(2.0 * st["theta_true"].imag),
        z_error_pct=100.0 * (st["z_shipped"].real / st["z_true"].real - 1.0),
        theta_error=float(abs(st["theta_shipped"] - st["theta_true"])),
        formula_gap=float(gap),
        formula_db=float(gap_db),
    )
    return st


def print_apriori(rows, title):
    print(f"\n  {title}")
    print(
        "    Nz  dz/mm  f/GHz   Z_ship   Z_true    dZ%    "
        "b*dz_ship b*dz_true |  split  Z only  th only  control |  grid formula"
    )
    for r in rows:
        print(
            f"    {r['nz']:3d} {r['dz'] * 1e3:6.3f} {r['f'] / 1e9:6.2f} "
            f"{r['z_shipped'].real:8.2f} {r['z_true'].real:8.2f} {r['z_error_pct']:+6.2f} "
            f"{r['beta_dz_shipped']:9.5f} {r['beta_dz_true']:9.5f} | "
            f"{db(abs(r['g_split'])):7.2f} {db(abs(r['g_split_z_only'])):7.2f} "
            f"{db(abs(r['g_split_theta_only'])):8.2f} {db(abs(r['g_split_control'])):8.2f} | "
            f"{r['formula_db']:8.2f}"
        )


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, complex):
        return [obj.real, obj.imag]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _cube_and_attribution(mesh, dt, f, configs, store, tag=""):
    """One control cube plus its attribution at one frequency."""
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    chains = {face: make_chain(mesh, dt, face) for face in (FACE_DRIVE, FACE_TEST)}
    ref = {
        face: build_modal_port(
            PortSpecMultiConductor(
                name="port1" if face is FACE_DRIVE else "port2", plane=face, epsilon_r=None
            ),
            mesh,
            m_eps,
            m_mu,
            dt=dt,
            f_calc=f,
        )
        for face in (FACE_DRIVE, FACE_TEST)
    }
    cube = {}
    for config in configs:
        try:
            cube[config] = run_cw(mesh, dt, f, config, ref, chains)
        except Exception as exc:  # noqa: BLE001
            print(f"    {config:13s} f {f / 1e9:5.2f} GHz  FAILED: {exc}", flush=True)
            cube[config] = None
            store["failures"].append(f"{tag}{config}@{f / 1e9:.2f}GHz: {exc}")
    rec = attribute(cube, f)
    ap_row = apriori_row(mesh, dt, f, ref=ref, chains=chains)
    if rec is not None:
        rec["apriori_split_db"] = db(abs(ap_row["g_split"]))
        rec["apriori_delta_db"] = rec["apriori_split_db"] - db(abs(rec["g_split"]))
        rec["nz"] = int(mesh.Nz)
        print_attribution(rec)
        print(
            f"      a-priori a/b split price {rec['apriori_split_db']:.2f} dB vs measured "
            f"{db(abs(rec['g_split'])):.2f} dB  (delta {rec['apriori_delta_db']:+.2f} dB)"
        )
    return cube, rec, ap_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freqs", default="5,10,15", help="comma-separated GHz for the control cube")
    ap.add_argument("--npw", type=int, default=NPW_SHIPPED, help="min_nodes_per_wavelength")
    ap.add_argument(
        "--npw-ladder",
        default="12,18,25,35,50",
        help="cell-size ladder for the a-priori split (comma-separated)",
    )
    ap.add_argument(
        "--refine-npw", type=int, default=40, help="second measured cell size (0 disables)"
    )
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--no-public", action="store_true", help="skip the public S-parameter runs")
    ap.add_argument("--fast", action="store_true", help="one frequency, no ladder, no refinement")
    ap.add_argument("--out", default="", help="write the rows as JSON")
    args = ap.parse_args()

    freqs = [float(x) * 1e9 for x in args.freqs.split(",") if x.strip()]
    configs = [c for c in args.configs.split(",") if c.strip()]
    ladder = [int(x) for x in args.npw_ladder.split(",") if x.strip()]
    refine = args.refine_npw
    if args.fast:
        freqs = freqs[-1:]
        ladder = []
        refine = 0

    print(
        "The user-visible modal-Mur S11 of a microstrip line, certified and decomposed\n"
        "(fixture: tutorial 09's own geometry, mesh control and band):"
    )
    store = {"public": {}, "attribution": [], "apriori": [], "ladder": [], "failures": []}
    recs, ap_rows = [], []

    mesh, dt = tutorial09_mesh(npw=args.npw)
    print(
        f"\n  mesh {mesh.Nx} x {mesh.Ny} x {mesh.Nz} = {mesh.Nx * mesh.Ny * mesh.Nz} cells  "
        f"(min_nodes_per_wavelength {args.npw}, f_max {F_MAX / 1e9:.0f} GHz, "
        f"line {LENGTH * 1e3:.0f} mm)  dt {dt * 1e15:.3f} fs"
    )

    # --- stage 1: the number the user reads ----------------------------
    if not args.no_public:
        pub = public_run(mesh)
        row = {k: v for k, v in pub.items() if k not in ("f_axis", "s11", "s21")}
        row["at_f"] = {}
        print(
            f"\n  public path (AnalysisScatteringTD.run -> compute_s_parameters), "
            f"tutorial 09 prints -32.9 dB:\n"
            f"    |S11| max {pub['s11_max_db']:.2f} dB at {pub['f_at_max'] / 1e9:.2f} GHz   "
            f"|S21| min {pub['s21_min_db']:+.3f} dB   ({pub['runtime_s']:.0f} s)"
        )
        for f in freqs:
            j = int(np.argmin(abs(pub["f_axis"] - f)))
            row["at_f"][f"{pub['f_axis'][j] / 1e9:.2f}"] = db(abs(pub["s11"][j]))
            print(
                f"      |S11| at {pub['f_axis'][j] / 1e9:5.2f} GHz  "
                f"{db(abs(pub['s11'][j])):8.2f} dB"
            )
        store["public"][str(args.npw)] = row

    # --- stage 2: the control cube -------------------------------------
    print("\n  CW control cube (same mesh, dt, driver and analysis channels throughout;")
    print("  only the port build and the postprocessor change):")
    for f in freqs:
        _, rec, ap_row = _cube_and_attribution(mesh, dt, f, configs, store)
        ap_rows.append(ap_row)
        if rec is not None:
            recs.append(rec)
            store["attribution"].append(_jsonable(rec))
        store["apriori"].append(_jsonable(ap_row))

    # --- stage 3: what caps the split ----------------------------------
    if ap_rows:
        print_apriori(ap_rows, "the a/b split, priced term by term (arithmetic, no run):")
        print(
            "    'Z only' keeps the shipped wave impedance and corrects the de-stagger\n"
            "    exponent; 'th only' does the reverse; 'control' corrects both and must\n"
            "    vanish, which is what certifies the two-parameter model of the split.\n"
            "    'grid formula' is (beta*dz/2)^3 (1-r^2)/6 divided by |2 cosh theta| --\n"
            "    the campaign's cap, which prices only the grid-dispersion part of the\n"
            "    de-stagger exponent."
        )

    # --- stage 4: cell-size trend --------------------------------------
    lad_rows = []
    for npw in ladder:
        m2, dt2 = tutorial09_mesh(npw=npw)
        lad_rows.append(apriori_row(m2, dt2, freqs[-1]))
    if lad_rows:
        store["ladder"] = [_jsonable(r) for r in lad_rows]
        print_apriori(lad_rows, f"cell-size ladder at {freqs[-1] / 1e9:.2f} GHz (a priori):")

    if refine:
        m2, dt2 = tutorial09_mesh(npw=refine)
        print(
            f"\n  refined confirmation: mesh {m2.Nx} x {m2.Ny} x {m2.Nz} = "
            f"{m2.Nx * m2.Ny * m2.Nz} cells (min_nodes_per_wavelength {refine}), "
            f"dt {dt2 * 1e15:.3f} fs"
        )
        if not args.no_public:
            pub2 = public_run(m2)
            print(
                f"    public |S11| max {pub2['s11_max_db']:.2f} dB at "
                f"{pub2['f_at_max'] / 1e9:.2f} GHz   |S21| min {pub2['s21_min_db']:+.3f} dB   "
                f"({pub2['runtime_s']:.0f} s)"
            )
            store["public"][str(refine)] = {
                k: v for k, v in pub2.items() if k not in ("f_axis", "s11", "s21")
            }
        _, rec2, _ = _cube_and_attribution(m2, dt2, freqs[-1], configs, store, tag="refined ")
        if rec2 is not None:
            recs.append(rec2)
            store["attribution"].append(_jsonable(rec2))

    acceptance(store, recs, ap_rows + lad_rows)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(store, fh, indent=1, default=str)
        print(f"\n  wrote {args.out}")


# ----------------------------------------------------------------------
# Acceptance
# ----------------------------------------------------------------------


def _gate(n, text, ok):
    print(f"    ({n}) {text} -> {'ok' if ok else 'OFF'}")
    return bool(ok)


def acceptance(store, recs, ap_rows):
    print("\n  Acceptance:")
    ok = True

    pub = store["public"].get(str(NPW_SHIPPED))
    if pub is None:
        print("    (1) public S11: not measured in this selection")
    else:
        d_pin = abs(pub["s11_max_db"] - PINNED_PUBLIC_MAX_DB)
        d_doc = abs(pub["s11_max_db"] - TUTORIAL_PRINTED_DB)
        ok &= _gate(
            1,
            f"public |S11| max {pub['s11_max_db']:.2f} dB reproduces the pin "
            f"{PINNED_PUBLIC_MAX_DB:.2f} dB to {d_pin:.2f} dB (line 0.30) and tutorial 09's "
            f"printed {TUTORIAL_PRINTED_DB:.1f} dB to {d_doc:.2f} dB (line 0.50)",
            d_pin <= 0.30 and d_doc <= 0.50,
        )

    if not recs:
        print("    (1b-9) control cube: not measured in this selection")
        return ok

    n_pin, worst_pin, worst_key = 0, 0.0, ""
    pin_ok = True
    for r in recs:
        pins = PINNED_TERMS.get((r["nz"], round(r["f"] / 1e9, 2)))
        if pins is None:
            continue
        got = {
            "reported": db(abs(r["g_user"])),
            "split": db(abs(r["g_split"])),
            "source": db(abs(r["g_source"])),
            "far": db(abs(r["g_far"])),
            "gain_far": r["gain_far_db"],
            "gain_split_source": r["gain_split_source_db"],
        }
        for key, pin in pins.items():
            n_pin += 1
            tol = 2.0 if pin < -80.0 else 0.5
            d = abs(got[key] - pin)
            if d > worst_pin:
                worst_pin, worst_key = d, f"Nz {r['nz']} {r['f'] / 1e9:.2f} GHz {key}"
            pin_ok &= d <= tol
    if n_pin:
        ok &= _gate(
            "1b",
            f"{n_pin} pinned terms reproduce; worst |delta| {worst_pin:.2f} dB ({worst_key})",
            pin_ok,
        )
    else:
        print("    (1b) pinned terms: none in this selection")

    floors = [r["g_floor"] for r in recs if r["g_floor"] is not None]
    if floors:
        worst = max(db(abs(v)) for v in floors)
        ok &= _gate(
            2,
            f"instrument floor (both ends exact) worst {worst:.2f} dB (line {FLOOR_DB:.0f})",
            worst <= FLOOR_DB,
        )
    worst_closure = min(r["closure_db"] for r in recs)
    ok &= _gate(
        3,
        f"three-term closure: residual is at least {worst_closure:.2f} dB below the "
        f"reported S11 at every point (line {CLOSURE_DB:.0f})",
        worst_closure >= CLOSURE_DB,
    )
    worst_ap = max(abs(r["apriori_delta_db"]) for r in recs)
    ok &= _gate(
        4,
        f"a-priori a/b split price matches the measured split to {worst_ap:.2f} dB (line 0.30)",
        worst_ap <= 0.30,
    )
    ctl = [db(abs(r["g_split_control"])) for r in ap_rows] if ap_rows else []
    if ctl:
        ok &= _gate(
            5,
            f"two-parameter model of the split is complete: worst control "
            f"{max(ctl):.2f} dB (line {SPLIT_CONTROL_DB:.0f})",
            max(ctl) <= SPLIT_CONTROL_DB,
        )
    worst_res = max(max(r["residuals"]) for r in recs)
    ok &= _gate(
        6,
        f"worst lock-in residual {worst_res:.2e} (line {RESIDUAL_MAX:.0e})",
        worst_res <= RESIDUAL_MAX,
    )

    # --- the verdict ---------------------------------------------------
    print("\n  Verdict (the question this certificate exists to answer):")
    for r in recs:
        print(
            f"    Nz {r['nz']:3d}  {r['f'] / 1e9:5.2f} GHz  reported {db(abs(r['g_user'])):7.2f} dB"
            f"   split {db(abs(r['g_split'])):7.2f}   source {db(abs(r['g_source'])):7.2f}"
            f"   far-port floor {db(abs(r['g_far'])):7.2f}"
            f"   | removing the floor buys {r['gain_far_db']:+5.2f} dB,"
            f" removing split+source buys {r['gain_split_source_db']:+6.2f} dB"
        )
    worst_gain = max(abs(r["gain_far_db"]) for r in recs)
    ok &= _gate(
        7,
        f"the far-port floor does NOT dominate: removing it entirely moves the reported "
        f"S11 by at most {worst_gain:.2f} dB at every measured point "
        f"(line {FLOOR_GAIN_DB:.1f})",
        worst_gain <= FLOOR_GAIN_DB,
    )
    # The band top on the SHIPPED mesh: that is where the user's max sits.
    top = max(recs, key=lambda r: (r["f"], -r["nz"]))
    ok &= _gate(
        8,
        f"the leverage is on the split and the source: at the band top "
        f"({top['f'] / 1e9:.2f} GHz, where |S11| peaks) removing both buys "
        f"{top['gain_split_source_db']:+.2f} dB (line {SPLIT_SOURCE_GAIN_DB:+.1f})",
        top["gain_split_source_db"] >= SPLIT_SOURCE_GAIN_DB,
    )
    n_fail = len(store["failures"])
    ok &= _gate(9, f"run points that raised: {n_fail}", n_fail == 0)
    print(
        "\n    Reading: the reported S11 is capped by the a/b split and the drive-port\n"
        "    source, which are the same defect twice -- the frequency-flat quasi-static\n"
        "    port mode -- and the far-port termination floor is a bystander.  Boundary\n"
        "    work cannot move the user-visible number; de-staggering can."
    )
    print(f"\n  overall: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    main()
