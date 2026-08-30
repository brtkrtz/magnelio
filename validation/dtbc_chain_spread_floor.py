"""DD-229 certificate: what a non-uniform feed chain costs in reflection.

The exact discrete transparent boundary (DD-054/DD-055) is derived for
*one* modal Courant number ``r``.  A real feed cross-section delivers a
spread of per-pair values and the termination is built from their
weighted mean, so the boundary is exact for a chain the interior only
approximates.  The uniform-chain gate decides how much of that is
tolerated — and until this measurement it was a *classifier* threshold
(1e-8, placed in the empty gap between roundoff at 1e-13 and the
material contrast of an inhomogeneous line at 1e-1), not an error
budget.  Nothing had measured what a given spread reflects.

Method.  Three fixtures that are exact discrete chains unperturbed: a
staircase parallel plate (TEM, pulsed through the high-level API,
measurement floor -132 dB median) and WR-90 (TE10 and TM11, CW lock-in
through the production operator, floors -160 and -164 dB — the pulsed
overview on a hollow guide is truncation-limited near the band edge
and cannot resolve this).  The transversal ``M_mu`` is then perturbed
*uniformly along the port normal*, so the feed stays translation-
invariant, the stage-2 slab certificate keeps passing, and the only
defect is the transversal non-uniformity this gate is about.  The
value is written into ``face_material`` the way the meshing-time pair
pass writes its own, so the 2D mode solve, the 3D update and the CFL
all read the same line.  The gate is relaxed for the run (measuring
what it protects against is the point) and the in-band |S11| is read
against the spread the port itself reports.

Two perturbation shapes — a smooth tilt and a single lifted column —
and they do not agree, which is the result:

    fixture / shape          fitted law            at spread 2e-6
    TEM plate   ramp     |G| = 1.03  d^2.00           -228 dB
    TEM plate   spike    |G| = 0.143 d^0.99           -129 dB
    WR-90 TE10  ramp     |G| = 0.99  d^2.01           -229 dB
    WR-90 TE10  spike    |G| = 0.21  d^1.94           -235 dB
    WR-90 TM11  ramp     |G| = 43.5  d^2.00           -195 dB
    WR-90 TM11  spike    |G| = 1.34  d^1.89           -213 dB

Fits over the resolvable range of each fixture (up to four decades of
spread, 80 to 120 dB of reflection).

The mechanism is first-order perturbation theory.  The parametrisation
annihilates the first-order error by construction, so what survives is
the overlap of the perturbation with the mode.  A linear ramp is
antisymmetric against a symmetric mode and that overlap vanishes — the
reflection falls to second order.  A localised defect does not vanish
against anything.  It stays first order only for TEM, though: on TE/TM
the chain also carries ``q``, taken from the 2D eigenvalue, and an
eigenvalue is a Rayleigh quotient — first-order accurate by
construction — so the localised defect is absorbed there too and the
reflection returns to second order.  **The pessimum is therefore a
localised defect on a TEM channel**, where there is no eigenvalue to
absorb it.

The scalar RMS spread cannot tell the shapes apart, so the gate is set
against the pessimum and, on top of it, against the deliberately crude
bound ``|Gamma| <= spread`` — a 17 dB margin on that measured 0.143
coefficient, which is itself geometry-dependent (a wider defect
overlaps the mode more per unit RMS).

Budget against the -100 dB acceptance line:

    spread     bound |G|<=d    measured worst    what sits there
    1e-8          -160 dB         -177 dB        the old gate
    2e-6          -114 dB         -129 dB        the gate (DD-229)
    1.7e-8        -155 dB         -172 dB        the one measured
                                                 conformal port spread
                                                 (DD-165, rejected by
                                                 the old gate)
    1e-4           -80 dB          -96 dB        upper edge of the
                                                 warning band
    2e-1           -14 dB          -30 dB        inhomogeneous QTEM
                                                 line: Mur's own floor

The old gate was protecting a -177 dB floor against a -100 dB
acceptance line — 77 dB of headroom nobody used — while a conformal
cross-section carrying ordinary B-Rep tolerance lands near 1e-6 and
was rejected onto a -30 dB absorber.  The last row is the other end:
near a spread of 0.2 the exact termination is no better than Mur, and
that is exactly where an inhomogeneous line sits, so the gate is right
to reject it there (and ``port_model="auto"`` is right to send it to
the band pipeline).

Limits.  Three fixtures, one defect width, ``mu``-side perturbations
only.  The coefficient of the first-order law is geometry-dependent,
which is why the gate is set against the bound rather than the fit.

Run:  CUPY_ACCELERATORS="" python validation/dtbc_chain_spread_floor.py
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.special import erf

from magnelio import AnalysisScatteringTD, Mesh
from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.geo._subcell import (
    FaceMaterialData,
    _build_A_face_H,
    _build_L_dual_H,
)
from magnelio.mesh import BoxFace
from magnelio.mesh.grid import GridLines
from magnelio.ports import PortSpecMultiConductor, PortSpecNumerical
from magnelio.ports._modal import ModeType, build_modal_port
from magnelio.ports._modal import operator as _op
from magnelio.ports._modal.dtbc import destagger_theta, dtbc_wave_impedance
from magnelio.ports.recorder import PortSignalRecorder
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import (
    compute_min_effective_eps,
    compute_min_effective_mu,
    courant_dt,
)

AMPLITUDES = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def parallel_plate_mesh() -> Mesh:
    """Uniform staircase plate line — an exact discrete TEM chain."""
    width_a, gap_b, length = 10e-3, 5e-3, 20e-3
    grid = GridLines(
        x=np.linspace(-width_a / 2, width_a / 2, 11),
        y=np.linspace(-gap_b / 2, gap_b / 2, 6),
        z=np.linspace(-length / 2, length / 2, 41),
    )
    return Mesh.from_grid(
        grid,
        boundary_conditions={
            "xmin": "PMC",
            "xmax": "PMC",
            "ymin": "PEC",
            "ymax": "PEC",
            "zmin": "PEC",
            "zmax": "PEC",
        },
    )


def wr90_mesh() -> Mesh:
    """WR-90 hollow guide — the TE/TM fixture of the DD-055 floors."""
    a, b, length = 22.86e-3, 10.16e-3, 40e-3
    grid = GridLines(
        x=np.linspace(0.0, a, 21),
        y=np.linspace(0.0, b, 10),
        z=np.linspace(0.0, length, 41),
    )
    return Mesh.from_grid(grid).with_boundary_conditions(
        {
            "xmin": "PEC",
            "xmax": "PEC",
            "ymin": "PEC",
            "ymax": "PEC",
            "zmin": "PEC",
            "zmax": "PEC",
        }
    )


def perturb_transversal_mu(mesh: Mesh, amplitude: float, shape: str) -> Mesh:
    """Break the transversal uniformity of M_mu, two ways.

    ``"ramp"`` tilts the masses linearly across the section; ``"spike"``
    lifts a single interior column and leaves the rest exact.  Both are
    constant along y and z, so the line stays translation-invariant
    along the port normal: the stage-2 slab certificate is untouched
    and the only defect is the transversal spread this gate is about.

    The two shapes are not interchangeable as a test.  The gate is a
    single scalar (weighted RMS), and a smooth tilt and a localised
    defect can produce the same scalar from very different
    cross-sections — a localised one is what KB-022 actually saw (one
    conformal edge beside a mirrored bore).  If the reflection law
    depended on the shape, the scalar gate would be measuring the
    wrong thing.

    Encoded the way ``couple_face_material_pairs`` encodes its own
    result (category 2, ``L_dual_free = L_dual``), so ``build_M_mu``,
    the 2D mode solvers and ``courant_dt`` all read the same line.
    """
    grid = mesh.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_Hx = (Nx + 1) * Ny * Nz
    n_Hy = Nx * (Ny + 1) * Nz
    A_face = _build_A_face_H(grid)
    L_dual = _build_L_dual_H(grid)
    fm = mesh.face_material
    if fm is None:
        # ``from_grid`` runs no classifier, so the staircase line has
        # no face record to override; create an all-bulk one.
        n = A_face.size
        fm = FaceMaterialData(
            category=np.zeros(n, dtype=np.int8),
            mu_avg=np.full(n, np.nan),
            A_face_free=np.full(n, np.nan),
            L_dual_free=np.full(n, np.nan),
        )
        mesh.face_material = fm

    def profile(n_along: int) -> np.ndarray:
        if shape == "ramp":
            t = np.arange(n_along, dtype=float) / max(n_along - 1, 1) - 0.5
            return 1.0 + amplitude * t
        f = np.ones(n_along)
        f[n_along // 2] = 1.0 + amplitude
        return f

    scale = np.ones(A_face.size)
    scale[:n_Hx] = np.broadcast_to(profile(Nx + 1)[:, None, None], (Nx + 1, Ny, Nz)).reshape(-1)
    scale[n_Hx : n_Hx + n_Hy] = np.broadcast_to(
        profile(Nx)[:, None, None], (Nx, Ny + 1, Nz)
    ).reshape(-1)

    touched = scale != 1.0
    fm.category[touched] = 2
    fm.mu_avg[touched] = 1.0
    fm.A_face_free[touched] = scale[touched] * A_face[touched]
    fm.L_dual_free[touched] = L_dual[touched]

    # The defect must stay invisible to the stage-2 slab certificate,
    # or the measurement would be about a different failure.  (linspace
    # leaves last-bit jitter in dz, hence a tolerance, not bit equality.)
    m_mu = build_M_mu(mesh)
    hx = m_mu[:n_Hx].reshape(Nx + 1, Ny, Nz)
    assert np.allclose(hx, hx[:, :, :1], rtol=1e-12, atol=0.0), "perturbation is not z-uniform"
    return mesh


# ----------------------------------------------------------------------
# Leg 1 — TEM, broadband pulsed through the high-level API
# ----------------------------------------------------------------------


def pulsed_floor(mesh: Mesh, f_max: float, band: tuple[float, float]):
    ana = AnalysisScatteringTD(
        mesh=mesh,
        ports=[
            PortSpecMultiConductor(name="port1", plane=BoxFace.Z_MIN, n_modes=1),
            PortSpecMultiConductor(name="port2", plane=BoxFace.Z_MAX, n_modes=1),
        ],
        f_max=f_max,
        verbose=False,
    )
    mode = ana.solve_ports()["port1"].modes[0]
    res = ana.run()
    s11 = res.db("port1", "port1")
    f = np.asarray(res.f_axis)
    sel = (f >= band[0]) & (f <= band[1])
    return float(mode.chain_spread), float(np.nanmedian(s11[sel]))


# ----------------------------------------------------------------------
# Leg 2 — TE/TM, CW lock-in (the DD-055 instrument; the pulsed floor
# on a hollow guide is truncation-limited near the band edge and cannot
# resolve what is measured here)
# ----------------------------------------------------------------------


def cw_floor(mesh: Mesh, mode_type: ModeType, f_calc: float, ratio: float):
    """Steady-state |S11| of port 1 at ``ratio`` times the discrete cut-off."""
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    # The perturbation lowers M_mu on some faces, which raises the
    # local wave speed: the bare grid CFL is no longer stable.  Thread
    # the effective material minima through, exactly as the production
    # analysis does.
    dt = courant_dt(
        mesh.grid,
        "normal",
        min_effective_eps=compute_min_effective_eps(mesh),
        min_effective_mu=compute_min_effective_mu(mesh),
    )
    ports = [
        build_modal_port(
            PortSpecNumerical(name=name, plane=plane, n_modes=1, mode_type=mode_type),
            mesh,
            m_eps,
            m_mu,
            dt=dt,
            f_calc=f_calc,
        )
        for name, plane in (("port1", BoxFace.Z_MIN), ("port2", BoxFace.Z_MAX))
    ]
    op1, op2 = ports
    assert op1.termination_kinds == ["dtbc"], op1.termination_kinds
    spread = float(op1._dtbc_pair_spread[0])
    r, q, z0 = op1.dtbc_line_params[0]

    w_dt = ratio * q
    period = 2.0 * math.pi / w_dt
    sigma = max(6.0 / ((ratio - 1.0) * q), 8.0 * period)
    s_hat = math.sin(w_dt / 2.0)
    sin_b2 = math.sqrt(max(s_hat**2 - (q / 2.0) ** 2, 1e-30)) / r
    v_g = r * r * math.sin(2.0 * math.asin(min(sin_b2, 1.0))) / math.sin(w_dt)
    n_win = int(30 * period)
    n_meas0 = int(10.0 * sigma + 40.0 * period + 3.0 * mesh.Nz / max(v_g, 1e-3))
    n_steps = n_meas0 + n_win + 2

    t0, sig_t, w_phys = 5.0 * sigma * dt, sigma * dt, w_dt / dt

    def waveform(t: float) -> float:
        amp = 0.5 * (1.0 + float(erf((t - t0) / (math.sqrt(2.0) * sig_t))))
        return amp * math.sin(w_phys * t)

    op1.set_excitation(0, waveform)
    recorder = PortSignalRecorder(dt=dt, ports=ports)
    solver = FITTimeDomainSolver(
        mesh=mesh,
        boundary_conditions={},
        ports=ports,
        recorder=recorder,
        total_time_steps=n_steps,
        dt=dt,
        verbose=False,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*neither a BoundaryCondition.*")
        solver.run()
    signals = recorder.finalize(n_steps_actual=n_steps)
    V_sig, I_sig = signals[("port1", 0)]

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
    return spread, 20.0 * math.log10(max(abs(b / a), 1e-300))


# ----------------------------------------------------------------------


def fit(points, floor_db):
    """Power-law fit over the points the measurement floor does not hide."""
    usable = [(d, g) for d, g in points if g > floor_db + 10.0 and d > 0.0]
    if len(usable) < 3:
        return None
    d = np.log10([x for x, _ in usable])
    g = np.array([y for _, y in usable]) / 20.0
    slope, icpt = np.polyfit(d, g, 1)
    return 10.0**icpt, slope, len(usable)


def sweep(label, run, floor_db):
    print(f"\n  {label}")
    print(f"    {'shape':>6} {'amp':>8} {'spread':>10} {'|S11|':>10}")
    for shape in ("ramp", "spike"):
        points = []
        for amp in AMPLITUDES:
            spread, s11 = run(amp, shape)
            points.append((spread, s11))
            print(f"    {shape:>6} {amp:8.0e} {spread:10.2e} {s11:10.1f}")
        f = fit(points, floor_db)
        if f is None:
            print(f"    {shape:>6} fit: not enough points above the floor")
        else:
            coeff, slope, n = f
            print(
                f"    {shape:>6} fit: |Gamma| = {coeff:.3f} * delta^{slope:.3f}"
                f"   ({n} points above floor{floor_db + 10:+.0f} dB)"
            )


def main() -> None:
    # The gate is what is under test; relaxing it is the experiment.
    tol = _op._DTBC_PAIR_SPREAD_TOL
    _op._DTBC_PAIR_SPREAD_TOL = 1.0
    try:
        print("DD-229 chain-spread reflection budget (exact DTBC forced):")

        base_spread, base_floor = pulsed_floor(parallel_plate_mesh(), 10e9, (0.25e9, 10e9))
        print(
            f"\n  TEM parallel plate, pulsed, band 0.25-10 GHz — unperturbed "
            f"spread {base_spread:.2e}, |S11| median {base_floor:.1f} dB"
        )

        def tem(amp, shape):
            mesh = perturb_transversal_mu(parallel_plate_mesh(), amp, shape)
            return pulsed_floor(mesh, 10e9, (0.25e9, 10e9))

        sweep("TEM parallel plate (pulsed median)", tem, base_floor)

        for mode_type, f_calc, ratio in (
            (ModeType.TE, 10.0e9, 1.5),
            (ModeType.TM, 20.0e9, 1.2),
        ):
            b_spread, b_floor = cw_floor(wr90_mesh(), mode_type, f_calc, ratio)
            name = f"WR-90 {'TE10' if mode_type is ModeType.TE else 'TM11'}"
            print(
                f"\n  {name}, CW lock-in at {ratio} f_c — unperturbed "
                f"spread {b_spread:.2e}, |S11| {b_floor:.1f} dB"
            )

            def run(amp, shape, _mt=mode_type, _fc=f_calc, _r=ratio):
                mesh = perturb_transversal_mu(wr90_mesh(), amp, shape)
                return cw_floor(mesh, _mt, _fc, _r)

            sweep(f"{name} (CW lock-in)", run, b_floor)
    finally:
        _op._DTBC_PAIR_SPREAD_TOL = tol


if __name__ == "__main__":
    main()
