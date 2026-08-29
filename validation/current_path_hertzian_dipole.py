#!/usr/bin/env python
"""DD-227: the impressed current filament against the Hertzian dipole.

An open ``SourceCurrentPath`` two and a half millimetres long, at the
centre of an air cube with CPML on all six faces, is the textbook short
dipole: a uniform current ``I`` over a length ``L`` much smaller than
the wavelength.  Its far field and radiated power are known in closed
form, so the whole chain — the sign of the injection, the ``β·I``
scaling, the rasterised chord, the near-to-far-field transform — is
measured against analysis rather than against another run.

Reference.  For a *uniform* filament of any length (not only ``kL → 0``)
the far-zone amplitude is

    r·E_θ = j η₀ k I L sin θ · sinc(kL cos θ / 2) / (4π)

and integrating its intensity over the sphere gives

    P = η₀ k² I² L² / (32π²) · 2π ∫₀^π sin³θ sinc²(kL cos θ / 2) dθ,

which collapses to the Hertzian ``η₀ (k I L)² / (12π)`` as ``kL → 0``.
Library phasors are effective amplitudes (``U = |E|²/η₀``, no ½), so
the value ``FarFieldResult.P_rad`` is compared against is twice that —
the same reading that makes a port's ``amplitude`` one watt CW.

What the ladder shows.  The measured power approaches the analytic
value from below with an order of about 1.3 in nodes per wavelength
and no offset left over: the residual at each resolution tracks the
near-to-far-field box's own closure deficit (``power_balance``), not
the source.

    nodes/λ   grid       P_rad / P_exact   power_balance
    10        80³        0.9642            0.9892
    12        92³        0.9719            0.9914
    14       104³        0.9767            0.9922
    18       128³        0.9831            0.9945

(2026-08-30, 45 mm half-domain, f₀ = 10 GHz, L = 2.5 mm, kL = 0.524.)
The pattern is ``sin θ`` throughout, ``|E_φ| / |E_θ|max`` stays near
2 %, and ``arg(E_θ / j)`` is 180° — the library's far-zone amplitude
is the conjugate of the ``e^{+jωt}`` textbook form (DD-204).  The
*sign* of the injection is not read off that phase but from the exact
charge-continuity identity in
``tests/integration/test_source_current_path.py``.

Run:  python validation/current_path_hertzian_dipole.py
        [--nodes 10 12 14 18] [--half 45e-3] [--length 2.5e-3]
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
from scipy.integrate import quad

import magnelio as mio
from magnelio import geo, monitors, signals, sources
from magnelio.constants import C0, ETA0

F0 = 10e9
F_MAX = 20e9
FACES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")


def exact_power(k: float, length: float) -> float:
    """Radiated power of a uniform filament, in the library's convention."""
    kl = k * length

    def integrand(theta: float) -> float:
        return np.sin(theta) ** 3 * np.sinc(kl * np.cos(theta) / 2 / np.pi) ** 2

    return ETA0 * k**2 * length**2 / (8 * np.pi) * quad(integrand, 0.0, np.pi)[0]


def run(nodes: int, half: float, length: float):
    model = mio.GeometryModel(boundary_conditions=dict.fromkeys(FACES, "CPML"))
    model.add(geo.Brick(origin=(-half,) * 3, size=(2 * half,) * 3, material="air"))
    model.add_source(
        sources.SourceCurrentPath(name="fil", path=[(0, 0, -length / 2), (0, 0, length / 2)])
    )
    mesh = mio.Mesh.from_geometry(
        model, mio.MeshControl(min_nodes_per_wavelength=nodes), f_max=F_MAX
    )
    # The source's own grid planes put both ends on nodes; read back the
    # length the rasteriser actually sees.
    z = np.asarray(mesh.grid.z)
    snapped = float(z[np.argmin(abs(z - length / 2))] - z[np.argmin(abs(z + length / 2))])

    ff = monitors.MonitorFarFieldFrequency(name="pattern", freqs=[F0])
    result = mio.AnalysisTD(mesh=mesh, monitors=[ff], verbose=False).run(
        excitations=[
            mio.Excitation("fil", waveform=signals.WaveformGaussian(f_max=F_MAX), amplitude=1.0)
        ],
        t_end=2000e-12,
        energy_stop_db=60,
    )
    result.renormalize("fil")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pattern = ff.result(F0)

    k = 2.0 * np.pi * F0 / C0
    reference = exact_power(k, snapped)
    sin_theta = np.sin(pattern.theta)
    equator = int(np.argmin(abs(pattern.theta - np.pi / 2)))
    profile = np.abs(pattern.E_theta).mean(axis=1)
    shape_error = float(np.abs(profile / profile[equator] - sin_theta).max())
    e_ref = (
        1j
        * ETA0
        * k
        * snapped
        / (4 * np.pi)
        * sin_theta[equator]
        * np.sinc(k * snapped * np.cos(pattern.theta[equator]) / 2 / np.pi)
    )
    phase = float(np.angle(pattern.E_theta[equator].mean() / e_ref, deg=True))
    return {
        "grid": (mesh.Nx, mesh.Ny, mesh.Nz),
        "length": snapped,
        "ratio": pattern.P_rad / reference,
        "balance": pattern.power_balance,
        "directivity": float(pattern.directivity.max()),
        "shape_error": shape_error,
        "cross_pol": float(np.abs(pattern.E_phi).max() / np.abs(pattern.E_theta).max()),
        "phase": phase,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", type=int, nargs="+", default=[10, 12, 14, 18])
    ap.add_argument("--half", type=float, default=45e-3)
    ap.add_argument("--length", type=float, default=2.5e-3)
    args = ap.parse_args()

    k = 2.0 * np.pi * F0 / C0
    print(f"kL = {k * args.length:.4f}   half-domain {args.half * 1e3:.0f} mm   f0 = 10 GHz")
    print(
        f"{'nodes/λ':>8} {'grid':>14} {'P/P_exact':>10} {'balance':>9} {'D_max':>7} "
        f"{'shape':>7} {'x-pol':>7} {'arg':>8}"
    )
    rows = []
    for nodes in args.nodes:
        r = run(nodes, args.half, args.length)
        rows.append((nodes, r["ratio"]))
        grid = "×".join(str(n) for n in r["grid"])
        print(
            f"{nodes:>8} {grid:>14} {r['ratio']:>10.4f} {r['balance']:>9.4f} "
            f"{r['directivity']:>7.4f} {r['shape_error']:>7.4f} {r['cross_pol']:>7.4f} "
            f"{r['phase']:>7.1f}°",
            flush=True,
        )
    if len(rows) >= 2:
        n = np.array([r[0] for r in rows], dtype=float)
        err = np.abs(1.0 - np.array([r[1] for r in rows]))
        order = -np.polyfit(np.log(n), np.log(err), 1)[0]
        print(f"convergence order in nodes per wavelength ≈ {order:.2f}")


if __name__ == "__main__":
    main()
