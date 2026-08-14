"""WP-U6 acceptance probes: multi-mode QTEM port on the shielded microstrip.

Two measurement legs beyond the high-level example
(``examples/straight_waveguide_microstrip.py``):

A. **Fundamental invariance.**  The QTEM fundamental of the
   multi-mode port must reproduce the single-mode (DD-064) numbers:
   the channel stays the Laplace QTEM profile with Mur termination;
   only the *projection* switches to the dual basis (the hybrid
   profiles overlap the fundamental).  Measured: |S11| and |S21| of
   the fundamental through ``n_modes=1`` vs. ``n_modes=3`` ports over
   the full band.

B. **Hybrid profile drift.**  ζ-pencil eigenprofiles are exact at
   ``f_calc`` and frequency-dependent (hybrid modes have no
   frequency-independent profile).  Measured: the M_ε-normalised
   profile mismatch ``1 - |<phi(f), phi(f_calc)>_W|`` of each hybrid
   channel across the band — the honest QTEM analogue of the
   homogeneous case, where profiles are frequency-independent and the
   merge is exact (PORT_MODES_PLAN.md WP-U6).

Results (2026-07-11, WP-U6):

    A. Below the first hybrid cut-off (2-16 GHz): max |dS11| =
       5.2e-2, max |dS21| = 5.5e-2 between the 1-mode and 3-mode
       ports — within the Mur measurement class of the default path
       (the 3-mode dual projection separates the evanescent hybrid
       tails that the 1-mode primal projection folds into the QTEM
       channel).  Full band (to 35 GHz): the 1-mode port is
       under-modelled above the hybrid cut-offs — the injected
       Laplace profile excites PROPAGATING hybrids that a 1-channel
       port neither absorbs nor separates: max |S11| -1.18 dB /
       |S21| overshoot +3.0 dB, against -21.2 dB / +0.6 dB on the
       3-mode port.  The multi-mode port is not an optional extra on
       a band crossing hybrid cut-offs; it is what makes the QTEM
       numbers meaningful there.
    B. Hybrid profile drift vs. the f_calc = 35 GHz profile
       (best-overlap matching): 4.7e-4 / 2.7e-4 at 34 GHz,
       2.1e-2 / 4.9e-3 at 30 GHz, 1.1e-1 / 1.3e-2 at 26 GHz,
       evan. / 3.6e-2 at 18.5 GHz — exact at f_calc, growing with
       band distance; the honest QTEM analogue of the homogeneous
       merge (where profiles are frequency-independent).

Run:  python validation/qtem_multimode_port_probe.py
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from magnelio import AnalysisScatteringTD, Material, Mesh, MeshControl
from magnelio._operators.curl import build_curl_matrix
from magnelio._operators.material_matrices import (
    build_M_eps,
    build_M_mu,
    flatten_port_plane_mass,
    flatten_port_plane_mu,
    flatten_port_plane_pec_mask,
)
from magnelio.geo import Brick, Difference, GeometryModel
from magnelio.ports import PortWaveguide
from magnelio.ports._modal import BoxFace
from magnelio.ports._modal.port_plane import PortPlane
from magnelio.ports._modal.zeta_pencil import (
    build_period_blocks,
    find_propagating_modes,
)
from magnelio.solver.stability import (
    compute_min_effective_eps,
    compute_min_effective_mu,
    courant_dt,
)

H_SUB, W_STRIP, T_STRIP = 0.8e-3, 1.5e-3, 0.2e-3
W_BOX, H_BOX, LENGTH = 8.0e-3, 5.0e-3, 12.0e-3
EPS_R, F_MAX = 4.3, 35.0e9


def microstrip_mesh():
    pec = Material.pec()
    air = Material.from_isotropic(name="air", epsilon=1.0)
    diel = Material.from_isotropic(name="FR4", epsilon=EPS_R)
    model = GeometryModel()
    model.add(Brick(origin=(-W_BOX / 2, 0.0, 0.0), size=(W_BOX, H_SUB, LENGTH), material=diel))
    air_cap = Brick(
        origin=(-W_BOX / 2, H_SUB, 0.0), size=(W_BOX, H_BOX - H_SUB, LENGTH), material=air
    )
    strip = Brick(origin=(-W_STRIP / 2, H_SUB, 0.0), size=(W_STRIP, T_STRIP, LENGTH), material=pec)
    model.add(Difference(air_cap, strip))
    model.add(strip)
    return Mesh.from_geometry(
        model,
        MeshControl(min_nodes_per_wavelength=15),
        f_max=F_MAX,
    )


def _analysis(n_modes: int) -> AnalysisScatteringTD:
    return AnalysisScatteringTD(
        mesh=microstrip_mesh().with_boundary_conditions(
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
            PortWaveguide(name="port1", plane="zmin", n_modes=n_modes),
            PortWaveguide(name="port2", plane="zmax", n_modes=n_modes),
        ],
        f_max=F_MAX,
        verbose=False,
    )


def leg_a_fundamental_invariance():
    f_axis = np.linspace(2.0e9, F_MAX, 121)
    res1 = _analysis(1).run(f_axis=f_axis, excited=[("port1", 0)])
    res3 = _analysis(3).run(f_axis=f_axis, excited=[("port1", 0)])
    # Below the first hybrid cut-off (~17 GHz on this mesh) the
    # single-mode and 3-mode ports must agree — hybrids are
    # evanescent there, so the only differences are the dual-basis
    # projection and the evanescent-tail absorption.  ABOVE it the
    # single-mode port is under-modelled: the injected Laplace
    # profile is not the exact QTEM(f) mode, the mismatch excites
    # PROPAGATING hybrids, and a 1-channel port neither absorbs nor
    # separates them — the very gap the multi-mode port closes.
    low = f_axis <= 16.0e9
    for name, out_port in (("S11", "port1"), ("S21", "port2")):
        s1 = res1.S(out_port, "port1")
        s3 = res3.S(out_port, "port1", mode_out=0, mode_in=0)
        d_low = np.max(np.abs(s3[low] - s1[low]))
        db1 = 20 * np.log10(np.abs(s1) + 1e-30)
        db3 = 20 * np.log10(np.abs(s3) + 1e-30)
        print(
            f"    {name}: below 16 GHz max |dS| {d_low:.3e}   "
            f"full-band max: n=1 {np.max(db1):+7.2f} dB vs "
            f"n=3 {np.max(db3):+7.2f} dB"
        )


def leg_b_profile_drift():
    mesh = microstrip_mesh().with_boundary_conditions(
        {
            "xmin": "PEC",
            "xmax": "PEC",
            "ymin": "PEC",
            "ymax": "PEC",
            "zmin": "PMC",
            "zmax": "PMC",
        }
    )
    m_eps = flatten_port_plane_mass(build_M_eps(mesh), mesh, BoxFace.Z_MIN)
    m_mu = flatten_port_plane_mu(build_M_mu(mesh), mesh, BoxFace.Z_MIN)
    object.__setattr__(
        mesh,
        "pec_mask_edges",
        flatten_port_plane_pec_mask(mesh.pec_mask_edges, mesh, BoxFace.Z_MIN),
    )
    dt = courant_dt(
        mesh.grid,
        "normal",
        min_effective_eps=compute_min_effective_eps(mesh),
        min_effective_mu=compute_min_effective_mu(mesh),
    )
    plane = PortPlane.from_mesh(BoxFace.Z_MIN, mesh)
    c_3d = build_curl_matrix(mesh.grid)
    chain = build_period_blocks(plane, mesh, m_eps, m_mu, c_3d, dt)
    dz = float(plane.normal_dx)
    C0 = 299_792_458.0
    w = chain.w_period

    def hybrid_profiles(f):
        w_dt = 2.0 * math.pi * f * dt
        theta0 = 2.0 * math.pi * f * math.sqrt(3.7) / C0 * dz
        zp, pp = find_propagating_modes(chain, w_dt, 1.5 * theta0)
        order = np.argsort([-abs(np.angle(z)) for z in zp])
        return zp[order], pp[:, order]

    z_ref, p_ref = hybrid_profiles(F_MAX)
    print(f"    reference at f_calc = {F_MAX / 1e9:.0f} GHz: {z_ref.size} propagating modes")
    print(
        f"    {'f [GHz]':>8} "
        + "".join(f"{'drift m' + str(j):>12}" for j in range(1, min(3, z_ref.size)))
    )

    def norm(p):
        return math.sqrt(abs(np.vdot(p, w * p)))

    for f in (34.0e9, 30.0e9, 26.0e9, 22.0e9, 18.5e9):
        zf, pf = hybrid_profiles(f)
        row = f"    {f / 1e9:8.1f} "
        for j in range(1, min(3, z_ref.size)):
            # Best-overlap matching: the phase-advance ordering is
            # frequency-dependent, so pair each f_calc profile with
            # its closest counterpart at f.
            best = max(
                (
                    abs(np.vdot(p_ref[:, j], w * pf[:, k])) / (norm(p_ref[:, j]) * norm(pf[:, k]))
                    for k in range(zf.size)
                ),
                default=0.0,
            )
            row += f"{1.0 - best:12.3e}" if best > 0.5 else f"{'evan.':>12}"
        print(row)


def main() -> None:
    warnings.filterwarnings("ignore", message=".*Mur.*")
    print("WP-U6 — multi-mode QTEM port probes (shielded microstrip)")
    print("Leg A: fundamental invariance (n_modes=1 vs 3, dual projection):")
    leg_a_fundamental_invariance()
    print("Leg B: hybrid profile drift vs the f_calc profile:")
    leg_b_profile_drift()


if __name__ == "__main__":
    main()
