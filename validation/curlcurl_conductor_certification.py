"""WP-U1: curl-curl TE/TM certification on conductor cross-sections.

The TE/TM curl-curl path (``PortSpecNumerical``) was documented for
*hollow* cross-sections only; the WP-U2 merge (TEM (+) TE/TM in one
modal port) needs it certified on multiply-connected domains.  Three
measurement legs (PORT_MODES_PLAN.md WP-U1):

A. **Coax TE11/TM01 cut-offs vs. the Bessel cross-product roots** on
   the conformally meshed RG-58 cross-section, over a resolution
   series — convergence order, not just one mesh.  The inner
   conductor enters the 2D eigenproblem as Dirichlet boundary through
   ``mesh.pec_mask_edges`` with no code change.

B. **PEC/PMC parallel-plate cut-off ladder** after WP-U0 stage 2
   (PMC declared on the model), magnetic wall ON the bbox face.  With PMC
   x-walls Hz is *Dirichlet* (H_tan = 0), so TE modes need m >= 1:
   TE ladder f_mn = sqrt((m c/2a)^2 + (n c/2b)^2), m >= 1; the
   ``n c/2b`` family lives in the TM spectrum (Ez = 0 on the PEC
   plates, natural on PMC): TM ladder with n >= 1.

C. **Discrete TEM x TE/TM cross-orthogonality** on the coax and the
   shielded two-wire cross-sections — the analytic-derivation gate
   before any WP-U2 merge code.  Measured through the *production*
   projections: ``project_V`` (M_eps inner product of the E profiles)
   and ``project_I`` (M_mu-weighted H profiles) of one family's
   operator applied to the other family's discrete mode fields,
   normalised to the self-projection.  For homogeneous fillings the
   continuum families are exactly orthogonal (TEM ⊕ TE ⊕ TM); the
   discrete numbers must reach solver tolerance.

Results (2026-07-11, WP-U1; leg A re-measured after the DD-067
port-plane mu-flatten, which changes the boundary Hz-M_mu the TE
solve consumes):

    A. coax TE11 rel. err +1.6e-2 (dx 0.24 mm) -> -1.8e-2 -> -6.7e-3
       -> -4.0e-4 (0.03 mm); TM01 -5.3e-2 -> +1.3e-2 -> +7.9e-3 ->
       +3.9e-5 (TM is untouched by DD-067 — the normal-face M_mu
       does not enter the TM node Laplacian).  Both signs occur: the
       conformal-boundary cut-off error oscillates with grid
       alignment about the analytic value (per-pair order estimates
       are therefore not meaningful individually); the error
       envelope falls from the few-percent class to the 1e-4..1e-3
       class over 8x refinement with NO systematic bias.  TE11
       degenerate-pair split <= 4.3e-7.
    B. TE ladder rel. err -7.2e-4 / -2.8e-3 / -2.1e-3;
       TM ladder -2.4e-3 / -2.1e-3 (0.4 mm cells - the mesh
       dispersion class; the pre-WP-U0 wall bias was -4.9e-2).
    C. coax TEM x TE11 pair:  V 3.5e-15, I 1.8e-14
       coax TEM x TM01:       V 8.1e-16, I 2.2e-14
       two-wire 2 TEM x 2 TE: V 5.1e-15, I 1.7e-14
       -> the discrete families are orthogonal at solver tolerance;
       the WP-U2 merge is analytically grounded.

Run:  python validation/curlcurl_conductor_certification.py
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import jv, jvp, yv, yvp

from magnelio import Material, Mesh, MeshControl
from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.geo import Brick, Cylinder, Difference, GeometryModel
from magnelio.ports import PortSpecNumerical
from magnelio.ports._modal import (
    BoxFace,
    ModeType,
    PortSpecMultiConductor,
    build_modal_port,
)
from magnelio.solver.stability import courant_dt

C0 = 299_792_458.0

# RG-58 class (straight_waveguide_coax_rg58.py)
R_I = 0.405e-3
R_A = 1.475e-3
EPS_R = 2.25


# ----------------------------------------------------------------------
# Analytic references — coax higher-mode cut-offs
# ----------------------------------------------------------------------


def coax_te_cutoff(m: int, r_i: float, r_a: float, eps_r: float) -> float:
    """First root of Jm'(k r_i) Ym'(k r_a) - Jm'(k r_a) Ym'(k r_i)."""

    def f(k):
        return jvp(m, k * r_i) * yvp(m, k * r_a) - jvp(m, k * r_a) * yvp(m, k * r_i)

    k0 = 2.0 / (r_i + r_a)  # TE11-class estimate
    k = brentq(f, 0.3 * k0, 2.0 * k0)
    return k * C0 / (2.0 * math.pi * math.sqrt(eps_r))


def coax_tm_cutoff(r_i: float, r_a: float, eps_r: float) -> float:
    """First root of J0(k r_i) Y0(k r_a) - J0(k r_a) Y0(k r_i)."""

    def f(k):
        return jv(0, k * r_i) * yv(0, k * r_a) - jv(0, k * r_a) * yv(0, k * r_i)

    k0 = math.pi / (r_a - r_i)  # TM01-class estimate
    k = brentq(f, 0.5 * k0, 1.5 * k0)
    return k * C0 / (2.0 * math.pi * math.sqrt(eps_r))


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def coax_mesh(dx: float) -> Mesh:
    pec = Material.pec()
    diel = Material.from_isotropic(name="polyethylene", epsilon=EPS_R)
    length = 12.0 * dx
    outer = Cylinder(origin=(0.0, 0.0, 0.0), radius=R_A, height=length, axis="z", material=diel)
    inner = Cylinder(origin=(0.0, 0.0, 0.0), radius=R_I, height=length, axis="z", material=pec)
    model = GeometryModel(background=pec)
    model.add(Difference(outer, inner))
    model.add(inner)
    return Mesh.from_geometry(
        model,
        MeshControl(min_nodes_per_wavelength=15, max_cell_size=dx),
        f_max=50.0e9,
    )


def two_wire_mesh() -> Mesh:
    w, s, box, length = 1.0e-3, 3.0e-3, 10.0e-3, 6.0e-3
    pec = Material.pec()
    air = Material.from_isotropic(name="air", epsilon=1.0)
    domain = Brick(origin=(-box / 2, -box / 2, 0.0), size=(box, box, length), material=air)
    wire1 = Brick(origin=(-s / 2 - w / 2, -w / 2, 0.0), size=(w, w, length), material=pec)
    wire2 = Brick(origin=(s / 2 - w / 2, -w / 2, 0.0), size=(w, w, length), material=pec)
    model = GeometryModel()
    model.add(Difference(domain, wire1, wire2))
    model.add(wire1)
    model.add(wire2)
    mesh = Mesh.from_geometry(
        model,
        MeshControl(min_nodes_per_wavelength=15),
        f_max=35.0e9,
    )
    return mesh.with_boundary_conditions(
        {
            "xmin": "PEC",
            "xmax": "PEC",
            "ymin": "PEC",
            "ymax": "PEC",
            "zmin": "PMC",
            "zmax": "PMC",
        }
    )


def parallel_plate_mesh() -> Mesh:
    a, b, length = 10.0e-3, 5.0e-3, 10.0e-3
    model = GeometryModel(
        boundary_conditions={
            "xmin": "PMC",
            "xmax": "PMC",
            "ymin": "PEC",
            "ymax": "PEC",
            "zmin": "PEC",
            "zmax": "PEC",
        }
    )
    model.add(
        Brick(
            origin=(-a / 2, -b / 2, 0.0),
            size=(a, b, length),
            material=Material.from_isotropic(name="air", epsilon=1.0),
        )
    )
    return Mesh.from_geometry(
        model,
        MeshControl(min_nodes_per_wavelength=15, max_cell_size=0.4e-3),
        f_max=40.0e9,
    )


def build_port(mesh, spec, f_calc):
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    dt = courant_dt(mesh.grid, "normal")
    return build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=f_calc)


def port_cutoffs_ghz(op) -> list[float]:
    return [dm.mode.omega_c / (2.0 * math.pi * 1e9) for dm in op.discrete_modes]


# ----------------------------------------------------------------------
# Leg A — coax cut-off convergence
# ----------------------------------------------------------------------


def leg_a():
    f_te = coax_te_cutoff(1, R_I, R_A, EPS_R)
    f_tm = coax_tm_cutoff(R_I, R_A, EPS_R)
    print(f"  analytic: TE11 {f_te / 1e9:.4f} GHz   TM01 {f_tm / 1e9:.4f} GHz")
    errs_te, errs_tm, dxs = [], [], []
    for dx in (0.24e-3, 0.12e-3, 0.06e-3, 0.03e-3):
        mesh = coax_mesh(dx)
        op_te = build_port(
            mesh,
            PortSpecNumerical(name="p", plane=BoxFace.Z_MIN, n_modes=2, mode_type=ModeType.TE),
            f_calc=40.0e9,
        )
        op_tm = build_port(
            mesh,
            PortSpecNumerical(name="p", plane=BoxFace.Z_MIN, n_modes=1, mode_type=ModeType.TM),
            f_calc=60.0e9,
        )
        te_pair = port_cutoffs_ghz(op_te)
        tm1 = port_cutoffs_ghz(op_tm)[0]
        err_te = (np.mean(te_pair) * 1e9 - f_te) / f_te
        err_tm = (tm1 * 1e9 - f_tm) / f_tm
        split = abs(te_pair[0] - te_pair[1]) / np.mean(te_pair)
        print(
            f"    dx {dx * 1e3:5.2f} mm  ({mesh.Nx}x{mesh.Ny})  "
            f"TE11 {np.mean(te_pair):8.4f} GHz "
            f"(err {err_te:+.2e}, pair split {split:.1e})   "
            f"TM01 {tm1:8.4f} GHz (err {err_tm:+.2e})"
        )
        errs_te.append(abs(err_te))
        errs_tm.append(abs(err_tm))
        dxs.append(dx)
    for name, errs in (("TE11", errs_te), ("TM01", errs_tm)):
        orders = [
            math.log(errs[i] / errs[i + 1]) / math.log(dxs[i] / dxs[i + 1])
            for i in range(len(errs) - 1)
        ]
        print(f"    {name} observed convergence orders: " + ", ".join(f"{o:.2f}" for o in orders))


# ----------------------------------------------------------------------
# Leg B — PEC/PMC parallel-plate ladder (after WP-U0 stage 2)
# ----------------------------------------------------------------------


def leg_b():
    a, b = 10.0e-3, 5.0e-3
    mesh = parallel_plate_mesh()

    def f_mn(m, n):
        return math.hypot(m * C0 / (2 * a), n * C0 / (2 * b)) / 1e9

    for mode_type, f_calc, ladder in (
        # PMC x-walls: Hz Dirichlet -> TE needs m >= 1.
        (ModeType.TE, 25.0e9, sorted([f_mn(1, 0), f_mn(2, 0), f_mn(1, 1)])),
        # PEC plates: Ez Dirichlet in y -> TM needs n >= 1.
        (ModeType.TM, 40.0e9, sorted([f_mn(0, 1), f_mn(1, 1)])),
    ):
        op = build_port(
            mesh,
            PortSpecNumerical(
                name="p", plane=BoxFace.Z_MIN, n_modes=len(ladder), mode_type=mode_type
            ),
            f_calc=f_calc,
        )
        got = sorted(port_cutoffs_ghz(op))
        print(
            f"    {mode_type.value}: analytic ladder [GHz]: "
            + ", ".join(f"{f:.3f}" for f in ladder)
        )
        print(
            f"    {mode_type.value}: measured        [GHz]: "
            + ", ".join(f"{f:.3f}" for f in got)
            + "   (rel. err: "
            + ", ".join(f"{(g - r) / r:+.1e}" for g, r in zip(got, ladder))
            + ")"
        )


# ----------------------------------------------------------------------
# Leg C — TEM x TE/TM cross-orthogonality through the production
#         projections
# ----------------------------------------------------------------------


def _mode_fields(op, n_e: int, n_h: int):
    """Flat E/H arrays carrying each discrete mode's plane profile."""
    fields = []
    pl = op.plane
    for dm in op.discrete_modes:
        e = np.zeros(n_e)
        e[pl.e_u_indices] = dm.e_u_profile
        e[pl.e_v_indices] = dm.e_v_profile
        h = np.zeros(n_h)
        h[pl.h_u_indices] = dm.h_u_profile
        h[pl.h_v_indices] = dm.h_v_profile
        fields.append((e, h))
    return fields


def cross_orthogonality(name, mesh, op_a, op_b):
    """Max normalised V/I crosstalk between the families of two ops."""
    n_e = build_M_eps(mesh).size
    n_h = build_M_mu(mesh).size
    worst_v = worst_i = 0.0
    for op_x, op_y in ((op_a, op_b), (op_b, op_a)):
        for j, (e, h) in enumerate(_mode_fields(op_y, n_e, n_h)):
            v_cross = op_x.project_V(e)
            v_self = op_y.project_V(e)[j]
            i_cross = op_x.project_I(h)
            i_self = op_y.project_I(h)[j]
            worst_v = max(worst_v, float(np.max(np.abs(v_cross))) / abs(v_self))
            worst_i = max(worst_i, float(np.max(np.abs(i_cross))) / abs(i_self))
    print(f"    {name:36s} max |V_cross|/V_self {worst_v:.2e}   max |I_cross|/I_self {worst_i:.2e}")


def leg_c():
    mesh = coax_mesh(0.12e-3)
    op_tem = build_port(
        mesh,
        PortSpecMultiConductor(name="p", plane=BoxFace.Z_MIN, epsilon_r=EPS_R, n_modes=1),
        f_calc=10.0e9,
    )
    op_te = build_port(
        mesh,
        PortSpecNumerical(name="p", plane=BoxFace.Z_MIN, n_modes=2, mode_type=ModeType.TE),
        f_calc=40.0e9,
    )
    op_tm = build_port(
        mesh,
        PortSpecNumerical(name="p", plane=BoxFace.Z_MIN, n_modes=1, mode_type=ModeType.TM),
        f_calc=60.0e9,
    )
    cross_orthogonality("coax TEM x TE11 pair", mesh, op_tem, op_te)
    cross_orthogonality("coax TEM x TM01", mesh, op_tem, op_tm)

    mesh = two_wire_mesh()
    op_tem = build_port(
        mesh,
        PortSpecMultiConductor(name="p", plane=BoxFace.Z_MIN, epsilon_r=1.0, n_modes=2),
        f_calc=10.0e9,
    )
    op_te = build_port(
        mesh,
        PortSpecNumerical(name="p", plane=BoxFace.Z_MIN, n_modes=2, mode_type=ModeType.TE),
        f_calc=25.0e9,
    )
    cross_orthogonality("two-wire 2x TEM x 2x TE", mesh, op_tem, op_te)


def main() -> None:
    print("WP-U1 — curl-curl certification on conductor cross-sections")
    print("Leg A: coax TE11/TM01 cut-off convergence (conformal mesh):")
    leg_a()
    print("Leg B: PEC/PMC parallel-plate TE ladder (PMC-on-model mesh):")
    leg_b()
    print("Leg C: TEM x TE/TM cross-orthogonality (production projections):")
    leg_c()


if __name__ == "__main__":
    main()
