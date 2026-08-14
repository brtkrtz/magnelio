"""Spike: pair-consistent conformal mu correction (STATUS construction site 0).

On staircase cross-sections the co-located pair identity

    M_eps[edge] * M_mu[face] = eps0*mu0 * eps_r * mu_r * dz * dz~

holds exactly for the transversal (Ex, Hy) / (Ey, Hx) pairs of a
z-translation-invariant section — which is what makes the DD-052 exact
discrete TEM travelling wave (and hence a reflection-clean port launch)
possible.  On conformal meshes the identity breaks because independent
geometric estimates enter the two sides (conformal free-AREA fraction
f_A of the E dual face in eps_avg; Krietenstein free-area fraction of
the H primal face), with a measured profile-weighted spread of 3.1 %
on the straight_coax 19x19 conformal section
(validation/straight_coax_conformal_pair_diag.py).

This spike measures the session-81 coupling proposal and finds that
the full fix needs TWO ingredients:

1.  **Pair-consistent mu** — for every transversal H face whose
    co-located (same transversal position, interior plane) E edge is
    unmasked, define the face mass through the pair identity

        M_mu[face] := eps0*mu0 * eps_pair * mu_bar * dz * dz~
                      / M_eps[edge]

    (written back into ``mesh.face_material`` as an equivalent
    ``A_face_free`` so build_M_mu, the 2D mode solvers' ``m_mu_flat``
    and ``courant_dt`` all see it).  This makes the pair product
    uniform by construction (spread exactly 0) and — because
    ``1/M_mu[face]`` becomes proportional to ``M_eps[edge]`` — the Ez
    rows of ``C^T M_mu^-1 C`` turn into the *same* weighted Laplacian
    the 2D TEM solve already zeroes: all free interior Ez rows drop to
    machine precision.  The E side (and with it the conformal z_line
    accuracy) is untouched.  **Alone this does NOT move the TD floor**
    (−32.5 → −32.8 dB).

2.  **Consistent longitudinal PEC masking** — the residual wave-
    equation violation localises exactly on the *conductor-footprint
    nodes*: the 2D solve treats them as Dirichlet (their Laplace rows
    are not enforced; they carry surface charge), but the conformal
    classifier leaves the co-located longitudinal z-edges unmasked
    (their line-solid f_L is 1 — they run parallel to the contour), so
    the 3D update evolves an e_z there that the purely transversal
    port profile can neither launch nor record.  Masking those
    z-edges applies the same surface condition the mode solver already
    assumes; afterwards the lifted travelling wave satisfies ALL free
    rows to machine precision — the exact discrete wave exists again.

Measured (19x19x25 conformal coax, D_i 0.41 mm / D_a 5 mm / eps_r 9):

    variant               spread   resid t     resid Ez     max|S11|  median   z_line
    conformal baseline    3.077%   1.3e-01   1.2 / 12.7     -32.50    -35.84   48.116
    + pair-consistent mu  0.000%   9.9e-11   1e-13 / 12.7   -32.79    -35.90   48.116
    + conductor z-mask    0.000%   9.9e-11   1e-13 / --     -44.06    -61.42   48.116

    (resid Ez = free interior rows / conductor-node rows, relative to
    max |Om^2 M_eps e|.  Staircase reference at the same resolution:
    -45.3 / -60.6 dB but z_line 44.73 Ohm; analytic 49.97 Ohm.)

The site-0 acceptance criterion — straight_coax port floor at the
staircase level *with* conformal z_line accuracy — is met on this
reproducer.  |S21| dev rises slightly (0.0055 → 0.0109 dB), the same
order as the staircase run.

Dead end recorded for completeness: replacing the phi-Laplace profile
by a "discrete harmonic" transversal profile (null space of the
stacked Hz-circulation + Ez-dual-curl constraints) does NOT work — the
joint null space is empty (sigma_min ~ 0.1) because the physical TEM
profile is a discrete *gradient*, and after the mu pairing the
interior Ez rows already vanish for it; the conductor-node rows are
*legitimately* nonzero (surface charge).  The profile was never the
problem — the inconsistent longitudinal masking was.

Production notes (design decision pending):

* ``eps_pair`` is the dielectric constant of the free (non-PEC) region
  at the pair's transversal position — exact here (homogeneous
  eps_r = 9).  A production mechanism can derive it per pair by
  storing the free-area fraction f_A in ``EdgeMaterialData``
  (eps_pair = eps_avg / f_A), giving the closed form
  ``M_mu = mu0 * mu_bar * dz * dz~ * (L_primal/A_dual) * (f_L/f_A)``.
* Both spike mechanisms are direction-dependent (z-pairing) resp.
  port-detection-based (conductor footprint); the production shape —
  general classifier rule vs port-driven — and the DD-051 curved-PEC
  benchmark non-regression (rotated cavity, round-WG TE11) are the
  open acceptance items.

Historical note (session 83, DD-053): the production mechanisms
landed — ``EdgeMaterialData.f_A``, the classifier's tangential-
surface re-masking (step 6b) and the meshing-time
``couple_face_material_pairs`` (unique-ladder rule with Krietenstein
fallback).  The "conformal baseline" measured by this script
therefore now already CONTAINS the fix (max |S11| −44.1 dB); the
docstring numbers above describe the pre-DD-053 state and the
override functions below act as (near-)no-ops on top of the coupled
pipeline.  This file stays as the decision record of the diagnosis.

Run:  python validation/coax_pair_consistent_mu_spike.py
"""

from __future__ import annotations

import copy
import math

import numpy as np

from magnelio import AnalysisScatteringTD, Material, Mesh, MeshControl
from magnelio._operators.curl import build_curl_matrix
from magnelio._operators.material_matrices import (
    EPS0,
    MU0,
    _build_avg_d,
    build_M_eps,
    build_M_mu,
)
from magnelio.geo import Cylinder, Difference, GeometryModel
from magnelio.geo._subcell import _build_A_face_H, _build_L_dual_H
from magnelio.mesh import BoxFace
from magnelio.ports import PortSpecMultiConductor
from magnelio.ports._modal.factory import build_modal_port
from magnelio.solver.stability import courant_dt

D_i, D_a, EPS_R, L, F_MAX = 0.41e-3, 5.0e-3, 9.0, 10.0e-3, 10.0e9


def build_model() -> GeometryModel:
    pec = Material.pec()
    diel = Material.from_isotropic(name="dielectric", epsilon=EPS_R)
    out_cyl = Cylinder(origin=(0, 0, 0), radius=D_a / 2, height=L, axis="z", material=diel)
    in_cyl = Cylinder(origin=(0, 0, 0), radius=D_i / 2, height=L, axis="z", material=pec)
    model = GeometryModel(background=pec)
    model.add(Difference(out_cyl, in_cyl))
    model.add(in_cyl)
    return model


def make_mesh(*, conformal: bool = True, refine: float = 1.0) -> Mesh:
    control = MeshControl(
        min_nodes_per_wavelength=int(8 * refine),
        min_cells_per_feature=max(3, int(3 * refine)),
        growth_factor=1.4,
        conformal=conformal,
        max_cell_size=0.4e-3 / refine,
        min_cell_size=50e-6 / refine,
        min_feature_gap=20e-6,
    )
    return Mesh.from_geometry(build_model(), control, f_max=F_MAX)


def apply_pair_consistent_mu(mesh: Mesh) -> None:
    """Write the pair-consistent transversal M_mu into mesh.face_material.

    For each transversal H face (Hy at (i+1/2, j, k+1/2) paired with the
    Ex edge at (i+1/2, j, kp); Hx at (i, j+1/2, k+1/2) paired with the
    Ey edge at (i, j+1/2, kp); kp = clip(k, 1, Nz-1) so the pairing
    plane is always interior and consistent with the port-plane
    flatten), the face mass is set through the pair identity with
    eps_pair = EPS_R (exact on this homogeneous section).

    The override is encoded as an equivalent ``A_face_free`` with
    ``L_dual_free = L_dual`` and category 2, so ``build_M_mu``
    reproduces it verbatim and ``compute_min_effective_mu`` scales the
    time step accordingly.
    """
    grid = mesh.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    em, fm = mesh.edge_material, mesh.face_material
    m_eps = build_M_eps(mesh)
    m_mu_old = build_M_mu(mesh)

    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Hx = (Nx + 1) * Ny * Nz

    A_face = _build_A_face_H(grid)
    L_dual = _build_L_dual_H(grid)
    dz_avg = _build_avg_d(grid.dz, Nz)
    kp = np.clip(np.arange(Nz), 1, Nz - 1)

    plans = [
        ("Hy<-Ex", n_Hx, (Nx, Ny + 1, Nz), 0, (Nx, Ny + 1, Nz + 1)),
        ("Hx<-Ey", 0, (Nx + 1, Ny, Nz), n_Ex, (Nx + 1, Ny, Nz + 1)),
    ]
    n_changed = 0
    factor_lo, factor_hi = math.inf, -math.inf
    z_jitter = 0.0
    for _tag, f_off, f_shape, e_off, e_shape in plans:
        n_f = int(np.prod(f_shape))
        n_e = int(np.prod(e_shape))
        cat_e = em.category[e_off : e_off + n_e].reshape(e_shape)
        me = m_eps[e_off : e_off + n_e].reshape(e_shape)
        mu_f = fm.mu_avg[f_off : f_off + n_f].reshape(f_shape)
        mu_f = np.where(np.isnan(mu_f), 1.0, mu_f)

        paired_cat = cat_e[:, :, kp]
        paired_me = me[:, :, kp]
        target = (
            EPS0
            * MU0
            * EPS_R
            * mu_f
            * grid.dz[None, None, :]
            * dz_avg[kp][None, None, :]
            / paired_me
        )
        # Re-couple EVERY pair with an unmasked E edge.  Bulk pairs
        # reproduce their bulk value exactly; this also covers
        # enlarged-cell DONOR edges (their M_eps carries the borrowed
        # neighbour mass on top of an otherwise bulk edge, which a
        # cat-2-only mask would miss — measured as a residual 0.49 %
        # spread with range up to 9.48).
        apply = paired_cat != 3

        # z-translation-invariance check: on this section the paired
        # M_eps must be k-independent per (i, j); tessellation jitter
        # here directly limits the achievable pair spread.
        interior = paired_me[:, :, 1 : Nz - 1] if Nz > 2 else paired_me
        span = interior.max(axis=2) - interior.min(axis=2)
        with np.errstate(invalid="ignore"):
            jit = span / interior.mean(axis=2)
        sel = apply[:, :, 1 : Nz - 1].any(axis=2) if Nz > 2 else apply.any(axis=2)
        if sel.any():
            z_jitter = max(z_jitter, float(np.nanmax(jit[sel])))

        flat = f_off + np.nonzero(apply.ravel())[0]
        tgt = target.ravel()[apply.ravel()]
        mu_flat = mu_f.ravel()[apply.ravel()]
        a_ff = tgt * L_dual[flat] / (MU0 * mu_flat)
        # Mirror of the build_M_mu 1 % floor: a pathological shrink
        # (paired edge nearly gone) keeps the Krietenstein value.
        ok = a_ff > 0.011 * A_face[flat]
        flat, tgt, mu_flat, a_ff = flat[ok], tgt[ok], mu_flat[ok], a_ff[ok]

        factor = tgt / m_mu_old[flat]
        changed = np.abs(factor - 1.0) > 1e-12
        n_changed += int(changed.sum())
        if changed.any():
            factor_lo = min(factor_lo, float(factor[changed].min()))
            factor_hi = max(factor_hi, float(factor[changed].max()))

        fm.category[flat] = 2
        fm.mu_avg[flat] = mu_flat
        fm.A_face_free[flat] = a_ff
        fm.L_dual_free[flat] = L_dual[flat]

    print(
        f"[override] {n_changed} transversal H faces re-coupled, "
        f"M_mu factor range [{factor_lo:.4f}, {factor_hi:.4f}], "
        f"paired-M_eps z-jitter {z_jitter:.2e}"
    )


def conductor_node_ij(mesh: Mesh) -> np.ndarray:
    """Flat (i*(Ny+1)+j) node footprint of the detected conductor groups.

    Mirrors the factory path (port-plane PEC-mask flatten, then
    auto-detection) on the Z_MIN plane; for the z-translation-invariant
    coax this footprint is the same at every k.
    """
    from magnelio._operators.material_matrices import (  # noqa: PLC0415
        flatten_port_plane_pec_mask,
    )
    from magnelio.ports._modal import PortPlane  # noqa: PLC0415
    from magnelio.ports._modal.auto_conductors import (  # noqa: PLC0415
        extract_conductor_groups_from_mesh,
    )

    m = copy.deepcopy(mesh)
    object.__setattr__(
        m,
        "pec_mask_edges",
        flatten_port_plane_pec_mask(m.pec_mask_edges, m, BoxFace.Z_MIN),
    )
    plane = PortPlane.from_mesh(BoxFace.Z_MIN, m)
    groups = extract_conductor_groups_from_mesh(plane, m)
    return np.unique(np.concatenate(groups))


def apply_conductor_z_mask(mesh: Mesh) -> None:
    """Mask the longitudinal E edges over the conductor node footprint.

    The 2D mode solve treats the detected conductor nodes as Dirichlet
    (their Laplace rows are not enforced — they carry surface charge),
    but the conformal classifier leaves the co-located z-edges
    *unmasked* in 3D (their line-solid fraction is 1 — they run
    parallel to the contour), so the FIT update evolves an e_z there
    that the purely transversal port profile can neither launch nor
    record.  Masking those z-edges applies the same surface condition
    the 2D solve already assumes, making the two systems consistent.
    """
    Nz = mesh.Nz
    nodes = conductor_node_ij(mesh)
    idx = (nodes[:, None] * Nz + np.arange(Nz)[None, :]).ravel()
    pec = mesh.pec_mask_edges
    newly = int((~pec[2, idx]).sum())
    pec[2, idx] = True
    print(
        f"[z-mask] conductor footprint {nodes.size} nodes, "
        f"{newly} additional longitudinal edges masked"
    )


def wave_residual_probe(mesh: Mesh, tag: str) -> None:
    """Feed the lifted travelling wave through the 3D operator.

    The exact leapfrog travelling wave satisfies
    ``C^T (1/M_mu) C e = Om^2 M_eps e`` on all free edges (Om = the
    discrete frequency, beta the discrete wavenumber).  Lifting the 2D
    mode profile with ``zeta^k`` and evaluating the residual row-wise
    separates the exactness conditions:

    * transversal rows (Ex/Ey) — the per-pair dispersion products
      (what the mu override fixes),
    * longitudinal rows (Ez), split by conductor-footprint membership —
      free interior rows measure the metric consistency of the profile
      (the mu pairing turns them into the 2D solve's own Laplacian);
      conductor-node rows carry the physical surface charge and are
      only satisfiable by masking those edges.
    """
    grid = mesh.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    dt = courant_dt(grid, accuracy="normal")
    spec = PortSpecMultiConductor(name="p1", plane=BoxFace.Z_MIN, epsilon_r=EPS_R, n_modes=1)
    op = build_modal_port(spec, copy.deepcopy(mesh), m_eps.copy(), m_mu, dt=dt, f_calc=F_MAX)
    plane = op.plane
    dm = op.discrete_modes[0]

    C = build_curl_matrix(grid)
    edges0 = np.concatenate([plane.e_u_indices, plane.e_v_indices])
    e_t = np.concatenate([np.asarray(dm.e_u_profile), np.asarray(dm.e_v_profile)])

    f0 = 0.5 * F_MAX
    omega = 2 * math.pi * f0
    Om = 2 * math.sin(omega * dt / 2) / dt
    dz = grid.dz[0]
    c0 = 299_792_458.0
    beta = 2 / dz * math.asin(Om * dz * math.sqrt(EPS_R) / (2 * c0))
    zeta = np.exp(-1j * beta * dz)

    e_c = np.zeros(C.shape[1], dtype=complex)
    for k in range(Nz + 1):
        e_c[edges0 + k] = e_t * zeta**k
    r = C.T @ ((C @ e_c) / m_mu) - Om**2 * (m_eps * e_c)
    scale = float(np.abs(Om**2 * (m_eps * e_c)).max())

    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    n_Ez = (Nx + 1) * (Ny + 1) * Nz
    pec = mesh.pec_mask_edges
    free = ~np.concatenate([pec[0, :n_Ex], pec[1, :n_Ey], pec[2, :n_Ez]])

    k_ex = np.arange(n_Ex) % (Nz + 1)
    k_ey = np.arange(n_Ey) % (Nz + 1)
    k_ez = np.arange(n_Ez) % Nz
    interior_t = (
        np.concatenate(
            [
                (k_ex >= 3) & (k_ex <= Nz - 3),
                (k_ey >= 3) & (k_ey <= Nz - 3),
                np.zeros(n_Ez, dtype=bool),
            ]
        )
        & free
    )
    interior_z = (
        np.concatenate(
            [
                np.zeros(n_Ex, dtype=bool),
                np.zeros(n_Ey, dtype=bool),
                (k_ez >= 3) & (k_ez <= Nz - 4),
            ]
        )
        & free
    )

    r_t = np.abs(r[interior_t]).max() / scale

    cond = conductor_node_ij(mesh)
    idx_z = np.nonzero(interior_z)[0]
    node2d = (idx_z - n_Ex - n_Ey) // Nz
    on_cond = np.isin(node2d, cond)
    r_all = np.abs(r[idx_z])
    r_z_cond = r_all[on_cond].max() / scale if on_cond.any() else 0.0
    r_z_free = r_all[~on_cond].max() / scale if (~on_cond).any() else 0.0
    print(
        f"[{tag}] wave residual @ {f0 / 1e9:.0f} GHz: "
        f"transversal rows {r_t:.3e}, Ez rows conductor-nodes "
        f"{r_z_cond:.3e} / elsewhere {r_z_free:.3e} "
        f"(rel. to max |Om^2 M_eps e|)"
    )


def dispersion_spread(mesh: Mesh, tag: str) -> float:
    """Profile-weighted spread of the co-located pair product.

    Identical metric to straight_coax_conformal_pair_diag.py: pairs the
    interior (k = 1) transversal E edges with the co-located H faces at
    k = 1/2 and reports the effective eps_r per pair.
    """
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    dt = courant_dt(mesh.grid, accuracy="normal")
    spec = PortSpecMultiConductor(name="p1", plane=BoxFace.Z_MIN, epsilon_r=EPS_R, n_modes=1)
    op = build_modal_port(spec, copy.deepcopy(mesh), m_eps.copy(), m_mu, dt=dt, f_calc=F_MAX)
    plane = op.plane
    dm = op.discrete_modes[0]

    dz = mesh.grid.dz[0]
    dz_tilde = 0.5 * (mesh.grid.dz[0] + mesh.grid.dz[1])

    prod_u = (m_eps[plane.e_u_indices_interior] * m_mu[plane.h_v_indices]) / (
        EPS0 * MU0 * dz * dz_tilde
    )
    prod_v = (m_eps[plane.e_v_indices_interior] * m_mu[plane.h_u_indices]) / (
        EPS0 * MU0 * dz * dz_tilde
    )

    w_u = m_eps[plane.e_u_indices_interior] * dm.e_u_profile**2
    w_v = m_eps[plane.e_v_indices_interior] * dm.e_v_profile**2
    prod = np.concatenate([prod_u, prod_v])
    w = np.concatenate([w_u, w_v])
    mask = w > 1e-6 * w.max()
    p, wm = prod[mask], w[mask]
    mean = float(np.average(p, weights=wm))
    std = math.sqrt(float(np.average((p - mean) ** 2, weights=wm)))
    print(
        f"[{tag}] mesh {mesh.Nx}x{mesh.Ny}x{mesh.Nz}: eps_r_eff per "
        f"pair (profile-weighted): mean {mean:.4f}, rel spread "
        f"{std / mean:.3%}, range [{p.min():.3f}, {p.max():.3f}]; "
        f"z_line = {op.port_report.z_line_num:.3f} Ohm"
    )
    return std / mean


def td_run(mesh: Mesh, tag: str) -> None:
    specs = [
        PortSpecMultiConductor(name="port1", plane=BoxFace.Z_MIN, epsilon_r=EPS_R, n_modes=1),
        PortSpecMultiConductor(name="port2", plane=BoxFace.Z_MAX, epsilon_r=EPS_R, n_modes=1),
    ]
    analysis = AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions({}), ports=specs, f_max=F_MAX, verbose=False
    )
    f_axis = np.linspace(F_MAX / 40, F_MAX, 81)
    result = analysis.run(f_axis=f_axis, excited=["port1"])
    s11 = 20 * np.log10(np.abs(result.S("port1", "port1")) + 1e-30)
    s21 = 20 * np.log10(np.abs(result.S("port2", "port1")) + 1e-30)
    print(
        f"[{tag}] max|S11| {s11.max():7.2f} dB, median "
        f"{np.median(s11):7.2f}; |S21| dev {np.max(np.abs(s21)):.4f} dB"
    )


def main() -> None:
    mesh = make_mesh(conformal=True)
    baseline = copy.deepcopy(mesh)

    dispersion_spread(mesh, "conformal baseline  ")
    wave_residual_probe(mesh, "conformal baseline  ")
    apply_pair_consistent_mu(mesh)
    dispersion_spread(mesh, "pair-consistent mu  ")
    wave_residual_probe(mesh, "pair-consistent mu  ")
    apply_conductor_z_mask(mesh)
    dispersion_spread(mesh, "paired mu + z-mask  ")
    wave_residual_probe(mesh, "paired mu + z-mask  ")

    td_run(baseline, "TD conformal baseline")
    td_run(mesh, "TD paired mu + z-mask")


if __name__ == "__main__":
    main()
