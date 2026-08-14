"""Diagnosis: conformal pair-inconsistency limits the straight_coax port floor.

Session-81 follow-up to WP7 / DD-052.  On *staircase* cross-sections
the co-located pair identity ``M_eps * M_mu = eps * mu * dz * dz~``
holds exactly, so the exact discrete TEM travelling wave exists and
the DD-052 launch/measurement chain is exact.  On *conformal* meshes
the E-edge sub-cell correction (eps side) and the Krietenstein H-face
correction (mu side) are computed independently from the geometry and
are NOT pairwise consistent — the identity breaks, no
frequency-independent mode profile can match the discrete wave
exactly, and the port |S11| floor is set by the profile-weighted
spread of the pair product.

Measured on the straight_coax demo cross-section (D_i = 0.41 mm,
D_a = 5 mm, eps_r = 9, 10 mm line, f_max = 10 GHz):

    case              pair spread   z_line      max |S11|   median
    conformal 19x19      3.1 %      48.12 Ohm   -32.5 dB    -35.8
    staircase 19x19      0 (exact)  44.73 Ohm   -45.3 dB    -60.6
    conformal 2x         3.7 %      49.60 Ohm   -41.2 dB    -43.5

i.e. staircase at the *same* resolution has the clean port (spread
exactly zero) but a 10 % z_line geometry error; conformal recovers
the impedance (analytic 49.97 Ohm) but pays with the port floor.
3 % impedance spread ~ reflection 0.015 ~ -36 dB, matching the
measurement.  The spread is concentrated at the inner conductor
where the 1/r field peaks; refinement improves the TD floor but not
the weighted spread itself.

Open construction site (STATUS): pair-consistent sub-cell correction —
couple the mu correction of the co-located transversal H faces to the
E-edge correction so that ``M_eps * M_mu = eps_bar * mu * dz * dz~``
holds per pair on translation-invariant sections; acceptance =
straight_coax port floor at the staircase level with conformal
z_line accuracy, without regressing the DD-051 curved-PEC benchmarks
(rotated cavity, round WG).

Run:  python validation/straight_coax_conformal_pair_diag.py
"""

from __future__ import annotations

import copy
import math

import numpy as np

from magnelio import AnalysisScatteringTD, Material, Mesh, MeshControl
from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.geo import Cylinder, Difference, GeometryModel
from magnelio.mesh import BoxFace
from magnelio.ports import PortSpecMultiConductor
from magnelio.ports._modal.factory import build_modal_port
from magnelio.solver.stability import courant_dt

D_i, D_a, EPS_R, L, F_MAX = 0.41e-3, 5.0e-3, 9.0, 10.0e-3, 10.0e9
EPS0 = 8.8541878128e-12
MU0 = 4e-7 * math.pi


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


def dispersion_spread(mesh: Mesh, tag: str) -> float:
    """Profile-weighted spread of the co-located pair product.

    Pairs the interior (k = 1) transversal E edges with the co-located
    H faces at k = 1/2 and reports
    ``M_eps * M_mu / (eps0 * mu0 * dz * dz~)`` — the effective eps_r
    per pair.  The exact discrete TEM wave requires this to be uniform
    across the profile support (spike WP7.1).
    """
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    dt = courant_dt(mesh.grid, accuracy="normal")
    spec = PortSpecMultiConductor(name="p1", plane=BoxFace.Z_MIN, epsilon_r=EPS_R, n_modes=1)
    # build_modal_port flattens the port-plane PEC mask in place.
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
    cases = [
        (dict(conformal=True), "conformal      "),
        (dict(conformal=False), "staircase      "),
        (dict(conformal=True, refine=2.0), "conformal 2x   "),
    ]
    meshes = []
    for kwargs, tag in cases:
        mesh = make_mesh(**kwargs)
        dispersion_spread(mesh, tag)
        meshes.append((mesh, tag))
    for mesh, tag in meshes:
        td_run(mesh, "TD " + tag)


if __name__ == "__main__":
    main()
