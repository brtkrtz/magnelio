"""Quick test: does eta=0 lift the Round-WG conformal-branch p_obs to >= 1.8?

The session-62 / DD-051 diagnostic
(``round_wg_subcell_diagnostic.py``) confirmed the OCC pipelines
(``compute_face_material_areas`` for ε̄, ``compute_edge_pec_fractions``
for f_L) are bit-identical between the variant-B run and a
1000×-finer-tessellation reference.  The hypothesis "tessellation
inconsistency is the convergence defect" is therefore *empirically
falsified*.

The same diagnostic showed that 15 / 80 cat-2 edges lie within
[0.35, 0.45] of the enlarged-cell threshold ``eta = 0.4``.  Those
edges sit at a *discontinuous* cat-3 (masked, borrowed-out) vs cat-2
(active, ``M_eps ∝ 1 / f_L``) classifier boundary.  Which edges fall
where is mesh-refinement-dependent and changes discretely between
``n_t = 25`` and ``n_t = 33``, exactly where the convergence study
saw the rel.err *grow* from −3.83 % to −4.36 %.

This script tests the "threshold discontinuity is the defect"
hypothesis by running the Round-WG TE11 cut-off measurement at two
resolutions (``n_t = 17``, ``n_t = 25``) with ``dey_mittra_eta = 0``
(no threshold ⇒ all curved-PEC sub-cell edges land in cat 2,
continuously in f_L).

Reading.
* If err(eta=0) ≪ err(eta=0.4) at both resolutions and the gap
  shrinks monotonically with refinement, the threshold is the
  defect and the fix is to give the eigensolver an
  ``eta = 0``-built M_eps while keeping ``eta = 0.4`` for the
  FIT-TD CFL bound.
* If err(eta=0) ≈ err(eta=0.4), the defect lies elsewhere — return
  to the OCC-consolidation drawing board.
"""

from __future__ import annotations

import math

import numpy as np

from magnelio import Material, Mesh, MeshControl
from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.geo import Brick, Cylinder, Difference, GeometryModel
from magnelio.ports._modal import (
    BoxFace,
    ModeType,
    PortSpecNumerical,
    build_modal_port,
)
from magnelio.solver.stability import courant_dt

R = 10.0e-3
D = 2.0 * R
S_BBOX = 1.2 * D
L_X = 60.0e-3
F_MAX = 14.0e9
F_CALC = 13.0e9
N_MODES = 2
C0 = 299_792_458.0
P11_PRIME = 1.84118
F_TE11 = C0 * P11_PRIME / (2.0 * math.pi * R)


def _geometry() -> GeometryModel:
    pec = Material.pec()
    vacuum = Material.air()
    bbox = Brick(
        origin=(0.0, -S_BBOX / 2, -S_BBOX / 2),
        size=(L_X, S_BBOX, S_BBOX),
        material=pec,
    )
    inner = Cylinder(
        origin=(0.0, 0.0, 0.0),
        radius=R,
        height=L_X,
        axis="x",
        material=vacuum,
    )
    model = GeometryModel()
    model.add(Difference(bbox, inner))
    model.add(inner)
    return model


def _build_mesh(n_t_nodes: int, conformal: bool, eta: float) -> Mesh:
    model = _geometry()
    y_nodes = np.linspace(-S_BBOX / 2, S_BBOX / 2, n_t_nodes).tolist()
    z_nodes = np.linspace(-S_BBOX / 2, S_BBOX / 2, n_t_nodes).tolist()
    x_nodes = np.linspace(0.0, L_X, 31).tolist()
    control = MeshControl(
        min_nodes_per_wavelength=8,
        min_cells_per_feature=0,
        growth_factor=1.5,
        max_cell_size=4.0 * S_BBOX / max(n_t_nodes - 1, 1),
        conformal=conformal,
        dey_mittra_eta=eta,
        forced_planes={"x": x_nodes, "y": y_nodes, "z": z_nodes},
    )
    return Mesh.from_geometry(model, control, f_max=F_MAX)


def _measure(n_t_nodes: int, conformal: bool, eta: float) -> float | None:
    """Return the numerical TE11 cut-off in GHz, or None on solver failure."""
    try:
        mesh = _build_mesh(n_t_nodes, conformal, eta)
        m_eps = build_M_eps(mesh)
        m_mu = build_M_mu(mesh)
        try:
            dt = courant_dt(mesh.grid, accuracy="normal")
        except Exception:
            dt = 1.0e-12
        spec = PortSpecNumerical(
            name="port1",
            plane=BoxFace.X_MIN,
            n_modes=N_MODES,
            mode_type=ModeType.TE,
        )
        op = build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=F_CALC)
        return float(op.discrete_modes[0].mode.omega_c) / (2.0 * math.pi) / 1e9
    except Exception as exc:
        print(f"    FAILURE at n_t={n_t_nodes}, eta={eta}: {type(exc).__name__}: {exc}")
        return None


def main():
    print("=" * 76)
    print("Round-WG TE11 cut-off — eta=0 vs eta=0.4 hypothesis test")
    print(f"  Analytical f_TE11 = {F_TE11 / 1e9:.4f} GHz")
    print("=" * 76)

    rows = []
    for n_t in (17, 25, 33):
        for eta in (0.4, 1e-6):
            f_c = _measure(n_t, conformal=True, eta=eta)
            err = (f_c - F_TE11 / 1e9) / (F_TE11 / 1e9) * 100 if f_c is not None else None
            rows.append((n_t, eta, f_c, err))
            tag = "(default)" if eta == 0.4 else "(eta→0, no threshold)"
            err_str = f"{err:+.2f} %" if err is not None else "FAIL"
            f_c_str = f"{f_c:.4f}" if f_c is not None else "FAIL"
            print(
                f"  n_t={n_t:3d}  eta={eta:.2f} {tag:<22s}"
                f"  f_c = {f_c_str} GHz   rel.err = {err_str}"
            )

    print()
    print("─" * 76)
    print(" Summary table")
    print("─" * 76)
    print(f"  {'n_t':>4} {'rel.err eta=0.4':>20} {'rel.err eta=0':>20} {'Δerr':>15}")
    by_n = {n_t: {} for n_t in (17, 25, 33)}
    for n_t, eta, f_c, err in rows:
        by_n[n_t][eta] = err
    for n_t in (17, 25, 33):
        e04 = by_n[n_t].get(0.4)
        e00 = by_n[n_t].get(1e-6)
        s04 = f"{e04:+.2f} %" if e04 is not None else "FAIL"
        s00 = f"{e00:+.2f} %" if e00 is not None else "FAIL"
        if e04 is not None and e00 is not None:
            ds = f"{e00 - e04:+.2f} %"
        else:
            ds = "—"
        print(f"  {n_t:>4} {s04:>20} {s00:>20} {ds:>15}")


if __name__ == "__main__":
    main()
