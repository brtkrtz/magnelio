"""Round-waveguide TE11 cut-off convergence study — conformal vs staircase.

Question.  Is the ~9 % vs 3 % cut-off-error gap on a circular hollow
waveguide with ``conformal=True`` versus ``conformal=False`` (observed
once at ``n_t = 17`` and recorded in ``STATUS.md`` open construction
site #2) an O(h^2) constant-of-proportionality issue, or a structural
defect of the conformal/Dey-Mittra coupling on curved PEC boundaries?

Method.  Build the same hollow circular waveguide as
``round_wg_te_stress_test.py`` (R = 10 mm, 1.2D x 1.2D x 60 mm bbox,
PEC bbox padding around the cylinder mantle) at four transversal
resolutions n_t in {17, 25, 33, 49}.  At each (n_t, conformal) point,
build the X_MIN port operator with mode_type=TE and read the lowest
non-trivial cut-off from ``op.discrete_modes[0].mode.omega_c``.  The
operator-consistent mode lives on the same 3D ``M_eps`` slab the
FIT-TD volume operator sees; its cut-off is therefore a direct probe
of the volume operator's behaviour on the curved PEC mantle.

We log the absolute error vs the analytical Bessel-zero cut-off
``f_TE11 = c0 * 1.84118 / (2 * pi * R)`` and estimate the observed
convergence order from successive resolutions via
``p_obs = log(err_k / err_{k+1}) / log(h_k / h_{k+1})``.

Reading.  If both branches converge with p_obs ~ 2 and the conformal
branch only carries a worse constant, the workaround can stand and the
issue downgrades to "ungunstige Konstante".  If the conformal branch's
p_obs is materially below 2 (or non-monotone) while staircase shows
clean p_obs ~ 2, the conformal/DM coupling is structurally broken on
curved PEC boundaries and a pipeline fix is justified.

This script touches only the public magnelio API.  It does not run
FIT-TD; the Cut-off comes from the mode-solver output that the modal
port factory exposes via ``op.discrete_modes[k].mode.omega_c``.
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


def _build_mesh(n_t_nodes: int, conformal: bool) -> Mesh:
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
        forced_planes={"x": x_nodes, "y": y_nodes, "z": z_nodes},
    )
    return Mesh.from_geometry(model, control, f_max=F_MAX)


def _measure_te11_cutoff(n_t_nodes: int, conformal: bool) -> float | None:
    """Return the numerical TE11 cut-off in GHz, or None on solver failure."""
    try:
        mesh = _build_mesh(n_t_nodes, conformal)
        m_eps = build_M_eps(mesh)
        m_mu = build_M_mu(mesh)
        # Mode solver does not need a physically-valid dt; use a robust
        # value so the build_modal_port sigma-heuristic is well-defined.
        # On conformal=True the courant_dt may collapse (M_eps=0 edges);
        # we sidestep that by clamping a floor here for the mode-solver
        # call only.  The mode-solver eigenvalue does not depend on dt.
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
        return float(op.discrete_modes[0].mode.omega_c) / (2.0 * math.pi)
    except Exception as exc:
        print(f"    [solver failed: {type(exc).__name__}: {exc}]")
        return None


def _convergence_table(label: str, conformal: bool, n_t_list: list[int]):
    print("\n" + "=" * 76)
    print(f"{label}  (conformal={conformal})")
    print("=" * 76)
    print(f"  Analytical TE11 cut-off:  {F_TE11 / 1e9:.4f} GHz")
    print(
        f"  {'n_t':>5} {'cells/D':>9} {'h/R':>10} "
        f"{'fc_num [GHz]':>15} {'rel.err':>11} {'abs.err [GHz]':>15} "
        f"{'p_obs':>9}"
    )
    rows = []
    for n_t in n_t_list:
        h = S_BBOX / max(n_t - 1, 1)
        fc_num = _measure_te11_cutoff(n_t, conformal)
        if fc_num is None:
            rows.append((n_t, h, None, None))
            print(
                f"  {n_t:>5} {2 * R / h:>9.2f} {h / R:>10.4f} "
                f"{'--':>15} {'--':>11} {'--':>15} {'--':>9}"
            )
            continue
        err_abs = fc_num - F_TE11
        err_rel = err_abs / F_TE11
        if rows and rows[-1][2] is not None:
            n_prev, h_prev, fc_prev, _ = rows[-1]
            err_prev = fc_prev - F_TE11
            if err_prev != 0.0 and err_abs != 0.0:
                p_obs = math.log(abs(err_prev) / abs(err_abs)) / math.log(h_prev / h)
                p_str = f"{p_obs:>+9.2f}"
            else:
                p_str = f"{'n/a':>9}"
        else:
            p_str = f"{'-':>9}"
        rows.append((n_t, h, fc_num, err_abs))
        print(
            f"  {n_t:>5} {2 * R / h:>9.2f} {h / R:>10.4f} "
            f"{fc_num / 1e9:>+15.4f} {err_rel * 100:>+10.2f}% "
            f"{err_abs / 1e9:>+15.4f} {p_str}"
        )
    return rows


def main() -> None:
    print("=" * 76)
    print("Round-WG TE11 cut-off convergence: conformal=True vs False")
    print("=" * 76)
    print(
        f"  Geometry: R = {R * 1e3:.1f} mm, bbox = {S_BBOX * 1e3:.2f} mm "
        f"(transversal), L_x = {L_X * 1e3:.1f} mm"
    )
    print(f"  Mode: TE11 (degenerate), p_11_prime = {P11_PRIME}")
    print(f"  Reference: f_TE11 = {F_TE11 / 1e9:.4f} GHz")

    n_t_list = [17, 25, 33, 49]

    rows_stair = _convergence_table(
        "Staircase branch",
        conformal=False,
        n_t_list=n_t_list,
    )
    rows_conf = _convergence_table(
        "Conformal branch",
        conformal=True,
        n_t_list=n_t_list,
    )

    print("\n" + "=" * 76)
    print("Summary")
    print("=" * 76)
    print(
        f"  {'n_t':>5} {'h/R':>10} "
        f"{'fc_stair [GHz]':>17} {'fc_conf [GHz]':>17} "
        f"{'rel.err stair':>15} {'rel.err conf':>14}"
    )
    for (n_t, h, fc_s, _), (_, _, fc_c, _) in zip(rows_stair, rows_conf):
        s_str = f"{fc_s / 1e9:>+17.4f}" if fc_s is not None else f"{'--':>17}"
        c_str = f"{fc_c / 1e9:>+17.4f}" if fc_c is not None else f"{'--':>17}"
        e_s = f"{(fc_s / F_TE11 - 1) * 100:>+14.2f}%" if fc_s is not None else f"{'--':>15}"
        e_c = f"{(fc_c / F_TE11 - 1) * 100:>+13.2f}%" if fc_c is not None else f"{'--':>14}"
        print(f"  {n_t:>5} {h / R:>10.4f} {s_str} {c_str} {e_s} {e_c}")


if __name__ == "__main__":
    main()
