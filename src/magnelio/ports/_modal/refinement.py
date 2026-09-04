"""Port-plane mesh refinement: the converged mode parameters (DD-244).

The port's 2D mode problem is, by design, the port-plane slice of the
3D FIT operator — which is what makes the discrete mode an exact
eigenvector of the grid the run marches on and the port reflection-
free.  The price is that the mode's parameters (``z_line``,
``epsilon_eff``, the cut-offs) carry the discretisation of the *user's*
3D mesh, and converging them by refining that mesh costs ``8×`` cells
per halving.

:func:`refine_port_modes` converges them on a **port slab** instead: a
thin slice of the model behind the port face, cut with the mesher's
own domain clip (a symmetry declaration at a grid node a few cells
in), meshed by the user's own mesh control as level 0 — which reproduces
the user's port-plane grid — and with every tangential cell split
``2^k`` ways at level ``k`` (``MeshControl.subdivide``).  Every level therefore
costs ``4×`` the cells of the previous one on a slab a handful of
cells deep, the level-0 value reproduces what ``solve_ports()`` prints
for the user's own mesh, and the ladder tells how far that number
stands from the converged cross-section.  The geometry is not
touched: the slab holds the same shapes, materials and lateral
boundary conditions as the model, and the port keeps its equidistant
buffer cells.

Richardson extrapolation uses the *observed* convergence order once
three levels exist (the conformal material matrices converge at
second order, a staircased feature at first) and assumes second order
before.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from magnelio.boundaries.boundary_conditions import BoundaryConditions, symmetry_entries
from magnelio.geo import GeometryModel
from magnelio.mesh.mesher import Mesh, MeshControl
from magnelio.ports.declarative import normalize_box_face

_VALID_TARGETS = ("auto", "z_line", "epsilon_eff", "f_cutoff")
#: What a mode family has to converge: a TEM/quasi-TEM line its
#: impedance, a TE/TM mode its cut-off (DD-252).
_AUTO_TARGET = {True: "z_line", False: "f_cutoff"}


@dataclass(frozen=True)
class RefinementLevel:
    """One rung of the refinement ladder.

    Attributes
    ----------
    level : int
        0 is the user's port-plane grid; level ``k`` bisects it ``k``
        times along both tangential axes.
    n_cells_port_plane : int
        Cross-section cell count on the port plane.
    n_cells_3d : int
        Cell count of the slab mesh.
    value : float
        The target quantity at this level (full-model value).
    rel_change : float
        ``|value − value_prev| / |value|``; ``nan`` at level 0.
    """

    level: int
    n_cells_port_plane: int
    n_cells_3d: int
    value: float
    rel_change: float


@dataclass(frozen=True)
class PortRefinementReport:
    """Convergence of one port's mode parameter under port-plane refinement.

    Attributes
    ----------
    port_name : str
    target : str
        ``"z_line"``, ``"epsilon_eff"`` or ``"f_cutoff"`` of mode
        ``mode`` -- the resolved quantity, also when ``"auto"`` was
        asked for.
    mode : int
    levels : tuple of RefinementLevel
        The ladder, level 0 first.
    tol : float
        The relative change the ladder was asked to reach.
    converged : bool
        Whether the last change fell below ``tol`` before the level
        cap.
    extrapolated : float or None
        Richardson estimate from the last two levels (``None`` with a
        single level).
    order : float or None
        Observed convergence order from the last three levels
        (``None`` with fewer).
    reports : tuple of PortReport
        The port report of every level, for the modes themselves.
    """

    port_name: str
    target: str
    mode: int
    levels: tuple[RefinementLevel, ...]
    tol: float
    converged: bool
    extrapolated: Optional[float] = None
    order: Optional[float] = None
    reports: tuple = dataclasses.field(default=(), repr=False)

    @property
    def value(self) -> float:
        """Best estimate: the extrapolated value, else the finest level's."""
        return self.levels[-1].value if self.extrapolated is None else self.extrapolated

    @property
    def estimated_error(self) -> float:
        """Conservative relative error of :attr:`value` — the last rung's change."""
        if len(self.levels) < 2:
            return float("inf")
        return abs(self.levels[-1].rel_change)

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    def summary(self) -> str:
        unit = {"z_line": "Ω", "epsilon_eff": "", "f_cutoff": "Hz"}[self.target]
        lines = [
            f"Port {self.port_name!r} — {self.target} of mode {self.mode} under "
            f"port-plane refinement ({'converged' if self.converged else 'not converged'} "
            f"at tol {self.tol:.1e})"
        ]
        for lv in self.levels:
            change = "" if math.isnan(lv.rel_change) else f"  Δ {100.0 * lv.rel_change:+.3f} %"
            lines.append(
                f"  level {lv.level}: {lv.n_cells_port_plane:7d} plane cells  "
                f"{lv.value:.6g} {unit}{change}"
            )
        if self.extrapolated is not None:
            order = "" if self.order is None else f", observed order {self.order:.2f}"
            lines.append(f"  extrapolated: {self.extrapolated:.6g} {unit}{order}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _declared_port(mesh: Mesh, port: str):
    for p in getattr(mesh, "ports", ()):
        if getattr(p, "name", None) == port:
            return p
    names = [getattr(p, "name", "?") for p in getattr(mesh, "ports", ())]
    raise ValueError(f"port {port!r} is not declared on the mesh; declared ports: {names}")


def _slab_boundary_conditions(model: GeometryModel, far_face: str, position: float):
    """The model's closure with ``far_face`` replaced by a clipping symmetry plane."""
    bc = model.boundary_conditions
    out: dict = {}
    sym = symmetry_entries(bc)
    for face, kind in bc.to_dict().items():
        if face in sym:
            out[face] = (
                (f"Symmetry{kind}", sym[face]) if sym[face] is not None else f"ForceSymmetry{kind}"
            )
        else:
            out[face] = kind
    axis = far_face[0]
    for face in list(out):
        # One symmetry plane per axis: a symmetry on the port's own axis
        # can only be the port face itself, which is not a symmetry.
        if face[0] == axis and face != far_face and face in sym:
            raise ValueError(
                f"the model declares a symmetry plane on {face!r}, the axis of port face "
                f"{far_face[0]}: the port slab cannot be cut on that axis."
            )
    out[far_face] = ("SymmetryPMC", float(position))
    return BoundaryConditions(**out, cpml_thickness_cells=int(bc.cpml_thickness_cells))


def _resolve_target(report, target: str, mode: int) -> str:
    """The quantity to converge: as asked, or the one the mode family defines."""
    if target != "auto":
        return target
    if mode >= len(report.modes):
        raise ValueError(
            f"port {report.name!r} solves {len(report.modes)} mode(s); mode {mode} does not exist"
        )
    return _AUTO_TARGET[report.modes[mode].z_line is not None]


def _extract(report, target: str, mode: int) -> float:
    if mode >= len(report.modes):
        raise ValueError(
            f"port {report.name!r} solves {len(report.modes)} mode(s); mode {mode} does not exist"
        )
    m = report.modes[mode]
    if target == "z_line":
        if m.z_line is None:
            raise ValueError(
                f"mode {mode} of port {report.name!r} has no line impedance: it is a "
                f"TE/TM mode, which the ladder converges by its cut-off -- pass "
                f"target='f_cutoff', or leave target='auto'."
            )
        return float(m.z_line)
    if target == "epsilon_eff":
        if m.epsilon_eff is None:
            raise ValueError(
                f"mode {mode} of port {report.name!r} has no ε_eff: it is a TE/TM "
                f"mode -- pass target='f_cutoff', or leave target='auto'."
            )
        return float(m.epsilon_eff)
    if target == "f_cutoff":
        return float(m.f_cutoff)
    raise ValueError(f"unknown target {target!r}; valid: {_VALID_TARGETS}")


def refine_port_modes(
    model: GeometryModel,
    control: MeshControl,
    mesh: Mesh,
    port: str,
    *,
    levels: int = 3,
    target: str = "auto",
    mode: int = 0,
    tol: float = 1e-3,
    slab_cells: int = 6,
    verbose: bool | None = None,
) -> PortRefinementReport:
    """Converge a port mode's parameter by refining the port plane alone.

    Builds a slab of the model behind the port face — the model's
    shapes, materials and lateral closure, cut ``slab_cells`` of the
    user's mesh in with the mesher's own domain clip — and meshes it
    with the user's mesh control at level 0 and every tangential cell
    split ``2^k`` ways at level ``k``.  Level 0 reproduces the port
    report of ``mesh``; the ladder converges the cross-section at
    ``4×`` the cells per level instead of the ``8×`` a 3D refinement
    costs, on a slab a few cells deep.

    Parameters
    ----------
    model : GeometryModel
        The model ``mesh`` was generated from, ports declared.
    control : MeshControl
        The mesh control ``mesh`` was generated with; every level keeps
        it and sets its ``subdivide`` on the tangential axes.
    mesh : Mesh
        The user's mesh (supplies the level-0 grid, ``f_max`` and the
        port declaration).
    port : str
        Name of a declared waveguide port.
    levels : int, default 3
        Number of ladder rungs including level 0 (at most).
    target : {"auto", "z_line", "epsilon_eff", "f_cutoff"}
        Quantity to converge.  ``"auto"`` (default) follows the mode
        family: the line impedance of a TEM or quasi-TEM mode, the
        cut-off frequency of a TE or TM mode, which has no line
        impedance.  The report names the quantity it converged.
    mode : int, default 0
        Mode index on the port.
    tol : float, default 1e-3
        Stop once the relative change between rungs falls below it.
    slab_cells : int, default 6
        Depth of the slab in cells of the user's mesh behind the port
        face (at least four: the port's equidistant buffer).
    verbose : bool, optional
        Report each level as it runs — the mesh build and port
        solve of the rung in progress, then its converged value.
        ``None`` (the default) follows
        :func:`magnelio.set_verbosity`.

    Returns
    -------
    PortRefinementReport

    Raises
    ------
    ValueError
        If the port is not declared on the mesh, the model declares a
        symmetry plane on the port's axis, or the target is not
        defined for the mode.
    """
    from magnelio.analysis.scattering_td import AnalysisScatteringTD  # noqa: PLC0415

    if target not in _VALID_TARGETS:
        raise ValueError(f"unknown target {target!r}; valid: {_VALID_TARGETS}")
    if levels < 1:
        raise ValueError("levels must be >= 1")
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if slab_cells < 4:
        raise ValueError("slab_cells must be >= 4 (the port's equidistant buffer)")
    if mesh.f_max is None:
        raise ValueError("mesh carries no f_max; generate it with Mesh.from_geometry")

    declared = _declared_port(mesh, port)
    if getattr(declared, "plane", None) is None:
        raise ValueError(f"port {port!r} is not a waveguide port (no plane)")
    face = normalize_box_face(declared.plane)
    n_axis = face.normal_axis
    axes = ("x", "y", "z")
    normal_nodes = np.asarray((mesh.grid.x, mesh.grid.y, mesh.grid.z)[n_axis], dtype=float)
    if normal_nodes.size <= slab_cells:
        raise ValueError(
            f"the mesh has only {normal_nodes.size - 1} cells along the port normal; "
            f"slab_cells={slab_cells} does not fit"
        )
    if face.is_max:
        position = float(normal_nodes[-1 - slab_cells])
        far_face = f"{axes[n_axis]}min"
    else:
        position = float(normal_nodes[slab_cells])
        far_face = f"{axes[n_axis]}max"
    tangential = [a for a in axes if a != axes[n_axis]]

    slab = GeometryModel(
        background=model.background,
        boundary_conditions=_slab_boundary_conditions(model, far_face, position),
        allow_overlaps=model.allow_overlaps,
    )
    for shape in model.shapes:
        slab.add(shape)
    slab.add_port(declared)

    from magnelio._progress import Reporter  # noqa: PLC0415

    # A rung is a full mesh build plus a port solve, so the ladder is
    # where a user waits longest with nothing to look at.  The rung is
    # announced *before* it runs, and its inner work reports through
    # the same setting rather than being silenced outright.
    rep = Reporter("refine", verbose)
    inner_verbose = rep.enabled

    rungs: list[RefinementLevel] = []
    reports = []
    prev: Optional[float] = None
    converged = False
    for k in range(levels):
        sub = dict(control.subdivide)
        for a in tangential:
            sub[a] = int(sub.get(a, 1)) * 2**k
        ctrl = dataclasses.replace(control, subdivide=sub)
        rep.note(f"level {k}/{levels - 1}: meshing and solving the port slab")
        mesh_k = Mesh.from_geometry(slab, ctrl, f_max=float(mesh.f_max), verbose=inner_verbose)
        report = AnalysisScatteringTD(mesh=mesh_k, verbose=inner_verbose).solve_ports()[port]
        target = _resolve_target(report, target, mode)
        value = _extract(report, target, mode)
        n_per_axis = (mesh_k.Nx, mesh_k.Ny, mesh_k.Nz)
        n_plane = int(np.prod([n_per_axis[i] for i in range(3) if i != n_axis]))
        rel = float("nan") if prev is None else abs(value - prev) / max(abs(value), 1e-300)
        rungs.append(
            RefinementLevel(
                level=k,
                n_cells_port_plane=n_plane,
                n_cells_3d=int(np.prod(n_per_axis)),
                value=value,
                rel_change=rel,
            )
        )
        reports.append(report)
        change = "" if prev is None else f"  Δ {100.0 * rel:+.3f} %"
        rep.note(f"level {k}: {n_plane} plane cells, {target} = {value:.6g}{change}")
        if prev is not None and rel < tol:
            converged = True
            prev = value
            break
        prev = value

    order = None
    extrapolated = None
    vals = [r.value for r in rungs]
    if len(vals) >= 3:
        d1 = vals[-2] - vals[-3]
        d2 = vals[-1] - vals[-2]
        if d1 != 0.0 and d2 != 0.0 and (d1 / d2) > 1.0:
            order = float(math.log2(abs(d1 / d2)))
    if len(vals) >= 2:
        p = 2.0 if order is None else max(order, 0.5)
        extrapolated = vals[-1] + (vals[-1] - vals[-2]) / (2.0**p - 1.0)

    rep.close()
    return PortRefinementReport(
        port_name=port,
        target=target,
        mode=mode,
        levels=tuple(rungs),
        tol=tol,
        converged=converged,
        extrapolated=extrapolated,
        order=order,
        reports=tuple(reports),
    )
