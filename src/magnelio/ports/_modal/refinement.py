"""Refinement-based mode-parameter extraction (Phase-2 cleanup item 3).

The 3D-grid-native mode solvers (:func:`solve_tem_laplace`,
:func:`solve_qtem_laplace`, :class:`Numerical2DModeSolver`) return
modes that are *exact* eigenvectors of the FIT-2D operator on the
user's 3D-mesh port plane.  This is the mode that drives the
:class:`PortOperatorModal` — operator-consistent, no projection
between meshes, reflection-free TF/SF coupling.

The mode's published parameters (``z_line``, ``epsilon_r``,
``v_phase``, etc.) carry the *3D-mesh discretisation* of the
geometry.  As the user's 3D mesh is finite, those parameters
typically deviate from the continuum values by the FIT
discretisation error — Cartesian-staircase O(h) without Dey-Mittra,
O(h²) with.  For engineering reports, sweeps, and comparisons
against closed-form references, the user wants the *converged*
(continuum) value of these parameters.

This module provides :func:`solve_modes_refined`: it rebuilds the
3D mesh with progressively tighter ``MeshControl`` (level k uses
``2**k``-fold finer cell-size and feature-resolution targets), runs
the same FIT-TD mode-solver pipeline on each, tracks the
convergence of the chosen target, and returns a
:class:`ModeRefinementReport` carrying the converged value plus
optional Richardson-extrapolated value and observed convergence
order.

Architectural note
------------------

This is the **transient-solver model** of refinement common in
commercial EM suites: same solver, same Cartesian topology, only
the mesh resolution varies between levels.  No separate 2D
pipeline, no projection between meshes.  The user's working 3D
mesh and the refinement levels are independent — refinement does
*not* alter the user's working operator, it produces a parallel
diagnostic report.

For comparison, frequency-domain solvers typically use an
unstructured triangular mesh on the port face, *separate* from
the 3D volume mesh, and project between them.  Magnelio's
FIT-TD architecture inherits the structured-grid simplicity: no
projection step, no associated reflection residual.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Optional, Union

from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.geo import GeometryModel
from magnelio.mesh.mesher import Mesh, MeshControl
from magnelio.ports._modal.factory import (
    PortSpecCoax,
    PortSpecMultiConductor,
    PortSpecNumerical,
    PortSpecRectWG,
    build_modal_port,
)
from magnelio.solver.stability import courant_dt

PortSpec = Union[
    PortSpecCoax,
    PortSpecRectWG,
    PortSpecNumerical,
    PortSpecMultiConductor,
]

from magnelio.constants import C0  # noqa: E402


@dataclass(frozen=True)
class LevelResult:
    """One refinement-level result inside :class:`ModeRefinementReport`.

    Attributes
    ----------
    level : int
        0-based refinement level (level 0 = ``base_control``,
        level ``k`` uses ``2**k``-fold scaled cell-size targets).
    n_cells_3d : int
        Total 3D cell count of the rebuilt mesh at this level.
    n_cells_port_plane : int
        Cross-section cell count on the port plane.
    value : float
        Target metric (e.g. ``z_line``) computed from the mode at
        this level.
    rel_change : float
        ``|value - value_prev| / |value|`` between this level and
        the previous one.  ``nan`` at level 0.
    """

    level: int
    n_cells_3d: int
    n_cells_port_plane: int
    value: float
    rel_change: float


@dataclass(frozen=True)
class ModeRefinementReport:
    """Refinement convergence history + extrapolated converged value.

    The report is *purely diagnostic*; it carries no operator-bound
    state.  The values stored here are computed on independent 3D
    meshes (each level rebuilds the mesh) and *must not* be used to
    drive the user's working :class:`PortOperatorModal`, which
    remains bound to its own 3D mesh.

    Attributes
    ----------
    target : str
        Name of the convergence target (e.g. ``"z_line"``).
    history : tuple of LevelResult
        Per-level results in refinement order (level 0 first).
    converged : bool
        ``True`` if the last ``rel_change`` fell below
        ``target_rel_err`` before ``max_levels`` was reached.
    target_rel_err : float
        The user's requested relative-error threshold.
    extrapolated_value : float or None
        Richardson-extrapolated value computed from the last two
        levels assuming O(h²) convergence:
        ``(4 · v_{h/2} - v_h) / 3``.  ``None`` if extrapolation
        was disabled or fewer than two levels were run.
    convergence_order : float or None
        Empirical convergence order from the last three levels,
        ``log2(|Δ_{k-1}| / |Δ_k|)`` with ``Δ_k = v_k - v_{k-1}``.
        Should match the expected O(h²) ≈ 2.0 for Dey-Mittra-
        refined geometries; lower values typically indicate
        Cartesian-staircase-dominated convergence (≈ 1) or a
        pathological setup.  ``None`` if fewer than three levels.
    """

    target: str
    history: tuple[LevelResult, ...]
    converged: bool
    target_rel_err: float
    extrapolated_value: Optional[float] = None
    convergence_order: Optional[float] = None

    @property
    def converged_value(self) -> float:
        """Best estimate of the converged value.

        Returns the Richardson-extrapolated value when available,
        otherwise the last level's value.
        """
        if self.extrapolated_value is not None:
            return self.extrapolated_value
        return self.history[-1].value

    @property
    def estimated_error(self) -> float:
        """Conservative relative-error estimate for ``converged_value``.

        Equal to the last ``rel_change`` in ``history``.  When
        Richardson extrapolation is used, the *actual* error of
        ``extrapolated_value`` is typically smaller than this
        estimate — it is intentionally conservative.
        """
        if len(self.history) < 2:
            return float("inf")
        return abs(self.history[-1].rel_change)

    @property
    def n_levels(self) -> int:
        return len(self.history)


_VALID_TARGETS = ("z_line", "epsilon_r", "v_phase")


def _extract_target(op, target: str) -> float:
    """Read the named scalar from the first discrete mode."""
    mode = op.discrete_modes[0].mode
    if target == "z_line":
        return float(mode.z_line)
    if target == "epsilon_r":
        return float(mode.epsilon_r)
    if target == "v_phase":
        return C0 / math.sqrt(mode.epsilon_r)
    raise ValueError(f"unknown refinement target {target!r}; valid choices are {_VALID_TARGETS!r}.")


def _scale_mesh_control(base: MeshControl, scale: int) -> MeshControl:
    """Return a MeshControl with cell-size targets refined by ``scale``.

    Scaling rules (level 0 returns ``base`` unchanged):

    - ``min_nodes_per_wavelength`` × ``scale``
    - ``min_cells_per_feature``  × ``scale``
    - ``max_cell_size``          ÷ ``scale`` (when not None)
    - ``min_cell_size``          ÷ ``scale`` (when not None)

    All other fields pass through unchanged.
    """
    if scale == 1:
        return base
    return dataclasses.replace(
        base,
        min_nodes_per_wavelength=base.min_nodes_per_wavelength * scale,
        min_cells_per_feature=base.min_cells_per_feature * scale,
        max_cell_size=(base.max_cell_size / scale if base.max_cell_size is not None else None),
        min_cell_size=(base.min_cell_size / scale if base.min_cell_size is not None else None),
    )


def solve_modes_refined(
    spec: PortSpec,
    geometry: GeometryModel,
    base_control: MeshControl,
    *,
    f_max: float,
    f_calc: Optional[float] = None,
    target: str = "z_line",
    target_rel_err: float = 1e-3,
    max_levels: int = 5,
    extrapolate: bool = True,
    verbose: bool = False,
) -> ModeRefinementReport:
    """Iteratively refine the 3D mesh and track mode-parameter convergence.

    Each level rebuilds the 3D mesh with a refined ``MeshControl``
    (``2**level``-fold tightened cell-size and feature-resolution
    targets), runs :func:`build_modal_port` to obtain the FIT-TD
    mode on that mesh, and reads the named target metric.
    Refinement stops when the relative change between successive
    levels falls below ``target_rel_err``, or after ``max_levels``
    iterations — whichever comes first.

    Parameters
    ----------
    spec : PortSpec
        Any of :class:`PortSpecCoax`, :class:`PortSpecRectWG`,
        :class:`PortSpecNumerical`, :class:`PortSpecMultiConductor`.
        Analytical specs (Coax, RectWG) yield mesh-independent
        values, so the report converges trivially after one level.
    geometry : GeometryModel
        Geometry to be re-meshed at each level.
    base_control : MeshControl
        Level-0 mesh control.  Subsequent levels scale it via
        :func:`_scale_mesh_control`.
    f_max : float
        Maximum frequency [Hz] passed to :meth:`Mesh.from_geometry`.
    f_calc : float or None, default None
        Mode-calculation frequency [Hz] used by
        :func:`build_modal_port`.  Defaults to ``f_max``.
    target : str, default "z_line"
        Convergence target.  One of ``"z_line"``, ``"epsilon_r"``,
        ``"v_phase"``.
    target_rel_err : float, default 1e-3
        Relative-error stopping threshold.
    max_levels : int, default 5
        Hard cap on refinement levels.  Level 4 typically rebuilds
        a 16× finer mesh per axis ≈ 4096× more cells than level 0
        — costs grow rapidly.
    extrapolate : bool, default True
        When ``True`` and ≥ 2 levels were run, the report carries a
        Richardson-extrapolated value assuming O(h²) convergence.
    verbose : bool, default False
        Print one line per level summarising mesh size and value.

    Returns
    -------
    ModeRefinementReport

    Notes
    -----
    The user's working 3D mesh is *not* used or modified by this
    function.  Every level builds an independent mesh from
    ``geometry`` + a scaled ``base_control``.  This function is
    expensive (each level fully rebuilds the mesh and material
    matrices); call it only when convergence diagnostics are
    actually needed.
    """
    if target not in _VALID_TARGETS:
        raise ValueError(
            f"unknown refinement target {target!r}; valid choices are {_VALID_TARGETS!r}."
        )
    if max_levels <= 0:
        raise ValueError("max_levels must be positive")
    if target_rel_err <= 0:
        raise ValueError("target_rel_err must be positive")

    f_calc_resolved = f_max if f_calc is None else f_calc

    levels: list[LevelResult] = []
    prev_value: Optional[float] = None
    converged = False

    for level in range(max_levels):
        scale = 2**level
        control = _scale_mesh_control(base_control, scale)
        mesh = Mesh.from_geometry(geometry, control, f_max=f_max)
        m_eps = build_M_eps(mesh)
        m_mu = build_M_mu(mesh)
        dt = courant_dt(mesh.grid, accuracy="normal")
        op = build_modal_port(
            spec,
            mesh,
            m_eps,
            m_mu,
            dt=dt,
            f_calc=f_calc_resolved,
        )
        value = _extract_target(op, target)

        n_3d = int(mesh.Nx * mesh.Ny * mesh.Nz)
        n_per_axis = (mesh.Nx, mesh.Ny, mesh.Nz)
        n_pp = int(n_per_axis[op.plane.face.u_axis] * n_per_axis[op.plane.face.v_axis])

        rel_change = (
            abs(value - prev_value) / abs(value)
            if prev_value is not None and value != 0.0
            else float("nan")
        )
        levels.append(
            LevelResult(
                level=level,
                n_cells_3d=n_3d,
                n_cells_port_plane=n_pp,
                value=value,
                rel_change=rel_change,
            )
        )
        if verbose:
            print(
                f"  refinement L{level}: "
                f"({mesh.Nx}x{mesh.Ny}x{mesh.Nz})  "
                f"{target}={value:.6g}  "
                f"rel_change={rel_change:.3e}"
            )

        if prev_value is not None and rel_change < target_rel_err:
            converged = True
            break
        prev_value = value

    extrap: Optional[float] = None
    if extrapolate and len(levels) >= 2:
        v_h = levels[-2].value
        v_h2 = levels[-1].value
        extrap = (4.0 * v_h2 - v_h) / 3.0

    order: Optional[float] = None
    if len(levels) >= 3:
        d1 = abs(levels[-2].value - levels[-3].value)
        d2 = abs(levels[-1].value - levels[-2].value)
        if d1 > 0.0 and d2 > 0.0:
            order = float(math.log2(d1 / d2))

    return ModeRefinementReport(
        target=target,
        history=tuple(levels),
        converged=converged,
        target_rel_err=target_rel_err,
        extrapolated_value=extrap,
        convergence_order=order,
    )
