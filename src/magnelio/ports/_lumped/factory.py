"""PortSpecLumped + build_lumped_port — declarative spec + builder.

Mirror of the modal-port factory pattern: a plain dataclass holds the
user's intent (label, endpoints, internal impedance), and a builder
function resolves it onto a concrete mesh, returning a runtime operator
that implements the :class:`magnelio.ports.base.Port` protocol.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from magnelio.circuit.companion import ParallelRLC, SeriesRLC
from magnelio.mesh.indexing import edge_index_Ex, edge_index_Ey, edge_index_Ez
from magnelio.ports._lumped.operator import LumpedElementOperator, PortOperatorLumped


@dataclass
class PortSpecLumped:
    """Declarative description of a lumped discrete port / RLC element.

    Parameters
    ----------
    name : str
        Unique port identifier (used as recorder channel key).
    start, end : tuple[float, float, float]
        Endpoints in metres.  The two points must differ along exactly
        one Cartesian axis after grid snapping.
    Z0 : float
        Power-wave reference impedance [Ω] (default 50 Ω).  Without an
        ``element`` it is also the internal Thévenin impedance — the
        classic discrete port.
    element : SeriesRLC or ParallelRLC, optional
        Trapezoidal companion element replacing the pure
        resistor as the port's internal impedance: an excited port
        becomes an RLC-backed source, an unexcited one a passive lumped
        RLC load.  ``None`` (default) means ``SeriesRLC(R=Z0)`` — the
        behaviour-identical classic port.  The element instance is
        deep-copied per run, so its transient state never leaks between
        excitations.
    """

    name: str
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    Z0: float = 50.0
    element: Optional[Union[SeriesRLC, ParallelRLC]] = None


def build_lumped_port(
    spec: PortSpecLumped,
    mesh,
    m_eps: np.ndarray,
    m_mu: np.ndarray,
    dt: float,
) -> PortOperatorLumped:
    """Build a :class:`PortOperatorLumped` from *spec* and *mesh*.

    Resolves ``spec.start`` / ``spec.end`` to flat E-edge indices on
    *mesh*, computes the lossless update coefficient ``β_E = dt / M_eps``
    at those edges, and assembles a ready-to-use lumped port operator.

    Parameters
    ----------
    spec : PortSpecLumped
    mesh : Mesh
    m_eps : np.ndarray
        Diagonal of the FIT ``M_eps`` matrix on the flat E layout.
    m_mu : np.ndarray
        Diagonal of ``M_mu`` (unused for lumped ports; accepted for
        symmetry with :func:`magnelio.ports._modal.build_modal_port`).
    dt : float
        Solver time step [s].

    Returns
    -------
    PortOperatorLumped
        Runtime operator implementing :class:`magnelio.ports.base.Port`.

    Raises
    ------
    ValueError
        When ``spec.start`` and ``spec.end`` snap to nodes that do not
        differ along exactly one axis, or coincide entirely.
    """
    del m_mu  # unused — kept in signature to mirror build_modal_port

    direction, flat_indices, ijk_list, dl_list = _snap_edge_chain(
        f"PortSpecLumped {spec.name!r}",
        spec.start,
        spec.end,
        mesh,
    )

    m_eps_port = np.asarray(m_eps[flat_indices], dtype=float)
    beta_E = dt / m_eps_port

    return PortOperatorLumped(
        name=spec.name,
        Z0=spec.Z0,
        direction=direction,
        flat_edge_indices=list(flat_indices),
        ijk_list=ijk_list,
        dl_list=dl_list,
        beta_E=beta_E,
        # Fresh companion state per built operator (operators are built
        # per excitation; the user's spec instance must stay pristine).
        element=copy.deepcopy(spec.element) if spec.element is not None else None,
    )


def build_lumped_element(
    spec,
    mesh,
    m_eps: np.ndarray,
    m_mu: np.ndarray,
    dt: float,
) -> LumpedElementOperator:
    """Build a passive :class:`LumpedElementOperator` from a declarative
    :class:`magnelio.circuit.LumpedElement` (DD-123).

    Same edge-chain resolution as :func:`build_lumped_port`, but the
    result is the plain element operator: no excitation is ever set, so
    it acts as a pure in-circuit load (``v_src = 0`` in the Thévenin
    update).  ``Z0 = 0`` — a passive element has no power-wave
    reference.

    Parameters
    ----------
    spec : magnelio.circuit.LumpedElement
    mesh : Mesh
    m_eps : np.ndarray
        Diagonal of the FIT ``M_eps`` matrix on the flat E layout.
    m_mu : np.ndarray
        Unused; accepted for symmetry with the port builders.
    dt : float
        Solver time step [s].
    """
    del m_mu

    direction, flat_indices, ijk_list, dl_list = _snap_edge_chain(
        f"LumpedElement {spec.name!r}",
        spec.start,
        spec.end,
        mesh,
    )

    m_eps_elem = np.asarray(m_eps[flat_indices], dtype=float)
    beta_E = dt / m_eps_elem

    component = {"x": 0, "y": 1, "z": 2}[direction]
    n = len(flat_indices)
    return LumpedElementOperator(
        name=spec.name,
        Z0=0.0,
        element=copy.deepcopy(spec.element),
        flat_edge_indices=list(flat_indices),
        ijk_list=ijk_list,
        dl_list=dl_list,
        edge_components=[component] * n,
        edge_signs=[1.0] * n,
        beta_E=beta_E,
    )


def _snap_edge_chain(
    what: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    mesh,
) -> tuple[str, list[int], list[tuple[int, int, int]], list[float]]:
    """Snap two endpoints to the grid and resolve the E-edge chain.

    Shared by :func:`build_lumped_port` and
    :func:`build_lumped_element`; *what* prefixes the error messages
    (e.g. ``"PortSpecLumped 'p1'"``).

    Returns
    -------
    tuple
        ``(direction, flat_indices, ijk_list, dl_list)`` with
        ``direction`` in ``{"x", "y", "z"}`` and ``flat_indices`` on
        the flat ``Ex|Ey|Ez`` layout.
    """
    grid = mesh.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

    ix_s = int(np.argmin(np.abs(grid.x - start[0])))
    iy_s = int(np.argmin(np.abs(grid.y - start[1])))
    iz_s = int(np.argmin(np.abs(grid.z - start[2])))
    ix_e = int(np.argmin(np.abs(grid.x - end[0])))
    iy_e = int(np.argmin(np.abs(grid.y - end[1])))
    iz_e = int(np.argmin(np.abs(grid.z - end[2])))

    diff = ((ix_s != ix_e), (iy_s != iy_e), (iz_s != iz_e))
    if sum(diff) != 1:
        raise ValueError(
            f"{what}: start and end must differ "
            f"along exactly one Cartesian axis after grid snapping. "
            f"Got start node ({ix_s},{iy_s},{iz_s}), "
            f"end node ({ix_e},{iy_e},{iz_e})."
        )

    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)

    flat_indices: list[int] = []
    ijk_list: list[tuple[int, int, int]] = []
    dl_list: list[float] = []

    if diff[0]:
        direction = "x"
        lo, hi = sorted((ix_s, ix_e))
        for ix in range(lo, hi):
            ix_c = max(0, min(ix, Nx - 1))
            iy_c = max(0, min(iy_s, Ny))
            iz_c = max(0, min(iz_s, Nz))
            flat_indices.append(edge_index_Ex(ix_c, iy_c, iz_c, Nx, Ny, Nz))
            ijk_list.append((ix_c, iy_c, iz_c))
            dl_list.append(float(grid.dx[ix_c]))
    elif diff[1]:
        direction = "y"
        lo, hi = sorted((iy_s, iy_e))
        for iy in range(lo, hi):
            ix_c = max(0, min(ix_s, Nx))
            iy_c = max(0, min(iy, Ny - 1))
            iz_c = max(0, min(iz_s, Nz))
            flat_indices.append(
                n_Ex + edge_index_Ey(ix_c, iy_c, iz_c, Nx, Ny, Nz),
            )
            ijk_list.append((ix_c, iy_c, iz_c))
            dl_list.append(float(grid.dy[iy_c]))
    else:
        direction = "z"
        lo, hi = sorted((iz_s, iz_e))
        for iz in range(lo, hi):
            ix_c = max(0, min(ix_s, Nx))
            iy_c = max(0, min(iy_s, Ny))
            iz_c = max(0, min(iz, Nz - 1))
            flat_indices.append(
                n_Ex + n_Ey + edge_index_Ez(ix_c, iy_c, iz_c, Nx, Ny, Nz),
            )
            ijk_list.append((ix_c, iy_c, iz_c))
            dl_list.append(float(grid.dz[iz_c]))

    if sum(dl_list) == 0.0:
        raise ValueError(
            f"{what}: zero-length edge chain — start and end coincide on the snapped grid.",
        )

    return direction, flat_indices, ijk_list, dl_list
