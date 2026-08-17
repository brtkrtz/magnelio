"""PortSpecLumped + build_lumped_port — declarative spec + builder.

Mirror of the modal-port factory pattern: a plain dataclass holds the
user's intent (label, endpoints, internal impedance), and a builder
function resolves it onto a concrete mesh, returning a runtime operator
that implements the :class:`magnelio.ports.base.Port` protocol.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from magnelio.circuit.companion import ParallelRLC, SeriesRLC
from magnelio.mesh.indexing import edge_index_Ex, edge_index_Ey, edge_index_Ez
from magnelio.ports._lumped.operator import LumpedElementOperator, PortOperatorLumped
from magnelio.ports._lumped.port_report import LumpedPortReport

_FACE_AXIS_SIDE = {
    "xmin": (0, "min"),
    "xmax": (0, "max"),
    "ymin": (1, "min"),
    "ymax": (1, "max"),
    "zmin": (2, "min"),
    "zmax": (2, "max"),
}


@dataclass
class PortSpecLumped:
    """Declarative description of a lumped discrete port / RLC element.

    Parameters
    ----------
    name : str
        Unique port identifier (used as recorder channel key).
    start, end : tuple[float, float, float]
        Endpoints in metres.  The two points must differ along exactly
        one Cartesian axis after grid snapping.  Under a clipping
        symmetry declaration they stay in full-model coordinates;
        ``Z0`` / ``element`` are full-model values throughout, and the
        builder derives the internally scaled half-model device.
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

    what = f"PortSpecLumped {spec.name!r}"
    start, end, report = _resolve_symmetry(what, spec.start, spec.end, mesh)
    direction, flat_indices, ijk_list, dl_list = _snap_edge_chain(
        what,
        start,
        end,
        mesh,
    )

    m_eps_port = np.asarray(m_eps[flat_indices], dtype=float)
    beta_E = dt / m_eps_port

    # DD-172: the user's Z0 / element describe the full-model device.
    # The meshed half carries the internally scaled device (series cut:
    # Z/2, parallel cut: 2·Z); with that, recorded power waves and the
    # injection pick up the modal √2-per-plane convention through the
    # shared port_report plumbing.
    z_int = report.z_internal_scale if report is not None else 1.0

    return PortOperatorLumped(
        name=spec.name,
        Z0=spec.Z0 * z_int,
        direction=direction,
        flat_edge_indices=list(flat_indices),
        ijk_list=ijk_list,
        dl_list=dl_list,
        beta_E=beta_E,
        # Fresh companion state per built operator (operators are built
        # per excitation; the user's spec instance must stay pristine).
        element=_scaled_element(spec.element, z_int),
        port_report=report,
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

    what = f"LumpedElement {spec.name!r}"
    start, end, report = _resolve_symmetry(what, spec.start, spec.end, mesh)
    direction, flat_indices, ijk_list, dl_list = _snap_edge_chain(
        what,
        start,
        end,
        mesh,
    )

    m_eps_elem = np.asarray(m_eps[flat_indices], dtype=float)
    beta_E = dt / m_eps_elem

    z_int = report.z_internal_scale if report is not None else 1.0

    component = {"x": 0, "y": 1, "z": 2}[direction]
    n = len(flat_indices)
    return LumpedElementOperator(
        name=spec.name,
        Z0=0.0,
        element=_scaled_element(spec.element, z_int),
        flat_edge_indices=list(flat_indices),
        ijk_list=ijk_list,
        dl_list=dl_list,
        edge_components=[component] * n,
        edge_signs=[1.0] * n,
        beta_E=beta_E,
        port_report=report,
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


def _scaled_element(
    element: Optional[Union[SeriesRLC, ParallelRLC]],
    z_scale: float,
) -> Optional[Union[SeriesRLC, ParallelRLC]]:
    """A fresh companion instance with all impedances scaled by *z_scale*.

    Both topologies scale identically (``R → R·s``, ``L → L·s``,
    ``C → C/s`` multiplies every branch impedance by *s*).  ``z_scale
    == 1`` reduces to the historic deep copy — fresh transient state,
    identical values.
    """
    if element is None:
        return None
    if z_scale == 1.0:
        return copy.deepcopy(element)
    return type(element)(
        R=None if element.R is None else element.R * z_scale,
        L=None if element.L is None else element.L * z_scale,
        C=None if element.C is None else element.C / z_scale,
    )


def _resolve_symmetry(
    what: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    mesh,
) -> tuple[tuple[float, float, float], tuple[float, float, float], Optional[LumpedPortReport]]:
    """Relate a declared edge chain to the declared symmetry planes.

    Validates the chain against every symmetry plane of the mesh's
    boundary conditions, clips a plane-crossing chain to the meshed
    half, and returns ``(start, end, report)`` with a
    :class:`LumpedPortReport` when at least one plane cuts the chain
    (``None`` otherwise).  See the case table in the raising docs
    below; meshes without symmetry declarations pass through untouched.
    """
    # Detection + validation matrix: DD-172.  The mirror-twin warning
    # follows the modal wording (ports/_modal/factory.py,
    # _with_symmetry_faces, DD-155).
    from magnelio.boundaries.boundary_conditions import (  # noqa: PLC0415
        bc_type_entries,
        symmetry_entries,
    )

    bc = getattr(mesh, "boundary_conditions", None)
    sym = symmetry_entries(bc)
    if not sym:
        return start, end, None

    types = bc_type_entries(bc)
    grid = mesh.grid
    axis_coords = (grid.x, grid.y, grid.z)
    axis_widths = (grid.dx, grid.dy, grid.dz)

    start = list(start)
    end = list(end)
    chain_axis = int(np.argmax([abs(end[i] - start[i]) for i in range(3)]))

    faces: list[tuple[str, str, str]] = []
    for face in sorted(sym):
        axis, side = _FACE_AXIS_SIDE[face]
        position = sym[face]
        as_built = position is None
        coords = axis_coords[axis]
        if as_built:
            wall = float(coords[0] if side == "min" else coords[-1])
        else:
            wall = float(position)
        widths = axis_widths[axis]
        cell = float(widths[0] if side == "min" else widths[-1])
        tol = 0.5 * cell
        kind = types[face]

        def _inward(c: float) -> float:
            # Signed distance into the meshed half (positive = kept side).
            return (c - wall) if side == "min" else (wall - c)

        s_in = _inward(start[axis])
        e_in = _inward(end[axis])

        if chain_axis == axis:
            lo_in, hi_in = sorted((s_in, e_in))
            if hi_in <= tol:
                raise ValueError(
                    f"{what}: the chain lies in the half-space removed by the "
                    f"symmetry declaration on face {face!r} (plane at "
                    f"{wall:.6g} m along {'xyz'[axis]}).  Declare the element "
                    f"in the kept half, or drop the symmetry declaration.",
                )
            if lo_in < -tol:
                # The plane bisects the chain.
                if kind == "PMC":
                    raise ValueError(
                        f"{what}: the chain crosses the magnetic symmetry "
                        f"plane on face {face!r}.  A current normal to a "
                        f"magnetic plane mirrors anti-parallel, so no "
                        f"full-model element corresponds to this "
                        f"declaration; a plane-crossing element needs an "
                        f"electric symmetry plane.",
                    )
                if abs(s_in + e_in) > tol:
                    raise ValueError(
                        f"{what}: the chain crosses the symmetry plane on "
                        f"face {face!r} asymmetrically (endpoint distances "
                        f"{abs(min(s_in, e_in)):.6g} m / "
                        f"{abs(max(s_in, e_in)):.6g} m).  An element crossing "
                        f"a symmetry plane must be mirror-symmetric about it.",
                    )
                # Clip to the meshed half: the outside terminal moves
                # onto the wall (an electric plane carries an exact
                # grid node there).
                if s_in < e_in:
                    start[axis] = wall
                else:
                    end[axis] = wall
                faces.append((face, kind, "crossing"))
            elif lo_in <= tol:
                # One terminal on the plane, chain body inside.
                if as_built:
                    # ForceSymmetry*: the model is declared as built
                    # (halved), so the half element ending on the plane
                    # IS the crossing declaration.
                    if kind == "PMC":
                        raise ValueError(
                            f"{what}: the chain ends on the magnetic "
                            f"symmetry plane on face {face!r} along the "
                            f"plane normal.  Its mirror continuation "
                            f"crosses the plane, which a magnetic plane "
                            f"does not support (the mirrored current is "
                            f"anti-parallel).",
                        )
                    if s_in < e_in:
                        start[axis] = wall
                    else:
                        end[axis] = wall
                    faces.append((face, kind, "crossing"))
                else:
                    raise ValueError(
                        f"{what}: a chain terminal lies on the symmetry "
                        f"plane on face {face!r}.  Endpoints are declared "
                        f"in full-model coordinates here — declare the "
                        f"full element crossing the plane, or declare the "
                        f"boundary as as-built symmetry if the geometry "
                        f"is meant to be halved.",
                    )
            # else: chain entirely inside, away from this plane.
        else:
            lo_in = min(s_in, e_in)
            hi_in = max(s_in, e_in)
            if hi_in < -tol:
                raise ValueError(
                    f"{what}: the chain lies in the half-space removed by the "
                    f"symmetry declaration on face {face!r} (plane at "
                    f"{wall:.6g} m along {'xyz'[axis]}).  Declare the element "
                    f"in the kept half, or drop the symmetry declaration.",
                )
            if abs(s_in) <= tol and abs(e_in) <= tol:
                # The chain lies in the plane.
                if kind == "PEC":
                    raise ValueError(
                        f"{what}: the chain lies in the electric symmetry "
                        f"plane on face {face!r}; tangential edges inside "
                        f"an electric wall are shorted.  Move the element "
                        f"off the plane, or declare a magnetic symmetry "
                        f"plane if the fields support it.",
                    )
                faces.append((face, kind, "containment"))
            elif lo_in < -tol:
                raise ValueError(
                    f"{what}: a chain terminal lies beyond the symmetry "
                    f"plane on face {face!r} (plane at {wall:.6g} m along "
                    f"{'xyz'[axis]}).  Declare the element in the kept "
                    f"half, or drop the symmetry declaration.",
                )
            # else: chain inside, away from this plane.

    if not faces:
        warnings.warn(
            f"{what}: the chain does not touch the declared symmetry "
            f"plane(s) {sorted(sym)} — in the full model this element "
            f"has a mirror twin, and the half-model run can only "
            f"realise the symmetric (in-phase) response of the twin "
            f"pair.  Model the element on the symmetry plane, or drop "
            f"the symmetry declaration if the twin is not intended.",
            UserWarning,
            stacklevel=3,
        )
        return tuple(start), tuple(end), None

    return tuple(start), tuple(end), LumpedPortReport(symmetry_faces=tuple(faces))
