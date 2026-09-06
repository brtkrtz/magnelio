"""Region operators — what a recorded field needs to state its energy and flux.

A monitor's frames are the solver's grid quantities on the Yee positions
of a region; the stored energy and the Poynting flux are FIT identities
on them — ``½ eᵀ Mε e + ½ hᵀ Mμ h`` and ``Σ e·h`` — that need nothing
but the material operators of the region and, at its edges, the share
of every dual patch that lies inside.  The monitor cuts the operators
from the solver's own diagonals when it is attached and the containers
carry them (:class:`RegionOperators`), so a frame can state its energy
and flux on its own, live or read back from a project.

Booking follows DD-155: a model solved on a symmetry-reduced domain
reports full-model joules and watts — a factor two per symmetry plane,
for the flux only per plane whose axis lies in the cross-section.
"""

# Design: DD-260 (energy and flux from a recording).

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_AXES = "xyz"
_E = ("Ex", "Ey", "Ez")
_H = ("Hx", "Hy", "Hz")
_FACES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
# How a region ends along an axis: cut through the interior, on a
# domain wall, or on a magnetic (PMC) wall half a cell outside.
_END_KINDS = ("cut", "wall", "pmc")


def _to_host(a):
    return a.get() if type(a).__module__.partition(".")[0] == "cupy" else np.asarray(a)


@dataclass(frozen=True)
class RegionOperators:
    """The material operators of a region and how its edges are booked.

    Attributes
    ----------
    m_eps : dict[str, np.ndarray]
        Diagonal of the permittivity operator on the region's E edges
        [F], one Yee-shaped array per component ``Ex``, ``Ey``, ``Ez``.
    m_mu : dict[str, np.ndarray]
        Diagonal of the permeability operator on the region's H faces
        [H], per ``Hx``, ``Hy``, ``Hz``.
    ends : tuple[tuple[str, str], ...]
        Per axis the kind of the region's low and high end: ``"cut"``
        (the region stops inside the domain — or on an electric
        symmetry plane, where the other half of the dual patch belongs
        to the mirror image), ``"wall"`` (a physical domain wall: PEC,
        absorber, port, periodic) or ``"pmc"`` (a magnetic wall, half a
        cell outside the last node, symmetry plane or not).
    symmetry : tuple[str, ...]
        Faces of the model's symmetry planes (``"xmin"``, …) — the
        full-model booking doubles per plane.
    """

    m_eps: dict
    m_mu: dict
    ends: tuple
    symmetry: tuple

    def __post_init__(self) -> None:
        for name in _E:
            if name not in self.m_eps:
                raise KeyError(f"m_eps lacks {name!r}")
        for name in _H:
            if name not in self.m_mu:
                raise KeyError(f"m_mu lacks {name!r}")
        if len(self.ends) != 3 or any(
            len(pair) != 2 or any(k not in _END_KINDS for k in pair) for pair in self.ends
        ):
            raise ValueError(f"ends must be three (low, high) pairs out of {_END_KINDS}")
        unknown = sorted(set(self.symmetry) - set(_FACES))
        if unknown:
            raise ValueError(f"unknown symmetry face(s) {unknown}")

    # ── booking ──────────────────────────────────────────────────────────

    def energy_factor(self) -> float:
        """Full-model factor of a stored energy: two per symmetry plane."""
        return float(2 ** len(self.symmetry))

    def flux_factor(self, axis: int) -> float:
        """Full-model factor of a flux normal to *axis*.

        A plane parallel to the cross-section leaves the aperture whole;
        every other symmetry plane halves it.
        """
        return float(2 ** sum(1 for face in self.symmetry if face[0] != _AXES[axis]))

    def edge_weights(self, grid, dual, *, purpose: str) -> tuple[np.ndarray, ...]:
        """Per-axis node weights that book a sample's dual patch inside the region.

        Interior nodes weigh one.  At an end the region is *cut*, the
        dual patch straddles the cut and only the half cell inside
        counts: ``(d/2) / dual`` — on an electric symmetry plane too,
        whose other half is the mirror image's.  On a physical domain
        wall the solver's own convention (the full end cell as the
        dual width) is kept for the energy, so a whole-domain
        recording reproduces the march's energy trace; the flux
        through a wall-bounded cross-section books the half cell at a
        wall (``½``) and the whole boundary cell at a magnetic wall,
        which sits half a cell outside — the weights
        :class:`~magnelio.monitors.MonitorFluxTime` uses.

        Parameters
        ----------
        grid : GridLines
            The region's grid.
        dual : tuple of np.ndarray or None
            The dual widths of the region's h samples (the solver's
            convention on *grid* when ``None``).
        purpose : {"energy", "flux"}
        """
        from magnelio.fields._interp import _solver_dual_widths  # noqa: PLC0415

        if purpose not in ("energy", "flux"):
            raise ValueError(f"purpose must be 'energy' or 'flux'; got {purpose!r}")
        out = []
        for a, d in enumerate((grid.dx, grid.dy, grid.dz)):
            d = np.asarray(d, dtype=float)
            dl = np.asarray(dual[a], dtype=float) if dual is not None else _solver_dual_widths(d)
            w = np.ones(d.size + 1)
            if d.size == 0:
                out.append(w)
                continue
            for pos, idx, cell in ((0, 0, d[0]), (1, -1, d[-1])):
                kind = self.ends[a][pos]
                if kind == "cut":
                    w[idx] = 0.5 * cell / dl[idx]
                elif purpose == "flux":
                    w[idx] = 1.0 if kind == "pmc" else 0.5
            out.append(w)
        return tuple(out)

    # ── serialisation (project store) ────────────────────────────────────

    def ends_codes(self) -> np.ndarray:
        """``ends`` as an ``int8`` array ``(3, 2)`` (index into the kinds)."""
        return np.array([[_END_KINDS.index(k) for k in pair] for pair in self.ends], dtype=np.int8)

    def symmetry_flags(self) -> np.ndarray:
        """``symmetry`` as ``int8`` flags over the six faces."""
        return np.array([1 if f in self.symmetry else 0 for f in _FACES], dtype=np.int8)

    @classmethod
    def from_arrays(cls, m_eps: dict, m_mu: dict, ends_codes, symmetry_flags) -> RegionOperators:
        """Inverse of :meth:`ends_codes` / :meth:`symmetry_flags`."""
        codes = np.asarray(ends_codes, dtype=int).reshape(3, 2)
        flags = np.asarray(symmetry_flags, dtype=int).reshape(6)
        return cls(
            m_eps={c: np.asarray(m_eps[c], dtype=float) for c in _E},
            m_mu={c: np.asarray(m_mu[c], dtype=float) for c in _H},
            ends=tuple((_END_KINDS[int(lo)], _END_KINDS[int(hi)]) for lo, hi in codes),
            symmetry=tuple(f for f, flag in zip(_FACES, flags) if flag),
        )


def region_operators(mesh, region, M_eps, M_mu) -> RegionOperators:
    """Cut the solver's material diagonals to a monitor region.

    Parameters
    ----------
    mesh : Mesh
        The run's mesh (grid and boundary declaration).
    region : MonitorRegion
        The region's cell slices ``ix, iy, iz``.
    M_eps, M_mu : array_like
        The solver's flat diagonals over all E edges (``Ex, Ey, Ez``
        blocks) and all H faces (``Hx, Hy, Hz``), in the run's own
        precision — port-plane flattening included.
    """
    from magnelio.boundaries.boundary_conditions import (  # noqa: PLC0415
        bc_type_entries,
        symmetry_entries,
    )
    from magnelio.fields._interp import _region_slices  # noqa: PLC0415
    from magnelio.fields.state import _yee_shapes  # noqa: PLC0415

    g = mesh.grid
    shapes = _yee_shapes(g.Nx, g.Ny, g.Nz)
    eps_flat = np.asarray(_to_host(M_eps), dtype=float)
    mu_flat = np.asarray(_to_host(M_mu), dtype=float)
    slices = (region.ix, region.iy, region.iz)

    def cut(flat, names):
        out, start = {}, 0
        for name in names:
            size = int(np.prod(shapes[name]))
            block = flat[start : start + size].reshape(shapes[name])
            out[name] = np.ascontiguousarray(block[_region_slices(*slices, name)])
            start += size
        if start != flat.size:
            raise ValueError(f"diagonal of length {flat.size} does not match the grid")
        return out

    bc = getattr(mesh, "boundary_conditions", None)
    types = bc_type_entries(bc) if bc is not None else {}
    symmetry = tuple(sorted(symmetry_entries(bc))) if bc is not None else ()
    n_cells = (g.Nx, g.Ny, g.Nz)
    ends = []
    for a, sl in enumerate(slices):
        pair = []
        for at_low, on_wall in ((True, sl.start == 0), (False, sl.stop == n_cells[a])):
            if not on_wall:
                pair.append("cut")
                continue
            face = f"{_AXES[a]}{'min' if at_low else 'max'}"
            if types.get(face) == "PMC":
                pair.append("pmc")
            elif face in symmetry:
                # An electric symmetry plane on a grid line: the mirror
                # image owns the other half of every dual patch on it.
                pair.append("cut")
            else:
                pair.append("wall")
        ends.append(tuple(pair))
    return RegionOperators(
        m_eps=cut(eps_flat, _E), m_mu=cut(mu_flat, _H), ends=tuple(ends), symmetry=symmetry
    )


# ── the identities ───────────────────────────────────────────────────────


def _transverse_weight(w: tuple, name: str) -> np.ndarray:
    """Broadcastable weight of a component's samples from the per-axis node weights.

    An E component is sampled on nodes across its own axis, an H
    component on nodes along it; the weight of a sample is the product
    of the node weights of the axes it sits on nodes of.
    """
    group, axis = name[0], _AXES.index(name[1])
    out = np.ones((1, 1, 1))
    for a in range(3):
        on_nodes = (a != axis) if group == "E" else (a == axis)
        if on_nodes:
            shape = [1, 1, 1]
            shape[a] = w[a].size
            out = out * w[a].reshape(shape)
    return out


def energy(raw, ops: RegionOperators, grid, dual, h_lead: float) -> float:
    """Stored energy [J] of one frame, full-model booking.

    *raw* holds the six grid quantities (a :class:`FieldArrays`).  For a
    real frame whose magnetic samples lead the electric ones by
    *h_lead* — the state of a leapfrog march, ``h_lead = dt/2`` — the
    magnetic term is the one the march conserves (DD-225), the pairing
    ``h(t−dt/2)·Mμ·h(t+dt/2)`` obtained from the frame alone through
    the discrete Faraday law: ``½ hᵀMμh + h_lead·(C e)ᵀh``.  A complex
    frame is an RMS phasor — the convention of a monitor's spectrum per
    1 W CW (DD-078: ``|a|² = P``) — so its time-averaged energy is
    ``½ eᵀMε e* + ½ hᵀMμ h*`` with no further factor.
    """
    w = ops.edge_weights(grid, dual, purpose="energy")
    is_complex = any(np.iscomplexobj(getattr(raw, c)) for c in _E + _H)
    total = 0.0
    for name in _E:
        e = np.asarray(_to_host(getattr(raw, name)))
        m = ops.m_eps[name] * _transverse_weight(w, name)
        total += float(np.sum(m * (np.abs(e) ** 2 if is_complex else e * e)))
    for name in _H:
        h = np.asarray(_to_host(getattr(raw, name)))
        m = ops.m_mu[name] * _transverse_weight(w, name)
        total += float(np.sum(m * (np.abs(h) ** 2 if is_complex else h * h)))
    total *= 0.5
    if not is_complex and h_lead:
        from magnelio._operators.curl import curl_e_stencil  # noqa: PLC0415
        from magnelio.fields.state import _yee_shapes  # noqa: PLC0415

        shapes = _yee_shapes(grid.Nx, grid.Ny, grid.Nz)
        e = [np.asarray(_to_host(getattr(raw, c)), dtype=float) for c in _E]
        curl = [np.empty(shapes[c]) for c in _H]
        curl_e_stencil(*e, *curl)
        for name, c in zip(_H, curl):
            h = np.asarray(_to_host(getattr(raw, name)), dtype=float)
            live = ops.m_mu[name] > 0.0
            wt = _transverse_weight(w, name)
            total += float(h_lead) * float(np.sum(np.where(live, wt * c * h, 0.0)))
    return total * ops.energy_factor()


def flux(raw, ops: RegionOperators, grid, dual, axis: int, k: int) -> float:
    """Poynting flux [W] through the region's cross-section at node *k* of *axis*.

    The FIT pairing ``Σ e·h`` of :class:`~magnelio.monitors.MonitorFluxTime`
    (DD-085) on the region's own samples, positive along the axis; a
    complex frame is an RMS phasor (a spectrum per 1 W CW) and yields
    the time-averaged real power ``Re Σ e·h*``.
    """
    w = ops.edge_weights(grid, dual, purpose="flux")
    b, c = (axis + 1) % 3, (axis + 2) % 3
    e_b, h_c = np.asarray(_to_host(getattr(raw, _E[b]))), np.asarray(_to_host(getattr(raw, _H[c])))
    e_c, h_b = np.asarray(_to_host(getattr(raw, _E[c]))), np.asarray(_to_host(getattr(raw, _H[b])))
    is_complex = any(np.iscomplexobj(a) for a in (e_b, h_c, e_c, h_b))
    n = (grid.Nx, grid.Ny, grid.Nz)[axis]
    if not (0 <= k <= n):
        raise IndexError(f"node {k} out of range along axis {axis} ({n} cells)")
    k = min(k, n - 1)
    idx = [slice(None)] * 3
    idx[axis] = k
    idx = tuple(idx)
    # The sliced planes keep the two other axes in ascending order.
    others = [a for a in range(3) if a != axis]

    def on_plane(weights, of_axis):
        shape = [1, 1]
        shape[others.index(of_axis)] = weights.size
        return weights.reshape(shape)

    w_c, w_b = on_plane(w[c], c), on_plane(w[b], b)
    if is_complex:
        term1 = np.sum(e_b[idx] * np.conj(h_c[idx]) * w_c)
        term2 = np.sum(e_c[idx] * np.conj(h_b[idx]) * w_b)
        p = float(np.real(term1 - term2))
    else:
        p = float(np.sum(e_b[idx] * h_c[idx] * w_c)) - float(np.sum(e_c[idx] * h_b[idx] * w_b))
    return p * ops.flux_factor(axis)


def plane_index(grid, axis: int, position: float) -> int:
    """The region node nearest to *position* [m] along *axis*."""
    nodes = np.asarray((grid.x, grid.y, grid.z)[axis], dtype=float)
    return int(np.argmin(np.abs(nodes - float(position))))


def no_operators(what: str) -> RuntimeError:
    return RuntimeError(
        f"{what} carries no material operators, so its energy and flux cannot be "
        "stated: a field monitor's recording or spectrum carries the operators "
        "of its region (a project store keeps them); a field assembled by hand "
        "or an eigenmode does not."
    )


__all__ = ["RegionOperators", "region_operators", "energy", "flux", "plane_index"]
