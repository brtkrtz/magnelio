"""Canonical curve rasteriser: ``Curve`` → ordered, directed grid E-edges.

This is the single shared kernel behind every Cluster-3 grid consumer
(voltage integration now; lumped-RLC and thin-wire later).  Keeping it the
ONE canonical rasteriser is a hard guardrail: if two consumers discretised a
curve independently, a voltage integral and a thin-wire current could
disagree on which edges the curve occupies.

Algorithm
---------
1. **Sample** the curve at quasi-uniform arc length no coarser than a
   fraction of the smallest cell (:func:`~magnelio.geo._occ_backend.
   sample_wire`), so consecutive samples never skip a grid node.
2. **Snap** each sample to its nearest primary node (per-axis independent on
   the rectilinear grid) and collapse consecutive repeats into a node chain.
3. **Staircase**: walk the node chain in unit steps.  Two adjacent chain
   nodes almost always differ by a single axis (dense sampling); a rare
   multi-axis jump is filled by a monotone x→y→z staircase.  Each unit step
   ``A→B`` emits the primary E-edge at the lower node with ``sign = ±1`` and
   its physical length ``dl``.

The signed edge chain makes the discrete line integral ``Σ sign·E·dl`` equal
to ``∫_curve E·dl``: for a uniform (or any conservative) field the result
depends only on the snapped endpoints, so an oblique curve — a helix is
oblique everywhere — integrates correctly despite the staircase.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magnelio.mesh.indexing import edge_index_Ex, edge_index_Ey, edge_index_Ez


@dataclass
class EdgePath:
    """An ordered, directed chain of primary-grid E-edges (a rasterised Curve).

    Entry ``k`` describes one traversed edge: ``axes[k]`` is ``'x'``/``'y'``/
    ``'z'``, ``ijk[k]`` is the edge's **lower-index** base node, ``signs[k]``
    is ``+1`` if the curve runs along ``+axis`` there (else ``-1``),
    ``dls[k]`` is the edge length [m], and ``flat_indices[k]`` indexes the
    flat E layout (``Ex|Ey|Ez`` concatenated — the ``FieldState`` / ``M_eps``
    ordering), so the same path serves both field-array and flat-vector
    consumers.
    """

    axes: list[str]
    ijk: list[tuple[int, int, int]]
    signs: list[int]
    dls: list[float]
    flat_indices: list[int]

    def __len__(self) -> int:
        return len(self.axes)

    @property
    def length(self) -> float:
        """Total traversed (staircase) edge length [m]."""
        return float(sum(self.dls))


def _nearest_node(p, grid) -> tuple[int, int, int]:
    """Nearest primary node to point *p* — per-axis on the rectilinear grid."""
    return (
        int(np.argmin(np.abs(grid.x - p[0]))),
        int(np.argmin(np.abs(grid.y - p[1]))),
        int(np.argmin(np.abs(grid.z - p[2]))),
    )


def rasterize_curve(
    curve, grid, *, samples_per_cell: int = 4, scale: float | None = None
) -> EdgePath:
    """Rasterise *curve* onto the primary E-edges of *grid*.

    Parameters
    ----------
    curve : Curve
        The abstract 3D locus to rasterise (any
        :class:`~magnelio.geo.curves.Curve`).
    grid : GridLines
        The simulation grid (``mesh.grid``).
    samples_per_cell : int, default 4
        Curve samples per smallest cell length; ``>= 2`` guarantees no node
        is skipped, higher values only refine the staircase geometry (the
        line integral of a conservative field is unaffected).
    scale : float or None, default None
        Model scale factor to build the curve's OCC wire at.
        ``None`` derives it from the curve's own analytic bounding box.

    Returns
    -------
    EdgePath
        The ordered, directed edge chain the curve occupies.

    Raises
    ------
    ValueError
        If the curve is shorter than one cell (rasterises to a single node),
        or *samples_per_cell* < 2.
    """
    from magnelio.geo._occ_backend import sample_wire  # noqa: PLC0415

    if samples_per_cell < 2:
        raise ValueError(
            f"samples_per_cell must be >= 2 to avoid skipping nodes; got {samples_per_cell}."
        )

    if scale is None:
        from magnelio.geo._scaling import choose_scale  # noqa: PLC0415

        scale = choose_scale(*curve._analytic_bbox())
    min_cell = min(grid.dx_min, grid.dy_min, grid.dz_min)
    pts = sample_wire(curve._occ_shape(scale), min_cell / samples_per_cell, scale=scale)

    nodes: list[tuple[int, int, int]] = []
    for p in pts:
        nd = _nearest_node(p, grid)
        if not nodes or nd != nodes[-1]:
            nodes.append(nd)

    if len(nodes) < 2:
        raise ValueError(
            "Curve rasterises to a single grid node — it is shorter than one "
            "cell along every axis, or the grid is too coarse to resolve it.",
        )

    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)

    axes: list[str] = []
    ijk: list[tuple[int, int, int]] = []
    signs: list[int] = []
    dls: list[float] = []
    flats: list[int] = []

    cur = list(nodes[0])
    for target in nodes[1:]:
        while (cur[0], cur[1], cur[2]) != target:
            # First differing axis → one unit step (monotone x→y→z staircase).
            for a in (0, 1, 2):
                if cur[a] == target[a]:
                    continue
                step = 1 if target[a] > cur[a] else -1
                base = list(cur)
                if step < 0:
                    base[a] -= 1  # edge base = lower node
                bi, bj, bk = base
                if a == 0:
                    axes.append("x")
                    dls.append(float(grid.dx[bi]))
                    flats.append(edge_index_Ex(bi, bj, bk, Nx, Ny, Nz))
                elif a == 1:
                    axes.append("y")
                    dls.append(float(grid.dy[bj]))
                    flats.append(n_Ex + edge_index_Ey(bi, bj, bk, Nx, Ny, Nz))
                else:
                    axes.append("z")
                    dls.append(float(grid.dz[bk]))
                    flats.append(
                        n_Ex + n_Ey + edge_index_Ez(bi, bj, bk, Nx, Ny, Nz),
                    )
                ijk.append((bi, bj, bk))
                signs.append(step)
                cur[a] += step
                break

    return EdgePath(axes=axes, ijk=ijk, signs=signs, dls=dls, flat_indices=flats)


def integrate_E(field, curve, grid, *, samples_per_cell: int = 4) -> float:
    """Line integral ``∫_curve E·dl`` [V] of an E field along *curve*.

    The first (read-only) consumer of :func:`rasterize_curve`: it sums the
    signed edge voltages ``Σ sign · E · dl`` along the rasterised chain.  For
    a conservative field the result depends only on the curve's endpoints, so
    it validates the rasteriser before any physics is built on it.

    Parameters
    ----------
    field : FieldState
        The E field to integrate (``field.Ex/Ey/Ez``).
    curve : Curve
        The path of integration.
    grid : GridLines
        The simulation grid the field lives on.
    samples_per_cell : int, default 4
        Forwarded to :func:`rasterize_curve`.

    Returns
    -------
    float
        The line integral [V] (the total voltage along the curve).
    """
    path = rasterize_curve(curve, grid, samples_per_cell=samples_per_cell)
    comp = {"x": field.Ex, "y": field.Ey, "z": field.Ez}
    v = 0.0
    for axis, (i, j, k), sign, dl in zip(
        path.axes,
        path.ijk,
        path.signs,
        path.dls,
    ):
        v += sign * float(comp[axis][i, j, k]) * dl
    return v
