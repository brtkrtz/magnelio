"""2D view of the mesh itself: cells by material, grid lines by origin.

Main entry point:
    plot_mesh_section — render an axis-aligned slice through a Mesh

Where :func:`~magnelio.post.plot_geometry.plot_cross_section` draws the
*geometry* and optionally the grid on top, this module draws the
*discretisation*: what the mesher made of the geometry, on the real
node coordinates, and every grid line in the style of the rule that
placed it (material face, bounding-box extent, geometry edge, thin
sheet, wire, symmetry face, forced position) — with the graded fill
nodes as hairlines and the absorber cells shaded.  The geometry section
outline can be overlaid so that the discretisation and the exact
contour are read against each other.

Three fills are offered, and they answer different questions:

``fill="coverage"`` (default)
    How much of each cell is conductor.  The primal faces normal to the
    cut, in the node plane nearest to it, carry the exact PEC-covered
    area the sub-cell classifier measured (the same area the magnetic
    material matrix uses).  Every cell is drawn in the colour of its
    classified material, blended towards PEC grey by that share — a
    conductor contour comes out as the smooth ring it is.
``fill="material"``
    The cell classification — the material whose volume contains the
    cell centre.  This is the staircase *baseline* of the material
    matrices, not the accuracy of the discretisation: on every cell the
    geometry cuts, the solver overrides it with area/length-weighted
    sub-cell values.  Thin sheets are invisible here by design.
``fill="conformal"``
    What the electric material matrix holds for the field component
    *normal* to the cut.  The primal edges normal to the cut own dual
    faces that lie *in* the cut plane and tile it (tiles centred on the
    grid nodes, bounded by the cell midpoints).  Each tile is coloured
    by the area-weighted permittivity the classifier put into the mass
    matrix for that edge: PEC counts zero, dielectrics by their area
    share, a PEC-masked edge is a PEC tile.  Around a conductor this is
    coarser than the coverage: edges that run along a conductor surface
    are held at the conductor's potential, so the masked tiles reach
    one node beyond the contour.

The edge layer (``edges=True``) adds the in-plane primal edges of the
node plane nearest to the cut: PEC-masked edges, edges only partly
outside PEC (their free-length fraction ``f_L`` sets the opacity), and
edges the enlarged-cell technique borrowed out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from magnelio.geo import GeometryModel
    from magnelio.mesh.grid import GridLines
    from magnelio.mesh.mesher import Mesh

# One style per plane kind.  A plane with several sources is drawn in
# the style of its highest-ranking kind (``_KIND_PRIORITY``): a sheet or
# a symmetry face says more about the model than the bounding box that
# happens to coincide with it.
_PLANE_STYLE: dict[str, dict] = {
    "sheet": dict(color="#8c564b", linestyle="-", linewidth=1.4),
    "symmetry": dict(color="#9467bd", linestyle=(0, (6, 2)), linewidth=1.2),
    "forced": dict(color="#2ca02c", linestyle=":", linewidth=1.2),
    "face": dict(color="#202020", linestyle="-", linewidth=1.0),
    "wire": dict(color="#c8963c", linestyle="-.", linewidth=1.0),
    "edge": dict(color="#1f77b4", linestyle="--", linewidth=0.9),
    "extent": dict(color="#9a9a9a", linestyle="-", linewidth=0.6),
}
_KIND_PRIORITY = ("sheet", "symmetry", "forced", "face", "wire", "edge", "extent")
_KIND_LABEL = {
    "sheet": "thin sheet",
    "symmetry": "symmetry face",
    "forced": "forced plane",
    "face": "material face",
    "wire": "wire vertex",
    "edge": "geometry edge",
    "extent": "bounding box",
}
_GRADED_STYLE = dict(color="#bbbbbb", linestyle="-", linewidth=0.3, alpha=0.7)
_SINGULAR_COLOR = "#d62728"
_PML_STYLE = dict(facecolor="#dddddd", edgecolor="#999999", hatch="///", alpha=0.35, linewidth=0)

# Edge layer.  Masked edges are the conductor as the solver enforces it
# (E_tangential = 0); partially free edges carry the Dey–Mittra length
# weighting, drawn stronger the more of them sits in metal.
_MASKED_EDGE_STYLE = dict(color="#404040", linewidth=1.8)
_PARTIAL_EDGE_COLOR = (0.851, 0.373, 0.008)  # "#d95f02"
_PARTIAL_EDGE_WIDTH = 1.8
_BORROWED_EDGE_STYLE = dict(
    linestyle="none", marker="x", markersize=5, color=_SINGULAR_COLOR, markeredgewidth=1.2
)
_PEC_GREY = (0.65, 0.65, 0.65)

_FILL_MODES = (None, "material", "coverage", "conformal")


def _style_of(record) -> tuple[str, dict]:
    kinds = record.kinds
    for kind in _KIND_PRIORITY:
        if kind in kinds:
            return kind, _PLANE_STYLE[kind]
    return "extent", _PLANE_STYLE["extent"]


# --------------------------------------------------------------------------
# Sub-cell helpers (pure NumPy; the data comes from ``mesh.edge_material``)
# --------------------------------------------------------------------------


def _edge_counts(grid: "GridLines") -> tuple[int, int, int]:
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    return Nx * (Ny + 1) * (Nz + 1), (Nx + 1) * Ny * (Nz + 1), (Nx + 1) * (Ny + 1) * Nz


def _edge_index(component: int, i, j, k, grid: "GridLines") -> tuple[np.ndarray, np.ndarray]:
    """Per-component and concatenated flat E-edge indices for broadcastable i, j, k.

    The per-component index addresses ``Mesh.pec_mask_edges[component]``;
    the concatenated one (``Ex | Ey | Ez``) addresses the arrays of
    :class:`~magnelio.geo._subcell.EdgeMaterialData`.  Same formulas as
    :mod:`magnelio.mesh.indexing`, vectorised.
    """
    Ny, Nz = grid.Ny, grid.Nz
    n_Ex, n_Ey, _ = _edge_counts(grid)
    i, j, k = np.broadcast_arrays(np.asarray(i), np.asarray(j), np.asarray(k))
    if component == 0:
        local = i * (Ny + 1) * (Nz + 1) + j * (Nz + 1) + k
        offset = 0
    elif component == 1:
        local = i * Ny * (Nz + 1) + j * (Nz + 1) + k
        offset = n_Ex
    else:
        local = i * (Ny + 1) * Nz + j * Nz + k
        offset = n_Ex + n_Ey
    return local, local + offset


def _face_index(component: int, i, j, k, grid: "GridLines") -> np.ndarray:
    """Concatenated flat H-face index (``Hx | Hy | Hz``) for broadcastable i, j, k.

    Addresses the arrays of
    :class:`~magnelio.geo._subcell.FaceMaterialData`; same formulas as
    :mod:`magnelio.mesh.indexing`, vectorised.
    """
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_Hx = (Nx + 1) * Ny * Nz
    n_Hy = Nx * (Ny + 1) * Nz
    i, j, k = np.broadcast_arrays(np.asarray(i), np.asarray(j), np.asarray(k))
    if component == 0:
        return i * Ny * Nz + j * Nz + k
    if component == 1:
        return n_Hx + i * (Ny + 1) * Nz + j * Nz + k
    return n_Hx + n_Hy + i * Ny * (Nz + 1) + j * (Nz + 1) + k


def _ijk(normal: str, u_idx: np.ndarray, v_idx: np.ndarray, n_idx: int):
    """(i, j, k) index grids of shape ``(len(u_idx), len(v_idx))`` for a cut."""
    from magnelio.post.plot_geometry import _AXIS_LABELS  # noqa: PLC0415

    u_label, v_label = _AXIS_LABELS[normal]
    U, V = np.meshgrid(np.asarray(u_idx), np.asarray(v_idx), indexing="ij")
    parts = {u_label: U, v_label: V, normal: np.full_like(U, n_idx)}
    return parts["x"], parts["y"], parts["z"]


def _dual_bounds(nodes: np.ndarray) -> np.ndarray:
    """Boundaries of the dual cells around ``nodes``: midpoints, ends clamped."""
    nodes = np.asarray(nodes, dtype=float)
    return np.concatenate(([nodes[0]], 0.5 * (nodes[:-1] + nodes[1:]), [nodes[-1]]))


def _conformal_tiles(mesh: "Mesh", normal: str, k: int) -> tuple[np.ndarray, float]:
    """Area-weighted ε̄ of the dual faces lying in cell layer ``k`` of the cut.

    One tile per in-plane node, indexed ``(u, v)``.  Bulk edges carry the
    staircase value of their owning cell (the clamped lookup of the mass
    matrix builder), boundary edges the classifier's ``eps_avg``; frozen
    edges (free area below the floor), interior-PEC edges and PEC-masked
    edges are 0.  Returns the tiles and the largest non-PEC permittivity
    of the material library along the normal.
    """
    from magnelio._operators.material_matrices import _FREE_AREA_FLOOR  # noqa: PLC0415
    from magnelio.post.plot_geometry import _AXIS_INDEX, _AXIS_LABELS  # noqa: PLC0415

    em = mesh.edge_material
    grid = mesh.grid
    comp = _AXIS_INDEX[normal]
    u_label, v_label = _AXIS_LABELS[normal]
    Nu = getattr(grid, f"N{u_label}")
    Nv = getattr(grid, f"N{v_label}")
    i, j, kk = _ijk(normal, np.arange(Nu + 1), np.arange(Nv + 1), k)
    local, flat = _edge_index(comp, i, j, kk, grid)

    n_ids = max(mesh.material_library) + 1
    eps_table = np.zeros(n_ids)
    for mid, mat in mesh.material_library.items():
        eps_table[mid] = 0.0 if mat.is_pec else float(mat.epsilon[comp])
    ci = np.minimum(i, grid.Nx - 1)
    cj = np.minimum(j, grid.Ny - 1)
    ck = np.minimum(kk, grid.Nz - 1)
    tiles = eps_table[mesh.material_id[ci, cj, ck]]

    cat = em.category[flat]
    averaged = (cat == 1) | (cat == 2)
    tiles[averaged] = em.eps_avg[flat][averaged]
    frozen = (cat == 2) & ~(em.f_A[flat] > _FREE_AREA_FLOOR)
    tiles[frozen | (cat == 3)] = 0.0
    tiles[mesh.pec_mask_edges[comp][local]] = 0.0
    eps_max = float(max(eps_table.max(), float(np.nanmax(tiles)) if tiles.size else 0.0))
    return tiles, eps_max


def _coverage_tiles(mesh: "Mesh", normal: str, k_n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """PEC-covered fraction of the primal faces in node plane ``k_n``, and tile colours.

    One tile per in-plane cell, indexed ``(u, v)``.  The fraction is
    the classifier's geometric ``A_face_pec / A_face`` of the faces
    normal to the cut; the colour is the classified material of the
    cell in layer ``k`` (white for a PEC-classified cell, whose
    remainder has no material of record) blended towards PEC grey by
    that fraction.  Returns ``(fraction (Nu, Nv), rgb (Nu, Nv, 3))``.
    """
    from magnelio.post._colors import material_color  # noqa: PLC0415
    from magnelio.post.plot_geometry import _AXIS_INDEX, _AXIS_LABELS  # noqa: PLC0415

    fm = mesh.face_material
    grid = mesh.grid
    comp = _AXIS_INDEX[normal]
    u_label, v_label = _AXIS_LABELS[normal]
    Nu = getattr(grid, f"N{u_label}")
    Nv = getattr(grid, f"N{v_label}")
    du = np.asarray(getattr(grid, f"d{u_label}"), dtype=float)
    dv = np.asarray(getattr(grid, f"d{v_label}"), dtype=float)
    i, j, kk = _ijk(normal, np.arange(Nu), np.arange(Nv), k_n)
    flat = _face_index(comp, i, j, kk, grid)
    area = du[:, None] * dv[None, :]
    fraction = np.clip(fm.A_face_pec[flat] / area, 0.0, 1.0)

    slab = np.take(mesh.material_id, k, axis=comp)  # (Nu, Nv), unflipped
    white = np.ones(3)
    base = np.ones((Nu, Nv, 3))
    for mid in np.unique(slab):
        mat = mesh.material_library.get(int(mid))
        if mat is None or mat.is_pec:
            continue
        r, g, b, a = material_color(mat)
        base[slab == mid] = white + (np.array([r, g, b]) - white) * a
    rgb = base * (1.0 - fraction[..., None]) + np.array(_PEC_GREY) * fraction[..., None]
    return fraction, rgb


def _eps_colormap(eps_max: float):
    """Colour map for ε̄: PEC grey at 0, white at 1, the dielectric tint above.

    The tint at the top of the range is the one
    :func:`~magnelio.post._colors.material_color` gives a material of
    that permittivity (blended over white as it appears in the geometry
    plots), so the tiles and the section outline agree in colour.
    """
    from matplotlib.colors import ListedColormap, Normalize  # noqa: PLC0415

    top = max(float(eps_max), 1.0)
    e = np.linspace(0.0, top, 256)
    grey = np.array(_PEC_GREY)
    white = np.ones(3)
    t = min((top - 1.0) / 11.0, 1.0)
    tint = np.array([0.4 + 0.5 * t, 0.7 - 0.2 * t, 0.9 - 0.6 * t])
    tint = 0.6 * tint + 0.4 * white
    colors = np.empty((e.size, 3))
    low = e <= 1.0
    w = e[low][:, None]
    colors[low] = grey + (white - grey) * w
    if (~low).any():
        w = ((e[~low] - 1.0) / (top - 1.0))[:, None]
        colors[~low] = white + (tint - white) * w
    return ListedColormap(colors), Normalize(0.0, top)


def _section_edges(mesh: "Mesh", normal: str, k_n: int) -> dict[str, np.ndarray]:
    """In-plane primal edges of node plane ``k_n``, sorted into classes.

    Returns segments in metres in unflipped ``(u, v)`` order:
    ``masked`` ``(n, 2, 2)``, ``partial`` ``(n, 2, 2)`` with ``f_L`` ``(n,)``,
    and ``borrowed`` midpoints ``(n, 2)``.
    """
    from magnelio.post.plot_geometry import _AXIS_INDEX, _AXIS_LABELS  # noqa: PLC0415

    em = mesh.edge_material
    grid = mesh.grid
    u_label, v_label = _AXIS_LABELS[normal]
    u_nodes = np.asarray(getattr(grid, u_label), dtype=float)
    v_nodes = np.asarray(getattr(grid, v_label), dtype=float)
    Nu, Nv = u_nodes.size - 1, v_nodes.size - 1

    segments, masked, f_L, borrowed = [], [], [], []
    for along, comp, i_idx, j_idx in (
        ("u", _AXIS_INDEX[u_label], np.arange(Nu), np.arange(Nv + 1)),
        ("v", _AXIS_INDEX[v_label], np.arange(Nu + 1), np.arange(Nv)),
    ):
        i, j, kk = _ijk(normal, i_idx, j_idx, k_n)
        local, flat = _edge_index(comp, i, j, kk, grid)
        U, V = np.meshgrid(i_idx, j_idx, indexing="ij")
        if along == "u":
            start = np.stack([u_nodes[U], v_nodes[V]], axis=-1)
            end = np.stack([u_nodes[U + 1], v_nodes[V]], axis=-1)
            L_primal = (u_nodes[U + 1] - u_nodes[U]).ravel()
        else:
            start = np.stack([u_nodes[U], v_nodes[V]], axis=-1)
            end = np.stack([u_nodes[U], v_nodes[V + 1]], axis=-1)
            L_primal = (v_nodes[V + 1] - v_nodes[V]).ravel()
        segments.append(np.stack([start, end], axis=-2).reshape(-1, 2, 2))
        masked.append(mesh.pec_mask_edges[comp][local.ravel()])
        with np.errstate(invalid="ignore", divide="ignore"):
            f_L.append(em.L_free[flat.ravel()] / L_primal)
        borrowed.append(em.enlarged_cell_donor[flat.ravel()] >= 0)

    segments = np.concatenate(segments)
    masked = np.concatenate(masked)
    f_L = np.concatenate(f_L)
    borrowed = np.concatenate(borrowed)
    partial = ~masked & np.isfinite(f_L) & (f_L < 1.0 - 1e-9)
    return {
        "masked": segments[masked],
        "partial": segments[partial],
        "f_L": f_L[partial],
        "borrowed": segments[borrowed].mean(axis=1),
    }


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------


def plot_mesh_section(
    mesh: "Mesh",
    normal: str,
    position: float,
    *,
    geometry: "GeometryModel | None" = None,
    scale_mm: bool = True,
    flip: bool = False,
    ax: "matplotlib.axes.Axes | None" = None,
    title: str | None = None,
    fill: str | None = "coverage",
    edges: bool = False,
    legend: bool = True,
) -> tuple["matplotlib.figure.Figure", "matplotlib.axes.Axes"]:
    """Plot an axis-aligned section of the mesh: cells and grid lines by origin.

    Every grid line of the two in-plane axes is drawn in the style of
    the rule that placed it (see the legend); lines that are graded fill
    between those planes are hairlines, absorber cells are hatched, and
    a plane holding a conductor edge with a field singularity carries
    red markers at its ends.  Pass the model as ``geometry`` to overlay
    the exact section outline.

    The fill answers one of three questions.  ``"coverage"`` (default)
    shows how much of each cell is conductor: the exact PEC-covered
    area of the primal faces normal to the cut, in the node plane
    nearest to it, as measured by the sub-cell classifier — every cell
    in the colour of its classified material, blended towards PEC grey
    by that share.  ``"material"`` shows the cell classification: the
    material containing each cell's centre, on the real cell size.  It
    is the staircase baseline the sub-cell values override on every
    cut cell, not the accuracy of the discretisation, and thin sheets
    do not appear in it.  ``"conformal"`` shows the dual faces of the
    edges normal to the cut — one tile per node, bounded by the cell
    midpoints, so the tiles sit half a cell off the grid lines — each
    coloured by the area-weighted permittivity that enters the
    electric material matrix for that edge (0 = PEC), with a colour
    bar.  Around a conductor the masked tiles reach one node beyond
    the contour: edges running along a conductor surface are held at
    its potential.

    Parameters
    ----------
    mesh : Mesh
        The mesh to draw.  Its ``planes`` record supplies the line
        styles; a mesh without one (built from a grid) draws every line
        as graded fill.  The ``"conformal"`` fill and the edge layer
        need the sub-cell data of a mesh built by ``Mesh.from_geometry``.
    normal : str
        Normal axis of the cutting plane: ``'x'``, ``'y'``, or ``'z'``.
    position : float
        Position along the normal axis in metres.  The cell layer
        containing it is shown; the edge layer takes the node plane
        nearest to it.
    geometry : GeometryModel or None, optional
        Overlay the exact section outline of this model (via
        :func:`~magnelio.plots.plot_cross_section` with ``fill=False``).
    scale_mm : bool, optional
        Display axes in millimetres (default) or metres.
    flip : bool, optional
        Swap the horizontal and vertical axes.
    ax : matplotlib.axes.Axes or None, optional
        Existing axes to draw into.  A new figure is created when *None*.
    title : str or None, optional
        Plot title.  Defaults to ``"Mesh section at <axis> = <pos> <unit>"``.
    fill : {"coverage", "material", "conformal", None}, optional
        ``"coverage"`` (default) shades every cell by its exact
        PEC-covered area; ``"material"`` colours every cell by its
        classified material; ``"conformal"`` colours the dual-face
        tiles of the normal edges by their area-weighted permittivity,
        with a colour bar; ``None`` draws the lines only.
    edges : bool, optional
        Add the in-plane primal edges of the nearest node plane:
        PEC-masked edges dark, edges partly inside PEC orange (the more
        metal, the stronger), edges borrowed out by the enlarged-cell
        technique with a red cross.  Free edges are not drawn.  Default
        False.
    legend : bool, optional
        Add a legend of the plane kinds and edge classes present
        (default True).

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Raises
    ------
    ValueError
        For an unknown ``normal`` or ``fill``, or when ``fill="coverage"``,
        ``fill="conformal"`` or ``edges=True`` is asked of a mesh
        without sub-cell data.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.collections import LineCollection  # noqa: PLC0415
    from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: PLC0415
    from matplotlib.lines import Line2D  # noqa: PLC0415
    from matplotlib.patches import Patch  # noqa: PLC0415

    from magnelio.post._colors import material_color  # noqa: PLC0415
    from magnelio.post.plot_geometry import (  # noqa: PLC0415
        _AXIS_INDEX,
        _AXIS_LABELS,
        plot_cross_section,
    )

    if normal not in _AXIS_LABELS:
        raise ValueError(f"normal must be 'x', 'y', or 'z'; got {normal!r}")
    if fill not in _FILL_MODES:
        raise ValueError(f"fill must be 'coverage', 'material', 'conformal' or None; got {fill!r}")
    no_subcell = (
        "mesh carries no sub-cell data (built from a grid?): fill='coverage', "
        "fill='conformal' and edges=True need a mesh from Mesh.from_geometry"
    )
    if (fill == "conformal" or edges) and mesh.edge_material is None:
        raise ValueError(no_subcell)
    if fill == "coverage" and (
        mesh.face_material is None or getattr(mesh.face_material, "A_face_pec", None) is None
    ):
        raise ValueError(no_subcell)

    scale = 1e3 if scale_mm else 1.0
    unit = "mm" if scale_mm else "m"
    h_label, v_label = _AXIS_LABELS[normal]
    if flip:
        h_label, v_label = v_label, h_label

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig: matplotlib.figure.Figure = ax.figure  # pyright: ignore[reportAssignmentType]

    grid = mesh.grid
    h_nodes = np.asarray(getattr(grid, h_label), dtype=float)
    v_nodes = np.asarray(getattr(grid, v_label), dtype=float)
    n_nodes = np.asarray(getattr(grid, normal), dtype=float)
    h_lo, h_hi = h_nodes[0] * scale, h_nodes[-1] * scale
    v_lo, v_hi = v_nodes[0] * scale, v_nodes[-1] * scale
    k = int(np.clip(np.searchsorted(n_nodes, position, side="right") - 1, 0, len(n_nodes) - 2))
    k_n = int(np.argmin(np.abs(n_nodes - position)))

    # -- fill -----------------------------------------------------------------------
    if fill == "material":
        slab = np.take(mesh.material_id, k, axis=_AXIS_INDEX[normal])
        # ``slab`` axes follow the unflipped (u, v) order of _AXIS_LABELS.
        cells = slab if flip else slab.T  # pcolormesh wants (n_v, n_h)
        ids = [int(i) for i in np.unique(cells)]
        colors = [material_color(mesh.material_library.get(i)) for i in ids]
        lookup = {mid: n for n, mid in enumerate(ids)}
        index = np.vectorize(lookup.get)(cells)
        ax.pcolormesh(
            h_nodes * scale,
            v_nodes * scale,
            index,
            cmap=ListedColormap(colors),
            norm=BoundaryNorm(np.arange(len(ids) + 1) - 0.5, len(ids)),
            shading="flat",
            edgecolors="none",
            zorder=0.5,
        )
    elif fill == "coverage":
        _fraction, rgb = _coverage_tiles(mesh, normal, k_n, k)
        ax.pcolormesh(
            h_nodes * scale,
            v_nodes * scale,
            rgb if flip else np.transpose(rgb, (1, 0, 2)),
            shading="flat",
            edgecolors="none",
            zorder=0.5,
        )
    elif fill == "conformal":
        tiles, eps_max = _conformal_tiles(mesh, normal, k)
        cmap, norm = _eps_colormap(eps_max)
        values = tiles if flip else tiles.T
        mappable = ax.pcolormesh(
            _dual_bounds(h_nodes) * scale,
            _dual_bounds(v_nodes) * scale,
            values,
            cmap=cmap,
            norm=norm,
            shading="flat",
            edgecolors="none",
            zorder=0.5,
        )
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("ε̄ (area-weighted, 0 = PEC)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    planes = getattr(mesh, "planes", None)
    kinds_present: set[str] = set()
    has_graded = False
    has_singular = False
    has_pml = False

    # -- absorber cells ---------------------------------------------------------
    if planes is not None:
        pml = planes.pml_cells
        for label, nodes, span in ((h_label, h_nodes, ax.axvspan), (v_label, v_nodes, ax.axhspan)):
            n_min = pml.get(f"{label}min", 0)
            n_max = pml.get(f"{label}max", 0)
            if n_min:
                span(nodes[0] * scale, nodes[n_min] * scale, zorder=0.7, **_PML_STYLE)
                has_pml = True
            if n_max:
                span(nodes[-1 - n_max] * scale, nodes[-1] * scale, zorder=0.7, **_PML_STYLE)
                has_pml = True

    # -- grid lines ---------------------------------------------------------------
    for label, nodes, lines, other_lo, other_hi in (
        (h_label, h_nodes, ax.vlines, v_lo, v_hi),
        (v_label, v_nodes, ax.hlines, h_lo, h_hi),
    ):
        records = planes.records(label) if planes is not None else ()
        plane_nodes = {r.node for r in records if r.node is not None}
        graded = [n * scale for i, n in enumerate(nodes) if i not in plane_nodes]
        if graded:
            has_graded = True
            lines(graded, other_lo, other_hi, zorder=1.5, **_GRADED_STYLE)
        for rec in records:
            kind, style = _style_of(rec)
            kinds_present.add(kind)
            pos = (rec.moved_to if rec.moved_to is not None else rec.position) * scale
            lines([pos], other_lo, other_hi, zorder=2.0, **style)
            if rec.singular:
                has_singular = True
                if lines is ax.vlines:
                    ax.plot(
                        [pos, pos],
                        [other_lo, other_hi],
                        linestyle="none",
                        marker="^",
                        markersize=4,
                        color=_SINGULAR_COLOR,
                        clip_on=False,
                        zorder=3,
                    )
                else:
                    ax.plot(
                        [other_lo, other_hi],
                        [pos, pos],
                        linestyle="none",
                        marker=">",
                        markersize=4,
                        color=_SINGULAR_COLOR,
                        clip_on=False,
                        zorder=3,
                    )

    # -- edge layer -----------------------------------------------------------------
    edge_classes: dict[str, bool] = {}
    if edges:
        classes = _section_edges(mesh, normal, k_n)
        cols = [1, 0] if flip else [0, 1]
        if len(classes["masked"]):
            ax.add_collection(
                LineCollection(
                    classes["masked"][:, :, cols] * scale, zorder=2.2, **_MASKED_EDGE_STYLE
                )
            )
            edge_classes["masked"] = True
        if len(classes["partial"]):
            alpha = 0.35 + 0.65 * (1.0 - classes["f_L"])
            rgba = np.column_stack([np.tile(_PARTIAL_EDGE_COLOR, (alpha.size, 1)), alpha])
            ax.add_collection(
                LineCollection(
                    classes["partial"][:, :, cols] * scale,
                    colors=rgba,
                    linewidths=_PARTIAL_EDGE_WIDTH,
                    zorder=2.2,
                )
            )
            edge_classes["partial"] = True
        if len(classes["borrowed"]):
            mid = classes["borrowed"][:, cols] * scale
            ax.plot(mid[:, 0], mid[:, 1], zorder=2.4, **_BORROWED_EDGE_STYLE)
            edge_classes["borrowed"] = True

    # -- geometry outline / axes cosmetics --------------------------------------
    if title is None:
        title = f"Mesh section at {normal} = {position * scale:.2f} {unit}"
        if edges or fill == "coverage":
            title += f" (node plane {n_nodes[k_n] * scale:.2f} {unit})"
    if geometry is not None:
        plot_cross_section(
            geometry,
            normal,
            position,
            ax=ax,
            mesh=None,
            scale_mm=scale_mm,
            flip=flip,
            fill=False,
            title=title,
        )
    else:
        ax.set_xlabel(f"{h_label} [{unit}]")
        ax.set_ylabel(f"{v_label} [{unit}]")
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
    ax.set_xlim(h_lo, h_hi)
    ax.set_ylim(v_lo, v_hi)

    # -- legend --------------------------------------------------------------------
    if legend:
        handles = [
            Line2D([], [], label=_KIND_LABEL[kind], **_PLANE_STYLE[kind])
            for kind in _KIND_PRIORITY
            if kind in kinds_present
        ]
        if has_graded:
            handles.append(Line2D([], [], label="graded fill", **_GRADED_STYLE))
        if has_singular:
            handles.append(
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker="^",
                    color=_SINGULAR_COLOR,
                    markersize=4,
                    label="singular edge",
                )
            )
        if edge_classes.get("masked"):
            handles.append(Line2D([], [], label="PEC-masked edge", **_MASKED_EDGE_STYLE))
        if edge_classes.get("partial"):
            handles.append(
                Line2D(
                    [],
                    [],
                    color=_PARTIAL_EDGE_COLOR,
                    linewidth=_PARTIAL_EDGE_WIDTH,
                    label="partially in PEC (0 < f_L < 1)",
                )
            )
        if edge_classes.get("borrowed"):
            handles.append(
                Line2D([], [], label="borrowed edge (enlarged cell)", **_BORROWED_EDGE_STYLE)
            )
        if has_pml:
            handles.append(Patch(label="absorber cells", **_PML_STYLE))
        if handles:
            ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.9)

    return fig, ax


__all__ = ["plot_mesh_section"]
