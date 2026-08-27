"""2D view of the mesh itself: cells by material, grid lines by origin.

Main entry point:
    plot_mesh_section — render an axis-aligned slice through a Mesh

Where :func:`~magnelio.post.plot_geometry.plot_cross_section` draws the
*geometry* and optionally the grid on top, this module draws the
*discretisation*: the material each cell was filled with on the real
node coordinates, and every grid line in the style of the rule that
placed it (material face, bounding-box extent, geometry edge, thin
sheet, wire, symmetry face, forced position) — with the graded fill
nodes as hairlines and the absorber cells shaded.  The geometry section
outline can be overlaid so that the staircase and the exact contour are
read against each other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from magnelio.geo import GeometryModel
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


def _style_of(record) -> tuple[str, dict]:
    kinds = record.kinds
    for kind in _KIND_PRIORITY:
        if kind in kinds:
            return kind, _PLANE_STYLE[kind]
    return "extent", _PLANE_STYLE["extent"]


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
    fill: str | None = "material",
    legend: bool = True,
) -> tuple["matplotlib.figure.Figure", "matplotlib.axes.Axes"]:
    """Plot an axis-aligned section of the mesh: cells and grid lines by origin.

    Every grid line of the two in-plane axes is drawn in the style of
    the rule that placed it (see the legend); lines that are graded fill
    between those planes are hairlines, absorber cells are hatched, and
    a plane holding a conductor edge with a field singularity carries
    red markers at its ends.  With ``fill="material"`` the cells of the
    section are coloured by the material the mesher filled them with,
    on the real (non-uniform) node coordinates.  Pass the model as
    ``geometry`` to overlay the exact section outline.

    Parameters
    ----------
    mesh : Mesh
        The mesh to draw.  Its ``planes`` record supplies the line
        styles; a mesh without one (built from a grid) draws every line
        as graded fill.
    normal : str
        Normal axis of the cutting plane: ``'x'``, ``'y'``, or ``'z'``.
    position : float
        Position along the normal axis in metres.  The cell layer
        containing it is shown.
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
    fill : {"material", None}, optional
        ``"material"`` (default) colours every cell by its material;
        ``None`` draws the lines only.
    legend : bool, optional
        Add a legend of the plane kinds present (default True).

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
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
    if fill not in (None, "material"):
        raise ValueError(f"fill must be 'material' or None; got {fill!r}")

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

    # -- cell fill by material ------------------------------------------------
    if fill == "material":
        k = int(np.clip(np.searchsorted(n_nodes, position, side="right") - 1, 0, len(n_nodes) - 2))
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

    # -- geometry outline / axes cosmetics --------------------------------------
    if title is None:
        title = f"Mesh section at {normal} = {position * scale:.2f} {unit}"
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
        if has_pml:
            handles.append(Patch(label="absorber cells", **_PML_STYLE))
        if handles:
            ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.9)

    return fig, ax


__all__ = ["plot_mesh_section"]
