"""
Unified 2D field plotting: scalar colour maps, contour fills, and quiver plots.

This module provides the shared rendering core used by both monitor
``plot()`` methods and port ``ModeResult.plot()``.  Geometry overlays
(3D cross-sections or 2D port geometries) are handled via a small
tagged-union dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure


# ===================================================================
# Geometry overlay types
# ===================================================================


@dataclass
class CrossSectionOverlay:
    """Overlay a 3D geometry cross-section on the plot.

    ``swap_axes`` marks that the host plot's (horizontal, vertical)
    axes are the *descending* pair of global axes (e.g. a port plane
    whose inward-pointing ``u x v`` convention yields ``u=z, v=y``);
    the cross-section, which always slices in ascending order, is then
    drawn flipped.  It combines with the plot's own ``flip`` as XOR.

    ``mirrors`` lists the in-plane symmetry planes as
    ``(slot, wall_m, at_low)`` — slot 0/1 is the first/second in-plane
    axis in ascending world order.  The overlay is then drawn once per
    mirror image, each clipped to its half-space and reflected via an
    artist transform: the display always shows what the solver saw —
    the simulated half plus its mirror — for full and half-modelled
    geometry alike.
    """

    geometry: object  # list of CSG shapes / GeometryModel
    normal: str  # "x", "y", or "z"
    position: float  # position along the normal axis [m]
    swap_axes: bool = False
    mirrors: tuple = ()  # Design: DD-154 (symmetry-plane mirroring)


GeometryOverlay = CrossSectionOverlay | None


def _draw_cross_section_image(
    overlay: CrossSectionOverlay,
    ax,
    scale_mm: bool,
    flip: bool,
    combo: tuple,
) -> None:
    """Draw one clipped (and possibly mirrored) cross-section image.

    ``combo`` selects, per overlay mirror, whether this image is the
    reflected copy.  The reflection is an artist transform (no geometry
    is rebuilt in mirrored coordinates) and the clip box keeps every
    image inside its own half-space, so an asymmetric far half of a
    fully modelled geometry never shows.
    """
    from matplotlib.patches import Rectangle  # noqa: PLC0415
    from matplotlib.transforms import Affine2D  # noqa: PLC0415

    eff_flip = flip != overlay.swap_axes
    sc = 1e3 if scale_mm else 1.0
    big = 1e6
    n_lines = len(ax.lines)
    n_patches = len(ax.patches)
    n_colls = len(ax.collections)
    overlay.geometry.plot_cross_section(
        overlay.normal,
        overlay.position,
        scale_mm=scale_mm,
        flip=eff_flip,
        ax=ax,
    )
    if not overlay.mirrors:
        return
    new_artists = (
        list(ax.lines[n_lines:]) + list(ax.patches[n_patches:]) + list(ax.collections[n_colls:])
    )
    tr = Affine2D()
    clip_x = [-big, big]
    clip_y = [-big, big]
    for (slot, wall, at_low), mirrored in zip(overlay.mirrors, combo):
        horizontal = (slot == 0) != eff_flip
        w = wall * sc
        lo, hi = (w, big) if at_low else (-big, w)  # the simulated side
        if mirrored:
            lo, hi = 2.0 * w - hi, 2.0 * w - lo
            if horizontal:
                tr = tr + Affine2D().scale(-1.0, 1.0).translate(2.0 * w, 0.0)
            else:
                tr = tr + Affine2D().scale(1.0, -1.0).translate(0.0, 2.0 * w)
        rng = clip_x if horizontal else clip_y
        rng[0], rng[1] = max(rng[0], lo), min(rng[1], hi)
    clip_rect = Rectangle(
        (clip_x[0], clip_y[0]),
        clip_x[1] - clip_x[0],
        clip_y[1] - clip_y[0],
        transform=ax.transData,
    )
    for a in new_artists:
        a.set_transform(tr + ax.transData)
        a.set_clip_path(clip_rect)


def render_geometry_overlay(
    overlay: GeometryOverlay,
    *,
    ax: "matplotlib.axes.Axes",
    scale_mm: bool = True,
    flip: bool = False,
) -> None:
    """Render a geometry overlay onto *ax*, silently skipping on error."""
    if overlay is None:
        return
    try:
        if isinstance(overlay, CrossSectionOverlay):
            combos = [()]
            for _ in overlay.mirrors:
                combos = [c + (False,) for c in combos] + [c + (True,) for c in combos]
            for combo in combos:
                _draw_cross_section_image(overlay, ax, scale_mm, flip, combo)
    except Exception:
        pass


# ===================================================================
# Scalar field plot (pcolormesh / contourf)
# ===================================================================


def plot_field_scalar(
    xc: np.ndarray,
    yc: np.ndarray,
    values: np.ndarray,
    *,
    xlabel: str = "u",
    ylabel: str = "v",
    title: str = "",
    clabel: str = "",
    ax: "matplotlib.axes.Axes | None" = None,
    scale_mm: bool = True,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = False,
    plot_type: str = "color",
    contour_levels: int = 16,
    flip: bool = False,
    geometry: GeometryOverlay = None,
) -> tuple["matplotlib.figure.Figure", "matplotlib.axes.Axes"]:
    """Scalar 2D field plot (pcolormesh or contourf).

    Parameters
    ----------
    xc, yc : np.ndarray
        Cell-centre coordinates of the two in-plane axes.
    values : np.ndarray
        Real-valued 2D array, shape ``(len(xc), len(yc))``.
    xlabel, ylabel : str
        Axis labels (without unit suffix).
    title : str
        Axes title.
    clabel : str
        Colour-bar label.
    scale_mm : bool
        If True, multiply coordinates by 1e3 and label in mm.
    cmap : str
        Matplotlib colourmap.
    vmin, vmax : float or None
        Explicit colour limits.
    symmetric : bool
        If True and vmin/vmax not set, use symmetric limits ``[-M, M]``.
    plot_type : str
        ``"color"`` (pcolormesh) or ``"contour"`` (filled contours).
    contour_levels : int
        Number of contour levels (contour mode only).
    flip : bool
        Swap horizontal and vertical axes.
    geometry : GeometryOverlay
        Optional geometry overlay.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    sc = 1e3 if scale_mm else 1.0
    unit = "mm" if scale_mm else "m"

    if flip:
        xc, yc = yc, xc
        xlabel, ylabel = ylabel, xlabel
        values = values.T

    # Auto colour limits
    if vmax is None and symmetric:
        vmax = float(np.max(np.abs(values))) if np.any(values != 0) else 1.0
    if vmin is None and symmetric:
        vmin = -vmax

    if plot_type == "color":
        X, Y = np.meshgrid(xc * sc, yc * sc, indexing="ij")
        mesh_kw: dict = dict(cmap=cmap, shading="auto")
        if vmin is not None:
            mesh_kw["vmin"] = vmin
        if vmax is not None:
            mesh_kw["vmax"] = vmax
        pcm = ax.pcolormesh(X, Y, values, **mesh_kw)
        fig.colorbar(pcm, ax=ax, label=clabel)

    elif plot_type == "contour":
        X, Y = np.meshgrid(xc * sc, yc * sc, indexing="ij")
        kw: dict = dict(levels=contour_levels, cmap=cmap, alpha=0.80)
        if vmin is not None and vmax is not None:
            kw["levels"] = np.linspace(vmin, vmax, contour_levels)
        cf = ax.contourf(X, Y, values, **kw)
        ax.contour(X, Y, values, levels=cf.levels, colors="k", linewidths=0.3, alpha=0.35)
        fig.colorbar(cf, ax=ax, label=clabel)

    else:
        raise ValueError(f"plot_type must be 'color' or 'contour'; got {plot_type!r}")

    render_geometry_overlay(geometry, ax=ax, scale_mm=scale_mm, flip=flip)

    ax.set_xlim(xc[0] * sc, xc[-1] * sc)
    ax.set_ylim(yc[0] * sc, yc[-1] * sc)
    ax.set_xlabel(f"{xlabel} [{unit}]")
    ax.set_ylabel(f"{ylabel} [{unit}]")
    ax.set_title(title)
    ax.set_aspect("equal")
    return fig, ax


# ===================================================================
# Vector field plot (quiver)
# ===================================================================


def _draw_normal_markers(
    ax,
    X: np.ndarray,
    Y: np.ndarray,
    ws: np.ndarray,
    mag: np.ndarray,
    dot_mask: np.ndarray,
    *,
    cmap: str,
    norm,
    wlabel: str,
) -> None:
    """Draw ⊙/⊗ markers where the field is dominated by the normal component.

    Each marker is a filled circle coloured by the same magnitude norm as
    the quiver arrows; the overlaid glyph encodes the sign of the normal
    component along the *positive* normal axis (⊙ = ``+wlabel``,
    ⊗ = ``-wlabel``) — deliberately axis-referenced rather than
    "towards the viewer", which would depend on axis handedness and
    ``flip``.
    """
    from matplotlib.lines import Line2D  # noqa: PLC0415

    for positive, glyph in ((True, "."), (False, "x")):
        m = dot_mask & ((ws > 0) if positive else (ws < 0))
        if not np.any(m):
            continue
        ax.scatter(
            X[m],
            Y[m],
            c=mag[m],
            cmap=cmap,
            norm=norm,
            marker="o",
            s=40,
            edgecolors="k",
            linewidths=0.5,
            zorder=4,
        )
        ax.scatter(
            X[m],
            Y[m],
            c="k",
            marker=glyph,
            s=8 if glyph == "." else 14,
            linewidths=0.8,
            zorder=5,
        )

    handles = [
        Line2D(
            [],
            [],
            linestyle="",
            marker="o",
            markerfacecolor="none",
            markeredgecolor="k",
            markersize=7,
            label=f"{sym} {sign}{wlabel}",
        )
        for sym, sign in (("⊙", "+"), ("⊗", "−"))
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.7)


def plot_field_vector(
    xc: np.ndarray,
    yc: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    w: np.ndarray | None = None,
    xlabel: str = "u",
    ylabel: str = "v",
    wlabel: str = "n",
    title: str = "",
    clabel: str | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
    scale_mm: bool = True,
    cmap: str = "viridis",
    density: int = 20,
    normalize_arrows: bool = False,
    vmax: float | None = None,
    threshold: float = 0.0,
    auto_scale: bool = True,
    quiver_scale: float | None = None,
    flip: bool = False,
    geometry: GeometryOverlay = None,
) -> tuple["matplotlib.figure.Figure", "matplotlib.axes.Axes"]:
    """Quiver plot for a 2D slice of a vector field.

    Arrows show the in-plane vector components.  When the out-of-plane
    component *w* is given, arrow colour encodes the full 3D magnitude,
    and grid points whose vector tilts out of the plane by more than
    ~72° (``|w| >= 3x`` the in-plane part, at significant magnitude)
    are drawn as circle markers instead of unreadable foreshortened
    arrows: a filled circle with a centre dot (⊙) where the field
    points along the positive normal axis, with a cross (⊗) along the
    negative one.  Without *w* the plot shows the in-plane projection
    only and the colour bar is labelled accordingly.

    Parameters
    ----------
    xc, yc : np.ndarray
        Cell-centre coordinates of the two in-plane axes.
    u, v : np.ndarray
        In-plane vector components, shape ``(len(xc), len(yc))``.
    w : np.ndarray or None
        Out-of-plane (normal) component on the same grid.  Enables the
        full-magnitude colouring and the ⊙/⊗ markers.
    xlabel, ylabel : str
        In-plane axis labels (without unit suffix).
    wlabel : str
        Name of the normal axis (e.g. ``"y"``), used in the ⊙/⊗
        marker legend.
    clabel : str or None
        Colour-bar label.  Default: ``"Field magnitude"`` when *w* is
        given, ``"In-plane field magnitude"`` otherwise.
    density : int
        Target number of arrows per axis (subsamples grid).
    normalize_arrows : bool
        If True, arrows have unit length and colour encodes magnitude
        (port-mode style).  If False, arrow length is proportional to
        field strength (monitor style).
    vmax : float or None
        Clip arrow length and colour at this magnitude.
    threshold : float
        Suppress arrows below this fraction of peak magnitude.  With
        *w*, ``max(threshold, 0.02)`` of the peak is also the
        significance floor below which no ⊙/⊗ marker is drawn.
    auto_scale : bool
        Compute the quiver scale so the peak magnitude (full 3D with
        *w*, in-plane without) maps to ~ one cell spacing; arrow
        length over colour then reads as the out-of-plane tilt.
    quiver_scale : float or None
        Explicit quiver ``scale`` override (e.g. for ``interact()``
        fixed-scale animations).  Overrides *auto_scale*.
    flip : bool
        Swap horizontal and vertical axes.
    geometry : GeometryOverlay
        Optional geometry overlay.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    sc = 1e3 if scale_mm else 1.0
    unit = "mm" if scale_mm else "m"

    if flip:
        xc, yc = yc, xc
        xlabel, ylabel = ylabel, xlabel
        u, v = v.T, u.T
        if w is not None:
            w = w.T

    # --- Subsample ---
    nx, ny = u.shape
    sx = max(1, nx // density)
    sy = max(1, ny // density)

    xs, ys = xc[::sx], yc[::sy]
    us, vs = u[::sx, ::sy], v[::sx, ::sy]
    ws = w[::sx, ::sy] if w is not None else None
    mag_in = np.sqrt(np.abs(us) ** 2 + np.abs(vs) ** 2)
    # Colour encodes the full 3D magnitude when the normal component is
    # known; arrow geometry (length, clipping, scale) always follows the
    # in-plane projection.
    mag = np.sqrt(mag_in**2 + np.abs(ws) ** 2) if ws is not None else mag_in

    # --- Clip to vmax ---
    if vmax is not None:
        safe_in = np.where(mag_in > 0, mag_in, 1.0)
        clip = np.where(mag_in > vmax, vmax / safe_in, 1.0)
        us = us * clip
        vs = vs * clip
        mag_in = np.minimum(mag_in, vmax)
        mag = np.minimum(mag, vmax)

    max_mag = float(np.max(mag)) if np.any(mag > 0) else 1.0
    max_mag_in = float(np.max(mag_in)) if np.any(mag_in > 0) else 1.0

    # --- Normal-dominated points -> ⊙/⊗ markers instead of arrows ---
    # Local tilt criterion: a vector pointing more than ~72 degrees out
    # of the plane (|w| >= 3x its in-plane part) projects to an arrow
    # too short to read — draw the marker instead.  Guarded by a global
    # significance floor so near-zero vectors stay blank.  A global
    # in-plane comparison would fail here: quiver auto-scales arrows to
    # the in-plane maximum, so a slice pierced almost at right angles
    # everywhere would still render as a full-length arrow picture.
    dot_mask = None
    if ws is not None:
        sig = max(threshold, 0.02)
        dot_mask = (np.abs(ws) >= 3.0 * mag_in) & (mag >= sig * max_mag)
        us = np.where(dot_mask, np.nan, us)
        vs = np.where(dot_mask, np.nan, vs)

    # --- Suppress weak arrows ---
    if threshold > 0:
        weak = mag < threshold * max_mag
        us = np.where(weak, np.nan, us)
        vs = np.where(weak, np.nan, vs)

    # --- Normalize arrows (port style) ---
    if normalize_arrows:
        safe = np.where(mag_in > 0, mag_in, 1.0)
        us = us / safe
        vs = vs / safe

    X, Y = np.meshgrid(xs * sc, ys * sc, indexing="ij")

    # --- Scale ---
    # With w, the scale references the full 3D maximum, so arrow length
    # over colour reads as the tilt everywhere: a field that mostly
    # pierces the slice keeps its small in-plane residues visually
    # small instead of being blown up to the in-plane maximum.
    if quiver_scale is not None:
        scale = quiver_scale
    elif auto_scale and not normalize_arrows:
        dx = float(np.mean(np.diff(xs))) * sc if len(xs) > 1 else 1.0
        scale_ref = max_mag if ws is not None else max_mag_in
        scale = scale_ref / dx if scale_ref > 0 else 1.0
    else:
        scale = None  # let matplotlib auto-scale

    norm = plt.Normalize(vmin=0.0, vmax=max_mag)
    quiver_kw: dict = dict(
        cmap=cmap,
        norm=norm,
        pivot="mid",
        zorder=3,
    )
    if scale is not None:
        quiver_kw["scale"] = scale
        quiver_kw["scale_units"] = "xy"
        quiver_kw["angles"] = "xy"

    Q = ax.quiver(X, Y, us, vs, mag, **quiver_kw)
    if clabel is None:
        clabel = "Field magnitude" if w is not None else "In-plane field magnitude"
    fig.colorbar(Q, ax=ax, label=clabel, fraction=0.046, pad=0.04)

    if dot_mask is not None and np.any(dot_mask):
        _draw_normal_markers(ax, X, Y, ws, mag, dot_mask, cmap=cmap, norm=norm, wlabel=wlabel)

    render_geometry_overlay(geometry, ax=ax, scale_mm=scale_mm, flip=flip)

    ax.set_xlim(xs[0] * sc, xs[-1] * sc)
    ax.set_ylim(ys[0] * sc, ys[-1] * sc)
    ax.set_xlabel(f"{xlabel} [{unit}]")
    ax.set_ylabel(f"{ylabel} [{unit}]")
    ax.set_title(title)
    ax.set_aspect("equal")
    return fig, ax
