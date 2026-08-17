"""Far-field pattern plots — polar cuts and the 3D radiation surface.

Computation-free drawing over plain arrays, per the library's two-tier
rule: :class:`~magnelio.post.FarFieldResult` (or any caller) computes
the pattern quantity and delegates here.  Both functions accept a
caller-made Axes of the matching projection, so multi-panel figures
compose the usual way.
"""

# Design: DD-174 (pattern plots).

from __future__ import annotations

import numpy as np


def _to_db(values: np.ndarray, floor_db: float) -> np.ndarray:
    floor_lin = 10.0 ** (floor_db / 10.0)
    return 10.0 * np.log10(np.maximum(np.asarray(values, dtype=float), floor_lin))


def plot_pattern_cut(
    angles: np.ndarray,
    values: np.ndarray,
    *,
    db: bool = True,
    floor_db: float = -40.0,
    ax=None,
    label: str | None = None,
    title: str | None = None,
):
    """Polar plot of one pattern cut.

    Follows the antenna-plot convention: the zero angle points up and
    angles run clockwise, so a θ-cut shows the zenith at the top.

    Parameters
    ----------
    angles : array_like
        Cut angles [rad].
    values : array_like
        Pattern quantity along the cut (linear, e.g. gain or
        directivity).
    db : bool, default True
        Radial axis in dB (with *floor_db*) instead of linear.
    floor_db : float, default -40.0
        Clip floor and radial-axis minimum for the dB display.
    label : str, optional
        Legend label for this trace.
    ax : matplotlib.projections.polar.PolarAxes, optional
        Polar axes to draw into; a new figure is created otherwise.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.projections.polar.PolarAxes
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    else:
        if getattr(ax, "name", None) != "polar":
            raise ValueError(
                "plot_pattern_cut needs a polar Axes; create one with "
                'plt.subplots(subplot_kw={"projection": "polar"}).'
            )
        fig = ax.get_figure()
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    angles = np.asarray(angles, dtype=float)
    if db:
        y = _to_db(values, floor_db)
        ax.plot(angles, y, label=label)
        top = float(np.max(y))
        ax.set_rlim(floor_db, top + 0.05 * (top - floor_db))
    else:
        ax.plot(angles, np.asarray(values, dtype=float), label=label)
    ax.grid(True, alpha=0.3)
    if label is not None:
        ax.legend(loc="lower left")
    if title is not None:
        ax.set_title(title)
    return fig, ax


def plot_pattern_3d(
    theta: np.ndarray,
    phi: np.ndarray,
    values: np.ndarray,
    *,
    db: bool = True,
    floor_db: float = -40.0,
    ax=None,
    cmap: str | None = None,
    title: str | None = None,
):
    """3D radiation surface: radius proportional to the pattern.

    In dB mode the radius is ``value_dB − floor_dB`` clipped at zero,
    so the floor collapses to the origin and nulls stay visible as
    indentations.

    Parameters
    ----------
    theta, phi : array_like
        Spherical angle grids [rad]; θ from +z, φ from +x.
    values : array_like
        Pattern quantity on the (θ, φ) grid (linear), shape
        ``(len(theta), len(phi))``.
    db : bool, default True
        Radius from the dB value (with *floor_db*) instead of linear.
    floor_db : float, default -40.0
        Radius origin for the dB display.
    cmap : str, optional
        Colormap for the radius shading (default ``"viridis"``).
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional
        3D axes to draw into; a new figure is created otherwise.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
    else:
        if getattr(ax, "name", None) != "3d":
            raise ValueError(
                'plot_pattern_3d needs a 3D Axes; create one with fig.add_subplot(projection="3d").'
            )
        fig = ax.get_figure()
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    values = np.asarray(values, dtype=float)
    if db:
        r = np.maximum(_to_db(values, floor_db) - floor_db, 0.0)
    else:
        r = np.maximum(values, 0.0)
    st, ct = np.sin(theta)[:, None], np.cos(theta)[:, None]
    x = r * st * np.cos(phi)[None, :]
    y = r * st * np.sin(phi)[None, :]
    z = r * ct
    r_max = float(np.max(r)) or 1.0
    norm = plt.Normalize(0.0, r_max)
    colors = plt.get_cmap(cmap or "viridis")(norm(r))
    ax.plot_surface(x, y, z, facecolors=colors, rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax.set_box_aspect((1, 1, 1))
    lim = r_max * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title)
    return fig, ax
