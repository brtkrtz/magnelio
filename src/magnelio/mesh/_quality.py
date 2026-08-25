"""
Mesh quality checks.

Warns the user if the mesh has problematic properties (grid gradient,
cell count, cells that cost time steps without buying resolution).
"""

from __future__ import annotations

import warnings

import numpy as np


def check_quality(mesh) -> None:
    """Run mesh quality checks and issue warnings for potential problems.

    Args:
        mesh: A :class:`~magnelio.mesh.mesher.Mesh` instance.
    """
    grid = mesh.grid

    dx = grid.dx
    dy = grid.dy
    dz = grid.dz

    h_min = min(dx.min(), dy.min(), dz.min())
    if h_min <= 0:
        raise ValueError(f"Mesh contains zero or negative cell size: {h_min}")

    n_cells = grid.n_cells
    if n_cells > 10_000_000:
        warnings.warn(
            f"Large mesh: {n_cells:,} cells. Simulation may require significant "
            "memory and time. Consider coarsening the mesh.",
            UserWarning,
            stacklevel=3,
        )

    # Check growth factor gradient in each axis
    for name, d in (("x", dx), ("y", dy), ("z", dz)):
        if len(d) > 1:
            ratios = d[1:] / d[:-1]
            if ratios.max() > 2.0 or ratios.min() < 0.5:
                # Not an accuracy limit.  The reflection off a grid
                # transition follows 2.5*(h_coarse/lambda)^2*(1 - 1/g^2)
                # (DD-105): the resolution dominates and the growth
                # factor saturates, so at the wavelength criterion even
                # g = 4 stays below -44 dB.  What a ratio above 2 does
                # mean is that a grading mesher stopped grading.
                warnings.warn(
                    f"Mesh growth factor in {name}-direction exceeds 2.0 "
                    f"(max ratio: {ratios.max():.2f}) — steeper than this "
                    "mesher grades. Expect a grid the geometry did not "
                    "ask for rather than a dispersion problem.",
                    UserWarning,
                    stacklevel=3,
                )


def check_grading_undershoot(
    grid_lines: dict[str, list[float]],
    axis_planes: dict[str, list[float]],
    axis_anchors: dict[str, set],
    h_fine_axis: dict[str, float],
    h_max: float,
    control,
    buffer_ends: dict[str, tuple[str, ...]] | None = None,
    ports_declared: bool = False,
    buffer_cells: int = 3,
    h_fine_planes: dict[str, list[float]] | None = None,
) -> None:
    """Warn when the time-step-setting cell is finer than anything asked for.

    The explicit time loop takes one global time step bounded by the
    *globally* smallest cell, so a cell that came out smaller than its
    interval requested costs steps everywhere and buys resolution
    nowhere.  The cause is that the cell count per interval is an
    integer -- most visibly in the symmetric graded subdivision, which
    takes the smallest count whose starting cell fits under ``h_fine``:
    missing that bound by a fraction of a percent forces one more cell
    and shrinks every cell in the interval.

    Only the interval holding the global minimum is reported — cells
    elsewhere do not bound the time step.  Anchor pairs (forced planes,
    thin sheets) are the user's own coordinates and are never reported,
    and neither is a cell already sitting on ``control.min_cell_size``.
    An axis whose fine size is not feature-driven at all
    (``h_fine >= h_max``) is skipped: there the wavelength criterion
    sets the cell count, and that is the user's accuracy choice, not
    slack to be reclaimed.  The one exception is a *buffered boundary
    interval* (DD-107 domain-face buffer): the three equidistant cells
    forced at a buffered face can undershoot the wavelength size on an
    otherwise wavelength-driven axis — a cell the user never asked
    for, reported with its own remedy.

    Parameters
    ----------
    grid_lines, axis_planes, axis_anchors, h_fine_axis
        Per-axis node positions, critical planes, user-anchored
        positions, and the feature-driven fine size, as built by
        :meth:`~magnelio.mesh.mesher.Mesh.from_geometry`.
    h_max : float
        The wavelength-driven bulk cell size.
    control : MeshControl
        Supplies the ``max_cell_size`` / ``min_cell_size`` clamps that
        also applied when the interval was subdivided.
    buffer_ends : dict, optional
        Per-axis tuple of buffered ends (``"lo"`` / ``"hi"``) as built
        by the mesher's port-buffer resolution; ``None`` means no
        buffer information (no buffer exemption applies).
    ports_declared : bool
        Whether the buffer set came from ports declared on the model
        (DD-109) rather than the buffer-all-faces fallback; selects
        the applicable remedy in the buffer warning.
    buffer_cells : int
        The mesher's equidistant-cell buffer count
        (``_BOUNDARY_BUFFER_CELLS``); a buffered interval is only
        reported when a plain fill would have held fewer cells than
        this — otherwise the undershoot is ordinary integer rounding,
        which the wavelength skip deliberately tolerates.
    h_fine_planes : dict, optional
        Per-axis fine size at each critical plane (the mesher's
        singularity refinement, which starts the grading at a
        conductor edge below ``h_fine`` by design); an interval is
        measured against the finer of its two ends.  ``None`` means
        ``h_fine_axis`` at every plane.
    """
    site = None  # (h_actual, axis, node_lo, node_hi)
    for axis in ("x", "y", "z"):
        arr = np.asarray(grid_lines[axis], dtype=float)
        if arr.size < 2:
            continue
        d = np.diff(arr)
        i = int(np.argmin(d))
        if site is None or d[i] < site[0]:
            site = (float(d[i]), axis, float(arr[i]), float(arr[i + 1]))

    if site is None:
        return
    h_actual, axis, node_lo, node_hi = site

    # A floor the user set themselves is not an undershoot.
    min_cell = control.min_cell_size or 0.0
    if min_cell > 0 and h_actual <= min_cell * (1.0 + 1e-9):
        return

    # An anchor pair is a user coordinate, kept verbatim by design.
    anchors = axis_anchors.get(axis, ())
    if node_lo in anchors and node_hi in anchors:
        return

    # Locate the critical interval that produced this cell, and rebuild
    # the h_fine it was subdivided with (mesher.py: per-interval clamps).
    planes = np.asarray(axis_planes[axis], dtype=float)
    k = int(np.searchsorted(planes, node_lo, side="right")) - 1
    if k < 0 or k + 1 >= planes.size:
        return
    p0, p1 = float(planes[k]), float(planes[k + 1])

    # Does the cell sit in a boundary interval whose domain face
    # carries the DD-107 buffer?
    buffered = set(buffer_ends.get(axis, ())) if buffer_ends else set()
    buffered_faces = []
    if k == 0 and "lo" in buffered:
        buffered_faces.append(f"{axis}min")
    if k + 2 == planes.size and "hi" in buffered:
        buffered_faces.append(f"{axis}max")

    # Wavelength-driven, not feature-driven: nothing to reclaim —
    # except the cells a domain-face buffer created on its own.
    wavelength_driven = h_fine_axis[axis] >= h_max * (1.0 - 1e-10)
    if wavelength_driven and not buffered_faces:
        return

    h_fine_here = h_fine_axis[axis]
    if h_fine_planes is not None and axis in h_fine_planes:
        fine = h_fine_planes[axis]
        if len(fine) == planes.size:
            h_fine_here = min(fine[k], fine[k + 1])
    h_wanted = min(h_fine_here, p1 - p0)
    if control.max_cell_size is not None:
        h_wanted = min(h_wanted, control.max_cell_size)
    h_wanted = max(h_wanted, min_cell)

    if h_actual >= h_wanted * 0.85:
        return

    if wavelength_driven:
        # The buffer is only at fault when it forced more cells than
        # the plain fill would have used; an interval that needs
        # ``buffer_cells`` or more anyway undershoots by ordinary
        # integer rounding, which stays skipped (the user's accuracy
        # choice, nothing to reclaim).
        n_plain = int(np.ceil((p1 - p0) / h_wanted * (1.0 - 1e-12)))
        if n_plain >= buffer_cells:
            return
        remedy = (
            "Declaring the analysis ports on the GeometryModel "
            "restricts the buffer to the port faces; alternatively "
            "widen this boundary interval."
            if not ports_declared
            else "Widen this boundary interval (the buffer is required by the port on this face)."
        )
        warnings.warn(
            f"Smallest cell {h_actual:.4g} m ({axis}, in the "
            f"{p1 - p0:.4g} m interval [{p0:.4g}, {p1:.4g}]) comes "
            f"from the equidistant-cell buffer at the "
            f"{' and '.join(buffered_faces)} domain face and is "
            f"{(1.0 - h_actual / h_wanted) * 100:.0f}% below the "
            f"{h_wanted:.4g} m this interval could host; it sets the "
            f"time step. {remedy}",
            UserWarning,
            stacklevel=3,
        )
        return

    warnings.warn(
        f"Smallest cell {h_actual:.4g} m ({axis}, in the {p1 - p0:.4g} m "
        f"interval [{p0:.4g}, {p1:.4g}]) is "
        f"{(1.0 - h_actual / h_wanted) * 100:.0f}% below the {h_wanted:.4g} m "
        f"this interval asked for, and it sets the time step. "
        f"MeshControl(min_cell_size={h_wanted:.3g}) removes the undershoot.",
        UserWarning,
        stacklevel=3,
    )
