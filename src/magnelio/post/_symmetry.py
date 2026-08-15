"""Symmetry-plane primitives shared by every plotting path (DD-154).

A model declared with symmetry planes is meshed and solved on the
reduced domain only.  Every user-facing picture, however, shows the
full structure: the simulated part plus its mirror images.  The
continuation rules are pure field theory and identical for a monitor
slice and for a port-mode profile, so they live here rather than in
either consumer.

:class:`MirrorSpec` names one plane; :func:`mirror_sign` gives the
component's continuation sign across it; :func:`mirror_extend` applies
it to one axis of a data array.  Monitors add their region bookkeeping
on top in :mod:`magnelio.monitors.base`; the port-mode report adds its
own window test in :mod:`magnelio.ports._modal.mode_report`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_AXES = ("x", "y", "z")


@dataclass(frozen=True)
class MirrorSpec:
    """One symmetry plane a plotted region touches.

    Attributes
    ----------
    axis : int
        World axis index (0/1/2) of the mirror normal.
    wall : float
        Mirror-plane position [m] — the *physical* wall: on a PEC face
        the outermost grid line, on a PMC face half the boundary cell
        outside it (where the natural magnetic wall of the staggered
        grid sits; after the mesher's pull-in that is exactly the
        declared plane).
    kind : str
        Wall type, ``"PEC"`` or ``"PMC"``.
    at_low : bool
        True when the mirror sits at the low end of the axis (the
        mirrored copy prepends).
    """

    axis: int
    wall: float
    kind: str
    at_low: bool


def mirror_spec_for_face(face: str, kind: str, axis_nodes: np.ndarray) -> MirrorSpec:
    """Build the spec for a declared symmetry *face* of the domain.

    *face* is a boundary name (``"ymin"``), *kind* its wall type, and
    *axis_nodes* the grid node coordinates along that face's axis.
    """
    axis = _AXES.index(face[0])
    at_low = face.endswith("min")
    n = np.asarray(axis_nodes, dtype=float)
    if kind == "PEC":
        wall = n[0] if at_low else n[-1]
    elif at_low:
        wall = n[0] - 0.5 * (n[1] - n[0])
    else:
        wall = n[-1] + 0.5 * (n[-1] - n[-2])
    return MirrorSpec(axis=axis, wall=float(wall), kind=kind, at_low=at_low)


def mirror_sign(field: str, comp_axis: int | None, mirror_axis: int, kind: str) -> float:
    """±1 continuation factor of a field component across a mirror.

    Across a magnetic (PMC) symmetry plane E continues like a polar
    vector (normal component odd, tangential even) and H like a
    pseudovector (normal even, tangential odd); across an electric
    (PEC) plane the roles swap.  Magnitudes (``comp_axis=None``) are
    always even.
    """
    if comp_axis is None:
        return 1.0
    is_normal = comp_axis == mirror_axis
    flips_normal = (kind == "PMC") if field == "E" else (kind == "PEC")
    return -1.0 if is_normal == flips_normal else 1.0


def mirror_extend(
    coords: np.ndarray,
    values: np.ndarray,
    spec: MirrorSpec,
    arr_axis: int,
    sign: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend one axis of a data array across a mirror plane.

    Returns the extended cell-centre coordinates and the values with
    the sign-weighted mirrored copy prepended (``at_low``) or appended.
    """
    reflected = 2.0 * spec.wall - coords[::-1]
    flipped = sign * np.flip(values, axis=arr_axis)
    if spec.at_low:
        return (
            np.concatenate([reflected, coords]),
            np.concatenate([flipped, values], axis=arr_axis),
        )
    return (
        np.concatenate([coords, reflected]),
        np.concatenate([values, flipped], axis=arr_axis),
    )
