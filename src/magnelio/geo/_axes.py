"""Axis-argument normalisation shared by primitives and transforms.

Every ``axis=`` parameter in the geometry API accepts either an axis
letter (``"x"``, ``"y"``, ``"z"``) or an arbitrary 3-vector; this module
canonicalises both forms to a unit direction tuple.
"""

from __future__ import annotations

import math

_AXIS_LETTERS = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}


def normalize_axis(axis, what: str = "axis") -> tuple[float, float, float]:
    """Return the unit direction for an axis letter or 3-vector.

    Parameters
    ----------
    axis : str or sequence of float
        ``"x"``/``"y"``/``"z"`` (case-insensitive) or any non-zero
        3-vector; the vector's length is ignored.
    what : str
        How to name the argument in an error message — the field or
        parameter it came from, e.g. ``"Cylinder.axis"``.

    Returns
    -------
    tuple of float
        Unit direction ``(dx, dy, dz)``.
    """
    if isinstance(axis, str):
        try:
            return _AXIS_LETTERS[axis.lower()]
        except KeyError:
            raise ValueError(f"{what} must be 'x', 'y', 'z' or a 3-vector; got {axis!r}") from None
    try:
        dx, dy, dz = (float(c) for c in axis)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be 'x', 'y', 'z' or a 3-vector; got {axis!r}") from None
    if not all(math.isfinite(c) for c in (dx, dy, dz)):
        raise ValueError(f"{what} must be finite; got {axis!r}")
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm == 0.0:
        raise ValueError(f"{what} vector must be non-zero")
    return (dx / norm, dy / norm, dz / norm)


def reference_dir(axis) -> tuple[float, float, float]:
    """The zero-angle direction for rotations measured about *axis*.

    Angles about an axis need a direction to count from.  This picks the
    first coordinate axis (x, then y, then z) that is not aligned with
    *axis* and squares it up against it, so ``'z'`` counts from ``+x``
    and ``'x'`` counts from ``+y``.  Together with ``axis x reference``
    as the quarter-turn direction the frame is right-handed, which makes
    a positive angle here the same rotation as a positive angle in
    :meth:`~magnelio.geo.Shape.rotated`.

    Both the CAD kernel frame and the analytic bounding boxes read the
    frame from here, so the two cannot drift apart.
    """
    d = normalize_axis(axis)
    candidate = min(_AXIS_LETTERS.values(), key=lambda e: abs(sum(a * b for a, b in zip(d, e))))
    projection = sum(a * b for a, b in zip(candidate, d))
    ref = tuple(c - projection * a for c, a in zip(candidate, d))
    norm = math.sqrt(sum(c * c for c in ref))
    return tuple(c / norm for c in ref)


def cross(a, b) -> tuple[float, float, float]:
    """Cross product of two 3-vectors."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )
