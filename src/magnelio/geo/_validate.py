"""Argument checking for the geometry API.

Geometry objects are dataclasses, so Python accepts whatever is passed
and the annotations are documentation only.  A wrong argument therefore
survives construction and surfaces much later — in the bounding box, in
the CAD kernel, or as a silently misplaced solid.  The helpers here turn
those into an error at the call that caused them, naming the field and
what it expects.

Checks run in the constructor, so they must stay cheap: shape of the
argument, sign and finiteness of a number, type of an operand.  Anything
needing the kernel (self-intersection, a chamfer larger than its edge)
belongs where the kernel builds the shape.

This module sits at the bottom of the import graph and must not import
anything else from the package.
"""

# Design: DD-176 (check geometry arguments where they are written).

from __future__ import annotations

import math

_POINT = "an (x, y, z) point in meters"
_VECTOR = "an (x, y, z) vector in meters"


def _coords(value, what: str, expects: str) -> tuple[float, ...]:
    """Coerce *value* to a tuple of floats, or raise naming *what*."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{what} must be {expects}; got the string {value!r}.")
    try:
        coords = tuple(float(c) for c in value)
    except TypeError:
        hint = ""
        if isinstance(value, (int, float)):
            hint = (
                f"  A single number is not three coordinates — write, for "
                f"example, (0, 0, {value!r}) for the z component alone."
            )
        raise TypeError(f"{what} must be {expects}; got {value!r}.{hint}") from None
    except ValueError:
        raise TypeError(f"{what} must be {expects} of numbers; got {value!r}.") from None
    if not all(math.isfinite(c) for c in coords):
        raise ValueError(f"{what} must be finite; got {value!r}.")
    return coords


def point3(value, what: str) -> tuple[float, float, float]:
    """Return *value* as an ``(x, y, z)`` tuple of floats."""
    coords = _coords(value, what, _POINT)
    if len(coords) != 3:
        raise ValueError(f"{what} must be {_POINT}; got {len(coords)} coordinates: {value!r}.")
    return coords


def vector3(value, what: str, *, nonzero: bool = False) -> tuple[float, float, float]:
    """Return *value* as a ``(dx, dy, dz)`` tuple of floats.

    With *nonzero*, reject the zero vector — a direction that would give
    the kernel a degenerate solid to build.
    """
    coords = _coords(value, what, _VECTOR)
    if len(coords) != 3:
        raise ValueError(f"{what} must be {_VECTOR}; got {len(coords)} components: {value!r}.")
    if nonzero and not any(coords):
        raise ValueError(f"{what} must not be the zero vector — it gives no direction.")
    return coords


def point_list(value, what: str, *, dim: int, minimum: int) -> tuple[tuple[float, ...], ...]:
    """Return *value* as a tuple of *dim*-dimensional point tuples.

    Catches the two ways a point sequence goes wrong: a flat list of
    coordinates instead of a list of points, and points of the wrong
    dimension for the context (``(u, v)`` in a plane, ``(x, y, z)`` in
    space).
    """
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{what} must be a sequence of points; got the string {value!r}.")
    try:
        raw = list(value)
    except TypeError:
        raise TypeError(f"{what} must be a sequence of points; got {value!r}.") from None
    form = {2: "(u, v)", 3: "(x, y, z)"}[dim]
    points = []
    for index, entry in enumerate(raw):
        if isinstance(entry, (int, float)):
            raise TypeError(
                f"{what}: point {index} is the bare number {entry!r}, not a "
                f"point of the form {form}.  Pass a sequence of points, not "
                f"a flat list of coordinates."
            )
        coords = _coords(entry, f"{what}: point {index}", f"a point of the form {form} in meters")
        if len(coords) != dim:
            raise ValueError(
                f"{what}: point {index} has {len(coords)} coordinates, but "
                f"points of the form {form} are expected here; got {entry!r}."
            )
        points.append(coords)
    if len(points) < minimum:
        raise ValueError(f"{what} needs at least {minimum} points; got {len(points)}.")
    return tuple(points)


def _number(value, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{what} must be a number; got {value!r}.") from None
    if not math.isfinite(number):
        raise ValueError(f"{what} must be finite; got {value!r}.")
    return number


def finite(value, what: str) -> float:
    """Return *value* as a finite float."""
    return _number(value, what)


def positive(value, what: str) -> float:
    """Return *value* as a finite float, rejecting zero and negatives."""
    number = _number(value, what)
    if number <= 0.0:
        raise ValueError(f"{what} must be positive; got {value!r}.")
    return number


def nonnegative(value, what: str) -> float:
    """Return *value* as a finite float, rejecting negatives."""
    number = _number(value, what)
    if number < 0.0:
        raise ValueError(f"{what} must not be negative; got {value!r}.")
    return number


def nonzero(value, what: str) -> float:
    """Return *value* as a finite float, rejecting zero.

    For an extent whose sign carries a meaning — a height extruded along
    ``-axis``, a metallisation grown on the far side of its sheet —
    where only zero is degenerate.
    """
    number = _number(value, what)
    if number == 0.0:
        raise ValueError(f"{what} must not be zero; got {value!r}.")
    return number


def count(value, what: str, *, minimum: int = 1) -> int:
    """Return *value* as an int of at least *minimum*."""
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        else:
            raise TypeError(f"{what} must be a whole number; got {value!r}.")
    if value < minimum:
        raise ValueError(f"{what} must be at least {minimum}; got {value!r}.")
    return value


def operand(value, what: str):
    """Return *value* unchanged if it is a geometry object.

    Identified by the leaf of the geometry protocol every shape, sheet
    and wire implements, so a ``Shape`` import — which would invert the
    import graph — is not needed.
    """
    if hasattr(value, "_occ_shape") and hasattr(value, "_analytic_bbox"):
        return value
    hint = ""
    if isinstance(value, (list, tuple)):
        hint = (
            "  The operands are passed one by one, not as a list: write "
            "Union(a, b), not Union([a, b])."
        )
    raise TypeError(f"{what} must be a geometry object; got {type(value).__name__}.{hint}")
