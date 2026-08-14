"""Per-axis coordinate-range resolution.

Shared vocabulary behind every ``from_ranges`` constructor
(:meth:`~magnelio.geo.Brick.from_ranges`, the monitor and source
``from_ranges`` classmethods): one axis is described by up to two of
its three keywords — lower bound (``x1``), upper bound (``x2``) and
extent (``dx``) — and resolves to a normalised ``(min, max)`` pair.
"""

from __future__ import annotations


def axis_range(
    axis: str, lo, hi, delta, *, required: bool = True
) -> tuple[float | None, float | None]:
    """Resolve one axis of a ``from_ranges`` spelling to ``(min, max)``.

    With ``required=True`` (the :class:`~magnelio.geo.Brick` contract)
    exactly two of the three keywords must be given; the third is
    redundant and is rejected rather than checked for consistency, so a
    typo cannot hide behind an agreeing pair.

    With ``required=False`` (the monitor/source contract, where an axis
    may be open toward the domain boundary) an axis may also be given
    nothing at all — it resolves to ``(None, None)``, i.e. unbounded on
    both sides — or a single bound, which leaves the other side ``None``.
    An extent alone is rejected in both modes: it anchors nothing.
    """
    keywords = ((f"{axis}1", lo), (f"{axis}2", hi), (f"d{axis}", delta))
    given = [n for n, v in keywords if v is not None]
    if required and len(given) != 2:
        raise ValueError(
            f"from_ranges needs exactly two of {axis}1, {axis}2, d{axis} "
            f"for the {axis} axis, got {len(given)} ({', '.join(given) or 'none'})."
        )
    if not required:
        if len(given) > 2:
            raise ValueError(
                f"from_ranges takes at most two of {axis}1, {axis}2, d{axis} "
                f"for the {axis} axis, got all three."
            )
        if len(given) == 0:
            return (None, None)
        if len(given) == 1:
            if delta is not None:
                raise ValueError(
                    f"from_ranges got d{axis} alone for the {axis} axis — an "
                    f"extent needs {axis}1 or {axis}2 to anchor it."
                )
            return (float(lo), None) if lo is not None else (None, float(hi))
    if delta is None:
        low, high = float(lo), float(hi)
    elif hi is None:
        low, high = float(lo), float(lo) + float(delta)
    else:
        low, high = float(hi) - float(delta), float(hi)
    return (low, high) if low <= high else (high, low)


def corners_from_ranges(x1, x2, dx, y1, y2, dy, z1, z2, dz) -> tuple[tuple, tuple]:
    """Resolve the lenient (monitor/source) form to two corner points.

    Unbounded sides come back as ``None`` components, the spelling the
    ``corners=`` parameters accept for "up to the domain boundary".
    """
    rx = axis_range("x", x1, x2, dx, required=False)
    ry = axis_range("y", y1, y2, dy, required=False)
    rz = axis_range("z", z1, z2, dz, required=False)
    return (rx[0], ry[0], rz[0]), (rx[1], ry[1], rz[1])
