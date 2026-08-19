"""
Path — a fluent builder for chained curves.

:class:`Path` is sugar over :class:`~magnelio.geo.Curve` and
:meth:`~magnelio.geo.Curve.joined`: it remembers where the previous
segment ended, so a profile reads as the pen stroke that draws it instead
of as a list of segments with every interior point spelled twice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from magnelio.geo._validate import point3
from magnelio.geo.curves import _JOIN_RTOL, Curve


def _arc_midpoint_about(u_start, u_end, normal, center, radius):
    """Mid-arc point of the CCW arc about *normal*, from u_start to u_end.

    *u_start*/*u_end* are unit radial directions; the arc is the one
    swept counter-clockwise about *normal*, which is well defined even
    for diametrically opposite ends.
    """
    from magnelio.geo._axes import normalize_axis  # noqa: PLC0415

    n = normalize_axis(normal)
    for label, u in (("start", u_start), ("end", u_end)):
        out_of_plane = sum(a * b for a, b in zip(u, n))
        if abs(out_of_plane) > 1e-6:
            raise ValueError(
                f"arc_to(center=..., normal=...) needs both ends in the "
                f"plane through the centre perpendicular to the normal, "
                f"but the {label} point is off it by "
                f"{abs(out_of_plane) * radius:.3e} m."
            )
    cross = (
        u_start[1] * u_end[2] - u_start[2] * u_end[1],
        u_start[2] * u_end[0] - u_start[0] * u_end[2],
        u_start[0] * u_end[1] - u_start[1] * u_end[0],
    )
    sweep = math.atan2(
        sum(a * b for a, b in zip(n, cross)),
        sum(a * b for a, b in zip(u_start, u_end)),
    )
    if sweep <= 0.0:
        sweep += 2.0 * math.pi
    half = 0.5 * sweep
    # u_start is perpendicular to n, so Rodrigues reduces to this.
    n_cross_u = (
        n[1] * u_start[2] - n[2] * u_start[1],
        n[2] * u_start[0] - n[0] * u_start[2],
        n[0] * u_start[1] - n[1] * u_start[0],
    )
    u_mid = tuple(a * math.cos(half) + b * math.sin(half) for a, b in zip(u_start, n_cross_u))
    return tuple(o + radius * x for o, x in zip(center, u_mid))


@dataclass(frozen=True)
class Path:
    """A pen that draws a chained :class:`~magnelio.geo.Curve`.

    Start at a point, then append one segment per call; each segment
    begins where the previous one ended, so only its *end* (and whatever
    shapes it) has to be given.  Finish with :meth:`curve` for an open
    path or :meth:`closed` for a loop, which is what
    :meth:`~magnelio.geo.Curve.covered` needs to make a sheet.

    **A Path is immutable.**  Every segment call returns a new Path and
    leaves the receiver alone, so a common prefix can be branched into
    several outlines.

    Parameters
    ----------
    start : tuple of float
        The 3D point ``(x, y, z)`` the pen starts from [meters].

    Examples
    --------
    A slot outline with two rounded ends::

        outline = (
            Path((0.0, -1e-3, 0.0))
            .line_to((6e-3, -1e-3, 0.0))
            .arc_to((6e-3, 1e-3, 0.0), center=(6e-3, 0.0, 0.0))
            .line_to((0.0, 1e-3, 0.0))
            .arc_to((0.0, -1e-3, 0.0), center=(0.0, 0.0, 0.0))
            .closed()
        )
        slot = outline.covered().extruded(vector=(0, 0, t), material=copper)
    """

    start: tuple
    _segments: tuple = field(default_factory=tuple)

    def __post_init__(self):
        object.__setattr__(self, "start", point3(self.start, "Path(start)"))

    @property
    def current(self) -> tuple:
        """The point the next segment will start from [meters]."""
        if not self._segments:
            return self.start
        return self._segments[-1]._ends[1]

    def _extended(self, segment: Curve) -> "Path":
        return Path(self.start, (*self._segments, segment))

    # ── segments ──────────────────────────────────────────────────────

    def line_to(self, point) -> "Path":
        """Append a straight segment ending at *point*.

        Parameters
        ----------
        point : tuple of float
            End point ``(x, y, z)`` [meters].

        Returns
        -------
        Path
            A new Path ending at *point*.
        """
        return self._extended(Curve.polyline([self.current, point3(point, "line_to(point)")]))

    def arc_to(self, end, *, via=None, center=None, normal=None, major=False) -> "Path":
        """Append a circular arc ending at *end*.

        Give exactly one of *via* or *center* — two ways of pinning down
        which arc is meant:

        - *via* — a point the arc passes through.  Always unambiguous,
          and the form to reach for when the centre is not what you
          know.
        - *center* — the centre of the circle, which must be equidistant
          from the current point and *end*.  This is the form for a slice
          of a round part, where the axis is the given quantity.

        Two arcs join any pair of points on a circle, and in 3D a pair of
        diametrically opposite points does not even fix the plane.  With
        *center*, add *normal* to settle both at once: the arc then runs
        counter-clockwise about *normal*, so swapping the two endpoints
        gives the complementary arc.  Without *normal* the shorter arc is
        drawn (or the longer one with *major*), and diametrically
        opposite ends are rejected.

        Parameters
        ----------
        end : tuple of float
            End point ``(x, y, z)`` [meters].
        via : tuple of float, optional
            A point on the arc, between start and end.
        center : tuple of float, optional
            Centre of the arc's circle.
        normal : str or sequence of float, optional
            With *center*: the axis the arc turns about — ``'x'``,
            ``'y'``, ``'z'``, or any non-zero 3-vector.  The arc runs
            counter-clockwise about it, seen from the tip of the axis.
            Both endpoints must lie in the plane through *center*
            perpendicular to it.
        major : bool
            With *center* and no *normal*: take the long way round
            (default False).

        Returns
        -------
        Path
            A new Path ending at *end*.

        Raises
        ------
        ValueError
            If neither or both of *via* and *center* are given, if
            *center* is not equidistant from both endpoints, if the
            endpoints do not lie in the plane *normal* describes, or if
            they are diametrically opposite and no *normal* was given to
            say which arc is meant.

        Examples
        --------
        The rounded end of a slot, turning about ``z``::

            path.arc_to((0.0, -1e-3, 0.0), center=(0.0, 0.0, 0.0), normal="z")
        """
        if (via is None) == (center is None):
            raise ValueError("arc_to() takes exactly one of via= or center=.")
        if normal is not None and center is None:
            raise ValueError("arc_to(normal=...) only applies together with center=.")
        p_start, p_end = self.current, point3(end, "arc_to(end)")
        if via is not None:
            return self._extended(Curve.arc(p_start, point3(via, "arc_to(via)"), p_end))

        c = point3(center, "arc_to(center)")
        r_start, r_end = math.dist(c, p_start), math.dist(c, p_end)
        r_max = max(r_start, r_end)
        if r_max <= 0.0:
            raise ValueError("arc_to(center=...) needs a centre distinct from the arc endpoints.")
        if abs(r_start - r_end) > _JOIN_RTOL * r_max:
            raise ValueError(
                f"arc_to(center=...) needs a centre equidistant from both "
                f"ends, but it is {r_start:.6e} m from the start and "
                f"{r_end:.6e} m from the end."
            )
        radius = 0.5 * (r_start + r_end)
        u_start = tuple((a - b) / r_start for a, b in zip(p_start, c))
        u_end = tuple((a - b) / r_end for a, b in zip(p_end, c))

        if normal is not None:
            p_via = _arc_midpoint_about(u_start, u_end, normal, c, radius)
        else:
            # The bisector of the two radial directions points at the
            # middle of the short arc, and away from it for the long one.
            bisector = [a + b for a, b in zip(u_start, u_end)]
            length = math.sqrt(sum(x * x for x in bisector))
            if length <= _JOIN_RTOL:
                raise ValueError(
                    "arc_to(center=...) cannot tell which arc is meant: "
                    "the two ends are diametrically opposite, so every "
                    "plane through them carries a half-circle.  Add "
                    "normal= to name the axis the arc turns about, or "
                    "give via= instead."
                )
            sign = -1.0 if major else 1.0
            p_via = tuple(o + sign * radius * x / length for o, x in zip(c, bisector))
        return self._extended(Curve.arc(p_start, p_via, p_end))

    def spline_to(self, *points) -> "Path":
        """Append a smooth spline through *points*, ending at the last one.

        Parameters
        ----------
        *points : tuple of float
            One or more 3D points [meters]; the spline interpolates all
            of them in order.

        Returns
        -------
        Path
            A new Path ending at the last point.
        """
        if not points:
            raise ValueError("spline_to() needs at least one point.")
        via = [point3(pt, f"spline_to() point {i}") for i, pt in enumerate(points)]
        return self._extended(Curve.spline([self.current, *via]))

    # ── terminals ─────────────────────────────────────────────────────

    def curve(self, *, name=None) -> Curve:
        """Return the drawn path as a :class:`~magnelio.geo.Curve`.

        The path is taken as drawn: open unless the last segment happens
        to end where the first one started.  Use :meth:`closed` to make a
        loop.

        Parameters
        ----------
        name : str, optional
            Optional label.

        Returns
        -------
        Curve
        """
        if not self._segments:
            raise ValueError("This Path has no segments yet — add at least one before closing it.")
        first, *rest = self._segments
        return first.joined(*rest, name=name)

    def closed(self, *, name=None) -> Curve:
        """Return the drawn path as a closed :class:`~magnelio.geo.Curve`.

        A straight segment back to the start point is appended unless the
        path already ends there, so the result is always a loop and can
        be :meth:`~magnelio.geo.Curve.covered` into a sheet.

        Parameters
        ----------
        name : str, optional
            Optional label.

        Returns
        -------
        Curve
            The closed curve.
        """
        chain = self.curve(name=name)
        if chain.is_closed:
            return chain
        return self.line_to(self.start).curve(name=name)
