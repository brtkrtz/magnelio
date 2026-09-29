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

from magnelio.geo._axes import cross, normalize_axis
from magnelio.geo._validate import finite, point3, positive, vector3
from magnelio.geo.curves import _JOIN_RTOL, Curve


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _direction(value, what):
    if isinstance(value, str):
        return normalize_axis(value, what)
    vector = vector3(value, what, nonzero=True)
    magnitude = max(abs(v) for v in vector)
    return normalize_axis(tuple(v / magnitude for v in vector), what)


def _up(tangent, up):
    direction = _direction(up, "Path up")
    projection = _dot(tangent, direction)
    perpendicular = tuple(u - projection * t for u, t in zip(direction, tangent))
    if math.hypot(*perpendicular) <= 1e-12:
        raise ValueError("Path tangent and up must not be parallel.")
    return normalize_axis(perpendicular)


def _rotate(vector, axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    axial = _dot(axis, vector) * (1 - c)
    return tuple(c * v + s * w + axial * n for v, w, n in zip(vector, cross(axis, vector), axis))


def _corner_up(old_tangent, new_tangent, up):
    axis = cross(old_tangent, new_tangent)
    sine = math.hypot(*axis)
    cosine = _dot(old_tangent, new_tangent)
    if sine <= 1e-12:
        return up  # A reversal rotates about up, retaining that direction.
    return _up(new_tangent, _rotate(up, normalize_axis(axis), math.atan2(sine, cosine)))


def _segment_pose(segment, tangent, up):
    """Transport the routing pose along the actual CAD segment (DD-275).

    Bishop transport satisfies du/dq = -t (u dot dt/dq). It has no
    Frenet singularity at inflections and adds no twist about the tangent.
    Each Path segment is a single smooth analytic/interpolated edge.
    """
    from OCC.Core.BRepAdaptor import BRepAdaptor_CompCurve  # noqa: PLC0415
    from OCC.Core.gp import gp_Pnt, gp_Vec  # noqa: PLC0415
    from scipy.integrate import solve_ivp  # noqa: PLC0415

    from magnelio.geo._scaling import choose_scale  # noqa: PLC0415

    scale = choose_scale(*segment._analytic_bbox())
    curve = BRepAdaptor_CompCurve(segment._occ_shape(scale))
    first, last = curve.FirstParameter(), curve.LastParameter()

    def derivatives(q):
        p, d1, d2 = gp_Pnt(), gp_Vec(), gp_Vec()
        curve.D2(q, p, d1, d2)
        v, a = (d1.X(), d1.Y(), d1.Z()), (d2.X(), d2.Y(), d2.Z())
        speed = math.hypot(*v)
        if speed <= 1e-30:
            raise ValueError("Path segment has a zero tangent; its pose is undefined.")
        t = tuple(x / speed for x in v)
        dt = tuple((x - _dot(t, a) * y) / speed for x, y in zip(a, t))
        return t, dt

    start_tangent = derivatives(first)[0]
    end_tangent = derivatives(last)[0]
    if up is None:
        return end_tangent, None
    initial = _corner_up(tangent, start_tangent, up)

    def transport(q, u):
        t, dt = derivatives(q)
        projection = _dot(u, dt)
        return [-projection * x for x in t]

    solution = solve_ivp(
        transport,
        (first, last),
        initial,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
        max_step=(last - first) / 16,
    )
    if not solution.success:
        raise RuntimeError("Path frame transport failed: " + solution.message)
    return end_tangent, _up(end_tangent, solution.y[:, -1])


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
    path or :meth:`closed` for a loop accepted by
    :meth:`~magnelio.geo.Profile.from_wires`.

    :meth:`from_pose` and :meth:`from_face` seed a tangent and perpendicular
    up direction for relative routing. Up is transported without twist along
    curved segments; it defines left as ``up x tangent``. The pose guides
    routing and does not select a pipe-sweep orientation mode.

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
        slot = Profile.from_wires(outline).extruded(vector=(0, 0, t), material=copper)
    """

    start: tuple
    _segments: tuple = field(default_factory=tuple)
    _tangent: tuple | None = field(default=None, init=False, repr=False)
    _up_direction: tuple | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "start", point3(self.start, "Path(start)"))

    @property
    def current(self) -> tuple:
        """The point the next segment will start from [meters]."""
        if not self._segments:
            return self.start
        return self._segments[-1]._ends[1]

    @property
    def tangent(self) -> tuple | None:
        """Current unit forward direction; None before an unposed first segment."""
        return self._tangent

    @property
    def up(self) -> tuple | None:
        """Current perpendicular unit up direction; None for an unposed path."""
        return self._up_direction

    @classmethod
    def from_pose(cls, point, tangent, up) -> "Path":
        """Start a route from a world point and explicit directions.

        Parameters
        ----------
        point : sequence of float
            Start point in meters.
        tangent : str or sequence of float
            Nonzero forward direction, or an axis letter.
        up : str or sequence of float
            Nonzero roll reference, projected perpendicular to tangent.
            Parallel directions are rejected.

        Returns
        -------
        Path
            An empty immutable path with an orthonormal pose.
        """
        direction = _direction(tangent, "Path tangent")
        result = cls(point)
        object.__setattr__(result, "_tangent", direction)
        object.__setattr__(result, "_up_direction", _up(direction, up))
        return result

    @classmethod
    def from_face(cls, face_ref, *, up) -> "Path":
        """Start at a planar owned face's centroid along its outward normal.

        Parameters
        ----------
        face_ref : FaceRef
            Selected planar face, retaining its world placement.
        up : str or sequence of float
            Roll reference, projected into the face plane.

        Returns
        -------
        Path
            Empty route pointing out of the face.

        Raises
        ------
        TypeError
            If the input is not a FaceRef.
        ValueError
            If the face is curved or up is parallel to its normal.
        """
        from magnelio.geo.topology import FaceRef  # noqa: PLC0415

        if not isinstance(face_ref, FaceRef):
            raise TypeError("Path.from_face requires a planar FaceRef.")
        if not face_ref.is_planar:
            raise ValueError(
                "Path.from_face requires a planar face with a constant outward normal."
            )
        return cls.from_pose(face_ref.centroid, face_ref.normal, up)

    def _extended(self, segment: Curve, *, pose=None) -> "Path":
        direction, up = pose if pose is not None else _segment_pose(segment, self.tangent, self.up)
        result = Path(self.start, (*self._segments, segment))
        object.__setattr__(result, "_tangent", direction)
        object.__setattr__(result, "_up_direction", up)
        return result

    def _require_pose(self, *, roll=False):
        if self.tangent is None or (roll and self.up is None):
            raise ValueError(
                "Relative routing needs a pose; start with Path.from_pose or Path.from_face."
            )

    def forward(self, distance) -> "Path":
        """Append a straight run along the current tangent.

        Parameters
        ----------
        distance : float
            Positive length in meters.

        Returns
        -------
        Path
            Route with unchanged tangent and up.
        """
        distance = positive(distance, "forward distance")
        self._require_pose()
        return self.line_to(tuple(p + distance * t for p, t in zip(self.current, self.tangent)))

    def _turn(self, axis, radius, angle):
        radial = cross(self.tangent, axis)
        center = tuple(p - radius * r for p, r in zip(self.current, radial))

        def point(theta):
            return tuple(c + radius * r for c, r in zip(center, _rotate(radial, axis, theta)))

        curve = Curve.arc(self.current, point(angle / 2), point(angle))
        tangent = normalize_axis(_rotate(self.tangent, axis, angle))
        return self._extended(curve, pose=(tangent, _up(tangent, _rotate(self.up, axis, angle))))

    def turn_left(self, *, radius, angle_deg) -> "Path":
        """Append an exact circular turn towards ``up x tangent``.

        Parameters
        ----------
        radius : float
            Positive bend radius in meters.
        angle_deg : float
            Positive angle strictly below 360 degrees; up is the rotation axis.

        Returns
        -------
        Path
            Route with the tangent rotated and up retained.
        """
        self._require_pose(roll=True)
        radius = positive(radius, "turn radius")
        angle = positive(angle_deg, "turn angle_deg")
        if angle >= 360:
            raise ValueError("turn angle_deg must be less than 360; split a full loop into turns.")
        return self._turn(self.up, radius, math.radians(angle))

    def turn_right(self, *, radius, angle_deg) -> "Path":
        """Append an exact circular turn towards ``tangent x up``.

        Parameters
        ----------
        radius : float
            Positive bend radius in meters.
        angle_deg : float
            Positive angle strictly below 360 degrees.

        Returns
        -------
        Path
            Route with the tangent rotated and up retained.
        """
        self._require_pose(roll=True)
        return self._right_turn(radius, angle_deg)

    def _right_turn(self, radius, angle_deg):
        radius = positive(radius, "turn radius")
        angle = positive(angle_deg, "turn angle_deg")
        if angle >= 360:
            raise ValueError("turn angle_deg must be less than 360; split a full loop into turns.")
        return self._turn(tuple(-x for x in self.up), radius, math.radians(angle))

    def turn_to(self, direction, *, radius) -> "Path":
        """Turn through the shortest circular arc to a world tangent.

        Parameters
        ----------
        direction : str or sequence of float
            Target tangent; its length is ignored.
        radius : float
            Positive bend radius in meters.

        Returns
        -------
        Path
            Route with up carried by the bend's rigid rotation. An already
            aligned target returns the receiver without adding an edge.

        Raises
        ------
        ValueError
            If the directions are opposite (the bend plane is undetermined).
            Use an explicit left or right 180-degree turn instead.
        """
        self._require_pose(roll=True)
        radius = positive(radius, "turn radius")
        target = _direction(direction, "turn_to direction")
        axis = cross(self.tangent, target)
        sine, cosine = math.hypot(*axis), _dot(self.tangent, target)
        if sine <= 1e-12:
            if cosine < 0:
                raise ValueError(
                    "Opposite turn_to directions do not determine a bend plane; "
                    "use turn_left or turn_right."
                )
            return self
        return self._turn(normalize_axis(axis), radius, math.atan2(sine, cosine))

    def straight_to_plane(self, normal, position) -> "Path":
        """Continue forward to the world plane ``normal dot point = position``.

        Parameters
        ----------
        normal : str or sequence of float
            Plane normal, normalized before applying position.
        position : float
            Signed plane offset from the world origin in meters.

        Returns
        -------
        Path
            Route ending on the plane; an already reached plane adds no edge.

        Raises
        ------
        ValueError
            If the ray is parallel to the plane or the intersection is behind it.
        """
        self._require_pose()
        normal = _direction(normal, "straight_to_plane normal")
        position = finite(position, "straight_to_plane position")
        gap = position - _dot(normal, self.current)
        magnitude = max(abs(position), sum(abs(n * p) for n, p in zip(normal, self.current)))
        roundoff = 8 * math.ulp(magnitude)
        if abs(gap) <= roundoff:
            return self
        slope = _dot(normal, self.tangent)
        if abs(slope) <= 1e-12:
            raise ValueError("Path tangent is parallel to the requested plane.")
        distance = gap / slope
        if distance < 0:
            raise ValueError("The requested plane lies behind the Path tangent.")
        return self.forward(distance)

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
        end = point3(point, "line_to(point)")
        delta = tuple(e - s for e, s in zip(end, self.current))
        direction = _direction(delta, "line_to direction")
        up = None if self.up is None else _corner_up(self.tangent, direction, self.up)
        return self._extended(Curve.line(self.current, end), pose=(direction, up))

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

    def ellipse_to(self, end, *, center, semi_axes, major_axis, normal) -> "Path":
        """Append an elliptical arc ending at *end*.

        The ellipse is given by its *center*, its two semi-axes and the
        direction of the first one; the current point and *end* must both
        lie on it.  Of the two arcs joining them, the one drawn runs
        counter-clockwise about *normal* — the same convention as
        :meth:`arc_to` with ``center=`` and ``normal=``, so reversing the
        direction of *normal* gives the complementary arc.

        Parameters
        ----------
        end : tuple of float
            End point ``(x, y, z)`` [meters], on the ellipse.
        center : tuple of float
            Centre of the ellipse [meters].
        semi_axes : tuple of float
            ``(a, b)`` [meters]: *a* along *major_axis*, *b* along
            ``normal x major_axis``.  Either may be the larger one — the
            names only say which direction each length belongs to.
        major_axis : str or sequence of float
            Direction of the *a* semi-axis: ``'x'``, ``'y'``, ``'z'``, or
            any non-zero 3-vector (its component along *normal* is ignored).
        normal : str or sequence of float
            Normal of the ellipse's plane and the sense of the arc.

        Returns
        -------
        Path
            A new Path ending at *end*.

        Raises
        ------
        ValueError
            If an endpoint is off the ellipse or its plane, if a semi-axis
            is not positive, or if *major_axis* is parallel to *normal*.

        Examples
        --------
        The iris of an accelerator cell, drawn in the x-z plane (x is the
        radius, z the axis): a quarter turn from the z-ward end of the
        ellipse down to the iris radius (normal ``-y`` picks the short
        way round)::

            path.ellipse_to(
                (0.035, 0.0, 0.012),
                center=(0.054, 0.0, 0.012),
                semi_axes=(0.012, 0.019),
                major_axis="z",
                normal=(0.0, -1.0, 0.0),
            )
        """
        return self._extended(
            Curve.ellipse_arc(
                self.current,
                point3(end, "ellipse_to(end)"),
                center=center,
                semi_axes=semi_axes,
                major_axis=major_axis,
                normal=normal,
            )
        )

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
        be made into a sheet with :meth:`~magnelio.geo.Profile.from_wires`.

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
