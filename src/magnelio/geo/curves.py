"""
Curve — one abstract, OCC-backed 3D locus.

A :class:`Curve` is a single one-dimensional path in space (a
``TopoDS_Wire``).  It is *not* a physical object and carries **no material**
— consumers decide how to use it: :func:`~magnelio.geo.sweep` turns a
profile + a spine Curve into a solid, :func:`~magnelio.geo.revolve`
uses an axis, and :class:`~magnelio.geo.ThinWire` rasterises a Curve onto
grid edges for the thin-wire sub-cell model.

Constructors (classmethods): :meth:`Curve.polyline`, :meth:`Curve.arc`,
:meth:`Curve.spline`, :meth:`Curve.helix`.  Several curves chain into one
profile with :meth:`Curve.joined`; a closed profile becomes a planar sheet
with :meth:`Curve.covered` and a conductor track with :meth:`Curve.traced`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from magnelio.geo._cache import cached_occ_shape
from magnelio.geo._validate import point3, point_list, positive

# Seam tolerance of :meth:`Curve.joined`, relative to the chain's own
# bounding-box diagonal.  Relative on purpose (DD-120): a micrometre-sized
# profile and a kilometre-sized one behave identically, and no absolute
# metre threshold can be right for both.
_JOIN_RTOL = 1e-6


@dataclass
class Curve:
    """An abstract 3D locus backed by an OCC wire (no material).

    Do not construct directly — use one of the classmethods
    (:meth:`polyline`, :meth:`arc`, :meth:`spline`, :meth:`helix`).  Each
    stores a builder callable that lazily produces the ``TopoDS_Wire`` on
    first use, cached like every other geometry object.

    A Curve exposes ``_occ_shape()`` (the wire) and :meth:`bounding_box`,
    but **no** ``material`` — a 1D locus is never a physical object on its
    own.
    """

    _build: object
    name: str | None = None
    # Conservative analytic AABB [m], set by every constructor — the
    # defining parameters live in the builder closure, so the box must
    # be captured at construction time (DD-120 scale choice).
    _bounds: tuple | None = None
    # Chain endpoints ((x, y, z) start, end) [m], or None when unknown —
    # a helix does not expose them.  Used for the eager connectivity and
    # closure checks, which is why they are captured, not recomputed.
    _ends: tuple | None = None
    # For a chained curve: the segments it was built from, so joining a
    # chain to another curve flattens instead of nesting.
    _segments: tuple = field(default_factory=tuple)

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        return self._build(scale)

    def bounding_box(
        self, scale: float | None = None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Return (min_corner, max_corner) in meters via OCC BRep bounding box."""
        from magnelio.geo._occ_backend import bounding_box  # noqa: PLC0415
        from magnelio.geo._scaling import choose_scale  # noqa: PLC0415

        if scale is None:
            scale = choose_scale(*self._analytic_bbox())
        return bounding_box(self._occ_shape(scale), scale=scale)

    def _analytic_bbox(self):
        if self._bounds is None:
            raise NotImplementedError(
                "Curve constructed without analytic bounds — every Curve "
                "constructor must set _bounds."
            )
        return self._bounds

    def _join_tol(self) -> float:
        """Seam tolerance [m] for this curve, relative to its own size."""
        from magnelio.geo._scaling import box_diagonal  # noqa: PLC0415

        diag = box_diagonal(self._analytic_bbox())
        return _JOIN_RTOL * diag if diag > 0.0 else 0.0

    @property
    def is_closed(self) -> bool:
        """Whether this curve's end meets its start, forming a loop.

        A closed curve is the input :meth:`covered` needs.  Curves whose
        endpoints are not known analytically — a helix — report ``False``;
        for those the check happens in the CAD kernel instead.
        """
        if self._ends is None:
            return False
        return math.dist(self._ends[0], self._ends[1]) <= self._join_tol()

    # ------------------------------------------------------------------
    # Chaining and conversion
    # ------------------------------------------------------------------

    def joined(self, *curves, name=None) -> "Curve":
        """Return one curve chaining this curve and *curves* end to start.

        Chaining is what turns the individual segment types into arbitrary
        profiles: an arc, a straight run and another arc become one
        boundary, and a boundary that closes on itself can be
        :meth:`covered` into a sheet and extruded or revolved into a
        solid.

        Segments must be given in order, each starting where the previous
        one ended.  They need not agree to the last bit — anything within
        one part per million of the chain's overall size counts as the
        same point — but a real gap is an error rather than something to
        bridge silently.

        Parameters
        ----------
        *curves : Curve
            The segments to append, in order.
        name : str, optional
            Optional label.

        Returns
        -------
        Curve
            The chained curve.  It is open or closed depending on whether
            the last segment ends where the first one starts; an open
            chain is a perfectly good sweep path.

        Raises
        ------
        TypeError
            If an argument is not a :class:`Curve`.
        ValueError
            If consecutive segments do not meet.

        Examples
        --------
        A D-shaped profile — a straight back and a semicircular front::

            back = Curve.polyline([(0, -5e-3, 0), (0, 5e-3, 0)])
            front = Curve.arc((0, 5e-3, 0), (5e-3, 0, 0), (0, -5e-3, 0))
            profile = back.joined(front)
            rod = profile.covered().extruded(vector=(0, 0, 20e-3), material=copper)
        """
        from magnelio.geo._scaling import box_diagonal, union_boxes  # noqa: PLC0415

        for curve in curves:
            if not isinstance(curve, Curve):
                raise TypeError(f"joined() takes Curve segments; got {type(curve).__name__}.")

        segments: list[Curve] = []
        for curve in (self, *curves):
            # Flatten: a chain of chains is one chain of leaf segments.
            segments.extend(curve._segments or (curve,))
        if len(segments) < 2:
            return segments[0]

        bounds = union_boxes([s._analytic_bbox() for s in segments])
        diag = box_diagonal(bounds)
        tol = _JOIN_RTOL * diag if diag > 0.0 else 0.0

        # Eager connectivity check wherever both endpoints are known: a
        # gap reported here names the segment, which the kernel cannot.
        for index, (prev, nxt) in enumerate(zip(segments, segments[1:]), start=1):
            if prev._ends is None or nxt._ends is None:
                continue
            gap = math.dist(prev._ends[1], nxt._ends[0])
            if gap > tol:
                raise ValueError(
                    f"Curve segment {index} starts {gap:.3e} m away from "
                    f"where segment {index - 1} ends (tolerance "
                    f"{tol:.3e} m).  Segments must be chained in order, "
                    f"each starting where the previous one ended."
                )

        def build(scale):
            from magnelio.geo._occ_backend import make_joined_wire  # noqa: PLC0415

            return make_joined_wire([s._occ_shape(scale) for s in segments], tol * scale)

        first, last = segments[0]._ends, segments[-1]._ends
        ends = (first[0], last[1]) if first is not None and last is not None else None
        return Curve(
            _build=build,
            name=name,
            _bounds=bounds,
            _ends=ends,
            _segments=tuple(segments),
        )

    def covered(self, *, material=None, name=None):
        """Return the planar sheet bounded by this closed curve.

        The free-form counterpart of :class:`~magnelio.geo.Face`, which is
        limited to axis-normal polygons: any closed planar boundary —
        arcs, splines and straight runs mixed — becomes a sheet here, and
        that sheet is the profile for
        :meth:`~magnelio.geo.Shape.extruded`,
        :meth:`~magnelio.geo.Shape.revolved`,
        :meth:`~magnelio.geo.Shape.swept` and
        :meth:`~magnelio.geo.Shape.thickened`.

        Parameters
        ----------
        material : Material, optional
            Material of the thin sheet.  ``None`` (default) makes it a
            construction profile, which is what you want when the sheet
            only exists to be grown into a solid.
        name : str, optional
            Optional label.

        Returns
        -------
        Shape
            The planar sheet.

        Raises
        ------
        ValueError
            If the curve is not closed, or — when the geometry is first
            built — not planar or self-intersecting.

        Examples
        --------
        A pad with one rounded end, 35 um of copper::

            outline = (
                Path((0.0, 0.0, 0.0))
                .line_to((10e-3, 0.0, 0.0))
                .arc_to((10e-3, 4e-3, 0.0), center=(10e-3, 2e-3, 0.0))
                .line_to((0.0, 4e-3, 0.0))
                .closed()
            )
            pad = outline.covered().extruded(vector=(0, 0, 35e-6), material=copper)
        """
        from magnelio.geo.modifications import cover  # noqa: PLC0415

        return cover(self, material=material, name=name)

    def traced(self, *, width, thickness, caps="round", normal=None, material=None, name=None):
        """Return the conductor track running along this curve.

        The curve is the track's centreline: it is widened by half the
        width to each side within its own plane, then given a
        metallisation thickness perpendicular to it.  That is the direct
        route from a routed path to the copper on a board, without
        assembling the track from separate straight and bent pieces.

        Corners of a polyline centreline come out rounded on the outside
        — a consequence of offsetting a path, and closer to a fabricated
        track than a mitred corner would be.

        Parameters
        ----------
        width : float
            Track width [meters].
        thickness : float
            Metallisation thickness [meters].  Negative grows the track
            on the other side of the centreline's plane.
        caps : {"round", "flat"}
            How an open track ends: ``"round"`` (default) closes it with
            a half-disc, ``"flat"`` cuts it off square, which is what a
            track meeting a port plane needs.  Ignored for a closed
            curve, which has no ends.
        normal : str or sequence of float, optional
            Normal of the plane the track lies in.  Only needed when the
            curve does not determine one — a straight centreline lies in
            infinitely many planes.
        material : Material, optional
            Material of the track.  ``None`` (default) makes it a
            construction body.
        name : str, optional
            Optional label.

        Returns
        -------
        Shape
            The track solid.

        Raises
        ------
        ValueError
            If the curve is not planar, if the plane is undetermined and
            no *normal* was given, or if the width is too large for the
            path's bends and clearances.

        Examples
        --------
        A 35 um copper feed line ending square at both ports::

            line = route.traced(
                width=0.6e-3, thickness=35e-6, caps="flat",
                normal="z", material=copper,
            )
        """
        from magnelio.geo.modifications import trace  # noqa: PLC0415

        return trace(
            self,
            width=width,
            thickness=thickness,
            caps=caps,
            normal=normal,
            material=material,
            name=name,
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def polyline(cls, points, *, name=None) -> "Curve":
        """A polyline (open, straight segments) through 3D *points*.

        Parameters
        ----------
        points : sequence of (float, float, float)
            At least 2 vertices [meters].
        name : str, optional
            Optional label.
        """
        pts = point_list(points, "Curve.polyline(points)", dim=3, minimum=2)

        def build(scale):
            from magnelio.geo._occ_backend import make_polyline  # noqa: PLC0415

            return make_polyline(pts, scale=scale)

        from magnelio.geo._scaling import box_of_points  # noqa: PLC0415

        return cls(_build=build, name=name, _bounds=box_of_points(pts), _ends=(pts[0], pts[-1]))

    @classmethod
    def arc(cls, start, through, end, *, name=None) -> "Curve":
        """A circular arc through three 3D points.

        Parameters
        ----------
        start, through, end : tuple of float
            The arc passes through all three points in order [meters].  The
            three points must not be collinear.
        name : str, optional
            Optional label.
        """
        p_start = point3(start, "Curve.arc(start)")
        p_through = point3(through, "Curve.arc(through)")
        p_end = point3(end, "Curve.arc(end)")

        def build(scale):
            from magnelio.geo._occ_backend import make_arc  # noqa: PLC0415

            return make_arc(p_start, p_through, p_end, scale=scale)

        from magnelio.geo._scaling import (  # noqa: PLC0415
            box_diagonal,
            box_of_points,
            circumcircle,
            pad_box,
        )

        # The arc lies on the circumcircle of the three points, so
        # center ± R contains it for any angular extent.
        circle = circumcircle(p_start, p_through, p_end)
        if circle is None:
            box = box_of_points([p_start, p_through, p_end])
            bounds = pad_box(box, box_diagonal(box))
        else:
            center, radius = circle
            bounds = pad_box((center, center), radius)
        return cls(_build=build, name=name, _bounds=bounds, _ends=(p_start, p_end))

    @classmethod
    def spline(cls, points, *, name=None) -> "Curve":
        """A smooth B-spline interpolating 3D *points*.

        Parameters
        ----------
        points : sequence of (float, float, float)
            At least 2 interpolation points [meters].
        name : str, optional
            Optional label.
        """
        pts = point_list(points, "Curve.spline(points)", dim=3, minimum=2)

        def build(scale):
            from magnelio.geo._occ_backend import make_spline  # noqa: PLC0415

            return make_spline(pts, scale=scale)

        from magnelio.geo._scaling import box_diagonal, box_of_points, pad_box  # noqa: PLC0415

        # An interpolating spline may overshoot the hull of its control
        # points; pad by half the hull diagonal (order of magnitude is
        # all the scale choice needs).
        box = box_of_points(pts)
        return cls(
            _build=build,
            name=name,
            _bounds=pad_box(box, 0.5 * box_diagonal(box)),
            _ends=(pts[0], pts[-1]),
        )

    @classmethod
    def helix(
        cls, *, radius, pitch, turns, origin=(0.0, 0.0, 0.0), axis="z", right_handed=True, name=None
    ) -> "Curve":
        """An exact helix on a cylinder.

        Parameters
        ----------
        radius : float
            Helix radius [meters].
        pitch : float
            Axial rise per full turn [meters].
        turns : float
            Number of turns (may be fractional).
        origin : tuple of float
            Base point on the axis [meters], as for
            :class:`~magnelio.geo.Cylinder`.
        axis : str
            Axis direction: ``'x'``, ``'y'``, or ``'z'`` (default).
        right_handed : bool
            If True (default) the helix ascends counter-clockwise about the
            axis; if False it is left-handed.
        name : str, optional
            Optional label.
        """
        if axis not in ("x", "y", "z"):
            raise ValueError(f"Helix axis must be 'x', 'y', or 'z'; got {axis!r}")
        radius = positive(radius, "Curve.helix(radius)")
        pitch = positive(pitch, "Curve.helix(pitch)")
        turns = positive(turns, "Curve.helix(turns)")
        origin = point3(origin, "Curve.helix(origin)")
        params = dict(
            radius=radius,
            pitch=pitch,
            turns=turns,
            origin=origin,
            axis=axis,
            right_handed=right_handed,
        )

        def build(scale):
            from magnelio.geo._occ_backend import make_helix  # noqa: PLC0415

            return make_helix(**params, scale=scale)

        from magnelio.geo._axes import normalize_axis  # noqa: PLC0415
        from magnelio.geo._scaling import axis_segment_box, pad_box  # noqa: PLC0415

        # The OCC helix wire is a B-spline approximation that can
        # overshoot the true cylinder radially by a fraction of a
        # percent — pad the exact bounds accordingly.
        bounds = pad_box(
            axis_segment_box(origin, normalize_axis(axis), pitch * turns, radius),
            0.1 * radius,
        )
        return cls(_build=build, name=name, _bounds=bounds)
