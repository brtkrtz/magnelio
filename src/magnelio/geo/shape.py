"""Dimensional standalone geometry bases.

:class:`Shape` carries the affine placement and bounding-box protocol shared
by curves, sheets, profiles and solids.  During the staged foundation
migration, construction verbs remain implemented here but validate their
dimensional inputs before reaching the CAD kernel.

The verbs delegate to implementations in ``transforms``/
``modifications``; those functions are internal, and this class is the
documented home of their behaviour.  All imports inside the methods are
deliberate: this module sits below ``operations``/``transforms``/
``modifications`` in the import graph and must not import them at module
level.
"""

from __future__ import annotations


class Shape:
    """Base class of immutable standalone geometry.

    Every standalone geometry object is a ``Shape``.  ``Shape`` is a base
    type, not something to instantiate directly; use one of its dimensional
    subclasses.

    **Shapes are immutable.**  Every operator and verb returns a *new*
    shape; the receiver is never modified.  That is what makes the calls
    chainable::

        pin = Cylinder(radius=0.5e-3, height=4e-3, material=pec)
        part = pin.rotated("y", 90.0).translated((0, 0, 1e-3)) - hole

    **Materials follow the base operand.**  A Boolean result takes the
    material of its base (:class:`~magnelio.geo.Difference`) resp. first
    (:class:`~magnelio.geo.Union`, :class:`~magnelio.geo.Intersection`)
    operand, and a transformed shape keeps the material of the shape it
    came from.  Tools and profiles therefore need no material of their
    own — see :class:`~magnelio.geo.Brick` for construction bodies.

    The dimensional subclasses are :class:`~magnelio.geo.Curve`,
    :class:`~magnelio.geo.Sheet` (including
    :class:`~magnelio.geo.Profile`) and :class:`~magnelio.geo.Solid`.
    Every category supports the same four affine transforms.  Boolean
    operations are restricted to ``Solid`` values.
    """

    # ── geometry queries ──────────────────────────────────────────────

    def bounding_box(
        self, scale: float | None = None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Return the axis-aligned bounding box of this shape.

        The box is computed from the CAD kernel's representation, so it
        accounts for the true geometry rather than the shape's nominal
        parameters — a rotated brick reports the box of the rotated
        solid.

        Parameters
        ----------
        scale : float, optional
            Unit scale factor at which to build the kernel shape.  Leave
            it unset: the scale is then derived from the shape itself so
            the result is correct for models spanning nanometres to
            kilometres.

        Returns
        -------
        tuple
            ``(min_corner, max_corner)``, each ``(x, y, z)`` in meters.
        """
        from magnelio.geo._occ_backend import bounding_box  # noqa: PLC0415
        from magnelio.geo._scaling import choose_scale  # noqa: PLC0415

        if scale is None:
            scale = choose_scale(*self._analytic_bbox())
        return bounding_box(self._occ_shape(scale), scale=scale)

    def volume(self, scale: float | None = None) -> float:
        """Return the volume enclosed by this shape.

        Computed from the CAD kernel's representation, so it accounts
        for the true geometry rather than the shape's nominal
        parameters: a Boolean difference reports what is left, and a
        chamfered block reports what the chamfer took away.  That makes
        it the direct way to check a construction — a filling factor, a
        metal volume, the agreement between two ways of building the
        same part.

        Parameters
        ----------
        scale : float, optional
            Unit scale factor at which to build the kernel shape.  Leave
            it unset: the scale is then derived from the shape itself so
            the result is correct for models spanning nanometres to
            kilometres.

        Returns
        -------
        float
            Volume in cubic meters.  A planar sheet
            (:class:`~magnelio.geo.Profile`) has no thickness and reports
            zero.

        Examples
        --------
        The fraction of a housing that is metal::

            fill = shell.volume() / block.volume()
        """
        from magnelio.geo._occ_backend import occ_volume  # noqa: PLC0415
        from magnelio.geo._scaling import choose_scale  # noqa: PLC0415

        if scale is None:
            scale = choose_scale(*self._analytic_bbox())
        # The kernel works in scaled units, so volumes come back scaled
        # by s^3 (lossless to undo: s is a power of two).
        return abs(occ_volume(self._occ_shape(scale))) / scale**3

    # ── CSG operators ─────────────────────────────────────────────────

    def __add__(self, other):
        """``a + b`` — Boolean union; see :class:`~magnelio.geo.Union`."""
        from magnelio.geo.operations import Union  # noqa: PLC0415

        if not _is_shape(other):
            return NotImplemented
        return Union(self, other)

    def __sub__(self, other):
        """``a - b`` — Boolean difference; see :class:`~magnelio.geo.Difference`."""
        from magnelio.geo.operations import Difference  # noqa: PLC0415

        if not _is_shape(other):
            return NotImplemented
        return Difference(self, other)

    def __and__(self, other):
        """``a & b`` — Boolean intersection; see :class:`~magnelio.geo.Intersection`."""
        from magnelio.geo.operations import Intersection  # noqa: PLC0415

        if not _is_shape(other):
            return NotImplemented
        return Intersection(self, other)

    # ── transforms ────────────────────────────────────────────────────

    def translated(self, vector, *, repeat=1, copy=False, unite=False, group=False):
        """Return this shape moved by *vector*.

        Parameters
        ----------
        vector : tuple of float
            ``(dx, dy, dz)`` translation [meters].
        repeat : int, optional
            Number of translated copies, starting at one vector displacement.
            Copy *i* is moved by ``i * vector``; defaults to 1.
        copy : bool, optional
            Include the untransformed original first; defaults to False.
        unite : bool, optional
            Return a Union of the copies. Only Solid geometry is eligible.
        group : bool, optional
            Return a Group, preserving each member's material. Mutually
            exclusive with *unite*.
        Returns
        -------
        Shape or list of Shape or Union or Group
            One translated value by default; otherwise a list, or the
            explicitly requested Union or Group, including for one copy.

        Examples
        --------
        A row of eight vias, including the original, fused into one body::

            fence = via.translated((2e-3, 0, 0), repeat=7, copy=True, unite=True)
        """
        from magnelio.geo.transforms import translate  # noqa: PLC0415

        return translate(self, vector, repeat=repeat, copy=copy, unite=unite, group=group)

    def rotated(
        self,
        axis,
        angle_deg,
        origin=(0.0, 0.0, 0.0),
        *,
        repeat=1,
        copy=False,
        unite=False,
        group=False,
    ):
        """Return this shape rotated about an axis.

        Parameters
        ----------
        axis : str or sequence of float
            Rotation axis: ``'x'``, ``'y'``, ``'z'``, or any non-zero
            3-vector (its length is ignored).
        angle_deg : float
            Rotation angle [degrees], right-handed about *axis*.  Copy
            *i* is rotated by ``i * angle_deg``.
        origin : tuple of float
            A point on the rotation axis (default: the coordinate
            origin).
        repeat : int, optional
            Number of rotated copies at ``angle_deg`` through
            ``repeat * angle_deg``; defaults to 1.
        copy : bool, optional
            Include the unrotated original first; defaults to False.
        unite : bool, optional
            Return a Union of the copies. Only Solid geometry is eligible.
        group : bool, optional
            Return a Group, preserving each member's material. Mutually
            exclusive with *unite*.
        Returns
        -------
        Shape or list of Shape or Union or Group
            One rotated value by default; otherwise a list, or the
            explicitly requested Union or Group, including for one copy.

        Examples
        --------
        Four posts at 90° spacing around the z axis::

            posts = post.rotated("z", 90.0, repeat=3, copy=True, group=True)
        """
        from magnelio.geo.transforms import rotate  # noqa: PLC0415

        return rotate(
            self, axis, angle_deg, origin, repeat=repeat, copy=copy, unite=unite, group=group
        )

    def scaled(self, factor, center=(0.0, 0.0, 0.0)):
        """Return this shape scaled uniformly about a fixed point.

        The scaling is uniform in all three directions; there is no
        per-axis factor, because a non-uniform scaling would turn
        cylinders into elliptic cylinders and spheres into ellipsoids,
        which the primitives cannot represent.

        Parameters
        ----------
        factor : float
            Uniform scale factor.  Note that this is *not* a way to
            mirror a shape: a negative factor inverts the shape through
            *center*, negating all three axes at once.  Use
            :meth:`mirrored` for a reflection.
        center : tuple of float
            Fixed point of the scaling (default: the coordinate origin).

        Returns
        -------
        Shape or Group
            The scaled shape; a :class:`~magnelio.geo.Group` is scaled
            member by member about the common *center*.
        """
        from magnelio.geo.transforms import Scale  # noqa: PLC0415

        return Scale(factor, center) @ self

    def mirrored(self, normal, position=0.0, *, copy=False, unite=False, group=False):
        """Return this shape reflected across a plane.

        The plane is the set of points ``p`` with ``p · normal ==
        position``, so for an axis letter *position* is simply the
        coordinate of the plane on that axis.

        A reflection is not a rotation: it leaves the two in-plane
        directions untouched and reverses only the normal one.  That is
        what makes it the correct operation for a structure symmetric
        about a plane but not about the axis normal to it — most planar
        circuits (dividers, couplers, filters) and every layer stack
        that differs from top to bottom.

        Unlike :meth:`translated` and :meth:`rotated` there is no
        *repeat*: mirroring twice across one plane reproduces the
        original.

        Parameters
        ----------
        normal : str or sequence of float
            Plane normal: ``'x'``, ``'y'``, ``'z'``, or any non-zero
            3-vector (its length is ignored).  ``normal='x'`` maps
            ``x -> 2 * position - x``.
        position : float
            Signed distance of the plane from the coordinate origin
            along *normal* [meters] (default 0).
        copy : bool, optional
            Include the original first; defaults to False.
        unite : bool, optional
            Return a Union of original and image. Requires *copy* and Solid
            geometry.
        group : bool, optional
            Return a Group of original and image, retaining member materials.
            Requires *copy*; mutually exclusive with *unite*.
        Returns
        -------
        Shape or list of Shape or Union or Group
            The image alone by default, otherwise ``[original, image]``
            or the explicitly requested Union or Group.

        Examples
        --------
        Complete a half-modelled power divider into one solid::

            full = half.mirrored("x", copy=True, unite=True)

        Mirror a feed line onto the far side of a board::

            far = line.mirrored("z", position=h / 2)
        """
        from magnelio.geo.transforms import mirror  # noqa: PLC0415

        return mirror(self, normal=normal, position=position, copy=copy, unite=unite, group=group)

    # ── modifications ─────────────────────────────────────────────────

    def chamfered(self, *, near=None, face_near=None, edges=None, distance):
        """Return this shape with a chamfer (a flat bevel) on selected edges.

        Exactly one of *near*, *face_near* or *edges* must be given —
        they are three ways of naming the edges to work on.

        Parameters
        ----------
        near : tuple or list of tuples, optional
            3D point(s) ``(x, y, z)`` near the edge(s) to chamfer.  A
            single point selects the one nearest edge; a list selects the
            nearest edge for each point.
        face_near : tuple of float, optional
            3D point near a face.  All edges of the nearest face are
            chamfered.
        edges : str, optional
            ``"all"`` to chamfer every edge of the shape.
        distance : float or tuple of float
            Chamfer distance [meters].  A single value gives a symmetric
            chamfer, a pair ``(d1, d2)`` an asymmetric one.

        Returns
        -------
        Shape
            A new shape with the chamfer applied, same material.
        """
        from magnelio.geo.modifications import chamfer  # noqa: PLC0415

        return chamfer(self, near=near, face_near=face_near, edges=edges, distance=distance)

    def filleted(self, *, near=None, face_near=None, edges=None, radius):
        """Return this shape with a fillet (a rounded edge) on selected edges.

        Exactly one of *near*, *face_near* or *edges* must be given.
        Rounding sharp metal edges is the usual reason: a right-angled
        edge concentrates the field far more than any real fabricated
        part does.

        Parameters
        ----------
        near : tuple or list of tuples, optional
            3D point(s) ``(x, y, z)`` near the edge(s) to fillet.
        face_near : tuple of float, optional
            3D point near a face.  All edges of the nearest face are
            filleted.
        edges : str, optional
            ``"all"`` to fillet every edge of the shape.
        radius : float
            Fillet radius [meters].

        Returns
        -------
        Shape
            A new shape with the fillet applied, same material.
        """
        from magnelio.geo.modifications import fillet  # noqa: PLC0415

        return fillet(self, near=near, face_near=face_near, edges=edges, radius=radius)

    def extruded(self, vector, *, face_near=None, material=None):
        """Extrude a face of this shape along a vector into a new solid.

        The result is a **standalone solid**, not fused with the shape it
        came from.  Two input forms:

        - a standalone sheet — a :class:`~magnelio.geo.Profile` or a curved
          :class:`~magnelio.geo.Surface` — the sheet *is* the profile and
          *face_near* is unused;
        - any solid — the face nearest *face_near* is extruded.

        Parameters
        ----------
        vector : tuple of float
            ``(dx, dy, dz)`` extrusion direction and length [meters].
        face_near : tuple of float, optional
            3D point near the face to extrude.  Required for a solid,
            ignored for a Profile.
        material : Material, optional
            Material of the extruded solid.  Defaults to this shape's
            material; required when extruding a construction sheet, which
            has none to inherit.

        Returns
        -------
        Shape
            The extruded solid.
        """
        from magnelio.geo.modifications import extrude  # noqa: PLC0415

        return extrude(self, vector=vector, face_near=face_near, material=material)

    def revolved(self, axis, angle_deg=360.0, *, origin=(0.0, 0.0, 0.0), material=None):
        """Revolve this planar profile about an axis into a solid of revolution.

        The result is a **standalone solid**.  The profile must not cross
        the revolution axis — that would produce a self-intersecting
        solid.

        Parameters
        ----------
        axis : str or sequence of float
            Revolution axis: ``'x'``, ``'y'``, ``'z'``, or any non-zero
            3-vector.
        angle_deg : float
            Revolution angle [degrees] (default 360, a full revolution).
        origin : tuple of float
            A point on the revolution axis (default: the coordinate
            origin).
        material : Material, optional
            Material of the revolved solid.  Defaults to this shape's
            material; required for a construction Profile.

        Returns
        -------
        Shape
            The solid of revolution.
        """
        from magnelio.geo.modifications import revolve  # noqa: PLC0415

        return revolve(self, axis=axis, angle_deg=angle_deg, origin=origin, material=material)

    def swept(self, spine, *, material=None):
        """Sweep this planar profile along a curve into a solid.

        The profile is moved for you: its centroid is placed on the
        spine's start point and its plane turned perpendicular to the
        spine's start tangent, then it follows the path.  The canonical
        example is a coil, ``Profile.rectangle(...).swept(Curve.helix(...))``.

        Parameters
        ----------
        spine : Curve
            The :class:`~magnelio.geo.Curve` giving the sweep path.
        material : Material, optional
            Material of the swept solid.  Defaults to this shape's
            material; required for a construction profile.

        Returns
        -------
        Shape
            The swept solid.
        """
        from magnelio.geo.modifications import sweep  # noqa: PLC0415

        return sweep(self, spine, material=material)

    def shelled(self, thickness, *, opening_face_near=None):
        """Return this solid hollowed out to a constant wall thickness.

        The walls are built inward, so the outer surface stays exactly
        where it was and the shape keeps its footprint — the difference
        between a solid block and the housing, waveguide or cavity a real
        part is.  Naming faces through *opening_face_near* leaves them
        out of the shell, turning them into openings: one for an open
        box, two opposite ones for a length of waveguide.

        Parameters
        ----------
        thickness : float
            Wall thickness [meters], positive.
        opening_face_near : tuple or list of tuples, optional
            3D point(s) near the face(s) to leave open.  Omit for a
            closed body with a sealed internal void.

        Returns
        -------
        Shape
            The hollowed solid, same material.

        Raises
        ------
        TypeError
            If this is a planar sheet — use :meth:`thickened` instead.
        RuntimeError
            If the wall does not fit: an offset surface stops being
            valid once the thickness approaches the smallest local
            dimension or curvature radius of the solid.

        Examples
        --------
        A length of rectangular waveguide, open at both ends::

            tube = block.shelled(
                thickness=2e-3, opening_face_near=[(0, 0, 0), (0, 0, L)]
            )
        """
        from magnelio.geo.modifications import shell  # noqa: PLC0415

        return shell(self, thickness=thickness, opening_face_near=opening_face_near)

    def thickened(self, thickness, *, direction="forward", material=None):
        """Grow this sheet into a solid of constant thickness.

        Only a sheet — a :class:`~magnelio.geo.Profile` or a curved
        :class:`~magnelio.geo.Surface` — can be thickened.  A planar sheet
        becomes a slab whose footprint is exactly the sheet, which makes
        this the direct way from a drawn outline to a metallisation of a
        given thickness, without spelling out the extrusion vector.  A
        curved sheet is offset along its own normal into a shell of
        constant thickness; where the kernel cannot build a valid offset
        (coarse sample grids, thickness near the curvature radius) the
        call fails with a pointer to :meth:`extruded`, which is always
        robust and, for a conductor, physically equivalent.

        Parameters
        ----------
        thickness : float
            Slab thickness [meters], positive.
        direction : {"forward", "backward", "symmetric"}
            Which side of the sheet to grow on.  ``"symmetric"`` puts
            half the thickness on each side, leaving the sheet as the
            slab's mid-plane (planar sheets only).  ``"forward"`` and ``"backward"`` are
            opposite sides of it; which one is "forward" follows from
            the plane and is fixed, so if a slab comes out on the wrong
            side, swap the value.
        material : Material, optional
            Material of the slab.  Defaults to the sheet's material;
            required for a construction profile, which has none.

        Returns
        -------
        Shape
            The solid slab.

        Raises
        ------
        TypeError
            If this is a solid — use :meth:`shelled` instead.

        Examples
        --------
        A copper patch from a drawn outline::

            patch = Profile.from_wires(outline).thickened(thickness=35e-6, material=copper)
        """
        from magnelio.geo.modifications import thicken  # noqa: PLC0415

        return thicken(self, thickness=thickness, direction=direction, material=material)

    def lofted(
        self, face_near, other, other_face_near, *, material=None, blend="spline", tension=None
    ):
        """Loft a solid between a face of this shape and one of *other*.

        Takes the outer wire of the face of this shape nearest
        *face_near* and of the face of *other* nearest
        *other_face_near*, then builds the transition between them — the
        way to model a taper between two different cross-sections, such
        as a waveguide-to-coax transition.

        Both points select by **proximity**, not by containment: the
        face nearest the point wins, and a point on a shared edge is
        equally near several faces.  Aim at the middle of the intended
        face, or just outside it along its normal, rather than at a
        corner.

        Parameters
        ----------
        face_near : tuple of float
            3D point near the start face, on this shape.
        other : Shape
            The shape providing the end profile.
        other_face_near : tuple of float
            3D point near the end face, on *other*.
        material : Material, optional
            Material of the lofted solid.  Defaults to this shape's
            material.
        blend : {'spline', 'ruled', 'tangent'}
            How the two profiles are joined.  ``'spline'`` (default) and
            ``'ruled'`` both run straight from one profile to the other
            and differ only in surface type, so the solid meets each face
            at whatever angle the straight connection happens to make.
            ``'tangent'`` leaves both faces along their outward normal:
            zero wall slope at each joint, so the transition meets both
            solids without a crease.  Between two faces that look at each
            other -- the two ends of a taper, coaxial or laterally offset
            -- the cross-section eases out of one profile and into the
            other along a straight run; between faces that point in
            different directions the profile is swept along a curved
            path that turns the corner.
        tension : float or tuple of float, optional
            Only for ``blend='tangent'``: how far the blend holds its
            normal direction before turning, as a fraction of the
            distance between the two faces.  A single value applies to
            both ends, a ``(start, end)`` pair to one each.  Defaults to
            ``1/3``, at which a taper's cross-section is spaced linearly
            along its axis; larger values reach further along the
            normals and eventually overshoot into a bulge.

        Returns
        -------
        Shape
            The lofted solid.

        Raises
        ------
        ValueError
            If *blend* is not one of the three modes, if *tension* is
            given for a mode that has no use for it, if the two faces
            share a centre point, or if two parallel faces look away
            from each other.

        Examples
        --------
        A stripline electrode bending into a coaxial inner conductor,
        meeting both at a right angle::

            transition = electrode.lofted(
                (0.0, 45.5e-3, 0.0), inner, (0.0, 48e-3, -10e-3),
                material=pec, blend="tangent",
            )

        A rectangular waveguide easing into a round one, with no crease
        at either flange::

            taper = rect_guide.lofted(
                (0.0, 0.0, 0.0), round_guide, (0.0, 0.0, 59e-3),
                material="air", blend="tangent",
            )
        """
        from magnelio.geo.modifications import loft  # noqa: PLC0415

        return loft(
            self,
            face_near,
            other,
            other_face_near,
            material=material,
            blend=blend,
            tension=tension,
        )


class Solid(Shape):
    """Base class of closed three-dimensional bodies.

    Primitive bodies, imported CAD, Boolean results and construction results
    derive from this category.  Boolean union, difference and intersection
    accept only ``Solid`` operands.
    """


def _is_shape(obj) -> bool:
    """True for anything the CSG operators can meaningfully combine.

    A Group passes on purpose: the Boolean constructors reject it with
    their own descriptive TypeError, which beats a generic
    'unsupported operand type' from returning NotImplemented.
    """
    from magnelio.geo.operations import Group  # noqa: PLC0415

    return hasattr(obj, "_occ_shape") or isinstance(obj, Group)
