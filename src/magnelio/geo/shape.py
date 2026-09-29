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

    def chamfered(self, *, near=None, face_near=None, edges=None, faces=None, distance):
        """Return a Solid with a flat bevel on selected owned edges.

        Select exactly one of ``edges``, ``faces``, ``near`` or ``face_near``.
        References must belong to this exact Solid; point forms use semantic
        selection and refuse ambiguity.

        Parameters
        ----------
        near : tuple or list of tuple, optional
            World points in metres selecting nearest edges.
        face_near : tuple of float, optional
            World point selecting a face's complete boundary.
        edges : EdgeRef or EdgeSetRef or sequence or str, optional
            Owned edges, or ``"all"`` for every edge.
        faces : FaceRef or FaceSetRef or sequence, optional
            Owned faces whose boundary edges are selected.
        distance : float or tuple of float
            Positive bevel distance in metres, or an asymmetric pair.

        Returns
        -------
        Solid
            Modified body, inheriting this Solid's material.
        """
        from magnelio.geo.modifications import chamfer  # noqa: PLC0415

        return chamfer(
            self, near=near, face_near=face_near, edges=edges, faces=faces, distance=distance
        )

    def filleted(self, *, near=None, face_near=None, edges=None, faces=None, radius):
        """Return a Solid with rounded selected owned edges.

        Select exactly one of ``edges``, ``faces``, ``near`` or ``face_near``.
        References must belong to this exact Solid; point forms refuse ties.

        Parameters
        ----------
        near : tuple or list of tuple, optional
            World points in metres selecting nearest edges.
        face_near : tuple of float, optional
            World point selecting a face's complete boundary.
        edges : EdgeRef or EdgeSetRef or sequence or str, optional
            Owned edges, or ``"all"`` for every edge.
        faces : FaceRef or FaceSetRef or sequence, optional
            Owned faces whose boundary edges are selected.
        radius : float
            Positive fillet radius in metres.

        Returns
        -------
        Solid
            Modified body, inheriting this Solid's material.
        """
        from magnelio.geo.modifications import fillet  # noqa: PLC0415

        return fillet(self, near=near, face_near=face_near, edges=edges, faces=faces, radius=radius)

    def extruded(self, vector, *, face_near=None, material=None):
        """Extrude a Sheet into an independent Solid, retaining holes.

        Parameters
        ----------
        vector : tuple of float
            Non-zero world extrusion vector in metres.
        face_near : tuple of float, optional
            For a Solid receiver, select a temporary FaceRef near this world
            point. Prefer ``solid.face(...).extruded(vector)``. A tied pick
            raises AmbiguousTopologyError. Invalid on standalone sheets.
        material : Material or str, optional
            Override the section's material; otherwise it is inherited.
            Materialless sections produce construction solids for Boolean use.

        Returns
        -------
        Solid
            Independent prism, without fusion to an input owner.
        """
        from magnelio.geo.modifications import extrude  # noqa: PLC0415

        return extrude(self, vector=vector, face_near=face_near, material=material)

    def revolved(self, axis, angle_deg=360.0, *, origin=(0.0, 0.0, 0.0), material=None):
        """Revolve a planar Sheet into an independent Solid.

        Parameters
        ----------
        axis : str or sequence of float
            Revolution axis letter or non-zero world vector.
        angle_deg : float, optional
            Non-zero right-handed angle, at most a full turn, in degrees.
        origin : tuple of float, optional
            World point on the revolution axis, in metres.
        material : Material or str, optional
            Override the section's material; otherwise it is inherited.
            Without material the result is a construction solid.

        Returns
        -------
        Solid
            Solid of revolution retaining all holes. A section crossing the
            axis can produce invalid or self-intersecting geometry.
        """
        from magnelio.geo.modifications import revolve  # noqa: PLC0415

        return revolve(self, axis=axis, angle_deg=angle_deg, origin=origin, material=material)

    def swept(
        self, spine, *, face_near=None, material=None, frame="corrected_frenet", binormal=None
    ):
        """Sweep a planar Sheet along a Curve into an independent Solid.

        The actual boundary is translated to the spine start and aligned by
        the shortest normal-to-tangent rotation, retaining in-plane roll.
        An already aligned section stays in its actual orientation. For an
        opposite normal, the section plane's X axis defines the half-turn.
        The pipe follows corrected Frenet transport unless another frame
        is selected explicitly.

        Parameters
        ----------
        spine : Curve
            World sweep path.
        face_near : tuple of float, optional
            For a Solid receiver, select a temporary FaceRef near this world
            point. Prefer ``solid.face(...).swept(spine)``. Tied picks raise.
            Invalid for standalone sheets.
        material : Material or str, optional
            Override the section's material; otherwise it is inherited.
            Without material the result is a construction solid.
        frame : {'corrected_frenet', 'frenet', 'fixed', 'fixed_binormal'}, optional
            Transport of the initially aligned section. The default retains
            corrected Frenet transport. Frenet follows curvature and torsion;
            fixed keeps sections parallel in world space. Fixed binormal
            preserves their angular relation to the supplied world direction.
        binormal : str or tuple of float, optional
            Required only for fixed binormal transport. Must not be parallel
            to the spine tangent. Path's up direction does not set this value.

        Returns
        -------
        Solid
            Independent pipe retaining the section's holes.
        """
        from magnelio.geo.modifications import sweep  # noqa: PLC0415

        return sweep(
            self, spine, face_near=face_near, material=material, frame=frame, binormal=binormal
        )

    def shelled(self, thickness, *, opening_face_near=None, openings=None):
        """Hollow this Solid inward to a constant wall thickness.

        Parameters
        ----------
        thickness : float
            Positive wall thickness in metres. The outer footprint is retained.
        opening_face_near : tuple or list of tuple, optional
            World points selecting temporary face references to leave open.
            Tied picks raise. Mutually exclusive with ``openings``.
        openings : FaceRef or FaceSetRef or sequence, optional
            Owned faces to leave open; must belong to this exact receiver.
            Omit both selection modes for a sealed internal void.

        Returns
        -------
        Solid
            Hollow body with inherited material.

        Raises
        ------
        TypeError
            For a non-Solid receiver; grow a sheet with thickened() instead.
        RuntimeError
            If the offset cannot form a valid closed body.
        """
        from magnelio.geo.modifications import shell  # noqa: PLC0415

        return shell(
            self, thickness=thickness, opening_face_near=opening_face_near, openings=openings
        )

    def thickened(self, thickness, *, direction="forward", material=None):
        """Grow this sheet into a solid of constant thickness.

        A :class:`~magnelio.geo.Sheet`, including a planar
        :class:`~magnelio.geo.Profile` or curved :class:`~magnelio.geo.Surface`,
        can be thickened. A planar sheet
        becomes a slab whose footprint is exactly the sheet, which makes
        this the direct way from a drawn outline to a metallisation of a
        given thickness, without spelling out the extrusion vector.  A
        curved sheet is offset along its own normal into a shell of
        constant thickness. An unsuitable thickness, curvature or sampled
        surface can prevent the CAD kernel from building a valid offset.

        Parameters
        ----------
        thickness : float
            Slab thickness [meters], positive.
        direction : {"forward", "backward", "symmetric"}
            Which side of the sheet to grow on.  ``"symmetric"`` puts
            half the thickness on each side, leaving the sheet as the
            slab's mid-plane (planar sheets only).  ``"forward"`` and ``"backward"`` are
            opposite sides of it; forward follows the oriented sheet normal.
            For a FaceRef, forward follows its outward normal.
        material : Material or str, optional
            Material of the slab. Defaults to the sheet's material.
            A materialless sheet produces a construction solid for Boolean use.

        Returns
        -------
        Solid
            The solid slab.

        Raises
        ------
        TypeError
            If this is a solid — use :meth:`shelled` instead.
        ValueError
            If the thickness or direction is invalid, or symmetric thickening
            is requested for a curved sheet.

        Examples
        --------
        A copper patch from a drawn outline::

            patch = Profile.from_wires(outline).thickened(thickness=35e-6, material=copper)
        """
        from magnelio.geo.modifications import thicken  # noqa: PLC0415

        return thicken(self, thickness=thickness, direction=direction, material=material)

    def lofted(
        self,
        face_near,
        other=None,
        other_face_near=None,
        *,
        material=None,
        blend="spline",
        tension=None,
    ):
        """Connect this section to another with an independent Solid.

        For sheets, use ``section.lofted(other_section)``. Suitable FaceRef
        values provide the same verb. All boundaries contribute, and hole
        counts must agree. The retained Solid convenience is
        ``body.lofted(face_near, other_body, other_face_near)``; its temporary
        FaceRefs use semantic selection and refuse tied picks.

        Parameters
        ----------
        face_near : Profile or Sheet or FaceRef or tuple
            Planar end section for a Sheet receiver. For the Solid convenience,
            a world point in metres selecting its start face.
        other : Solid, optional
            End owner for the Solid point convenience only.
        other_face_near : tuple of float, optional
            World point selecting the end face for the Solid convenience.
        material : Material or str, optional
            Override the start section's material. Without a material the
            result is a construction solid.
        blend : {'spline', 'ruled', 'tangent'}, optional
            Smooth or straight interpolation, or a transition leaving both
            oriented normals. Tangent mode requires faces looking towards
            each other and retains holes with the same spine conditions.
        tension : float or tuple of float, optional
            Positive finite tangent reach fractions, only in tangent mode.
            Defaults to one third at each end.

        Returns
        -------
        Solid
            Independent transition matching corresponding boundaries.
        """
        from magnelio.geo.modifications import loft  # noqa: PLC0415

        if other is None:
            from magnelio.geo.modifications import loft_profiles

            return loft_profiles(self, face_near, material=material, blend=blend, tension=tension)
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

    def face(self, name=None, *, near=None, normal=None, surface_type=None):
        """Select one owned face, or retrieve a registered name.

        Parameters
        ----------
        name : str, optional
            Registered singular selection name; excludes semantic constraints.
        near : tuple of float, optional
            World point in metres; distance is measured to the complete subshape.
        normal : str or tuple of float, optional
            Outward unit normal constraint. Curved faces require near; without
            near this filters planar faces only.
        surface_type : str, optional
            plane, cylinder, cone, sphere, torus, bspline or other.

        Returns
        -------
        FaceRef
            Read-only view bound to this Solid. At least one semantic constraint
            is required for unnamed selection.

        Raises
        ------
        TopologySelectionError
            If no candidate or matching registered name exists.
        AmbiguousTopologyError
            If a singular selection has equally eligible candidates. Supply
            additional constraints or use a plural selector deliberately.
        """
        from magnelio.geo.topology import select

        return select(
            self, "face", name, near=near, normal=normal, surface_type=surface_type, plural=False
        )

    def tag_face(self, name, *, near=None, normal=None, surface_type=None):
        """Register an immutable named face selection.

        Parameters
        ----------
        name : str
            Non-empty name, unique within this topology kind on this owner.
        near : tuple of float, optional
            World point in metres; distance is measured to the complete subshape.
        normal : str or tuple of float, optional
            Outward unit normal constraint. Curved faces require near; without
            near this filters planar faces only.
        surface_type : str, optional
            plane, cylinder, cone, sphere, torus, bspline or other.

        Returns
        -------
        Solid
            New owner with the selection registered. The receiver is unchanged.
            Affine transforms preserve it; construction requires provable OCC
            successors and raises TopologyEvolutionError if identity is lost.
        """
        from magnelio.geo.topology import tag

        return tag(
            self, "face", name, near=near, normal=normal, surface_type=surface_type, plural=False
        )

    def faces(self, name=None, *, near=None, normal=None, surface_type=None):
        """Select a deliberate set of owned faces, or retrieve a registered name.

        Parameters
        ----------
        name : str, optional
            Registered set selection name; excludes semantic constraints.
        near : tuple of float, optional
            World point in metres; distance is measured to the complete subshape.
        normal : str or tuple of float, optional
            Outward unit normal constraint. Curved faces require near; without
            near this filters planar faces only.
        surface_type : str, optional
            plane, cylinder, cone, sphere, torus, bspline or other.

        Returns
        -------
        FaceSetRef
            Read-only view bound to this Solid. With no constraints, selects all members.

        Raises
        ------
        TopologySelectionError
            If no candidate or matching registered name exists.
        AmbiguousTopologyError
            If a singular selection has equally eligible candidates. Supply
            additional constraints or use a plural selector deliberately.
        """
        from magnelio.geo.topology import select

        return select(
            self, "face", name, near=near, normal=normal, surface_type=surface_type, plural=True
        )

    def tag_faces(self, name, *, near=None, normal=None, surface_type=None):
        """Register an immutable named set of faces.

        Parameters
        ----------
        name : str
            Non-empty name, unique within this topology kind on this owner.
        near : tuple of float, optional
            World point in metres; distance is measured to the complete subshape.
        normal : str or tuple of float, optional
            Outward unit normal constraint. Curved faces require near; without
            near this filters planar faces only.
        surface_type : str, optional
            plane, cylinder, cone, sphere, torus, bspline or other.

        Returns
        -------
        Solid
            New owner with the selection registered. The receiver is unchanged.
            Affine transforms preserve it; construction requires provable OCC
            successors and raises TopologyEvolutionError if identity is lost.
        """
        from magnelio.geo.topology import tag

        return tag(
            self, "face", name, near=near, normal=normal, surface_type=surface_type, plural=True
        )

    def edge(self, name=None, *, near=None, curve_type=None):
        """Select one owned edge, or retrieve a registered name.

        Parameters
        ----------
        name : str, optional
            Registered singular selection name; excludes semantic constraints.
        near : tuple of float, optional
            World point in metres; distance is measured to the complete subshape.
        curve_type : str, optional
            line, circle, ellipse, hyperbola, parabola, bezier, bspline or other.

        Returns
        -------
        EdgeRef
            Read-only view bound to this Solid. At least one semantic constraint
            is required for unnamed selection.

        Raises
        ------
        TopologySelectionError
            If no candidate or matching registered name exists.
        AmbiguousTopologyError
            If a singular selection has equally eligible candidates. Supply
            additional constraints or use a plural selector deliberately.
        """
        from magnelio.geo.topology import select

        return select(self, "edge", name, near=near, curve_type=curve_type, plural=False)

    def tag_edge(self, name, *, near=None, curve_type=None):
        """Register an immutable named edge selection.

        Parameters
        ----------
        name : str
            Non-empty name, unique within this topology kind on this owner.
        near : tuple of float, optional
            World point in metres; distance is measured to the complete subshape.
        curve_type : str, optional
            line, circle, ellipse, hyperbola, parabola, bezier, bspline or other.

        Returns
        -------
        Solid
            New owner with the selection registered. The receiver is unchanged.
            Affine transforms preserve it; construction requires provable OCC
            successors and raises TopologyEvolutionError if identity is lost.
        """
        from magnelio.geo.topology import tag

        return tag(self, "edge", name, near=near, curve_type=curve_type, plural=False)

    def edges(self, name=None, *, near=None, curve_type=None):
        """Select a deliberate set of owned edges, or retrieve a registered name.

        Parameters
        ----------
        name : str, optional
            Registered set selection name; excludes semantic constraints.
        near : tuple of float, optional
            World point in metres; distance is measured to the complete subshape.
        curve_type : str, optional
            line, circle, ellipse, hyperbola, parabola, bezier, bspline or other.

        Returns
        -------
        EdgeSetRef
            Read-only view bound to this Solid. With no constraints, selects all members.

        Raises
        ------
        TopologySelectionError
            If no candidate or matching registered name exists.
        AmbiguousTopologyError
            If a singular selection has equally eligible candidates. Supply
            additional constraints or use a plural selector deliberately.
        """
        from magnelio.geo.topology import select

        return select(self, "edge", name, near=near, curve_type=curve_type, plural=True)

    def tag_edges(self, name, *, near=None, curve_type=None):
        """Register an immutable named set of edges.

        Parameters
        ----------
        name : str
            Non-empty name, unique within this topology kind on this owner.
        near : tuple of float, optional
            World point in metres; distance is measured to the complete subshape.
        curve_type : str, optional
            line, circle, ellipse, hyperbola, parabola, bezier, bspline or other.

        Returns
        -------
        Solid
            New owner with the selection registered. The receiver is unchanged.
            Affine transforms preserve it; construction requires provable OCC
            successors and raises TopologyEvolutionError if identity is lost.
        """
        from magnelio.geo.topology import tag

        return tag(self, "edge", name, near=near, curve_type=curve_type, plural=True)

    def vertex(self, name=None, *, near=None):
        """Select one owned vertex, or retrieve a registered name.

        Parameters
        ----------
        name : str, optional
            Registered singular selection name; excludes semantic constraints.
        near : tuple of float, optional
            World point in metres; distance is measured to the complete subshape.

        Returns
        -------
        VertexRef
            Read-only view bound to this Solid. At least one semantic constraint
            is required for unnamed selection.

        Raises
        ------
        TopologySelectionError
            If no candidate or matching registered name exists.
        AmbiguousTopologyError
            If a singular selection has equally eligible candidates. Supply
            additional constraints or use a plural selector deliberately.
        """
        from magnelio.geo.topology import select

        return select(self, "vertex", name, near=near, plural=False)

    def tag_vertex(self, name, *, near=None):
        """Register an immutable named vertex selection.

        Parameters
        ----------
        name : str
            Non-empty name, unique within this topology kind on this owner.
        near : tuple of float, optional
            World point in metres; distance is measured to the complete subshape.

        Returns
        -------
        Solid
            New owner with the selection registered. The receiver is unchanged.
            Affine transforms preserve it; construction requires provable OCC
            successors and raises TopologyEvolutionError if identity is lost.
        """
        from magnelio.geo.topology import tag

        return tag(self, "vertex", name, near=near, plural=False)


def _is_shape(obj) -> bool:
    """True for anything the CSG operators can meaningfully combine.

    A Group passes on purpose: the Boolean constructors reject it with
    their own descriptive TypeError, which beats a generic
    'unsupported operand type' from returning NotImplemented.
    """
    from magnelio.geo.operations import Group  # noqa: PLC0415

    return hasattr(obj, "_occ_shape") or isinstance(obj, Group)
