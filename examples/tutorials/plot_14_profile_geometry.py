"""
Profiles: shapes the primitives do not cover
============================================

Bricks, cylinders, cones and spheres carry most models, and Boolean
operations carry most of the rest.  Sooner or later a part resists both:
a curved electrode that is a slice of a tube, a pad with one rounded
end, a track that follows a route, a taper between two different
cross-sections.  Cutting those out of primitives means inventing tool
bodies whose only purpose is to be subtracted, and getting their
dimensions right by hand.

The way out is the one a draughtsman uses: draw the outline, then give
it a third dimension.  This tutorial builds that path — a curve, a
closed profile, a solid — and then the three verbs that operate on
whole shapes rather than outlines: hollowing, tracking, and lofting.

Nothing here runs a simulation; it is geometry only, and takes seconds.
"""

import math

import magnelio as mio
from magnelio import geo, plots

copper = mio.Material.lossy_metal(name="copper", sigma=5.8e7)

# %%
# A worked example: a curved electrode
# ------------------------------------
#
# A 20 degree slice of a tube — 24 mm across, 2 mm wall, 30 mm tall.
# Two independent ways to build it, and it is worth seeing both, because
# the choice between them comes up again for every part.
#
# The first is a primitive.  A cylinder takes a bore and an angular
# extent, so the electrode is one call:

R_OUT, R_IN, HEIGHT, SPAN = 12.0e-3, 10.0e-3, 30.0e-3, 20.0

electrode = geo.Cylinder(
    radius=R_OUT,
    inner_radius=R_IN,
    height=HEIGHT,
    angle_deg=(0.0, SPAN),
    material="pec",
)

fig, ax = plots.plot_cross_section(
    [electrode], "z", HEIGHT / 2, title="electrode, cut across the axis"
)

# %%
# The same part, drawn as a profile
# ---------------------------------
#
# The second way draws the cross-section and extrudes it.  A
# :class:`~magnelio.geo.Path` is a pen: it starts at a point and
# remembers where the last segment ended, so each call only names where
# the segment goes.
#
# Two of the four sides are arcs about the axis.  An arc through a
# centre has two solutions — the short way round and the long way — so
# ``normal=`` names the axis the arc turns about, and the arc runs
# counter-clockwise about it.  That is the same sense as
# :meth:`~magnelio.geo.Shape.rotated`, and it settles the ambiguity even
# for a half-circle, where the two ends alone say nothing at all.

CENTRE = (0.0, 0.0, 0.0)


def on_circle(radius, angle_deg):
    """A point on a circle about the origin, in the z = 0 plane."""
    angle = math.radians(angle_deg)
    return (radius * math.cos(angle), radius * math.sin(angle), 0.0)


outline = (
    geo.Path(on_circle(R_IN, 0.0))
    .line_to(on_circle(R_OUT, 0.0))
    .arc_to(on_circle(R_OUT, SPAN), center=CENTRE, normal="z")
    .line_to(on_circle(R_IN, SPAN))
    .arc_to(on_circle(R_IN, 0.0), center=CENTRE, normal=(0.0, 0.0, -1.0))
    .closed()
)

section = geo.Profile.from_wires(outline)
drawn = section.extruded(vector=(0.0, 0.0, HEIGHT), material="pec")

# The two routes describe the same solid, and `volume()` is the way to
# say so: it reports what the CAD kernel actually built, not what the
# parameters nominally asked for.  They are built by different kernel
# operations, so they agree to numerical precision rather than bit for
# bit -- the level of agreement to expect whenever one part can be
# built two ways.
print(f"primitive: {electrode.volume() * 1e9:.6f} mm^3")
print(f"drawn:     {drawn.volume() * 1e9:.6f} mm^3")
print(f"relative difference: {abs(drawn.volume() / electrode.volume() - 1.0):.2e}")

# %%
# Curves, profiles and solids share one placement grammar
# -------------------------------------------------------
#
# The outline is a :class:`~magnelio.geo.Curve`, its bounded region is a
# :class:`~magnelio.geo.Profile`, and the extrusion is a
# :class:`~magnelio.geo.Solid`.  All three accept the same named transform
# methods.  An immutable transform value is useful when the same placement
# belongs to several pieces of a component.  In ``A @ B @ geometry``, ``B``
# acts first:

placement = geo.Translation((30.0e-3, 0.0, 0.0)) @ geo.Rotation("z", 15.0)
placed_outline = placement @ outline
placed_section = placement @ section
placed_electrode = placement @ drawn

print(
    isinstance(placed_outline, geo.Curve),
    isinstance(placed_section, geo.Profile),
    isinstance(placed_electrode, geo.Solid),
)

# Regular arrays use the named methods' convenience options. Seven rotated
# copies plus the original make eight electrodes, fused into one PEC body.
# Curves can use the same placement pattern with ``group=True``, retaining
# their one-dimensional category. A mirrored pair needs only ``copy=True``.

electrodes = drawn.rotated("z", 45, repeat=7, copy=True, unite=True)
outlines = outline.rotated("z", 45, repeat=7, copy=True, group=True)
electrode_pair = drawn.mirrored("x", copy=True, group=True)

# %%
# .. note::
#
#    The inner arc turns about ``-z`` while the outer turns about
#    ``+z``.  The pen walks the outline as a loop, so it comes back
#    along the inside — and "counter-clockwise" is then the other way
#    round.  If a profile comes out crossing itself, this is the first
#    thing to check.
#
# Which route to prefer?  The primitive, whenever it fits: it is one
# line, it carries its dimensions as named parameters, and the mesher
# gets an exact analytic surface.  The profile route earns its keep the
# moment the outline is not a plain sector — a broken edge, a keyway, a
# flat on one side.  Adding a chamfer to the inner corner is one more
# segment in the pen stroke, and no new tool body:

CHAMFER = 1.0e-3

chamfered = geo.Profile.from_wires(
    geo.Path(on_circle(R_IN, 0.0))
    .line_to(on_circle(R_OUT - CHAMFER, 0.0))
    .line_to((*on_circle(R_OUT, 0.0)[:2], 0.0))
    .arc_to(on_circle(R_OUT, SPAN), center=CENTRE, normal="z")
    .line_to(on_circle(R_IN, SPAN))
    .arc_to(on_circle(R_IN, 0.0), center=CENTRE, normal=(0.0, 0.0, -1.0))
    .closed()
).extruded(vector=(0.0, 0.0, HEIGHT), material="pec")

fig, ax = plots.plot_cross_section(
    [chamfered], "z", HEIGHT / 2, title="the same electrode, drawn and modified"
)

# %%
# A profile can also be revolved or swept
# ---------------------------------------
#
# The profile is the input, not the extrusion.  The same closed curve
# feeds three verbs: :meth:`~magnelio.geo.Shape.extruded` pushes it
# along a vector, :meth:`~magnelio.geo.Shape.revolved` turns it about an
# axis, and :meth:`~magnelio.geo.Shape.swept` runs it along a path.
#
# Revolving is the direct way to any rotationally symmetric part whose
# outline is not a rectangle — a rounded-nose centre conductor, a
# stepped transformer, a bead.  Here the outline is drawn in the x-z
# plane and turned about z:

nose = geo.Profile.from_wires(
    geo.Path((0.0, 0.0, 0.0))
    .line_to((0.003, 0.0, 0.0))
    .line_to((0.003, 0.0, 0.008))
    .arc_to((0.0, 0.0, 0.011), via=(0.0021, 0.0, 0.0101))
    .closed()
).revolved(axis="z", material="pec")

fig, ax = plots.plot_cross_section([nose], "y", 0.0, title="revolved profile: a rounded pin")

# %%
# Exact boundaries and intrinsic holes
# -------------------------------------
#
# A circle or ellipse is one exact analytic wire, not a many-sided polygon.
# A Profile bounds the region inside it. Holes belong to that region, so
# extruding or sweeping the profile carries the bore without a separate cut.
# Here an annulus sweeps round a half-circle into a hollow elbow:

outer = geo.Curve.circle((0, 0, 0), 2e-3)
inner = geo.Curve.circle((0, 0, 0), 1e-3)
annulus = geo.Profile.from_wires(outer, holes=[inner], material="pec")
spine = geo.Curve.arc((8e-3, 0, 0), (0, 8e-3, 0), (-8e-3, 0, 0))
elbow = annulus.swept(spine)

fig, ax = plots.plot_cross_section([elbow], "z", 0.0, title="an exact hollow elbow")
print(
    f"elbow volume / (profile area x spine length): "
    f"{elbow.volume() / (annulus.area * spine.length):.8f}"
)

# An ellipse and an explicitly oriented rectangle need only world vectors.
# The first semi-axis follows ``major_axis`` in the plane of ``normal``;
# rectangle width follows ``x_direction`` and height ``normal x x_direction``.
# Extracted boundaries remain independent, transformable Curve values.

ellipse = geo.Curve.ellipse((0, 0, 0), (3e-3, 2e-3), major_axis="x")
elliptical_pad = geo.Profile.from_wires(ellipse).extruded((0, 0, 35e-6), material=copper)
rectangle = geo.Profile.rectangle((0, 0, 0), (6e-3, 4e-3), normal=(1, 1, 1), x_direction=(1, -1, 0))
(edge_loop,) = rectangle.boundary()
print(f"ellipse area: {geo.Profile.from_wires(ellipse).area * 1e6:.6f} mm^2")
print(f"rectangle perimeter: {edge_loop.length * 1e3:.6f} mm")

# %%
# Hollowing a solid
# -----------------
#
# A housing is a block with its inside removed, and the inside of
# anything but a box is not the outside scaled down.
# :meth:`~magnelio.geo.Shape.shelled` builds it directly: walls grow
# inward, so the outer dimensions stay exactly as designed, and naming a
# face leaves it out of the shell to become an opening.

WALL = 1.5e-3
BOX = (40.0e-3, 25.0e-3, 12.0e-3)

housing = geo.Brick(origin=(0.0, 0.0, 0.0), size=BOX, material="pec").shelled(
    thickness=WALL,
    opening_face_near=[(0.0, BOX[1] / 2, BOX[2] / 2), (BOX[0], BOX[1] / 2, BOX[2] / 2)],
)

fig, ax = plots.plot_cross_section([housing], "z", BOX[2] / 2, title="a housing, open at both ends")

# %%
# Tracks that follow a route
# --------------------------
#
# A feed line is a centreline with a width and a metal thickness.
# :meth:`~magnelio.geo.Curve.traced` takes it that way, so a bend is one
# segment of the path rather than a separate body to place.
#
# ``caps="flat"`` matters more than it looks: a track that ends at a
# port has to meet the port plane squarely, and the default rounded end
# would leave a sliver of air there.  Outside corners come out rounded,
# which is what a fabricated track does too.

W_TRACK, T_COPPER = 0.6e-3, 35.0e-6

route = (
    geo.Path((0.0, 0.0, 0.0))
    .line_to((8.0e-3, 0.0, 0.0))
    .spline_to((14.0e-3, 3.0e-3, 0.0), (20.0e-3, 3.0e-3, 0.0))
    .curve()
)

track = route.traced(
    width=W_TRACK,
    thickness=T_COPPER,
    caps="flat",
    normal="z",
    material=copper,
)

fig, ax = plots.plot_cross_section(
    [track], "z", T_COPPER / 2, title="a routed track, square at both ends"
)

# %%
# Tapers between two cross-sections
# ---------------------------------
#
# The last gap the primitives leave is a transition: a horn, a matching
# section, a change from a round cross-section to a rectangular one.
# :class:`~magnelio.geo.Loft` takes the cross-sections themselves, in
# the order the solid passes through them, and interpolates.
#
# ``blend="ruled"`` joins them with straight surfaces, which is what a
# machined taper is; the default ``blend="spline"`` passes one smooth
# surface through all of them, for a flared horn.
#
# Where the two ends are faces of solids that already exist, the
# :meth:`~magnelio.geo.FaceRef.lofted` verb takes those instead, and adds a
# third mode: ``blend="tangent"`` leaves each face along its own normal,
# so the transition meets both parts without a crease.  Between two
# faces that look at each other -- the two ends of a waveguide taper --
# the cross-section eases out of one profile and into the other, with
# zero wall slope at both flanges; between faces that point in different
# directions the profile is swept round a curved path instead.


def square(half, z):
    """A square outline of half-width *half*, at height *z*."""
    return geo.Profile.rectangle((0, 0, z), (2 * half, 2 * half))


taper = geo.Loft(square(4.0e-3, 0.0), square(10.0e-3, 18.0e-3), blend="ruled", material="pec")

fig, ax = plots.plot_cross_section([taper], "y", 0.0, title="a ruled taper between two squares")

# %%
# A selected face is a section too
# --------------------------------
#
# The same loft grammar consumes a Profile or the end face of an existing body.
# Both annular faces contribute their bore, so the transition is hollow.

start_body = geo.Cylinder(radius=2e-3, inner_radius=1e-3, height=2e-3, material="pec")
end_body = geo.Cylinder(
    origin=(0, 0, 8e-3), radius=3e-3, inner_radius=1.5e-3, height=2e-3, material="pec"
)
transition = start_body.face(normal="z").lofted(end_body.face(normal=(0, 0, -1)), blend="ruled")
assert transition.volume() > 0
fig, ax = plots.plot_cross_section(
    [start_body, transition, end_body],
    "y",
    0,
    title="Face-to-face loft: matching annular boundaries",
)

# %%
# What to take away
# -----------------
#
# * Reach for a primitive first — a cylinder with ``inner_radius`` and
#   ``angle_deg`` covers tubes and sectors without any drawing.
# * When the outline is the thing you actually know, draw it with
#   :class:`~magnelio.geo.Path`, close it, and pass it to ``Profile.from_wires``.  The resulting
#   sheet is a profile for ``extruded`` / ``revolved`` / ``swept`` /
#   ``thickened``.
# * An arc through a centre is ambiguous; ``normal=`` removes the
#   ambiguity, and the direction reverses when the pen comes back along
#   the far side of a loop.
# * ``ellipse_to`` draws an elliptical arc the same way — centre, the
#   two semi-axes and the direction of the first, ``normal=`` for the
#   sense — for the outlines of accelerator cells and lens profiles.
# * ``shelled`` hollows, ``traced`` follows a route, ``Loft``
#   interpolates cross-sections.  Each replaces a construction that
#   would otherwise be assembled from tool bodies by hand.
# * :meth:`~magnelio.geo.Shape.volume` checks a construction against
#   what it was supposed to be — the metal fraction of a housing, or
#   two routes to the same part agreeing.
#
# A profile carrying no material is a *construction* profile: it is not
# a physical object and cannot be meshed on its own. Its solid-producing
# verbs also allow materialless Boolean tools. Supply ``material=`` to the
# operation when the result is intended for model assembly.
