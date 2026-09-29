"""
Topology selection: a coax end face follows its owner
=====================================================

An end face has a physical role before the body is placed. Name that role,
rotate the body and retrieve the same face in its new placement. This tutorial
uses a coax shield: its selected end is an annulus, so detaching it also shows
that the bore is part of the face topology.

This is geometry only and takes seconds. The face reference stays bound to its
owner; its construction verbs provide independent solids for an extension.
An explicit path pose then bends the placed face towards a domain plane.
"""

import math

from magnelio import geo, plots

# %%
# Name before placement
# ----------------------
#
# ``near`` measures distance to the complete face. ``normal`` fixes the
# oriented end, even if the near point also approaches another boundary.
# Registration returns a new body; the untagged input remains unchanged.

shield = geo.Cylinder(axis="x", radius=2e-3, inner_radius=1e-3, height=6e-3, material="pec")
shield = shield.tag_face("port", near=(6e-3, 1.5e-3, 0), normal="x")
shield = shield.rotated("z", 22.5)
port = shield.face("port")

assert math.isclose(port.area, 3 * math.pi * 1e-6, rel_tol=1e-12)
assert port.is_planar
assert len(port.edges) == 2

# %%
# Sweep the selected face directly
# --------------------------------
#
# A FaceRef has an owner and no independent placement methods. Detachment
# returns a Profile when independent placement is needed. To continue the
# existing shield, consume the FaceRef directly. Its actual annular boundary
# and owner material feed the sweep; no profile reconstruction is required.

profile = port.detached()
spine = geo.Path.from_face(port, up="z").forward(3e-3).curve()
extension = port.swept(spine)
assert math.isclose(extension.volume(), port.area * 3e-3, rel_tol=1e-12)

fig, ax = plots.plot_cross_section(
    [shield, extension], "z", 0, title="Named coax end face: the placed bore continues"
)

# %%
# Relative bend to a domain plane
# --------------------------------
#
# ``from_face`` starts at the annulus centroid, along its outward normal.
# With up along +z, right points towards ``tangent x up``. A 22.5-degree
# right turn straightens the oblique coax towards +x; the following run ends
# at the world plane x = 20 mm. Radius fixes the exact circular bend.
# The original path can be branched because its pose is immutable.

XMAX = 20e-3
BEND_RADIUS = 8e-3
route = (
    geo.Path.from_face(port, up="z")
    .turn_right(radius=BEND_RADIUS, angle_deg=22.5)
    .straight_to_plane(normal="x", position=XMAX)
)
bend_spine = route.curve()
elbow = port.swept(bend_spine)
assert math.isclose(route.current[0], XMAX, rel_tol=1e-12)
assert all(math.isclose(t, e, abs_tol=1e-12) for t, e in zip(route.tangent, (1, 0, 0)))
assert math.isclose(elbow.volume(), port.area * bend_spine.length, rel_tol=1e-9)
outlet = elbow.face(normal="x")
assert len(outlet.edges) == 2
assert math.isclose(outlet.centroid[0], XMAX, rel_tol=1e-12)
fig, ax = plots.plot_cross_section(
    [shield, elbow], "z", 0, title="Relative coax bend: the open bore reaches the domain plane"
)
ax.axvline(XMAX * 1e3, color="gray", linestyle="--", label="Domain plane")
xmin, xmax = ax.get_xlim()
margin = 0.05 * (xmax - xmin)
ax.set_xlim(xmin - margin, xmax + margin)
ax.legend()

# %%
# Referenced openings and boundary edges
# ---------------------------------------
#
# References passed to an owner modification must belong to that exact body.
# Shell openings take faces; fillet and chamfer take faces or their edges.

housing_block = geo.Brick(size=(10e-3, 8e-3, 6e-3), material="pec")
opening = housing_block.face(normal="z")
housing = housing_block.shelled(0.5e-3, openings=opening)
rounded = housing_block.filleted(edges=opening.edges, radius=0.2e-3)
bevelled = housing_block.chamfered(faces=opening, distance=0.2e-3)
assert housing.volume() < housing_block.volume()
assert rounded.volume() < housing_block.volume()
assert bevelled.volume() < housing_block.volume()

# %%
# Deliberate sets for a split
# ----------------------------
#
# A singular name cannot silently become one of several faces. A deliberate
# set says that every successor belongs to the named region. Cutting a slot
# through this block leaves two cap faces, both retained under ``cap``.

block = geo.Brick(size=(4e-3, 4e-3, 2e-3), material="pec").tag_faces("cap", normal="z")
slot = geo.Brick(origin=(1.8e-3, -1e-3, -1e-3), size=(0.4e-3, 6e-3, 4e-3))
split = block - slot
assert len(split.faces("cap")) == 2

fig, ax = plots.plot_cross_section(
    [split], "z", 1e-3, title="A named cap set follows both sides of the slot"
)

# %%
# Retaining a selection
# ----------------------
#
# * Use a normal or analytic type constraint to distinguish a physical face.
#   A point on a shared boundary can tie; ambiguous singular picks raise.
# * Name before placement, transform the owner and look up the name there.
# * Read measurements from references; detach explicitly for standalone shapes.
# * Construction follows kernel history. A lost or unprovable name raises at
#   construction; a deliberate set can retain multiple successors.
# * Project geometry replays named histories when read back.
