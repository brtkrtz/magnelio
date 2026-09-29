"""
Topology selection: a coax end face follows its owner
=====================================================

An end face has a physical role before the body is placed. Name that role,
rotate the body and retrieve the same face in its new placement. This tutorial
uses a coax shield: its selected end is an annulus, so detaching it also shows
that the bore is part of the face topology.

This is geometry only and takes seconds. The face reference stays bound to its
owner; its construction verbs provide independent solids for an extension.
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

shield = geo.Cylinder(radius=2e-3, inner_radius=1e-3, height=6e-3, material="pec")
shield = shield.tag_face("port", near=(1.5e-3, 0, 6e-3), normal="z")
shield = shield.rotated("y", 30)
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
spine = geo.Curve.line(
    port.centroid, tuple(c + 3e-3 * n for c, n in zip(port.centroid, port.normal))
)
extension = port.swept(spine)
assert math.isclose(extension.volume(), port.area * 3e-3, rel_tol=1e-12)

fig, ax = plots.plot_cross_section(
    [shield, extension], "y", 0, title="Named coax end face: the placed bore continues"
)

# %%
# Exact bend from a placed face
# -------------------------------
#
# An absolute arc already determines the bend geometrically. Here the coax
# starts along +y, turns toward -x and retains its bore. The length times the
# face area is an independent check on the resulting metal volume.

bend_shield = geo.Cylinder(
    origin=(8e-3, 0, 0), axis="y", radius=2e-3, inner_radius=1e-3, height=-3e-3, material="pec"
)
bend_port = bend_shield.face(normal="y")
bend_spine = geo.Curve.arc((8e-3, 0, 0), (8e-3 / 2**0.5, 8e-3 / 2**0.5, 0), (0, 8e-3, 0))
elbow = bend_port.swept(bend_spine)
assert math.isclose(elbow.volume(), bend_port.area * bend_spine.length, rel_tol=1e-9)
fig, ax = plots.plot_cross_section(
    [bend_shield, elbow], "z", 0, title="Face-based coax sweep: the bore follows the bend"
)

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
