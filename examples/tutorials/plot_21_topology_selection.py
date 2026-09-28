"""
Topology selection: a coax end face follows its owner
=====================================================

An end face has a physical role before the body is placed. Name that role,
rotate the body and retrieve the same face in its new placement. This tutorial
uses a coax shield: its selected end is an annulus, so detaching it also shows
that the bore is part of the face topology.

This is geometry only and takes seconds. The face reference stays bound to its
owner; a detached profile provides the standalone geometry for an extension.
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
# Detach to continue the shield
# ------------------------------
#
# A FaceRef has an owner and no independent placement methods. Detachment
# returns a Profile with the actual orientation and bore, carrying the owner's
# material. Extrude it along the face's outward normal to grow a continuation.

profile = port.detached()
extension = profile.extruded(tuple(3e-3 * component for component in port.normal))
assert math.isclose(extension.volume(), port.area * 3e-3, rel_tol=1e-12)

fig, ax = plots.plot_cross_section(
    [shield, extension], "y", 0, title="Named coax end face: the placed bore continues"
)

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
