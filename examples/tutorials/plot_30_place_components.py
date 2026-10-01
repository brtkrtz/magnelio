"""
30 — Place copies of a named component
======================================

Define a component once, then place independent copies with reusable affine
values. The named contact face follows its own shell through each placement.
"""

import numpy as np

from magnelio import geo

# %%
# Define a mixed component in world coordinates
# ---------------------------------------------

shell = geo.Brick(size=(2e-3, 1e-3, 1e-3), material="pec", name="shell")
shell = shell.tag_face("contact", normal="z")
insert = geo.Brick(
    origin=(0.5e-3, 0.25e-3, 0),
    size=(1e-3, 0.5e-3, 1e-3),
    material="air",
    name="insert",
)
datum = geo.Curve.line((0, 0, 0), (2e-3, 0, 0), name="datum")
component = geo.Group(geo.Group(shell, insert, name="layers"), datum, name="component")

# %%
# Compose each placement and retrieve the owned contact
# ------------------------------------------------------

poses = [geo.Translation((5e-3 * i, 0, 0)) @ geo.Rotation("z", 90 * i) for i in range(3)]
copies = [pose @ component for pose in poses]
shells = [next(copy.members()) for copy in copies]
contacts = [placed_shell.face("contact") for placed_shell in shells]

for pose, placed_shell, contact in zip(poses, shells, contacts):
    assert contact.owner is placed_shell
    assert np.allclose(contact.centroid, pose.point(shell.face("contact").centroid))
    assert np.isclose(contact.area, shell.face("contact").area)
    assert placed_shell.material == shell.material
assert np.allclose(shell.face("contact").centroid, (1e-3, 0.5e-3, 1e-3))

# Reflection and uniform scaling also retain the contact name. A factor of
# two multiplies its area by four and the shell volume by eight.
reflected = geo.Translation((15e-3, 0, 0)) @ geo.Mirror("x") @ geo.Scale(2) @ component
reflected_shell = next(reflected.members())
assert np.isclose(reflected_shell.face("contact").area, 4 * shell.face("contact").area)
assert np.isclose(reflected_shell.volume(), 8 * shell.volume())
