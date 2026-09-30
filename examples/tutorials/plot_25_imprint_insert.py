"""
25 — Imprint a selectable face and insert a dielectric
======================================================

Imprint divides the chosen receiver's boundary faces without changing its
volume. Insert assigns every overlapping volume to an explicitly prioritized
material body. A material-less void removes volume from all bodies.
"""

import math

import matplotlib.pyplot as plt

from magnelio import GeometryModel, geo, plots

# %%
# Mark a contact on the housing's outside face
# --------------------------------------------
#
# The construction tool crosses the outer x face. Its intersection divides
# that face into selectable regions while the housing remains one Solid.

side = 2e-3
housing = geo.Brick(origin=(0, 0, 0), size=(side,) * 3, material="pec")
window_tool = geo.Brick(origin=(1.8e-3, 0.5e-3, 0.5e-3), size=(0.4e-3, 1e-3, 1e-3))
marked = housing.imprint(window_tool)
assert math.isclose(marked.volume(), housing.volume(), rel_tol=1e-10)
assert len(marked.faces()) > len(housing.faces())
marked = marked.tag_face("contact", near=(side, 1e-3, 1e-3), normal="x")
assert math.isclose(marked.face("contact").area, 1e-6, rel_tol=1e-10)

# %%
# Give an enclosed dielectric precedence over the metal
# ------------------------------------------------------
#
# The host loses precisely the dielectric volume. Both material bodies
# remain in the group and may enter a GeometryModel in either order.

dielectric = geo.Brick(origin=(0.5e-3,) * 3, size=(1e-3,) * 3, material="air")
assembly = geo.insert(marked, dielectric, priorities=(0, 1))
metal, dielectric_region = tuple(assembly.members())
assert math.isclose(metal.volume(), 7e-9, rel_tol=1e-10)
assert math.isclose(dielectric_region.volume(), 1e-9, rel_tol=1e-10)
assert metal.face("contact").area == marked.face("contact").area
GeometryModel().add(assembly).validate()

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
plots.plot_cross_section(
    [marked.face("contact").detached()],
    "x",
    side,
    ax=axes[0],
    title="Selected imprinted contact",
)
plots.plot_cross_section(
    list(assembly.members()), "z", 1e-3, ax=axes[1], title="Dielectric wins overlap"
)
fig.tight_layout()

# %%
# A void removes material without becoming a material body
# ---------------------------------------------------------
#
# This tool reaches the dielectric and cuts a smaller hole through it.

void = geo.Brick(origin=(0.75e-3,) * 3, size=(0.5e-3,) * 3)
drilled = geo.insert(marked, dielectric, priorities=(0, 1), voids=(void,))
assert len(tuple(drilled.members())) == 2
assert math.isclose(
    sum(part.volume() for part in drilled.members()),
    housing.volume() - void.volume(),
    rel_tol=1e-10,
)
