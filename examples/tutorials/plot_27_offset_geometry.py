"""
27 — Clearance profiles and normal-offset construction sheets
=============================================================

Offsets move geometry by a distance, rather than scaling it about a centre.
A planar profile grows around its material boundary; a curved sheet follows
its local normal. Both operations return independent geometry.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from magnelio import geo

# %%
# Give a plated annulus a clearance
# ---------------------------------
#
# The exterior radius grows while the hole radius shrinks. The result stays
# a zero-thickness profile until a physical thickness is constructed.

outer_radius = 2e-3
hole_radius = 1e-3
clearance_distance = 0.1e-3
annulus = geo.Profile.from_wires(
    geo.Curve.circle((0, 0, 0), outer_radius),
    holes=[geo.Curve.circle((0, 0, 0), hole_radius)],
    material="pec",
)
(clearance,) = annulus.offset(clearance_distance)
expected_area = math.pi * (
    (outer_radius + clearance_distance) ** 2 - (hole_radius - clearance_distance) ** 2
)
assert math.isclose(clearance.area, expected_area, rel_tol=1e-9)
assert math.isclose(annulus.area, math.pi * (outer_radius**2 - hole_radius**2))
spacer = clearance.thickened(35e-6)
assert spacer.volume() > 0

# A directed open trace has a different sign rule: positive is to the left
# when viewed along the chosen plane normal, and its ends stay uncapped.
feed = geo.Curve.line((0, 0, 0), (3e-3, 0, 0))
(feed_clearance,) = feed.offset(clearance_distance, normal="z")
assert math.isclose(feed_clearance.length, feed.length)
assert math.isclose(feed_clearance.bounding_box()[0][1], clearance_distance)

# %%
# Move a bounded housing wall along its normal
# ---------------------------------------------
#
# The selected cylinder face remains owned by the housing. Detaching it
# creates the standalone sheet that can be offset for another construction.

housing_radius = 5e-3
housing = geo.Cylinder(radius=housing_radius, height=10e-3)
wall = housing.face(near=(housing_radius, 0, 5e-3))
(outer_sheet,) = wall.detached().offset(0.2e-3)
assert wall.owner is housing
assert abs(outer_sheet.bounding_box()[1][0] - 5.2e-3) < 2e-6

# The drawing is a cross-section of the analytically checked boundaries.
# Radii come from the geometry parameters above; it is not a solver plot.
angles = np.linspace(0, 2 * math.pi, 256)
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for radius, style, label in (
    (outer_radius, "--", "original exterior"),
    (outer_radius + clearance_distance, "-", "clearance exterior"),
    (hole_radius, "--", "original hole"),
    (hole_radius - clearance_distance, "-", "clearance hole"),
):
    axes[0].plot(radius * np.cos(angles) * 1e3, radius * np.sin(angles) * 1e3, style, label=label)
for radius, style, label in (
    (housing_radius, "--", "housing wall"),
    (housing_radius + 0.2e-3, "-", "normal-offset sheet"),
):
    axes[1].plot(radius * np.cos(angles) * 1e3, radius * np.sin(angles) * 1e3, style, label=label)
for ax in axes:
    ax.set(xlabel="x (mm)", ylabel="y (mm)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
axes[0].set_title("Planar region clearance")
axes[1].set_title("Curved sheet offset")
fig.tight_layout()
