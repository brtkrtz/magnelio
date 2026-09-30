"""
26 — Project a conductor path onto a bounded housing face
==========================================================

A routed centreline in a world plane can be mapped to a curved housing wall.
Projection uses the selected face, including its physical boundary. Parallel
rays, perspective rays and nearest-point projection are separate choices.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from magnelio import geo

# %%
# Select the actual curved face
# -----------------------------
#
# The construction cylinder supplies an owned FaceRef. The projected Curve
# is independent of that owner. A directed ray from the sketch first reaches
# the near wall; the optional all_hits result also includes the far wall.

radius = 5e-3
housing = geo.Cylinder(radius=radius, height=10e-3)
wall = housing.face(near=(radius, 0, 5e-3))
sketch = geo.Curve.line((8e-3, -2e-3, 5e-3), (8e-3, 2e-3, 5e-3))
(trace,) = sketch.projected_onto(wall, direction=(-1, 0, 0))
both = sketch.projected_onto(wall, direction=(-1, 0, 0), all_hits=True)
assert len(both) == 2
assert math.isclose(trace.length, 2 * radius * math.asin(2e-3 / radius))
assert wall.owner is housing

# A thin PEC wire can use the projected centreline. Its radius is a physical
# sub-cell wire parameter; the Curve itself carries no material.
wire = geo.ThinWire(trace, radius=20e-6)
assert wire.curve is trace

# This cross-section draws the analytic cylinder reference. The assertions
# above check the exact projected CAD curve used by the wire declaration.
y = np.linspace(-2e-3, 2e-3, 200)
x = np.sqrt(radius**2 - y**2)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.full_like(y, 8.0), y * 1e3, "--", label="source sketch")
ax.plot(x * 1e3, y * 1e3, label="first-hit wall")
ax.plot(-x * 1e3, y * 1e3, ":", label="second wall")
ax.set(xlabel="x (mm)", ylabel="y (mm)", title="Directed projection at z = 5 mm")
ax.set_aspect("equal")
ax.legend()
fig.tight_layout()

# %%
# Keep only the portion inside a bounded patch
# --------------------------------------------
#
# A hole is not part of the target. The default call reports partial
# coverage; clip=True explicitly keeps two independent curve pieces.

outer = geo.Curve.polyline(
    [(-2e-3, -2e-3, 0), (2e-3, -2e-3, 0), (2e-3, 2e-3, 0), (-2e-3, 2e-3, 0), (-2e-3, -2e-3, 0)]
)
hole = geo.Curve.circle((0, 0, 0), 0.5e-3)
patch = geo.Profile.from_wires(outer, holes=[hole])
across = geo.Curve.line((-1e-3, 0, 1e-3), (1e-3, 0, 1e-3))
pieces = across.projected_onto(patch, direction=(0, 0, -1), clip=True)
assert len(pieces) == 2
assert all(math.isclose(piece.length, 0.5e-3) for piece in pieces)
