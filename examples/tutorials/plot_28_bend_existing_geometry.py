"""
28 — Bend a layered component around one neutral surface
=========================================================

A bend can deform existing solids and sheets together. Every layer follows the
same neutral surface: outer layers grow longer, inner layers become shorter,
and the straight material outside the affected interval stays attached.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from magnelio import geo

# %%
# Declare the common neutral surface
# ----------------------------------
#
# The surface parameters correspond to distance along the original part and
# distance across it. Its starting edge lies on the unchanged straight part.

radius = 10e-3
angle = math.pi / 2
length = radius * angle
width = 2e-3
half_width = width / 2
layer = 0.2e-3
straight = 3e-3

neutral = geo.Surface.parametric(
    lambda u, v: (
        radius * np.sin(u / radius),
        v,
        radius * (1 - np.cos(u / radius)),
    ),
    u=(0, length),
    v=(-half_width, half_width),
    samples=(65, 9),
)
bend = geo.Bend(
    neutral,
    origin=(0, 0, 0),
    along="x",
    across="y",
    u=(0, length),
    v=(-half_width, half_width),
    max_strain=0.01,
)

# %%
# Bend both physical layers with the same mapping
# ------------------------------------------------
#
# The upper layer lies nearer the centre of curvature after bending. There
# are straight continuations on both sides of the curved interval.

outer = geo.Brick(
    origin=(-straight, -half_width, -layer),
    size=(length + 2 * straight, width, layer),
    material="pec",
)
inner = geo.Brick(
    origin=(-straight, -half_width, 0),
    size=(length + 2 * straight, width, layer),
    material="air",
)
assembly = bend @ geo.Group(outer, inner, name="laminated bend")
outer_bent, inner_bent = tuple(assembly.members())


def expected_volume(w0, w1):
    curved = width * angle * ((radius - w0) ** 2 - (radius - w1) ** 2) / 2
    attached = 2 * straight * width * (w1 - w0)
    return curved + attached


assert math.isclose(outer_bent.volume(), expected_volume(-layer, 0), rel_tol=1e-5)
assert math.isclose(inner_bent.volume(), expected_volume(0, layer), rel_tol=1e-5)
assert outer_bent.volume() > inner_bent.volume()
assert outer_bent.material == outer.material
assert inner_bent.material == inner.material

# %%
# Read the bending construction in cross-section
# ------------------------------------------------
#
# The drawing shows the declared neutral arc and the two layer boundaries.
# It is a construction sketch; the volume checks above use the CAD results.

phi = np.linspace(0, angle, 200)
fig, ax = plt.subplots(figsize=(6, 5))
for w, label, style in (
    (-layer, "outer layer", "-"),
    (0, "neutral surface", "--"),
    (layer, "inner layer", "-"),
):
    r = radius - w
    ax.plot(r * np.sin(phi) * 1e3, (radius - r * np.cos(phi)) * 1e3, style, label=label)
    ax.plot([-straight * 1e3, 0], [w * 1e3, w * 1e3], style, color="0.5")
    ax.plot([r * 1e3, r * 1e3], [radius * 1e3, (radius + straight) * 1e3], style, color="0.5")
ax.set(xlabel="x (mm)", ylabel="z (mm)", title="One neutral surface for both layers")
ax.set_aspect("equal")
ax.legend()
fig.tight_layout()
