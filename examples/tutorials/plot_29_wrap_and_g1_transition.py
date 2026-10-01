"""
29 — Wrap a layered trace and make a tangent transition
========================================================

Map a flat substrate and its conductor to one doubly curved patch. The same
chart keeps their placement together; a declared strain limit bounds the
surface deformation. A separate face-to-face tangent loft makes a smooth
transition between prismatic parts.
"""

import matplotlib.pyplot as plt
import numpy as np

from magnelio import geo

# %%
# Use one chart for the substrate and conductor
# ----------------------------------------------

length = 20e-3
half_width = 5e-3
rise = 0.25e-3


def patch(u, v):
    return (u, v, rise * np.sin(np.pi * u / length) * np.cos(np.pi * v / (2 * half_width)))


target = geo.Surface.parametric(patch, u=(0, length), v=(-half_width, half_width), samples=(41, 21))
wrap = geo.Wrap(
    target,
    origin=(0, 0, 0),
    along="x",
    across="y",
    u=(0, length),
    v=(-half_width, half_width),
    max_strain=0.05,
    tolerance=2e-6,
)
substrate = geo.Brick(
    origin=(1e-3, -4e-3, -0.3e-3),
    size=(18e-3, 8e-3, 0.3e-3),
    material="air",
)
trace = geo.Brick(
    origin=(2e-3, -0.5e-3, 0),
    size=(16e-3, 1e-3, 0.035e-3),
    material="pec",
)
assembly = wrap @ geo.Group(substrate, trace, name="conformal trace")
board_wrapped, trace_wrapped = tuple(assembly.members())
assert board_wrapped.volume() > 0
assert trace_wrapped.volume() > 0
assert board_wrapped.material == substrate.material
assert trace_wrapped.material == trace.material

# %%
# Join two straight components with matching end tangents
# --------------------------------------------------------
#
# Tangency is the G1 goal: the transition's wall normals agree with the
# straight adjoining walls at each end. Curvature matching is not requested.

small = geo.Brick(origin=(-2e-3, -1e-3, 0), size=(4e-3, 2e-3, 1e-3), material="pec")
large = geo.Brick(origin=(-3e-3, -1.5e-3, 5e-3), size=(6e-3, 3e-3, 1e-3), material="pec")
transition = small.face(normal="z").lofted(large.face(normal=(0, 0, -1)), blend="tangent")
assert transition.volume() > 0

# %%
# Read the intended surface and conductor placement
# ---------------------------------------------------

u, v = np.meshgrid(np.linspace(0, length, 45), np.linspace(-half_width, half_width, 25))
z = rise * np.sin(np.pi * u / length) * np.cos(np.pi * v / (2 * half_width))
fig = plt.figure(figsize=(7, 4))
ax = fig.add_subplot(projection="3d")
ax.plot_surface(u * 1e3, v * 1e3, z * 1e3, alpha=0.45, color="steelblue")
centre = np.linspace(2e-3, 18e-3, 100)
for edge in (-0.5e-3, 0.5e-3):
    x, y, neutral_z = patch(centre, edge)
    ax.plot(x * 1e3, np.full_like(x, y * 1e3), (neutral_z + 0.035e-3) * 1e3, color="firebrick")
ax.set(xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)")
fig.tight_layout()
