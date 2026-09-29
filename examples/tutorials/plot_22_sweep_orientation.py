"""
22 — Sweep orientation: keeping transition apertures parallel
=============================================================

A curved transition can keep its cross-sections perpendicular to the route
or keep its apertures parallel in world space. Both use the same exact path
and the same starting section. Their different volumes follow directly from
the displacement through the section plane.

``frame="corrected_frenet"`` is the established default. ``frame="frenet"``
follows curvature and torsion, while ``frame="fixed"`` keeps the section's
world orientation. ``frame="fixed_binormal", binormal="z"`` keeps its angular
relation to the bend plane constant in this planar example. A spatial path can
produce oblique sections in that last mode. Path's up direction is routing
data and does not impose any of these sweep modes.
"""

import math

import matplotlib.pyplot as plt

import magnelio as mio
from magnelio import geo, plots

# %%
# One section, two transports
# ---------------------------
#
# The asymmetric rectangle makes orientation visible. Width is along x,
# height along z, and the initial normal is along y. The 45-degree arc advances
# monotonically through the parallel section planes, so both constructions
# form valid bodies.

radius = 8e-3
angle = math.pi / 4
air = mio.Material.air()
aperture = geo.Profile.rectangle(
    (radius, 0, 0), (1e-3, 0.3e-3), normal="y", x_direction="x", material=air
)
spine = geo.Curve.arc(
    (radius, 0, 0),
    (radius * math.cos(angle / 2), radius * math.sin(angle / 2), 0),
    (radius * math.cos(angle), radius * math.sin(angle), 0),
)
perpendicular = aperture.swept(spine)
parallel = aperture.swept(spine, frame="fixed")

assert math.isclose(perpendicular.volume(), aperture.area * spine.length, rel_tol=1e-8)
assert math.isclose(parallel.volume(), aperture.area * radius * math.sin(angle), rel_tol=1e-8)
assert math.isclose(parallel.face(normal="y").area, aperture.area, rel_tol=1e-8)
assert math.isclose(
    perpendicular.face(normal=(-math.sin(angle), math.cos(angle), 0)).area,
    aperture.area,
    rel_tol=1e-8,
)

# %%
# Inspect the apertures
# ---------------------
#
# The outlet is normal to the arc in the first body and parallel to the inlet
# in the second. Both pictures use the same geometry scale.

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
plots.plot_cross_section(
    [perpendicular], "z", 0, ax=axes[0], title="Default: perpendicular sections"
)
plots.plot_cross_section([parallel], "z", 0, ax=axes[1], title="Fixed: parallel apertures")
for get_limits, set_limits in (("get_xlim", "set_xlim"), ("get_ylim", "set_ylim")):
    limits = [getattr(ax, get_limits)() for ax in axes]
    lower = min(pair[0] for pair in limits)
    upper = max(pair[1] for pair in limits)
    padding = 0.08 * (upper - lower)
    for ax in axes:
        getattr(ax, set_limits)((lower - padding, upper + padding))
fig.tight_layout()

# %%
# Choosing a mode
# ----------------
#
# Use the default for the normal swept waveguide grammar. Choose fixed world
# orientation when parallel apertures are the actual construction requirement.
# Avoid a folded route or a section too large for its bend radius. Frame modes
# preserve holes and material inheritance; they do not certify the absence of
# self-intersection or provide a twist/draft law.
