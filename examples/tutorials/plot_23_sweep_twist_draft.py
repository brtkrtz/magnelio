"""
23 — Sweep twist and draft: rotating and tapering sections
==========================================================

Twist is a total additional roll, distributed uniformly over the travelled
path length. Draft is a constant section-offset angle: positive values grow
the exterior and shrink holes. Both operate on the actual planar section,
including its material and bores.
"""

import math

import matplotlib.pyplot as plt

from magnelio import geo, plots

# %%
# A rectangular transition with ninety degrees of total roll
# ----------------------------------------------------------
#
# The inlet starts without extra roll; the middle has 45 degrees and the
# outlet 90 degrees. The straight transition retains area times length.

length = 4e-3
route = geo.Curve.line((0, 0, 0), (0, 0, length))
aperture = geo.Profile.rectangle((0, 0, 0), (2e-3, 1e-3), material="air")
twisted = aperture.swept(route, twist_deg=90)
assert math.isclose(twisted.volume(), aperture.area * length, rel_tol=2e-6)
assert math.isclose(twisted.face(normal="z").area, aperture.area, rel_tol=2e-7)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
for ax, fraction in zip(axes, (0.05, 0.5, 0.95)):
    plots.plot_cross_section(
        [twisted],
        "z",
        fraction * length,
        ax=ax,
        title=f"{fraction * 90:g} degrees of roll",
    )
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
fig.tight_layout()

# %%
# A drafted annulus
# ------------------
#
# At one degree of positive draft, the outer radius grows by
# ``length * tan(1 degree)`` and the bore radius shrinks by the same amount.
# The area is therefore linear in distance. Its integral predicts the volume
# independently of the CAD construction.

outer, inner, draft = 2e-3, 0.3e-3, 1
annulus = geo.Profile.from_wires(
    geo.Curve.circle((0, 0, 0), outer),
    [geo.Curve.circle((0, 0, 0), inner)],
    material="pec",
)
drafted = annulus.swept(route, draft_deg=draft)
slope = math.tan(math.radians(draft))
delta = length * slope
expected_volume = math.pi * ((outer**2 - inner**2) * length + (outer + inner) * slope * length**2)
expected_outlet = math.pi * ((outer + delta) ** 2 - (inner - delta) ** 2)
assert math.isclose(drafted.volume(), expected_volume, rel_tol=2e-6)
assert math.isclose(drafted.face(normal="z").area, expected_outlet, rel_tol=2e-7)
assert len(drafted.face(normal="z").edges) == 2

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
plots.plot_cross_section([drafted], "y", 0, ax=axes[0], title="Drafted walls and open bore")
plots.plot_cross_section([drafted], "z", 0.95 * length, ax=axes[1], title="Near the outlet")
fig.tight_layout()

# %%
# Combining the laws
# -------------------
#
# Either section can use both keywords together. Roll is added about the
# transported section normal, so it preserves fixed/oblique section planes.
# ``tolerance=`` controls the sampled section fit in metres; its default is
# one millionth of the section diagonal. Holes must remain open along the
# route. On curved routes, draft specifies section offset per arc length;
# the spatial wall angle also depends on transport and curvature. Draft does
# not replace a collapsing boundary with a solid section.
