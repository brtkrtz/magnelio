"""
24 — Partition a body and take a filled section
================================================

Partitioning makes independently owned material regions from a cutting plane
or another body or sheet. A section reads the intersection as exact curves;
``filled=True`` asks for planar regions with holes instead. Both operations
use world geometry and leave the source untouched.
"""

import math

import matplotlib.pyplot as plt

from magnelio import geo, plots

# %%
# Split a hollow component at a construction plane
# -------------------------------------------------
#
# The source material reaches both independent halves. Their volume sum is
# the original volume; each half can be placed or used in another operation.

outer_radius = 2e-3
inner_radius = 0.7e-3
length = 6e-3
part = geo.Cylinder(
    origin=(0, 0, 0),
    radius=outer_radius,
    inner_radius=inner_radius,
    height=length,
    material="pec",
)
halves = part.partition(normal="z", position=length / 2)
assert len(halves) == 2
assert all(half.material is part.material for half in halves)
assert math.isclose(sum(half.volume() for half in halves), part.volume(), rel_tol=1e-10)

# %%
# Read its section as curves or a filled annulus
# ----------------------------------------------
#
# The ordinary section returns the exterior and bore loops independently.
# The filled section is one Profile whose inner loop is a hole, ready for
# another profile-consuming construction operation.

outlines = part.section(normal="z", position=length / 2)
(annulus,) = part.section(normal="z", position=length / 2, filled=True)
expected_area = math.pi * (outer_radius**2 - inner_radius**2)
assert len(outlines) == 2
assert all(curve.is_closed for curve in outlines)
assert len(annulus.boundary()) == 2
assert math.isclose(annulus.area, expected_area, rel_tol=1e-10)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
plots.plot_cross_section(halves, "y", 0, ax=axes[0], title="Independent half bodies")
plots.plot_cross_section([part], "z", length / 2, ax=axes[1], title="Filled annular section")
fig.tight_layout()

# %%
# A cutter need not be axis aligned
# ---------------------------------
#
# A bounded oblique sheet splits a block in world coordinates. Both halves
# keep the source material; kernel output order is not a design parameter.

block = geo.Brick(origin=(0, 0, 0), size=(2e-3,) * 3, material="air")
oblique = geo.Profile.rectangle((1e-3,) * 3, (5e-3,) * 2, normal=(1, 1, 1))
regions = block.partition(oblique)
assert len(regions) == 2
assert all(math.isclose(region.volume(), block.volume() / 2, rel_tol=1e-10) for region in regions)
