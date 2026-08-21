"""
Periodic structures: the dispersion diagram of a TESLA cell
============================================================

Every cavity so far was a closed box with a discrete set of resonances.
An accelerating structure is different: a *chain* of identical cells,
coupled through their irises, and what matters is how the field in one
cell relates to the field in the next.  For an infinite chain that
relation is Floquet's theorem — the field in cell :math:`n+1` is the
field in cell :math:`n` times a phase factor :math:`e^{-\\mathrm j
\\varphi}` — and the resonant frequency becomes a *function* of the
**phase advance** :math:`\\varphi` per cell.  That function, plotted
over :math:`0 \\le \\varphi \\le \\pi`, is the **dispersion diagram**
(or Brillouin diagram) of the structure.

This tutorial computes it for the mid-cell of the TESLA 9-cell
cavity, the 1.3 GHz superconducting resonator of the TESLA Test
Facility and, since then, of the European XFEL and LCLS-II.  Two
things are new: a face pair declared ``"Periodic"`` with a phase
advance on the eigenmode analysis, and the cell's outline — an
elliptical arc, a straight wall and a circular arc joined tangentially
— drawn with :class:`~magnelio.geo.Path`.
"""

# sphinx_gallery_thumbnail_number = 3

# %%
# The cell
# --------
#
# The TESLA mid-cell is described by a handful of numbers (Aune et
# al., *Phys. Rev. ST Accel. Beams* 3, 092001 (2000), Table 3).  Its
# half-cell contour, drawn in the :math:`(z, r)` plane from the iris
# plane to the equator plane, is:
#
# - an **elliptical arc** around the iris, centred on the iris plane
#   at :math:`r = R_\mathrm{iris} + b` with half-axes
#   :math:`a = 12` mm along the beam axis and :math:`b = 19` mm
#   radially, so the aperture is smallest exactly on the iris plane;
# - a **circular arc** of radius :math:`R_\mathrm{arc} = 42` mm at
#   the equator, centred below the equator point so the contour is
#   flat there;
# - the straight **wall** between them, tangent to both.
#
# ===========================  =========
# quantity                     value
# ===========================  =========
# equator radius               103.3 mm
# iris radius                  35.0 mm
# circular-arc radius          42.0 mm
# iris half-axis :math:`a`     12.0 mm
# iris half-axis :math:`b`     19.0 mm
# half-cell length             57.7 mm
# ===========================  =========
#
# The wall angle is *not* a free parameter: once the two curves are
# placed, the straight wall is their common tangent.  A short root
# search finds the point on the ellipse whose tangent line touches the
# circle.

import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

import magnelio as mio
from magnelio import geo, plots

R_EQ, R_IRIS, R_ARC = 103.3e-3, 35.0e-3, 42.0e-3
A_IRIS, B_IRIS = 12.0e-3, 19.0e-3
L_HALF = 57.7e-3
PERIOD = 2.0 * L_HALF

# Centres in the (z, r) plane.
c_iris = (0.0, R_IRIS + B_IRIS)
c_eq = (L_HALF, R_EQ - R_ARC)


def on_ellipse(theta):
    """Point (z, r) on the iris ellipse; theta = -pi/2 is the iris tip."""
    return (A_IRIS * math.cos(theta), c_iris[1] + B_IRIS * math.sin(theta))


def ellipse_tangent(theta):
    """Unit tangent (z, r) of the iris ellipse at theta."""
    d = (-A_IRIS * math.sin(theta), B_IRIS * math.cos(theta))
    n = math.hypot(*d)
    return (d[0] / n, d[1] / n)


def tangent_gap(theta):
    """Distance of the circle centre from the ellipse's tangent line, minus R_ARC."""
    p, t = on_ellipse(theta), ellipse_tangent(theta)
    normal = (t[1], -t[0])
    return (c_eq[0] - p[0]) * normal[0] + (c_eq[1] - p[1]) * normal[1] - R_ARC


theta = brentq(tangent_gap, -math.pi / 2 + 1e-6, -1e-6)
p_ell = on_ellipse(theta)
t_wall = ellipse_tangent(theta)
p_arc = (c_eq[0] - R_ARC * t_wall[1], c_eq[1] + R_ARC * t_wall[0])
wall_angle = math.degrees(math.atan2(t_wall[0], t_wall[1]))

print(f"wall leaves the ellipse at z = {p_ell[0] * 1e3:.2f} mm, r = {p_ell[1] * 1e3:.2f} mm")
print(f"wall meets the arc at     z = {p_arc[0] * 1e3:.2f} mm, r = {p_arc[1] * 1e3:.2f} mm")
print(f"wall angle from the radial direction: {wall_angle:.2f} deg")

# %%
# Drawing the profile
# -------------------
#
# The profile is drawn in the x-z plane (x plays the part of the
# radius) and revolved about z.  The pen starts at the iris tip, follows
# the ellipse, the wall and the equator arc to the mid-plane, then
# retraces the same three segments mirrored for the second half of the
# cell.  Two details from tutorial 14 apply:
#
# - each arc names its centre and a ``normal`` to fix which way round
#   it goes; on the way *back* the turning sense reverses, so the
#   equator arcs take ``normal=(0, -1, 0)`` where the iris arcs take
#   ``"y"``;
# - a profile must not touch the revolution axis.  The outline is
#   therefore closed slightly *inside* the iris radius, revolved into a
#   ring, and united with a plain cylinder that supplies the beam tube
#   and the axis.


def xz(zr):
    """(z, r) -> (x, y, z) with the radius along x."""
    return (zr[1], 0.0, zr[0])


def mirrored(zr):
    """The same (z, r) point in the second half of the cell."""
    return (PERIOD - zr[0], zr[1])


air = mio.Material.air()
pec = mio.Material.pec()

outline = (
    geo.Path(xz((0.0, R_IRIS)))
    .ellipse_to(
        xz(p_ell), center=xz(c_iris), semi_axes=(A_IRIS, B_IRIS), major_axis="z", normal="y"
    )
    .line_to(xz(p_arc))
    .arc_to(xz((L_HALF, R_EQ)), center=xz(c_eq), normal=(0.0, -1.0, 0.0))
    .arc_to(xz(mirrored(p_arc)), center=xz(mirrored(c_eq)), normal=(0.0, -1.0, 0.0))
    .line_to(xz(mirrored(p_ell)))
    .ellipse_to(
        xz((PERIOD, R_IRIS)),
        center=xz(mirrored(c_iris)),
        semi_axes=(A_IRIS, B_IRIS),
        major_axis="z",
        normal="y",
    )
    .line_to(xz((PERIOD, R_IRIS - 2e-3)))
    .line_to(xz((0.0, R_IRIS - 2e-3)))
    .closed()
    .covered()
)
ring = outline.revolved(axis="z", material=air)
tube = geo.Cylinder(origin=(0, 0, 0), radius=R_IRIS, height=PERIOD, axis="z", material=air)
cell = geo.Union(ring, tube, material=air)

fig, ax = plots.plot_cross_section([cell], "y", 0.0, title="TESLA mid-cell, one period")

# %%
# Unit cell, quarter model
# ------------------------
#
# The model is one period of the chain: from one iris plane to the
# next.  Both z-faces are declared ``"Periodic"`` — the pair is the
# statement that the structure continues identically beyond them.  Two
# mirror planes through the axis cut the work by four: the modes of
# interest have their electric field in the r-z plane, which is
# *tangential* to the planes x = 0 and y = 0, so the correct wall there
# is the magnetic one (tutorial 09).

model = mio.GeometryModel(
    background=pec,
    boundary_conditions={
        "xmin": "SymmetryPMC",
        "ymin": "SymmetryPMC",
        "zmin": "Periodic",
        "zmax": "Periodic",
    },
)
model.add(cell)

mesh = mio.Mesh.from_geometry(model, mio.MeshControl(max_cell_size=4e-3), f_max=1.5e9)
print(f"grid: {mesh.Nx} x {mesh.Ny} x {mesh.Nz} cells")

# %%
# Band edges the classical way
# ----------------------------
#
# Before the sweep, the two ends of the passband by the traditional
# route.  At :math:`\varphi = 0` every cell carries the same field, and
# the iris plane is a mirror plane across which the accelerating field
# :math:`E_z` is *even* — a plane where the tangential electric field
# vanishes, i.e. an electric wall.  At :math:`\varphi = \pi` the field
# flips sign from cell to cell, :math:`E_z` is *odd* across the iris
# plane and the wall there is magnetic.  Two ordinary cavity solves,
# nothing periodic about them:


def lowest_mode(bcs):
    half = mio.GeometryModel(background=pec, boundary_conditions=bcs)
    half.add(cell)
    m = mio.Mesh.from_geometry(half, mio.MeshControl(max_cell_size=4e-3), f_max=1.5e9)
    return mio.AnalysisEigenmode(mesh=m, n_modes=1, verbose=False).run().frequencies[0]


mirror = {"xmin": "SymmetryPMC", "ymin": "SymmetryPMC"}
f_0_walls = lowest_mode({**mirror, "zmin": "PEC", "zmax": "PEC"})
f_pi_walls = lowest_mode({**mirror, "zmin": "PMC", "zmax": "PMC"})
print(f"electric walls on the iris planes (0-mode):  {f_0_walls / 1e9:.4f} GHz")
print(f"magnetic walls on the iris planes (pi-mode): {f_pi_walls / 1e9:.4f} GHz")

# %%
# The sweep
# ---------
#
# Everything in between needs the periodic pair.  ``phase_advance_deg``
# on :class:`~magnelio.AnalysisEigenmode` is the phase by which the
# field in one period leads the next; the mesh is built once and the
# analysis is repeated for each value.  At 0 and 180 degrees the
# problem is real; in between the mode fields are complex — travelling
# waves — and the solver switches to a complex Hermitian formulation
# without anything to configure.

phases = np.linspace(0.0, 180.0, 7)
freqs = []
for deg in phases:
    result = mio.AnalysisEigenmode(mesh=mesh, n_modes=1, verbose=False, phase_advance_deg=deg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        freqs.append(result.run().frequencies[0])
    print(f"phase advance {deg:5.1f} deg: {freqs[-1] / 1e9:.4f} GHz")
freqs = np.array(freqs)

# %%
# The dispersion diagram
# ----------------------
#
# A chain of electrically coupled cells follows
#
# .. math::
#
#     f(\varphi)^2 = f_{\pi/2}^2\,(1 - k \cos\varphi)
#
# (Wangler, *RF Linear Accelerators*, eq. 3.31), where the
# **cell-to-cell coupling** :math:`k` is also the fractional width of
# the passband, :math:`(f_\pi - f_0)/f_{\pi/2}`.  Fitting the two
# parameters to the computed points checks the curve's *shape*; the
# published values are 1.300 GHz for the :math:`\pi`-mode and
# :math:`k = 1.87\,\%`.

phi = np.radians(phases)
design = np.column_stack([np.ones_like(phi), -np.cos(phi)])
coef, *_ = np.linalg.lstsq(design, freqs**2, rcond=None)
f_half = math.sqrt(coef[0])
k_cell = coef[1] / coef[0]
phi_fine = np.linspace(0.0, math.pi, 181)
f_fit = f_half * np.sqrt(1.0 - k_cell * np.cos(phi_fine))

print(f"pi-mode:              {freqs[-1] / 1e9:.4f} GHz (design 1.3000 GHz)")
print(f"cell-to-cell coupling: {100 * k_cell:.2f} % (published 1.87 %)")

fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.plot(
    np.degrees(phi_fine),
    f_fit / 1e9,
    color="0.6",
    label="$f_{\\pi/2}\\sqrt{1 - k\\cos\\varphi}$ fit",
)
ax.plot(phases, freqs / 1e9, "o", label="periodic eigenmode solves")
ax.plot(
    [0.0, 180.0],
    [f_0_walls / 1e9, f_pi_walls / 1e9],
    "s",
    mfc="none",
    ms=10,
    label="wall-type band edges",
)
ax.set_xlabel("phase advance per cell (deg)")
ax.set_ylabel("frequency (GHz)")
ax.set_xticks(np.arange(0, 181, 30))
ax.set_title("TM$_{010}$ passband of the TESLA mid-cell")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()

# %%
# The band edges from the periodic solves land on the wall-type solves
# — the same two numbers by two routes, to a fraction of a permille
# (the wall-type models are meshed separately, and a magnetic wall
# pulls the grid in differently from a periodic face) — and the curve
# between them has the cosine shape the circuit model predicts.  The
# :math:`\pi`-mode, where the chain is operated, is the top of the
# band: the field reverses from cell to cell in the time a relativistic
# particle takes to cross one, which is what makes the period
# :math:`c/2f`.
#
# The field: standing wave and travelling wave
# --------------------------------------------
#
# The TESLA cavity is operated in the :math:`\pi`-mode (Aune et al.,
# Table 2): the field reverses from cell to cell in the time a
# relativistic particle takes to cross one, which is what fixes the
# period at :math:`c/2f`.  As a band edge it is a *standing* wave and,
# as the wall-type solve above showed, needs no periodic boundary at
# all.  Every other point of the diagram is a *travelling* wave — the
# field pattern advances by :math:`\varphi` per cell and the mode
# fields are complex — and exists only through the periodic pair.
# The :math:`2\pi/3` phase advance is the textbook choice of
# travelling-wave linacs, so it serves as the example: the plot shows
# the real snapshot of its electric field next to the
# :math:`\pi`-mode, both on the meridian plane of the quarter model.
# The standing wave is symmetric about the cell's mid-plane; the
# travelling wave is not, and no combination of walls would produce it.

pi_mode = mio.AnalysisEigenmode(mesh=mesh, n_modes=1, verbose=False, phase_advance_deg=180.0).run()
tw_mode = mio.AnalysisEigenmode(mesh=mesh, n_modes=1, verbose=False, phase_advance_deg=120.0).run()

fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
for ax, res, label in (
    (axes[0], pi_mode, "$\\pi$-mode (standing wave)"),
    (axes[1], tw_mode, "$2\\pi/3$-mode (travelling wave)"),
):
    res.plot(
        mode=0, component="E", normal="y", position=0.0, plot_type="vector", geometry=model, ax=ax
    )
    ax.set_title(f"{label}, {res.frequencies[0] / 1e9:.4f} GHz")
fig.tight_layout()

# %%
# Where to go next
# ----------------
#
# New in this tutorial: a ``"Periodic"`` face pair and
# ``phase_advance_deg`` turn the eigenmode analysis into a unit-cell
# solver for infinite periodic structures, and sweeping the phase
# advance traces the dispersion diagram; elliptical arcs join the
# profile vocabulary; and a tangent construction settles what a
# parameter table leaves implicit.  The same pair of declarations
# serves any periodic structure whose unit cell fits in a box —
# disk-loaded waveguides, photonic-crystal slabs, frequency-selective
# surfaces at normal incidence.
