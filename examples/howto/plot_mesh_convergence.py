"""
Mesh convergence: how fine is fine enough
=========================================

Every result from a grid carries a discretisation error, and nothing
inside a single run tells you how large it is.  The only honest
instrument is a *ladder*: the same model on successively finer
meshes, the same quantity read off each rung, and a look at how it
moves.  This page is the recipe — one ladder for an eigenmode
quantity, one for S-parameters — and ends with the one thing a ladder
cannot see.

The mesh has two scales: ``min_nodes_per_wavelength`` sets the bulk
cell, ``min_cells_per_feature`` the cell at material interfaces.  A
ladder must scale *both* in a fixed ratio — a ladder over the
wavelength knob alone stalls as long as the interface cells dominate
the grid, then jumps once the bulk cell finally falls below them.
Nothing else about the model changes between rungs.
"""

# sphinx_gallery_thumbnail_number = 1

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jn_zeros

import magnelio as mio
from magnelio import geo, ports
from magnelio.constants import C0

# %%
# Ladder 1 — an eigenfrequency
# ----------------------------
#
# A cylindrical metal cavity (pillbox), because its lowest mode has a
# closed form to hold the ladder against: :math:`f_{010} = 2.4048\,c_0
# / (2\pi R)`, independent of the height.  The curved wall is where
# the grid has to work — it is represented by the conformal
# sub-cell treatment of partially filled cells, not by a staircase.

R_CAV, H_CAV = 10e-3, 6e-3
F_EXACT = float(jn_zeros(0, 1)[0]) * C0 / (2 * np.pi * R_CAV)


def pillbox():
    model = mio.GeometryModel(background="pec")
    model.add(geo.Cylinder(origin=(0, 0, 0), radius=R_CAV, height=H_CAV, axis="z", material="air"))
    return model


def rung(mnpw):
    """Both mesh scales from one number: 4 bulk cells per interface cell."""
    return mio.MeshControl(min_nodes_per_wavelength=mnpw, min_cells_per_feature=mnpw // 4)


def f0_on(mnpw, f_max=14e9, f_shift=11e9):
    """Lowest eigenfrequency [Hz], cell count and wall time for one rung."""
    t = time.perf_counter()
    mesh = mio.Mesh.from_geometry(pillbox(), rung(mnpw), f_max=f_max)
    result = mio.AnalysisEigenmode(
        mesh=mesh, n_modes=2, sigma=(2 * np.pi * f_shift) ** 2, verbose=False
    ).run()
    return float(result.frequencies[0]), mesh.Nx * mesh.Ny * mesh.Nz, time.perf_counter() - t


ladder = [12, 16, 24, 32, 48]
rows = [f0_on(n) for n in ladder]

print(f"exact TM010: {F_EXACT / 1e9:.4f} GHz")
print("mnpw   f0 [GHz]   cells   time   change    error")
prev = None
for n, (f, cells, dt) in zip(ladder, rows):
    change = "" if prev is None else f"{(f - prev) / prev * 100:+.2f} %"
    print(
        f"{n:4d}   {f / 1e9:8.4f}   {cells:6d}   {dt:4.1f} s   {change:8s} "
        f"{(f - F_EXACT) / F_EXACT * 100:+.2f} %"
    )
    prev = f

# %%
# Reading the ladder
# ------------------
#
# Two things to look at.  First the *change between rungs*: when it
# drops below the tolerance your specification allows, the coarser
# rung of that pair is fine enough for this quantity.  Second the
# *trend*: a second-order scheme converges like :math:`h^2`, so
# plotting the result against :math:`1/N^2` (with :math:`N` the nodes
# per wavelength) should give a straight line once the mesh is in the
# asymptotic regime, and the intercept at :math:`1/N^2 \to 0` is an
# estimate of the converged value.  If the points do not fall on a
# line, the mesh is not yet in that regime and the extrapolation is
# not to be trusted.  Here the exact value is known, so the plot also
# shows how good the estimate is — and it is a caution as much as a
# demonstration: the coarse rungs wobble with the way the grid lines
# cut the curved wall, the fine rungs approach the exact value from
# below and then slightly overshoot it, and the extrapolation lands a
# few tenths of a percent off.  A conformal boundary carries an error
# component that does not scale as :math:`h^2`.  Take the extrapolation
# as an estimate, and the change between rungs as the criterion.

f0 = np.array([r[0] for r in rows])
x = 1.0 / np.asarray(ladder, dtype=float) ** 2
slope, intercept = np.polyfit(x[-3:], f0[-3:], 1)  # fit the three finest rungs

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(x, f0 / 1e9, "o-", label="ladder")
xx = np.linspace(0.0, x.max(), 50)
ax.plot(xx, (slope * xx + intercept) / 1e9, "--", label="fit through the three finest rungs")
ax.plot(0.0, intercept / 1e9, "s", label=f"extrapolated {intercept / 1e9:.4f} GHz")
ax.axhline(F_EXACT / 1e9, color="k", lw=0.8, label=f"exact {F_EXACT / 1e9:.4f} GHz")
ax.set_xlabel("1 / (nodes per wavelength)²")
ax.set_ylabel("f₀ [GHz]")
ax.set_title("TM₀₁₀ of a pillbox versus resolution")
ax.grid(True, alpha=0.3)
ax.legend()

print(
    f"extrapolated f0: {intercept / 1e9:.4f} GHz ({(intercept - F_EXACT) / F_EXACT * 1e6:+.0f} ppm)"
)

# %%
# When the ladder does not settle
# -------------------------------
#
# A high-contrast dielectric body on a conformal grid is a different
# story.  A ceramic puck (εᵣ = 45) in a box, run through the same
# ladder, moves by ±3 % *in either direction* between rungs: the
# result depends on how the grid lines happen to cut the ceramic
# boundary, and that changes from rung to rung.  There is no straight
# line to extrapolate.  The spread across the rungs is then the honest
# error estimate for the absolute frequency — and the reason to design
# on ratios where you can.

# %%
# Ratios converge faster than absolutes
# -------------------------------------
#
# When the design quantity is a *ratio* of two results from the same
# mesh — a coupling coefficient from two eigenfrequencies, a quality
# factor, a relative bandwidth — both carry nearly the same
# discretisation error and most of it cancels.  Tutorial 13 measures
# this on a filter design: between two grids f₀ moves by a percent —
# half the filter's passband — while the coupling coefficient moves
# by a fraction of a percent of itself, which is that same fraction of
# the bandwidth.  Run the ladder on the quantity you actually design
# with, and read it against that quantity's own tolerance.

# %%
# Ladder 2 — S-parameters
# -----------------------
#
# The same recipe for a driven run: a 50 Ω microstrip between two
# waveguide ports.  The quantity compared is the largest change in
# ``|S21|`` and ``|S11|`` (in dB) over the band between successive
# rungs, read on one common frequency axis.

h_sub, w_strip, t_strip = 0.8e-3, 1.2e-3, 0.2e-3
W_box, H_box, L = 8.0e-3, 5.0e-3, 20.0e-3
f_max = 15.0e9
fr4 = mio.Material("FR4", epsilon=(4.3,) * 3)


def microstrip():
    substrate = geo.Brick(origin=(-W_box / 2, 0, 0), size=(W_box, h_sub, L), material=fr4)
    air = geo.Brick(origin=(-W_box / 2, h_sub, 0), size=(W_box, H_box - h_sub, L), material="air")
    strip = geo.Brick(origin=(-w_strip / 2, h_sub, 0), size=(w_strip, t_strip, L), material="pec")
    model = mio.GeometryModel(background="pec")
    model.add(substrate)
    model.add(air - strip)
    model.add(strip)
    model.add_port(ports.PortWaveguide(name="port1", plane="zmin", n_modes=1))
    model.add_port(ports.PortWaveguide(name="port2", plane="zmax", n_modes=1))
    return model


f_common = np.linspace(1.0e9, f_max, 141)


def s_on(mnpw):
    """|S21| and |S11| in dB on the common axis, cells and wall time."""
    t = time.perf_counter()
    mesh = mio.Mesh.from_geometry(microstrip(), rung(mnpw), f_max=f_max)
    result = mio.AnalysisScatteringTD(mesh=mesh, verbose=False).run(excited=["port1"])
    f = result.f_axis
    s21 = np.interp(f_common, f, 20 * np.log10(np.abs(result.S("port2", "port1"))))
    s11 = np.interp(f_common, f, 20 * np.log10(np.abs(result.S("port1", "port1"))))
    return s21, s11, mesh.Nx * mesh.Ny * mesh.Nz, time.perf_counter() - t


s_ladder = [8, 12, 16, 24]
s_rows = [s_on(n) for n in s_ladder]

print("mnpw    cells   time   max Δ|S21|   max Δ|S11|")
for k, (n, (s21, s11, cells, dt)) in enumerate(zip(s_ladder, s_rows)):
    if k == 0:
        print(f"{n:4d}   {cells:6d}   {dt:4.1f} s")
        continue
    d21 = np.abs(s21 - s_rows[k - 1][0]).max()
    d11 = np.abs(s11 - s_rows[k - 1][1]).max()
    print(f"{n:4d}   {cells:6d}   {dt:4.1f} s   {d21:7.3f} dB   {d11:7.2f} dB")

fig, ax = plt.subplots(figsize=(6.4, 4.2))
for n, (s21, s11, _cells, _dt) in zip(s_ladder, s_rows):
    ax.plot(f_common / 1e9, s11, label=f"|S11|, mnpw {n}")
ax.set_xlabel("frequency [GHz]")
ax.set_ylabel("dB")
ax.set_title("return loss across the ladder")
ax.grid(True, alpha=0.3)
ax.legend()

# %%
# The change in ``|S21|`` is small from the start — a matched line has
# little to get wrong — while ``|S11|`` at −30 dB and below moves by
# whole decibels between rungs: a small quantity measured against a
# discretisation error of fixed absolute size.  Judge each quantity
# against its own tolerance.

# %%
# What the ladder cannot see
# --------------------------
#
# A ladder finds errors that *shrink* with the cell size.  It is blind
# to a feature the grid does not contain at all: a chamfer or fillet
# smaller than half a cell has no effect on the result, then appears
# in one step when a rung finally resolves it — the ladder shows a
# jump, not a trend.  The mesher places grid planes on such edges so
# the feature occupies a cell layer of its own, and warns when an edge
# would need a cell finer than ``h_max / max_edge_refinement`` and is
# dropped.  Read those warnings before trusting a ladder; the remedy
# they name (``MeshControl(max_edge_refinement=...)``) is part of the
# model, not of the resolution.
#
# The flip side: a feature layer the mesher *does* resolve keeps its
# size — one cell across the chamfer, whatever the rung — so the grid
# is no longer refined uniformly and the :math:`h^2` trend is
# disturbed.  With a 0.5 mm chamfer on the ceramic puck of the section
# above, a wavelength-only ladder produces the same grid on its first
# two rungs and the rest do not fall on a line.  Run the ladder on the
# sharp-edged model to judge the resolution, then add the features
# back.  The mechanism is explained under
# :doc:`../methods/meshing-conformal`.

plt.show()
