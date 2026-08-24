"""
Characterising a lumped port termination: coaxial line
======================================================

A :class:`~magnelio.ports.PortLumped` terminates a line in a single
grid edge — cheap, DC-capable, and available where a waveguide port
window does not fit.  The price is that it is not an exact line
termination: a residual self-reflection and a phase error remain, and
both depend on the grid resolution at the conductor gap, on the gap
geometry and position, and on the port impedance.  Rules of thumb
exist, but none of them tells you how good *your* termination is on
*your* mesh, or up to which frequency you can trust it.

This page is two things at once:

- a **walkthrough** of what governs a lumped termination's quality —
  the sweeps below show which knob moves which error;
- a **pre-flight tool**: before a real simulation, fill in the given
  quantities of your target model, run, and read the scoreboard —
  *"with these settings you are this good"*.  Tweak the knobs until
  the numbers meet your spec, then carry the settings over.

The measurement setup: a waveguide port — reflection-free by
construction, with a floor far below anything a lumped element
reaches — launches the exact line mode down a short uniform coax onto
the lumped port under test.  ``|S11|`` at the waveguide port *is* the
termination's self-reflection.  The phase ruler is a second, equally
short run of the same line with waveguide ports at both ends: its
transmission phase is the exact propagation of *this grid* over the
reference length, so the difference to the lumped run is the
termination's phase error — no textbook dispersion formula involved,
which is what lets the same recipe serve dispersive lines
(microstrip, CPW) unchanged.

Every number this page prints is a property of *this* example's grid,
not a constant of the method: re-measure whenever the cross-section,
the resolution or the band changes.
"""

# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import numpy as np

import magnelio as mio
from magnelio import geo, plots, ports
from magnelio.constants import ETA0

# %%
# Given quantities
# ----------------
#
# What the target simulation dictates: the cross-section, the band,
# and the cell size.  ``cell`` is the cross-section cell size the
# test mesh may not fall below — set it to the size your production
# mesh will actually have at the port, because the termination
# quality is a property of the discretised gap, not of the continuous
# geometry.

r_i = 0.405e-3  # inner conductor radius [m]
r_o = 1.475e-3  # shield (dielectric outer) radius [m]
eps_r = 2.25  # solid polyethylene
f_max = 15e9  # upper band edge [Hz]
cell = 0.5 * r_i  # production cross-section cell size [m]

# %%
# The knobs
# ---------
#
# What you optimise — the compromise you will carry into the target
# simulation:
#
# - ``gap`` — length of the end gap between inner conductor and the
#   shorted end plate; the lumped port element spans it.  Governs the
#   broadband reflection level.
# - ``gap_position`` — where the gap *starts* relative to the
#   reference plane (positive = beyond it).  Governs the phase error:
#   the termination does not act at the plate but somewhere inside
#   the gap region, so the gap must sit off the reference plane by
#   that amount.  On a coax the compensation can be perfect at all
#   frequencies (TEM, dispersion-free); on microstrip or CPW it is
#   frequency-dependent and the chosen position is a genuine
#   compromise.
# - ``z0_port`` — the lumped port impedance.  ``None`` uses the line
#   impedance *of the grid*, as the waveguide-port solver measures it
#   below; a number (e.g. the catalogue 50 Ω) uses that instead.

gap = 0.4 * (r_o - r_i)  # end gap length [m]
gap_position = 0.0  # gap start relative to the reference plane [m]
z0_port = None  # None -> grid line impedance; or e.g. 50.0

# %%
# Derived quantities
# ------------------
#
# The reference plane sits ``L`` behind the waveguide port; a few
# shield radii of uniform line are enough to keep the waveguide
# port's evanescent near-field away from the device under test while
# the runs stay near-instant.  ``z_formula`` is only a plausibility
# reference for the impedance printout — no result below depends on
# it.

L = 5.0 * r_o  # waveguide port to reference plane [m]
z_formula = ETA0 / (2.0 * np.pi * np.sqrt(eps_r)) * np.log(r_o / r_i)
print(f"closed-form line impedance: {z_formula:.2f} Ohm")

# %%
# The measurement machinery
# -------------------------
#
# One builder for the coax cross-section, used twice: ``reference()``
# runs the plain line with waveguide ports at both ends — its
# transmission phase is the grid's own propagation over ``L``, the
# ruler every phase error below is read against.  ``measure()``
# terminates the line with the lumped port candidate instead: inner
# conductor up to the gap start, gap, shorted end plate; the lumped
# port bridges the gap on the axis.
#
# The test grid pins ``max_cell_size = min_cell_size``, so the feed
# is uniform and the two runs see the same line per unit length.


def _base_model(length, pin_length):
    model = mio.GeometryModel(background="pec")
    dielectric = geo.Cylinder(
        origin=(0.0, 0.0, 0.0),
        radius=r_o,
        height=length,
        axis="z",
        material=mio.Material.from_isotropic(name="polyethylene", epsilon=eps_r),
    )
    inner = geo.Cylinder(
        origin=(0.0, 0.0, 0.0),
        radius=r_i,
        height=pin_length,
        axis="z",
        material="pec",
    )
    model.add(geo.Difference(dielectric, inner))
    model.add(inner)
    model.add_port(ports.PortWaveguide(name="wg", plane="zmin"))
    return model


def _mesh(model):
    return mio.Mesh.from_geometry(
        model,
        mio.MeshControl(min_cell_size=cell, max_cell_size=cell),
        f_max=f_max,
    )


def reference():
    model = _base_model(L, pin_length=L)
    model.add_port(ports.PortWaveguide(name="far", plane="zmax"))
    mesh = _mesh(model)
    analysis = mio.AnalysisScatteringTD(mesh=mesh, verbose=False)
    return analysis.run(excited=[("wg", 0)])


def measure(gap, gap_position, z0=None):
    z_pin = L + gap_position  # inner conductor ends here
    z_end = z_pin + gap  # shorted end plate
    model = _base_model(z_end, pin_length=z_pin)
    mesh = _mesh(model)

    analysis = mio.AnalysisScatteringTD(mesh=mesh, verbose=False)
    z_line = analysis.solve_ports()["wg"].z_line_num

    model.add_port(
        ports.PortLumped(
            name="dut",
            start=(0.0, 0.0, z_end),
            end=(0.0, 0.0, z_pin),
            Z0=float(z0 if z0 is not None else z_line),
        )
    )
    analysis = mio.AnalysisScatteringTD(
        mesh=mesh,
        ports=list(model.ports),
        verbose=False,
    )
    result = analysis.run(excited=[("wg", 0)])
    return result, z_line, model, mesh


ref = reference()
f = np.asarray(ref.f_axis)
band = f <= f_max
phase_ruler = ref.phase("far", "wg")


def phase_error(result):
    return result.phase("dut", "wg") - phase_ruler


def scoreboard(result, label="current settings"):
    s11_db = result.db("wg", "wg")
    good = s11_db[band] < -20.0
    f_edge = f[band][np.argmin(good)] if not good.all() else f[band][-1]
    err = np.abs(phase_error(result)[band]).max()
    print(f"--- {label} ---")
    print(f"worst |S11| in band : {s11_db[band].max():6.1f} dB")
    print(f"|S11| < -20 dB up to: {f_edge / 1e9:6.2f} GHz")
    print(f"max |phase error|   : {err:6.2f} deg")
    return s11_db, phase_error(result)


# %%
# Your termination, measured
# --------------------------
#
# The scoreboard for the knob settings above — this is the loop you
# iterate: edit the knobs, re-run, read these three numbers.

result, z_line, model, mesh = measure(gap, gap_position, z0_port)
print(f"line impedance on this grid: {z_line:.2f} Ohm ({100 * (z_line / z_formula - 1):+.1f} %)")
s11_db, ph_err = scoreboard(result)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6))
ax1.plot(f[band] / 1e9, s11_db[band])
ax1.set_xlabel("frequency [GHz]")
ax1.set_ylabel("|S11| [dB]")
ax1.set_title("Self-reflection")
ax1.grid(True, alpha=0.3)
ax2.plot(f[band] / 1e9, ph_err[band])
ax2.set_xlabel("frequency [GHz]")
ax2.set_ylabel("phase error [deg]")
ax2.set_title("Phase error at the reference plane")
ax2.grid(True, alpha=0.3)
fig.tight_layout()

# %%
# The cross-section with the mesh it is actually solved on — check
# that the gap region resolves the way your production model does
# before trusting any number above.

fig, ax = plots.plot_cross_section(model, "z", L / 2, mesh=mesh, title="Test coax cross-section")

# %%
# Sensitivity: gap length
# -----------------------
#
# The gap length moves the broadband reflection level.  Re-running
# the measurement over a few values shows how sharp the optimum is on
# this grid — this sweep *is* the optimisation loop, and wrapping
# ``measure`` in :func:`scipy.optimize.minimize_scalar` automates the
# last fraction of a dB if you need it.

fig, ax = plt.subplots()
for factor in (0.2, 0.4, 0.6):
    g = factor * (r_o - r_i)
    res_g, _, _, _ = measure(g, gap_position, z0_port)
    ax.plot(
        f[band] / 1e9,
        res_g.db("wg", "wg")[band],
        label=f"gap = {factor:.1f} × (r_o − r_i)",
    )
ax.set_xlabel("frequency [GHz]")
ax.set_ylabel("|S11| [dB]")
ax.set_title("Gap-length sensitivity of the self-reflection")
ax.grid(True, alpha=0.3)
ax.legend()

# %%
# Sensitivity: gap position
# -------------------------
#
# The termination does not act at the end plate: the fields detour
# around the pin end, so the effective plane sits somewhere inside —
# and the gap has to be shifted off the reference plane to
# compensate.  Each candidate below is its own geometry and its own
# run, exactly as it will be in the target simulation, where this one
# position is the compromise you commit to.  On this TEM line the
# best position works across the whole band; repeat the sweep on a
# dispersive line and the curves tilt — then the choice depends on
# which part of the band matters most.

fig, ax = plt.subplots()
for k in (0.0, -1.0, -2.0, -3.0):
    res_k, _, _, _ = measure(gap, k * gap, z0_port)
    err_k = phase_error(res_k)
    ax.plot(f[band] / 1e9, err_k[band], label=f"gap start at {k:.0f}·gap")
    print(f"gap start at {k:.0f}·gap: max |phase error| {np.abs(err_k[band]).max():6.2f} deg")
ax.set_xlabel("frequency [GHz]")
ax.set_ylabel("phase error [deg]")
ax.set_title("Gap-position sensitivity of the phase error")
ax.grid(True, alpha=0.3)
ax.legend()

# %%
# Sensitivity: port impedance
# ---------------------------
#
# The same termination with the catalogue 50 Ω instead of the grid's
# own line impedance: the constant mismatch
# :math:`(Z - Z_0)/(Z + Z_0)` sets a frequency-independent floor
# under the reflection.  With the grid impedance the floor drops to
# whatever the gap geometry itself scatters.

res_50, _, _, _ = measure(gap, gap_position, 50.0)

fig, ax = plt.subplots()
ax.plot(f[band] / 1e9, s11_db[band], label="Z0 = grid line impedance")
ax.plot(f[band] / 1e9, res_50.db("wg", "wg")[band], label="Z0 = 50 Ohm")
ax.set_xlabel("frequency [GHz]")
ax.set_ylabel("|S11| [dB]")
ax.set_title("Port-impedance sensitivity of the self-reflection")
ax.grid(True, alpha=0.3)
ax.legend()

# %%
# Carry it over
# -------------
#
# Three knobs, three effects, all of them properties of *this* grid:
# the **gap length** sets the reflection level, the **gap position**
# sets the phase error, the **port impedance** sets the low-frequency
# reflection floor.  Once the scoreboard meets your spec, transfer
# ``gap``, ``gap_position`` (relative to where your reference plane
# is) and the port impedance into the target model — and when its
# cross-section, resolution or band changes, run this page again: the
# optimum does not transfer between grids.

scoreboard(result, label="current settings — tune until this meets your spec")
plt.show()
