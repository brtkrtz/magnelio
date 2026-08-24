"""
Characterising a lumped port termination: coaxial line
======================================================

A :class:`~magnelio.ports.PortLumped` terminates a line in a single
grid edge — cheap, DC-capable, and available where a waveguide port
window does not fit.  The price is that it is not an exact line
termination: a residual self-reflection and a phase offset remain,
and both depend on the grid resolution at the conductor gap, on the
gap geometry, and on the port impedance.  Rules of thumb exist, but
none of them tells you how good *your* termination is on *your* mesh,
or up to which frequency you can trust it.

This guide measures it instead of guessing.  A waveguide port —
reflection-free by construction, with a measured floor far below
anything a lumped element reaches — launches the exact line mode down
a short uniform coax toward the lumped port under test:

- ``|S11|`` at the waveguide port *is* the lumped termination's
  self-reflection (the uniform line in between adds none of its own);
- the transmission phase, after de-embedding the line with
  :meth:`~magnelio.analysis.result_interface.ScatteringResultMixin.deembed`,
  is the termination's phase error against an ideal load at the
  chosen reference plane.

Run it with your own cross-section, resolution and band, then turn
the three knobs at the top — gap length, gap position relative to the
reference plane, port impedance — until reflection and phase meet
your requirements.  The numbers this page prints are properties of
*this* example's grid, not constants of the method: re-measure them
whenever the cross-section or the resolution changes.
"""

# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import numpy as np

import magnelio as mio
from magnelio import geo, plots, ports
from magnelio.constants import ETA0

# %%
# Parameters
# ----------
#
# The coax is an RG-58 class cable.  ``cell`` is the cross-section
# cell size the test mesh may not fall below — set it to the size your
# production mesh will actually have at the port, because the
# termination quality is a property of the discretised gap, not of the
# continuous geometry.
#
# The knobs:
#
# - ``gap`` — length of the end gap between inner conductor and the
#   end plane; the lumped port element spans it.
# - the **reference plane** the termination should emulate.  It costs
#   nothing to move: de-embedding re-references the *same* run, so
#   the phase section below sweeps it without re-simulating.
# - ``z0_port`` — the lumped port impedance.  ``None`` uses the line
#   impedance *of the grid*, as the waveguide-port solver measures it
#   below; a number (e.g. the catalogue 50 Ω) uses that instead.

r_i = 0.405e-3  # inner conductor radius [m]
r_o = 1.475e-3  # shield (dielectric outer) radius [m]
eps_r = 2.25  # solid polyethylene
f_max = 15e9  # upper band edge [Hz]
cell = 0.5 * r_i  # production cross-section cell size [m]

L = 10e-3  # test line length, waveguide port to end plane [m]
gap = 0.4 * (r_o - r_i)  # end gap length [m]
z0_port = None  # None -> grid line impedance; or e.g. 50.0

z_formula = ETA0 / (2.0 * np.pi * np.sqrt(eps_r)) * np.log(r_o / r_i)
print(f"closed-form line impedance: {z_formula:.2f} Ohm")

# %%
# One function builds and measures a candidate termination, so the
# sensitivity sweeps below are one-liners.  The geometry is the coax
# of the tutorials — dielectric annulus, PEC inner conductor, shield
# from the PEC background — with the inner conductor stopping ``gap``
# short of the dielectric's end plane, and the lumped port bridging
# that gap on the axis.  The waveguide port at ``zmin`` is the
# measuring instrument.


def measure(gap, z0=None, cell=cell):
    z_pin = L - gap  # inner conductor stops here

    model = mio.GeometryModel(background="pec")
    dielectric = geo.Cylinder(
        origin=(0.0, 0.0, 0.0),
        radius=r_o,
        height=L,
        axis="z",
        material=mio.Material.from_isotropic(name="polyethylene", epsilon=eps_r),
    )
    inner = geo.Cylinder(
        origin=(0.0, 0.0, 0.0),
        radius=r_i,
        height=z_pin,
        axis="z",
        material="pec",
    )
    model.add(geo.Difference(dielectric, inner))
    model.add(inner)
    model.add_port(ports.PortWaveguide(name="wg", plane="zmin"))

    # Uniform test grid: pinning max = min removes mesh grading along
    # the line, so the feed the de-embedding assumes is exactly the
    # grid that is there.  The measured termination quality is then a
    # property of the discretised gap alone.
    mesh = mio.Mesh.from_geometry(
        model,
        mio.MeshControl(min_cell_size=cell, max_cell_size=cell),
        f_max=f_max,
    )

    analysis = mio.AnalysisScatteringTD(mesh=mesh, verbose=False)
    z_line = analysis.solve_ports()["wg"].z_line_num

    model.add_port(
        ports.PortLumped(
            name="dut",
            start=(0.0, 0.0, L),
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


result, z_line, model, mesh = measure(gap, z0_port)
print(f"line impedance on this grid: {z_line:.2f} Ohm ({100 * (z_line / z_formula - 1):+.1f} %)")

# %%
# The cross-section with the mesh it will actually be solved on —
# check that the gap region resolves the way your production model
# does before trusting any number below.

fig, ax = plots.plot_cross_section(model, "z", L / 2, mesh=mesh, title="Test coax cross-section")

# %%
# Self-reflection
# ---------------
#
# The waveguide port launches 1 W of the exact TEM grid mode; the
# uniform line is reflection-free, so everything that comes back is
# the lumped termination.  The print states the highest frequency up
# to which the termination stays below −20 dB.

f = np.asarray(result.f_axis)
band = f <= f_max
s11_db = result.db("wg", "wg")

fig, ax = plt.subplots()
ax.plot(f[band] / 1e9, s11_db[band])
ax.set_xlabel("frequency [GHz]")
ax.set_ylabel("|S11| [dB]")
ax.set_title("Self-reflection of the lumped termination")
ax.grid(True, alpha=0.3)

good = s11_db[band] < -20.0
f_ok = f[band][np.argmin(good)] if not good.all() else f[band][-1]
print(f"|S11| < -20 dB up to {f_ok / 1e9:.2f} GHz")

# %%
# Phase error, and where the termination actually sits
# ----------------------------------------------------
#
# ``deembed`` shifts the waveguide port's reference plane by exactly
# the propagation the grid applied — so if the lumped port behaved as
# an ideal load at that plane, the de-embedded transmission phase
# would be zero across the band.  What remains is the termination's
# phase error *relative to that plane*.
#
# Because de-embedding is post-processing, sweeping the reference
# plane costs nothing: the loop below re-references the **same run**
# to a few candidate planes around the end plane.  The flattest curve
# tells you where the termination effectively sits on this grid; in
# your production model, place the gap so that this plane lands where
# the reference should be — or simply de-embed your production result
# by the offset found here.

fig, ax = plt.subplots()
for k in (0.0, 1.0, 2.0, 3.0):
    de = result.deembed({"wg": L + k * gap})
    phase_err = de.phase("dut", "wg", unwrap=False)
    ax.plot(f[band] / 1e9, phase_err[band], label=f"plane at L + {k:.0f}·gap")
    print(f"plane at L + {k:.0f}·gap: max |phase error| {np.abs(phase_err[band]).max():6.2f} deg")
ax.set_xlabel("frequency [GHz]")
ax.set_ylabel("arg S21, de-embedded [deg]")
ax.set_title("Phase error vs. assumed reference plane (one run, re-referenced)")
ax.grid(True, alpha=0.3)
ax.legend()

# %%
# Sensitivity: gap length
# -----------------------
#
# Re-run the measurement over a few gap lengths to see how sharp the
# optimum is on this grid.  This sweep *is* the optimisation loop —
# wrap ``measure`` in :func:`scipy.optimize.minimize_scalar` if you
# want the last fraction of a dB automated.

fig, ax = plt.subplots()
for factor in (0.2, 0.4, 0.6):
    g = factor * (r_o - r_i)
    res_g, _, _, _ = measure(g, z0_port)
    fg = np.asarray(res_g.f_axis)
    bg = fg <= f_max
    ax.plot(
        fg[bg] / 1e9,
        res_g.db("wg", "wg")[bg],
        label=f"gap = {factor:.1f} × (r_o − r_i)",
    )
ax.set_xlabel("frequency [GHz]")
ax.set_ylabel("|S11| [dB]")
ax.set_title("Gap-length sensitivity of the self-reflection")
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

res_50, _, _, _ = measure(gap, 50.0)
f50 = np.asarray(res_50.f_axis)
b50 = f50 <= f_max

fig, ax = plt.subplots()
ax.plot(f[band] / 1e9, s11_db[band], label="Z0 = grid line impedance")
ax.plot(f50[b50] / 1e9, res_50.db("wg", "wg")[b50], label="Z0 = 50 Ohm")
ax.set_xlabel("frequency [GHz]")
ax.set_ylabel("|S11| [dB]")
ax.set_title("Port-impedance sensitivity of the self-reflection")
ax.grid(True, alpha=0.3)
ax.legend()

# %%
# Reading the result
# ------------------
#
# Three knobs, three effects, all of them properties of *this* grid:
#
# - **gap length** moves the broadband reflection level;
# - **gap position** relative to the reference plane rotates the
#   phase — tune it when the phase error matters (multi-port devices,
#   deliberate reference-plane matching);
# - **port impedance** sets the low-frequency floor; the grid's own
#   line impedance from ``solve_ports`` removes the deterministic
#   part of the mismatch.
#
# When the cross-section, the resolution or the band of your
# production model changes, run this page again — the optimum does
# not transfer between grids.

plt.show()
