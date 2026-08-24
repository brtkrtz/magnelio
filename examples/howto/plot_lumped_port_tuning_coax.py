"""
Lumped port tuning: coaxial line
================================

A pre-flight check for a lumped port terminating a coaxial line:
fill in the given quantities of your target model, run, and the
scoreboard tells you how good the termination is — worst reflection,
usable band, phase error.  Edit the three knobs and re-run until the
numbers meet your spec, then carry the settings over.

The measurement: a waveguide port launches the exact grid mode onto
the lumped port under test (``|S11|`` is the termination's
self-reflection), and a reference run of the same line with waveguide
ports at both ends provides the grid-exact phase ruler.  The page
:doc:`plot_lumped_port_investigations` explains the setup and shows
which knob moves which error; every number here is a property of
*your* grid, so re-run whenever cross-section, resolution or band
change.
"""

# sphinx_gallery_thumbnail_number = 1

import matplotlib.pyplot as plt
import numpy as np

import magnelio as mio
from magnelio import geo, plots, ports
from magnelio.constants import ETA0

# %%
# Given quantities
# ----------------
#
# What the target simulation dictates.  ``cell`` is the cross-section
# cell size the test mesh may not fall below — copy the size your
# production mesh will actually have at the port.

r_i = 0.405e-3  # inner conductor radius [m]
r_o = 1.475e-3  # shield (dielectric outer) radius [m]
eps_r = 2.25  # solid polyethylene
f_max = 15e9  # upper band edge [Hz]
cell = 0.5 * r_i  # production cross-section cell size [m]

# %%
# The knobs
# ---------
#
# The compromise you will carry into the target simulation:
#
# - ``gap`` — end gap between inner conductor and shorted end plate;
#   sets the broadband reflection level.
# - ``gap_position`` — where the gap starts relative to the reference
#   plane (negative = before it); sets the phase error.
# - ``z0_port`` — ``None`` uses the line impedance of the grid, as
#   measured by the waveguide-port solver; a number (e.g. 50.0) uses
#   that instead.  Sets the low-frequency reflection floor.

gap = 0.4 * (r_o - r_i)
gap_position = 0.0
z0_port = None

# %%
# Derived quantities
# ------------------

L = 5.0 * r_o  # waveguide port to reference plane [m]
z_formula = ETA0 / (2.0 * np.pi * np.sqrt(eps_r)) * np.log(r_o / r_i)
print(f"closed-form line impedance: {z_formula:.2f} Ohm")

# %%
# Measurement machinery — a uniform test grid (``max = min`` cell
# size), the plain line as phase reference, and the candidate
# termination: inner conductor to the gap start, gap, shorted end
# plate, lumped port bridging the gap on the axis.


def _model(length, pin_length):
    model = mio.GeometryModel(background="pec")
    dielectric = geo.Cylinder(
        origin=(0.0, 0.0, 0.0),
        radius=r_o,
        height=length,
        axis="z",
        material=mio.Material.from_isotropic(name="polyethylene", epsilon=eps_r),
    )
    inner = geo.Cylinder(
        origin=(0.0, 0.0, 0.0), radius=r_i, height=pin_length, axis="z", material="pec"
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


ref_model = _model(L, pin_length=L)
ref_model.add_port(ports.PortWaveguide(name="far", plane="zmax"))
ref = mio.AnalysisScatteringTD(mesh=_mesh(ref_model), verbose=False).run(excited=[("wg", 0)])

z_pin = L + gap_position
z_end = z_pin + gap
model = _model(z_end, pin_length=z_pin)
mesh = _mesh(model)
z_line = mio.AnalysisScatteringTD(mesh=mesh, verbose=False).solve_ports()["wg"].z_line_num
print(f"line impedance on this grid: {z_line:.2f} Ohm ({100 * (z_line / z_formula - 1):+.1f} %)")

model.add_port(
    ports.PortLumped(
        name="dut",
        start=(0.0, 0.0, z_end),
        end=(0.0, 0.0, z_pin),
        Z0=float(z0_port if z0_port is not None else z_line),
    )
)
result = mio.AnalysisScatteringTD(mesh=mesh, ports=list(model.ports), verbose=False).run(
    excited=[("wg", 0)]
)

# %%
# The scoreboard
# --------------
#
# The phase error is polarity-normalised: the sign of a mode profile
# is a convention, so the error is referenced to the nearest multiple
# of 180° at the low end of the band.

f = np.asarray(result.f_axis)
band = f <= f_max
s11_db = result.db("wg", "wg")
err = result.phase("dut", "wg") - ref.phase("far", "wg")
err -= 180.0 * np.round(err[int(np.argmax(f >= f_max / 15.0))] / 180.0)

good = s11_db[band] < -20.0
f_edge = f[band][np.argmin(good)] if not good.all() else f[band][-1]
print("--- current settings — tune until this meets your spec ---")
print(f"worst |S11| in band : {s11_db[band].max():6.1f} dB")
print(f"|S11| < -20 dB up to: {f_edge / 1e9:6.2f} GHz")
print(f"max |phase error|   : {np.abs(err[band]).max():6.2f} deg")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6))
ax1.plot(f[band] / 1e9, s11_db[band])
ax1.set_xlabel("frequency [GHz]")
ax1.set_ylabel("|S11| [dB]")
ax1.set_title("Self-reflection")
ax1.grid(True, alpha=0.3)
ax2.plot(f[band] / 1e9, err[band])
ax2.set_xlabel("frequency [GHz]")
ax2.set_ylabel("phase error [deg]")
ax2.set_title("Phase error at the reference plane")
ax2.grid(True, alpha=0.3)
fig.tight_layout()

# %%
# The cross-section with the mesh it is actually solved on — check
# that the gap region resolves the way your production model does.

fig, ax = plots.plot_cross_section(model, "z", L / 2, mesh=mesh, title="Test coax cross-section")

# %%
# Carry it over
# -------------
#
# Once the three numbers meet your spec, transfer ``gap``,
# ``gap_position`` (relative to where your reference plane is) and
# the port impedance into the target model.  Which knob moves which
# number — and why — is shown in
# :doc:`plot_lumped_port_investigations`; when the cross-section,
# resolution or band changes, run this page again.
