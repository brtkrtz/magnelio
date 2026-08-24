"""
Lumped port tuning: coplanar waveguide
======================================

A pre-flight check for lumped ports terminating a coplanar waveguide:
fill in the given quantities of your target model, run, and the
scoreboard tells you how good the termination is — worst reflection,
usable band, phase error.  Edit the knobs and re-run until the
numbers meet your spec, then carry the settings over.

The CPW even mode returns its current through *both* ground strips,
so the termination loads **both slots**: a lumped port across one, a
plain resistor across the other, each with twice the line impedance.
Two fixture rules come with that (measured, not guessed — see
:doc:`plot_lumped_port_investigations`): the slots must be *closed*
one slot-width behind the termination plane, or they run on as
resonating slotline stubs; and the test shield must stay single-mode
over the band, or its own box modes masquerade as termination error.
Every number here is a property of *your* grid.
"""

# sphinx_gallery_thumbnail_number = 1

import matplotlib.pyplot as plt
import numpy as np

import magnelio as mio
from magnelio import circuit, geo, plots, ports

# %%
# Given quantities
# ----------------
#
# The cross-section of the target model plus band and resolution.
# Keep the test shield tight: it must be single-mode up to ``f_max``
# (a roomy box invalidates the top of the measured band).

w_strip = 1.2e-3  # centre strip width [m]
s_slot = 0.4e-3  # slot width [m]
h_sub = 0.8e-3  # substrate height [m]
t_met = 0.2e-3  # metallisation thickness [m]
eps_r = 4.3  # substrate permittivity (FR4)
W_box = 4.0e-3  # shield width [m] — tight, see above
H_box = 2.5e-3  # shield height [m]
f_max = 15e9  # upper band edge [Hz]
n_per_lambda = 25  # mesh resolution [cells per wavelength]

# %%
# The knobs
# ---------
#
# - ``end_position`` — where the centre strip ends relative to the
#   reference plane (negative = before it); sets the phase error.
#   The natural scale of the CPW end effect is the slot width.
# - ``z0_port`` — line impedance to match: ``None`` uses the grid's
#   own value from the waveguide-port solver; each slot load is
#   built with **twice** this value, so the parallel pair presents it
#   to the mode.

end_position = 0.0
z0_port = None

# %%
# Derived quantities
# ------------------

L = 5.0 * w_strip  # waveguide port to reference plane [m]
tail = 2.5 * h_sub  # substrate/air continuing beyond the closing plate [m]

# %%
# Measurement machinery.  The candidate termination: centre strip to
# the end position, lumped port across one slot, resistor across the
# other, and a closing plate over strip and slots one slot-width
# behind — the CPW's equivalent of the coax end plate.  The resistor
# is declared after the mesh, so it is handed to the analysis
# explicitly via ``elements=`` (late-declared elements do not travel
# with the mesh).


def _model(length, strip_len, close_from=None):
    fr4 = mio.Material.from_isotropic(name="FR4", epsilon=eps_r)
    model = mio.GeometryModel(background="pec")
    model.add(geo.Brick(origin=(-W_box / 2, 0.0, 0.0), size=(W_box, h_sub, length), material=fr4))
    air = geo.Brick(
        origin=(-W_box / 2, h_sub, 0.0), size=(W_box, H_box - h_sub, length), material="air"
    )
    metal = [
        geo.Brick(
            origin=(-w_strip / 2, h_sub, 0.0), size=(w_strip, t_met, strip_len), material="pec"
        )
    ]
    for sign in (+1, -1):
        x0, x1 = sign * (w_strip / 2 + s_slot), sign * W_box / 2
        metal.append(
            geo.Brick(
                origin=(min(x0, x1), h_sub, 0.0),
                size=(abs(x1 - x0), t_met, length),
                material="pec",
            )
        )
    if close_from is not None:
        metal.append(
            geo.Brick(
                origin=(-(w_strip / 2 + s_slot), h_sub, close_from),
                size=(w_strip + 2 * s_slot, t_met, length - close_from),
                material="pec",
            )
        )
    model.add(geo.Difference(air, geo.Union(*metal)))
    for m in metal:
        model.add(m)
    model.add_port(ports.PortWaveguide(name="wg", plane="zmin", n_modes=1))
    return model


def _mesh(model):
    return mio.Mesh.from_geometry(
        model,
        mio.MeshControl(min_nodes_per_wavelength=n_per_lambda),
        f_max=f_max,
    )


ref_model = _model(L, strip_len=L)
ref_model.add_port(ports.PortWaveguide(name="far", plane="zmax", n_modes=1))
ref = mio.AnalysisScatteringTD(mesh=_mesh(ref_model), verbose=False).run(excited=[("wg", 0)])

z_pin = L + end_position
model = _model(z_pin + tail, strip_len=z_pin, close_from=z_pin + s_slot)
mesh = _mesh(model)
z_line = mio.AnalysisScatteringTD(mesh=mesh, verbose=False).solve_ports()["wg"].modes[0].z_line
print(f"line impedance on this grid: {z_line:.2f} Ohm")

z_slot = 2.0 * float(z0_port if z0_port is not None else z_line)
model.add_port(
    ports.PortLumped(
        name="dut",
        start=(w_strip / 2, h_sub, z_pin),
        end=(w_strip / 2 + s_slot, h_sub, z_pin),
        Z0=z_slot,
    )
)
load2 = circuit.LumpedElement(
    name="load2",
    start=(-w_strip / 2, h_sub, z_pin),
    end=(-(w_strip / 2 + s_slot), h_sub, z_pin),
    element=circuit.SeriesRLC(R=z_slot),
)
result = mio.AnalysisScatteringTD(
    mesh=mesh, ports=list(model.ports), elements=[load2], verbose=False
).run(excited=[("wg", 0)])

# %%
# The test fixture
# ----------------
#
# A top view of the metallisation plane, cut along the propagation
# direction: centre strip and ground strips with the two slots, the
# strip ending at ``end_position``, the lumped port across one slot
# and the resistor across the other, and the closing plate one
# slot-width behind the termination plane.

model.add_element(load2)  # declare on the model so the plot shows it
fig, ax = plots.plot_cross_section(
    model,
    "y",
    h_sub + t_met / 2,
    flip=True,
    slab=t_met,
    title="CPW test fixture (top view)",
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
# The cross-section with the mesh it is actually solved on.

fig, ax = plots.plot_cross_section(model, "z", L / 2, mesh=mesh, title="Test CPW cross-section")

# %%
# Carry it over
# -------------
#
# Transfer ``end_position``, the slot loading (port plus resistor,
# each at twice the matched impedance) and the closing of the slots
# into the target model once the numbers meet your spec.  Background
# and sweeps: :doc:`plot_lumped_port_investigations`.  When the
# cross-section, resolution or band changes, run this page again.
