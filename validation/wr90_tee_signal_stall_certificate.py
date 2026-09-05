"""Certificate: port-signal stall watchdog on the WR-90 magic tee (DD-122).

The E-arm drive of the magic tee leaves TE10 band-edge ringing at the
cut-off (6.56 GHz, vanishing group velocity) whose modal-port |V|
envelope plateaus near -57 dB below peak — just above the -60 dB
``port_signal_stop_db="auto"`` threshold.  Before DD-122 the default
unbounded run marched indefinitely (>40 000 extra steps observed with
no envelope movement); the stall watchdog must now prove the threshold
unreachable before the runtime cap, accept the plateau as the
effective floor, and stop within about one detection window past the
arming step.

Asserts, with pure defaults (plus ``taper_signals=True`` on both runs
for a like-for-like S comparison):

* the run ends with ``stop_reason == "port_signal_stall"`` well before
  the auto runtime cap,
* the booked plateau level sits between the arming floor (-40 dB) and
  the auto threshold (-60 dB),
* |dS| against a ``port_signal_stop_db=50`` reference (the tutorial-06
  workaround) stays below 2e-3 across the design band.
"""

import tempfile
import time
import warnings

import numpy as np

import magnelio as mio
from magnelio import geo, ports

a = 22.86e-3  # WR-90 broad wall
b = 10.16e-3  # WR-90 narrow wall
arm = 30.0e-3
f_min = 8.2e9
f_max = 12.4e9


def build_analysis(project=None):
    pec = mio.Material.pec()
    air = mio.Material.from_isotropic(name="air", epsilon=1.0)
    model = mio.GeometryModel(background=pec)
    model.add(
        geo.Brick(origin=(-(a / 2 + arm), -a / 2, 0.0), size=(a + 2 * arm, a, b), material=air)
    )
    model.add(geo.Brick(origin=(-a / 2, 0.0, 0.0), size=(a, a / 2 + arm, b), material=air))
    model.add(geo.Brick(origin=(-b / 2, -a / 2, 0.0), size=(b, a, b + arm), material=air))
    model.add_port(ports.PortWaveguide(name="port1", plane="xmin", n_modes=1))
    model.add_port(ports.PortWaveguide(name="port2", plane="xmax", n_modes=1))
    model.add_port(ports.PortWaveguide(name="port3", plane="ymax", n_modes=1))
    model.add_port(ports.PortWaveguide(name="port4", plane="zmax", n_modes=1))
    mesh = mio.Mesh.from_geometry(
        model,
        mio.MeshControl(min_nodes_per_wavelength=15, min_cell_size=1.59e-3),
        f_max=f_max,
    )
    return mio.AnalysisScatteringTD(
        mesh=mesh, f_min=f_min, f_max=f_max, verbose=False, project=project, geometry=model
    )


# Reference: the tutorial-06 workaround (criterion fires before the
# plateau) — the converged answer this certificate compares against.
t0 = time.perf_counter()
ref = build_analysis().run(excited=["port4"], port_signal_stop_db=50.0, taper_signals=True)
t_ref = time.perf_counter() - t0
print(f"reference (psd=50): {ref.settings.n_actual_steps} steps  [{t_ref:.1f} s]")

# Certificate: pure defaults.  Streamed so the run index books the
# stop reason and the achieved plateau level.
proj_dir = tempfile.mkdtemp(prefix="stall_cert_") + "/tee"
t0 = time.perf_counter()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    proj = build_analysis(project=proj_dir).run(excited=["port4"], taper_signals=True)
t_cert = time.perf_counter() - t0

info = proj.runs["port4_mode0"]
n_stall = info.n_steps
level = info.final_port_signal_db
print(
    f"defaults: {n_stall} steps, stop_reason={info.get('stop_reason')!r}, "
    f"plateau at {level:.1f} dB  [{t_cert:.1f} s]"
)
stall_warnings = [w for w in caught if "stalled" in str(w.message)]
print(f"warnings: {len(stall_warnings)} stall warning(s)")

d_s = 0.0
for out in ("port1", "port2", "port3", "port4"):
    d = np.max(np.abs(proj.S(out, "port4") - ref.S(out, "port4")))
    d_s = max(d_s, float(d))
    print(f"|dS({out},port4)| = {d:.2e}")

assert info.stop_reason == "port_signal_stall", info.stop_reason
assert len(stall_warnings) == 1
assert -60.0 < level < -40.0, level
# Well before the auto cap (40x the step estimate >> the stall step):
# one detection window past arming, not tens of thousands of steps.
assert n_stall < 4 * ref.settings.n_actual_steps, n_stall
assert d_s < 2e-3, d_s
print(f"CERTIFICATE PASSED: stall stop at step {n_stall} ({level:.1f} dB), max |dS| = {d_s:.2e}")
