"""
31 — Use a named end face for a port and an observation plane
==============================================================

Retrieve a face from the placed owner, then declare a boundary waveguide port
and a field-frequency recording from the same physical rectangle.
"""

from magnelio import GeometryModel, geo, monitors, ports

# %%
# Place and name the end face
# ---------------------------

source = geo.Brick(size=(2e-3, 3e-3, 4e-3), material="air")
source = source.tag_face("output", normal="z")
placed = geo.Translation((5e-3, 0, 0)) @ source
model = GeometryModel(background="pec").add(placed)
end = placed.face("output")

# %%
# Declare the EM consumers from that owned face
# ---------------------------------------------

port = ports.PortWaveguide.from_face(end, model=model, name="output")
model.add_port(port)
recording = monitors.MonitorFieldFrequency.from_face(
    end, freqs=[10e9], fields=["Ex", "Ey"], name="output_fields"
)
assert port.plane == "zmax"
assert port.corners == recording.corners
assert recording.corners[0][0] == 5e-3

# Field recordings may also use a different selected plane.
side = placed.face(normal="x")
side_recording = monitors.MonitorFieldFrequency.from_face(
    side, freqs=[10e9], fields=["Ez"], name="side_fields"
)
assert side_recording.corners[0][0] == 7e-3
