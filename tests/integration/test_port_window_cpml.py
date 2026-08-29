"""Waveguide-port windows in absorbing (CPML) faces (DD-198).

A PEC tube (20 × 10 mm bore, 3 mm walls) reaches the ``xmin`` face of
an air-filled box whose six faces are CPML.  The port sits in the
tube's cross-section on that face.  Two fixtures:

- **through**: the tube spans the box, a second window port on the
  PEC ``xmax`` face.  The guided TE10 wave must cross the absorbing
  layer behind the port untouched — the DD-198 footprint switch-off —
  and both DTBC-terminated ports must be transparent.
- **open end**: the tube ends inside the box and radiates.  The
  reflection of an open waveguide end is a textbook −10 dB class
  number; the far-field monitor's radiated power must balance the
  accepted power; and the same model with a PEC ``xmin`` face (an
  infinite flange) must agree within the flange's own effect.
"""

from __future__ import annotations

import numpy as np
import pytest

import magnelio as mio
from magnelio import geo, monitors, ports
from magnelio.constants import C0

A, B, WALL = 20e-3, 10e-3, 3e-3
F_C = C0 / (2 * A)  # TE10 cut-off, 7.49 GHz
F_MAX = 12e9
F_AXIS = np.linspace(8.6e9, 11.8e9, 33)
CELL = 2e-3


def _tube_model(*, length, box_x, xmin="CPML", extra_ports=()):
    faces = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
    bcs = {f: "CPML" for f in faces}
    bcs["xmin"] = xmin
    if length >= box_x:
        bcs["xmax"] = "PEC"
    tube = geo.Brick(
        origin=(0.0, -A / 2 - WALL, -B / 2 - WALL),
        size=(length, A + 2 * WALL, B + 2 * WALL),
        material="pec",
    )
    bore = geo.Brick(origin=(-1e-3, -A / 2, -B / 2), size=(length + 2e-3, A, B), material="air")
    model = mio.GeometryModel(boundary_conditions=bcs)
    model.add(geo.Difference(tube, bore))
    # An air box fixes the domain extent beyond the tube.
    model.add(
        geo.Difference(
            geo.Brick(origin=(0.0, -0.03, -0.025), size=(box_x, 0.06, 0.05), material="air"),
            tube,
        )
    )
    window = ((None, -A / 2, -B / 2), (None, A / 2, B / 2))
    model.add_port(ports.PortWaveguide(name="feed", plane="xmin", corners=window))
    for p in extra_ports:
        model.add_port(p)
    return model


def _mesh(model):
    return mio.Mesh.from_geometry(model, mio.MeshControl(max_cell_size=CELL), f_max=F_MAX)


def _in_band(f):
    return f >= 1.15 * F_C


class TestThroughTube:
    def test_guided_wave_crosses_the_absorber_behind_the_port(self):
        model = _tube_model(
            length=40e-3,
            box_x=40e-3,
            extra_ports=(
                ports.PortWaveguide(
                    name="far",
                    plane="xmax",
                    corners=((None, -A / 2, -B / 2), (None, A / 2, B / 2)),
                ),
            ),
        )
        mesh = _mesh(model)
        analysis = mio.AnalysisScatteringTD(mesh=mesh, f_min=8.5e9, verbose=False)
        report = analysis.solve_ports()["feed"]
        assert report.modes[0].f_cutoff == pytest.approx(F_C, rel=0.02)
        result = analysis.run(f_axis=F_AXIS, excited=["feed"])
        s11 = np.abs(result.S("feed", "feed"))[_in_band(F_AXIS)]
        s21 = np.abs(result.S("far", "feed"))[_in_band(F_AXIS)]
        assert 20 * np.log10(s21).min() > -0.5
        assert 20 * np.log10(s11).max() < -40.0


class TestOpenEnd:
    @pytest.fixture(scope="class")
    def radiating(self):
        model = _tube_model(length=25e-3, box_x=60e-3)
        mesh = _mesh(model)
        ff = monitors.MonitorFarFieldFrequency(freqs=[10e9], name="ff")
        analysis = mio.AnalysisScatteringTD(mesh=mesh, f_min=8.5e9, monitors=(ff,), verbose=False)
        result = analysis.run(f_axis=F_AXIS, excited=["feed"])
        return result, ff

    def test_reflection_is_that_of_an_open_waveguide_end(self, radiating):
        result, _ = radiating
        s11_db = 20 * np.log10(np.abs(result.S("feed", "feed")))[_in_band(F_AXIS)]
        assert -16.0 < s11_db.max() < -5.0
        assert np.all(np.abs(result.S("feed", "feed")) <= 1.0 + 1e-6)

    def test_radiated_power_balances_the_accepted_power(self, radiating):
        result, ff = radiating
        pattern = ff.result(10e9)
        s11 = np.interp(10e9, F_AXIS, np.abs(result.S("feed", "feed")))
        # Within the feed-guide approximation: the currents on the guide's
        # outer wall beyond the box are what the absorber removes (measured 0.97).
        assert pattern.P_rad == pytest.approx(1.0 - s11**2, rel=0.06)

    def test_flanged_variant_agrees(self, radiating):
        result, _ = radiating
        model = _tube_model(length=25e-3, box_x=60e-3, xmin="PEC")
        mesh = _mesh(model)
        flanged = mio.AnalysisScatteringTD(mesh=mesh, f_min=8.5e9, verbose=False).run(
            f_axis=F_AXIS, excited=["feed"]
        )
        sel = _in_band(F_AXIS)
        a = 20 * np.log10(np.abs(result.S("feed", "feed")))[sel]
        b = 20 * np.log10(np.abs(flanged.S("feed", "feed")))[sel]
        assert np.abs(a - b).max() < 3.0


class TestRefusals:
    def test_whole_face_port_on_an_absorbing_face(self):
        model = _tube_model(length=25e-3, box_x=60e-3)
        model.ports.clear()
        model.add_port(ports.PortWaveguide(name="feed", plane="xmin"))
        mesh = _mesh(model)
        with pytest.raises(ValueError, match="covers the whole face"):
            mio.AnalysisScatteringTD(mesh=mesh, f_min=8.5e9, verbose=False).solve_ports()

    def test_window_wider_than_the_guide(self):
        model = _tube_model(length=25e-3, box_x=60e-3)
        model.ports.clear()
        model.add_port(
            ports.PortWaveguide(
                name="feed",
                plane="xmin",
                corners=((None, -A / 2 - 2 * WALL, -B / 2), (None, A / 2 + 2 * WALL, B / 2)),
            )
        )
        mesh = _mesh(model)
        with pytest.raises(ValueError, match="not enclosed by conductor"):
            mio.AnalysisScatteringTD(mesh=mesh, f_min=8.5e9, verbose=False).solve_ports()
