"""The dispersive port source through the public path (DD-248).

Tutorial 09's straight 50 ohm microstrip reflects nothing in theory, so
its reported |S11| is the port's own error.  With the frozen source it
reads about -32.9 dB; the rank-r source, which carries the exact
per-frequency decomposition with it, has to do materially better.
"""

from __future__ import annotations

import numpy as np
import pytest

import magnelio as mio
from magnelio import geo
from magnelio import ports as pub_ports

H_SUB, W_STRIP, T_STRIP = 0.8e-3, 1.2e-3, 0.2e-3
W_BOX, H_BOX, LENGTH = 8.0e-3, 5.0e-3, 20.0e-3
EPS_R, F_MAX, NPW = 4.3, 15.0e9, 25


def _model():
    fr4 = mio.Material.from_isotropic(name="FR4", epsilon=EPS_R)
    model = mio.GeometryModel(boundary_conditions={"xmin": "SymmetryPMC"})
    model.add(geo.Brick(origin=(-W_BOX / 2, 0.0, 0.0), size=(W_BOX, H_SUB, LENGTH), material=fr4))
    air = geo.Brick(
        origin=(-W_BOX / 2, H_SUB, 0.0), size=(W_BOX, H_BOX - H_SUB, LENGTH), material="air"
    )
    strip = geo.Brick(
        origin=(-W_STRIP / 2, H_SUB, 0.0), size=(W_STRIP, T_STRIP, LENGTH), material="pec"
    )
    model.add(geo.Difference(air, strip))
    model.add(strip)
    model.add_port(pub_ports.PortWaveguide(name="port1", plane="zmin", n_modes=1))
    model.add_port(pub_ports.PortWaveguide(name="port2", plane="zmax", n_modes=1))
    return model


@pytest.fixture(scope="module")
def mesh():
    return mio.Mesh.from_geometry(
        _model(), mio.MeshControl(min_nodes_per_wavelength=NPW), f_max=F_MAX
    )


def _worst_s11_db(mesh, port_source):
    result = mio.AnalysisScatteringTD(mesh=mesh, verbose=False, port_source=port_source).run(
        excited=["port1"]
    )
    s11 = np.abs(np.asarray(result.S("port1", "port1")))
    return 20.0 * np.log10(float(s11.max())), result


class TestDispersiveSourceRun:
    def test_it_beats_the_frozen_source_on_a_microstrip(self, mesh):
        frozen, _ = _worst_s11_db(mesh, "frozen")
        disp, res = _worst_s11_db(mesh, "dispersive")
        # Frozen reproduces the tutorial's printed number.
        assert frozen == pytest.approx(-32.9, abs=0.6)
        # The gain measured on this fixture is ~6 dB; the gate keeps
        # margin for grid and platform noise but fails if the coupling
        # to the per-frequency split is ever broken, which shows up as
        # the dispersive arm being *worse* than the frozen one.
        assert disp < frozen - 3.0
        assert res.port_source_used == "dispersive"

    def test_transmission_stays_on_a_lossless_line(self, mesh):
        """|S21| stays at unity within the split's own passivity error.

        The per-frequency decomposition overshoots slightly and does so
        already for the frozen source — measured max |S21| 1.0030 there
        against 1.0078 here, both growing with frequency and neither
        depending on the source rank.  The gate keeps the transmission
        pinned to a lossless line without asserting a passivity the
        decomposition does not currently deliver.
        """
        _, res = _worst_s11_db(mesh, "dispersive")
        s21 = np.abs(np.asarray(res.S("port2", "port1")))
        assert s21.max() <= 1.02
        assert 20.0 * np.log10(float(s21.min())) > -0.5

    def test_frozen_stays_the_default(self, mesh):
        res = mio.AnalysisScatteringTD(mesh=mesh, verbose=False).run(excited=["port1"])
        assert res.port_source_used == "frozen"

    def test_an_unknown_source_is_refused(self, mesh):
        with pytest.raises(ValueError, match="port_source"):
            mio.AnalysisScatteringTD(mesh=mesh, port_source="nope")
