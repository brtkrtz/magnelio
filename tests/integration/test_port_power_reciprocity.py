"""Integration: mixed-port power consistency (DD-095).

The round -> square PTFE coax transition pairs a conformal TEM port
with a staircase TEM port.  Same-type port pairs cancel any per-port
power-scale error exactly (S21 = s2/s1 * T), so this mixed fixture is
the canonical detector for the DD-095 defect class: before the
conformality patch the modal pipeline measured S12 - S21 =
+0.5423 dB frequency-flat here (predicted +0.555 dB from the absolute
power gates; internal dossier investigations/port_power/DERIVATION.md).

Two gates:

* reciprocity: |S12| - |S21| within 0.05 dB across the band
  (post-fix measurement: -0.0001 dB, spread 0.003 dB);
* absolute power: the energy recorded at the output port equals the
  discrete Poynting energy through a mid-line flux plane (the DD-095
  reference definition) to the flux-gate floor.  Separate
  single-excitation run: MonitorFluxTime accumulates across runs.
"""

from __future__ import annotations

import numpy as np

from magnelio import AnalysisScatteringTD, Material, Mesh, MeshControl
from magnelio.geo import Brick, Cylinder, Difference, GeometryModel
from magnelio.monitors import MonitorFluxTime
from magnelio.ports import PortWaveguide

EPS_R = 2.08
RI, RA = 1.27e-3 / 2, 4.11e-3 / 2
LH = 8e-3
F_MIN, F_MAX = 4.0e9, 12.4e9


def _model_and_mesh():
    ptfe = Material.from_isotropic(name="PTFE", epsilon=EPS_R)
    pin_mat = Material.pec()
    pin_mat.name = "COPPER"
    model = GeometryModel(background=Material.pec())
    round_diel = Cylinder(material=ptfe, axis="z", origin=(0, 0, -LH), radius=RA, height=LH)
    square_diel = Brick(material=ptfe, origin=(-RA, -RA, 0), size=(2 * RA, 2 * RA, LH))
    pin_round = Cylinder(material=pin_mat, axis="z", origin=(0, 0, -LH), radius=RI, height=LH)
    pin_square = Brick(material=pin_mat, origin=(-RI, -RI, 0), size=(2 * RI, 2 * RI, LH))
    model.add(
        (
            Difference(round_diel, pin_round),
            Difference(square_diel, pin_square),
            pin_round,
            pin_square,
        )
    )
    mesh = Mesh.from_geometry(
        model,
        MeshControl(min_nodes_per_wavelength=12),
        f_max=F_MAX,
    )
    return model, mesh


def _analysis(model, mesh, monitors=()):
    return AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions(
            {f: "PEC" for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
        ),
        ports=[
            PortWaveguide(name="round", plane="zmin", corners=((-RA, -RA, None), (RA, RA, None))),
            PortWaveguide(name="square", plane="zmax", corners=((-RA, -RA, None), (RA, RA, None))),
        ],
        f_min=F_MIN,
        f_max=F_MAX,
        verbose=False,
        monitors=monitors,
        geometry=model,
    )


class TestMixedPortPower:
    def test_reciprocity(self):
        """|S12| == |S21| across the band (DD-095 detector gate)."""
        model, mesh = _model_and_mesh()
        f_axis = np.linspace(F_MIN, F_MAX, 43)
        res = _analysis(model, mesh).run(
            f_axis=f_axis,
            excited=["round", "square"],
        )
        s21 = res.S("square", "round")
        s12 = res.S("round", "square")
        d_db = 20.0 * np.log10(np.abs(s12) / np.abs(s21))
        assert np.all(np.isfinite(d_db))
        assert np.max(np.abs(d_db)) < 0.05, (
            f"mixed-pair reciprocity violated: S12-S21 in "
            f"[{d_db.min():+.4f}, {d_db.max():+.4f}] dB "
            f"(pre-DD-095 defect: +0.54 dB)"
        )

    def test_port_power_equals_flux(self):
        """Recorded |b|^2 energy == discrete Poynting energy (reference).

        The flux plane sits in the staircase half between the junction
        and the output port: with matched DTBC terminations every
        transmitted joule crosses it exactly once.
        """
        model, mesh = _model_and_mesh()
        mon = MonitorFluxTime(
            plane=("z", LH / 2),
            name="flux",
        )
        res = _analysis(model, mesh, monitors=(mon,)).run(
            excited=[("round", 0)],
        )
        b = res.b("square")
        e_port = float(np.trapezoid(b.values**2, dx=b.dt))
        s2 = e_port / float(mon.total_energy)
        assert 0.97 < s2 < 1.03, (
            f"port/flux power ratio s^2 = {s2:.4f} outside the gate "
            f"floor band (pre-DD-095: conformal coax measured 1.072)"
        )
