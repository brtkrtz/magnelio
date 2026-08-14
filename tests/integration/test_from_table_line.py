"""DD-086 full-chain gate: a from_table material on the DD-084 TEM line.

Table -> vector fit -> Material.dispersive -> FIT-TD solve -> two-length
gamma extraction, compared against gamma computed from the TABLE itself
(complex interpolation) — the reference never touches the fitted model,
so the gate covers the whole chain including the fit error.

Fixture and thresholds follow test_dispersive_materials.py (the DD-081
two-length ratio cancels the lossless-eps_inf port mismatch; the
low-frequency band edge keeps the Fabry-Perot residual, so alpha gates
from 4 GHz).
"""

from __future__ import annotations

import numpy as np

from magnelio import (
    AnalysisScatteringTD,
    Material,
    Mesh,
)
from magnelio.materials import DispersionModel
from magnelio.mesh import BoxFace
from magnelio.mesh.grid import GridLines
from magnelio.ports import PortSpecMultiConductor

EPS0 = 8.8541878128e-12
MU0 = 1.2566370614e-6

WIDTH_A = 10e-3
GAP_B = 5e-3
F_MAX = 10e9
DZ = 0.5e-3
L1, L2 = 10e-3, 40e-3
F_AXIS = np.linspace(2e9, 10e9, 41)

_BCS = {
    "xmin": "PMC",
    "xmax": "PMC",
    "ymin": "PEC",
    "ymax": "PEC",
    "zmin": "PEC",
    "zmax": "PEC",
}


def _line_grid(length: float) -> GridLines:
    nz = int(round(length / DZ))
    return GridLines(
        x=np.linspace(-WIDTH_A / 2, WIDTH_A / 2, 11),
        y=np.linspace(-GAP_B / 2, GAP_B / 2, 6),
        z=np.linspace(0, length, nz + 1),
    )


def _measure_gamma(material: Material) -> np.ndarray:
    s21 = {}
    for length in (L1, L2):
        mesh = Mesh.from_grid(_line_grid(length), background=material)
        analysis = AnalysisScatteringTD(
            mesh=mesh.with_boundary_conditions(_BCS),
            ports=[
                PortSpecMultiConductor(name="port1", plane=BoxFace.Z_MIN, n_modes=1),
                PortSpecMultiConductor(name="port2", plane=BoxFace.Z_MAX, n_modes=1),
            ],
            f_max=F_MAX,
            verbose=False,
        )
        result = analysis.run(f_axis=F_AXIS, excited=["port1"])
        s21[length] = result.S("port2", "port1")
    ratio = s21[L2] / s21[L1]
    return -(np.log(np.abs(ratio)) + 1j * np.unwrap(np.angle(ratio))) / (L2 - L1)


def test_from_table_line_gamma_vs_table():
    """Two-relaxation table (knees at 3 and 8 GHz, both in band) fitted
    by from_table, run on the TEM line; gamma is gated against the
    complex-interpolated TABLE — the full-chain acceptance."""
    # the table generator is only used to WRITE the table
    gen = DispersionModel.debye(
        eps_inf=2.0,
        delta_eps=(1.2, 0.8),
        tau=(1.0 / (2 * np.pi * 3e9), 1.0 / (2 * np.pi * 8e9)),
    )
    f_tab = np.linspace(1e9, 12e9, 45)
    eps_tab = gen.evaluate(2.0 * np.pi * f_tab)

    model = DispersionModel.from_table(f_tab, eps_tab)
    fitted = Material.dispersive("tab_fill", model)

    g_meas = _measure_gamma(fitted)

    # reference straight from the table: complex interpolation onto the
    # gate axis, then gamma = jw*sqrt(mu0*eps0*eps_c)
    eps_axis = np.interp(F_AXIS, f_tab, eps_tab.real) + 1j * np.interp(F_AXIS, f_tab, eps_tab.imag)
    g_ref = 1j * 2 * np.pi * F_AXIS * np.sqrt(MU0 * EPS0 * eps_axis)

    hi = F_AXIS >= 4e9  # Fabry-Perot residual below (module docstring)
    alpha_err = np.abs(g_meas.real[hi] / g_ref.real[hi] - 1).max()
    beta_err = np.abs(g_meas.imag / g_ref.imag - 1).max()
    assert alpha_err < 0.03, f"alpha rel error {alpha_err:.4f}"
    assert beta_err < 0.01, f"beta rel error {beta_err:.4f}"
