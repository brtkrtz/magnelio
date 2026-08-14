"""Physics gates for lossy bulk materials (DD-081, Cluster A of
MATERIAL_MODELS_PLAN.md).

All fixtures are parallel-plate TEM lines (BC plates, PMC side walls,
QTEM auto-conductor ports — the ``test_parallel_plate_sparams`` pattern)
so the measured propagation constant can be gated against the EXACT
analytic

    gamma = j*omega*sqrt(mu*eps) * sqrt((1 - j*sigma/(omega*eps))
                                        * (1 - j*sigma_m/(omega*mu)))

The two-length ratio S21(L2)/S21(L1) = exp(-gamma*(L2-L1)) cancels the
port's lossy-fill mismatch (the modal port is built lossless; measured
|S11| ~ -25 dB on the sigma line) up to a multiple-reflection residual
~ r^2*exp(-2*alpha*L1) — the dominant remaining alpha error at the
low-frequency band edge.

Measured (session 105, 11x6 cross-section, dz = 0.5 mm, 2-10 GHz,
L1/L2 = 10/40 mm):
- sigma line (eps_r=2, sigma=0.05):   alpha 1.0 %, beta 0.22 %
- sigma_m line (eps_r=2, sm=3500):    alpha 1.2 %, beta 0.22 %
- transverse-only sigma:              |S21| within [-0.006, +0.002] dB
- lossy half-space (sigma=5):         |r| 0.6 %, complex r 0.9 %
"""

from __future__ import annotations

import numpy as np

from magnelio import (
    AnalysisScatteringTD,
    Material,
    Mesh,
)
from magnelio.mesh import BoxFace
from magnelio.mesh.grid import GridLines
from magnelio.ports import PortSpecMultiConductor

EPS0 = 8.8541878128e-12
MU0 = 1.2566370614e-6
C0 = 299_792_458.0

WIDTH_A = 10e-3
GAP_B = 5e-3
F_MAX = 10e9
DZ = 0.5e-3
L1, L2 = 10e-3, 40e-3

F_AXIS = np.linspace(2e9, 10e9, 41)


def _line_grid(length: float) -> GridLines:
    nz = int(round(length / DZ))
    return GridLines(
        x=np.linspace(-WIDTH_A / 2, WIDTH_A / 2, 11),
        y=np.linspace(-GAP_B / 2, GAP_B / 2, 6),
        z=np.linspace(0, length, nz + 1),
    )


_BCS = {
    "xmin": "PMC",
    "xmax": "PMC",
    "ymin": "PEC",
    "ymax": "PEC",
    "zmin": "PEC",
    "zmax": "PEC",
}


def _run_two_port(mesh: Mesh):
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
    return result.S("port2", "port1"), result.S("port1", "port1")


def _measure_gamma(material: Material) -> np.ndarray:
    """Two-length gamma extraction on a homogeneously filled line."""
    s21 = {}
    for length in (L1, L2):
        mesh = Mesh.from_grid(_line_grid(length), background=material)
        s21[length], _ = _run_two_port(mesh)
    ratio = s21[L2] / s21[L1]
    return -(np.log(np.abs(ratio)) + 1j * np.unwrap(np.angle(ratio))) / (L2 - L1)


def _gamma_analytic(f, eps_r, mu_r, sigma, sigma_m):
    w = 2 * np.pi * f
    eps = EPS0 * eps_r
    mu = MU0 * mu_r
    return (
        1j
        * w
        * np.sqrt(mu * eps)
        * np.sqrt((1 - 1j * sigma / (w * eps)) * (1 - 1j * sigma_m / (w * mu)))
    )


def test_lossy_line_sigma_gamma():
    """Electric conductivity: measured gamma vs the exact lossy-line gamma."""
    mat = Material.from_isotropic("lossy", epsilon=2.0, sigma=0.05)
    g_meas = _measure_gamma(mat)
    g_ref = _gamma_analytic(F_AXIS, 2.0, 1.0, 0.05, 0.0)

    alpha_err = np.abs(g_meas.real / g_ref.real - 1).max()
    beta_err = np.abs(g_meas.imag / g_ref.imag - 1).max()
    assert alpha_err < 0.02, f"alpha rel error {alpha_err:.4f} (measured 0.010)"
    assert beta_err < 0.005, f"beta rel error {beta_err:.4f} (measured 0.0022)"


def test_lossy_line_sigma_m_gamma():
    """Magnetic loss sigma_m drives the dual attenuation (DD-081, A3)."""
    mat = Material("mag_lossy", epsilon=(2.0,) * 3, sigma_m=(3500.0,) * 3)
    g_meas = _measure_gamma(mat)
    g_ref = _gamma_analytic(F_AXIS, 2.0, 1.0, 0.0, 3500.0)

    alpha_err = np.abs(g_meas.real / g_ref.real - 1).max()
    beta_err = np.abs(g_meas.imag / g_ref.imag - 1).max()
    assert alpha_err < 0.025, f"alpha rel error {alpha_err:.4f} (measured 0.012)"
    assert beta_err < 0.005, f"beta rel error {beta_err:.4f} (measured 0.0022)"


def test_transverse_sigma_is_lossless():
    """Anisotropy sanity: sigma on x/z only cannot attenuate the Ey mode."""
    mat = Material("aniso", epsilon=(2.0,) * 3, sigma=(5.0, 0.0, 5.0))
    mesh = Mesh.from_grid(_line_grid(L2), background=mat)
    s21, _ = _run_two_port(mesh)
    s21_db = 20 * np.log10(np.abs(s21))
    assert np.abs(s21_db).max() < 0.02, (
        f"transverse sigma attenuates: |S21| in [{s21_db.min():.4f}, {s21_db.max():.4f}] dB"
    )


def test_lossy_half_space_fresnel():
    """Normal-incidence reflection off a lossy half-space vs Fresnel.

    Intermediate-to-good-conductor regime (sigma/(omega*eps0) = 45..9
    in band).  The 25 mm lossy region kills the round trip to the PEC
    back wall (> 85 dB), so S11 is the interface reflection alone.

    Reference plane: the staircase M_sigma samples the E-edges ON the
    interface grid plane from the lossy cell (one-sided clamp), so the
    discrete interface sits half a cell in FRONT of the material plane
    — de-embed with d_eff = d_vac - dz/2 (shift measured: -dz/2 gives
    0.9 % complex error vs 9.9 % at the nominal plane).
    """
    d_vac, d_lossy, sigma = 5e-3, 25e-3, 5.0
    grid = _line_grid(d_vac + d_lossy)
    lossy = Material.from_isotropic("half_space", epsilon=1.0, sigma=sigma)
    mesh = Mesh.from_grid(
        grid,
        regions=[(lossy, (-1.0, -1.0, d_vac, 1.0, 1.0, 1.0))],
    )
    analysis = AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions(_BCS),
        ports=[PortSpecMultiConductor(name="port1", plane=BoxFace.Z_MIN, n_modes=1)],
        f_max=F_MAX,
        verbose=False,
    )
    result = analysis.run(f_axis=F_AXIS, excited=["port1"])
    S11 = result.S("port1", "port1")

    w = 2 * np.pi * F_AXIS
    eps_c = 1.0 - 1j * sigma / (w * EPS0)
    r_ref = (1 - np.sqrt(eps_c)) / (1 + np.sqrt(eps_c))
    d_eff = d_vac - DZ / 2
    r_meas = S11 * np.exp(2j * w / C0 * d_eff)

    mag_err = np.abs(np.abs(r_meas) / np.abs(r_ref) - 1).max()
    cplx_err = np.abs(r_meas / r_ref - 1).max()
    assert mag_err < 0.012, f"|r| rel error {mag_err:.4f} (measured 0.006)"
    assert cplx_err < 0.02, f"complex r rel error {cplx_err:.4f} (measured 0.009)"


def test_lossy_metal_is_pec_in_field_solve():
    """A lossy-metal obstacle produces the BIT-IDENTICAL field solution
    to the same obstacle in PEC (DD-081, A1: is_pec classification is
    untouched; finite sigma is consumed only by loss models)."""
    obstacle_box = (-2e-3, -GAP_B / 2, 8e-3, 2e-3, 0.0, 10e-3)
    results = {}
    for name, mat in (
        ("pec", Material.pec()),
        ("lossy", Material.lossy_metal("copper", sigma=5.8e7)),
    ):
        mesh = Mesh.from_grid(_line_grid(20e-3), regions=[(mat, obstacle_box)])
        results[name] = _run_two_port(mesh)
    for a, b in zip(results["pec"], results["lossy"]):
        np.testing.assert_array_equal(a, b)
