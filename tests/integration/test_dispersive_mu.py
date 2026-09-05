"""Physics gates for mu(omega) dispersive materials (DD-089).

The magnetic mirror of test_dispersive_materials.py, on the same DD-081
parallel-plate TEM line: the two-length ratio S21(L2)/S21(L1) =
exp(-gamma*dL) cancels the port mismatch (modal ports are built lossless
at mu_inf, so an in-band mu'(omega) above mu_inf reflects — the same
Fabry-Perot residual ~ r^2*exp(-2*alpha*L1) that made the DD-084 alpha
gates start at 4 GHz, and for the same reason).

Everything is gated against the EXACT

    gamma = j*omega*sqrt(mu0*mu_c(omega) * eps0*eps_c(omega)),
    mu_c  = mu(omega) - j*sigma_m/(omega*mu0),

i.e. the DD-081 sigma_m form with mu_r promoted to the pole-residue
model — which is precisely what DD-089 claims to realise.

Measured (session 109), on par with the DD-084 eps mirror (1.8 %/0.46 %):
- mu-Debye line (mu 3->2, eps 1):      alpha 2.27 %, beta 0.54 %
- mu-Debye + sigma_m = 3500:           alpha 1.92 %, beta 0.57 %
- eps-Debye AND mu-Debye in one fill:  alpha 0.54 %, beta 0.14 %
Every counterfeit (static mu_inf; sigma_m without poles; either channel
alone) is rejected at 19-21 % beta — two orders above the gate.

PHASE-BRANCH CONSTRAINT (measured the hard way, session 109): the
two-length extraction unwraps the ratio's phase over frequency, so it
recovers beta only if beta*(L2-L1) < pi at the FIRST frequency —
np.unwrap fixes relative jumps, never the absolute branch of f[0].  An
eps-Debye AND mu-Debye fill of eps_inf = mu_inf = 2 reaches n ~ 2.9 at
2 GHz, i.e. 3.65 rad > pi, and the gate then reads beta off by exactly
2*pi/dL (measured error 1.728 — the arithmetic matches to three
digits).  The fixtures therefore keep n below ~1.7 at the band start,
the same zone the DD-084 fixtures live in, and _assert_phase_branch
guards it so a future edit cannot walk back into the wrap silently.
"""

from __future__ import annotations

import numpy as np
import pytest

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
BAND = F_AXIS >= 4e9  # the DD-084 alpha-gate band (see docstring)

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


def _gamma_exact(f, mu_c, eps_c) -> np.ndarray:
    return 1j * 2 * np.pi * f * np.sqrt(MU0 * mu_c * EPS0 * eps_c)


def _assert_phase_branch(exact: np.ndarray) -> None:
    """The two-length extraction is only valid below the first phase
    wrap — see the module docstring."""
    first = exact.imag[0] * (L2 - L1)
    assert first < 0.9 * np.pi, (
        f"fixture beta*dL = {first:.2f} rad at {F_AXIS[0] / 1e9:.1f} GHz — "
        f"the unwrap cannot recover the branch of f[0]"
    )


def _mu_debye() -> DispersionModel:
    # A magnetic relaxation with its knee inside the band: mu' falls
    # 2.86 -> 2.20 across 2-10 GHz, mu'' peaks near 1/(2*pi*tau) ~ 5 GHz.
    return DispersionModel.debye(2.0, 1.0, tau=3.2e-11, f_band=(1e9, 2e10))


def _weak_debye() -> DispersionModel:
    """A gentler relaxation (1.0 -> 1.5) for the both-dispersive fixture:
    two of these keep n below the phase branch while still swinging
    mu'/eps' by ~30 % in band."""
    return DispersionModel.debye(1.0, 0.5, tau=3.2e-11, f_band=(1e9, 2e10))


@pytest.fixture(scope="module")
def _mu_debye_gamma():
    model = _mu_debye()
    # epsilon = 1 keeps n at 1.7 in the band start — the DD-084 zone,
    # safely below the phase branch (module docstring).
    mat = Material.dispersive_mu("mu_debye", model, epsilon=1.0)
    return model, _measure_gamma(mat)


def test_mu_debye_line_vs_exact_gamma(_mu_debye_gamma):
    """A mu-Debye fill against the exact gamma with mu(omega)."""
    model, gamma = _mu_debye_gamma
    omega = 2 * np.pi * F_AXIS
    mu_c = model.evaluate(omega)
    exact = _gamma_exact(F_AXIS, mu_c, 1.0)
    _assert_phase_branch(exact)

    # The dispersion must actually be doing something in-band, or the
    # gate would pass on a static mu_inf fill.
    assert mu_c.real.max() / mu_c.real.min() > 1.3
    assert mu_c.imag.min() < -0.1 or mu_c.imag.max() > 0.1

    a_err = np.abs(gamma.real[BAND] / exact.real[BAND] - 1).max()
    b_err = np.abs(gamma.imag / exact.imag - 1).max()
    assert a_err < 0.05, f"alpha error {a_err:.3f}"
    assert b_err < 0.01, f"beta error {b_err:.3f}"


def test_static_mu_inf_counterfeit_is_rejected(_mu_debye_gamma):
    """The same line read as a dispersionless mu_inf fill: the gate must
    reject it, or it is not testing mu(omega) at all."""
    model, gamma = _mu_debye_gamma
    counterfeit = _gamma_exact(F_AXIS, model.eps_inf, 1.0)
    b_err = np.abs(gamma.imag / counterfeit.imag - 1).max()
    assert b_err > 0.05, f"static mu_inf fits the measured beta to {b_err:.3f}"


def test_mu_dispersion_and_sigma_m_coexist():
    """mu(omega) poles and the static sigma_m channel share one
    denominator — gate them combined, and check the sigma_m-only
    reference is rejected."""
    sigma_m = 3500.0
    model = _mu_debye()
    mat = Material.dispersive_mu("mix", model, epsilon=1.0, sigma_m=sigma_m)
    gamma = _measure_gamma(mat)

    omega = 2 * np.pi * F_AXIS
    mu_c = model.evaluate(omega) - 1j * sigma_m / (omega * MU0)
    exact = _gamma_exact(F_AXIS, mu_c, 1.0)
    _assert_phase_branch(exact)
    a_err = np.abs(gamma.real[BAND] / exact.real[BAND] - 1).max()
    b_err = np.abs(gamma.imag / exact.imag - 1).max()
    assert a_err < 0.05, f"alpha error {a_err:.3f}"
    assert b_err < 0.01, f"beta error {b_err:.3f}"

    # Dropping the poles (sigma_m on a static mu_inf) must not fit.
    only_sm = _gamma_exact(
        F_AXIS,
        model.eps_inf - 1j * sigma_m / (omega * MU0),
        1.0,
    )
    assert np.abs(gamma.imag / only_sm.imag - 1).max() > 0.05


def test_eps_and_mu_dispersion_together():
    """Both sides dispersive in ONE material: the full denominator
    interplay against gamma with eps(omega) AND mu(omega)."""
    # Two weak relaxations: n stays ~1.45 at 2 GHz (below the phase
    # branch) while mu' and eps' each swing ~30 % in band.
    eps_model = DispersionModel.debye(1.0, 0.5, tau=2.0e-11, f_band=(1e9, 2e10))
    mu_model = _weak_debye()
    mat = Material(
        name="both",
        epsilon=(eps_model.eps_inf,) * 3,
        mu=(mu_model.eps_inf,) * 3,
        dispersion=eps_model,
        dispersion_mu=mu_model,
    )
    gamma = _measure_gamma(mat)

    omega = 2 * np.pi * F_AXIS
    exact = _gamma_exact(
        F_AXIS,
        mu_model.evaluate(omega),
        eps_model.evaluate(omega),
    )
    _assert_phase_branch(exact)
    a_err = np.abs(gamma.real[BAND] / exact.real[BAND] - 1).max()
    b_err = np.abs(gamma.imag / exact.imag - 1).max()
    assert a_err < 0.06, f"alpha error {a_err:.3f}"
    assert b_err < 0.01, f"beta error {b_err:.3f}"

    # Each channel alone must be rejected — this is the gate that proves
    # the two ADEs are both live and independent.
    eps_only = _gamma_exact(F_AXIS, mu_model.eps_inf, eps_model.evaluate(omega))
    mu_only = _gamma_exact(F_AXIS, mu_model.evaluate(omega), eps_model.eps_inf)
    assert np.abs(gamma.imag / eps_only.imag - 1).max() > 0.05
    assert np.abs(gamma.imag / mu_only.imag - 1).max() > 0.05


def test_resume_mu_dispersive_bit_exact(tmp_path):
    """A mu-dispersive run resumed from disk is bit-identical to an
    uninterrupted one: dispersion_mu survives the mesh.h5 material
    round-trip (recipe rebuild) and the complex Lorentz pole currents on
    the H-faces survive checkpoint.h5 — with an eps-dispersive channel
    in the same material, so both ADEs cross the seam together.
    TEM parallel plate on purpose (deterministic dense-Laplace port
    build — the WP-S8 caveat)."""
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    from magnelio import MeshControl, open_project, resume
    from magnelio.geo import (
        Brick,
        GeometryModel,  # noqa: PLC0415
    )
    from magnelio.ports import PortWaveguide

    mu_model = DispersionModel.lorentz(
        1.5,
        0.4,
        omega0=2 * np.pi * 8e9,
        delta=2 * np.pi * 1e9,
    )
    fill = Material(
        name="resonant_mu_fill",
        epsilon=(1.5,) * 3,
        mu=(mu_model.eps_inf,) * 3,
        dispersion=DispersionModel.debye(1.5, 0.3, tau=2e-11),
        dispersion_mu=mu_model,
        sigma_m=(50.0,) * 3,
    )

    def analysis(project=None):
        gm = GeometryModel(boundary_conditions=_BCS)
        gm.add(
            Brick(
                origin=(-WIDTH_A / 2, -GAP_B / 2, 0.0),
                size=(WIDTH_A, GAP_B, 20e-3),
                material=fill,
            )
        )
        mesh = Mesh.from_geometry(
            gm,
            MeshControl(min_nodes_per_wavelength=8),
            f_max=F_MAX,
        )
        return AnalysisScatteringTD(
            mesh=mesh,
            ports=[
                PortWaveguide(name="port1", plane="zmin", n_modes=1),
                PortWaveguide(name="port2", plane="zmax", n_modes=1),
            ],
            f_max=F_MAX,
            verbose=False,
            project=project,
            geometry=gm,
        )

    n1, n_total = 120, 300
    ref = analysis().run(
        excited=[("port1", 0)],
        energy_stop_db=None,
        total_time_steps=n_total,
    )
    ref_vi = {
        k: (v[0].values.copy(), v[1].values.copy()) for k, v in ref.signals[("port1", 0)].items()
    }
    p = tmp_path / "disp_mu"
    analysis(project=p).run(
        excited=[("port1", 0)],
        energy_stop_db=None,
        total_time_steps=n1,
        checkpoint_interval=40,
    )
    assert open_project(p).runs["port1_mode0"].n_steps == n1
    stored = open_project(p).mesh.material_library
    assert any(m.dispersion_mu is not None for m in stored.values())
    proj = resume(p, excited=("port1", 0), total_time_steps=n_total, verbose=False)
    assert proj.runs["port1_mode0"].n_steps == n_total
    for chan, (rv, ri) in ref_vi.items():
        gv, gi = proj.signals[("port1", 0)][chan]
        np.testing.assert_array_equal(rv, gv.values)
        np.testing.assert_array_equal(ri, gi.values)
