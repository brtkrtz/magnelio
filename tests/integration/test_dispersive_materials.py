"""Physics gates for pole-residue dispersive materials (DD-083/DD-084,
Cluster C of MATERIAL_MODELS_PLAN.md).

Line fixtures are the DD-081 parallel-plate TEM pattern: the two-length
ratio S21(L2)/S21(L1) = exp(-gamma*dL) cancels the port mismatch (modal
ports are built lossless at eps_inf, so an in-band eps'(omega) above
eps_inf reflects; measured |S11| ~ -16 dB on the Debye line vs ~ -25 dB
on the DD-081 sigma line).  The multiple-reflection residual
~ r^2 exp(-2*alpha*L1) therefore dominates where alpha is small — the
low-frequency band edge — exactly as in DD-081, only r^2 is ~5x larger;
alpha gates start at 4 GHz for the Debye line for that reason (the
band-edge deviation was measured at -9.9 % with the +-oscillating
signature of the Fabry-Perot ripple, not a systematic ADE error).

Slab fixtures compare against the exact vacuum/slab/vacuum transfer
matrix with the model's own eps(omega).  The staircase interface
convention (DD-081: the clamped one-sided sampling shifts the discrete
interface dz/2 in FRONT of the material plane) moves the slab but keeps
its thickness — transmission needs no de-embedding, reflection does.
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
C0 = 299_792_458.0

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


def _gamma_of_eps(f: np.ndarray, eps_c: np.ndarray) -> np.ndarray:
    return 1j * 2 * np.pi * f * np.sqrt(MU0 * EPS0 * eps_c)


def _slab_s21(f: np.ndarray, eps_c: np.ndarray, d: float, L: float) -> np.ndarray:
    """Vacuum line of length L with a slab (eps_c, thickness d): exact
    plane-wave transfer through the slab plus the vacuum path phase."""
    w = 2 * np.pi * f
    n = np.sqrt(eps_c)
    k0, k1 = w / C0, n * w / C0
    r01 = (1 - n) / (1 + n)
    t = (1 - r01**2) * np.exp(-1j * k1 * d) / (1 - r01**2 * np.exp(-2j * k1 * d))
    return t * np.exp(-1j * k0 * (L - d))


# ═════════════════════════════════════════════════════════════════════
# Debye
# ═════════════════════════════════════════════════════════════════════


def test_debye_line_gamma():
    """Debye-filled line vs the exact gamma(omega) with eps(omega) —
    the relaxation sits mid-band (5 GHz), so both the dispersive beta
    and the relaxation-loss alpha are exercised across their knee."""
    model = DispersionModel.debye(2.0, 1.5, tau=1.0 / (2 * np.pi * 5e9))
    g_meas = _measure_gamma(Material.dispersive("debye_fill", model))
    g_ref = _gamma_of_eps(F_AXIS, model.evaluate(2 * np.pi * F_AXIS))

    hi = F_AXIS >= 4e9  # below: Fabry-Perot residual, see module docstring
    alpha_err = np.abs(g_meas.real[hi] / g_ref.real[hi] - 1).max()
    beta_err = np.abs(g_meas.imag / g_ref.imag - 1).max()
    assert alpha_err < 0.03, f"alpha rel error {alpha_err:.4f} (measured 0.018)"
    assert beta_err < 0.01, f"beta rel error {beta_err:.4f} (measured 0.0046)"


def test_debye_sigma_half_space_fresnel():
    """Normal-incidence reflection off a Debye + static-sigma half-space
    vs Fresnel with eps_c = eps_debye(omega) - j*sigma/(omega*eps0).

    Gates the ADE and the semi-implicit sigma channel COMBINED in one
    material (they share the E-update denominator, DD-084).  The sigma
    term kills the round trip to the PEC back wall (DD-081 fixture); the
    reference plane is de-embedded at d - dz/2 (staircase interface
    convention, DD-081)."""
    d_vac, d_lossy, sigma = 5e-3, 25e-3, 2.0
    model = DispersionModel.debye(2.0, 3.0, tau=1.0 / (2 * np.pi * 5e9))
    mat = Material.dispersive("debye_sigma", model, sigma=sigma)
    mesh = Mesh.from_grid(
        _line_grid(d_vac + d_lossy),
        regions=[(mat, (-1.0, -1.0, d_vac, 1.0, 1.0, 1.0))],
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
    eps_c = model.evaluate(w) - 1j * sigma / (w * EPS0)
    r_ref = (1 - np.sqrt(eps_c)) / (1 + np.sqrt(eps_c))
    r_meas = S11 * np.exp(2j * w / C0 * (d_vac - DZ / 2))

    cplx_err = np.abs(r_meas / r_ref - 1).max()
    assert cplx_err < 0.02, f"complex r rel error {cplx_err:.4f} (measured 0.0068)"

    # The Debye pole set moves r measurably beyond the sigma-only value
    # (measured 0.152): a sigma-only reference must NOT pass the gate.
    eps_sig_only = 2.0 - 1j * sigma / (w * EPS0)
    r_sig = (1 - np.sqrt(eps_sig_only)) / (1 + np.sqrt(eps_sig_only))
    assert np.abs(r_meas / r_sig - 1).max() > 0.08


# ═════════════════════════════════════════════════════════════════════
# Lorentz / Drude slabs
# ═════════════════════════════════════════════════════════════════════


def test_lorentz_slab_transmission():
    """Lorentz slab: in-band resonance absorption dip vs the exact
    vacuum/slab/vacuum transfer matrix (complex S21 over the band)."""
    model = DispersionModel.lorentz(
        2.0,
        0.5,
        omega0=2 * np.pi * 6e9,
        delta=2 * np.pi * 0.3e9,
    )
    d, L = 5e-3, 30e-3
    mesh = Mesh.from_grid(
        _line_grid(L),
        regions=[(Material.dispersive("lorentz", model), (-1.0, -1.0, 12.5e-3, 1.0, 1.0, 17.5e-3))],
    )
    s21, _ = _run_two_port(mesh)
    s21_ref = _slab_s21(F_AXIS, model.evaluate(2 * np.pi * F_AXIS), d, L)

    # The resonance dip must be present and land on frequency.
    i_meas, i_ref = np.argmin(np.abs(s21)), np.argmin(np.abs(s21_ref))
    assert abs(i_meas - i_ref) <= 1, (
        f"dip at {F_AXIS[i_meas] / 1e9:.1f} GHz, reference {F_AXIS[i_ref] / 1e9:.1f} GHz"
    )
    assert np.abs(s21_ref).min() < 0.6  # a real dip, not a ripple
    err = np.abs(s21 - s21_ref).max()
    assert err < 0.03, f"complex S21 abs error {err:.4f}"


def test_drude_slab_cutoff():
    """Drude slab: evanescent below the plasma frequency, transparent
    above — |S21| vs the exact slab transfer on both sides of f_p."""
    f_p = 6e9
    model = DispersionModel.drude(
        1.0,
        omega_p=2 * np.pi * f_p,
        gamma=2 * np.pi * 0.3e9,
    )
    d, L = 10e-3, 30e-3
    mesh = Mesh.from_grid(
        _line_grid(L),
        regions=[(Material.dispersive("plasma", model), (-1.0, -1.0, 10e-3, 1.0, 1.0, 20e-3))],
    )
    s21, _ = _run_two_port(mesh)
    s21_ref = _slab_s21(F_AXIS, model.evaluate(2 * np.pi * F_AXIS), d, L)

    # Cutoff switch: evanescent tunneling at the low edge (kappa*d ~ 1.2
    # still passes 0.37 — the *reference* says so too), near-transparent
    # at the top.
    assert np.abs(s21[0]) < 0.45 and np.abs(s21[-1]) > 0.9
    # The exact-physics gate: planar grid-aligned interfaces make the
    # discrete slab nearly exact (measured 0.005 dB / 0.0012 complex).
    err_db = np.abs(20 * np.log10(np.abs(s21)) - 20 * np.log10(np.abs(s21_ref)))
    assert err_db.max() < 0.05, f"|S21| error {err_db.max():.3f} dB"
    assert np.abs(s21 - s21_ref).max() < 0.01


# ═════════════════════════════════════════════════════════════════════
# Djordjevic–Sarkar — the causal constant-tan-delta substrate
# ═════════════════════════════════════════════════════════════════════


def test_djordjevic_sarkar_line_causal():
    """DS-filled line vs exact gamma with eps_DS(omega), and against the
    narrowband sigma_eff shortcut (constant eps', eps'' ~ 1/f): the
    measurement must match the causal model and REJECT the shortcut —
    tan delta of the shortcut falls as 1/f, so its alpha is wrong by
    tens of percent at the band edges, and its constant eps' misses the
    causal phase slope."""
    model = DispersionModel.djordjevic_sarkar(
        4.3,
        0.02,
        f_ref=5e9,
        f1=1e6,
        f2=1e12,
    )
    g_meas = _measure_gamma(Material.dispersive("substrate", model))

    w = 2 * np.pi * F_AXIS
    g_ds = _gamma_of_eps(F_AXIS, model.evaluate(w))
    sigma_eff = 2 * np.pi * 5e9 * EPS0 * 4.3 * 0.02
    g_sc = _gamma_of_eps(F_AXIS, 4.3 - 1j * sigma_eff / (w * EPS0))

    alpha_err = np.abs(g_meas.real / g_ds.real - 1).max()
    beta_err = np.abs(g_meas.imag / g_ds.imag - 1).max()
    assert alpha_err < 0.03, f"alpha vs DS {alpha_err:.4f}"
    assert beta_err < 0.005, f"beta vs DS {beta_err:.4f}"

    # tan delta measured flat: the 1/f shortcut misses alpha by ~2x at
    # the band edges — the measurement must resolve that difference.
    alpha_sc_err = np.abs(g_meas.real / g_sc.real - 1).max()
    assert alpha_sc_err > 0.30, (
        f"sigma_eff shortcut alpha off by only {alpha_sc_err:.3f} — "
        f"cannot distinguish causal DS from the shortcut"
    )
    # Causal eps'(omega) slope: the shortcut's constant-eps' beta is
    # distinguishably worse than the DS beta match.
    beta_sc_err = np.abs(g_meas.imag / g_sc.imag - 1).max()
    assert beta_sc_err > 3.0 * beta_err


# ═════════════════════════════════════════════════════════════════════
# Disk resume across a checkpoint seam (DD-070 WP-S8 chain)
# ═════════════════════════════════════════════════════════════════════


def test_resume_dispersive_bit_exact(tmp_path):
    """A dispersive run resumed from disk is bit-identical to an
    uninterrupted run: the dispersion model survives the mesh.h5
    material round-trip (recipe rebuild), and the complex Lorentz pole
    currents plus the real Debye channel survive checkpoint.h5.

    TEM parallel plate on purpose (deterministic dense-Laplace port
    build — the WP-S8 caveat), with a Lorentz+Debye+sigma fill so all
    three state kinds cross the seam."""
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    from magnelio import MeshControl, open_project, resume
    from magnelio.geo import (
        Brick,
        GeometryModel,  # noqa: PLC0415
    )
    from magnelio.ports import PortWaveguide

    model_l = DispersionModel.lorentz(
        2.0,
        0.4,
        omega0=2 * np.pi * 8e9,
        delta=2 * np.pi * 1e9,
    )
    fill = Material.dispersive("resonant_fill", model_l, sigma=0.005)

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

    p = tmp_path / "disp"
    analysis(project=p).run(
        excited=[("port1", 0)],
        energy_stop_db=None,
        total_time_steps=n1,
        checkpoint_interval=40,
    )
    assert open_project(p).runs["port1_mode0"]["n_steps"] == n1

    proj = resume(p, excited=("port1", 0), total_time_steps=n_total, verbose=False)
    assert proj.runs["port1_mode0"]["n_steps"] == n_total
    for chan, (rv, ri) in ref_vi.items():
        gv, gi = proj.signals[("port1", 0)][chan]
        np.testing.assert_array_equal(rv, gv.values)
        np.testing.assert_array_equal(ri, gi.values)


# ═════════════════════════════════════════════════════════════════════
# eps_inf CFL
# ═════════════════════════════════════════════════════════════════════


def test_cfl_uses_eps_inf():
    """The production dt chain (courant_dt on the mesh's effective
    minima — the exact call AnalysisScatteringTD.run makes) yields the
    identical dt for a dispersive fill and for a static fill at
    eps = eps_inf: pole strength must not enter the CFL (DD-084 — the
    high-frequency wave speed is the stability-relevant one; in-band
    eps' > eps_inf only slows the wave)."""
    from magnelio.solver.stability import (  # noqa: PLC0415
        compute_min_effective_eps,
        compute_min_effective_mu,
        courant_dt,
    )

    model = DispersionModel.debye(2.0, 40.0, tau=1e-11)
    dts = {}
    for name, mat in (
        ("disp", Material.dispersive("d", model)),
        ("static", Material.from_isotropic("s", epsilon=2.0)),
    ):
        mesh = Mesh.from_grid(_line_grid(L1), background=mat)
        dts[name] = courant_dt(
            mesh.grid,
            "normal",
            min_effective_eps=compute_min_effective_eps(mesh),
            min_effective_mu=compute_min_effective_mu(mesh),
        )
    assert dts["disp"] == dts["static"]
