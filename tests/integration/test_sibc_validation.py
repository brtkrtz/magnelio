"""WP-D6 validation gates of the SIBC roadmap (SIBC_PLAN.md).

The one mandatory bar (DERIVATION.md §6) is **Gate B**: a finite-sigma
sequence must converge to the PEC solution at the analytic O(1/sqrt(sigma))
rate, end-to-end through the analysis run (Gate A — the structural
bit-identity no-op — is gated in ``tests/unit/test_sibc_operator.py``).
Everything else here records the SIBC_PLAN measurements with the
DERIVATION.md §7 a-priori targets as windows around the measured values
(developer decision at planning: measurements, not pre-code acceptance
bars).

Measured (session 116, numpy backend):

* Gate B (parallel plate, sigma = 5.8e3/5.8e5/5.8e7 vs the lossless
  perturbative run): slope of max|S21 - S21_pec| = -0.4975, slope of
  mean(-ln|S21|) = -0.4991 against the analytic -1/2; copper endpoint
  max|S21 - S21_pec| = 5.9e-4; the lossless baseline itself sits at
  ||S21| - 1| = 7e-8 — three decades of clean sigma^(-1/2) approach.
* Rect-WG TE10 (WR-90, 12x6x30 cells, sigma = 5.8e3, 8.5-12 GHz):
  -ln|S21| / (alpha_c L) = 0.944 … 0.976 against the closed-form
  alpha_c = R_s/(b eta0 sqrt(1-(fc/f)^2)) (1 + 2b/a (fc/f)^2) — the
  O(h) H-sample-position class of §7, largest toward cut-off where the
  analytic-vs-discrete cut-off deviation is amplified.  Lossy-wall
  port floor |S11| = -43…-48 dB = the recorded O(Z_s/eta0) DTBC
  limitation, far below physical reflections.
* Round coax (conformal, D_i = 2 mm, D_a = 5 mm, vacuum, sigma =
  5.8e3, 3-9 GHz, outer conductor as an explicit padded PEC shell —
  the historical bbox-tangent fixture silently lost ~16 % of the
  outer wall area to unregistered sub-cell slivers, DD-098
  addendum), WITH the DD-098 curvature pullback (default):
  -ln|S21| / (alpha L) = 0.850 … 0.953 at 0.16 mm cells (~6 cells
  per inner radius; tangent fixture read 0.801 … 0.903).  The
  pullback removes exactly the DD-087 H-sample-position share
  (x1.108, analytic prediction 1.115); the remaining deficit is the
  near-wall measured-field class at this resolution (DD-098).
  Resolution caveat measured on the way: at ~2 cells per inner
  radius (the DD-053 impedance fixture, D_i = 0.41 mm) the same
  mechanism under-reads alpha by ~2x — alpha measurements need the
  curved conductor resolved.  Port floor -26 dB at 3 GHz =
  alpha/(2 beta), the lossy-line DTBC mismatch class of §7.
* Pillbox TM010 (conformal, 1 mm cells = 10 cells/radius, sigma =
  5.8e3): ring-down Q_sibc = 65.9 vs perturbative wall_loss_Q = 64.4
  (Q_sibc/Q_pert = 1.023 — agreement inside the combined §7 budgets)
  vs closed form 72.2 (Q_sibc/Q_closed = 0.913, the DD-087 position
  class; the perturbative value sits at 0.892 — the SIBC is the
  better one, as the plan expected).  Tail-fit skip-robust
  (identical rate at skip 500/1000/1500).
* Rough/smooth (plate, Huray cannonball Rz = 1 mm at sigma = 5.8e3 so
  the skin depth sweeps the roughness knee across the band):
  attenuation ratio tracks K(f) per bin, ratio/K = 0.950 … 0.969 over
  K = 2.45 → 3.58; the band-edge 2 GHz bin reads 0.88 (excitation
  band edge, excluded).  The residual few-% deficit scales with |Z_s|
  (the l_dual/2 offset error grows with K) — it does NOT cancel fully
  in the ratio on BC walls, unlike the DD-087 position bias it was
  argued for.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import j0, jn_zeros

from magnelio import AnalysisScatteringTD, Mesh
from magnelio.mesh import BoxFace
from magnelio.mesh.grid import GridLines
from magnelio.ports import PortSpecMultiConductor
from magnelio.ports._modal import ModeType, PortSpecNumerical
from magnelio.post.wall_loss import surface_resistance

EPS0 = 8.8541878128e-12
MU0 = 1.2566370614e-6
C0 = 299_792_458.0
ETA0 = float(np.sqrt(MU0 / EPS0))

# --------------------------------------------------------------------
# Parallel plate (Gate B + rough/smooth) — the WP-D5 fixture
# --------------------------------------------------------------------

W_A, GAP_B, LENGTH = 10e-3, 5e-3, 30e-3
FREQS = np.linspace(2e9, 10e9, 9)
SIGMA_LO = 5.8e3


def _plate_analysis(wall_model, sigma=None, roughness=None):
    grid = GridLines(
        x=np.linspace(0, W_A, 11),
        y=np.linspace(0, GAP_B, 6),
        z=np.linspace(0, LENGTH, 61),
    )
    return AnalysisScatteringTD(
        mesh=Mesh.from_grid(
            grid,
            boundary_conditions={
                "xmin": "PMC",
                "xmax": "PMC",
                "ymin": "PEC",
                "ymax": "PEC",
                "zmin": "PEC",
                "zmax": "PEC",
            },
        ),
        ports=[
            PortSpecMultiConductor(name="p1", plane=BoxFace.Z_MIN, n_modes=1),
            PortSpecMultiConductor(name="p2", plane=BoxFace.Z_MAX, n_modes=1),
        ],
        f_max=10e9,
        f_min=2e9,
        verbose=False,
        wall_model=wall_model,
        wall_sigma=sigma,
        wall_roughness=roughness,
    )


def _plate_s21(wall_model, sigma=None, roughness=None):
    res = _plate_analysis(wall_model, sigma, roughness).run(
        f_axis=FREQS,
        excited=["p1"],
    )
    return res.S("p2", "p1")


@pytest.fixture(scope="module")
def plate_s21_pec():
    return _plate_s21("perturbative")


@pytest.fixture(scope="module")
def plate_s21_smooth():
    return _plate_s21("sibc", sigma=SIGMA_LO)


def test_bc_wall_material_parity(plate_s21_smooth):
    """DD-099 WP-B2 gate: PECBoundary objects carrying their own wall
    material reproduce the analysis-global ``wall_sigma`` run exactly
    (same walls, same fitted Z_s, no global fallback needed)."""
    from magnelio.boundaries import PECBoundary

    grid = GridLines(
        x=np.linspace(0, W_A, 11),
        y=np.linspace(0, GAP_B, 6),
        z=np.linspace(0, LENGTH, 61),
    )
    ana = AnalysisScatteringTD(
        mesh=Mesh.from_grid(
            grid,
            boundary_conditions={
                "xmin": "PMC",
                "xmax": "PMC",
                "ymin": PECBoundary("ymin", wall_sigma=SIGMA_LO),
                "ymax": PECBoundary("ymax", wall_sigma=SIGMA_LO),
                "zmin": "PEC",
                "zmax": "PEC",
            },
        ),
        ports=[
            PortSpecMultiConductor(name="p1", plane=BoxFace.Z_MIN, n_modes=1),
            PortSpecMultiConductor(name="p2", plane=BoxFace.Z_MAX, n_modes=1),
        ],
        f_max=10e9,
        f_min=2e9,
        verbose=False,
        wall_model="sibc",
        wall_sigma=None,
    )
    res = ana.run(f_axis=FREQS, excited=["p1"])
    s21 = res.S("p2", "p1")
    assert np.allclose(s21, plate_s21_smooth, rtol=0.0, atol=1e-12), (
        "per-face PECBoundary wall material must reproduce the global-fallback run exactly"
    )


def test_gate_b_sigma_convergence_to_pec(plate_s21_pec, plate_s21_smooth):
    """The mandatory Gate B (DERIVATION.md §6b), end-to-end through the analysis:
    Z_s ~ sigma^(-1/2) uniformly on the band, so the deviation from the
    PEC run must vanish at exactly that rate, and the copper endpoint
    must land near the PEC solution."""
    sigmas = np.array([5.8e3, 5.8e5, 5.8e7])
    s21 = [plate_s21_smooth, _plate_s21("sibc", sigma=5.8e5), _plate_s21("sibc", sigma=5.8e7)]

    # the lossless reference itself is clean far below the endpoint
    assert np.max(np.abs(np.abs(plate_s21_pec) - 1.0)) < 1e-6

    dev = np.array([np.max(np.abs(s - plate_s21_pec)) for s in s21])
    nl = np.array([np.mean(-np.log(np.abs(s))) for s in s21])
    assert (np.diff(dev) < 0).all(), f"not monotone: {dev}"

    slope_dev = np.polyfit(np.log(sigmas), np.log(dev), 1)[0]
    slope_nl = np.polyfit(np.log(sigmas), np.log(nl), 1)[0]
    assert slope_dev == pytest.approx(-0.5, abs=0.03), (
        f"deviation slope {slope_dev:.4f} (measured -0.4975)"
    )
    assert slope_nl == pytest.approx(-0.5, abs=0.03), (
        f"attenuation slope {slope_nl:.4f} (measured -0.4991)"
    )
    assert dev[-1] < 1e-3, f"copper endpoint {dev[-1]:.2e} vs PEC (measured 5.9e-4)"


def test_rough_smooth_alpha_ratio_tracks_k(plate_s21_smooth):
    """DD-088 argument in the field solution: the rough/smooth
    attenuation ratio reproduces K(f) per DFT bin.  Rz is scaled so the
    skin depth sweeps the roughness knee INSIDE the band (K = 2.45 →
    3.58) — a constant factor cannot pass the tracking check.  The
    2 GHz excitation-band-edge bin is excluded (measured 0.88 there,
    0.950…0.969 inside)."""
    from magnelio.materials.roughness import Huray

    rough = Huray.cannonball(rz=1.0e-3)
    s21_rough = _plate_s21("sibc", sigma=SIGMA_LO, roughness=rough)

    inner = slice(1, None)
    nl_s = -np.log(np.abs(plate_s21_smooth))[inner]
    nl_r = -np.log(np.abs(s21_rough))[inner]
    k = rough.factor(FREQS, SIGMA_LO)[inner]
    assert k[0] > 2.3 and k[-1] > 3.4  # the knee is inside the band

    ratio = nl_r / nl_s
    assert (ratio / k > 0.93).all() and (ratio / k < 1.0).all(), (
        f"ratio/K = {ratio / k} (measured 0.950…0.969)"
    )
    # per-bin tracking: the measured ratio rises with f like K does
    track = (ratio[-1] / ratio[0]) / (k[-1] / k[0])
    assert track == pytest.approx(1.0, abs=0.05), f"frequency tracking {track:.4f} (measured 1.02)"


# --------------------------------------------------------------------
# Rectangular-WG TE10 attenuation
# --------------------------------------------------------------------

WG_A, WG_B, WG_L = 22.86e-3, 10.16e-3, 30e-3
WG_FREQS = np.linspace(8.5e9, 12.0e9, 8)


def test_rectwg_te10_alpha_vs_analytic():
    """One pulsed run: |S21| of a length-L WR-90 line under SIBC walls
    vs the closed-form TE10 conductor attenuation (windows in the
    module docstring)."""
    sigma = 5.8e3
    grid = GridLines(
        x=np.linspace(0, WG_A, 13),
        y=np.linspace(0, WG_B, 7),
        z=np.linspace(0, WG_L, 31),
    )
    ana = AnalysisScatteringTD(
        mesh=Mesh.from_grid(
            grid,
            boundary_conditions={
                "xmin": "PEC",
                "xmax": "PEC",
                "ymin": "PEC",
                "ymax": "PEC",
                "zmin": "PEC",
                "zmax": "PEC",
            },
        ),
        ports=[
            PortSpecNumerical(name="p1", plane=BoxFace.Z_MIN, n_modes=1, mode_type=ModeType.TE),
            PortSpecNumerical(name="p2", plane=BoxFace.Z_MAX, n_modes=1, mode_type=ModeType.TE),
        ],
        f_max=12.4e9,
        f_min=8.2e9,
        verbose=False,
        wall_model="sibc",
        wall_sigma=sigma,
    )
    res = ana.run(f_axis=WG_FREQS, excited=["p1"])
    s21 = np.abs(res.S("p2", "p1"))
    s11 = np.abs(res.S("p1", "p1"))

    fc = C0 / (2 * WG_A)
    rs = surface_resistance(WG_FREQS, sigma)
    root = np.sqrt(1.0 - (fc / WG_FREQS) ** 2)
    alpha = rs / (WG_B * ETA0 * root) * (1.0 + 2.0 * WG_B / WG_A * (fc / WG_FREQS) ** 2)
    ratio = -np.log(s21) / (alpha * WG_L)
    assert ratio.min() > 0.92 and ratio.max() < 1.01, (
        f"TE10 alpha ratio {ratio} (measured 0.944…0.976)"
    )
    # lossy-wall port floor: the DERIVATION.md §7 O(Z_s/eta0) class
    assert (20 * np.log10(s11) < -40).all()


# --------------------------------------------------------------------
# Conformal round coax attenuation
# --------------------------------------------------------------------

CX_DI, CX_DA, CX_L = 2.0e-3, 5.0e-3, 10e-3
CX_FREQS = np.linspace(3e9, 9e9, 7)


def test_conformal_coax_alpha():
    """Curved-wall SIBC booking end-to-end: vacuum round coax vs the
    closed form alpha = R_s/(2 eta0 ln(b/a)) (1/a + 1/b).

    The outer conductor is an explicit PEC shell (to 3.25 mm) so the
    wall at r = b stays clear of the domain bbox (the historical
    bbox-tangent registration void, DD-098 addendum — fixed by the
    DD-099 boundary-layer seed, see test_tangent_coax_alpha).

    Measured 0.838 … 0.929 at 0.16 mm with the DD-098 curvature
    pullback AND the DD-099 port continuation masking (window
    unchanged 0.83 … 0.97).  The pre-DD-099 baseline 0.850 … 0.953
    included ~1 % phantom port-plane cross-section booking (the
    conformal cell path booked the conductor cross-sections at the
    port planes as walls); the remaining deficit is the near-wall
    measured-field class (O(h), DD-097 C-control) — see the internal
    dossier investigations/boundary_wall/MEASUREMENTS.md."""
    sigma = 5.8e3
    from magnelio import Material, MeshControl
    from magnelio.geo import Cylinder, Difference, GeometryModel

    pec, vac = Material.pec(), Material.air()
    shell_out = Cylinder(origin=(0, 0, 0), radius=3.25e-3, height=CX_L, axis="z", material=pec)
    out_cyl = Cylinder(origin=(0, 0, 0), radius=CX_DA / 2, height=CX_L, axis="z", material=vac)
    in_cyl = Cylinder(origin=(0, 0, 0), radius=CX_DI / 2, height=CX_L, axis="z", material=pec)
    model = GeometryModel(background=pec)
    model.add(Difference(shell_out, out_cyl))
    model.add(Difference(out_cyl, in_cyl))
    model.add(in_cyl)
    control = MeshControl(
        min_nodes_per_wavelength=8,
        min_cells_per_feature=3,
        growth_factor=1.4,
        conformal=True,
        max_cell_size=0.16e-3,
        min_cell_size=20e-6,
        min_feature_gap=20e-6,
    )
    mesh = Mesh.from_geometry(model, control, f_max=10e9)
    ana = AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions({}),
        ports=[
            PortSpecMultiConductor(name="p1", plane=BoxFace.Z_MIN, n_modes=1),
            PortSpecMultiConductor(name="p2", plane=BoxFace.Z_MAX, n_modes=1),
        ],
        f_max=10e9,
        f_min=3e9,
        verbose=False,
        wall_model="sibc",
        wall_sigma=sigma,
    )
    res = ana.run(f_axis=CX_FREQS, excited=["p1"])
    s21 = np.abs(res.S("p2", "p1"))

    a, b = CX_DI / 2, CX_DA / 2
    rs = surface_resistance(CX_FREQS, sigma)
    alpha = rs / (2 * ETA0 * np.log(b / a)) * (1 / a + 1 / b)
    ratio = -np.log(s21) / (alpha * CX_L)
    assert ratio.min() > 0.83 and ratio.max() < 0.97, (
        f"coax alpha ratio {ratio} (measured 0.838…0.929 at this "
        f"resolution with the DD-098 pullback and the DD-099 port "
        f"continuation masking — the remaining deficit is the "
        f"near-wall measured-field class, DD-098 addendum)"
    )


def test_tangent_coax_alpha():
    """DD-099 gate: the bbox-tangent coax — outer conductor tangent to
    the domain box, the geometry that historically lost four ~20 deg
    zones of the outer wall to the candidate-gate registration void —
    reads the SAME alpha as the padded fixture, with no padding and no
    declared BCs.  The boundary-layer seed registers the flat boundary
    wall; port planes get continuation masking.

    Measured 0.838 … 0.929 at 0.16 mm — identical to
    test_conformal_coax_alpha (fixture equivalence is the point);
    same window."""
    sigma = 5.8e3
    from magnelio import Material, MeshControl
    from magnelio.geo import Cylinder, Difference, GeometryModel

    pec, vac = Material.pec(), Material.air()
    out_cyl = Cylinder(origin=(0, 0, 0), radius=CX_DA / 2, height=CX_L, axis="z", material=vac)
    in_cyl = Cylinder(origin=(0, 0, 0), radius=CX_DI / 2, height=CX_L, axis="z", material=pec)
    model = GeometryModel(background=pec)
    model.add(Difference(out_cyl, in_cyl))
    model.add(in_cyl)
    control = MeshControl(
        min_nodes_per_wavelength=8,
        min_cells_per_feature=3,
        growth_factor=1.4,
        conformal=True,
        max_cell_size=0.16e-3,
        min_cell_size=20e-6,
        min_feature_gap=20e-6,
    )
    mesh = Mesh.from_geometry(model, control, f_max=10e9)
    ana = AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions({}),
        ports=[
            PortSpecMultiConductor(name="p1", plane=BoxFace.Z_MIN, n_modes=1),
            PortSpecMultiConductor(name="p2", plane=BoxFace.Z_MAX, n_modes=1),
        ],
        f_max=10e9,
        f_min=3e9,
        verbose=False,
        wall_model="sibc",
        wall_sigma=sigma,
    )
    res = ana.run(f_axis=CX_FREQS, excited=["p1"])
    s21 = np.abs(res.S("p2", "p1"))

    a, b = CX_DI / 2, CX_DA / 2
    rs = surface_resistance(CX_FREQS, sigma)
    alpha = rs / (2 * ETA0 * np.log(b / a)) * (1 / a + 1 / b)
    ratio = -np.log(s21) / (alpha * CX_L)
    assert ratio.min() > 0.83 and ratio.max() < 0.97, (
        f"tangent coax alpha ratio {ratio} (measured 0.838…0.929, "
        f"identical to the padded fixture — DD-099 candidate-gate "
        f"seed + continuation masking)"
    )


# --------------------------------------------------------------------
# Pillbox TM010: SIBC ring-down Q vs perturbative wall_loss_Q
# --------------------------------------------------------------------

PB_R, PB_H, PB_BBOX = 10e-3, 8e-3, 26e-3
PB_SIGMA = 5.8e3


def _pillbox_mesh(h=1e-3):
    from magnelio.geo import Brick, Cylinder, Difference, GeometryModel
    from magnelio.materials.material import Material
    from magnelio.mesh.mesher import MeshControl

    pec, vac = Material.pec(), Material.air()
    pad = 3 * h
    brick = Brick(
        origin=(-PB_BBOX / 2, -PB_BBOX / 2, -pad),
        size=(PB_BBOX, PB_BBOX, PB_H + 2 * pad),
        material=pec,
    )
    hole = Cylinder(origin=(0, 0, 0), radius=PB_R, height=PB_H, axis="z", material=vac)
    model = GeometryModel()
    model.add(Difference(brick, hole))
    model.add(hole)
    n_t = int(round(PB_BBOX / h)) + 1
    n_z = int(round((PB_H + 2 * pad) / h)) + 1
    ctrl = MeshControl(
        min_nodes_per_wavelength=8,
        min_cells_per_feature=0,
        growth_factor=1.5,
        max_cell_size=2 * h,
        conformal=True,
        forced_planes={
            "x": np.linspace(-PB_BBOX / 2, PB_BBOX / 2, n_t).tolist(),
            "y": np.linspace(-PB_BBOX / 2, PB_BBOX / 2, n_t).tolist(),
            "z": np.linspace(-pad, PB_H + pad, n_z).tolist(),
        },
    )
    return Mesh.from_geometry(model, ctrl, f_max=10e9)


class _EnergySampler:
    def __init__(self, solver):
        self._s = solver
        self.energy: list[float] = []

    def record(self, fields, n, t, dt):
        e, h = fields.e_flat, fields.h_flat
        self.energy.append(
            0.5 * (float((self._s._M_eps_diag * e) @ e) + float((self._s._M_mu_diag * h) @ h))
        )


def test_pillbox_tm010_q_sibc_vs_perturbative():
    """High-Q consistency (SIBC_PLAN WP-D6): the TM010 ring-down Q of
    the conformal pillbox under SIBC walls agrees with the DD-082/087
    perturbative ``wall_loss_Q`` inside the combined §7 budgets, and
    sits CLOSER to the closed form (no post-hoc sampling mismatch —
    the loss the solver inserts is the loss the decay shows).

    Measured (1 mm cells = 10 cells/radius): Q_pert = 64.4 (-10.8 % vs
    closed form — the DD-087 record at this resolution), Q_sibc = 65.9
    (-8.7 %), Q_sibc/Q_pert = 1.023, tail-fit identical for skip
    500/1000/1500."""
    from magnelio.analysis.eigenmode import AnalysisEigenmode
    from magnelio.boundaries.pec import PECBoundary
    from magnelio.materials.surface_impedance import fit_wall_impedances
    from magnelio.mesh._surfaces import (
        enumerate_sibc_surfaces,
        resolve_wall_conductors,
    )
    from magnelio.post.wall_loss import wall_loss_Q
    from magnelio.solver._sibc import SIBCSpec
    from magnelio.solver.fit_td import FITTimeDomainSolver
    from magnelio.solver.stability import courant_dt

    x01 = float(jn_zeros(0, 1)[0])
    mesh = _pillbox_mesh()

    # perturbative reference on the same mesh (ARPACK needs n_modes=4
    # on this fixture, see test_conformal_wall_area.py)
    eig = AnalysisEigenmode(mesh=mesh, n_modes=4, verbose=False).run()
    f0 = float(eig.frequencies[0])
    q_pert = wall_loss_Q(eig, sigma=PB_SIGMA).Q
    rs0 = float(surface_resistance(f0, PB_SIGMA))
    q_closed = x01 * ETA0 / (2 * rs0 * (1 + PB_R / PB_H))

    # SIBC ring-down through the solver (component level: AnalysisEigenmode keeps
    # the perturbative route by plan non-goal, so the TD march is the
    # SIBC instrument)
    surfs = enumerate_sibc_surfaces(mesh)
    resolved = resolve_wall_conductors(mesh, surfs, sigma=PB_SIGMA)
    fits = fit_wall_impedances(resolved, 1e9, 1e11)
    spec = SIBCSpec(surfaces=tuple(surfs), fits=fits)

    dt = courant_dt(mesh.grid)
    faces = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
    s = FITTimeDomainSolver(
        mesh=mesh,
        boundary_conditions={f: PECBoundary(f) for f in faces},
        dt=dt,
        total_time_steps=3000,
        verbose=False,
        sibc=spec,
    )
    s.setup()

    # TM010 seed: Ez = J0(x01 r/R) inside the cavity, frozen edges
    # cleared (a seeded frozen edge would hold constant energy and
    # floor the rate fit)
    g = mesh.grid
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    zc = 0.5 * (g.z[:-1] + g.z[1:])
    rr = np.sqrt(g.x[:, None] ** 2 + g.y[None, :] ** 2)
    prof = np.where(rr < PB_R - 1e-3, j0(x01 * rr / PB_R), 0.0)
    ez = prof[:, :, None] * ((zc > 0.0) & (zc < PB_H))[None, None, :]
    s._fields.e_flat[n_Ex + n_Ey :] = ez.ravel()
    s._fields.e_flat[s._beta_E == 0.0] = 0.0

    sampler = _EnergySampler(s)
    s.monitors.append(sampler)
    s.run()

    energy = np.asarray(sampler.energy)
    assert np.isfinite(energy).all()
    tail = energy[800:]
    n = np.arange(tail.size, dtype=float)
    lam = -np.polyfit(n * dt, np.log(tail), 1)[0]
    q_sibc = 2 * np.pi * f0 / lam

    assert q_sibc / q_pert == pytest.approx(1.023, abs=0.05), (
        f"Q_sibc = {q_sibc:.1f} vs perturbative {q_pert:.1f} (measured ratio 1.023)"
    )
    assert 0.88 < q_sibc / q_closed < 1.0, (
        f"Q_sibc/Q_closed = {q_sibc / q_closed:.3f} (measured 0.913, "
        f"the DD-087 O(h) position class at 10 cells/radius)"
    )
