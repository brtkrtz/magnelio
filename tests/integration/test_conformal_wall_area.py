"""DD-087 gates: conformal wall areas + pillbox TM010 wall-loss Q.

Staircase face counting over-books a curved PEC wall by exactly 4/pi
(measured 1.2732 both resolutions) — the reason the pillbox Q gate was
deferred in DD-082.  DD-087 books the wall area per cut cell from the
divergence vector ``w = sum(A_face_pec*n_out)`` with the flat
(grid-aligned) family split off via the signed PEC-coverage jump, and
repairs the two H-sampling defects the conformal path exposed (flux
route state→H conversion; Faraday-dead faces redirected to the air
side).  Internal record: ``investigations/conformal_wall_area/FINDINGS.md``.

The axis-aligned DD-082 fixtures (TE101, plate, TE10 guide) never
enter the conformal branch and stay on the staircase path — their
gates live in ``test_wall_loss_q.py`` / ``test_wall_loss_monitor.py``
and keep guarding that path.
"""

from __future__ import annotations

import numpy as np
from scipy.special import jn_zeros

from magnelio.analysis.eigenmode import AnalysisEigenmode
from magnelio.geo import Brick, Cylinder, Difference, GeometryModel
from magnelio.materials.material import Material
from magnelio.mesh._surfaces import enumerate_pec_surfaces, enumerate_sibc_surfaces
from magnelio.mesh.mesher import Mesh, MeshControl
from magnelio.post.wall_loss import wall_loss_Q

MU0 = 1.2566370614e-6
EPS0 = 8.8541878128e-12
ETA0 = float(np.sqrt(MU0 / EPS0))
SIGMA_CU = 5.8e7
X01 = float(jn_zeros(0, 1)[0])

R_CYL = 10e-3
H_CAV = 8e-3
BBOX = 26e-3


def _cavity_mesh(h: float, lids: bool, shift: float = 0.0) -> Mesh:
    """PEC brick with a cylindrical air cavity.

    ``lids=False``: the hole pierces the brick (mantle only).
    ``lids=True``:  the brick extends past the hole in z (pillbox).
    ``shift``:      displaces the cylinder centre (generic alignment).
    """
    pec, vac = Material.pec(), Material.air()
    pad = 3 * h if lids else 0.0
    brick = Brick(
        origin=(-BBOX / 2, -BBOX / 2, -pad), size=(BBOX, BBOX, H_CAV + 2 * pad), material=pec
    )
    hole = Cylinder(origin=(shift, shift, 0), radius=R_CYL, height=H_CAV, axis="z", material=vac)
    model = GeometryModel()
    model.add(Difference(brick, hole))
    model.add(hole)
    n_t = int(round(BBOX / h)) + 1
    n_z = int(round((H_CAV + 2 * pad) / h)) + 1
    ctrl = MeshControl(
        min_nodes_per_wavelength=8,
        min_cells_per_feature=0,
        growth_factor=1.5,
        max_cell_size=2 * h,
        conformal=True,
        forced_planes={
            "x": np.linspace(-BBOX / 2, BBOX / 2, n_t).tolist(),
            "y": np.linspace(-BBOX / 2, BBOX / 2, n_t).tolist(),
            "z": np.linspace(-pad, H_CAV + pad, n_z).tolist(),
        },
    )
    return Mesh.from_geometry(model, ctrl, f_max=10e9)


def test_cylinder_side_area_vs_analytic():
    """Isolated-area gate: through-cylinder mantle vs 2*pi*R*L to
    < 0.5 % — where the staircase counts 4/pi = +27.3 %."""
    mesh = _cavity_mesh(1e-3, lids=False)
    surfs = enumerate_pec_surfaces(mesh)
    assert len(surfs) == 1
    a_exact = 2 * np.pi * R_CYL * H_CAV
    assert abs(surfs[0].area / a_exact - 1) < 5e-3, (
        f"mantle area {surfs[0].area * 1e6:.2f} mm² vs "
        f"analytic {a_exact * 1e6:.2f} (staircase would give 4/pi)"
    )


def test_pillbox_total_area_corner_cells():
    """Corner cells (lid + mantle in ONE cell): the plain divergence
    norm books |a+b| < |a|+|b| (measured 0.9546); the signed-jump
    split recovers the total to < 0.2 %."""
    mesh = _cavity_mesh(1e-3, lids=True)
    surfs = enumerate_pec_surfaces(mesh)
    assert len(surfs) == 1
    a_exact = 2 * np.pi * R_CYL * H_CAV + 2 * np.pi * R_CYL**2
    assert abs(surfs[0].area / a_exact - 1) < 2e-3, (
        f"pillbox area {surfs[0].area * 1e6:.2f} mm² vs "
        f"analytic {a_exact * 1e6:.2f} (corner-cell booking)"
    )


def test_sibc_topology_reuses_conformal_booking():
    """WP-D3 gate: on a conformal scene the SIBC update topology is the
    DD-087 cell booking VERBATIM (DERIVATION.md §4) — same rows, same
    weights, same area; ``g`` is exactly ``weight * inv_l_dual**2``."""
    mesh = _cavity_mesh(1e-3, lids=True)
    ws = enumerate_pec_surfaces(mesh)[0]
    ss = enumerate_sibc_surfaces(mesh)[0]
    assert ss.tag == ws.tag
    assert ss.area_total == ws.area_total
    key_w = np.lexsort((ws.flat_idx, ws.comp))
    key_s = np.lexsort((ss.flat_idx, ss.comp))
    np.testing.assert_array_equal(ss.comp[key_s], ws.comp[key_w])
    np.testing.assert_array_equal(ss.flat_idx[key_s], ws.flat_idx[key_w])
    np.testing.assert_array_equal(ss.weight[key_s], ws.weight[key_w])
    np.testing.assert_array_equal(ss.inv_l_dual[key_s], ws.inv_l_dual[key_w])
    np.testing.assert_allclose(
        ss.g,
        ss.weight * ss.inv_l_dual**2,
        rtol=0.0,
        atol=0.0,
    )


def _pillbox_q_error(h: float, shift: float = 0.0) -> float:
    """Relative TM010 wall-loss Q error vs the closed form.

    ``shift`` displaces the cylinder centre by (shift, shift) in
    metres — sub-cell shifts probe generic grid/geometry alignment
    (the centred fixture is anomalously benign: 10² = 6² + 8² puts
    lattice points exactly ON the circle)."""
    mesh = _cavity_mesh(h, lids=True, shift=shift)
    # ARPACK returns no modes on this fixture at n_modes=1
    res = AnalysisEigenmode(mesh=mesh, n_modes=4, verbose=False).run()
    f0 = float(res.frequencies[0])
    wl = wall_loss_Q(res, sigma=SIGMA_CU, mu=1.0)
    rs = float(np.sqrt(np.pi * f0 * MU0 / SIGMA_CU))
    q_ref = X01 * ETA0 / (2.0 * rs * (1.0 + R_CYL / H_CAV))
    return wl.Q / q_ref - 1.0


def test_pillbox_tm010_q_converges():
    """The deferred DD-082 gate: pillbox TM010 Q vs the closed form
    ``Q = x01*eta0/(2*Rs*(1+R/H))``.

    The estimator samples H_tan on the nearest UNCUT faces along the
    inward wall normal (cut-face states are not clean grid integrals —
    a resolution-independent power over-read; Faraday-dead faces read
    0).  The DD-098 curvature pullback removes the O(h) inward
    sample-position bias that dominated the raw booking (−10.8 % at
    1 mm → −8.0 % at 0.5 mm uncorrected).  What remains is a small
    signed mix of booking coverage, near-wall field error and the
    second-order factor residue: measured −2.5 % (1 mm), −3.7 %
    (0.5 mm), −2.9 % (0.5 mm generic phase) — inside a ±5 % envelope
    but no longer monotone between two resolutions (the dominant
    O(h) term is gone).  Gate: the post-factor envelope on BOTH
    resolutions plus phase robustness."""
    err_coarse = _pillbox_q_error(1e-3)
    err_fine = _pillbox_q_error(0.5e-3)
    err_shift = _pillbox_q_error(0.5e-3, shift=0.25e-3)
    assert -0.06 < err_fine < 0.02, (
        f"pillbox TM010 Q error at 0.5 mm: {err_fine * 100:+.2f} % "
        f"(measured −3.7 % with the DD-098 pullback; −8.0 % without; "
        f"staircase areas alone sit ~21 % low)"
    )
    assert -0.06 < err_coarse < 0.02, (
        f"pillbox TM010 Q error at 1 mm: {err_coarse * 100:+.2f} % "
        f"(measured −2.5 % with the DD-098 pullback; −10.8 % without)"
    )
    assert abs(err_shift - err_fine) < 0.04, (
        f"phase sensitivity: centred {err_fine * 100:+.2f} % vs "
        f"shifted {err_shift * 100:+.2f} % (the pre-repair estimator "
        f"scattered +0.5 % → −18 % across phases)"
    )
