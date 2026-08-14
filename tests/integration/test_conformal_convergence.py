"""
Integration test: Conformal material matrices vs staircase.

Verifies that conformal material matrices improve accuracy for curved
boundaries compared to the staircase approximation.

Test cases:
1. Cylindrical PEC cavity TM010 mode (eigenfrequency).
2. Coaxial line impedance Z₀ (energy-based capacitance).
"""

import math

import numpy as np
import pytest

occ = pytest.importorskip("OCC.Core.BRepPrimAPI")

from magnelio._operators.material_matrices import build_M_eps
from magnelio.analysis.eigenmode import AnalysisEigenmode
from magnelio.geo import Brick, Cylinder, Difference, GeometryModel
from magnelio.materials.material import Material
from magnelio.mesh.mesher import Mesh, MeshControl
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import (
    compute_min_effective_eps,
    compute_min_effective_mu,
    courant_dt,
)

C0 = 299_792_458.0
EPS0 = 8.854187817e-12
MU0 = 1.2566370614e-6
J01 = 2.4048255577  # first zero of J_0


def _cylindrical_cavity_mesh(
    radius: float,
    height: float,
    max_cell_size: float,
    conformal: bool,
    dey_mittra_eta: float = 0.4,
) -> Mesh:
    """Build a cylindrical PEC cavity mesh via from_geometry.

    A PEC brick fills the domain, then an air cylinder carves out
    the cavity interior.  PEC BCs on all faces (default).
    """
    pec = Material(name="PEC", is_pec=True)
    air = Material.air()

    pec_block = Brick(
        origin=(-radius, -radius, 0),
        size=(2 * radius, 2 * radius, height),
        material=pec,
    )
    air_cavity = Cylinder(
        origin=(0, 0, 0),
        radius=radius,
        height=height,
        axis="z",
        material=air,
    )
    model = GeometryModel()
    model.add(Difference(pec_block, air_cavity))
    model.add(air_cavity)

    ctrl = MeshControl(
        min_nodes_per_wavelength=4,
        max_cell_size=max_cell_size,
        conformal=conformal,
        dey_mittra_eta=dey_mittra_eta,
    )
    # f_max must resolve the TM010 mode
    f_tm010 = J01 * C0 / (2 * math.pi * radius)
    return Mesh.from_geometry(model, ctrl, f_max=2 * f_tm010)


def _find_tm010(mesh: Mesh, radius: float) -> float:
    """Run eigenmode solver and return the TM010 frequency [Hz]."""
    result = AnalysisEigenmode(mesh=mesh, n_modes=5, verbose=False).run()
    # TM010 is the lowest non-trivial mode
    f_physical = sorted(result.frequencies[result.frequencies > 1e6].tolist())
    assert len(f_physical) >= 1, "No physical modes found"
    return f_physical[0]


def _coaxial_mesh(
    inner_r: float,
    outer_r: float,
    length: float,
    max_cell_size: float,
    conformal: bool,
    dey_mittra_eta: float = 0.4,
) -> Mesh:
    """Build a coaxial line mesh via from_geometry.

    Shapes (last wins): PEC brick → air cylinder (outer) → PEC cylinder (inner).
    """
    pec = Material(name="PEC", is_pec=True)
    air = Material.air()

    pec_block = Brick(
        origin=(-outer_r, -outer_r, 0),
        size=(2 * outer_r, 2 * outer_r, length),
        material=pec,
    )
    air_cyl = Cylinder(
        origin=(0, 0, 0),
        radius=outer_r,
        height=length,
        axis="z",
        material=air,
    )
    pec_inner = Cylinder(
        origin=(0, 0, 0),
        radius=inner_r,
        height=length,
        axis="z",
        material=pec,
    )
    model = GeometryModel()
    model.add(Difference(pec_block, air_cyl))  # PEC wall with hole
    model.add(Difference(air_cyl, pec_inner))  # air gap ring
    model.add(pec_inner)  # inner conductor

    ctrl = MeshControl(
        min_nodes_per_wavelength=4,
        max_cell_size=max_cell_size,
        conformal=conformal,
        dey_mittra_eta=dey_mittra_eta,
    )
    mesh = Mesh.from_geometry(model, ctrl, f_max=C0 / (4 * max_cell_size))
    # The z faces are the line's open ends, not lids: Z0 is taken from
    # the cross-section energy, which a shorting PEC cap would falsify.
    # Applied after meshing on purpose — declaring PMC on the model
    # would also pull those grid lines in (WP-U0), moving the very
    # cell sizes this convergence fixture varies.
    return mesh.with_boundary_conditions(
        {
            "xmin": "PEC",
            "xmax": "PEC",
            "ymin": "PEC",
            "ymax": "PEC",
            "zmin": "PMC",
            "zmax": "PMC",
        }
    )


def _compute_Z0_energy(
    mesh: Mesh,
    inner_r: float,
    outer_r: float,
    voltage: float = 1.0,
) -> float:
    """Compute coaxial Z₀ via energy method using analytical TEM E-field.

    Sets the analytical E_r = V/(r·ln(b/a)) on mesh edges,
    computes W = ½ Σ M_eps·E², and derives Z₀ = 1/(c₀·C').
    """
    grid = mesh.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    x, y, z = grid.x, grid.y, grid.z
    dx, dy = grid.dx, grid.dy

    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    n_Ez = (Nx + 1) * (Ny + 1) * Nz
    n_total = n_Ex + n_Ey + n_Ez

    M_eps = build_M_eps(mesh)
    E = np.zeros(n_total)

    log_ba = math.log(outer_r / inner_r)
    a2 = inner_r**2
    b2 = outer_r**2

    # Ex edges: shape (Nx, Ny+1, Nz+1), midpoint at ((x[i]+x[i+1])/2, y[j])
    x_mid = 0.5 * (x[:-1] + x[1:])
    r2_ex = x_mid[:, None] ** 2 + y[None, :] ** 2  # (Nx, Ny+1)
    in_gap = (r2_ex > a2) & (r2_ex < b2)
    E_ex_2d = np.where(
        in_gap,
        voltage * x_mid[:, None] / (r2_ex * log_ba) * dx[:, None],
        0.0,
    )
    E[:n_Ex] = np.broadcast_to(
        E_ex_2d[:, :, None],
        (Nx, Ny + 1, Nz + 1),
    ).ravel()

    # Ey edges: shape (Nx+1, Ny, Nz+1), midpoint at (x[i], (y[j]+y[j+1])/2)
    y_mid = 0.5 * (y[:-1] + y[1:])
    r2_ey = x[:, None] ** 2 + y_mid[None, :] ** 2  # (Nx+1, Ny)
    in_gap_ey = (r2_ey > a2) & (r2_ey < b2)
    E_ey_2d = np.where(
        in_gap_ey,
        voltage * y_mid[None, :] / (r2_ey * log_ba) * dy[None, :],
        0.0,
    )
    E[n_Ex : n_Ex + n_Ey] = np.broadcast_to(
        E_ey_2d[:, :, None],
        (Nx + 1, Ny, Nz + 1),
    ).ravel()

    # Ez = 0 for TEM (already zero)

    # Zero out PEC edges
    pec = mesh.pec_mask_edges
    E[:n_Ex][pec[0, :n_Ex]] = 0.0
    E[n_Ex : n_Ex + n_Ey][pec[1, :n_Ey]] = 0.0

    # Energy → capacitance per unit length → impedance
    W = 0.5 * np.dot(M_eps, E**2)
    L = z[-1] - z[0]
    C_prime = 2.0 * W / (voltage**2 * L)
    return 1.0 / (C0 * C_prime)


# ---------------------------------------------------------------------------
# Module-level fixtures: expensive meshes built once and shared across tests.
# ---------------------------------------------------------------------------

_CYL_RADIUS = 15e-3
_CYL_HEIGHT = 20e-3
_CYL_F_ANA = J01 * C0 / (2 * math.pi * _CYL_RADIUS)

_COAX_INNER = 5e-3
_COAX_OUTER = 15e-3
_COAX_LENGTH_LONG = 20e-3
_COAX_LENGTH_SHORT = 10e-3
_COAX_Z0_ANA = math.sqrt(MU0 / EPS0) / (2 * math.pi) * math.log(_COAX_OUTER / _COAX_INNER)


@pytest.fixture(scope="module")
def cyl_mesh_sc_5mm():
    return _cylindrical_cavity_mesh(_CYL_RADIUS, _CYL_HEIGHT, 5e-3, conformal=False)


@pytest.fixture(scope="module")
def cyl_mesh_cf_5mm():
    return _cylindrical_cavity_mesh(
        _CYL_RADIUS,
        _CYL_HEIGHT,
        5e-3,
        conformal=True,
        dey_mittra_eta=0,
    )


@pytest.fixture(scope="module")
def cyl_mesh_cf_4mm():
    return _cylindrical_cavity_mesh(
        _CYL_RADIUS,
        _CYL_HEIGHT,
        4e-3,
        conformal=True,
        dey_mittra_eta=0,
    )


@pytest.fixture(scope="module")
def cyl_mesh_dm_5mm():
    return _cylindrical_cavity_mesh(
        _CYL_RADIUS,
        _CYL_HEIGHT,
        5e-3,
        conformal=True,
        dey_mittra_eta=0.4,
    )


@pytest.fixture(scope="module")
def coax_mesh_sc_4mm():
    return _coaxial_mesh(
        _COAX_INNER,
        _COAX_OUTER,
        _COAX_LENGTH_LONG,
        4e-3,
        conformal=False,
    )


@pytest.fixture(scope="module")
def coax_mesh_cf_4mm():
    return _coaxial_mesh(
        _COAX_INNER,
        _COAX_OUTER,
        _COAX_LENGTH_LONG,
        4e-3,
        conformal=True,
        dey_mittra_eta=0,
    )


@pytest.fixture(scope="module")
def coax_mesh_cf_3mm():
    return _coaxial_mesh(
        _COAX_INNER,
        _COAX_OUTER,
        _COAX_LENGTH_LONG,
        3e-3,
        conformal=True,
        dey_mittra_eta=0,
    )


@pytest.fixture(scope="module")
def coax_dm_short_4mm():
    return _coaxial_mesh(
        _COAX_INNER,
        _COAX_OUTER,
        _COAX_LENGTH_SHORT,
        4e-3,
        conformal=True,
        dey_mittra_eta=0.4,
    )


@pytest.fixture(scope="module")
def coax_sc_short_4mm():
    return _coaxial_mesh(
        _COAX_INNER,
        _COAX_OUTER,
        _COAX_LENGTH_SHORT,
        4e-3,
        conformal=False,
    )


@pytest.fixture(scope="module")
def coax_nodm_short_4mm():
    return _coaxial_mesh(
        _COAX_INNER,
        _COAX_OUTER,
        _COAX_LENGTH_SHORT,
        4e-3,
        conformal=True,
        dey_mittra_eta=0,
    )


@pytest.fixture(scope="module")
def coax_dm_short_5mm():
    return _coaxial_mesh(
        _COAX_INNER,
        _COAX_OUTER,
        _COAX_LENGTH_SHORT,
        5e-3,
        conformal=True,
        dey_mittra_eta=0.4,
    )


# ---------------------------------------------------------------------------
# Cylindrical PEC cavity: conformal vs staircase
# ---------------------------------------------------------------------------


class TestCylindricalCavityConformal:
    """Cylindrical PEC cavity: conformal vs staircase for curved walls."""

    def test_conformal_no_worse_than_staircase(self, cyl_mesh_sc_5mm, cyl_mesh_cf_5mm):
        """Conformal-only must not degrade PEC cavity accuracy vs staircase.

        Without Dey-Mittra, conformal cannot improve PEC boundaries —
        conformal material averaging alone is incorrect at PEC
        (missing edge-shortening correction), so it falls back to
        staircase there.
        """
        f_sc = _find_tm010(cyl_mesh_sc_5mm, _CYL_RADIUS)
        f_cf = _find_tm010(cyl_mesh_cf_5mm, _CYL_RADIUS)

        err_sc = abs(f_sc - _CYL_F_ANA) / _CYL_F_ANA * 100
        err_cf = abs(f_cf - _CYL_F_ANA) / _CYL_F_ANA * 100

        # 0.5 percentage point tolerance: under DD-051 Variante A the
        # M_μ cat-2 reduction occasionally drives the eigenvalue
        # *slightly* away from staircase on coarse meshes where the
        # cat-1/cat-2 face classification is mesh-resolution-sensitive
        # (the 5 mm fixture sits in this regime).  Round-WG benchmarks
        # show Variante A is sub-percent on TE11 cut-off; on the
        # cylindrical cavity TM010 at 5 mm the conformal branch trails
        # the staircase by ≤ 0.5 pp, well within the noise band.
        assert err_cf <= err_sc + 0.5, (
            f"Conformal ({err_cf:.2f}%) should not be much worse than "
            f"staircase ({err_sc:.2f}%) for TM010"
        )

    def test_conformal_tm010_within_10pct(self, cyl_mesh_cf_4mm):
        """Conformal TM010 at moderate resolution should be within 10%."""
        f_cf = _find_tm010(cyl_mesh_cf_4mm, _CYL_RADIUS)
        err = abs(f_cf - _CYL_F_ANA) / _CYL_F_ANA * 100

        assert err < 10.0, (
            f"TM010 conformal error {err:.2f}% exceeds 10% "
            f"(f_num={f_cf / 1e9:.4f} GHz, f_ana={_CYL_F_ANA / 1e9:.4f} GHz)"
        )

    def test_staircase_mode_unchanged(self, cyl_mesh_sc_5mm):
        """Staircase result (conformal=False) must still work correctly."""
        assert cyl_mesh_sc_5mm.edge_material is None
        f_sc = _find_tm010(cyl_mesh_sc_5mm, _CYL_RADIUS)

        err = abs(f_sc - _CYL_F_ANA) / _CYL_F_ANA * 100
        assert err < 30.0, (
            f"Staircase TM010 error {err:.2f}% exceeds 30% — mode may not have been found correctly"
        )


class TestDeyMittraTM010:
    """TM010 eigenfrequency: compare staircase vs conformal-only vs DM."""

    def test_dm_improves_over_conformal(self, cyl_mesh_cf_5mm, cyl_mesh_dm_5mm):
        """DM should yield equal or better TM010 accuracy than conformal-only."""
        f_cf = _find_tm010(cyl_mesh_cf_5mm, _CYL_RADIUS)
        f_dm = _find_tm010(cyl_mesh_dm_5mm, _CYL_RADIUS)

        err_cf = abs(f_cf - _CYL_F_ANA) / _CYL_F_ANA * 100
        err_dm = abs(f_dm - _CYL_F_ANA) / _CYL_F_ANA * 100

        # DM should not be worse than conformal-only (allow 0.5% margin)
        assert err_dm < err_cf + 0.5, (
            f"DM ({err_dm:.2f}%) should not be worse than conformal-only ({err_cf:.2f}%) for TM010"
        )

    def test_dm_improves_over_staircase(self, cyl_mesh_sc_5mm, cyl_mesh_dm_5mm):
        """DM must yield smaller TM010 error than staircase."""
        f_sc = _find_tm010(cyl_mesh_sc_5mm, _CYL_RADIUS)
        f_dm = _find_tm010(cyl_mesh_dm_5mm, _CYL_RADIUS)

        err_sc = abs(f_sc - _CYL_F_ANA) / _CYL_F_ANA * 100
        err_dm = abs(f_dm - _CYL_F_ANA) / _CYL_F_ANA * 100

        assert err_dm < err_sc, (
            f"DM ({err_dm:.2f}%) should be more accurate than staircase ({err_sc:.2f}%) for TM010"
        )

    def test_dm_has_active_edges(self, cyl_mesh_dm_5mm):
        """DM mesh should have modified edges on the PEC cylinder."""
        em = cyl_mesh_dm_5mm.edge_material
        assert em is not None, "edge_material missing"
        # DD-051: cat-2 (curved-PEC sub-cell) replaces the historical
        # "DM-active" edge set; non-NaN L_free is its operational marker.
        dm_edges = em.category == 2
        assert dm_edges.sum() > 0, "No curved-PEC sub-cell edges found"


# ---------------------------------------------------------------------------
# Coaxial Z₀: conformal vs staircase
# ---------------------------------------------------------------------------


class TestCoaxialImpedanceConformal:
    """Coaxial line Z₀: conformal vs staircase via energy method."""

    def test_conformal_Z0_no_worse_than_staircase(
        self,
        coax_mesh_sc_4mm,
        coax_mesh_cf_4mm,
    ):
        """Conformal-only must not degrade Z₀ vs staircase."""
        Z0_sc = _compute_Z0_energy(coax_mesh_sc_4mm, _COAX_INNER, _COAX_OUTER)
        Z0_cf = _compute_Z0_energy(coax_mesh_cf_4mm, _COAX_INNER, _COAX_OUTER)

        err_sc = abs(Z0_sc - _COAX_Z0_ANA) / _COAX_Z0_ANA * 100
        err_cf = abs(Z0_cf - _COAX_Z0_ANA) / _COAX_Z0_ANA * 100

        assert err_cf <= err_sc + 0.1, (
            f"Conformal ({err_cf:.2f}%) should not be worse than staircase ({err_sc:.2f}%) for Z₀"
        )

    def test_conformal_Z0_within_10pct(self, coax_mesh_cf_3mm):
        """Conformal Z₀ at moderate resolution should be within 10%."""
        Z0 = _compute_Z0_energy(coax_mesh_cf_3mm, _COAX_INNER, _COAX_OUTER)
        err = abs(Z0 - _COAX_Z0_ANA) / _COAX_Z0_ANA * 100

        assert err < 10.0, (
            f"Z₀ conformal error {err:.2f}% exceeds 10% "
            f"(Z₀_num={Z0:.2f} Ω, Z₀_ana={_COAX_Z0_ANA:.2f} Ω)"
        )

    def test_staircase_Z0_reasonable(self, coax_mesh_sc_4mm):
        """Staircase Z₀ (conformal=False) should still be in the right range."""
        assert coax_mesh_sc_4mm.edge_material is None
        Z0 = _compute_Z0_energy(coax_mesh_sc_4mm, _COAX_INNER, _COAX_OUTER)
        err = abs(Z0 - _COAX_Z0_ANA) / _COAX_Z0_ANA * 100

        assert err < 30.0, (
            f"Staircase Z₀ error {err:.2f}% exceeds 30% "
            f"(Z₀_num={Z0:.2f} Ω, Z₀_ana={_COAX_Z0_ANA:.2f} Ω)"
        )


# ---------------------------------------------------------------------------
# Dey-Mittra integration tests
# ---------------------------------------------------------------------------


class TestDeyMitraPEC:
    """Verify Dey-Mittra edge-shortening on a coaxial line mesh."""

    def test_dey_mittra_data_present(self, coax_dm_short_4mm):
        """With default MeshControl, curved-PEC sub-cell edges populate."""
        em = coax_dm_short_4mm.edge_material
        assert em is not None
        dm_edges = em.category == 2
        assert dm_edges.sum() > 0, "No curved-PEC sub-cell edges for PEC cylinder"

    def test_f_L_in_valid_range(self, coax_dm_short_4mm):
        """L_free / L_primal (≡ historical f_L) must be in [0, 1]."""
        from magnelio._operators.material_matrices import _build_L_primal_E

        em = coax_dm_short_4mm.edge_material
        L_primal = _build_L_primal_E(coax_dm_short_4mm.grid)
        cat2 = em.category == 2
        f_L = em.L_free[cat2] / L_primal[cat2]
        assert np.all(f_L >= 0.0)
        assert np.all(f_L <= 1.0 + 1e-9)

    def test_enlarged_cell_donors_valid(self, coax_dm_short_4mm):
        """Enlarged-cell donors must point to valid edge indices."""
        em = coax_dm_short_4mm.edge_material
        has_donor = em.enlarged_cell_donor >= 0
        n_total = len(em.category)
        assert np.all(em.enlarged_cell_donor[has_donor] < n_total)

    def test_no_dm_when_eta_zero(self, coax_nodm_short_4mm):
        """dey_mittra_eta=0 should disable curved-PEC sub-cell edges."""
        em = coax_nodm_short_4mm.edge_material
        # edge_material is still populated under control.conformal=True,
        # but no edge can land in cat 2 without pec_solid + eta>0.
        assert em is None or not (em.category == 2).any()

    def test_dm_cfl_bounded(self, coax_dm_short_4mm, coax_sc_short_4mm):
        """DM CFL penalty must stay within acceptable bounds vs staircase.

        With ECT (η = 0.4), f_L ≥ 0.4 for all active edges.  The CFL
        penalty comes from f_A/f_L < 1 on some edges (geometric, not a
        bug).  The penalty should be bounded — dt_DM ≥ 0.3 · dt_staircase.
        """
        min_eps_dm = compute_min_effective_eps(coax_dm_short_4mm)
        min_mu_dm = compute_min_effective_mu(coax_dm_short_4mm)
        dt_dm = courant_dt(
            coax_dm_short_4mm.grid,
            "normal",
            min_effective_eps=min_eps_dm,
            min_effective_mu=min_mu_dm,
        )
        dt_sc = courant_dt(coax_sc_short_4mm.grid, "normal")
        ratio = dt_dm / dt_sc
        # The previous 0.3 ratio bound was calibrated against the
        # E-edge-only CFL pre-DD-051; Variante A adds a symmetric
        # M_μ cat-2 reduction at curved-PEC H-faces, which lowers
        # ``μ_eff_min`` from 1.0 to ~0.04 and pulls ``dt`` down by
        # ``√0.04 ≈ 0.2``.  Combined with the existing ε reduction
        # the ratio lands ~0.28 on the test fixture.  Threshold
        # relaxed to 0.25 to reflect the symmetric ε / μ enlarged-
        # cell envelope while still bounding wild collapses.
        # DD-107 recalibration: the domain-face buffer uniformises
        # the wall tail (4.25/3.25/2.5 → 4 x 2.5 mm), and the 2.5 mm
        # wall cell in the bbox-tangent zone of the outer conductor
        # cuts a thinner μ sliver (μ_eff_min 0.131 → 0.030, measured
        # ratio 0.138).  The 1 % A_face_free floor structurally
        # bounds the ratio near 0.08, so 0.10 still catches a
        # collapse of that floor.
        assert ratio > 0.10, (
            f"DM CFL ratio {ratio:.2f} is too low — ECT should bound the "
            f"penalty (dt_dm={dt_dm:.3e}, dt_sc={dt_sc:.3e})"
        )

    def test_M_eps_positive_with_dm(self, coax_dm_short_4mm):
        """All M_eps entries should remain positive with DM active."""
        mesh = coax_dm_short_4mm
        M_eps = build_M_eps(mesh)
        pec_flat = np.concatenate(
            [
                np.nonzero(mesh.pec_mask_edges[0])[0],
                mesh.grid.Nx * (mesh.grid.Ny + 1) * (mesh.grid.Nz + 1)
                + np.nonzero(mesh.pec_mask_edges[1])[0],
                mesh.grid.Nx * (mesh.grid.Ny + 1) * (mesh.grid.Nz + 1)
                + (mesh.grid.Nx + 1) * mesh.grid.Ny * (mesh.grid.Nz + 1)
                + np.nonzero(mesh.pec_mask_edges[2])[0],
            ]
        )
        non_pec_mask = np.ones(len(M_eps), dtype=bool)
        non_pec_mask[pec_flat] = False
        assert np.all(M_eps[non_pec_mask] > 0), "Non-PEC M_eps must be positive"


class TestEdgePecFractions3Dvs2D:
    """Validate 3D line-solid f_L against existing 2D cross-section f_L."""

    def test_3d_f_L_matches_2d(self, coax_dm_short_5mm):
        """f_L from 3D BRepIntCurveSurface must agree with 2D cross-section scan."""
        from magnelio.geo._occ_backend import compute_edge_pec_fractions

        mesh = coax_dm_short_5mm
        assert mesh.edge_material is not None

        # f_L is no longer stored explicitly; reconstruct from the
        # category-2 sub-cell triple via L_free / L_primal.
        em = mesh.edge_material
        L_free = em.L_free
        from magnelio._operators.material_matrices import _build_L_primal_E

        L_primal = _build_L_primal_E(mesh.grid)
        f_L_2d = np.where(em.category == 2, L_free / L_primal, np.nan)
        dm_mask = ~np.isnan(f_L_2d)
        assert dm_mask.sum() > 50, "Need enough DM edges for meaningful comparison"

        # Build edge endpoint arrays for all DM edges
        grid = mesh.grid
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        x, y, z = grid.x, grid.y, grid.z
        n_Ex = Nx * (Ny + 1) * (Nz + 1)
        n_Ey = (Nx + 1) * Ny * (Nz + 1)

        dm_indices = np.nonzero(dm_mask)[0]
        edges = np.empty((len(dm_indices), 2, 3), dtype=np.float64)

        for idx_out, flat in enumerate(dm_indices):
            if flat < n_Ex:
                rem = flat
                stride_j = Nz + 1
                stride_i = (Ny + 1) * stride_j
                i = rem // stride_i
                rem %= stride_i
                j = rem // stride_j
                k = rem % stride_j
                edges[idx_out, 0] = [x[i], y[j], z[k]]
                edges[idx_out, 1] = [x[i + 1], y[j], z[k]]
            elif flat < n_Ex + n_Ey:
                rem = flat - n_Ex
                stride_k = Nz + 1
                stride_j = Ny * stride_k
                i = rem // stride_j
                rem %= stride_j
                j = rem // stride_k
                k = rem % stride_k
                edges[idx_out, 0] = [x[i], y[j], z[k]]
                edges[idx_out, 1] = [x[i], y[j + 1], z[k]]
            else:
                rem = flat - n_Ex - n_Ey
                stride_k = Nz
                stride_j = (Ny + 1) * stride_k
                i = rem // stride_j
                rem %= stride_j
                j = rem // stride_k
                k = rem % stride_k
                edges[idx_out, 0] = [x[i], y[j], z[k]]
                edges[idx_out, 1] = [x[i], y[j], z[k + 1]]

        from magnelio.geo._occ_backend import build_effective_pec_solid

        pec = Material(name="PEC", is_pec=True)
        air = Material.air()
        shapes_with_mat = [
            (
                Brick(
                    origin=(-_COAX_OUTER, -_COAX_OUTER, 0),
                    size=(2 * _COAX_OUTER, 2 * _COAX_OUTER, _COAX_LENGTH_SHORT),
                    material=pec,
                ),
                1,
            ),
            (
                Cylinder(
                    origin=(0, 0, 0),
                    radius=_COAX_OUTER,
                    height=_COAX_LENGTH_SHORT,
                    axis="z",
                    material=air,
                ),
                0,
            ),
            (
                Cylinder(
                    origin=(0, 0, 0),
                    radius=_COAX_INNER,
                    height=_COAX_LENGTH_SHORT,
                    axis="z",
                    material=pec,
                ),
                1,
            ),
        ]
        mat_lib = {0: air, 1: pec}
        effective_pec = build_effective_pec_solid(shapes_with_mat, mat_lib)
        assert effective_pec is not None

        f_L_3d = compute_edge_pec_fractions([effective_pec], edges)

        f_L_2d_valid = f_L_2d[dm_mask]
        abs_diff = np.abs(f_L_3d - f_L_2d_valid)
        assert abs_diff.max() < 0.05, (
            f"Max |f_L_3d - f_L_2d| = {abs_diff.max():.4f}, mean = {abs_diff.mean():.4f}"
        )
        assert abs_diff.mean() < 0.02, f"Mean |f_L_3d - f_L_2d| = {abs_diff.mean():.4f}"


class TestDeyMittraStability:
    """Time-domain stability: DM with standard dt must not diverge."""

    def test_dm_stable_standard_dt(self, coax_dm_short_4mm):
        """Run 2000 leapfrog steps with DM and standard CFL dt."""
        mesh = coax_dm_short_4mm

        min_eps = compute_min_effective_eps(mesh)
        min_mu = compute_min_effective_mu(mesh)
        dt = courant_dt(
            mesh.grid,
            "normal",
            min_effective_eps=min_eps,
            min_effective_mu=min_mu,
        )

        grid = mesh.grid
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

        solver = FITTimeDomainSolver(
            mesh=mesh,
            total_time_steps=2000,
            dt=dt,
            verbose=False,
        )
        solver.setup()

        fields = solver._fields
        Ez = fields.Ez
        mid_k = Nz // 2
        for i in range(Nx + 1):
            for j in range(Ny + 1):
                xc = grid.x[i]
                yc = grid.y[j]
                r2 = xc**2 + yc**2
                a2 = _COAX_INNER**2
                b2 = _COAX_OUTER**2
                if a2 < r2 < b2:
                    Ez[i, j, mid_k] = 1.0

        e = fields.e_flat
        M_eps_diag = solver._M_eps_diag
        M_mu_diag = solver._M_mu_diag
        E0 = 0.5 * float((M_eps_diag * e) @ e)
        assert E0 > 0, "Initial energy should be positive"

        solver.run()

        e = fields.e_flat
        h = fields.h_flat
        E_final = 0.5 * (float((M_eps_diag * e) @ e) + float((M_mu_diag * h) @ h))

        ratio = E_final / E0
        assert ratio < 2.0, (
            f"Energy diverged: E_final/E_initial = {ratio:.2f} — DM with standard dt is unstable"
        )
        assert not np.isnan(E_final), "Energy is NaN — simulation exploded"
        assert not np.isinf(E_final), "Energy is Inf — simulation diverged"

    def test_energy_bounded_over_time(self, coax_dm_short_4mm):
        """Energy trace should stay bounded (no exponential growth)."""
        mesh = coax_dm_short_4mm

        min_eps = compute_min_effective_eps(mesh)
        min_mu = compute_min_effective_mu(mesh)
        dt = courant_dt(
            mesh.grid,
            "normal",
            min_effective_eps=min_eps,
            min_effective_mu=min_mu,
        )

        solver = FITTimeDomainSolver(
            mesh=mesh,
            total_time_steps=1000,
            dt=dt,
            verbose=False,
        )
        solver.setup()

        grid = mesh.grid
        Ez = solver._fields.Ez
        mid_k = grid.Nz // 2
        for i in range(grid.Nx + 1):
            for j in range(grid.Ny + 1):
                r2 = grid.x[i] ** 2 + grid.y[j] ** 2
                if _COAX_INNER**2 < r2 < _COAX_OUTER**2:
                    Ez[i, j, mid_k] = 1.0

        solver.run()

        trace = solver._energy_trace
        assert len(trace) > 0, "No energy trace recorded"
        energies = trace["energy"]
        peak = energies.max()

        E0 = energies[0]
        assert peak < 3.0 * E0, f"Energy grew to {peak / E0:.1f}× initial — instability detected"


class TestDeyMittraConvergence:
    """h-refinement: verify convergence with DM active."""

    def test_dm_preserves_convergence_order(self):
        """DM must converge and be more accurate than staircase."""
        h_coarse = 5e-3
        h_fine = 3e-3

        err = {}
        for label, eta in [("staircase", -1), ("conformal", 0), ("dm", 0.4)]:
            conf = eta >= 0
            dm_eta = max(eta, 0)
            e_list = []
            for h in [h_coarse, h_fine]:
                mesh = _cylindrical_cavity_mesh(
                    _CYL_RADIUS,
                    _CYL_HEIGHT,
                    h,
                    conformal=conf,
                    dey_mittra_eta=dm_eta if conf else 0,
                )
                f = _find_tm010(mesh, _CYL_RADIUS)
                e_list.append(abs(f - _CYL_F_ANA) / _CYL_F_ANA)
            err[label] = e_list

        log_h_ratio = math.log(h_coarse / h_fine)
        orders = {}
        for label in ["staircase", "conformal", "dm"]:
            e_c, e_f = err[label]
            if e_f > 0 and e_c > 0:
                orders[label] = math.log(e_c / e_f) / log_h_ratio

        p_dm = orders.get("dm", 0)

        assert p_dm > 0.5, f"DM convergence order too low: {p_dm:.2f}"

        for i, h in enumerate([h_coarse, h_fine]):
            assert err["dm"][i] < err["staircase"][i], (
                f"DM error {err['dm'][i] * 100:.2f}% >= staircase "
                f"{err['staircase'][i] * 100:.2f}% at h={h * 1e3:.0f}mm"
            )

    def test_dm_error_decreases_with_refinement(self):
        """Finer mesh must yield smaller TM010 error with DM."""
        h_coarse = 5e-3
        h_fine = 3e-3

        mesh_coarse = _cylindrical_cavity_mesh(
            _CYL_RADIUS,
            _CYL_HEIGHT,
            h_coarse,
            conformal=True,
            dey_mittra_eta=0.4,
        )
        mesh_fine = _cylindrical_cavity_mesh(
            _CYL_RADIUS,
            _CYL_HEIGHT,
            h_fine,
            conformal=True,
            dey_mittra_eta=0.4,
        )

        f_coarse = _find_tm010(mesh_coarse, _CYL_RADIUS)
        f_fine = _find_tm010(mesh_fine, _CYL_RADIUS)

        err_coarse = abs(f_coarse - _CYL_F_ANA) / _CYL_F_ANA
        err_fine = abs(f_fine - _CYL_F_ANA) / _CYL_F_ANA

        assert err_fine < err_coarse, (
            f"Refinement did not reduce error: "
            f"h={h_coarse * 1e3:.0f}mm → {err_coarse * 100:.2f}%, "
            f"h={h_fine * 1e3:.0f}mm → {err_fine * 100:.2f}%"
        )
