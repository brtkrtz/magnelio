"""Unit tests for the public modal-port factory.

Per DD-048 the factory builds two modes per port:

- a *reference* mode (analytical or refined-2D) that fills
  ``op.port_report.z_line_ref`` / ``cutoff_ref`` for user reporting;
- an *operator-consistent* mode (solved on the 3D-mesh transversal
  slice) that drives the FIT-TD coupling.

These tests cover:

1. ``PortSpecRectWG`` / ``PortSpecCoax`` UV-convention: width_a/height_b
   and center swap correctly between MIN and MAX faces (both reference
   and operator-consistent path see the right geometry).
2. ``ExcitationSpec`` wires into the operator.
3. ``port_report`` carries both reference and operator values; for a
   refined coax mesh, ``z_line_num`` ≈ ``z_line_ref``.
4. ``PortSpecCoax`` auto-detection rejects geometries that do not
   match the spec to within one mesh cell.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from magnelio import Material
from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.geo import Brick, Cylinder, Difference, GeometryModel
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh, MeshControl
from magnelio.ports._modal import (
    BoxFace,
    ExcitationSpec,
    PortOperatorModal,
    PortOperatorReport,
    PortPlane,
    PortSpecCoax,
    PortSpecRectWG,
    build_modal_port,
)
from magnelio.signals.waveforms import gaussian, modulated_gaussian
from magnelio.solver.stability import courant_dt

WR90_A = 22.86e-3
WR90_B = 10.16e-3


@pytest.fixture
def wr90_mesh():
    L_x = 30e-3
    grid = GridLines(
        x=np.linspace(0.0, L_x, 31),
        y=np.linspace(0.0, WR90_A, 24),
        z=np.linspace(0.0, WR90_B, 11),
    )
    mesh = Mesh.from_grid(grid, boundary_conditions=_BC_OPEN)
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    dt = courant_dt(grid, accuracy="normal")
    return mesh, m_eps, m_mu, dt


@pytest.fixture
def coax_mesh():
    L_x = 30e-3
    L_yz = 3e-3
    grid = GridLines(
        x=np.linspace(0.0, L_x, 121),
        y=np.linspace(0.0, L_yz, 13),
        z=np.linspace(0.0, L_yz, 13),
    )
    mesh = Mesh.from_grid(grid, boundary_conditions=_BC_OPEN)
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    dt = courant_dt(grid, accuracy="normal")
    return mesh, m_eps, m_mu, dt, L_yz


def _operator_signature(op: PortOperatorModal) -> dict:
    """Capture the user-observable state of an operator for equality checks."""
    return {
        "label": op.name,
        "n_modes": op.n_modes,
        "face": op.plane.face,
        "normal_dx": op.plane.normal_dx,
        "mur_r": np.array(op._mur_r, copy=True),
        "v_p": np.array(op._v_p, copy=True),
        "tau_m": np.array(op._tau_m, copy=True),
        "e_u_profiles": [np.array(dm.e_u_profile) for dm in op.discrete_modes],
        "e_v_profiles": [np.array(dm.e_v_profile) for dm in op.discrete_modes],
        "h_u_profiles": [np.array(dm.h_u_profile) for dm in op.discrete_modes],
        "h_v_profiles": [np.array(dm.h_v_profile) for dm in op.discrete_modes],
        "mode_omega_c": [dm.mode.omega_c for dm in op.discrete_modes],
        "excitation_mode": op._excitation_mode,
    }


def _assert_signatures_equal(sig_a: dict, sig_b: dict) -> None:
    assert sig_a["label"] == sig_b["label"]
    assert sig_a["n_modes"] == sig_b["n_modes"]
    assert sig_a["face"] == sig_b["face"]
    assert sig_a["normal_dx"] == sig_b["normal_dx"]
    assert sig_a["excitation_mode"] == sig_b["excitation_mode"]
    np.testing.assert_allclose(sig_a["mur_r"], sig_b["mur_r"])
    np.testing.assert_allclose(sig_a["v_p"], sig_b["v_p"])
    np.testing.assert_allclose(sig_a["tau_m"], sig_b["tau_m"])
    for key in ("e_u_profiles", "e_v_profiles", "h_u_profiles", "h_v_profiles"):
        for p_a, p_b in zip(sig_a[key], sig_b[key]):
            np.testing.assert_allclose(p_a, p_b)
    assert sig_a["mode_omega_c"] == sig_b["mode_omega_c"]


# ---------------------------------------------------------------------------
# Coax — OCC fixture for two-path tests (DD-048 needs a real PEC mask)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def coax_occ_mesh():
    """Real OCC coax: PEC bbox, dielectric annulus, PEC inner cylinder.

    Provides a mesh whose ``pec_mask_edges`` carries staircased PEC
    contours of two conductors — the topology DD-048 path-(b) needs.
    """
    D_i = 0.41e-3
    D_a = 5.0e-3
    L = 10.0e-3
    eps_r = 9.0
    s_bbox = 1.2 * D_a
    pec = Material.pec()
    diel = Material(name="dielectric", epsilon=(eps_r,) * 3)
    bbox = Brick(
        origin=(-s_bbox / 2, -s_bbox / 2, 0.0),
        size=(s_bbox, s_bbox, L),
        material=pec,
    )
    out_cyl = Cylinder(
        origin=(0.0, 0.0, 0.0),
        radius=D_a / 2,
        height=L,
        axis="z",
        material=diel,
    )
    in_cyl = Cylinder(
        origin=(0.0, 0.0, 0.0),
        radius=D_i / 2,
        height=L,
        axis="z",
        material=pec,
    )
    model = GeometryModel()
    model.add(Difference(bbox, out_cyl))
    model.add(Difference(out_cyl, in_cyl))
    model.add(in_cyl)
    control = MeshControl(
        min_nodes_per_wavelength=8,
        min_cells_per_feature=3,
        growth_factor=1.4,
        conformal=True,
        max_cell_size=0.4e-3,
        min_cell_size=50e-6,
        min_feature_gap=20e-6,
    )
    mesh = Mesh.from_geometry(model, control, f_max=10e9)
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    dt = courant_dt(mesh.grid, accuracy="normal")
    return mesh, m_eps, m_mu, dt, D_i, D_a, eps_r


def test_coax_factory_port_report_two_paths(coax_occ_mesh):
    """PortSpecCoax fills both reference and operator-consistent fields."""
    mesh, m_eps, m_mu, dt, D_i, D_a, eps_r = coax_occ_mesh
    spec = PortSpecCoax(
        name="p1",
        plane=BoxFace.Z_MIN,
        inner_radius=D_i / 2,
        outer_radius=D_a / 2,
        epsilon_r=eps_r,
        n_modes=1,
    )
    op = build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=10e9)
    rep = op.port_report
    assert isinstance(rep, PortOperatorReport)
    # Reference is the closed-form 50 Ω of the continuous geometry.
    eta0 = 376.730313668
    z_anal = (eta0 / math.sqrt(eps_r)) / (2.0 * math.pi) * math.log(D_a / D_i)
    assert rep.z_line_ref == pytest.approx(z_anal, rel=1e-9)
    # Operator-consistent value is the staircased FIT Z_line — within
    # 5 % of the continuous value at this mesh density.
    assert rep.z_line_num is not None
    assert abs(rep.z_line_num - z_anal) / z_anal < 0.05
    # delta_relative is populated.
    assert rep.z_line_delta_relative is not None


def test_coax_factory_xmin_xmax_match(coax_occ_mesh):
    """Coax at Z_MIN and Z_MAX produce the same physical mode."""
    mesh, m_eps, m_mu, dt, D_i, D_a, eps_r = coax_occ_mesh
    spec_min = PortSpecCoax(
        name="p1",
        plane=BoxFace.Z_MIN,
        inner_radius=D_i / 2,
        outer_radius=D_a / 2,
        epsilon_r=eps_r,
    )
    spec_max = PortSpecCoax(
        name="p2",
        plane=BoxFace.Z_MAX,
        inner_radius=D_i / 2,
        outer_radius=D_a / 2,
        epsilon_r=eps_r,
    )
    op_min = build_modal_port(spec_min, mesh, m_eps, m_mu, dt=dt, f_calc=10e9)
    op_max = build_modal_port(spec_max, mesh, m_eps, m_mu, dt=dt, f_calc=10e9)
    assert op_min.port_report.z_line_num == pytest.approx(
        op_max.port_report.z_line_num,
        rel=1e-9,
    )
    assert op_min.port_report.z_line_ref == pytest.approx(
        op_max.port_report.z_line_ref,
        rel=1e-9,
    )


def test_coax_factory_excitation_routing(coax_occ_mesh):
    """ExcitationSpec(gaussian) wires through to operator."""
    mesh, m_eps, m_mu, dt, D_i, D_a, eps_r = coax_occ_mesh
    spec = PortSpecCoax(
        name="p1",
        plane=BoxFace.Z_MIN,
        inner_radius=D_i / 2,
        outer_radius=D_a / 2,
        epsilon_r=eps_r,
        excitation=ExcitationSpec(
            f_min=0.0,
            f_max=10e9,
            mode_index=0,
            waveform="gaussian",
        ),
    )
    op = build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=10e9)
    assert op._excitation_mode == 0
    t_test = float(np.linspace(0.0, 4.0 / 10e9, 7)[3])
    # DD-078: set_excitation scales the user waveform (a(t) in √W) into
    # the operator's internal basis units by source_scale.
    expected = float(gaussian(t_test, 10e9)) * float(op._source_scale[0])
    actual = op._excitation_waveform(t_test)
    assert actual == pytest.approx(expected)


def test_coax_factory_rejects_displaced_center(coax_occ_mesh):
    """Auto-detection cross-check: spec.center far from real centroid."""
    mesh, m_eps, m_mu, dt, D_i, D_a, eps_r = coax_occ_mesh
    # Real coax sits at (0, 0); pretend the user typed (1 mm, 0).
    spec = PortSpecCoax(
        name="p1",
        plane=BoxFace.Z_MIN,
        inner_radius=D_i / 2,
        outer_radius=D_a / 2,
        epsilon_r=eps_r,
        center=(1e-3, 0.0),
    )
    with pytest.raises(ValueError, match=r"differs from spec\.center"):
        build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=10e9)


# ---------------------------------------------------------------------------
# Rectangular waveguide — bare-grid fallback covers Mesh.from_grid setups
# ---------------------------------------------------------------------------


def test_rectwg_factory_xmin_xmax_share_te10_cutoff(wr90_mesh):
    """Both ports report the same TE10 cutoff (UV-convention)."""
    mesh, m_eps, m_mu, dt = wr90_mesh
    f_calc = 10e9
    spec_min = PortSpecRectWG(
        name="port1",
        plane=BoxFace.X_MIN,
        width_a=WR90_A,
        height_b=WR90_B,
        n_modes=3,
    )
    spec_max = PortSpecRectWG(
        name="port2",
        plane=BoxFace.X_MAX,
        width_a=WR90_A,
        height_b=WR90_B,
        n_modes=3,
    )
    op_min = build_modal_port(spec_min, mesh, m_eps, m_mu, dt=dt, f_calc=f_calc)
    op_max = build_modal_port(spec_max, mesh, m_eps, m_mu, dt=dt, f_calc=f_calc)

    expected_te10_omega_c = (math.pi / WR90_A) * 299_792_458.0

    # Path (a) is mesh-independent — must match the closed form exactly.
    assert op_min.port_report.cutoff_ref == pytest.approx(
        expected_te10_omega_c / (2.0 * math.pi),
        rel=1e-9,
    )
    assert op_max.port_report.cutoff_ref == pytest.approx(
        op_min.port_report.cutoff_ref,
        rel=1e-9,
    )

    # Path (b) is operator-consistent — staircased y-resolution shifts
    # the discrete cutoff by a small amount; both ports must agree.
    assert op_min.port_report.cutoff_num == pytest.approx(
        op_max.port_report.cutoff_num,
        rel=1e-9,
    )
    # Discrete cutoff lies within ~1 % of the analytical value at this mesh.
    rel_err = (
        abs(op_min.port_report.cutoff_num - op_min.port_report.cutoff_ref)
        / op_min.port_report.cutoff_ref
    )
    assert rel_err < 0.01


def test_rectwg_factory_xmax_uv_convention_via_port_report(wr90_mesh):
    """At X_MAX the (u, v) swap must yield the same physical TE10 cutoff."""
    mesh, m_eps, m_mu, dt = wr90_mesh
    spec_max = PortSpecRectWG(
        name="p",
        plane=BoxFace.X_MAX,
        width_a=WR90_A,
        height_b=WR90_B,
        n_modes=1,
    )
    op = build_modal_port(spec_max, mesh, m_eps, m_mu, dt=dt, f_calc=10e9)
    expected_f_c = 299_792_458.0 / (2.0 * WR90_A)
    assert op.port_report.cutoff_ref == pytest.approx(expected_f_c, rel=1e-9)


def test_rectwg_factory_excitation_modulated_gaussian(wr90_mesh):
    """ExcitationSpec(modulated_gaussian) wires through to operator."""
    mesh, m_eps, m_mu, dt = wr90_mesh
    f_calc = 10e9

    spec = PortSpecRectWG(
        name="port1",
        plane=BoxFace.X_MIN,
        width_a=WR90_A,
        height_b=WR90_B,
        excitation=ExcitationSpec(f_min=8.2e9, f_max=12.4e9, mode_index=0),
    )
    op = build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=f_calc)
    assert op._excitation_mode == 0

    t_test = 4.0 / (12.4e9 - 8.2e9)
    # DD-078: user waveform (√W) × source_scale = internal basis units.
    expected = float(modulated_gaussian(t_test, 12.4e9, 8.2e9)) * float(op._source_scale[0])
    actual = op._excitation_waveform(t_test)
    assert actual == pytest.approx(expected)


def test_excitation_spec_unknown_waveform_raises():
    spec = ExcitationSpec(f_min=1e9, f_max=2e9, waveform="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown waveform"):
        spec.build_waveform()


def test_build_modal_port_invalid_spec_type(wr90_mesh):
    mesh, m_eps, m_mu, dt = wr90_mesh
    with pytest.raises(TypeError, match="unsupported port spec type"):
        build_modal_port(
            "not-a-spec",  # type: ignore[arg-type]
            mesh,
            m_eps,
            m_mu,
            dt=dt,
            f_calc=10e9,
        )


def test_build_modal_port_rejects_bad_dt_and_fcalc(wr90_mesh):
    mesh, m_eps, m_mu, _ = wr90_mesh
    spec = PortSpecRectWG(
        name="p",
        plane=BoxFace.X_MIN,
        width_a=WR90_A,
        height_b=WR90_B,
    )
    with pytest.raises(ValueError, match="dt must be positive"):
        build_modal_port(spec, mesh, m_eps, m_mu, dt=0.0, f_calc=10e9)
    with pytest.raises(ValueError, match="f_calc must be positive"):
        build_modal_port(spec, mesh, m_eps, m_mu, dt=1e-12, f_calc=0.0)


# =====================================================================
# §2.4 "three equidistant cells at the port" rule (commercial-mesher prereq)
# =====================================================================


def _wr90_mesh_with_x_nodes(x_nodes):
    """Build the standard WR-90 mesh from explicit x-node coordinates."""
    grid = GridLines(
        x=np.asarray(x_nodes, dtype=float),
        y=np.linspace(0.0, WR90_A, 24),
        z=np.linspace(0.0, WR90_B, 11),
    )
    mesh = Mesh.from_grid(grid, boundary_conditions=_BC_OPEN)
    return mesh, build_M_eps(mesh), build_M_mu(mesh), courant_dt(grid)


def test_three_eq_cells_uniform_grid_accepted():
    """Uniform port-normal mesh — passes both X_MIN and X_MAX checks."""
    mesh, m_eps, m_mu, dt = _wr90_mesh_with_x_nodes(np.linspace(0.0, 30e-3, 31))
    for face in (BoxFace.X_MIN, BoxFace.X_MAX):
        spec = PortSpecRectWG(
            name="p",
            plane=face,
            width_a=WR90_A,
            height_b=WR90_B,
        )
        build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=10e9)


def test_three_eq_cells_violation_at_xmin_raises():
    """Geometric progression — first three cells differ; X_MIN fails."""
    n = 30
    x_nodes = np.cumsum(np.concatenate([[0.0], 1.05 ** np.arange(n)]))
    x_nodes = x_nodes / x_nodes[-1] * 30e-3
    mesh, m_eps, m_mu, dt = _wr90_mesh_with_x_nodes(x_nodes)
    spec = PortSpecRectWG(
        name="p",
        plane=BoxFace.X_MIN,
        width_a=WR90_A,
        height_b=WR90_B,
    )
    with pytest.raises(
        ValueError,
        match=r"three equidistant cells",
    ):
        build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=10e9)


def test_three_eq_cells_violation_at_xmax_raises():
    """Reverse-progressive grid — last three cells differ; X_MAX fails."""
    n = 30
    sizes = 1.05 ** np.arange(n)
    x_nodes = np.cumsum(np.concatenate([[0.0], sizes[::-1]]))
    x_nodes = x_nodes / x_nodes[-1] * 30e-3
    mesh, m_eps, m_mu, dt = _wr90_mesh_with_x_nodes(x_nodes)
    spec = PortSpecRectWG(
        name="p",
        plane=BoxFace.X_MAX,
        width_a=WR90_A,
        height_b=WR90_B,
    )
    with pytest.raises(
        ValueError,
        match=r"three equidistant cells",
    ):
        build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=10e9)


def test_three_eq_cells_with_buffer_at_xmin_accepted():
    """3-cell equidistant buffer at the port + variable interior — passes."""
    buf = 0.5e-3
    n_buf = 3
    inner_n = 24
    inner_total = 30e-3 - 2 * n_buf * buf
    inner_sizes = 1.2 ** np.arange(inner_n)
    inner_sizes = inner_sizes / inner_sizes.sum() * inner_total
    x_nodes = np.concatenate(
        [
            np.arange(n_buf + 1) * buf,
            n_buf * buf + np.cumsum(inner_sizes),
            30e-3 - np.arange(n_buf - 1, -1, -1) * buf,
        ]
    )
    mesh, m_eps, m_mu, dt = _wr90_mesh_with_x_nodes(x_nodes)
    for face in (BoxFace.X_MIN, BoxFace.X_MAX):
        spec = PortSpecRectWG(
            name="p",
            plane=face,
            width_a=WR90_A,
            height_b=WR90_B,
        )
        build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=10e9)


def test_three_eq_cells_too_few_cells_raises():
    """Grid with only 2 cells along port-normal axis is rejected."""
    mesh, m_eps, m_mu, dt = _wr90_mesh_with_x_nodes(
        np.linspace(0.0, 30e-3, 3)  # 2 cells, 3 nodes
    )
    spec = PortSpecRectWG(
        name="p",
        plane=BoxFace.X_MIN,
        width_a=WR90_A,
        height_b=WR90_B,
    )
    with pytest.raises(
        ValueError,
        match=r"at least 3 cells required",
    ):
        build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=10e9)


# =====================================================================
# PortSpecNumerical — Phase 2a step 4: numerical mode-solver dispatch
# =====================================================================


from magnelio.ports._modal import (  # noqa: E402  (group with PortSpecNumerical tests)
    ModeType,
    PortSpecNumerical,
)

# DD-103: the closure these fixtures always assumed.  A face
# with no BC used to evolve under the free curl operator —
# which IS the natural magnetic wall, hence "PMC".
_BC_OPEN = {
    "xmin": "PMC",
    "xmax": "PMC",
    "ymin": "PMC",
    "ymax": "PMC",
    "zmin": "PMC",
    "zmax": "PMC",
}


@pytest.fixture
def wr90_mesh_high_res():
    """Higher-resolution WR-90 setup for the numerical Test-1N convergence."""
    L_x = 10e-3
    grid = GridLines(
        x=np.linspace(0.0, L_x, 5),
        y=np.linspace(0.0, WR90_A, 61),
        z=np.linspace(0.0, WR90_B, 31),
    )
    mesh = Mesh.from_grid(grid, boundary_conditions=_BC_OPEN)
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    dt = courant_dt(grid, accuracy="normal")
    return mesh, m_eps, m_mu, dt


class TestPortSpecNumericalFactory:
    """PortSpecNumerical dispatches to the numerical 2D mode solver."""

    def test_factory_returns_modal_port_operator(self, wr90_mesh_high_res):
        mesh, m_eps, m_mu, dt = wr90_mesh_high_res
        spec = PortSpecNumerical(
            name="port_num",
            plane=BoxFace.X_MIN,
            n_modes=2,
            epsilon_r=1.0,
            mode_type=ModeType.TE,
        )
        op = build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=10e9)
        assert isinstance(op, PortOperatorModal)
        assert op.n_modes == 2
        assert len(op.discrete_modes) == 2

    def test_te10_te20_cutoffs_within_one_percent(self, wr90_mesh_high_res):
        """End-to-end Test 1N through the public factory.

        Bare ``Mesh.from_grid`` ⇒ empty PEC mask ⇒ factory falls back to
        the four-lateral-bbox-faces hollow-WG default (DD-050).
        """
        mesh, m_eps, m_mu, dt = wr90_mesh_high_res
        spec = PortSpecNumerical(
            name="port_num",
            plane=BoxFace.X_MIN,
            n_modes=4,
            epsilon_r=1.0,
        )
        op = build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=10e9)
        omega_c = sorted(m.mode.omega_c for m in op.discrete_modes)
        f_c = [w / (2 * math.pi) for w in omega_c]
        # Analytical TE_10, TE_20 (a > 2b ⇒ TE_20 below TE_01), TE_01.
        c0 = 2.99792458e8
        f_te10 = c0 / (2 * WR90_A)
        f_te20 = c0 / WR90_A
        f_te01 = c0 / (2 * WR90_B)
        assert abs(f_c[0] - f_te10) / f_te10 < 0.01
        assert abs(f_c[1] - f_te20) / f_te20 < 0.01
        assert abs(f_c[2] - f_te01) / f_te01 < 0.02

    def test_unified_multimode_port_mixes_te_and_tm(
        self,
        wr90_mesh_high_res,
    ):
        """mode_type=None merges the TE and TM families sorted by
        cut-off in ONE operator — the WP-R3 unified multi-mode port
        (formerly two colliding operators, STATUS site 1).  For WR-90
        the first mixed-type entries are the degenerate TE11/TM11
        pair above TE10/TE20/TE01."""
        mesh, m_eps, m_mu, dt = wr90_mesh_high_res
        spec = PortSpecNumerical(
            name="port_num",
            plane=BoxFace.X_MIN,
            n_modes=5,
        )
        op = build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=20e9)
        assert op.n_modes == 5
        kinds = [dm.mode.mode_type for dm in op.discrete_modes]
        assert ModeType.TE in kinds and ModeType.TM in kinds
        omega_c = [dm.mode.omega_c for dm in op.discrete_modes]
        assert omega_c == sorted(omega_c)

    def test_modes_are_phase2_numerical_path(self, wr90_mesh_high_res):
        """Resulting Mode objects carry discrete_*_profile (no field_evaluator)."""
        mesh, m_eps, m_mu, dt = wr90_mesh_high_res
        spec = PortSpecNumerical(
            name="port_num",
            plane=BoxFace.X_MIN,
            n_modes=1,
        )
        op = build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=10e9)
        m = op.discrete_modes[0].mode
        assert m.field_evaluator is None
        assert m.discrete_e_u_profile is not None
        assert m.discrete_e_v_profile is not None
        assert m.discrete_h_u_profile is not None
        assert m.discrete_h_v_profile is not None

    def test_excitation_routed_through_factory(self, wr90_mesh_high_res):
        mesh, m_eps, m_mu, dt = wr90_mesh_high_res
        excitation = ExcitationSpec(
            f_min=8.2e9,
            f_max=12.4e9,
            mode_index=0,
            waveform="modulated_gaussian",
        )
        spec = PortSpecNumerical(
            name="port_num",
            plane=BoxFace.X_MIN,
            n_modes=1,
            excitation=excitation,
        )
        op = build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=10e9)
        assert op._excitation_mode == 0
        # Excited waveform matches the modulated-gaussian closure scaled
        # into basis units (DD-078 source_scale).
        t_test = 4.0 / (12.4e9 - 8.2e9)
        expected = float(modulated_gaussian(t_test, 12.4e9, 8.2e9)) * float(op._source_scale[0])
        assert op._excitation_waveform(t_test) == pytest.approx(expected)

    def test_pec_mask_consolidated_via_with_pec_boundaries(
        self,
        wr90_mesh_high_res,
    ):
        """DD-050: ``Mesh.with_pec_boundaries`` populates the canonical
        PEC mask, and the factory picks it up — no explicit
        ``lateral_pec_faces`` argument needed (the field is gone)."""
        mesh, m_eps, m_mu, dt = wr90_mesh_high_res
        # Consolidate "ymin/ymax/zmin/zmax = PEC" into ``pec_mask_edges``
        # via the canonical DD-046 pathway.
        mesh_pec = mesh.with_boundary_conditions(
            {
                "ymin": "PEC",
                "ymax": "PEC",
                "zmin": "PEC",
                "zmax": "PEC",
                "xmin": "PMC",
                "xmax": "PMC",
            }
        )
        spec = PortSpecNumerical(
            name="port_num",
            plane=BoxFace.X_MIN,
            n_modes=2,
        )
        op = build_modal_port(
            spec,
            mesh_pec,
            m_eps,
            m_mu,
            dt=dt,
            f_calc=10e9,
        )
        omega_c = sorted(m.mode.omega_c for m in op.discrete_modes)
        f_c0 = omega_c[0] / (2 * math.pi)
        c0 = 2.99792458e8
        f_te10 = c0 / (2 * WR90_A)
        # Same TE10 cutoff as the bare-mesh fallback path — the two
        # PEC-information sources must yield identical results.
        assert abs(f_c0 - f_te10) / f_te10 < 0.01


class TestLateralPecEdgeMaskHelper:
    """Unit tests for the internal ``_build_lateral_pec_edge_mask`` helper."""

    def test_port_plane_face_rejected_as_lateral(self, wr90_mesh_high_res):
        """A face that shares the port plane's normal axis is not a
        lateral wall and must be rejected by the helper."""
        from magnelio.ports._modal.factory import _build_lateral_pec_edge_mask

        mesh, _, _, _ = wr90_mesh_high_res
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        with pytest.raises(
            ValueError,
            match="shares the port plane's normal axis",
        ):
            _build_lateral_pec_edge_mask(plane, mesh, (BoxFace.X_MAX,))

    def test_lateral_mask_marks_correct_edges(self, wr90_mesh_high_res):
        """Internal helper output matches the test-side _wall_pec_mask."""
        from magnelio.ports._modal.factory import _build_lateral_pec_edge_mask

        mesh, _, _, _ = wr90_mesh_high_res
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        mask = _build_lateral_pec_edge_mask(
            plane,
            mesh,
            (BoxFace.Y_MIN, BoxFace.Y_MAX, BoxFace.Z_MIN, BoxFace.Z_MAX),
        )
        # Recompute with the curl_curl_2d test's helper convention.
        eps = 1e-9 * max(WR90_A, WR90_B)
        u_v = plane.u_edge_uv[:, 1]
        v_u = plane.v_edge_uv[:, 0]
        u_pec = (np.abs(u_v) < eps) | (np.abs(u_v - WR90_B) < eps)
        v_pec = (np.abs(v_u) < eps) | (np.abs(v_u - WR90_A) < eps)
        expected = np.concatenate([u_pec, v_pec])
        np.testing.assert_array_equal(mask, expected)
