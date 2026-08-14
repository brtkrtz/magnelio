"""Unit tests for the per-frequency zeta-pencil machinery (WP-R4a).

Covers the period-block extraction certificates, the frequency-local
scalar-chain fit (exactness at the drive frequency), the homogeneous
cross-check against the scalar DTBC symbol, the CW port factory and
the operator extensions (chain overrides, dual-basis projection).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from magnelio._operators.curl import build_curl_matrix
from magnelio._operators.material_matrices import (
    build_M_eps,
    build_M_mu,
    flatten_port_plane_mass,
    flatten_port_plane_pec_mask,
)
from magnelio.geo import Brick, GeometryModel
from magnelio.materials.material import Material
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh, MeshControl
from magnelio.ports._modal import (
    BoxFace,
    PortSpecMultiConductor,
    build_cw_true_mode_port,
)
from magnelio.ports._modal.dtbc import lambda_symbol
from magnelio.ports._modal.port_plane import PortPlane
from magnelio.ports._modal.zeta_pencil import (
    build_period_blocks,
    chain_fit,
    normalize_gauge,
    profile_reality,
    solve_zeta_modes,
)
from magnelio.solver.stability import courant_dt

# DD-103: the closure these fixtures always assumed.  A face
# with no BC used to evolve under the free curl operator —
# which IS the natural magnetic wall, hence "PMC".
_BC_PEC_Y = {
    "ymin": "PEC",
    "ymax": "PEC",
    "xmin": "PMC",
    "xmax": "PMC",
    "zmin": "PMC",
    "zmax": "PMC",
}

C0 = 299_792_458.0


def _segments(*breaks_and_counts):
    out = []
    for lo, hi, n in breaks_and_counts:
        seg = np.linspace(lo, hi, n + 1)
        out.extend(seg if not out else seg[1:])
    return [float(v) for v in out]


def _layered_mesh_along(prop_axis: str):
    """The ``layered`` fixture geometry with propagation along any axis.

    Axis permutation of the half-filled parallel plate: the dielectric
    layering stays transverse to the propagation axis, PEC closes the
    layer axis, PMC everything else.  The discrete operator is an exact
    permutation of the z-along original, so the period-chain spectrum
    must be identical to roundoff.
    """
    w, hy, h_if, n_len, d_len = 10.0e-3, 8.0e-3, 4.0e-3, 12, 1.0e-3
    length = n_len * d_len
    width_ax, layer_ax, prop_ax = {
        "z": ("x", "y", "z"),
        "x": ("z", "y", "x"),
        "y": ("x", "z", "y"),
    }[prop_axis]

    def vec(width, layer, prop):
        d = {width_ax: width, layer_ax: layer, prop_ax: prop}
        return (d["x"], d["y"], d["z"])

    diel = Material(name="diel", epsilon=(4.0,) * 3)
    model = GeometryModel()
    model.add(Brick(origin=vec(0, 0, 0), size=vec(w, h_if, length), material=diel))
    model.add(
        Brick(origin=vec(0, h_if, 0), size=vec(w, hy - h_if, length), material=Material.air())
    )
    control = MeshControl(
        min_nodes_per_wavelength=4,
        min_cells_per_feature=0,
        max_cell_size=5.1e-3,
        forced_planes={
            width_ax: _segments((0.0, w, 2)),
            layer_ax: _segments((0.0, h_if, 4), (h_if, hy, 4)),
            prop_ax: _segments((0.0, length, n_len)),
        },
    )
    mesh = Mesh.from_geometry(model, control, f_max=8.0e9)
    mesh = mesh.with_boundary_conditions(
        {
            f"{ax}{side}": ("PEC" if ax == layer_ax else "PMC")
            for ax in "xyz"
            for side in ("min", "max")
        }
    )
    return mesh, courant_dt(mesh.grid, "normal")


@pytest.fixture(scope="module")
def layered():
    """Half-filled parallel plate, 12 cells long (production mesh)."""
    return _layered_mesh_along("z")


def _chain_for(mesh, dt, face=BoxFace.Z_MIN):
    m_eps = flatten_port_plane_mass(build_M_eps(mesh), mesh, face)
    object.__setattr__(
        mesh,
        "pec_mask_edges",
        flatten_port_plane_pec_mask(mesh.pec_mask_edges, mesh, face),
    )
    m_mu = build_M_mu(mesh)
    plane = PortPlane.from_mesh(face, mesh)
    c_3d = build_curl_matrix(mesh.grid)
    return build_period_blocks(plane, mesh, m_eps, m_mu, c_3d, dt), plane


class TestPeriodBlocks:
    def test_homogeneous_reproduces_scalar_symbol(self):
        """Uniform vacuum line: fundamental zeta = TEM chain symbol."""
        grid = GridLines(
            x=np.array([0.0, 5.0e-3, 10.0e-3]),
            y=np.arange(7) * 1.0e-3,
            z=np.arange(13) * 1.0e-3,
        )
        mesh = Mesh.from_grid(grid, boundary_conditions=_BC_PEC_Y)
        dt = courant_dt(grid, "normal")
        chain, plane = _chain_for(mesh, dt)
        f = 5.0e9
        w_dt = 2.0 * math.pi * f * dt
        r = C0 * dt / 1.0e-3
        theta0 = 2.0 * math.asin(math.sin(w_dt / 2.0) / r)
        zs, ps = solve_zeta_modes(chain, w_dt, [np.exp(-1j * theta0)], k=6)
        lam_ref = complex(lambda_symbol((1.0 + 1e-12) * np.exp(1j * w_dt), r, 0.0))
        best = np.min(np.abs(zs - lam_ref))
        assert best < 1e-9

        pick = int(np.argmin(np.abs(zs - lam_ref)))
        r_fit, q_fit = chain_fit(complex(zs[pick]), ps[:, pick], chain, w_dt)
        assert abs(r_fit - r) < 1e-8
        assert q_fit**2 < 1e-12

    def test_fit_exact_at_drive_frequency(self, layered):
        """|lambda(r_eff, q_eff)(w) - zeta| at float noise (QTEM)."""
        mesh, dt = layered
        chain, _ = _chain_for(mesh, dt)
        f = 4.2e9
        w_dt = 2.0 * math.pi * f * dt
        theta0 = 2.0 * math.pi * f * math.sqrt(1.7) / C0 * 1.0e-3
        zs, ps = solve_zeta_modes(chain, w_dt, [np.exp(-1j * theta0)], k=6)
        prop = np.abs(np.abs(zs) - 1.0) <= 1e-6
        assert np.any(prop)
        j = int(np.where(prop)[0][0])
        phi = normalize_gauge(ps[:, j], chain.n_t)
        zeta = complex(zs[j]) if zs[j].imag < 0 else complex(np.conj(zs[j]))
        if zs[j].imag > 0:
            phi = np.conj(phi)
        r_fit, q_fit = chain_fit(zeta, phi, chain, w_dt)
        assert 0.0 < r_fit < 1.0
        lam = complex(lambda_symbol((1.0 + 1e-12) * np.exp(1j * w_dt), r_fit, q_fit))
        assert abs(lam - zeta) < 1e-9
        assert profile_reality(phi, chain.n_t) < 1e-10

    @pytest.mark.parametrize(
        "prop_axis,face",
        [("x", BoxFace.X_MIN), ("y", BoxFace.Y_MIN)],
    )
    def test_x_and_y_normal_faces_match_z_reference(self, layered, prop_axis, face):
        """KB-009: e_u/e_v carry different normal strides off z-normal.

        The rotated fixture is an exact axis permutation of the
        z-along one, so the chain must build (it used to raise
        ``RuntimeError``) and the fundamental eigenpair must agree
        with the z-normal reference to roundoff.
        """
        mesh_z, dt = layered
        chain_z, _ = _chain_for(mesh_z, dt)
        mesh_r, dt_r = _layered_mesh_along(prop_axis)
        assert dt_r == pytest.approx(dt, rel=1e-15)
        chain_r, _ = _chain_for(mesh_r, dt, face=face)

        # Same free-DOF layout, but genuinely mixed strides.
        assert chain_r.n_t == chain_z.n_t
        assert isinstance(chain_r.et_step, np.ndarray)
        assert np.unique(chain_r.et_step).size == 2

        f = 4.2e9
        w_dt = 2.0 * math.pi * f * dt
        theta0 = 2.0 * math.pi * f * math.sqrt(1.7) / C0 * 1.0e-3

        def fundamental(chain):
            zs, ps = solve_zeta_modes(chain, w_dt, [np.exp(-1j * theta0)], k=6)
            prop = np.abs(np.abs(zs) - 1.0) <= 1e-6
            assert np.any(prop)
            j = int(np.where(prop)[0][0])
            zeta = complex(zs[j]) if zs[j].imag < 0 else complex(np.conj(zs[j]))
            phi = ps[:, j] if zs[j].imag < 0 else np.conj(ps[:, j])
            return zeta, chain_fit(zeta, normalize_gauge(phi, chain.n_t), chain, w_dt)

        zeta_z, (r_z, q_z) = fundamental(chain_z)
        zeta_r, (r_r, q_r) = fundamental(chain_r)
        assert zeta_r == pytest.approx(zeta_z, abs=1e-9)
        assert r_r == pytest.approx(r_z, rel=1e-9)
        assert q_r == pytest.approx(q_z, abs=1e-9)

    def test_non_uniform_section_rejected(self):
        """z-varying material violates the invariance certificate."""
        grid = GridLines(
            x=np.array([0.0, 5.0e-3, 10.0e-3]),
            y=np.arange(7) * 1.0e-3,
            z=np.arange(13) * 1.0e-3,
        )
        mesh = Mesh.from_grid(grid, boundary_conditions=_BC_PEC_Y)
        dt = courant_dt(grid, "normal")
        m_eps = build_M_eps(mesh)
        # Perturb the eps mass on plane 2 — inside the extraction
        # rows of the period-2 blocks, so the p0=1 / p0=2 comparison
        # must disagree.  (A perturbation deeper than plane 4 is the
        # legitimate "device starts here" case and is NOT flagged.)
        n_Ex = mesh.Nx * (mesh.Ny + 1) * (mesh.Nz + 1)
        ex = m_eps[:n_Ex].reshape(mesh.Nx, mesh.Ny + 1, mesh.Nz + 1)
        ex[:, :, 2] *= 1.5
        m_eps = flatten_port_plane_mass(m_eps, mesh, BoxFace.Z_MIN)
        m_mu = build_M_mu(mesh)
        plane = PortPlane.from_mesh(BoxFace.Z_MIN, mesh)
        c_3d = build_curl_matrix(mesh.grid)
        with pytest.raises(ValueError, match="translation invariant"):
            build_period_blocks(plane, mesh, m_eps, m_mu, c_3d, dt)

    def test_too_few_cells_rejected(self):
        grid = GridLines(
            x=np.array([0.0, 5.0e-3, 10.0e-3]),
            y=np.arange(7) * 1.0e-3,
            z=np.arange(4) * 1.0e-3,
        )
        mesh = Mesh.from_grid(grid, boundary_conditions=_BC_PEC_Y)
        dt = courant_dt(grid, "normal")
        m_eps = build_M_eps(mesh)
        m_mu = build_M_mu(mesh)
        plane = PortPlane.from_mesh(BoxFace.Z_MIN, mesh)
        c_3d = build_curl_matrix(mesh.grid)
        with pytest.raises(ValueError, match="4 cells"):
            build_period_blocks(plane, mesh, m_eps, m_mu, c_3d, dt)


class TestCWPortFactory:
    def test_layered_port(self, layered):
        mesh, dt = layered
        m_eps = build_M_eps(mesh)
        m_mu = build_M_mu(mesh)
        spec = PortSpecMultiConductor(name="p1", plane=BoxFace.Z_MIN, epsilon_r=None)
        op = build_cw_true_mode_port(spec, mesh, m_eps, m_mu, dt=dt, f_cw=4.2e9)
        assert op.termination_kinds == ["dtbc"]
        data = op.cw_data
        ch = data.channels[0]
        assert abs(abs(ch.zeta) - 1.0) < 1e-9
        assert ch.zeta.imag < 0.0
        assert ch.q**2 >= 0.0
        assert 0.0 < ch.r < 1.0
        # Effective permittivity between the fillings, above the DC
        # value of the series-capacitor layered line (1.6).
        assert 1.55 < ch.eps_eff_hat < 4.0
        # Phasor pair non-degenerate: the a/b system is solvable.
        det = ch.v_in * ch.i_out - ch.v_out * ch.i_in
        assert abs(det) > 1e-3 * abs(ch.v_in * ch.i_in)
        # Reflected-wave convention: conjugate V, sign-flipped
        # conjugate I (transmission-line reverse-current sign).
        assert ch.v_out == pytest.approx(np.conj(ch.v_in))
        assert ch.i_out == pytest.approx(-np.conj(ch.i_in))
        assert data.solve_seconds > 0.0

    def test_max_face_port_matches_min_face(self, layered):
        mesh, dt = layered
        m_eps = build_M_eps(mesh)
        m_mu = build_M_mu(mesh)
        ops = [
            build_cw_true_mode_port(
                PortSpecMultiConductor(name=lbl, plane=face, epsilon_r=None),
                mesh,
                m_eps,
                m_mu,
                dt=dt,
                f_cw=4.2e9,
            )
            for lbl, face in (("p1", BoxFace.Z_MIN), ("p2", BoxFace.Z_MAX))
        ]
        c1 = ops[0].cw_data.channels[0]
        c2 = ops[1].cw_data.channels[0]
        assert c1.zeta == pytest.approx(c2.zeta, abs=1e-9)
        assert c1.r == pytest.approx(c2.r, rel=1e-9)
        assert abs(c1.v_in) == pytest.approx(abs(c2.v_in), rel=1e-9)
        assert abs(c1.i_in) == pytest.approx(abs(c2.i_in), rel=1e-9)

    def test_n_channels_demand_raises_below_cuton(self, layered):
        mesh, dt = layered
        m_eps = build_M_eps(mesh)
        m_mu = build_M_mu(mesh)
        spec = PortSpecMultiConductor(name="p1", plane=BoxFace.Z_MIN, epsilon_r=None)
        with pytest.raises(ValueError, match="propagating"):
            build_cw_true_mode_port(spec, mesh, m_eps, m_mu, dt=dt, f_cw=1.0e9, n_channels=2)


class TestOperatorExtensions:
    def test_chain_override_forces_dtbc(self, layered):
        """A QTEM mode fails the pair gate; the override certifies it."""
        from magnelio.ports._modal.factory import build_modal_port

        mesh, dt = layered
        m_eps = build_M_eps(mesh)
        m_mu = build_M_mu(mesh)
        spec = PortSpecMultiConductor(name="p1", plane=BoxFace.Z_MIN, epsilon_r=None)
        op_mur = build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=4.2e9)
        assert op_mur.termination_kinds == ["mur"]

        from magnelio.ports._modal.operator import PortOperatorModal

        op = PortOperatorModal(
            "p1",
            op_mur.plane,
            op_mur.discrete_modes,
            m_eps,
            m_mu,
            dt=dt,
            omega_calc=2.0 * math.pi * 4.2e9,
            chain_overrides={0: (0.5, 0.01)},
            calibrate=False,
        )
        assert op.termination_kinds == ["dtbc"]
        assert op.dtbc_line_params[0] == (0.5, 0.01, None)

    def test_dual_projection_removes_cross_talk(self, layered):
        """Non-orthogonal profiles: dual projection is exact."""
        from magnelio.ports._modal.factory import build_modal_port
        from magnelio.ports._modal.operator import PortOperatorModal

        mesh, dt = layered
        m_eps = build_M_eps(mesh)
        m_mu = build_M_mu(mesh)
        spec = PortSpecMultiConductor(name="p1", plane=BoxFace.Z_MIN, epsilon_r=None)
        base = build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=4.2e9)
        dm = base.discrete_modes[0]

        # Second, deliberately non-orthogonal profile: a rolled copy
        # of the fundamental, re-normalised in the W metric (a random
        # elementwise perturbation is negligible against the O(1e6)
        # M_eps-normalised profile scale and degenerates the Gram).
        from magnelio.ports._modal.discrete import DiscreteMode

        plane = base.plane
        me_u = m_eps[plane.e_u_indices]
        me_v = m_eps[plane.e_v_indices]
        e_u2 = np.roll(dm.e_u_profile, 3)
        e_v2 = np.roll(dm.e_v_profile, 3)
        n2 = math.sqrt(float(np.dot(me_u, e_u2**2) + np.dot(me_v, e_v2**2)))
        e_u2 /= n2
        e_v2 /= n2
        dm2 = DiscreteMode(
            mode=dm.mode,
            e_u_profile=e_u2,
            e_v_profile=e_v2,
            h_u_profile=dm.h_u_profile,
            h_v_profile=dm.h_v_profile,
        )
        plane = base.plane
        me_u = m_eps[plane.e_u_indices]
        me_v = m_eps[plane.e_v_indices]
        prof_u = np.stack([dm.e_u_profile, e_u2])
        prof_v = np.stack([dm.e_v_profile, e_v2])
        gram = (prof_u * me_u) @ prof_u.T + (prof_v * me_v) @ prof_v.T
        ginv = np.linalg.inv(gram)
        dual_u = ginv @ prof_u
        dual_v = ginv @ prof_v

        op = PortOperatorModal(
            "p1",
            plane,
            [dm, dm2],
            m_eps,
            m_mu,
            dt=dt,
            omega_calc=2.0 * math.pi * 4.2e9,
            dual_e_profiles=[(dual_u[0], dual_v[0]), (dual_u[1], dual_v[1])],
            calibrate=False,
        )
        e = np.zeros(m_eps.size)
        e[plane.e_u_indices] = 2.0 * dm.e_u_profile
        e[plane.e_v_indices] = 2.0 * dm.e_v_profile
        V = op.project_V(e)
        assert V[0] == pytest.approx(2.0, rel=1e-12)
        assert abs(V[1]) < 1e-10
