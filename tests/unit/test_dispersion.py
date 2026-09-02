"""Per-frequency port dispersion (DD-244): continuation, impedance, decomposition."""

from __future__ import annotations

import numpy as np
import pytest

from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.geo import Brick, GeometryModel
from magnelio.materials.material import Material
from magnelio.mesh.mesher import Mesh, MeshControl
from magnelio.ports._modal import BoxFace, PortSpecMultiConductor, build_modal_port
from magnelio.ports._modal.dispersion import decompose_power_waves, solve_port_dispersion
from magnelio.ports._modal.factory import build_port_dispersion_record
from magnelio.solver.stability import courant_dt

F_MAX = 8.0e9


def _segments(*breaks_and_counts):
    out = []
    for lo, hi, n in breaks_and_counts:
        seg = np.linspace(lo, hi, n + 1)
        out.extend(seg if not out else seg[1:])
    return [float(v) for v in out]


def _parallel_plate(eps_lower: float):
    """Half-filled parallel plate along z: PEC top/bottom, PMC sides."""
    w, hy, h_if, n_len, d_len = 10.0e-3, 8.0e-3, 4.0e-3, 12, 1.0e-3
    length = n_len * d_len
    lower = Material(name="lower", epsilon=(eps_lower,) * 3)
    model = GeometryModel()
    model.add(Brick(origin=(0, 0, 0), size=(w, h_if, length), material=lower))
    model.add(Brick(origin=(0, h_if, 0), size=(w, hy - h_if, length), material=Material.air()))
    control = MeshControl(
        min_nodes_per_wavelength=4,
        min_cells_per_feature=0,
        max_cell_size=5.1e-3,
        forced_planes={
            "x": _segments((0.0, w, 2)),
            "y": _segments((0.0, h_if, 4), (h_if, hy, 4)),
            "z": _segments((0.0, length, n_len)),
        },
    )
    mesh = Mesh.from_geometry(model, control, f_max=F_MAX)
    mesh = mesh.with_boundary_conditions(
        {
            "ymin": "PEC",
            "ymax": "PEC",
            "xmin": "PMC",
            "xmax": "PMC",
            "zmin": "PMC",
            "zmax": "PMC",
        }
    )
    return mesh


def _record(mesh):
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    dt = courant_dt(mesh.grid, "normal")
    op = build_modal_port(
        PortSpecMultiConductor(name="p", plane=BoxFace.Z_MIN, epsilon_r=None),
        mesh,
        m_eps,
        m_mu,
        dt=dt,
        f_calc=F_MAX,
    )
    return op, build_port_dispersion_record(op, mesh, m_eps, m_mu, F_MAX)


@pytest.fixture(scope="module")
def layered():
    return _record(_parallel_plate(4.0))


@pytest.fixture(scope="module")
def homogeneous():
    return _record(_parallel_plate(1.0))


def test_tracking_matches_the_full_search(layered):
    _, rec = layered
    f = np.linspace(1e9, 7e9, 25)
    full = solve_port_dispersion(rec, f, search="full")
    track = solve_port_dispersion(rec, f, search="track")
    assert full.n_full_searches == f.size
    assert track.n_full_searches == 2
    assert np.nanmax(np.abs(track.zeta - full.zeta)) < 1e-9
    assert np.nanmax(np.abs(track.z_line / full.z_line - 1.0)) < 1e-8
    assert np.nanmax(np.abs(track.z_ref / full.z_ref - 1.0)) < 1e-8


def test_homogeneous_line_reproduces_the_quasi_static_impedance(homogeneous):
    op, rec = homogeneous
    z_qs = float(op.discrete_modes[0].mode.z_line)
    d = solve_port_dispersion(rec, np.array([1e9, 3e9, 7e9]))
    assert rec.signal_nodes is not None and len(rec.signal_nodes) == 1
    # Power–current impedance from the true fields, and the channel's
    # own reference, both coincide with the Laplace line impedance on a
    # homogeneous cross-section — the discrete TEM wave is that mode.
    assert np.allclose(d.z_line[0], z_qs, rtol=1e-8)
    assert np.allclose(d.z_ref[0], z_qs, rtol=1e-8)
    assert np.allclose(d.epsilon_eff[0], 1.0, atol=1e-6)
    assert np.all(np.abs(d.alpha[0]) < 1e-6)


def test_layered_line_disperses_physically(layered):
    op, rec = layered
    z_qs = float(op.discrete_modes[0].mode.z_line)
    eps_qs = float(op.discrete_modes[0].mode.epsilon_r)
    f = np.array([0.2e9, 1e9, 3e9, 5e9, 7e9])
    d = solve_port_dispersion(rec, f)
    assert d.assigned.all()
    # Static limit: the power–current impedance and ε_eff of the true
    # mode meet the quasi-static Laplace values.
    assert abs(d.z_line[0, 0] / z_qs - 1.0) < 3e-3
    assert abs(d.epsilon_eff[0, 0] / eps_qs - 1.0) < 3e-3
    # Normal dispersion: ε_eff rises monotonically toward the filling.
    assert np.all(np.diff(d.epsilon_eff[0]) > 0.0)
    assert np.all(d.epsilon_eff[0] < 4.0)
    assert np.all(np.abs(d.alpha[0]) < 1e-6)
    # The channel's reference departs from Z_PI on an inhomogeneous
    # line: the recording profile is the quasi-static one.
    assert not np.allclose(d.z_ref[0], d.z_line[0], rtol=1e-4)


def test_decomposition_recovers_power_waves(layered):
    _, rec = layered
    f = np.array([2e9, 4e9, 6e9])
    d = solve_port_dispersion(rec, f)
    a_true = np.array([1.0 + 0.2j, 0.7 - 0.1j, -0.4 + 0.9j])
    b_true = np.array([0.05 - 0.02j, 0.3 + 0.1j, 0.01 + 0.0j])
    scale = 1.0 / np.sqrt(d.power[0])
    V = a_true * d.v_in[:, 0, 0] * scale + b_true * d.v_out[:, 0, 0] * scale
    I = a_true * d.i_in[:, 0, 0] * scale + b_true * d.i_out[:, 0, 0] * scale
    a, b = decompose_power_waves(d, V[None, :], I[None, :])
    assert np.allclose(a[0], a_true, rtol=1e-10, atol=1e-12)
    assert np.allclose(b[0], b_true, rtol=1e-10, atol=1e-12)


def test_axis_order_is_immaterial(layered):
    _, rec = layered
    f = np.array([5e9, 1e9, 3e9])
    d = solve_port_dispersion(rec, f)
    d_sorted = solve_port_dispersion(rec, np.sort(f))
    for k, fk in enumerate(f):
        j = int(np.argmin(np.abs(np.sort(f) - fk)))
        assert abs(d.zeta[0, k] - d_sorted.zeta[0, j]) < 1e-9
        assert abs(d.z_line[0, k] - d_sorted.z_line[0, j]) < 1e-8


def test_gamma_is_the_log_of_zeta(layered):
    _, rec = layered
    d = solve_port_dispersion(rec, np.array([2e9]))
    assert np.isclose(d.beta[0, 0], -np.angle(d.zeta[0, 0]) / d.dz)
    assert d.beta[0, 0] > 0.0


def test_rejects_bad_axes(layered):
    _, rec = layered
    with pytest.raises(ValueError, match="positive"):
        solve_port_dispersion(rec, np.array([0.0, 1e9]))
    with pytest.raises(ValueError, match="search"):
        solve_port_dispersion(rec, np.array([1e9]), search="guess")
