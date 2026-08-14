"""Tests for magnelio.ports.recorder — unified V/I signal recorder."""

from __future__ import annotations

import math

import numpy as np
import pytest

from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.ports._modal import (
    BoxFace,
    CoaxAnalyticalModeSolver,
    PortOperatorModal,
    PortPlane,
    RectWGAnalyticalModeSolver,
    discretize_modes,
)
from magnelio.ports.recorder import PortSignalRecorder
from magnelio.signals.signal_1d import Signal1D
from magnelio.solver.stability import courant_dt

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

WR90_A = 22.86e-3
WR90_B = 10.16e-3


def _wr90_setup(n_modes: int = 1, f_calc: float = 10e9):
    """Build a small WR-90 box with a port operator on X_MIN."""
    grid = GridLines(
        x=np.linspace(0.0, 30e-3, 6),
        y=np.linspace(0.0, WR90_A, 15),
        z=np.linspace(0.0, WR90_B, 8),
    )
    mesh = Mesh.from_grid(grid)
    plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    modes = RectWGAnalyticalModeSolver(
        width_a=WR90_A,
        height_b=WR90_B,
    ).solve(n_modes=n_modes, f_calc=f_calc)
    discrete = discretize_modes(modes, plane, m_eps)
    dt = courant_dt(grid, accuracy="normal")
    op = PortOperatorModal(
        name="port1",
        plane=plane,
        discrete_modes=discrete,
        m_eps_flat=m_eps,
        m_mu_flat=m_mu,
        dt=dt,
        omega_calc=2 * math.pi * f_calc,
    )
    return mesh, plane, op, dt, m_eps, m_mu


def _n_H(mesh: Mesh) -> int:
    """Total flat-H length for a given mesh."""
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    return (Nx + 1) * Ny * Nz + Nx * (Ny + 1) * Nz + Nx * Ny * (Nz + 1)


def _coax_port_on(face: BoxFace, label: str, mesh: Mesh, dt: float, m_eps, m_mu, n_modes: int = 1):
    """Build a coaxial-port operator on the given bbox face."""
    plane = PortPlane.from_mesh(face, mesh)
    L_yz = float(mesh.grid.y[-1])
    modes = CoaxAnalyticalModeSolver(
        inner_radius=0.3e-3,
        outer_radius=1.0e-3,
        center=(L_yz / 2, L_yz / 2),
    ).solve(n_modes=n_modes)
    discrete = discretize_modes(modes, plane, m_eps)
    op = PortOperatorModal(
        name=label,
        plane=plane,
        discrete_modes=discrete,
        m_eps_flat=m_eps,
        m_mu_flat=m_mu,
        dt=dt,
        omega_calc=2 * math.pi * 10e9,
    )
    return op


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


class TestPortSignalRecorderConstruction:
    def test_attributes_populated(self):
        _, _, op, dt, _, _ = _wr90_setup()
        rec = PortSignalRecorder(dt=dt, ports=[op])
        assert rec.n_steps_recorded == 0
        assert rec.channels == [("port1", 0)]

    def test_channels_for_multi_mode_port(self):
        _, _, op, dt, _, _ = _wr90_setup(n_modes=3, f_calc=12e9)
        rec = PortSignalRecorder(dt=dt, ports=[op])
        assert rec.channels == [("port1", 0), ("port1", 1), ("port1", 2)]

    def test_channels_for_multi_port(self):
        # Build a coax box with two ports on opposing faces.
        L_x = 30e-3
        L_yz = 3e-3
        grid = GridLines(
            x=np.linspace(0.0, L_x, 21),
            y=np.linspace(0.0, L_yz, 13),
            z=np.linspace(0.0, L_yz, 13),
        )
        mesh = Mesh.from_grid(grid)
        m_eps = build_M_eps(mesh)
        m_mu = build_M_mu(mesh)
        dt = courant_dt(grid, accuracy="normal")
        op_min = _coax_port_on(BoxFace.X_MIN, "p1", mesh, dt, m_eps, m_mu)
        op_max = _coax_port_on(BoxFace.X_MAX, "p2", mesh, dt, m_eps, m_mu)
        rec = PortSignalRecorder(dt=dt, ports=[op_min, op_max])
        assert rec.channels == [("p1", 0), ("p2", 0)]

    def test_rejects_zero_dt(self):
        _, _, op, _, _, _ = _wr90_setup()
        with pytest.raises(ValueError, match="dt"):
            PortSignalRecorder(dt=0.0, ports=[op])

    def test_rejects_empty_ports(self):
        with pytest.raises(ValueError, match="non-empty"):
            PortSignalRecorder(dt=1e-12, ports=[])

    def test_rejects_duplicate_names(self):
        _, _, op1, dt, _, _ = _wr90_setup()
        _, _, op2, _, _, _ = _wr90_setup()
        # both have name "port1"
        with pytest.raises(ValueError, match="unique"):
            PortSignalRecorder(dt=dt, ports=[op1, op2])


# ----------------------------------------------------------------------
# Recording behaviour
# ----------------------------------------------------------------------


class TestRecordStep:
    def test_record_increments_step_count(self):
        mesh, _, op, dt, m_eps, _ = _wr90_setup()
        rec = PortSignalRecorder(dt=dt, ports=[op])
        e = np.zeros_like(m_eps)
        h = np.zeros(_n_H(mesh))
        for k in range(5):
            rec.record(e, h)
            assert rec.n_steps_recorded == k + 1

    def test_record_zero_field_gives_zero_buffers(self):
        mesh, _, op, dt, m_eps, _ = _wr90_setup()
        rec = PortSignalRecorder(dt=dt, ports=[op])
        e = np.zeros_like(m_eps)
        h = np.zeros(_n_H(mesh))
        for _ in range(7):
            rec.record(e, h)
        signals = rec.finalize()
        V_sig, I_sig = signals[("port1", 0)]
        assert V_sig.values.shape == (7,)
        assert I_sig.values.shape == (7,)
        np.testing.assert_array_equal(V_sig.values, 0.0)
        np.testing.assert_array_equal(I_sig.values, 0.0)

    def test_record_picks_up_modal_e_field(self):
        """E set to ê_m profile records V_m = record_scale (κ), I_m = 0."""
        mesh, plane, op, dt, m_eps, _ = _wr90_setup()
        e = np.zeros_like(m_eps)
        # Inject the ê_0 profile at port-plane edges; orthonormalisation
        # in M_ε guarantees ⟨ê_0, ê_0⟩_Mε = 1, and the recorder scales
        # the unit coefficient to physical units by κ (DD-078).
        dm = op.discrete_modes[0]
        e[plane.e_u_indices] = dm.e_u_profile
        e[plane.e_v_indices] = dm.e_v_profile
        h = np.zeros(_n_H(mesh))
        rec = PortSignalRecorder(dt=dt, ports=[op])
        rec.record(e, h)
        signals = rec.finalize()
        V_sig, I_sig = signals[("port1", 0)]
        kappa = float(op.record_scale[0])
        assert kappa > 0.0
        assert V_sig.values[0] == pytest.approx(kappa, rel=1e-10)
        assert I_sig.values[0] == pytest.approx(0.0, abs=1e-12 * kappa)


# ----------------------------------------------------------------------
# Finalize
# ----------------------------------------------------------------------


class TestFinalize:
    def test_returns_signal1d_pair_per_channel(self):
        _, _, op, dt, _, _ = _wr90_setup(n_modes=2, f_calc=12e9)
        rec = PortSignalRecorder(dt=dt, ports=[op])
        signals = rec.finalize()
        assert set(signals.keys()) == {("port1", 0), ("port1", 1)}
        for V_sig, I_sig in signals.values():
            assert isinstance(V_sig, Signal1D)
            assert isinstance(I_sig, Signal1D)

    def test_time_axis_uses_solver_dt(self):
        mesh, _, op, dt, m_eps, _ = _wr90_setup()
        rec = PortSignalRecorder(dt=dt, ports=[op])
        e = np.zeros_like(m_eps)
        h = np.zeros(_n_H(mesh))
        for _ in range(4):
            rec.record(e, h)
        signals = rec.finalize()
        V_sig, _ = signals[("port1", 0)]
        np.testing.assert_allclose(V_sig.t, np.arange(4) * dt)
        assert V_sig.dt == pytest.approx(dt)

    def test_labels_carry_port_and_mode(self):
        _, _, op, dt, _, _ = _wr90_setup(n_modes=2, f_calc=12e9)
        rec = PortSignalRecorder(dt=dt, ports=[op])
        signals = rec.finalize()
        V_sig0, I_sig0 = signals[("port1", 0)]
        V_sig1, I_sig1 = signals[("port1", 1)]
        assert V_sig0.label == "port1_mode0_V"
        assert I_sig0.label == "port1_mode0_I"
        assert V_sig1.label == "port1_mode1_V"
        assert I_sig1.label == "port1_mode1_I"

    def test_multi_port_channels_independent(self):
        L_x = 30e-3
        L_yz = 3e-3
        grid = GridLines(
            x=np.linspace(0.0, L_x, 21),
            y=np.linspace(0.0, L_yz, 13),
            z=np.linspace(0.0, L_yz, 13),
        )
        mesh = Mesh.from_grid(grid)
        m_eps = build_M_eps(mesh)
        m_mu = build_M_mu(mesh)
        dt = courant_dt(grid, accuracy="normal")
        op_min = _coax_port_on(BoxFace.X_MIN, "p1", mesh, dt, m_eps, m_mu)
        op_max = _coax_port_on(BoxFace.X_MAX, "p2", mesh, dt, m_eps, m_mu)
        rec = PortSignalRecorder(dt=dt, ports=[op_min, op_max])
        # Inject only at port p1: V should be non-zero on p1, zero on p2.
        e = np.zeros_like(m_eps)
        plane_min = op_min.plane
        dm_min = op_min.discrete_modes[0]
        e[plane_min.e_u_indices] = dm_min.e_u_profile
        e[plane_min.e_v_indices] = dm_min.e_v_profile
        h = np.zeros(_n_H(mesh))
        rec.record(e, h)
        sigs = rec.finalize()
        kappa = float(op_min.record_scale[0])
        assert sigs[("p1", 0)][0].values[0] == pytest.approx(kappa, rel=1e-10)
        assert sigs[("p2", 0)][0].values[0] == pytest.approx(0.0, abs=1e-12 * kappa)


# ----------------------------------------------------------------------
# Device staging (WP-G1)
# ----------------------------------------------------------------------


class _FakeDeviceArray(np.ndarray):
    """ndarray subclass with a CuPy-style ``.get()``.

    Lets the unit suite drive the recorder's device-staging machinery
    (ring buffer, drain on tail/finalize/full) without a CUDA device:
    ``hasattr(e, "get")`` selects the staged path, while all arithmetic
    stays NumPy — so the staged result must be *bit-identical* to the
    immediate path on the same values.
    """

    def get(self) -> np.ndarray:
        return np.asarray(self).copy()


def _as_fake(a: np.ndarray) -> _FakeDeviceArray:
    return np.ascontiguousarray(a).view(_FakeDeviceArray)


def _two_port_coax_recorder():
    """Two-port coax fixture + a fresh recorder and field-vector shapes."""
    L_x = 30e-3
    L_yz = 3e-3
    grid = GridLines(
        x=np.linspace(0.0, L_x, 21),
        y=np.linspace(0.0, L_yz, 13),
        z=np.linspace(0.0, L_yz, 13),
    )
    mesh = Mesh.from_grid(grid)
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    dt = courant_dt(grid, accuracy="normal")
    op1 = _coax_port_on(BoxFace.X_MIN, "p1", mesh, dt, m_eps, m_mu)
    op2 = _coax_port_on(BoxFace.X_MAX, "p2", mesh, dt, m_eps, m_mu)
    ports = [op1, op2]
    return (
        PortSignalRecorder(dt=dt, ports=ports),
        PortSignalRecorder(dt=dt, ports=ports),
        m_eps.size,
        _n_H(mesh),
        dt,
    )


def _random_steps(n_e, n_h, n_steps, seed=7):
    rng = np.random.default_rng(seed)
    return [(rng.standard_normal(n_e), rng.standard_normal(n_h)) for _ in range(n_steps)]


class TestDeviceStaging:
    def test_numpy_path_never_stages(self):
        rec, _, n_e, n_h, _ = _two_port_coax_recorder()
        for e, h in _random_steps(n_e, n_h, 3):
            rec.record(e, h)
        assert rec._staged is False
        assert all(s is None for s in rec._stage_list)
        assert len(rec._V_buffers[("p1", 0)]) == 3

    def test_staged_bit_identical_and_tail_drains(self):
        rec_staged, rec_ref, n_e, n_h, _ = _two_port_coax_recorder()
        steps = _random_steps(n_e, n_h, 10)
        for e, h in steps:
            rec_staged.record(_as_fake(e), _as_fake(h))
            rec_ref.record(e, h)
        # Samples really were staged, not recorded per step.
        assert rec_staged._staged is True
        assert len(rec_staged._V_buffers[("p1", 0)]) == 0
        assert rec_staged.n_steps_recorded == 10
        # tail() drains and returns exactly the reference samples.
        t_staged = rec_staged.tail(0)
        t_ref = rec_ref.tail(0)
        for key in t_ref:
            assert np.array_equal(t_staged[key][0], t_ref[key][0])
            assert np.array_equal(t_staged[key][1], t_ref[key][1])
        # Continue past the drain seam: tail(start) serves the rest.
        for e, h in _random_steps(n_e, n_h, 4, seed=11):
            rec_staged.record(_as_fake(e), _as_fake(h))
            rec_ref.record(e, h)
        t_staged = rec_staged.tail(10)
        t_ref = rec_ref.tail(10)
        for key in t_ref:
            assert t_staged[key][0].shape == (4,)
            assert np.array_equal(t_staged[key][0], t_ref[key][0])
            assert np.array_equal(t_staged[key][1], t_ref[key][1])
        # finalize after mixed drains stays bit-identical.
        sig_staged = rec_staged.finalize()
        sig_ref = rec_ref.finalize()
        for key in sig_ref:
            assert np.array_equal(sig_staged[key][0].values, sig_ref[key][0].values)
            assert np.array_equal(sig_staged[key][1].values, sig_ref[key][1].values)

    def test_drain_on_buffer_full(self):
        rec_staged, rec_ref, n_e, n_h, _ = _two_port_coax_recorder()
        steps = _random_steps(n_e, n_h, 11, seed=3)
        e0, h0 = steps[0]
        rec_staged.record(_as_fake(e0), _as_fake(h0))
        rec_ref.record(e0, h0)
        for stage in rec_staged._stage_list:
            assert stage is not None
            stage.capacity = 4  # force frequent full-drains
        for e, h in steps[1:]:
            rec_staged.record(_as_fake(e), _as_fake(h))
            rec_ref.record(e, h)
        # 11 samples, capacity 4: at least two forced drains happened
        # and the host lists carry the drained prefix already.
        assert len(rec_staged._V_buffers[("p1", 0)]) >= 8
        sig_staged = rec_staged.finalize()
        sig_ref = rec_ref.finalize()
        for key in sig_ref:
            assert sig_staged[key][0].values.shape == (11,)
            assert np.array_equal(sig_staged[key][0].values, sig_ref[key][0].values)
            assert np.array_equal(sig_staged[key][1].values, sig_ref[key][1].values)

    def test_finalize_trims_staged_samples(self):
        rec_staged, rec_ref, n_e, n_h, _ = _two_port_coax_recorder()
        for e, h in _random_steps(n_e, n_h, 9, seed=5):
            rec_staged.record(_as_fake(e), _as_fake(h))
            rec_ref.record(e, h)
        sig_staged = rec_staged.finalize(n_steps_actual=5)
        sig_ref = rec_ref.finalize(n_steps_actual=5)
        for key in sig_ref:
            assert sig_staged[key][0].values.shape == (5,)
            assert np.array_equal(sig_staged[key][0].values, sig_ref[key][0].values)
            assert np.array_equal(sig_staged[key][1].values, sig_ref[key][1].values)

    def test_mixed_stageable_and_immediate_ports(self):
        class _StubPort:
            n_modes = 1

            def __init__(self, name, idx_e, idx_h):
                self.name = name
                self._ie = np.asarray(idx_e)
                self._ih = np.asarray(idx_h)

            def _host(self, x):
                return x.get() if hasattr(x, "get") else x

            def project_V(self, e):
                return np.array([float(self._host(e[self._ie]).sum())])

            def project_I(self, h):
                return np.array([float(self._host(h[self._ih]).sum())])

        class _StagedStubPort(_StubPort):
            @property
            def record_gather_indices(self):
                return self._ie, self._ih

            def project_V_samples(self, e_samples):
                return np.array([float(np.asarray(e_samples).sum())])

            def project_I_samples(self, h_samples):
                return np.array([float(np.asarray(h_samples).sum())])

        staged_p = _StagedStubPort("staged", [0, 2, 4], [1, 3])
        plain_p = _StubPort("plain", [1, 5], [0, 2])
        rec = PortSignalRecorder(dt=1e-12, ports=[staged_p, plain_p])
        rng = np.random.default_rng(9)
        ref_V = {"staged": [], "plain": []}
        for _ in range(6):
            e = rng.standard_normal(8)
            h = rng.standard_normal(8)
            ref_V["staged"].append(e[[0, 2, 4]].sum())
            ref_V["plain"].append(e[[1, 5]].sum())
            rec.record(_as_fake(e), _as_fake(h))
        # The plain port records immediately, the staged one defers.
        assert len(rec._V_buffers[("plain", 0)]) == 6
        assert len(rec._V_buffers[("staged", 0)]) == 0
        sigs = rec.finalize()
        assert np.array_equal(sigs[("staged", 0)][0].values, np.asarray(ref_V["staged"]))
        assert np.array_equal(sigs[("plain", 0)][0].values, np.asarray(ref_V["plain"]))
