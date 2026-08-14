"""WP-G2 gates — fused port-plane transfers of ``PortOperatorModal``.

On a device-array backend the operator takes ONE concatenated gather
per projection and ONE concatenated scatter for the port-plane
write-back instead of per-array round trips.  A gather-then-split (or
concat-then-scatter on disjoint indices) moves the identical float64
values, and the host dot products are shared code — so the fused path
must be *bit-identical* to the unfused CPU path on the same field
vector.  These gates drive the fused branch without a CUDA device via
an ``np.ndarray`` subclass with a CuPy-style ``.get()`` (the same
trick as the WP-G1 recorder staging gates); the real-GPU end-to-end
coverage lives in ``tests/integration/test_gpu_backend.py``.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.ports._modal import (
    BoxFace,
    PortOperatorModal,
    PortPlane,
    RectWGAnalyticalModeSolver,
    discretize_modes,
)
from magnelio.solver.stability import courant_dt

WR90_A = 22.86e-3
WR90_B = 10.16e-3


class _FakeDeviceArray(np.ndarray):
    """ndarray subclass with a CuPy-style ``.get()`` (selects the
    fused-transfer branch while all arithmetic stays NumPy)."""

    def get(self) -> np.ndarray:
        return np.asarray(self).copy()


def _as_fake(a: np.ndarray) -> _FakeDeviceArray:
    return np.ascontiguousarray(a).view(_FakeDeviceArray)


def _wr90_op(n_modes: int = 2, f_calc: float = 12e9):
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
    n_H = (
        (mesh.Nx + 1) * mesh.Ny * mesh.Nz
        + mesh.Nx * (mesh.Ny + 1) * mesh.Nz
        + mesh.Nx * mesh.Ny * (mesh.Nz + 1)
    )
    return op, m_eps.size, n_H, dt


class TestFusedProjections:
    def test_project_v_fused_bit_identical(self):
        op, n_e, _, _ = _wr90_op()
        rng = np.random.default_rng(2)
        for _ in range(3):
            e = rng.standard_normal(n_e)
            assert np.array_equal(op.project_V(_as_fake(e)), op.project_V(e))

    def test_project_v_interior_fused_bit_identical(self):
        op, n_e, _, _ = _wr90_op()
        rng = np.random.default_rng(3)
        for _ in range(3):
            e = rng.standard_normal(n_e)
            assert np.array_equal(op.project_V_interior(_as_fake(e)), op.project_V_interior(e))

    def test_project_i_fused_bit_identical(self):
        op, _, n_h, _ = _wr90_op()
        rng = np.random.default_rng(4)
        for _ in range(3):
            h = rng.standard_normal(n_h)
            assert np.array_equal(op.project_I(_as_fake(h)), op.project_I(h))

    def test_fused_indices_cached_once(self):
        op, n_e, _, _ = _wr90_op()
        e = np.zeros(n_e)
        op.project_V(_as_fake(e))
        idx = op._dev_idx
        assert idx is not None
        op.project_V_interior(_as_fake(e))
        assert op._dev_idx is idx  # no rebuild per call


class TestFusedWriteBack:
    def test_update_e_fused_bit_identical(self):
        """A full ``update_e`` on the fused branch (fake device array)
        must leave ``e`` and the operator's Mur state bit-identical to
        the unfused CPU branch on the same input — the write-back
        scatter of the concatenated block writes the same values to the
        same slots."""
        op_fused, n_e, n_h, dt = _wr90_op()
        op_plain, _, _, _ = _wr90_op()
        rng = np.random.default_rng(5)
        e0 = rng.standard_normal(n_e)
        h0 = rng.standard_normal(n_h)

        # ``update_e`` touches only ``fields.e_flat`` — a namespace
        # stand-in keeps the gate free of FieldState plumbing.
        e_fused = _as_fake(e0.copy())
        op_fused.update_e(
            SimpleNamespace(e_flat=e_fused, h_flat=_as_fake(h0.copy())),
            t=dt,
            dt=dt,
        )

        e_plain = e0.copy()
        op_plain.update_e(
            SimpleNamespace(e_flat=e_plain, h_flat=h0.copy()),
            t=dt,
            dt=dt,
        )

        assert np.array_equal(np.asarray(e_fused), e_plain)
        assert np.array_equal(op_fused._V_port_prev, op_plain._V_port_prev)
        assert np.array_equal(op_fused._V_interior_prev, op_plain._V_interior_prev)

    def test_multi_step_march_bit_identical(self):
        """Several coupled steps (projection feeds the Mur recursion
        feeds the write-back) stay bit-identical along the whole path."""
        op_fused, n_e, n_h, dt = _wr90_op()
        op_plain, _, _, _ = _wr90_op()
        rng = np.random.default_rng(6)
        e = rng.standard_normal(n_e)
        h = rng.standard_normal(n_h)
        e_f, h_f = e.copy(), h.copy()
        for k in range(5):
            # emulate an (arbitrary, identical) field change per step
            bump = rng.standard_normal(n_e) * 0.1
            e += bump
            e_f += bump
            op_plain.update_e(SimpleNamespace(e_flat=e, h_flat=h), (k + 1) * dt, dt)
            op_fused.update_e(
                SimpleNamespace(e_flat=_as_fake(e_f), h_flat=_as_fake(h_f)),
                (k + 1) * dt,
                dt,
            )
            assert np.array_equal(e_f, e)
