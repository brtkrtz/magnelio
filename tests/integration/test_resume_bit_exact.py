"""Bit-exact resume round-trip — the WP-S6 checkpoint gate (DD-070).

Proves that ``FITTimeDomainSolver.state_dict`` captures *every* piece of
evolving leapfrog state.  A single omitted state variable — a Mur
previous-value, the TF/SF source ring buffer, one element of the exact
DTBC convolution history, or a CPML ψ field — makes a resumed trajectory
diverge, so requiring the resume to reproduce the uninterrupted run
*exactly* is the completeness check.

Determinism note.  The FIT field update is element-wise (bit-reproducible
regardless of thread count), but building a modal port runs an eigen
mode-solve whose ARPACK start vector is random, so two *independently
built* solvers differ at ~1e-13.  To isolate the checkpoint round-trip
from that build noise, the test rewinds **the same solver** (same
operators): run to N, snapshot, continue to N+M (reference), then reload
the snapshot into those same operators and re-march.  Identical operators
+ complete state ⇒ ``max|Δ| == 0`` exactly; a missing state variable
shows up as a non-zero residual.
"""

from __future__ import annotations

import numpy as np

from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.boundaries.cpml import CPMLBoundary
from magnelio.boundaries.pec import PECBoundary
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.ports._modal import (
    BoxFace,
    ExcitationSpec,
    PortSpecRectWG,
    build_modal_port,
)
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import courant_dt

WR90_A = 22.86e-3
WR90_B = 10.16e-3


def _wr90_solver(n_steps):
    """One straight WR-90 line, TE10 excited at port1 (built once)."""
    grid = GridLines(
        x=np.linspace(0.0, 30e-3, 31),
        y=np.linspace(0.0, WR90_A, 24),
        z=np.linspace(0.0, WR90_B, 11),
    )
    mesh = Mesh.from_grid(grid)
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    dt = courant_dt(grid, accuracy="normal")
    excitation = ExcitationSpec(f_min=8.2e9, f_max=12.4e9, mode_index=0)
    op_src = build_modal_port(
        PortSpecRectWG(
            name="port1",
            plane=BoxFace.X_MIN,
            width_a=WR90_A,
            height_b=WR90_B,
            n_modes=1,
            excitation=excitation,
        ),
        mesh,
        m_eps,
        m_mu,
        dt=dt,
        f_calc=10.0e9,
    )
    op_load = build_modal_port(
        PortSpecRectWG(
            name="port2",
            plane=BoxFace.X_MAX,
            width_a=WR90_A,
            height_b=WR90_B,
            n_modes=1,
        ),
        mesh,
        m_eps,
        m_mu,
        dt=dt,
        f_calc=10.0e9,
    )
    return FITTimeDomainSolver(
        mesh=mesh,
        dt=dt,
        total_time_steps=n_steps,
        ports=[op_src, op_load],
        boundary_conditions={
            "ymin": PECBoundary("ymin"),
            "ymax": PECBoundary("ymax"),
            "zmin": PECBoundary("zmin"),
            "zmax": PECBoundary("zmax"),
            "xmin": "PMC",
            "xmax": "PMC",
        },
        verbose=False,
    )


def _rewind_roundtrip(solver, n_half, n_total):
    """Run to n_half, snapshot, continue to n_total (reference), then reload
    the snapshot into the same operators and re-march; return (ref, resumed)
    final (e, h) pairs."""
    solver.total_time_steps = n_half
    solver.run()  # 0 -> n_half
    snapshot = solver.state_dict()
    solver.total_time_steps = n_total
    solver.run()  # n_half -> n_total (uninterrupted)
    ref = (solver._fields.e_flat.copy(), solver._fields.h_flat.copy())

    solver.load_state_dict(snapshot)  # rewind to n_half (same operators)
    solver.total_time_steps = n_total
    solver.run()  # replay n_half -> n_total
    resumed = (solver._fields.e_flat.copy(), solver._fields.h_flat.copy())
    return snapshot, ref, resumed


def test_resume_bit_exact_modal_dtbc_port():
    """DTBC convolution history + Mur previous-values + TF/SF buffer."""
    solver = _wr90_solver(n_steps=260)
    snapshot, (ref_e, ref_h), (res_e, res_h) = _rewind_roundtrip(
        solver,
        n_half=120,
        n_total=260,
    )

    assert snapshot["n_completed"] == 120
    # the scenario must actually exercise the hard reflection-free state:
    # both ports carry a full-length exact DTBC convolution history and
    # the fields are non-trivial at the checkpoint.
    assert snapshot["ports"]["port1"]["dtbc"]["0"]["n"] == 120
    assert snapshot["ports"]["port2"]["dtbc"]["0"]["n"] == 120
    assert float(np.abs(snapshot["e"]).max()) > 0.0

    assert np.array_equal(ref_e, res_e), (
        f"E not bit-exact after resume: max|Δ| = {float(np.max(np.abs(ref_e - res_e))):.3e}"
    )
    assert np.array_equal(ref_h, res_h), (
        f"H not bit-exact after resume: max|Δ| = {float(np.max(np.abs(ref_h - res_h))):.3e}"
    )


def test_resume_bit_exact_cpml_psi():
    """CPML ψ auxiliary convolution fields survive a checkpoint round-trip."""
    grid = GridLines(
        x=np.linspace(0.0, 30e-3, 24),
        y=np.linspace(0.0, 30e-3, 24),
        z=np.linspace(0.0, 30e-3, 24),
    )
    mesh = Mesh.from_grid(grid)
    dt = courant_dt(grid, accuracy="normal")
    solver = FITTimeDomainSolver(
        mesh=mesh,
        dt=dt,
        total_time_steps=60,
        ports=[],
        boundary_conditions={
            f: CPMLBoundary(f, grid, thickness_cells=6)
            for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
        },
        verbose=False,
    )
    # A Gaussian E_z bump at the box centre radiates into the six PML walls,
    # so the ψ convolution fields carry non-trivial state at the checkpoint.
    solver.setup()
    Ez = solver._fields.Ez
    nx, ny, nz = Ez.shape
    xs, ys, zs = (
        np.arange(nx)[:, None, None],
        np.arange(ny)[None, :, None],
        np.arange(nz)[None, None, :],
    )
    r2 = (xs - nx / 2) ** 2 + (ys - ny / 2) ** 2 + (zs - nz / 2) ** 2
    solver._fields.Ez = np.exp(-r2 / 6.0)

    snapshot, (ref_e, ref_h), (res_e, res_h) = _rewind_roundtrip(
        solver,
        n_half=60,
        n_total=140,
    )

    assert any(k.startswith("_psi_") for k in snapshot["boundaries"]["xmin"])
    assert np.array_equal(ref_e, res_e)
    assert np.array_equal(ref_h, res_h)
