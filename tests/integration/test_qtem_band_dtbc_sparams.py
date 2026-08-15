"""Integration: broadband band-subspace DTBC port on a QTEM line.

WP-R4b-impl acceptance (DD-057), scaled down for CI: a short
half-filled layered parallel plate, ONE pulsed run through the full
production chain (``build_band_dtbc_port`` ->
``set_excitation_band`` -> ``FITTimeDomainSolver`` ->
``PortSignalRecorder`` -> ``compute_band_s_parameters``) and the
per-frequency true-mode |S11| across the measurement span.  The
benchmark (``validation/qtem_band_dtbc_port_floors.py``)
measures below -155 dB on the full-size lines; the bounds here are
generous, and the kernel grid is resolved well enough that they
actually are -- see the ``n_grid`` note below.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.geo import Brick, GeometryModel
from magnelio.materials.material import Material
from magnelio.mesh.mesher import Mesh, MeshControl
from magnelio.ports._modal import (
    BoxFace,
    PortSpecMultiConductor,
    build_band_dtbc_port,
)
from magnelio.ports.recorder import PortSignalRecorder
from magnelio.post import compute_band_s_parameters
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import courant_dt


def _segments(*breaks_and_counts):
    out = []
    for lo, hi, n in breaks_and_counts:
        seg = np.linspace(lo, hi, n + 1)
        out.extend(seg if not out else seg[1:])
    return [float(v) for v in out]


@pytest.fixture(scope="module")
def band_run():
    w, hy, h_if, nz, dz = 10.0e-3, 8.0e-3, 4.0e-3, 20, 1.0e-3
    length = nz * dz
    diel = Material(name="diel", epsilon=(4.0,) * 3)
    model = GeometryModel()
    model.add(Brick(origin=(0, 0, 0), size=(w, h_if, length), material=diel))
    model.add(Brick(origin=(0, h_if, 0), size=(w, hy - h_if, length), material=Material.air()))
    control = MeshControl(
        min_nodes_per_wavelength=4,
        min_cells_per_feature=0,
        max_cell_size=5.1e-3,
        forced_planes={
            "x": _segments((0.0, w, 2)),
            "y": _segments((0.0, h_if, 4), (h_if, hy, 4)),
            "z": _segments((0.0, length, nz)),
        },
    )
    mesh = Mesh.from_geometry(model, control, f_max=8.0e9)
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
    dt = courant_dt(mesh.grid, "normal")

    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    ops = []
    for label, face in (("port1", BoxFace.Z_MIN), ("port2", BoxFace.Z_MAX)):
        ops.append(
            build_band_dtbc_port(
                PortSpecMultiConductor(name=label, plane=face, epsilon_r=None),
                mesh,
                m_eps,
                m_mu,
                dt=dt,
                f_band=(0.3e9, 8.3e9),
                # The floor this fixture asserts is a kernel-fit
                # residual, and the fit's resolution dominates it: at
                # n_grid=9 the worst point read -120.06 dB against the
                # -120 dB bound, at 11 it reads -138.9 and at 13
                # -150.8, with individual frequency points moving up to
                # 52 dB.  Nine points left the gate defending 0.06 dB
                # of margin on an under-resolved fit rather than on the
                # port's accuracy.
                n_grid=13,
                n_kernel_init=4096,
            )
        )
    op1, op2 = ops
    op1.set_excitation_band(0, (1.8e9, 6.8e9), n_syn=3072)

    n_steps = 4064
    recorder = PortSignalRecorder(dt=dt, ports=[op1, op2])
    solver = FITTimeDomainSolver(
        mesh=mesh,
        boundary_conditions={},
        ports=[op1, op2],
        recorder=recorder,
        total_time_steps=n_steps,
        dt=dt,
        verbose=False,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*neither a BoundaryCondition.*",
        )
        solver.run()
    signals = recorder.finalize(n_steps_actual=n_steps)

    f_axis = np.array([1.8e9, 3.0e9, 4.3e9, 5.5e9, 6.8e9])
    S = compute_band_s_parameters(
        signals,
        [op1, op2],
        ("port1", 0),
        f_axis,
    )
    return signals, S, f_axis


class TestBandDTBCSParams:
    def test_s11_floor(self, band_run):
        _, S, f_axis = band_run
        s11_db = 20.0 * np.log10(np.abs(S[("port1", 0)]) + 1e-300)
        assert np.all(np.isfinite(s11_db))
        assert s11_db.max() < -120.0

    def test_s21_flat(self, band_run):
        _, S, _ = band_run
        s21_db = 20.0 * np.log10(np.abs(S[("port2", 0)]) + 1e-300)
        assert np.all(np.abs(s21_db) < 0.05)

    def test_record_decays(self, band_run):
        signals, _, _ = band_run
        v1 = signals[("port1", 0)][0].values
        assert np.abs(v1[-128:]).max() < 1e-9 * np.abs(v1).max()
