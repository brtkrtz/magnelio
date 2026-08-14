"""Integration: CW true-mode port floor on an inhomogeneous line.

WP-R4a acceptance criterion, scaled down for CI: a short half-filled
layered parallel plate driven CW at one frequency through the full
production chain (``build_cw_true_mode_port`` -> ``FITTimeDomainSolver``
-> ``PortSignalRecorder`` -> ``cw_lockin_phasors`` / ``cw_decompose``).
The benchmark (``validation/qtem_cw_dtbc_port_floors.py``)
measures -208 dB on the full-size line; the bound here is generous.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy.special import erf

from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.geo import Brick, GeometryModel
from magnelio.materials.material import Material
from magnelio.mesh.mesher import Mesh, MeshControl
from magnelio.ports._modal import (
    BoxFace,
    PortSpecMultiConductor,
    build_cw_true_mode_port,
    cw_decompose,
    cw_lockin_phasors,
)
from magnelio.ports.recorder import PortSignalRecorder
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import courant_dt


def _segments(*breaks_and_counts):
    out = []
    for lo, hi, n in breaks_and_counts:
        seg = np.linspace(lo, hi, n + 1)
        out.extend(seg if not out else seg[1:])
    return [float(v) for v in out]


@pytest.fixture(scope="module")
def layered_line():
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
    return mesh, courant_dt(mesh.grid, "normal")


def test_cw_floor_and_transmission(layered_line):
    mesh, dt = layered_line
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    f = 4.2e9
    op1 = build_cw_true_mode_port(
        PortSpecMultiConductor(name="port1", plane=BoxFace.Z_MIN, epsilon_r=None),
        mesh,
        m_eps,
        m_mu,
        dt=dt,
        f_cw=f,
    )
    op2 = build_cw_true_mode_port(
        PortSpecMultiConductor(name="port2", plane=BoxFace.Z_MAX, epsilon_r=None),
        mesh,
        m_eps,
        m_mu,
        dt=dt,
        f_cw=f,
    )
    assert op1.termination_kinds == ["dtbc"]
    ch1 = op1.cw_data.channels[0]
    w_dt = op1.cw_data.w_dt

    period = 2.0 * math.pi / w_dt
    sigma = 6.0 * period
    n_win = int(20 * period)
    theta = abs(np.angle(ch1.zeta))
    v_g = ch1.r**2 * math.sin(theta) / math.sin(w_dt)
    n_steps = int(8.0 * sigma + 20.0 * period + 3.0 * mesh.Nz / max(v_g, 1e-3)) + n_win + 2

    t0 = 5.0 * sigma * dt
    sig_t = sigma * dt
    w_phys = w_dt / dt

    def waveform(t: float) -> float:
        amp = 0.5 * (1.0 + float(erf((t - t0) / (math.sqrt(2.0) * sig_t))))
        return amp * math.sin(w_phys * t)

    op1.set_excitation(0, waveform)
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

    V1, I1 = signals[("port1", 0)]
    Vp, Ip, res_fit = cw_lockin_phasors(V1.values, I1.values, w_dt, n_win)
    a1, b1 = cw_decompose(Vp, Ip, ch1)
    V2, I2 = signals[("port2", 0)]
    Vp2, Ip2, _ = cw_lockin_phasors(V2.values, I2.values, w_dt, n_win)
    a2, b2 = cw_decompose(Vp2, Ip2, op2.cw_data.channels[0])

    s11_db = 20.0 * math.log10(abs(b1 / a1))
    s21_db = 20.0 * math.log10(abs(b2 / a1))
    assert res_fit < 1e-6
    assert s11_db < -120.0
    assert abs(s21_db) < 0.05
    # Nothing re-enters at the matched output port.
    assert abs(a2 / a1) < 1e-5
