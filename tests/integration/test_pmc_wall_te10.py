"""WP-U0 stage 1 regression: TD PMC wall consistency (natural BC).

Before WP-U0 the TD ``PMCBoundary`` zeroed tangential H on the
cell-centre layer (wall dx/2 *inside*) while both mode solvers use the
natural boundary (wall dx/2 *outside*) — one full cell apart.  On this
fixture that produced a non-passive S-matrix (|S21| up to +14.6 dB
between the two cut-offs) and in-band |S11| never better than −17 dB.

Three gates on the PMC-walled parallel plate (PMC x faces, PEC y
plates, TE(1,0) ports on the z faces), see
``validation/pmc_wall_te10_port_floor.py`` for the full
measurement:

1. CW lock-in |S11| floor < −120 dB at 1.05 f_c_hat (acceptance line
   is −100 dB; measured −156.6 dB on the fine fixture).
2. Wall-convention pin: the port's discrete cut-off is
   ``c/2(a+dx)`` — the natural wall dx/2 outside each PMC face — and
   NOT ``c/2a`` (wall on the grid line) or ``c/2(a-dx)`` (the legacy
   TD wall), which differ by ∓9 % on this mesh.
3. Pulsed passivity: |S11|² + |S21|² bounded by the truncation-limited
   measurement class of the accepted all-PEC WR-90 reference (≤ 1.06
   at the band edge), max |S21| ≤ +0.1 dB — the pre-fix window showed
   +14.6 dB.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import erf

from magnelio import (
    AnalysisScatteringTD,
)
from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.boundaries import PECBoundary, PMCBoundary
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.ports import PortSpecNumerical
from magnelio.ports._modal import (
    BoxFace,
    ModeType,
    build_modal_port,
)
from magnelio.ports._modal.dtbc import destagger_theta, dtbc_wave_impedance
from magnelio.ports.recorder import PortSignalRecorder
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import courant_dt

A = 10.0e-3  # width between the PMC walls
B = 5.0e-3  # PEC plate spacing
LENGTH = 20.0e-3
DX = 1.0e-3
C0 = 299_792_458.0


def _parallel_plate():
    grid = GridLines(
        x=np.linspace(-A / 2, A / 2, 11),  # dx = 1 mm, PMC walls
        y=np.linspace(-B / 2, B / 2, 6),
        z=np.linspace(-LENGTH / 2, LENGTH / 2, 21),
    )
    mesh = Mesh.from_grid(grid).with_boundary_conditions(
        {
            "ymin": "PEC",
            "ymax": "PEC",
            "xmin": "PMC",
            "xmax": "PMC",
            "zmin": "PMC",
            "zmax": "PMC",
        }
    )
    return mesh, courant_dt(grid, "normal")


def _lateral_bcs(grid):
    return {
        "xmin": PMCBoundary("xmin", grid),
        "xmax": PMCBoundary("xmax", grid),
        "ymin": PECBoundary("ymin"),
        "ymax": PECBoundary("ymax"),
    }


def _te10_ports(mesh, dt):
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    return [
        build_modal_port(
            PortSpecNumerical(name=lbl, plane=face, n_modes=1, mode_type=ModeType.TE),
            mesh,
            m_eps,
            m_mu,
            dt=dt,
            f_calc=20.0e9,
        )
        for lbl, face in (("port1", BoxFace.Z_MIN), ("port2", BoxFace.Z_MAX))
    ]


def _cw_s11_db(mesh, dt, ratio):
    """|S11| at ``ratio * f_c_hat`` by production CW lock-in."""
    ops = _te10_ports(mesh, dt)
    assert ops[0].termination_kinds == ["dtbc"]
    r, q, z0 = ops[0].dtbc_line_params[0]

    w_dt = ratio * q
    period = 2.0 * math.pi / w_dt
    sigma = max(6.0 / ((ratio - 1.0) * q), 8.0 * period)
    s_hat = math.sin(w_dt / 2.0)
    sin_b2 = math.sqrt(max(s_hat**2 - (q / 2.0) ** 2, 1e-30)) / r
    v_g = r * r * math.sin(2.0 * math.asin(min(sin_b2, 1.0))) / math.sin(w_dt)
    n_win = int(30 * period)
    n_meas0 = int(10.0 * sigma + 40.0 * period + 3.0 * mesh.Nz / max(v_g, 1e-3))
    n_steps = n_meas0 + n_win + 2

    t0 = 5.0 * sigma * dt
    sig_t = sigma * dt
    w_phys = w_dt / dt

    def waveform(t: float) -> float:
        amp = 0.5 * (1.0 + float(erf((t - t0) / (math.sqrt(2.0) * sig_t))))
        return amp * math.sin(w_phys * t)

    ops[0].set_excitation(0, waveform)
    recorder = PortSignalRecorder(dt=dt, ports=ops)
    solver = FITTimeDomainSolver(
        mesh=mesh,
        boundary_conditions=_lateral_bcs(mesh.grid),
        ports=ops,
        recorder=recorder,
        total_time_steps=n_steps,
        dt=dt,
        verbose=False,
    )
    solver.run()
    V_sig, I_sig = recorder.finalize(n_steps_actual=n_steps)[("port1", 0)]

    n_grid = np.arange(n_meas0, n_meas0 + n_win)
    basis = np.column_stack([np.cos(w_dt * n_grid), np.sin(w_dt * n_grid)])
    cv, *_ = np.linalg.lstsq(basis, V_sig.values[n_grid], rcond=None)
    ci, *_ = np.linalg.lstsq(basis, I_sig.values[n_grid], rcond=None)
    V = cv[0] - 1j * cv[1]
    I = (ci[0] - 1j * ci[1]) * np.exp(1j * w_dt / 2.0)

    theta = destagger_theta(np.array([w_dt]), r, q)[0]
    Z = dtbc_wave_impedance(np.array([w_dt]), q, z0, ModeType.TE.value)[0]
    sz = np.sqrt(Z)
    ep, em = np.exp(theta), np.exp(-theta)
    a = (V / sz * ep + sz * I) / (ep + em)
    b = (V / sz * em - sz * I) / (ep + em)
    return 20.0 * math.log10(max(abs(b / a), 1e-300))


class TestPMCWallTE10:
    def test_cw_floor(self):
        mesh, dt = _parallel_plate()
        s11 = _cw_s11_db(mesh, dt, 1.05)
        assert s11 < -120.0, f"|S11| = {s11:.1f} dB at 1.05 f_c"

    def test_wall_convention_pin(self):
        """f_c_hat = c/2(a+dx): the natural wall dx/2 outside each
        PMC face — the convention shared by the 2D mode solver,
        EigenmodeSolver3D and (since WP-U0) the TD update."""
        mesh, dt = _parallel_plate()
        ops = _te10_ports(mesh, dt)
        _, q, _ = ops[0].dtbc_line_params[0]
        f_c_hat = q / (2.0 * math.pi * dt)
        f_natural = C0 / (2.0 * (A + DX))
        assert abs(f_c_hat - f_natural) / f_natural < 0.01, (
            f"f_c_hat = {f_c_hat / 1e9:.4f} GHz, natural wall predicts {f_natural / 1e9:.4f} GHz"
        )
        # ... and is cleanly distinguishable from the other two wall
        # positions (on the grid line / the legacy TD wall).
        for f_wrong in (C0 / (2.0 * A), C0 / (2.0 * (A - DX))):
            assert abs(f_c_hat - f_wrong) / f_wrong > 0.05

    def test_pulsed_passivity(self):
        mesh, _ = _parallel_plate()
        analysis = AnalysisScatteringTD(
            mesh=mesh.with_boundary_conditions(
                {
                    "xmin": "PMC",
                    "xmax": "PMC",
                    "ymin": "PEC",
                    "ymax": "PEC",
                    "zmin": "PEC",
                    "zmax": "PEC",
                }
            ),
            ports=[
                PortSpecNumerical(
                    name="port1", plane=BoxFace.Z_MIN, n_modes=1, mode_type=ModeType.TE
                ),
                PortSpecNumerical(
                    name="port2", plane=BoxFace.Z_MAX, n_modes=1, mode_type=ModeType.TE
                ),
            ],
            f_max=20.0e9,
            verbose=False,
        )
        report = analysis.solve_ports()["port1"]
        f_c_hat = report.cutoff_num
        f_axis = np.linspace(1.001 * f_c_hat, 20.0e9, 81)
        res = analysis.run(f_axis=f_axis)
        s11 = res.S("port1", "port1")
        s21 = res.S("port2", "port1")

        # The 1.00x f_c edge bins are truncation-limited (algebraic
        # band-edge ring-down; on this coarse mesh the first bin reads
        # ~1.6).  From 1.02 f_c upward the excess must stay within the
        # measurement class of the accepted all-PEC WR-90 reference —
        # this window fully covers the former double-cut-off defect,
        # which extended to c/2(a-dx) = 1.17 f_c at |S21| +14.6 dB.
        band02 = f_axis >= 1.02 * f_c_hat
        power = (np.abs(s11) ** 2 + np.abs(s21) ** 2)[band02]
        assert np.max(power) < 1.06, (
            f"max |S11|^2+|S21|^2 = {np.max(power):.4f} above "
            f"1.02 f_c exceeds the truncation-limited measurement class"
        )
        s21_db = 20.0 * np.log10(np.abs(s21[band02]) + 1e-300)
        assert np.max(s21_db) < 0.1, f"max |S21| = {np.max(s21_db):+.2f} dB (pre-fix: +14.6 dB)"
        band = f_axis >= 1.05 * f_c_hat
        s11_db = 20.0 * np.log10(np.abs(s11) + 1e-300)
        assert np.max(s11_db[band]) < -40.0, (
            f"pulsed |S11| above 1.05 f_c: {np.max(s11_db[band]):.1f} dB"
        )


# ----------------------------------------------------------------------
# WP-U0 stage 2 — mesher places the PMC wall ON the bbox face
# ----------------------------------------------------------------------


def _from_geometry_te10_cutoff(dx, pmc_faces):
    """f_c_hat of the TE(1,0) port on the meshed parallel-plate brick."""
    from magnelio import Material, MeshControl
    from magnelio.geo import Brick, GeometryModel

    closure = {f: "PEC" for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    closure.update({f: "PMC" for f in (pmc_faces or ())})
    model = GeometryModel(boundary_conditions=closure)
    model.add(
        Brick(
            origin=(-A / 2, -B / 2, -LENGTH / 2),
            size=(A, B, LENGTH),
            material=Material.from_isotropic(name="air", epsilon=1.0),
        )
    )
    mesh = Mesh.from_geometry(
        model,
        MeshControl(min_nodes_per_wavelength=10, max_cell_size=dx),
        f_max=40.0e9,
    )
    dt = courant_dt(mesh.grid, "normal")
    ops = _te10_ports(mesh, dt)
    _, q, _ = ops[0].dtbc_line_params[0]
    return q / (2.0 * math.pi * dt), mesh


class TestPMCWallStage2:
    def test_wall_on_bbox_face(self):
        """The natural wall (half a boundary cell outside the
        outermost line) lands exactly on the requested bbox face."""
        _, mesh = _from_geometry_te10_cutoff(0.5e-3, pmc_faces=["xmin", "xmax"])
        x = mesh.grid.x
        wall_lo = x[0] - (x[1] - x[0]) / 2.0
        wall_hi = x[-1] + (x[-1] - x[-2]) / 2.0
        assert abs(wall_lo - (-A / 2)) < 1e-12
        assert abs(wall_hi - (A / 2)) < 1e-12

    def test_te10_cutoff_second_order(self):
        """TE(1,0) cut-off converges to c/2a at O(dx^2) — the WP-U0
        stage-2 gate.  Measured: rel. error -1.14e-3 (dx = 0.5 mm)
        -> -2.71e-4 (0.25 mm), ratio 4.2; without ``pmc_faces`` the
        O(dx) half-cell bias is -4.9e-2."""
        f_ref = C0 / (2.0 * A)
        f_coarse, _ = _from_geometry_te10_cutoff(0.5e-3, pmc_faces=["xmin", "xmax"])
        f_fine, _ = _from_geometry_te10_cutoff(0.25e-3, pmc_faces=["xmin", "xmax"])
        err_coarse = abs(f_coarse - f_ref) / f_ref
        err_fine = abs(f_fine - f_ref) / f_ref
        assert err_fine < 1e-3, f"rel. error {err_fine:.2e} at 0.25 mm"
        assert err_coarse / err_fine > 2.5, (
            f"convergence ratio {err_coarse / err_fine:.2f} on dx "
            f"halving — expected ~4 (second order)"
        )

    def test_one_type_per_face(self):
        """A face cannot carry two closures (DD-103).

        This used to be a runtime check against ``pml_faces`` and
        ``pmc_faces`` overlapping.  With the closure declared as one
        map, the conflict is unrepresentable — a face key holds a
        single type, and an invalid one is rejected on construction.
        """
        from magnelio.geo import GeometryModel

        model = GeometryModel(boundary_conditions={"xmin": "CPML"})
        assert model.boundary_conditions.xmin == "CPML"
        with pytest.raises(ValueError, match="not valid"):
            GeometryModel(boundary_conditions={"xmin": "CPML+PMC"})
