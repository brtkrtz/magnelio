"""DD-096: Mur complement absorber + port-signal stop criterion.

Fixture: the DD-056 half-filled layered parallel plate (inhomogeneous
cross-section, so the DD-064 modal default falls back to Mur on every
channel) — the WP-M0 minimal reproduction of the late-time trapped-
family growth (internal dossier investigations/mur_stability/).  Pre-fix, the boundary-
closed step operator carried |lambda| up to 1 + 8.6e-5 per step
(measured AND reproduced by the exact companion spectrum); with the
complement absorber every port-coupled family decays and the port
V-tails collapse to the machine floor.

The port-signal stop criterion is exercised on an unbounded run: the
stored energy plateaus on TM-cut-off (k_z = 0) cavity content that no
port-plane scheme can reach (zero tangential E), so ``energy_stop_db``
alone would never fire — the |V|-envelope criterion terminates.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.geo import Brick, GeometryModel
from magnelio.materials.material import Material
from magnelio.mesh.mesher import Mesh, MeshControl
from magnelio.ports import PortWaveguide
from magnelio.ports._modal.factory import ExcitationSpec, build_modal_port
from magnelio.ports.declarative import resolve_declarative_port
from magnelio.ports.recorder import PortSignalRecorder
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import (
    compute_min_effective_eps,
    compute_min_effective_mu,
    courant_dt,
)

F_CALC = 13.0e9


def _forced(lo, hi, n):
    return np.linspace(lo, hi, n + 1)


def _layered_mesh(nz: int, eps_lower: float) -> Mesh:
    w, hy, h_if, dz = 10.0e-3, 8.0e-3, 4.0e-3, 1.0e-3
    length = nz * dz
    model = GeometryModel()
    model.add(
        Brick(
            origin=(0, 0, 0),
            size=(w, h_if, length),
            material=Material(name="diel", epsilon=(eps_lower,) * 3),
        )
    )
    model.add(Brick(origin=(0, h_if, 0), size=(w, hy - h_if, length), material=Material.air()))
    control = MeshControl(
        min_nodes_per_wavelength=4,
        min_cells_per_feature=0,
        max_cell_size=5.1e-3,
        forced_planes={
            "x": _forced(0.0, w, 2),
            "y": np.concatenate(
                [
                    _forced(0.0, h_if, 4),
                    _forced(h_if, hy, 4)[1:],
                ]
            ),
            "z": _forced(0.0, length, nz),
        },
    )
    mesh = Mesh.from_geometry(model, control, f_max=8.0e9)
    return mesh.with_boundary_conditions(
        {
            "ymin": "PEC",
            "ymax": "PEC",
            "zmin": "PEC",
            "zmax": "PEC",
            "xmin": "PMC",
            "xmax": "PMC",
        }
    )


def _build_ports(mesh, eps_lower: float, excite_mode: int | None):
    m_eps, m_mu = build_M_eps(mesh), build_M_mu(mesh)
    dt = courant_dt(
        mesh.grid,
        "normal",
        min_effective_eps=compute_min_effective_eps(mesh),
        min_effective_mu=compute_min_effective_mu(mesh),
    )
    ops = []
    for label, plane in (("p1", "zmin"), ("p2", "zmax")):
        spec = resolve_declarative_port(
            PortWaveguide(name=label, plane=plane, n_modes=2),
            mesh,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ops.append(build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=F_CALC))
    if excite_mode is not None:
        f_c = getattr(ops[0].discrete_modes[excite_mode].mode, "omega_c", 0.0) / (2.0 * math.pi)
        exc = ExcitationSpec(f_min=max(f_c, 1.0e9), f_max=F_CALC, waveform="modulated_gaussian")
        ops[0].set_excitation(excite_mode, exc.build_waveform())
    return ops, dt


def _run(mesh, ops, dt, **kwargs):
    solver = FITTimeDomainSolver(
        mesh=mesh,
        boundary_conditions={},
        ports=ops,
        dt=dt,
        verbose=False,
        backend="numpy",
        precision="double",
        **kwargs,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*neither a BoundaryCondition.*",
        )
        solver.run()
    return solver


class TestComplementAbsorber:
    def test_factory_configures_absorber_on_mur_fallback_port(self):
        mesh = _layered_mesh(8, 4.0)
        ops, _ = _build_ports(mesh, 4.0, None)
        for op in ops:
            assert op._comp_r_u is not None
            assert op._complement_active  # all channels on Mur
            assert np.all(op._comp_r_u <= 0.0)  # c dt < dx_n (Courant)
            assert np.all(op._comp_r_u > -1.0)

    def test_absorber_dormant_on_fully_dtbc_port(self):
        mesh = _layered_mesh(8, 1.0)  # uniform: DTBC certifies
        ops, _ = _build_ports(mesh, 1.0, None)
        for op in ops:
            assert op._comp_r_u is not None  # configured...
            assert not op._complement_active  # ...but dormant

    def test_trapped_family_no_growth_and_port_tail_at_floor(self):
        # WP-M0 strongest measured pump (nz 24: +8.6e-5/step pre-fix,
        # amplitude x13 over this horizon).  Post-fix: V tails at the
        # machine floor, stored energy free of late-time growth (the
        # remaining plateau is the port-untouchable TM-cut-off
        # content).
        n_steps = 40_000
        mesh = _layered_mesh(24, 4.0)
        ops, dt = _build_ports(mesh, 4.0, excite_mode=1)
        recorder = PortSignalRecorder(dt=dt, ports=ops)
        solver = _run(
            mesh, ops, dt, total_time_steps=n_steps, recorder=recorder, energy_check_interval=100
        )

        signals = recorder.finalize(n_steps_actual=n_steps)
        v_exc = signals[("p1", 1)][0].values
        peak = np.abs(v_exc).max()
        for (label, m), (v_sig, _) in signals.items():
            tail_rms = float(np.sqrt(np.mean(v_sig.values[-n_steps // 4 :] ** 2)))
            assert tail_rms < 1.0e-8 * peak, (label, m, tail_rms / peak)

        tr = solver._energy_trace
        e_tr = np.asarray(tr["energy"], dtype=float)
        n_tr = np.asarray(tr["step"], dtype=float)
        half = len(e_tr) // 2
        slope = np.polyfit(n_tr[half:], np.log(np.maximum(e_tr[half:], 1e-300)), 1)[0]
        assert slope < 5.0e-7  # measured plateau noise ~7e-8

    def test_complement_state_resume_roundtrip(self):
        mesh = _layered_mesh(8, 4.0)
        ops, dt = _build_ports(mesh, 4.0, excite_mode=1)
        _run(mesh, ops, dt, total_time_steps=50)
        op = ops[0]
        sd = op.state_dict()
        assert "complement" in sd
        assert np.any(sd["complement"]["port_u"] != 0.0)
        saved = {k: v.copy() for k, v in sd["complement"].items()}
        op._comp_port_prev_u[:] = 0.0
        op.load_state_dict(sd)
        np.testing.assert_array_equal(op._comp_port_prev_u, saved["port_u"])
        # Pre-DD-096 checkpoint (no complement key): absorber restarts
        # from rest instead of raising.
        sd_old = {k: v for k, v in sd.items() if k != "complement"}
        op.load_state_dict(sd_old)
        assert np.all(op._comp_port_prev_u == 0.0)


class TestSpectralRadiusGate:
    def test_no_eigenvalue_outside_unit_circle_in_trapped_band(self):
        # WP-M3 permanent gate: the exact boundary-closed one-step
        # operator of the small Mur-fallback fixture (production code
        # path, complement state included) has spectral radius <= 1 in
        # the trapped-family band.  Pre-DD-096 this held +8.6e-5-class
        # eigenvalues; the method is the WP-M1 companion assembly
        # (one production step per unit vector, deterministic).
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        nz = 8
        mesh = _layered_mesh(nz, 4.0)
        ops, dt = _build_ports(mesh, 4.0, None)
        solver = FITTimeDomainSolver(
            mesh=mesh,
            boundary_conditions={},
            ports=ops,
            total_time_steps=1,
            dt=dt,
            verbose=False,
            backend="numpy",
            precision="double",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solver.setup()

        n_e = solver._fields.e_flat.size
        n_h = solver._fields.h_flat.size
        comp_keys = (
            "_comp_int_prev_u",
            "_comp_int_prev_v",
            "_comp_port_prev_u",
            "_comp_port_prev_v",
        )
        dim = n_e + n_h + sum(4 + sum(getattr(op, k).size for k in comp_keys) for op in ops)

        def apply(x):
            solver._fields.e_flat[:] = x[:n_e]
            solver._fields.h_flat[:] = x[n_e : n_e + n_h]
            off = n_e + n_h
            for op in ops:
                op._V_port_prev[:] = x[off : off + 2]
                op._V_interior_prev[:] = x[off + 2 : off + 4]
                off += 4
                for k in comp_keys:
                    n = getattr(op, k).size
                    setattr(op, k, np.array(x[off : off + n]))
                    off += n
            solver._resume_step = 0
            solver._stop_requested = False
            solver.run()
            out = np.empty(dim)
            out[:n_e] = solver._fields.e_flat
            out[n_e : n_e + n_h] = solver._fields.h_flat
            off = n_e + n_h
            for op in ops:
                out[off : off + 2] = op._V_port_prev
                out[off + 2 : off + 4] = op._V_interior_prev
                off += 4
                for k in comp_keys:
                    n = getattr(op, k).size
                    out[off : off + n] = getattr(op, k)
                    off += n
            return out

        # Hidden state would break linearity — assert it first.
        rng = np.random.default_rng(7)
        xa, xb = rng.normal(size=dim), rng.normal(size=dim)
        lin = np.linalg.norm(apply(xa + 0.5 * xb) - apply(xa) - 0.5 * apply(xb)) / np.linalg.norm(
            apply(xa)
        )
        assert lin < 1e-12

        cols = []
        x = np.zeros(dim)
        for j in range(dim):
            x[:] = 0.0
            x[j] = 1.0
            cols.append(sp.csc_matrix(apply(x)).T)
        M = sp.hstack(cols, format="csc").astype(complex)

        worst = -1.0
        for f0 in np.linspace(13.0e9, 19.0e9, 7):
            theta = 2.0 * math.pi * f0 * dt
            sigma = (1.0 + 1.0e-4) * complex(math.cos(theta), math.sin(theta))
            lam = spla.eigs(M, k=6, sigma=sigma, return_eigenvectors=False, tol=1e-10)
            worst = max(worst, float(np.max(np.abs(lam))) - 1.0)
        # The port-untouchable TM-cut-off modes sit exactly on the
        # unit circle; anything measurably above is a regression.
        assert worst < 1.0e-10, worst


class TestPortSignalStop:
    def test_unbounded_terminates_on_signal_despite_energy_plateau(self):
        mesh = _layered_mesh(12, 4.0)
        ops, dt = _build_ports(mesh, 4.0, excite_mode=1)
        solver = _run(
            mesh,
            ops,
            dt,
            total_time_steps=None,
            port_signal_stop_db=100.0,
            energy_check_interval=200,
        )
        assert solver._actual_steps < 20_000
        tr = solver._energy_trace
        e_tr = np.asarray(tr["energy"], dtype=float)
        final_db = 10.0 * math.log10(e_tr[-1] / e_tr.max())
        # The TM-cut-off plateau keeps the energy far above a -40 dB
        # energy criterion — the signal criterion is what terminated.
        assert final_db > -40.0
        assert solver._peak_signal > 0.0

    def test_unbounded_needs_some_criterion(self):
        mesh = _layered_mesh(8, 4.0)
        ops, dt = _build_ports(mesh, 4.0, None)
        solver = FITTimeDomainSolver(
            mesh=mesh,
            boundary_conditions={},
            ports=ops,
            dt=dt,
            total_time_steps=None,
            verbose=False,
            backend="numpy",
        )
        with pytest.raises(ValueError, match="port_signal_stop_db"):
            solver.run()

    def test_signal_stop_needs_a_modal_port(self):
        mesh = _layered_mesh(8, 4.0)
        _, dt = _build_ports(mesh, 4.0, None)
        solver = FITTimeDomainSolver(
            mesh=mesh,
            boundary_conditions={},
            ports=[],
            dt=dt,
            total_time_steps=100,
            port_signal_stop_db=60.0,
            verbose=False,
            backend="numpy",
        )
        with pytest.raises(ValueError, match="modal port"):
            solver.run()
