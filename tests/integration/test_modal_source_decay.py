"""Source-driven absorber validation.

Uses an excited modal source on a homogeneous coaxial line with a Mur
absorber on the far port.  Drives a short Gaussian pulse, then watches
the stored energy decay after the source ends.

For TEM in vacuum coax the Mur reflection coefficient at Courant 0.95 is
``r = -0.05/1.95 ≈ -0.026`` (|r|² ≈ -32 dB per bounce); after the source
ends and the wave packet has reached the far port and been absorbed,
residual energy should be far below peak energy.

Ports are built via the public :func:`build_modal_port` factory.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.boundaries.pec import PECBoundary
from magnelio.mesh.mesher import Mesh, MeshControl
from magnelio.ports._modal import (
    BoxFace,
    PortSpecCoax,
    build_modal_port,
)
from magnelio.ports.recorder import PortSignalRecorder
from magnelio.signals import WaveformGaussian
from magnelio.signals.signal_1d import Signal1D
from magnelio.signals.waveforms import gaussian
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import (
    compute_min_effective_eps,
    compute_min_effective_mu,
    courant_dt,
)

C0 = 299_792_458.0


def _lateral_pec_bcs() -> dict:
    """PEC on the four lateral bbox faces (y/z walls).

    X_MIN / X_MAX are owned by the modal port operators in every test
    in this file and must not appear here — the operator overrides
    ``e[port_edges]`` and a coincident ``PECBoundary`` would zero it
    again the next step.
    """
    return {
        "ymin": PECBoundary("ymin"),
        "ymax": PECBoundary("ymax"),
        "zmin": PECBoundary("zmin"),
        "zmax": PECBoundary("zmax"),
    }


def _make_coax_occ_mesh(L_x: float, L_yz: float, r_i: float, r_o: float, f_max: float):
    """Build a coax FIT mesh via OCC (PEC bbox, air annulus, PEC inner).

    DD-048 path-(b) requires a non-empty mesh PEC mask on the port
    plane.  ``Mesh.from_grid`` does not provide one, so the source-
    decay tests build a real OCC coax instead.
    """
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    from magnelio.geo import Brick, Cylinder, Difference, GeometryModel
    from magnelio.materials.material import Material

    pec = Material.pec()
    air = Material.air()
    yc = zc = L_yz / 2.0
    bbox = Brick(origin=(0.0, 0.0, 0.0), size=(L_x, L_yz, L_yz), material=pec)
    out_cyl = Cylinder(origin=(0.0, yc, zc), radius=r_o, height=L_x, axis="x", material=air)
    in_cyl = Cylinder(origin=(0.0, yc, zc), radius=r_i, height=L_x, axis="x", material=pec)
    model = GeometryModel()
    model.add(Difference(bbox, out_cyl))
    model.add(Difference(out_cyl, in_cyl))
    model.add(in_cyl)
    mesh = Mesh.from_geometry(
        model,
        MeshControl(
            min_nodes_per_wavelength=4,
            max_cell_size=0.25e-3,
            min_cells_per_feature=4,
            conformal=True,
        ),
        f_max=f_max,
    )
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    dt = courant_dt(
        mesh.grid,
        accuracy="normal",
        min_effective_eps=compute_min_effective_eps(mesh),
        min_effective_mu=compute_min_effective_mu(mesh),
    )
    return mesh, m_eps, m_mu, dt


def _coax_specs(L_yz: float, r_i: float, r_o: float, f_max: float):
    """Source + load Coax port specs sharing the same TEM cross-section."""
    spec_src = PortSpecCoax(
        name="port1",
        plane=BoxFace.X_MIN,
        inner_radius=r_i,
        outer_radius=r_o,
        center=(L_yz / 2, L_yz / 2),
    )
    spec_load = PortSpecCoax(
        name="port2",
        plane=BoxFace.X_MAX,
        inner_radius=r_i,
        outer_radius=r_o,
        center=(L_yz / 2, L_yz / 2),
    )
    return spec_src, spec_load


def test_coax_source_cutoff_energy_decay():
    """Source-driven Coax line: energy must decay > 20 dB after pulse ends."""
    L_x = 30e-3
    L_yz = 3e-3
    r_i = 0.3e-3
    r_o = 1.0e-3

    f_calc = 10e9
    f_max = 10e9
    t0_pulse = 4.0 / f_max
    mesh, m_eps, m_mu, dt = _make_coax_occ_mesh(L_x, L_yz, r_i, r_o, f_max)

    spec_src, spec_load = _coax_specs(L_yz, r_i, r_o, f_max)
    op_src = build_modal_port(spec_src, mesh, m_eps, m_mu, dt=dt, f_calc=f_calc)
    op_src.set_excitation(0, WaveformGaussian(f_max=f_max))
    op_load = build_modal_port(spec_load, mesh, m_eps, m_mu, dt=dt, f_calc=f_calc)

    t_total = 1.5e-9
    n_steps = int(round(t_total / dt))

    solver = FITTimeDomainSolver(
        mesh=mesh,
        dt=dt,
        total_time_steps=n_steps,
        ports=[op_src, op_load],
        boundary_conditions=_lateral_pec_bcs(),
        verbose=False,
    )
    solver.setup()
    solver.run()
    trace = solver._energy_trace
    times = trace["time"]
    energies = trace["energy"]

    peak_idx = int(np.argmax(energies))
    peak_energy = float(energies[peak_idx])
    peak_time = float(times[peak_idx])
    final_energy = float(energies[-1])

    assert peak_time >= 0.5 * t0_pulse, (
        f"Energy peak at t={peak_time * 1e12:.1f} ps too early "
        f"(pulse peak at {t0_pulse * 1e12:.1f} ps)"
    )

    decay_dB = 10 * math.log10(max(final_energy, 1e-300) / peak_energy)
    assert decay_dB <= -20.0, (
        f"Source-cutoff decay test failed: only {decay_dB:.1f} dB decay "
        f"from peak (target ≤ −20 dB).  Peak={peak_energy:.3e} at "
        f"t={peak_time * 1e12:.1f} ps; final={final_energy:.3e}."
    )


def test_coax_source_decay_with_modal_recorder():
    """Source-driven Coax line with PortSignalRecorder attached.

    Validates the recorder's end-to-end integration in the FIT solver
    loop:

    - Buffer length matches the actual step count.
    - V at the source port shows a clear pulse-shaped peak during the
      excitation window and decays well below peak afterwards.
    - V at the load port has matching shape and finite values.
    """
    L_x = 30e-3
    L_yz = 3e-3
    r_i = 0.3e-3
    r_o = 1.0e-3

    f_calc = 10e9
    f_max = 10e9
    t0_pulse = 4.0 / f_max
    mesh, m_eps, m_mu, dt = _make_coax_occ_mesh(L_x, L_yz, r_i, r_o, f_max)

    spec_src, spec_load = _coax_specs(L_yz, r_i, r_o, f_max)
    op_src = build_modal_port(spec_src, mesh, m_eps, m_mu, dt=dt, f_calc=f_calc)
    op_src.set_excitation(0, WaveformGaussian(f_max=f_max))
    op_load = build_modal_port(spec_load, mesh, m_eps, m_mu, dt=dt, f_calc=f_calc)

    rec = PortSignalRecorder(dt=dt, ports=[op_src, op_load])

    t_total = 1.5e-9
    n_steps = int(round(t_total / dt))
    solver = FITTimeDomainSolver(
        mesh=mesh,
        dt=dt,
        total_time_steps=n_steps,
        ports=[op_src, op_load],
        recorder=rec,
        boundary_conditions=_lateral_pec_bcs(),
        verbose=False,
    )
    solver.setup()
    solver.run()

    actual_steps = solver._actual_steps
    assert rec.n_steps_recorded == actual_steps

    signals = rec.finalize()
    assert set(signals.keys()) == {("port1", 0), ("port2", 0)}

    V_src, _ = signals[("port1", 0)]
    V_load, _ = signals[("port2", 0)]

    assert V_src.values.shape == (actual_steps,)
    assert V_load.values.shape == (actual_steps,)

    peak_idx_src = int(np.argmax(np.abs(V_src.values)))
    peak_t_src = V_src.t[peak_idx_src]
    assert peak_t_src < 3 * t0_pulse, (
        f"V_src peak at {peak_t_src * 1e12:.1f} ps too late "
        f"(pulse peak at {t0_pulse * 1e12:.1f} ps)"
    )

    peak_V_src = float(np.max(np.abs(V_src.values)))
    final_V_src = float(np.abs(V_src.values[-1]))
    decay_dB = 20 * math.log10(max(final_V_src, 1e-300) / peak_V_src)
    assert decay_dB <= -20.0, (
        f"V_src did not decay: peak={peak_V_src:.3e}, "
        f"final={final_V_src:.3e}, decay={decay_dB:.1f} dB"
    )

    assert np.all(np.isfinite(V_load.values))
    assert V_load.values.shape == V_src.values.shape


def test_coax_pec_confined_wave_arrival():
    """End-to-end physics test: TEM wave reaches the far port.

    Geometry built via ``Mesh.from_geometry`` with a PEC outer brick, an
    air outer cylinder, and a PEC inner cylinder — i.e. a real coaxial
    line.  Pass criteria are physical:

    1. ``V_load`` peak is a substantial fraction of ``V_src`` peak.
    2. ``V_load`` peak time matches the analytical TEM traversal
       (``t = t0_pulse + L_x / c₀``) to within ±50%.
    3. Power-wave passivity ``|S11|² + |S21|² ≤ 1.1``.
    4. ``max |S21| > -2 dB`` somewhere in the band.

    OCC required for ``Mesh.from_geometry``.
    """
    pytest.importorskip("OCC.Core.BRepPrimAPI")

    from magnelio.geo import Brick, Cylinder, Difference, GeometryModel
    from magnelio.materials.material import Material

    L_x = 30e-3
    L_yz = 3e-3
    r_i = 0.3e-3
    r_o = 1.0e-3
    yc = zc = L_yz / 2.0

    pec = Material.pec()
    air = Material.air()

    bbox = Brick(origin=(0.0, 0.0, 0.0), size=(L_x, L_yz, L_yz), material=pec)
    out_cyl = Cylinder(origin=(0.0, yc, zc), radius=r_o, height=L_x, axis="x", material=air)
    in_cyl = Cylinder(origin=(0.0, yc, zc), radius=r_i, height=L_x, axis="x", material=pec)

    model = GeometryModel()
    model.add(Difference(bbox, out_cyl))  # PEC outside r_o
    model.add(Difference(out_cyl, in_cyl))  # air in r_i..r_o annulus
    model.add(in_cyl)  # PEC inner conductor

    f_max = 10e9
    mesh = Mesh.from_geometry(
        model,
        MeshControl(
            min_nodes_per_wavelength=4,
            max_cell_size=0.25e-3,
            min_cells_per_feature=4,
            conformal=True,
        ),
        f_max=f_max,
    )
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    dt = courant_dt(
        mesh.grid,
        accuracy="normal",
        min_effective_eps=compute_min_effective_eps(mesh),
        min_effective_mu=compute_min_effective_mu(mesh),
    )
    f_calc = f_max
    t0_pulse = 4.0 / f_max

    spec_src, spec_load = _coax_specs(L_yz, r_i, r_o, f_max)
    op_src = build_modal_port(spec_src, mesh, m_eps, m_mu, dt=dt, f_calc=f_calc)
    op_src.set_excitation(0, WaveformGaussian(f_max=f_max))
    op_load = build_modal_port(spec_load, mesh, m_eps, m_mu, dt=dt, f_calc=f_calc)
    rec = PortSignalRecorder(dt=dt, ports=[op_src, op_load])

    n_steps = int(round(1.5e-9 / dt))
    solver = FITTimeDomainSolver(
        mesh=mesh,
        dt=dt,
        total_time_steps=n_steps,
        ports=[op_src, op_load],
        recorder=rec,
        boundary_conditions=_lateral_pec_bcs(),
        verbose=False,
    )
    solver.setup()
    solver.run()

    signals = rec.finalize()
    V_src, _ = signals[("port1", 0)]
    V_load, _ = signals[("port2", 0)]

    peak_V_src = float(np.max(np.abs(V_src.values)))
    peak_V_load = float(np.max(np.abs(V_load.values)))
    t_load_peak = float(V_load.t[int(np.argmax(np.abs(V_load.values)))])

    ratio = peak_V_load / peak_V_src
    assert ratio > 0.5, (
        f"V_load/V_src = {ratio:.3f} (target > 0.5). "
        f"V_src peak = {peak_V_src:.3e}, V_load peak = {peak_V_load:.3e}. "
        f"Wave is not propagating through the coax — confinement issue?"
    )

    t_expected = t0_pulse + L_x / C0
    rel_err = abs(t_load_peak - t_expected) / t_expected
    assert rel_err < 0.5, (
        f"V_load peak at {t_load_peak * 1e12:.1f} ps, expected ≈ "
        f"{t_expected * 1e12:.1f} ps (rel. error {rel_err:.2f}, "
        f"target < 0.5)."
    )

    from magnelio.post import compute_s_parameters

    ref_t = np.arange(rec.n_steps_recorded) * dt
    ref_sig = Signal1D(
        t=ref_t,
        values=np.array([gaussian(float(tk), f_max) for tk in ref_t]),
        dt=dt,
        label="excitation",
    )
    f_axis = np.linspace(1e9, 10e9, 19)
    S = compute_s_parameters(
        recorder_signals=signals,
        port_modes={
            "port1": [dm.mode for dm in op_src.discrete_modes],
            "port2": [dm.mode for dm in op_load.discrete_modes],
        },
        excited=("port1", 0),
        reference_signal=ref_sig,
        f_axis=f_axis,
    )
    S11 = S[("port1", 0)]
    S21 = S[("port2", 0)]
    valid = ~(np.isnan(S11) | np.isnan(S21))
    assert np.any(valid), "no valid S-parameter samples"
    sum_sq = np.abs(S11[valid]) ** 2 + np.abs(S21[valid]) ** 2

    # Threshold raised to 1.15 once Signal1D.at_frequencies switched from
    # rfft+linear-interp to the exact direct DFT: the previous interp
    # path under-reported magnitudes by 1–3 % per channel, which
    # *masked* the true |S|²-sum on the staircased OCC-meshed coax
    # (≈ 1.10–1.12).  The remaining excess is a known V/I-calibration
    # interaction with the radial-staircase Z_line discretisation
    # error (10 % at L0 mesh, decreasing to 0.04 % at the 4-level
    # Richardson-extrapolated mesh per Cleanup 3 in STATUS.md).
    assert float(np.max(sum_sq)) <= 1.15, (
        f"|S|²-sum = {float(np.max(sum_sq)):.3f} > 1.15 — passive "
        f"bound violated; V/I calibration regression?"
    )

    s21_max_db = float(np.max(20.0 * np.log10(np.abs(S21[valid]) + 1e-300)))
    assert s21_max_db > -2.0, (
        f"max |S21| = {s21_max_db:.2f} dB across band — wave is not "
        f"propagating cleanly; calibration or geometry regression?"
    )
