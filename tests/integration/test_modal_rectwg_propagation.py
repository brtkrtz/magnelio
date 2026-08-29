"""End-to-end Rect-WG validation (Test 4 of the Phase-1 ladder).

Source-driven straight WR-90 line, vacuum, port-to-port.  Excite TE10
at port1 with a band-limited modulated Gaussian, observe transmission
to port2, and assert physical pass criteria:

1.  ``V_load`` peak amplitude is a substantial fraction of ``V_src`` —
    the mode actually traverses the line.
2.  ``V_load`` peak time matches the analytical TE10 group-velocity
    traversal ``t = t0_pulse + L_x / v_g``.
3.  Power-wave passivity ``|S11|² + |S21|² <= 1 + slack`` everywhere in
    the propagating band.
4.  ``max |S21|`` close to 0 dB in the propagating band — matched-line
    transmission limited only by the Mur-1st residual reflection.

Setup follows the commercial-suite workflow: a vacuum WR-90 cavity with
``PECBoundary`` on the four lateral faces (y/z walls).  X_MIN / X_MAX
are owned by the modal-port operators and must not get a PEC BC —
the operator overrides ``e[port_edges]`` each step.

Ports are built via the public :func:`build_modal_port` factory; the
spec-side (u, v)-frame swap on X_MAX is handled internally so the test
gives the same ``width_a=WR90_A, height_b=WR90_B`` description at both
ends.
"""

from __future__ import annotations

import math

import numpy as np

from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.boundaries.pec import PECBoundary
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.ports._modal import (
    BoxFace,
    PortSpecRectWG,
    build_modal_port,
)
from magnelio.ports.recorder import PortSignalRecorder
from magnelio.post import compute_s_parameters
from magnelio.signals import Waveform, WaveformGaussianModulated
from magnelio.signals.signal_1d import Signal1D
from magnelio.signals.waveforms import modulated_gaussian
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import courant_dt

C0 = 299_792_458.0
WR90_A = 22.86e-3
WR90_B = 10.16e-3


def _lateral_pec_bcs() -> dict:
    """PEC on the four lateral faces — the WR-90 walls.

    X_MIN / X_MAX are owned by the modal port operators and must
    not appear here.
    """
    return {
        "ymin": PECBoundary("ymin"),
        "ymax": PECBoundary("ymax"),
        "zmin": PECBoundary("zmin"),
        "zmax": PECBoundary("zmax"),
    }


def _wr90_grid_30mm() -> GridLines:
    L_x = 30e-3
    return GridLines(
        x=np.linspace(0.0, L_x, 31),
        y=np.linspace(0.0, WR90_A, 24),
        z=np.linspace(0.0, WR90_B, 11),
    )


def _build_ports(
    mesh: Mesh,
    m_eps: np.ndarray,
    m_mu: np.ndarray,
    dt: float,
    f_calc: float,
    n_modes: int,
    waveform: Waveform,
):
    spec_src = PortSpecRectWG(
        name="port1",
        plane=BoxFace.X_MIN,
        width_a=WR90_A,
        height_b=WR90_B,
        n_modes=n_modes,
    )
    spec_load = PortSpecRectWG(
        name="port2",
        plane=BoxFace.X_MAX,
        width_a=WR90_A,
        height_b=WR90_B,
        n_modes=n_modes,
    )
    op_src = build_modal_port(spec_src, mesh, m_eps, m_mu, dt=dt, f_calc=f_calc)
    op_src.set_excitation(0, waveform)
    op_load = build_modal_port(spec_load, mesh, m_eps, m_mu, dt=dt, f_calc=f_calc)
    return op_src, op_load


def test_rectwg_te10_wave_arrival():
    """Source-driven TE10 traverses a straight WR-90 line."""
    grid = _wr90_grid_30mm()
    L_x = float(grid.x[-1] - grid.x[0])
    mesh = Mesh.from_grid(grid)
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    dt = courant_dt(grid, accuracy="normal")

    f_min = 8.2e9
    f_max = 12.4e9
    f_calc = 10.0e9

    waveform = WaveformGaussianModulated(f_min=f_min, f_max=f_max)
    op_src, op_load = _build_ports(
        mesh,
        m_eps,
        m_mu,
        dt,
        f_calc,
        n_modes=1,
        waveform=waveform,
    )
    rec = PortSignalRecorder(dt=dt, ports=[op_src, op_load])

    bandwidth = f_max - f_min
    t0_pulse = 4.0 / bandwidth

    f_c_te10 = C0 / (2.0 * WR90_A)
    v_g_calc = C0 * math.sqrt(max(0.0, 1.0 - (f_c_te10 / f_calc) ** 2))
    t_traversal = L_x / v_g_calc
    t_total = 2.5 * t0_pulse + 5.0 * t_traversal
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

    signals = rec.finalize()
    V_src, _ = signals[("port1", 0)]
    V_load, _ = signals[("port2", 0)]

    peak_V_src = float(np.max(np.abs(V_src.values)))
    peak_V_load = float(np.max(np.abs(V_load.values)))
    t_load_peak = float(V_load.t[int(np.argmax(np.abs(V_load.values)))])

    # (1) V_load amplitude
    ratio = peak_V_load / peak_V_src
    assert ratio > 0.5, (
        f"V_load/V_src = {ratio:.3f} (target > 0.5). "
        f"V_src peak = {peak_V_src:.3e}, V_load peak = {peak_V_load:.3e}."
    )

    # (2) Arrival time
    t_expected = t0_pulse + t_traversal
    rel_err = abs(t_load_peak - t_expected) / t_expected
    assert rel_err < 0.5, (
        f"V_load peak at {t_load_peak * 1e12:.1f} ps, expected ~ "
        f"{t_expected * 1e12:.1f} ps (rel. error {rel_err:.2f}, target < 0.5)."
    )

    # (3) S-parameter passivity
    f_axis = np.linspace(f_min, f_max, 21)
    ref_t = np.arange(rec.n_steps_recorded) * dt

    def _waveform(t: float) -> float:
        return float(modulated_gaussian(t, f_max, f_min))

    ref_sig = Signal1D(
        t=ref_t,
        values=np.array([_waveform(float(tk)) for tk in ref_t]),
        dt=dt,
        label="excitation",
    )
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
    assert float(np.max(sum_sq)) <= 1.1, (
        f"|S|²-sum = {float(np.max(sum_sq)):.3f} > 1.1 — passivity broken."
    )

    # (4) max |S21| close to 0 dB
    s21_max_db = float(np.max(20.0 * np.log10(np.abs(S21[valid]) + 1e-300)))
    assert s21_max_db > -2.0, (
        f"max |S21| = {s21_max_db:.2f} dB in [{f_min / 1e9:.1f}, "
        f"{f_max / 1e9:.1f}] GHz — TE10 is not propagating cleanly."
    )


def test_rectwg_te10_no_higher_mode_leakage():
    """Multi-mode regression (Test 5 scope, restricted to in-band evanescents).

    n_modes = 3 at both ports (TE10 propagating + TE20, TE01 evanescent
    across [8.2, 12.4] GHz).  Excite TE10 at port1 and assert that the
    cross-mode |S21| / |S11| stays at least 20 dB below |S21_TE10|.
    """
    grid = _wr90_grid_30mm()
    L_x = float(grid.x[-1] - grid.x[0])
    mesh = Mesh.from_grid(grid)
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    dt = courant_dt(grid, accuracy="normal")

    f_min = 8.2e9
    f_max = 12.4e9
    f_calc = 10.0e9

    waveform = WaveformGaussianModulated(f_min=f_min, f_max=f_max)
    op_src, op_load = _build_ports(
        mesh,
        m_eps,
        m_mu,
        dt,
        f_calc,
        n_modes=3,
        waveform=waveform,
    )
    rec = PortSignalRecorder(dt=dt, ports=[op_src, op_load])

    bandwidth = f_max - f_min
    t0_pulse = 4.0 / bandwidth

    f_c_te10 = C0 / (2.0 * WR90_A)
    v_g_calc = C0 * math.sqrt(max(0.0, 1.0 - (f_c_te10 / f_calc) ** 2))
    t_traversal = L_x / v_g_calc
    t_total = 2.5 * t0_pulse + 5.0 * t_traversal
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

    signals = rec.finalize()
    f_axis = np.linspace(f_min, f_max, 21)
    ref_t = np.arange(rec.n_steps_recorded) * dt

    def _waveform(t: float) -> float:
        return float(modulated_gaussian(t, f_max, f_min))

    ref_sig = Signal1D(
        t=ref_t,
        values=np.array([_waveform(float(tk)) for tk in ref_t]),
        dt=dt,
        label="excitation",
    )
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

    s21_te10 = np.abs(S[("port2", 0)])
    s21_te20 = np.abs(S[("port2", 1)])
    s21_te01 = np.abs(S[("port2", 2)])
    s11_te20 = np.abs(S[("port1", 1)])
    s11_te01 = np.abs(S[("port1", 2)])
    valid = ~(np.isnan(s21_te10) | np.isnan(s21_te20) | np.isnan(s21_te01))
    assert np.any(valid), "no valid S samples"

    s21_te10_db = 20.0 * np.log10(s21_te10[valid] + 1e-300)
    s21_te20_db = 20.0 * np.log10(s21_te20[valid] + 1e-300)
    s21_te01_db = 20.0 * np.log10(s21_te01[valid] + 1e-300)
    s11_te20_db = 20.0 * np.log10(s11_te20[valid] + 1e-300)
    s11_te01_db = 20.0 * np.log10(s11_te01[valid] + 1e-300)

    assert float(np.min(s21_te10_db - s21_te20_db)) > 20.0, (
        f"TE10->TE20 margin only {float(np.min(s21_te10_db - s21_te20_db)):.1f} dB."
    )
    assert float(np.min(s21_te10_db - s21_te01_db)) > 20.0, (
        f"TE10->TE01 margin only {float(np.min(s21_te10_db - s21_te01_db)):.1f} dB."
    )
    assert float(np.min(s21_te10_db - s11_te20_db)) > 20.0, (
        "port1 TE20 return-channel level too high."
    )
    assert float(np.min(s21_te10_db - s11_te01_db)) > 20.0, (
        "port1 TE01 return-channel level too high."
    )


def test_rectwg_te10_te20_propagating_no_leakage():
    """Test 5 §7 — TE10 -> TE20 cross-mode regression with TE20 propagating.

    Same WR-90 setup as Test 4, but band shifted into ``[13.5, 14.5] GHz``
    so that TE20 (cutoff 13.114 GHz) is *above its cut-off* throughout
    the test band.  TE01 (14.744 GHz) and TE11/TM11 (16.146 GHz) stay
    evanescent at every f in the band, so the only mode change vs the
    in-band Test 5 is that TE20 itself has a real propagation constant.

    On a homogeneous straight WG, TE10 -> TE20 coupling is structurally
    zero (mode orthogonality in the M_ε inner product), so any non-zero
    ``|S21_TE20|`` here would expose a propagating-only leakage path —
    e.g. an operator-side mode mixing in the TF/SF buffer or in the V/I
    calibration that only manifests when ``Z_TE20`` becomes real.

    Pass criteria (§7 Test 5 strict bound):
    1.  ``max |S21_TE10|  >  -2 dB``      — TE10 traverses cleanly.
    2.  ``max |S21_TE20|  < -80 dB``      — propagating TE20 stays at
        the FFT noise floor; no spurious leakage path.
    3.  cross-mode margin ``|S21_TE10| - |S21_TE20|  >  60 dB``.
    """
    grid = _wr90_grid_30mm()
    L_x = float(grid.x[-1] - grid.x[0])
    mesh = Mesh.from_grid(grid)
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    dt = courant_dt(grid, accuracy="normal")

    f_min = 13.5e9
    f_max = 14.5e9
    f_calc = 14.0e9

    waveform = WaveformGaussianModulated(f_min=f_min, f_max=f_max)
    # n_modes=2 (TE10 + TE20).  The original Phase-1 test asked for 5
    # modes to keep the analytical solver in a TE/TM mixed regime; with
    # the DD-048 path-(b) numerical solver, requesting TM on a coarse
    # 24×11 grid pulls in spurious low-omega TM modes.  This test only
    # uses S21[0] (TE10) and S21[1] (TE20), so 2 modes suffice.
    op_src, op_load = _build_ports(
        mesh,
        m_eps,
        m_mu,
        dt,
        f_calc,
        n_modes=2,
        waveform=waveform,
    )

    # Sanity: physical TE20 (mode index 1 at both ports) is propagating
    # at every f in the band.  At X_MAX the local-frame label is TE02_uv
    # for the same physical wave; the factory's (u, v) swap routes mode
    # index 1 to TE20 at both ends regardless.
    for f in (f_min, f_calc, f_max):
        for label, op in (("X_MIN", op_src), ("X_MAX", op_load)):
            mode = op.discrete_modes[1].mode
            g = mode.gamma(2.0 * math.pi * f)
            assert g.real == 0.0 and g.imag > 0.0, (
                f"physical TE20 at {label} must be propagating at f={f / 1e9:.2f} GHz, got γ={g}."
            )

    rec = PortSignalRecorder(dt=dt, ports=[op_src, op_load])

    bandwidth = f_max - f_min
    t0_pulse = 4.0 / bandwidth

    f_c_te10 = C0 / (2.0 * WR90_A)
    v_g_calc = C0 * math.sqrt(max(0.0, 1.0 - (f_c_te10 / f_calc) ** 2))
    t_traversal = L_x / v_g_calc
    t_total = 2.5 * t0_pulse + 5.0 * t_traversal
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

    signals = rec.finalize()
    f_axis = np.linspace(f_min, f_max, 21)
    ref_t = np.arange(rec.n_steps_recorded) * dt

    def _waveform(t: float) -> float:
        return float(modulated_gaussian(t, f_max, f_min))

    ref_sig = Signal1D(
        t=ref_t,
        values=np.array([_waveform(float(tk)) for tk in ref_t]),
        dt=dt,
        label="excitation",
    )
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

    s21_te10 = np.abs(S[("port2", 0)])
    s21_te20 = np.abs(S[("port2", 1)])
    valid = ~(np.isnan(s21_te10) | np.isnan(s21_te20))
    assert np.any(valid), "no valid S samples"

    s21_te10_db = 20.0 * np.log10(s21_te10[valid] + 1e-300)
    s21_te20_db = 20.0 * np.log10(s21_te20[valid] + 1e-300)

    # (1) TE10 transmits cleanly (matched-line in the propagating band).
    assert float(np.max(s21_te10_db)) > -2.0, (
        f"max |S21_TE10| = {float(np.max(s21_te10_db)):.2f} dB — "
        f"TE10 is not propagating cleanly in "
        f"[{f_min / 1e9:.1f}, {f_max / 1e9:.1f}] GHz."
    )

    # (2) §7 Test 5 strict bound: propagating TE20 has no spurious floor.
    assert float(np.max(s21_te20_db)) < -80.0, (
        f"max |S21_TE20| = {float(np.max(s21_te20_db)):.2f} dB — "
        f"propagating TE20 has a spurious leakage floor above the §7 "
        f"-80 dB bound."
    )

    # (3) Cross-mode margin (matches the in-band Test 5 pattern; tighter
    # here because the §7 propagating-TE20 floor must be much lower).
    assert float(np.min(s21_te10_db - s21_te20_db)) > 60.0, (
        f"TE10 -> TE20 cross-mode margin only {float(np.min(s21_te10_db - s21_te20_db)):.1f} dB."
    )
