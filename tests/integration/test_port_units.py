"""DD-078 gates: physical √W amplitude convention across port types.

Three gates pin the per-port power normalisation (``record_scale`` /
``source_scale``, computed in ``PortOperatorModal._calibrate_v_i``):

1. **κ is physical** — on a parallel-plate TEM plane, the recorded
   voltage of a physical 1 V gap field must be 1 V after scaling
   (fast unit-level check on the built operator, no time marching).
2. **Heterogeneous modal↔modal unitarity** — a TEM line with an
   ε_r = 1 → 4 dielectric step obeys the exact Fresnel result
   |S11| = 1/3 AND lossless unitarity |S11|² + |S21|² = 1.  Before
   DD-078 the basis-scale mismatch made |S21| = 2·√(8/9) and the sum
   3.67.
3. **Mixed lumped↔modal commensurability** — a discrete port feeding
   a matched TEM line reaches the modal receiver at |S21| ≈ 0 dB
   (was −78 dB: the naked basis scale).  The small residual deficit
   is the lumped port's O(1/Nx) transverse-coupling factor (session
   103 FINDINGS), not a units error.
"""

from __future__ import annotations

import numpy as np

from magnelio import (
    AnalysisScatteringTD,
    Material,
    Mesh,
)
from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.mesh import BoxFace
from magnelio.mesh.grid import GridLines
from magnelio.ports import PortSpecLumped, PortSpecMultiConductor

GAP, WIDTH, LENGTH = 5e-3, 16e-3, 60e-3

_BC = {"xmin": "PMC", "xmax": "PMC", "ymin": "PEC", "ymax": "PEC", "zmin": "PEC", "zmax": "PEC"}


def _plate_grid(n_z: int = 121) -> GridLines:
    return GridLines(
        x=np.linspace(-WIDTH / 2, WIDTH / 2, 9),
        y=np.linspace(-GAP / 2, GAP / 2, 6),
        z=np.linspace(-LENGTH / 2, LENGTH / 2, n_z),
    )


def _two_port_analysis(mesh: Mesh) -> AnalysisScatteringTD:
    return AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions(dict(_BC)),
        ports=[
            PortSpecMultiConductor(name="m1", plane=BoxFace.Z_MIN, n_modes=1),
            PortSpecMultiConductor(name="m2", plane=BoxFace.Z_MAX, n_modes=1),
        ],
        f_max=6e9,
        verbose=False,
    )


def test_record_scale_recovers_physical_volts():
    """Gate 1: κ·⟨ê, E_1V⟩ = 1 V for a physical 1 V gap field."""
    mesh = Mesh.from_grid(_plate_grid())
    ana = _two_port_analysis(mesh)
    ana.solve_ports()
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    op = ana._build_operator(ana.ports[0], m_eps, m_mu, 0.9e-12, 3e9)

    e = np.zeros(m_eps.shape[0])
    # uniform E_y of a 1 V gap field as FIT grid quantities e = E·l
    # (DD-085: the states are edge voltages)
    e[op.plane.e_v_indices] = op.plane.v_edge_lengths / GAP
    v_recorded = float(op.record_scale[0] * op.project_V(e)[0])
    assert abs(abs(v_recorded) - 1.0) < 1e-2, (
        f"1 V field records as {v_recorded:.4f} V after record_scale"
    )
    # source_scale is the exact inverse chain: a 1 √W excitation must
    # correspond to √(Z·1W) volts of incident wave.
    z_line = op.discrete_modes[0].mode.z_line
    v_inc = op.record_scale[0] * op._source_scale[0]  # κ·(√Z/κ) = √Z
    assert abs(v_inc - np.sqrt(z_line)) / np.sqrt(z_line) < 1e-12


def test_dielectric_step_fresnel_unitarity():
    """Gate 2: ε_r 1→4 step — exact Fresnel + lossless unitarity."""
    eps_r = 4.0
    diel = Material.from_isotropic("diel4", epsilon=eps_r)
    mesh = Mesh.from_grid(
        _plate_grid(),
        regions=[
            (diel, (-WIDTH, -GAP, 0.0, WIDTH, GAP, LENGTH)),
        ],
    )
    ana = _two_port_analysis(mesh)
    reports = ana.solve_ports()
    z1 = reports["m1"].z_line_num
    z2 = reports["m2"].z_line_num
    gamma_exact = (z2 - z1) / (z2 + z1)

    f_axis = np.array([0.5e9, 1e9, 2e9])
    res = ana.run(f_axis=f_axis, excited=["m1"], total_time_steps=20000, energy_stop_db=None)
    s11 = np.abs(res.S("m1", "m1"))
    s21 = np.abs(res.S("m2", "m1"))

    assert np.all(np.abs(s11 - abs(gamma_exact)) < 5e-3), (
        f"|S11| = {s11} vs Fresnel {abs(gamma_exact):.4f}"
    )
    assert np.all(np.abs(s21 - np.sqrt(1 - gamma_exact**2)) < 5e-3), (
        f"|S21| = {s21} vs Fresnel {np.sqrt(1 - gamma_exact**2):.4f}"
    )
    unitarity = s11**2 + s21**2
    assert np.all(np.abs(unitarity - 1.0) < 1e-3), (
        f"|S11|²+|S21|² = {unitarity} (lossless line must be unitary)"
    )


def test_mixed_lumped_modal_s21_commensurate():
    """Gate 3: discrete feed → modal receiver lands at ≈ 0 dB, not −78 dB."""
    grid = _plate_grid()
    mesh = Mesh.from_grid(grid)
    zline = _two_port_analysis(mesh).solve_ports()["m1"].z_line_num

    ana = AnalysisScatteringTD(
        mesh=Mesh.from_grid(grid, boundary_conditions={**_BC, "zmin": "PMC"}),
        ports=[
            PortSpecLumped(
                name="feed",
                start=(0.0, -GAP / 2, grid.z[2]),
                end=(0.0, GAP / 2, grid.z[2]),
                Z0=zline,
            ),
            PortSpecMultiConductor(name="rx", plane=BoxFace.Z_MAX, n_modes=1),
        ],
        f_max=6e9,
        verbose=False,
    )
    res = ana.run(
        f_axis=np.array([0.5e9]), excited=["feed"], total_time_steps=20000, energy_stop_db=None
    )
    s11_db = 20 * np.log10(np.abs(res.S("feed", "feed"))[0])
    s21_db = 20 * np.log10(np.abs(res.S("rx", "feed"))[0])

    assert s11_db < -20.0, f"matched lumped feed: |S11| = {s11_db:.1f} dB"
    # Units commensurate: within the O(1/Nx) transverse-coupling deficit
    # of a single-column port (Nx=8 → ~−0.5 dB), NOT the −78 dB naked
    # basis scale of the pre-DD-078 state.
    assert -1.5 < s21_db <= 0.1, f"mixed-family |S21| = {s21_db:.1f} dB (units mismatch?)"


def test_frequency_monitor_fields_per_1w_cw():
    """Gate 4 (work item ii): renormalised freq-monitor = fields at 1 W CW.

    With DD-078 the reference waveform is a(t) in √W, so dividing the
    accumulated DFT by its spectrum (``MonitorFieldFrequency.renormalize``)
    yields the field of a 1 W CW excitation: |E_y| = √(z_line·1 W)/gap for
    the forward TEM wave on the plate.
    """
    from magnelio.monitors.field_frequency import MonitorFieldFrequency

    freqs = np.array([1e9, 2e9])
    mon = MonitorFieldFrequency(
        corners=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), freqs=freqs, fields=["Ey"], name="probe"
    )
    mesh = Mesh.from_grid(_plate_grid())
    ana = AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions(dict(_BC)),
        ports=[
            PortSpecMultiConductor(name="m1", plane=BoxFace.Z_MIN, n_modes=1),
            PortSpecMultiConductor(name="m2", plane=BoxFace.Z_MAX, n_modes=1),
        ],
        f_max=6e9,
        monitors=(mon,),
        verbose=False,
    )
    zline = ana.solve_ports()["m1"].z_line_num
    res = ana.run(f_axis=freqs, excited=["m1"], total_time_steps=20000, energy_stop_db=None)

    mon.renormalize(res.reference_signal)
    ey = np.abs(mon.data["Ey"].reshape(len(freqs), -1)[:, 0])
    e_expect = np.sqrt(zline * 1.0) / GAP
    assert np.all(np.abs(ey / e_expect - 1.0) < 1e-3), (
        f"|Ey| at 1 W CW = {ey} vs expected {e_expect:.1f} V/m"
    )


def test_source_scale_thevenin_sqrtw():
    """Discrete port: waveform is a(t) in √W ⇒ v_src = 2√Z0·a(t)."""
    from magnelio.ports._lumped.operator import PortOperatorLumped

    op = PortOperatorLumped(
        name="p",
        Z0=50.0,
        direction="y",
        flat_edge_indices=[0],
        ijk_list=[(0, 0, 0)],
        dl_list=[1e-3],
        beta_E=np.array([0.0]),
    )
    op.set_excitation(0, lambda t: 1.0)
    assert abs(op._waveform_fn(0.0) - 2.0 * np.sqrt(50.0)) < 1e-12
