"""DD-085 gates: physical volume states + physical monitors.

The volume states are pinned to physical FIT grid quantities at the
source (C = 1; measured pre-fix: C = 1/dy-class, grid-dependent —
internal record ``investigations/state_scale/FINDINGS.md``).  Monitors
convert states
to physical fields/power locally:

- Field monitors return E [V/m], H [A/m] at cell centres.
- ``MonitorFluxTime`` is the FIT identity ``P = Σ e·h`` (boundary-h ×½)
  in watts.
- The SAME absolute values must come out on a uniform and a graded
  grid (grid independence — the point of the fix); pre-fix they
  differed by the grid factor.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio import (
    AnalysisScatteringTD,
    Mesh,
)
from magnelio.mesh import BoxFace
from magnelio.mesh.grid import GridLines
from magnelio.monitors.field_frequency import MonitorFieldFrequency
from magnelio.monitors.flux import MonitorFluxTime
from magnelio.ports import PortSpecMultiConductor

EPS0 = 8.8541878128e-12
MU0 = 1.2566370614e-6
ETA0 = float(np.sqrt(MU0 / EPS0))

W, B, L = 10e-3, 5e-3, 30e-3
F_GATE = np.array([3e9])

_BC = {"xmin": "PMC", "xmax": "PMC", "ymin": "PEC", "ymax": "PEC", "zmin": "PEC", "zmax": "PEC"}


def _grid_uniform() -> GridLines:
    return GridLines(
        x=np.linspace(0, W, 11),
        y=np.linspace(0, B, 6),
        z=np.linspace(0, L, 31),
    )


def _grid_graded_y() -> GridLines:
    y = np.concatenate([np.linspace(0, 2e-3, 5), np.linspace(2.5e-3, B, 4)])
    return GridLines(x=np.linspace(0, W, 11), y=y, z=np.linspace(0, L, 31))


def _run_plate(grid, monitors):
    ana = AnalysisScatteringTD(
        mesh=Mesh.from_grid(grid, boundary_conditions=dict(_BC)),
        ports=[
            PortSpecMultiConductor(name="p1", plane=BoxFace.Z_MIN, n_modes=1),
            PortSpecMultiConductor(name="p2", plane=BoxFace.Z_MAX, n_modes=1),
        ],
        f_max=10e9,
        monitors=tuple(monitors),
        verbose=False,
    )
    zline = ana.solve_ports()["p1"].z_line_num
    res = ana.run(f_axis=F_GATE, excited=["p1"])
    return res, zline


@pytest.mark.parametrize("grid_fn", [_grid_uniform, _grid_graded_y], ids=["uniform", "graded_y"])
def test_absolute_fields_per_1w_cw(grid_fn):
    """|Ey| and |Hx| per 1 W CW at an interior point match the analytic
    TEM line values on the uniform AND the graded grid (grid
    independence + absolute scale in one gate)."""
    mon = MonitorFieldFrequency(
        corners=((W / 2, B / 2, L / 2), (W / 2, B / 2, L / 2)),
        freqs=F_GATE,
        fields=["Ey", "Hx"],
        name="pt",
    )
    res, zline = _run_plate(grid_fn(), [mon])
    mon.renormalize(res.reference_signal)

    ey = float(np.abs(np.asarray(mon.data["Ey"]).reshape(-1)[0]))
    hx = float(np.abs(np.asarray(mon.data["Hx"]).reshape(-1)[0]))
    # The magnetic walls sit half an outer x-cell beyond the outermost
    # grid lines (from_grid keeps them in place): the simulated line is
    # w_eff = W + dx wide, and z_line reports exactly that line.
    dx = np.diff(grid_fn().x)
    w_eff = W + 0.5 * (dx[0] + dx[-1])
    e_expect = np.sqrt(zline * 1.0) / B  # V = √(Z·1W) over the gap
    h_expect = np.sqrt(1.0 / zline) / w_eff  # I = √(1W/Z) over the width

    assert abs(ey / e_expect - 1) < 0.02, f"|Ey| per 1 W = {ey:.2f} V/m vs analytic {e_expect:.2f}"
    assert abs(hx / h_expect - 1) < 0.02, f"|Hx| per 1 W = {hx:.4f} A/m vs analytic {h_expect:.4f}"


@pytest.mark.parametrize("grid_fn", [_grid_uniform, _grid_graded_y], ids=["uniform", "graded_y"])
def test_flux_monitor_watts(grid_fn):
    """Time-integrated flux through a mid-plane equals the incident
    pulse energy minus the reflected/back-absorbed part — in joules,
    on both grids.  (S11 sits at the DTBC floor, so the transmitted
    fraction is 1 to well below the gate tolerance.)"""
    flux = MonitorFluxTime(plane=("z", L / 2), name="fluxz")
    res, _ = _run_plate(grid_fn(), [flux])

    a1 = res.a("p1")
    e_in = float(np.trapezoid(a1.values**2, a1.t))
    e_thru = float(np.trapezoid(flux.power, flux.t))

    assert abs(e_thru / e_in - 1) < 0.01, (
        f"flux energy {e_thru:.4e} J vs incident {e_in:.4e} J (ratio {e_thru / e_in:.4f})"
    )


def test_grid_independence_absolute_values():
    """The same physical problem on a uniform and a graded grid yields
    the same absolute monitor values (pre-fix: off by the grid factor,
    e.g. flux by (C·l)² ≈ 0.5…3.3 across the graded cells)."""
    vals = {}
    for tag, grid_fn in (("u", _grid_uniform), ("g", _grid_graded_y)):
        mon = MonitorFieldFrequency(
            corners=((W / 2, B / 2, L / 2), (W / 2, B / 2, L / 2)),
            freqs=F_GATE,
            fields=["Ey"],
            name="pt",
        )
        flux = MonitorFluxTime(plane=("z", L / 2), name="fluxz")
        res, _ = _run_plate(grid_fn(), [mon, flux])
        mon.renormalize(res.reference_signal)
        a1 = res.a("p1")
        vals[tag] = (
            float(np.abs(np.asarray(mon.data["Ey"]).reshape(-1)[0])),
            float(np.trapezoid(flux.power, flux.t)) / float(np.trapezoid(a1.values**2, a1.t)),
        )

    ey_u, p_u = vals["u"]
    ey_g, p_g = vals["g"]
    assert abs(ey_g / ey_u - 1) < 0.02, (ey_u, ey_g)
    assert abs(p_g / p_u - 1) < 0.01, (p_u, p_g)


def test_modal_state_scale_pinned_at_source():
    """Per-edge identity on the graded plate: the injected profile per
    unit incident √W equals the physical grid quantity of the 1-√W TEM
    wave, ê·source_scale = (√Z/B)·l_edge — exactly the C = 1 statement.
    The measured pre-pin scale is kept on the operator for
    introspection and must be the grid factor (≠ 1)."""
    from magnelio._operators.material_matrices import build_M_eps, build_M_mu

    mesh = Mesh.from_grid(_grid_graded_y())
    ana = AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions(dict(_BC)),
        ports=[
            PortSpecMultiConductor(name="p1", plane=BoxFace.Z_MIN, n_modes=1),
            PortSpecMultiConductor(name="p2", plane=BoxFace.Z_MAX, n_modes=1),
        ],
        f_max=10e9,
        verbose=False,
    )
    ana.solve_ports()
    op = ana._build_operator(
        ana.ports[0],
        build_M_eps(mesh),
        build_M_mu(mesh),
        0.9e-12,
        3e9,
    )
    dm = op.discrete_modes[0]
    z = float(dm.mode.z_modal(2 * np.pi * 3e9).real)
    src = float(op._source_scale[0])

    prof = dm.e_v_profile
    lens = op.plane.v_edge_lengths
    keep = np.abs(prof) > 1e-6 * np.abs(prof).max()
    target = np.sqrt(z) / B * lens[keep]
    ratio = np.abs(prof[keep]) * src / target
    np.testing.assert_allclose(ratio, 1.0, rtol=1e-9)

    # the pre-pin state scale (grid factor) is recorded per mode
    assert op.state_scale[0] > 100.0  # 1/dy-class, ≈ 1455 /m here


def test_discrete_port_reads_grid_quantity_volts():
    """LumpedElementOperator.project_V is the plain signed voltage sum
    (grid-quantity form) — exact on a graded grid, where the historic
    field-interpretation ``Σ E·dl`` misscaled."""
    from magnelio._operators.material_matrices import build_M_eps, build_M_mu
    from magnelio.ports import PortSpecLumped
    from magnelio.ports._lumped.factory import build_lumped_port

    mesh = Mesh.from_grid(_grid_graded_y())
    spec = PortSpecLumped(
        name="d1",
        start=(W / 2, 0.0, L / 2),
        end=(W / 2, B, L / 2),
        Z0=50.0,
    )
    m_eps = build_M_eps(mesh)
    op = build_lumped_port(spec, mesh, m_eps, build_M_mu(mesh), dt=1e-12)

    # synthetic uniform-E gap field as grid quantities: e_p = (V0/B)·l_p
    v0 = 2.5
    e = np.zeros(m_eps.shape[0])
    dy = np.asarray(mesh.grid.dy, dtype=float)
    for flat, (_, j, _unused), sign in zip(
        op.flat_edge_indices,
        op.ijk_list,
        op.edge_signs,
    ):
        e[flat] = sign * v0 / B * dy[j]
    v = float(op.project_V(e)[0])
    assert abs(abs(v) - v0) < 1e-12 * v0, f"read {v} V, want ±{v0} V"


def test_plane_wave_amplitude_is_physical():
    """A 1 V/m plane wave read back by a field monitor peaks at 1 V/m
    (few-% pulse-propagation tolerance) — the source injects grid
    quantities, the monitor converts back."""
    from magnelio.boundaries.pec import PECBoundary
    from magnelio.monitors.field_time import MonitorFieldTime
    from magnelio.solver.fit_td import FITTimeDomainSolver
    from magnelio.solver.stability import courant_dt
    from magnelio.sources.plane_wave import PlaneWaveSource

    grid = GridLines(
        x=np.linspace(0, 8e-3, 9),
        y=np.linspace(0, 8e-3, 9),
        z=np.linspace(0, 16e-3, 17),
    )
    mesh = Mesh.from_grid(grid)
    dt = courant_dt(grid, accuracy="normal")
    f_max = 20e9
    x, y, z = grid.x, grid.y, grid.z
    src = PlaneWaveSource(
        direction=(0, 0, 1.0),
        polarization=(1.0, 0, 0),
        corners=((x[2], y[2], z[2]), (x[6], y[6], z[14])),
        f_max=f_max,
        amplitude=1.0,
        waveform="gaussian",
    )
    c0 = 299_792_458.0
    t_end = 4.0 / f_max + z[10] / c0 + 2.0 / f_max
    n_steps = int(np.ceil(t_end / dt)) + 5

    mon = MonitorFieldTime(
        corners=((0.004, 0.004, 0.008), (0.004, 0.004, 0.008)),
        times=np.arange(n_steps) * dt,
        fields=["Ex"],
        name="probe",
    )
    solver = FITTimeDomainSolver(
        mesh=mesh,
        boundary_conditions={
            f: PECBoundary(f) for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
        },
        sources=[src],
        monitors=[mon],
        total_time_steps=n_steps,
        dt=dt,
        verbose=False,
    )
    solver.run()

    peak = float(np.max(np.abs(mon.data["Ex"])))
    assert abs(peak - 1.0) < 0.05, f"plane-wave monitor peak {peak:.4f} V/m, want 1 V/m"
