"""DD-080 line gates: thin wire in a square PEC duct (T1-T3).

The fixture is the genuine single-edge transmission line that DD-075 F4
deferred the quantitative discrete-port gates to: a thin wire centered
in a square PEC duct, fed and terminated through lumped gaps in the
wire (the DD-079 discrete ports drive exactly the gap edge).  Closed
form for the square coax (side s, round inner conductor d = 2a,
s/d > 4):  Z0 = (eta0/2pi) * ln(1.0787 * s / d).

T1 — the corrected wire presents the analytic Z0 across a 4x radius
     sweep (median input impedance on a matched line, 5%), tracks the
     ln(a) law, and terminates matched (max |S11| <= -15 dB; the
     residual is the DD-078 (1+f) single-column bookkeeping, measured
     separately by the T7 investigation, not asserted here).  A graded
     transverse axis pins the geometric-mean r0 rule.
T2 — the lambda/4 stub null: a wire shorted to the zmin wall with an
     open end resonates at f = c/(4*L_eff), L_eff = span - gap cell
     (exact for the two-stub series: cot(bL1) = tan(bL2) at
     b(L1+L2) = pi/2); gate 4%.
T3 — mechanism + calibration: with the correction OFF the line is
     radius-blind; the bare plateau solves to the grid's own
     equivalent radius r0/cell in [0.15, 0.30] (KAPPA0 anchor).
T5 — a wire ending ON a PMC wall sees an ideal open there (image
     theory: the perpendicular current mirrors anti-directed, current
     null at the wall — a PMC wall is the line-theory OPEN, with no
     fringing capacitance), so the short(PEC)->open(PMC) resonator
     hits c/(4 L_eff) even tighter than T2's fringing open end.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from magnelio import AnalysisScatteringTD, Mesh
from magnelio.circuit import SeriesRLC
from magnelio.circuit.rasterize import EdgePath
from magnelio.mesh._thin_wire import apply_thin_wire_path
from magnelio.mesh.grid import GridLines
from magnelio.mesh.indexing import edge_index_Ez
from magnelio.ports import PortSpecLumped

ETA0_2PI = 59.9585  # eta0 / 2pi [ohm]
C0 = 299_792_458.0
# The duct walls are set per run through the solver's BC dict; the mesh
# itself carries no closure of its own (DD-103: "PMC" is the free curl
# update, i.e. what an undeclared face used to do).
_BC_OPEN = {f: "PMC" for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}

D = 1e-3  # uniform cell [m]
S_CELLS = 12  # duct side [cells]
L_CELLS = 60  # duct length [cells]
K_FEED = 2  # gap edge for the feed port
N_STEPS = 6000

_CACHE: dict = {}


def duct_z0(a: float) -> float:
    """Square-coax closed form (round wire, side S_CELLS*D)."""
    return ETA0_2PI * math.log(1.0787 * (S_CELLS * D) / (2.0 * a))


def _graded_ax() -> np.ndarray:
    """Symmetric transverse axis, fine at the wire (node exactly at 6 mm).

    Applied to BOTH transverse axes: the correction is derived for
    locally isotropic transverse cells (single-axis grading at the wire
    triggers the anisotropy warning and degrades Z0 — a documented
    DD-080 v1 limitation).
    """
    half = np.cumsum(0.55e-3 * 1.18 ** np.arange(8))
    half = half / half[-1] * (S_CELLS * D / 2)
    return np.concatenate([6e-3 - half[::-1], [6e-3], 6e-3 + half])


def _line_run(
    a_radius,
    *,
    correction=True,
    graded=False,
    open_end=False,
    pmc_end=False,
    port_z0=None,
    term_r=None,
    f_axis=None,
):
    key = (
        a_radius,
        correction,
        graded,
        open_end,
        pmc_end,
        port_z0,
        term_r,
        None if f_axis is None else (f_axis[0], f_axis[-1], len(f_axis)),
    )
    if key in _CACHE:
        return _CACHE[key]

    uni = np.linspace(0, S_CELLS * D, S_CELLS + 1)
    grid = GridLines(
        x=_graded_ax() if graded else uni.copy(),
        y=_graded_ax() if graded else uni.copy(),
        z=np.linspace(0, L_CELLS * D, L_CELLS + 1),
    )
    mesh = Mesh.from_grid(grid, boundary_conditions=_BC_OPEN)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    i0 = int(np.argmin(np.abs(grid.x - 6e-3)))
    j0 = S_CELLS // 2

    k_term = Nz - 3
    k_end = Nz - 10 if open_end else Nz  # open stub ends at 50 mm
    gaps = {K_FEED} if (open_end or pmc_end) else {K_FEED, k_term}
    ks = [k for k in range(k_end) if k not in gaps]
    path = EdgePath(
        axes=["z"] * len(ks),
        ijk=[(i0, j0, k) for k in ks],
        signs=[1] * len(ks),
        dls=[float(grid.dz[k]) for k in ks],
        flat_indices=[n_Ex + n_Ey + edge_index_Ez(i0, j0, k, Nx, Ny, Nz) for k in ks],
    )
    for flat in path.flat_indices:
        mesh.pec_mask_edges[2, flat - n_Ex - n_Ey] = True
    if correction:
        apply_thin_wire_path(mesh, path, a_radius, name="wire")

    z0p = port_z0 if port_z0 is not None else duct_z0(a_radius)
    x0, y0 = float(grid.x[i0]), float(grid.y[j0])
    ports = [
        PortSpecLumped(
            name="feed",
            start=(x0, y0, grid.z[K_FEED]),
            end=(x0, y0, grid.z[K_FEED + 1]),
            Z0=z0p,
        )
    ]
    if not (open_end or pmc_end):
        r_term = term_r if term_r is not None else duct_z0(a_radius)
        ports.append(
            PortSpecLumped(
                name="term",
                start=(x0, y0, grid.z[k_term]),
                end=(x0, y0, grid.z[k_term + 1]),
                Z0=r_term,
                element=SeriesRLC(R=r_term),
            )
        )
    if f_axis is None:
        f_axis = np.linspace(0.5e9, 4.5e9, 9)
    bc = {f: "PEC" for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    if pmc_end:
        bc["zmax"] = "PMC"
    ana = AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions(bc),
        ports=ports,
        f_max=5e9,
        verbose=False,
    )
    res = ana.run(f_axis=f_axis, excited=["feed"], total_time_steps=N_STEPS, energy_stop_db=None)
    s11 = res.S("feed", "feed")
    zin = z0p * (1 + s11) / (1 - s11)
    out = (np.asarray(f_axis), s11, zin)
    _CACHE[key] = out
    return out


# ---------------------------------------------------------------------------
# T1 — Z0 radius sweep + ln tracking + matched termination
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a_frac", [0.05, 0.1, 0.2])
def test_t1_z0_radius_sweep(a_frac):
    a = a_frac * D
    _, s11, zin = _line_run(a)
    z0a = duct_z0(a)
    z_med = float(np.median(zin.real))
    assert abs(z_med - z0a) < 0.05 * z0a, (
        f"a={a_frac}D: median Zin {z_med:.1f} vs analytic {z0a:.1f}"
    )
    s11_db = 20 * np.log10(np.max(np.abs(s11)))
    assert s11_db <= -15.0, f"matched termination |S11| = {s11_db:.1f} dB"


def test_t1_ln_tracking():
    """Z0(0.05D) - Z0(0.2D) = (eta0/2pi) ln 4 = 83.1 ohm."""
    _, _, zin_thin = _line_run(0.05 * D)
    _, _, zin_fat = _line_run(0.2 * D)
    dz_meas = float(np.median(zin_thin.real) - np.median(zin_fat.real))
    dz_ref = ETA0_2PI * math.log(4.0)
    assert abs(dz_meas - dz_ref) < 0.15 * dz_ref, (
        f"ln tracking: measured {dz_meas:.1f} vs {dz_ref:.1f} ohm"
    )


def test_t1_graded_transverse_axis():
    """The geometric-mean r0 rule holds on a graded transverse grid."""
    a = 0.05 * D
    _, s11, zin = _line_run(a, graded=True)
    z0a = duct_z0(a)
    z_med = float(np.median(zin.real))
    assert abs(z_med - z0a) < 0.05 * z0a, f"graded: median Zin {z_med:.1f} vs analytic {z0a:.1f}"


# ---------------------------------------------------------------------------
# T2 — lambda/4 stub null (the resurrected DD-075 F4 gate)
# ---------------------------------------------------------------------------


def test_t2_quarter_wave_stub_null():
    a = 0.05 * D
    f_axis = np.linspace(1.0e9, 2.1e9, 111)
    _, _, zin = _line_run(a, open_end=True, f_axis=f_axis)
    # Wire span: zmin wall short to the open end at (L_CELLS-10) cells,
    # minus the feed-gap cell (series stubs: b*(L1+L2) = pi/2).
    l_eff = (L_CELLS - 10) * D - D
    f_ref = C0 / (4.0 * l_eff)
    f_min = float(f_axis[int(np.argmin(np.abs(zin)))])
    assert abs(f_min - f_ref) < 0.04 * f_ref, (
        f"lambda/4 null at {f_min / 1e9:.3f} GHz vs {f_ref / 1e9:.3f} GHz"
    )


def test_t5_pmc_end_is_ideal_open():
    """Wire ending ON the zmax PMC wall: short(PEC) -> open(PMC) resonator.

    The PMC wall is the ideal line open (current null, no fringing):
    |Zin| minimum at b*(L1+L2) = pi/2 with L1+L2 = wall span minus the
    feed-gap cell.
    """
    a = 0.05 * D
    f_axis = np.linspace(1.05e9, 1.55e9, 101)
    _, _, zin = _line_run(a, pmc_end=True, f_axis=f_axis)
    l_eff = L_CELLS * D - D
    f_ref = C0 / (4.0 * l_eff)
    f_min = float(f_axis[int(np.argmin(np.abs(zin)))])
    assert abs(f_min - f_ref) < 0.02 * f_ref, (
        f"PMC-open lambda/4 null at {f_min / 1e9:.4f} GHz vs {f_ref / 1e9:.4f} GHz"
    )


# ---------------------------------------------------------------------------
# T3 — radius-blindness + the grid's own equivalent radius
# ---------------------------------------------------------------------------


def test_t3_bare_line_r0_calibration():
    """Correction OFF: the line solves to r0/cell in [0.15, 0.30]."""
    _, _, zin = _line_run(0.05 * D, correction=False, port_z0=200.0, term_r=200.0)
    z_bare = float(np.median(zin.real))
    r0 = 1.0787 * (S_CELLS * D) / (2.0 * math.exp(z_bare / ETA0_2PI))
    assert 0.15 <= r0 / D <= 0.30, f"bare grid r0 = {r0 / D:.3f} cells (Z = {z_bare:.1f} ohm)"
    # Mechanism: without the correction the thin-wire target is missed
    # by far more than the T1 tolerance.
    assert abs(z_bare - duct_z0(0.05 * D)) > 50.0


def test_t3_bare_line_radius_blind():
    """The bare masked chain carries no radius at all: identical runs."""
    _, s11_a, _ = _line_run(0.05 * D, correction=False, port_z0=200.0, term_r=200.0)
    _, s11_b, _ = _line_run(0.2 * D, correction=False, port_z0=200.0, term_r=200.0)
    np.testing.assert_array_equal(s11_a, s11_b)


# ---------------------------------------------------------------------------
# T6 — checkpoint/resume with a thin wire in the mesh (bit-exact)
# ---------------------------------------------------------------------------


def test_t6_resume_with_wire_bit_exact(tmp_path):
    """The wire lives in the stored consolidated mesh: resume is free.

    Clone of the DD-079 resume gate on the wire-in-duct line — an
    aborted project-backed run resumed across the checkpoint seam is
    bit-identical to one uninterrupted run.
    """
    from magnelio import open_project, resume

    a = 0.05 * D

    def _build(project=None):
        uni = np.linspace(0, S_CELLS * D, S_CELLS + 1)
        grid = GridLines(x=uni.copy(), y=uni.copy(), z=np.linspace(0, 30 * D, 31))
        mesh = Mesh.from_grid(grid)
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        n_Ex = Nx * (Ny + 1) * (Nz + 1)
        n_Ey = (Nx + 1) * Ny * (Nz + 1)
        i0 = j0 = S_CELLS // 2
        ks = [k for k in range(Nz) if k != K_FEED]
        path = EdgePath(
            axes=["z"] * len(ks),
            ijk=[(i0, j0, k) for k in ks],
            signs=[1] * len(ks),
            dls=[float(grid.dz[k]) for k in ks],
            flat_indices=[n_Ex + n_Ey + edge_index_Ez(i0, j0, k, Nx, Ny, Nz) for k in ks],
        )
        for flat in path.flat_indices:
            mesh.pec_mask_edges[2, flat - n_Ex - n_Ey] = True
        apply_thin_wire_path(mesh, path, a, name="wire")
        x0, y0 = float(grid.x[i0]), float(grid.y[j0])
        return AnalysisScatteringTD(
            mesh=mesh.with_boundary_conditions(
                {f: "PEC" for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
            ),
            ports=[
                PortSpecLumped(
                    name="feed",
                    start=(x0, y0, grid.z[K_FEED]),
                    end=(x0, y0, grid.z[K_FEED + 1]),
                    Z0=duct_z0(a),
                )
            ],
            f_max=5e9,
            verbose=False,
            project=project,
        )

    n1, n_total = 120, 300
    ref = _build().run(excited=["feed"], energy_stop_db=None, total_time_steps=n_total)
    rv, ri = (s.values.copy() for s in ref.signals[("feed", 0)][("feed", 0)])

    p = tmp_path / "wire"
    _build(project=p).run(
        excited=["feed"], energy_stop_db=None, total_time_steps=n1, checkpoint_interval=40
    )
    assert open_project(p).runs["feed_mode0"].n_steps == n1

    proj = resume(p, excited=("feed", 0), total_time_steps=n_total, verbose=False)
    gv, gi = proj.signals[("feed", 0)][("feed", 0)]
    np.testing.assert_array_equal(rv, gv.values)
    np.testing.assert_array_equal(ri, gi.values)
