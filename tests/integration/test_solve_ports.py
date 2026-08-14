"""Integration tests for AnalysisScatteringTD.solve_ports (WP5.1, F5).

``solve_ports()`` exposes the per-port 2D mode solutions —
z_line (numerical + analytical reference), cut-offs, mode types, and
transverse-profile plots — without running a time-domain simulation.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from magnelio import (
    AnalysisScatteringTD,
    Mesh,
)
from magnelio.boundaries import CPMLBoundary, PECBoundary
from magnelio.mesh import BoxFace
from magnelio.mesh.grid import GridLines
from magnelio.ports import PortReport, PortSpecLumped, PortSpecMultiConductor, PortSpecRectWG
from magnelio.ports._modal import ModeType

WR90_A = 22.86e-3
WR90_B = 10.16e-3
ETA0 = 376.730313668


def _wr90_analysis() -> AnalysisScatteringTD:
    grid = GridLines(
        x=np.linspace(0.0, 30e-3, 31),
        y=np.linspace(0.0, WR90_A, 24),
        z=np.linspace(0.0, WR90_B, 11),
    )
    specs = [
        PortSpecRectWG(
            name="port1",
            plane=BoxFace.X_MIN,
            width_a=WR90_A,
            height_b=WR90_B,
            n_modes=1,
        ),
        PortSpecRectWG(
            name="port2",
            plane=BoxFace.X_MAX,
            width_a=WR90_A,
            height_b=WR90_B,
            n_modes=1,
        ),
    ]
    return AnalysisScatteringTD(
        mesh=Mesh.from_grid(
            grid, boundary_conditions={f: PECBoundary(f) for f in ("ymin", "ymax", "zmin", "zmax")}
        ),
        ports=specs,
        f_max=12.4e9,
        f_min=8.2e9,
        verbose=False,
    )


def _parallel_plate_analysis() -> AnalysisScatteringTD:
    grid = GridLines(
        x=np.linspace(-5e-3, 5e-3, 11),
        y=np.linspace(-2.5e-3, 2.5e-3, 6),
        z=np.linspace(-10e-3, 10e-3, 21),
    )
    specs = [
        PortSpecMultiConductor(name="port1", plane=BoxFace.Z_MIN, n_modes=1),
        PortSpecMultiConductor(name="port2", plane=BoxFace.Z_MAX, n_modes=1),
    ]
    return AnalysisScatteringTD(
        mesh=Mesh.from_grid(
            grid,
            boundary_conditions={
                "xmin": "PMC",
                "xmax": "PMC",
                "ymin": "PEC",
                "ymax": "PEC",
                "zmin": "PEC",
                "zmax": "PEC",
            },
        ),
        ports=specs,
        f_max=10e9,
        verbose=False,
    )


def test_solve_ports_rectwg_cutoff_and_types():
    """WR-90 report carries both cut-off paths and the TE10 mode entry."""
    reports = _wr90_analysis().solve_ports()
    assert set(reports) == {"port1", "port2"}

    rep = reports["port1"]
    assert isinstance(rep, PortReport)
    f_c_ref = 299_792_458.0 / (2.0 * WR90_A)

    # Reference path: analytical TE10 cut-off, exact.
    np.testing.assert_allclose(rep.cutoff_ref, f_c_ref, rtol=1e-9)
    # Numerical path: grid-dispersion limited.
    np.testing.assert_allclose(rep.cutoff_num, f_c_ref, rtol=2e-3)
    # Hollow pipe: no line impedance on either path.
    assert rep.z_line_num is None and rep.z_line_ref is None

    assert len(rep.modes) == 1
    m = rep.modes[0]
    assert m.mode_type is ModeType.TE
    assert m.port_name == "port1"
    np.testing.assert_allclose(m.f_cutoff, rep.cutoff_num, rtol=1e-12)
    assert m.z_line is None
    # Frequency relations delegate to the underlying Mode: above
    # cut-off the TE wave impedance is real and > eta0.
    z = m.z_modal(10e9)
    assert z.imag == pytest.approx(0.0, abs=1e-9)
    assert z.real > ETA0
    # Propagating at 10 GHz: gamma purely imaginary.
    g = m.gamma(10e9)
    assert g.real == pytest.approx(0.0, abs=1e-12) and g.imag > 0.0


def test_solve_ports_tem_z_line_matches_analytic():
    """Parallel-plate TEM z_line = η₀·b/w_eff, machine-exact.

    ``from_grid`` keeps the outermost lines in place, so the magnetic
    walls sit half an x-cell beyond them: the simulated width is
    ``w + dx`` (11 mm here), and the PMC-aware capacitance booking
    reports exactly that line's impedance.
    """
    reports = _parallel_plate_analysis().solve_ports()
    rep = reports["port1"]

    z_analytic = ETA0 * 5e-3 / 11e-3  # 171.24 Ω
    np.testing.assert_allclose(rep.z_line_num, z_analytic, rtol=1e-9)
    assert rep.cutoff_num is None  # TEM path reports z_line, not cut-off

    m = rep.modes[0]
    assert m.mode_type is ModeType.TEM
    assert m.f_cutoff == 0.0
    np.testing.assert_allclose(m.z_line, rep.z_line_num, rtol=1e-12)
    # TEM z_modal is the line impedance at any frequency.
    np.testing.assert_allclose(
        complex(m.z_modal(1e9)),
        complex(m.z_line),
        rtol=1e-12,
    )

    # summary() renders without error and names the key numbers.
    text = str(rep)
    assert "port1" in text and "z_line" in text and "TEM" in text


def test_solve_ports_lumped_port_reports_z0():
    """A PortSpecLumped yields an empty mode tuple and z_line = Z0."""
    grid = GridLines(
        x=np.linspace(0, 3e-3, 4),
        y=np.linspace(0, 3e-3, 4),
        z=np.linspace(0, 30e-3, 31),
    )
    spec = PortSpecLumped(
        name="p1",
        start=(1.5e-3, 1.5e-3, 0.0),
        end=(1.5e-3, 1.5e-3, 1e-3),
        Z0=50.0,
    )
    bcs = {f: PECBoundary(f) for f in ("xmin", "xmax", "ymin", "ymax", "zmin")}
    bcs["zmax"] = CPMLBoundary("zmax", grid, thickness_cells=8)
    analysis = AnalysisScatteringTD(
        mesh=Mesh.from_grid(grid, boundary_conditions=bcs),
        ports=[spec],
        f_max=5e9,
        verbose=False,
    )
    rep = analysis.solve_ports()["p1"]
    assert rep.modes == ()
    assert rep.z_line_num == 50.0


class TestModePlot:
    """modes[m].plot() renders the staggered edge profiles on cell centres."""

    def test_te10_e_and_h_profile(self):
        rep = _wr90_analysis().solve_ports()["port1"]
        m = rep.modes[0]

        fig, ax = m.plot(field="E")
        # One quiver + colourbar on a fresh figure; default title set.
        assert "TE" in ax.get_title() and "port1" in ax.get_title()
        plt.close(fig)

        fig, ax = m.plot(field="H", title="custom")
        assert ax.get_title() == "custom"
        plt.close(fig)

        with pytest.raises(ValueError, match="'E' or 'H'"):
            m.plot(field="Ez")

    def test_avg_nonzero_boundary_destaggering(self):
        """Cell centres next to a conductor keep the live edge's value
        instead of averaging it with the in-conductor zero."""
        from magnelio.ports._modal.mode_report import _avg_nonzero

        a = np.array([1.0, 0.0, 0.0, -2.0])
        b = np.array([3.0, 4.0, 0.0, -4.0])
        np.testing.assert_allclose(_avg_nonzero(a, b), [2.0, 4.0, 0.0, -3.0])

    def test_geometry_overlay_wiring(self):
        """geometry= slices half a boundary cell inward; descending (u, v)
        faces (u x v inward) get the axis swap for the overlay."""

        class _RecordingModel:
            def __init__(self):
                self.calls = []

            def plot_cross_section(self, normal, position, **kw):
                self.calls.append((normal, position, kw["flip"]))

        reports = _wr90_analysis().solve_ports()

        model = _RecordingModel()
        fig, ax = reports["port1"].modes[0].plot(field="E", geometry=model)
        plt.close(fig)
        ((normal, pos, flip),) = model.calls
        assert normal == "x"
        assert pos == pytest.approx(0.5e-3)  # X_MIN at x=0, dx=1 mm
        assert flip is False  # X_MIN: (u, v) = (y, z), ascending

        model = _RecordingModel()
        fig, ax = reports["port2"].modes[0].plot(field="E", geometry=model)
        plt.close(fig)
        ((normal, pos, flip),) = model.calls
        assert normal == "x"
        assert pos == pytest.approx(29.5e-3)  # X_MAX at x=30 mm
        assert flip is True  # X_MAX: (u, v) = (z, y), descending

    def test_tem_profile_points_across_gap(self):
        """Parallel-plate TEM E profile is dominated by the v (=y) component."""
        rep = _parallel_plate_analysis().solve_ports()["port1"]
        m = rep.modes[0]

        from magnelio.ports._modal.mode_report import _edge_grid

        dm = m._discrete
        plane = m._plane
        grid_u, _, _ = _edge_grid(dm.e_u_profile, plane.u_edge_uv)
        grid_v, _, _ = _edge_grid(dm.e_v_profile, plane.v_edge_uv)
        # Port plane is z-normal: u = x (along plates), v = y (gap).
        # The TEM field is E_y: v-magnitude dominates u by orders.
        assert np.abs(grid_v).max() > 100.0 * np.abs(grid_u).max()

        fig, ax = m.plot(ax=plt.subplots()[1])
        plt.close(ax.get_figure())
