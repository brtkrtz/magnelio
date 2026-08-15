"""Integration tests for AnalysisScatteringTD.solve_ports (WP5.1, F5).

``solve_ports()`` exposes the per-port 2D mode solutions —
z_line (numerical + analytical reference), cut-offs, mode types, and
transverse-profile plots — without running a time-domain simulation.
"""

from __future__ import annotations

from types import SimpleNamespace

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


def _graded_z(n: int, ratio: float) -> np.ndarray:
    """WR90 height in ``n`` cells, each ``ratio`` times its predecessor."""
    d = ratio ** np.arange(n)
    return WR90_B * np.concatenate([[0.0], np.cumsum(d)]) / d.sum()


_GRADED_Z = _graded_z(10, 1.15)


def _graded_wr90_analysis() -> AnalysisScatteringTD:
    """WR90 with a transversally graded z-grid (cell sizes spread 3.5x)."""
    grid = GridLines(
        x=np.linspace(0.0, 30e-3, 16),
        y=np.linspace(0.0, WR90_A, 24),
        z=_GRADED_Z,
    )
    return AnalysisScatteringTD(
        mesh=Mesh.from_grid(
            grid, boundary_conditions={f: PECBoundary(f) for f in ("ymin", "ymax", "zmin", "zmax")}
        ),
        ports=[
            PortSpecRectWG(
                name="port1",
                plane=BoxFace.X_MIN,
                width_a=WR90_A,
                height_b=WR90_B,
                n_modes=1,
            )
        ],
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

    def test_symmetry_cut_port_plots_the_full_window(self):
        """A port window cut by a declared symmetry plane is drawn whole:
        the solved half plus its mirror image (DD-154)."""
        import magnelio as mio
        from magnelio import geo
        from magnelio.mesh import MeshControl
        from magnelio.ports import PortWaveguide

        f_max = 12e9
        a, b = 22.86e-3, 10.16e-3
        # WR-90 halved by a magnetic wall through the TE10 field maximum:
        # the E field is even across it, so the mirrored half must repeat
        # the solved values with the same sign.
        model = mio.GeometryModel(
            background=mio.Material.air(),
            boundary_conditions={"xmin": "SymmetryPMC"},
        )
        model.add(
            geo.Brick(
                origin=(-a / 2, 0.0, 0.0),
                size=(a, b, 30e-3),
                material=mio.Material.air(),
            )
        )
        model.add_port(PortWaveguide(name="port1", plane="zmin", n_modes=1))
        mesh = Mesh.from_geometry(model, MeshControl(), f_max=f_max)
        rep = AnalysisScatteringTD(mesh=mesh, f_max=f_max).solve_ports()["port1"]
        mode = rep.modes[0]
        assert len(mode._mirrors) == 1
        assert mode._mirrors[0].kind == "PMC"

        fig, ax = mode.plot(field="E")
        assert "full model" in ax.get_title()
        # The picture spans the full guide width (cell centres, so one
        # cell short of the walls) and is centred on the symmetry plane.
        xlim = ax.get_xlim()
        assert xlim[0] == pytest.approx(-xlim[1], rel=1e-9)
        assert xlim[1] - xlim[0] > 0.9 * a * 1e3
        # ... and it is mirror-symmetric about the wall.
        quiv = ax.collections[0]
        pos = quiv.get_offsets()
        vals = np.hypot(np.asarray(quiv.U), np.asarray(quiv.V))
        left = pos[:, 0] < -1e-9
        right = pos[:, 0] > 1e-9
        assert left.sum() == right.sum() > 0
        order_l = np.lexsort((np.abs(pos[left, 0]), pos[left, 1]))
        order_r = np.lexsort((pos[right, 0], pos[right, 1]))
        np.testing.assert_allclose(vals[left][order_l], vals[right][order_r], rtol=1e-9)
        plt.close(fig)

    def test_symmetry_mirrors_reach_the_geometry_overlay(self):
        """The overlay is mirrored along with the field, so conductors
        and arrows agree on both halves."""
        import magnelio as mio
        from magnelio import geo
        from magnelio.mesh import MeshControl
        from magnelio.ports import PortWaveguide

        class _RecordingModel:
            def __init__(self):
                self.calls = 0

            def plot_cross_section(self, normal, position, **kw):
                self.calls += 1

        f_max = 12e9
        model = mio.GeometryModel(
            background=mio.Material.air(),
            boundary_conditions={"xmin": "SymmetryPMC"},
        )
        model.add(
            geo.Brick(
                origin=(-11.43e-3, 0.0, 0.0),
                size=(22.86e-3, 10.16e-3, 30e-3),
                material=mio.Material.air(),
            )
        )
        model.add_port(PortWaveguide(name="port1", plane="zmin", n_modes=1))
        mesh = Mesh.from_geometry(model, MeshControl(), f_max=f_max)
        rep = AnalysisScatteringTD(mesh=mesh, f_max=f_max).solve_ports()["port1"]
        recorder = _RecordingModel()
        fig, ax = rep.modes[0].plot(field="E", geometry=recorder)
        # One image per half-space: the solved side and its mirror.
        assert recorder.calls == 2
        plt.close(fig)

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

    def test_profiles_are_grid_quantities_not_field_samples(self):
        """The 2D solvers return FIT grid quantities — the edge voltage
        ``ê = E·l``, not a field sample.  On a graded transversal grid
        the two differ by the local cell size: TE10's E_z is uniform
        along z, its raw DoF ramps with dz over the full grading."""
        rep = _graded_wr90_analysis().solve_ports()["port1"]
        m = rep.modes[0]
        plane = m._plane
        # Port plane is x-normal: u = y (broad wall), v = z (graded).
        uv = plane.v_edge_uv
        us, vs = np.unique(uv[:, 0]), np.unique(uv[:, 1])
        dz = np.diff(_GRADED_Z)
        grading = dz.max() / dz.min()
        assert grading > 3.0  # the grid actually asks the question

        def row(values):
            g = np.zeros((us.size, vs.size))
            g[np.searchsorted(us, uv[:, 0]), np.searchsorted(vs, uv[:, 1])] = values
            return np.abs(g[us.size // 2])

        raw_e = row(m._discrete.e_v_profile)
        phys_e = row(m._field_profiles("E")[1])
        assert raw_e.max() / raw_e.min() == pytest.approx(grading, rel=1e-6)
        assert phys_e.max() / phys_e.min() == pytest.approx(1.0, abs=1e-9)

        # H carries the dual voltage ĥ = H·l_dual with its own per-face
        # length.  |H_u/E_v| is spatially constant for a TE mode, which
        # only the physical profiles reproduce.
        ratio_raw = row(m._discrete.h_u_profile) / raw_e
        ratio_phys = row(m._field_profiles("H")[0]) / phys_e
        assert ratio_raw.max() / ratio_raw.min() == pytest.approx(grading, rel=1e-6)
        assert ratio_phys.max() / ratio_phys.min() == pytest.approx(1.0, abs=1e-9)

    def test_analytical_families_keep_their_sampled_profiles(self):
        """Closed-form mode families sample V/m and A/m at the midpoints
        already — those must pass through the metric untouched."""
        from magnelio.ports._modal.discrete import DiscreteMode
        from magnelio.ports._modal.mode_report import ModeReport

        e_u, e_v = np.array([1.0, 2.0]), np.array([3.0])
        report = ModeReport(
            port_name="p",
            name="analytic",
            mode_type=ModeType.TEM,
            f_cutoff=0.0,
            z_line=50.0,
            _discrete=DiscreteMode(
                mode=SimpleNamespace(field_evaluator=lambda u, v: None),
                e_u_profile=e_u,
                e_v_profile=e_v,
                h_u_profile=e_v,
                h_v_profile=e_u,
            ),
            _plane=None,
            _h_dual_lengths=(np.array([7.0]), np.array([7.0, 7.0])),
        )
        got_u, got_v = report._field_profiles("E")
        np.testing.assert_array_equal(got_u, e_u)
        np.testing.assert_array_equal(got_v, e_v)
        np.testing.assert_array_equal(report._field_profiles("H")[0], e_v)
