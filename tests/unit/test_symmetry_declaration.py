"""DD-154: symmetry planes are boundary declarations with a domain clip.

A symmetry face is physically a PEC/PMC wall plus the semantic "the
mirror image exists beyond it".  The declaration normalises to the
physical wall type (so every type-dispatching consumer keeps working)
with the symmetry recorded in a separate map, and a declared position
clips the computational domain to the kept half-space — the full
geometry may be modelled, the mirror half is never meshed.

Vocabulary (DD-159): ``"SymmetryPEC"``/``"SymmetryPMC"`` clip at the
plane (0.0, or the position of a ``("SymmetryPEC", pos)`` tuple);
``"ForceSymmetryPEC"``/``"ForceSymmetryPMC"`` declare the domain as
built (no clip).
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio.boundaries.boundary_conditions import (
    BoundaryConditions,
    bc_type_entries,
    resolve_boundary_conditions,
    symmetry_entries,
)
from magnelio.geo import Brick, GeometryModel
from magnelio.materials.material import Material
from magnelio.mesh.mesher import Mesh, MeshControl

F_MAX = 30e9
CONTROL = MeshControl(min_nodes_per_wavelength=6, max_cell_size=1e-3)


class TestDeclaration:
    def test_string_shorthand_normalises_to_physical_type(self):
        bc = BoundaryConditions(xmin="SymmetryPMC", zmax="ForceSymmetryPEC")
        assert bc.xmin == "PMC"
        assert bc.zmax == "PEC"
        # A bare clip string declares the plane at the origin; a Force
        # string declares the domain as built (no clip position).
        assert bc.symmetry == {"xmin": 0.0, "zmax": None}

    def test_tuple_carries_the_plane_position(self):
        bc = BoundaryConditions(xmin=("SymmetryPMC", 1.5e-3))
        assert bc.xmin == "PMC"
        assert bc.symmetry == {"xmin": 1.5e-3}

    def test_type_entries_report_the_physical_wall(self):
        entries = bc_type_entries(
            {"xmin": ("SymmetryPMC", 0.0), "ymax": "ForceSymmetryPEC", "zmin": "SymmetryPEC"},
        )
        assert entries["xmin"] == "PMC"
        assert entries["ymax"] == "PEC"
        assert entries["zmin"] == "PEC"

    def test_symmetry_entries_reads_every_declaration_form(self):
        decl = {
            "xmin": ("SymmetryPEC", 2e-3),
            "ymin": "ForceSymmetryPMC",
            "zmax": "SymmetryPMC",
            "zmin": "PEC",
        }
        assert symmetry_entries(decl) == {"xmin": 2e-3, "ymin": None, "zmax": 0.0}
        resolved = resolve_boundary_conditions(decl)
        assert isinstance(resolved, BoundaryConditions)
        assert symmetry_entries(resolved) == {"xmin": 2e-3, "ymin": None, "zmax": 0.0}
        assert resolved.xmin == "PEC"
        assert resolved.ymin == "PMC"
        assert resolved.zmax == "PMC"
        assert symmetry_entries(None) == {}

    def test_two_parallel_mirror_planes_rejected(self):
        with pytest.raises(ValueError, match="infinite image chain"):
            BoundaryConditions(xmin="SymmetryPMC", xmax="SymmetryPEC")

    def test_malformed_symmetry_declarations_are_rejected(self):
        with pytest.raises(ValueError, match="not valid"):
            BoundaryConditions(xmin="SymmetryCPML")
        with pytest.raises(ValueError, match="symmetry tuple"):
            BoundaryConditions(xmin=("SymmetryPEC",))
        with pytest.raises(ValueError, match="not a number"):
            BoundaryConditions(xmin=("SymmetryPEC", "abc"))
        with pytest.raises(ValueError, match="takes no plane position"):
            BoundaryConditions(xmin=("ForceSymmetryPEC", 0.0))

    def test_direct_symmetry_map_must_name_a_wall_face(self):
        with pytest.raises(ValueError, match="PEC or PMC"):
            BoundaryConditions(zmax="CPML", symmetry={"zmax": None})


def _layered_model(x_min: float, symmetry_decl) -> GeometryModel:
    """Substrate + air stack, x extent [x_min, 4 mm], symmetric in x."""
    substrate = Material(name="sub", epsilon=(4.0, 4.0, 4.0))
    model = GeometryModel(boundary_conditions=symmetry_decl)
    model.add(
        Brick(
            origin=(x_min, 0.0, 0.0),
            size=(4e-3 - x_min, 4e-3, 1e-3),
            material=substrate,
        )
    )
    model.add(
        Brick(
            origin=(x_min, 0.0, 1e-3),
            size=(4e-3 - x_min, 4e-3, 3e-3),
            material=Material.air(),
        )
    )
    return model


class TestDomainClip:
    def test_clip_starts_the_grid_exactly_on_the_plane(self):
        # Bare clip string: plane position defaults to 0.0.
        model = _layered_model(-4e-3, {"xmin": "SymmetryPEC"})
        mesh = Mesh.from_geometry(model, CONTROL, F_MAX)
        assert mesh.grid.x[0] == 0.0
        assert mesh.grid.x[-1] == pytest.approx(4e-3)

    @pytest.mark.parametrize("kind", ["PEC", "PMC"])
    def test_full_model_clipped_equals_half_model_as_built(self, kind):
        full = _layered_model(-4e-3, {"xmin": f"Symmetry{kind}"})
        half = _layered_model(0.0, {"xmin": f"ForceSymmetry{kind}"})
        mesh_full = Mesh.from_geometry(full, CONTROL, F_MAX)
        mesh_half = Mesh.from_geometry(half, CONTROL, F_MAX)
        assert np.array_equal(mesh_full.grid.x, mesh_half.grid.x)
        assert np.array_equal(mesh_full.grid.y, mesh_half.grid.y)
        assert np.array_equal(mesh_full.grid.z, mesh_half.grid.z)
        assert np.array_equal(mesh_full.material_id, mesh_half.material_id)
        assert np.array_equal(mesh_full.pec_mask_edges, mesh_half.pec_mask_edges)

    def test_thin_wire_in_the_discarded_half_is_skipped(self):
        # DD-172 full-model rule: the mirror-half wire is never meshed,
        # like a solid there — instead of dying in the rasteriser.
        from magnelio.geo import Curve, ThinWire

        full = _layered_model(-4e-3, {"xmin": "SymmetryPEC"})
        full.add(
            ThinWire(
                Curve.polyline([(2e-3, 2e-3, 1e-3), (2e-3, 2e-3, 3e-3)]),
                radius=0.05e-3,
                name="kept",
            )
        )
        full.add(
            ThinWire(
                Curve.polyline([(-2e-3, 2e-3, 1e-3), (-2e-3, 2e-3, 3e-3)]),
                radius=0.05e-3,
                name="mirrored",
            )
        )
        half = _layered_model(0.0, {"xmin": "ForceSymmetryPEC"})
        half.add(
            ThinWire(
                Curve.polyline([(2e-3, 2e-3, 1e-3), (2e-3, 2e-3, 3e-3)]),
                radius=0.05e-3,
                name="kept",
            )
        )
        mesh_full = Mesh.from_geometry(full, CONTROL, F_MAX)
        mesh_half = Mesh.from_geometry(half, CONTROL, F_MAX)
        assert np.array_equal(mesh_full.pec_mask_edges, mesh_half.pec_mask_edges)

    def test_pmc_pull_in_lands_the_wall_on_the_plane(self):
        model = _layered_model(-4e-3, {"xmin": "SymmetryPMC"})
        mesh = Mesh.from_geometry(model, CONTROL, F_MAX)
        x0, x1 = mesh.grid.x[0], mesh.grid.x[1]
        assert x0 > 0.0
        # The natural magnetic wall sits half a boundary cell outside
        # the outermost primal line (see boundaries/pmc.py).
        assert x0 - (x1 - x0) / 2.0 == pytest.approx(0.0, abs=1e-15)

    def test_features_in_the_discarded_half_leave_no_planes(self):
        model = _layered_model(-4e-3, {"xmin": ("SymmetryPEC", 0.0)})
        # The blob sits inside the air layer; the overlap audit runs on
        # the full model (a mirrored-half overlap is just as real), so
        # last-wins semantics are fine for this grid-only check.
        model.allow_overlaps = True
        model.add(
            Brick(
                origin=(-3e-3, 1e-3, 2e-3),
                size=(1e-3, 1e-3, 1e-3),
                material=Material(name="blob", epsilon=(9.0, 9.0, 9.0)),
            )
        )
        mesh = Mesh.from_geometry(model, CONTROL, F_MAX)
        assert mesh.grid.x[0] == 0.0
        assert np.all(mesh.grid.x >= 0.0)

    def test_forced_planes_beyond_the_plane_drop_with_a_warning(self):
        control = MeshControl(
            min_nodes_per_wavelength=6,
            max_cell_size=1e-3,
            forced_planes={"x": [-2e-3, 1e-3]},
        )
        model = _layered_model(-4e-3, {"xmin": "SymmetryPEC"})
        with pytest.warns(UserWarning, match="beyond the symmetry plane"):
            mesh = Mesh.from_geometry(model, control, F_MAX)
        assert mesh.grid.x[0] == 0.0
        assert np.any(np.isclose(mesh.grid.x, 1e-3))


class TestRedeclaration:
    def test_with_boundary_conditions_cannot_clip_after_the_fact(self):
        model = _layered_model(0.0, None)
        mesh = Mesh.from_geometry(model, CONTROL, F_MAX)
        with pytest.raises(ValueError, match="re-mesh"):
            mesh.with_boundary_conditions({"xmin": "SymmetryPMC"})

    def test_redeclaring_the_built_closure_is_allowed(self):
        decl = {"xmin": ("SymmetryPMC", 0.0)}
        model = _layered_model(-4e-3, decl)
        mesh = Mesh.from_geometry(model, CONTROL, F_MAX)
        again = mesh.with_boundary_conditions(decl)
        assert symmetry_entries(again.boundary_conditions) == {"xmin": 0.0}

    def test_forced_symmetry_can_be_declared_after_the_fact(self):
        model = _layered_model(0.0, {"xmin": "PMC"})
        mesh = Mesh.from_geometry(model, CONTROL, F_MAX)
        redeclared = mesh.with_boundary_conditions({"xmin": "ForceSymmetryPMC"})
        assert symmetry_entries(redeclared.boundary_conditions) == {"xmin": None}


ETA0 = 376.730313668


class TestPortReportFullModel:
    """DD-154 stage D: half-port reports publish full-model impedances.

    Parallel plate (PEC top/bottom, PMC sides): the TEM line impedance
    is exactly η₀·d/W on the uniform grid, so cutting the window with a
    symmetry plane doubles (PMC cut, half width) or halves (PEC cut,
    half height) the raw solver value — the published report must
    restore the full-model impedance in both cases.
    """

    def _parallel_plate_port(self, W, d, bc):
        from magnelio._operators.material_matrices import build_M_eps, build_M_mu
        from magnelio.mesh.grid import GridLines
        from magnelio.ports._modal import (
            BoxFace,
            PortSpecMultiConductor,
            build_modal_port,
        )
        from magnelio.solver.stability import courant_dt

        grid = GridLines(
            x=np.linspace(0, 10e-3, 11),
            y=np.linspace(0, W, int(round(W / 0.5e-3)) + 1),
            z=np.linspace(0, d, int(round(d / 0.5e-3)) + 1),
        )
        mesh = Mesh.from_grid(grid, boundary_conditions=bc)
        m_eps = build_M_eps(mesh)
        m_mu = build_M_mu(mesh)
        dt = courant_dt(grid, accuracy="normal")
        spec = PortSpecMultiConductor(
            name="pp",
            plane=BoxFace.X_MIN,
            conductors=None,
            epsilon_r=1.0,
            n_modes=1,
        )
        return build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=10e9)

    _BC_FULL = {"xmin": "PMC", "xmax": "PMC", "ymin": "PMC", "ymax": "PMC"}

    def _pp_geometry_port(self, y_extent, bc):
        """from_geometry variant: the PMC pull-in puts the magnetic
        wall exactly on the declared plane, so the half/full impedance
        ratio is exactly 2 (from_grid leaves the wall half a boundary
        cell outside and the ratio picks up that offset)."""
        from magnelio._operators.material_matrices import build_M_eps, build_M_mu
        from magnelio.ports._modal import (
            BoxFace,
            PortSpecMultiConductor,
            build_modal_port,
        )
        from magnelio.solver.stability import courant_dt

        model = GeometryModel(boundary_conditions=bc)
        model.add(
            Brick(
                origin=(0.0, 0.0, 0.0),
                size=(10e-3, y_extent, 2e-3),
                material=Material.air(),
            )
        )
        mesh = Mesh.from_geometry(
            model,
            MeshControl(min_nodes_per_wavelength=6, max_cell_size=0.5e-3),
            10e9,
        )
        m_eps = build_M_eps(mesh)
        m_mu = build_M_mu(mesh)
        dt = courant_dt(mesh.grid, accuracy="normal")
        spec = PortSpecMultiConductor(
            name="pp",
            plane=BoxFace.X_MIN,
            conductors=None,
            epsilon_r=1.0,
            n_modes=1,
        )
        return build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=10e9)

    # CPML on the port-normal faces: an open face without the PMC
    # pull-in, which would break the port's three-equidistant-cells
    # gate right at the port plane.
    _BC_GEO = {"xmin": "CPML", "xmax": "CPML", "ymin": "PMC", "ymax": "PMC"}

    def test_pmc_cut_report_restores_full_model_z_line(self):
        from magnelio.ports import PortReport

        op_full = self._pp_geometry_port(8e-3, dict(self._BC_GEO))
        # The target workflow: full geometry, clipped at the declared
        # symmetry plane through the middle of the parallel plate.
        op_half = self._pp_geometry_port(
            8e-3,
            dict(self._BC_GEO, ymax=("SymmetryPMC", 4e-3)),
        )
        assert op_half.port_report.symmetry_faces == (("ymax", "PMC"),)
        # Raw solver value carries the half-window factor 2 ...
        assert op_half.port_report.z_line_num == pytest.approx(
            2.0 * op_full.port_report.z_line_num,
            rel=1e-9,
        )
        # ... and the published report removes it.
        rep_full = PortReport.from_operator(op_full)
        rep_half = PortReport.from_operator(op_half)
        assert rep_half.z_line_num == pytest.approx(rep_full.z_line_num, rel=1e-9)
        assert rep_half.z_line_num == pytest.approx(ETA0 * 2e-3 / 8e-3, rel=1e-6)
        assert rep_half.modes[0].z_line == pytest.approx(rep_half.z_line_num, rel=1e-12)
        # The Mode object itself keeps the raw normalisation.
        assert op_half.discrete_modes[0].mode.z_line == pytest.approx(
            2.0 * rep_half.z_line_num,
            rel=1e-9,
        )
        assert "full-model" in rep_half.summary()
        assert "full-model" not in rep_full.summary()

    def test_pec_cut_report_restores_full_model_z_line(self):
        from magnelio.ports import PortReport

        op_full = self._parallel_plate_port(8e-3, 2e-3, dict(self._BC_FULL))
        op_half = self._parallel_plate_port(
            8e-3,
            1e-3,
            dict(self._BC_FULL, zmax="ForceSymmetryPEC"),
        )
        assert op_half.port_report.symmetry_faces == (("zmax", "PEC"),)
        assert op_half.port_report.z_line_num == pytest.approx(
            0.5 * op_full.port_report.z_line_num,
            rel=1e-9,
        )
        rep_full = PortReport.from_operator(op_full)
        rep_half = PortReport.from_operator(op_half)
        assert rep_half.z_line_num == pytest.approx(rep_full.z_line_num, rel=1e-9)

    def test_plain_pmc_wall_is_not_a_symmetry_cut(self):
        from magnelio.ports import PortReport

        op = self._parallel_plate_port(8e-3, 2e-3, dict(self._BC_FULL))
        assert op.port_report.symmetry_faces == ()
        assert op.port_report.z_line_full_scale == 1.0
        rep = PortReport.from_operator(op)
        assert rep.z_line_num == pytest.approx(op.port_report.z_line_num)

    def test_port_away_from_the_plane_warns_about_its_mirror_twin(self):
        # The port window (x_min face) touches the four transverse
        # faces but never xmax — declaring the symmetry there leaves
        # the port with a mirror twin in the full model.
        with pytest.warns(UserWarning, match="mirror twin"):
            op = self._parallel_plate_port(
                8e-3,
                2e-3,
                dict(self._BC_FULL, xmax="ForceSymmetryPMC"),
            )
        assert op.port_report.symmetry_faces == ()


class TestFullModelPowerSemantics:
    """DD-155: sources declare full-model watts under symmetry.

    The excitation injects ×1/√2 per plane cutting the port window
    (fields at full-model level, half the full-model power into the
    meshed half-space) and the recorder restores ×√2 per plane on the
    recorded V/I — so a/b, the S-matrix and both store paths all see
    full-model wave amplitudes from the one scale applied at the
    recording layer.
    """

    def test_report_power_wave_scale(self):
        from magnelio.ports._modal.port_report import PortOperatorReport

        assert PortOperatorReport().power_wave_full_scale == 1.0
        one_cut = PortOperatorReport(symmetry_faces=(("ymin", "PMC"),))
        assert one_cut.power_wave_full_scale == pytest.approx(np.sqrt(2.0))
        two_cuts = PortOperatorReport(
            symmetry_faces=(("ymin", "PMC"), ("zmax", "PEC")),
        )
        assert two_cuts.power_wave_full_scale == pytest.approx(2.0)

    class _StubPort:
        def __init__(self, name, symmetry_faces):
            from magnelio.ports._modal.port_report import PortOperatorReport

            self.name = name
            self.n_modes = 1
            self.port_report = PortOperatorReport(symmetry_faces=symmetry_faces)

        def project_V(self, e):
            return np.array([1.0])

        def project_I(self, h):
            return np.array([2.0])

    def _recorded(self, symmetry_faces):
        from magnelio.ports.recorder import PortSignalRecorder

        port = self._StubPort("p1", symmetry_faces)
        rec = PortSignalRecorder(dt=1e-12, ports=[port])
        rec.record(np.zeros(3), np.zeros(3))
        V, I = rec.finalize()[("p1", 0)]
        return float(V.values[0]), float(I.values[0])

    def test_recorder_restores_full_model_amplitudes_on_a_cut_port(self):
        V, I = self._recorded((("ymin", "PMC"),))
        assert V == pytest.approx(np.sqrt(2.0), rel=1e-15)
        assert I == pytest.approx(2.0 * np.sqrt(2.0), rel=1e-15)

    def test_recorder_keeps_uncut_ports_bit_identical(self):
        V, I = self._recorded(())
        assert V == 1.0
        assert I == 2.0

    def test_excitation_scale_is_the_inverse_wave_scale(self):
        from magnelio.analysis.scattering_td import _excitation_scale

        assert _excitation_scale(self._StubPort("p", ())) == 1.0
        assert _excitation_scale(
            self._StubPort("p", (("ymin", "PMC"),)),
        ) == pytest.approx(1.0 / np.sqrt(2.0))
        assert _excitation_scale(object()) == 1.0  # lumped: no report


class TestMirrorRules:
    """DD-154 stage E: sign table and mirroring of plane slices."""

    def test_sign_table(self):
        from magnelio.monitors.base import mirror_sign

        # Across PMC: E continues like a polar vector, H like a
        # pseudovector; across PEC the roles swap.
        assert mirror_sign("E", 0, 0, "PMC") == -1.0
        assert mirror_sign("E", 1, 0, "PMC") == 1.0
        assert mirror_sign("E", 0, 0, "PEC") == 1.0
        assert mirror_sign("E", 1, 0, "PEC") == -1.0
        assert mirror_sign("H", 0, 0, "PMC") == 1.0
        assert mirror_sign("H", 1, 0, "PMC") == -1.0
        assert mirror_sign("H", 0, 0, "PEC") == -1.0
        assert mirror_sign("H", 1, 0, "PEC") == 1.0
        # Magnitudes are mirror-even.
        assert mirror_sign("E", None, 0, "PMC") == 1.0

    def test_mirror_plane_arrays_extends_axis_and_signs(self):
        from magnelio.monitors.base import (
            MirrorSpec,
            PlaneView,
            mirror_plane_arrays,
        )

        cc_x = np.array([0.5, 1.5, 2.5])
        cc_y = np.array([0.25, 0.75])
        pv = PlaneView(
            free=[(0, cc_x), (1, cc_y)],
            normal_idx=2,
            normal_pos=0.0,
            slice_index=None,
        )
        spec = MirrorSpec(axis=0, wall=0.0, kind="PMC", at_low=True)
        vals = np.arange(6.0).reshape(3, 2)
        c0, c1, (ex, ey) = mirror_plane_arrays(
            pv,
            (spec,),
            cc_x,
            cc_y,
            [(vals, "E", 0), (vals, "E", 1)],
        )
        np.testing.assert_allclose(c0, [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
        np.testing.assert_array_equal(c1, cc_y)
        # Ex is normal to the PMC mirror -> odd continuation.
        np.testing.assert_allclose(ex[:3], -vals[::-1])
        np.testing.assert_allclose(ex[3:], vals)
        # Ey is tangential -> even continuation.
        np.testing.assert_allclose(ey[:3], vals[::-1])

    def test_resolve_mirrors_wall_positions(self):
        from magnelio.monitors.base import resolve_mirrors, resolve_region

        mesh_pmc = Mesh.from_geometry(
            _layered_model(0.0, {"xmin": "ForceSymmetryPMC"}),
            CONTROL,
            F_MAX,
        )
        region = resolve_region(None, mesh_pmc.grid)
        (m,) = resolve_mirrors(region, mesh_pmc)
        assert (m.axis, m.kind, m.at_low) == (0, "PMC", True)
        # The mesher pull-in puts the magnetic wall exactly on the
        # declared plane at x = 0.
        assert m.wall == pytest.approx(0.0, abs=1e-15)

        mesh_pec = Mesh.from_geometry(
            _layered_model(0.0, {"xmin": "ForceSymmetryPEC"}),
            CONTROL,
            F_MAX,
        )
        (m_pec,) = resolve_mirrors(
            resolve_region(None, mesh_pec.grid),
            mesh_pec,
        )
        assert m_pec.kind == "PEC"
        assert m_pec.wall == mesh_pec.grid.x[0] == 0.0

    def test_region_short_of_the_plane_is_not_mirrored(self):
        from magnelio.monitors.base import resolve_mirrors, resolve_region

        mesh = Mesh.from_geometry(
            _layered_model(0.0, {"xmin": "ForceSymmetryPEC"}),
            CONTROL,
            F_MAX,
        )
        region = resolve_region(((1e-3, None, None), (None, None, None)), mesh.grid)
        assert resolve_mirrors(region, mesh) == ()

    def test_plot_shows_the_full_model(self):
        import matplotlib

        matplotlib.use("Agg")
        from magnelio.monitors.field_time import MonitorFieldTime

        mesh = Mesh.from_geometry(
            _layered_model(0.0, {"xmin": "ForceSymmetryPMC"}),
            CONTROL,
            F_MAX,
        )
        mon = MonitorFieldTime(times=[0.0], fields=["E"], name="m")
        mon.attach(mesh)
        r = mon.region
        shape = (
            r.ix.stop - r.ix.start,
            r.iy.stop - r.iy.start,
            r.iz.stop - r.iz.start,
        )
        mon._snapshots = [{"Ez": np.ones(shape)}]
        mon._recorded_times = [0.0]
        _fig, ax = mon.plot(
            "Ez",
            normal="z",
            position=2e-3,
            plot_type="color",
        )
        # The mirrored half extends the x axis to ~[-4, 4] mm.
        assert ax.get_xlim()[0] < -3.0
        import matplotlib.pyplot as plt

        plt.close("all")


class TestFluxUnderSymmetry:
    """DD-155: the flux record books full-model watts.

    Every symmetry plane whose axis lies in the cross-section halves
    the meshed aperture, so the record doubles per such plane; a plane
    parallel to the monitor surface leaves the aperture whole.  This is
    source-independent because the sources declare full-model
    amplitudes (the modal excitation injects ×1/√2 per cutting plane);
    pinned end-to-end in
    ``validation/symmetry_full_vs_half_certificate.py``.
    """

    class _Ones:
        def __init__(self, grid):
            Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
            self.Ex = np.ones((Nx, Ny + 1, Nz + 1))
            self.Ey = np.ones((Nx + 1, Ny, Nz + 1))
            self.Ez = np.ones((Nx + 1, Ny + 1, Nz))
            self.Hx = np.ones((Nx + 1, Ny, Nz))
            self.Hy = np.ones((Nx, Ny + 1, Nz))
            self.Hz = np.ones((Nx, Ny, Nz + 1))

    def _flux_power(self, decl):
        from magnelio.monitors.flux import MonitorFluxTime

        mesh = Mesh.from_geometry(_layered_model(0.0, decl), CONTROL, F_MAX)
        mon = MonitorFluxTime(plane=("z", 2e-3), name="f")
        mon.attach(mesh)
        mon.record(self._Ones(mesh.grid), 0, 0.0, 1e-12)
        return float(mon.power[0])

    def test_cutting_plane_doubles_the_record(self):
        p_plain = self._flux_power({"xmin": "PMC"})
        p_sym = self._flux_power({"xmin": "ForceSymmetryPMC"})
        assert p_sym == pytest.approx(2.0 * p_plain, rel=1e-12)

    def test_parallel_plane_leaves_the_record_unscaled(self):
        p_plain = self._flux_power({"zmin": "PMC"})
        p_sym = self._flux_power({"zmin": "ForceSymmetryPMC"})
        assert p_sym == pytest.approx(p_plain, rel=1e-12)


class TestWallLossFullModel:
    """DD-154: wall-loss fractions carry full-model semantics and a
    symmetry face never books as a physical wall."""

    def _monitor(self, decl, *, reference_plane, bc_faces=("zmin",)):
        from magnelio.monitors.wall_loss import MonitorWallLoss

        mesh = Mesh.from_geometry(_layered_model(0.0, decl), CONTROL, F_MAX)
        mon = MonitorWallLoss(
            freqs=[10e9],
            reference_plane=reference_plane,
            sigma=5.8e7,
            bc_faces=bc_faces,
        )
        mon.attach(mesh)
        return mon

    def test_transverse_symmetry_leaves_the_fraction_factor_at_one(self):
        mon = self._monitor(
            {"xmin": "ForceSymmetryPMC"},
            reference_plane=("y", 2e-3),
        )
        assert mon._sym_fraction_factor == 1.0

    def test_parallel_symmetry_doubles_the_fraction_factor(self):
        mon = self._monitor(
            {"xmin": "ForceSymmetryPMC"},
            reference_plane=("x", 2e-3),
        )
        assert mon._sym_fraction_factor == 2.0

    def test_symmetry_face_listed_as_wall_is_dropped_loudly(self):
        with pytest.warns(UserWarning, match="not a physical wall"):
            mon = self._monitor(
                {"xmin": "ForceSymmetryPEC"},
                reference_plane=("y", 2e-3),
                bc_faces=("xmin", "zmin"),
            )
        assert mon.bc_faces == ("zmin",)


class TestOverlayMirroring:
    """DD-154: the geometry overlay shows the simulated half plus its
    mirror image."""

    def _overlay_artists(self, decl, with_geometry):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from magnelio.monitors.field_time import MonitorFieldTime

        model = _layered_model(0.0, decl)
        mesh = Mesh.from_geometry(model, CONTROL, F_MAX)
        mon = MonitorFieldTime(times=[0.0], fields=["E"], name="m")
        mon.attach(mesh)
        r = mon.region
        shape = (
            r.ix.stop - r.ix.start,
            r.iy.stop - r.iy.start,
            r.iz.stop - r.iz.start,
        )
        mon._snapshots = [{"Ez": np.ones(shape)}]
        mon._recorded_times = [0.0]
        _fig, ax = mon.plot(
            "Ez",
            normal="z",
            position=2e-3,
            plot_type="color",
            geometry=model if with_geometry else None,
        )
        n = len(ax.lines) + len(ax.patches) + len(ax.collections)
        plt.close("all")
        return n

    def test_overlay_is_drawn_once_per_mirror_image(self):
        base = self._overlay_artists({"xmin": "PMC"}, with_geometry=False)
        plain = self._overlay_artists({"xmin": "PMC"}, with_geometry=True)
        mirrored = self._overlay_artists({"xmin": "ForceSymmetryPMC"}, with_geometry=True)
        assert plain - base > 0  # the overlay actually drew something
        assert mirrored - base == 2 * (plain - base)


class TestParaViewSymmetry:
    """DD-154 stage F: the generated pipeline mirrors half-model data."""

    def test_symmetry_config_reads_the_stored_mesh(self):
        from types import SimpleNamespace

        from magnelio.io.paraview import _symmetry_config

        mesh = Mesh.from_geometry(
            _layered_model(0.0, {"xmin": "ForceSymmetryPMC"}),
            CONTROL,
            F_MAX,
        )
        (entry,) = _symmetry_config(SimpleNamespace(mesh=mesh))
        axis, wall, at_low, kind = entry
        # The wall type travels with the plane: it decides the sign every
        # field component picks up across it (DD-169).
        assert (axis, at_low, kind) == ("x", True, "PMC")
        assert wall == pytest.approx(0.0, abs=1e-15)
        assert _symmetry_config(SimpleNamespace(mesh=None)) == []

    def test_script_carries_the_reflect_stage(self, tmp_path):
        from magnelio.io.paraview import write_paraview_script

        script = write_paraview_script(
            tmp_path / "paraview_open.py",
            {
                "geometry": None,
                "materials": [],
                "monitors": [],
                "symmetry": [["x", 0.0, True, "PMC"]],
            },
        )
        text = script.read_text(encoding="utf-8")
        compile(text, str(script), "exec")  # the generated file must parse
        assert "simple.Reflect" in text
        assert '"symmetry"' in text or "symmetry" in text
        # Both property sets of the reflection filter must stay in the
        # script: renderers disagree on which one they expose, and the
        # session showed half a model for want of the newer names
        # (DD-169).
        for prop in ("PlaneMode", "ReflectionPlane", "Plane", "Center"):
            assert prop in text
        for flag in ("ReflectAllInputArrays", "FlipAllInputArrays"):
            assert flag in text


class TestStoreRoundTrip:
    def test_symmetry_survives_the_mesh_round_trip(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        from magnelio.io.project import _load_mesh, _save_mesh

        decl = {"xmin": "SymmetryPMC", "ymin": "ForceSymmetryPEC"}
        model = _layered_model(-4e-3, decl)
        mesh = Mesh.from_geometry(model, CONTROL, F_MAX)
        path = tmp_path / "mesh.h5"
        with h5py.File(path, "w") as f:
            _save_mesh(f, mesh)
        with h5py.File(path, "r") as f:
            reloaded = _load_mesh(f)
        assert symmetry_entries(reloaded.boundary_conditions) == {
            "xmin": 0.0,
            "ymin": None,
        }
        assert reloaded.boundary_conditions.xmin == "PMC"
        assert reloaded.boundary_conditions.ymin == "PEC"
