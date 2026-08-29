"""DD-172: lumped ports and elements on symmetry planes.

The user declares the full-model device (endpoints, Z0, companion
values); the builder relates the edge chain to every declared symmetry
plane, clips a crossing chain to the meshed half, scales the internal
device, and stamps a ``LumpedPortReport`` so the shared recorder /
injection plumbing restores full-model power semantics.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from magnelio.boundaries.boundary_conditions import BoundaryConditions
from magnelio.circuit.companion import ParallelRLC, SeriesRLC
from magnelio.circuit.element import LumpedElement
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.ports._lumped.factory import (
    PortSpecLumped,
    build_lumped_element,
    build_lumped_port,
)
from magnelio.ports._lumped.port_report import LumpedPortReport

H = 1e-3  # uniform cell size of the test grids


def _mesh(**bc) -> Mesh:
    """9-line uniform half-model grid [0, 8 mm]^3 with the given BCs."""
    ax = np.arange(9) * H
    return Mesh.from_grid(
        GridLines(ax.copy(), ax.copy(), ax.copy()),
        boundary_conditions=BoundaryConditions(**bc) if bc else None,
    )


def _m_eps(mesh: Mesh) -> np.ndarray:
    g = mesh.grid
    n = (
        g.Nx * (g.Ny + 1) * (g.Nz + 1)
        + (g.Nx + 1) * g.Ny * (g.Nz + 1)
        + (g.Nx + 1) * (g.Ny + 1) * g.Nz
    )
    return np.ones(n)


def _port(mesh, start, end, Z0=50.0, element=None):
    spec = PortSpecLumped(name="p", start=start, end=end, Z0=Z0, element=element)
    return build_lumped_port(spec, mesh, _m_eps(mesh), None, dt=1e-12)


def _element(mesh, start, end, element):
    spec = LumpedElement(name="e", start=start, end=end, element=element)
    return build_lumped_element(spec, mesh, _m_eps(mesh), None, dt=1e-12)


MID = 4 * H  # chain position well inside the domain


class TestPECCrossing:
    """Chain along the plane normal, bisected by an electric plane."""

    def test_symmetric_crossing_is_clipped_and_booked(self):
        mesh = _mesh(zmin=("SymmetryPEC", 0.0))
        op = _port(mesh, (MID, MID, -2 * H), (MID, MID, 2 * H))
        # Clipped to [0, 2 mm]: two z-edges starting at the wall node.
        assert [ijk[2] for ijk in op.ijk_list] == [0, 1]
        assert op.port_report.symmetry_faces == (("zmin", "PEC", "crossing"),)
        # Series cut: the meshed half carries half the device.
        assert op.Z0 == pytest.approx(25.0)
        assert op.element.R == pytest.approx(25.0)
        assert op.port_report.power_wave_full_scale == pytest.approx(math.sqrt(2.0))

    def test_asymmetric_crossing_is_rejected(self):
        mesh = _mesh(zmin=("SymmetryPEC", 0.0))
        with pytest.raises(ValueError, match="asymmetric"):
            _port(mesh, (MID, MID, -1 * H), (MID, MID, 3 * H))

    def test_pmc_crossing_is_rejected(self):
        mesh = _mesh(zmin=("SymmetryPMC", 0.0))
        with pytest.raises(ValueError, match="magnetic symmetry"):
            _port(mesh, (MID, MID, -2 * H), (MID, MID, 2 * H))

    def test_chain_in_discarded_half_is_rejected(self):
        mesh = _mesh(zmin=("SymmetryPEC", 0.0))
        with pytest.raises(ValueError, match="removed by the symmetry"):
            _port(mesh, (MID, MID, -3 * H), (MID, MID, -1 * H))

    def test_terminal_on_clip_plane_is_rejected_with_guidance(self):
        mesh = _mesh(zmin=("SymmetryPEC", 0.0))
        with pytest.raises(ValueError, match="full-model coordinates"):
            _port(mesh, (MID, MID, 0.0), (MID, MID, 2 * H))


class TestAsBuiltDeclaration:
    """ForceSymmetry*: the geometry is declared halved."""

    def test_terminal_on_plane_books_the_crossing(self):
        mesh = _mesh(zmin="ForceSymmetryPEC")
        op = _port(mesh, (MID, MID, 0.0), (MID, MID, 2 * H))
        assert [ijk[2] for ijk in op.ijk_list] == [0, 1]
        assert op.port_report.symmetry_faces == (("zmin", "PEC", "crossing"),)
        assert op.Z0 == pytest.approx(25.0)

    def test_terminal_on_pmc_plane_is_rejected(self):
        mesh = _mesh(zmin="ForceSymmetryPMC")
        with pytest.raises(ValueError, match="magnetic"):
            _port(mesh, (MID, MID, 0.0), (MID, MID, 2 * H))


class TestContainment:
    """Chain lying in a symmetry plane (parallel cut)."""

    def test_pmc_containment_doubles_the_internal_device(self):
        mesh = _mesh(xmin=("SymmetryPMC", 0.0))
        op = _element(
            mesh,
            (0.0, MID, 2 * H),
            (0.0, MID, 4 * H),
            SeriesRLC(R=100.0, L=2e-9, C=1e-12),
        )
        assert op.port_report.symmetry_faces == (("xmin", "PMC", "containment"),)
        assert op.element.R == pytest.approx(200.0)
        assert op.element.L == pytest.approx(4e-9)
        assert op.element.C == pytest.approx(0.5e-12)

    def test_pec_containment_is_rejected(self):
        mesh = _mesh(xmin=("SymmetryPEC", 0.0))
        with pytest.raises(ValueError, match="electric symmetry plane"):
            _element(mesh, (0.0, MID, 2 * H), (0.0, MID, 4 * H), SeriesRLC(R=100.0))

    def test_terminal_beyond_a_parallel_plane_is_rejected(self):
        mesh = _mesh(xmin=("SymmetryPMC", 0.0))
        with pytest.raises(ValueError, match="removed by the symmetry"):
            _element(mesh, (-2 * H, MID, 2 * H), (-2 * H, MID, 4 * H), SeriesRLC(R=100.0))


class TestAwayFromPlane:
    def test_mirror_twin_warning(self):
        mesh = _mesh(xmin=("SymmetryPMC", 0.0))
        with pytest.warns(UserWarning, match="mirror twin"):
            op = _port(mesh, (MID, MID, 2 * H), (MID, MID, 4 * H))
        assert op.port_report is None
        assert op.Z0 == pytest.approx(50.0)

    def test_no_symmetry_keeps_the_historic_path(self):
        mesh = _mesh()
        op = _port(mesh, (MID, MID, 2 * H), (MID, MID, 4 * H))
        assert op.port_report is None
        assert op.Z0 == pytest.approx(50.0)
        assert op.element.R == pytest.approx(50.0)


class TestComposition:
    def test_crossing_plus_containment(self):
        mesh = _mesh(zmin=("SymmetryPEC", 0.0), xmin=("SymmetryPMC", 0.0))
        op = _port(mesh, (0.0, MID, -2 * H), (0.0, MID, 2 * H))
        assert op.port_report.symmetry_faces == (
            ("xmin", "PMC", "containment"),
            ("zmin", "PEC", "crossing"),
        )
        # Series ×0.5 and parallel ×2 cancel in the internal impedance;
        # the power-wave scale composes to 2.
        assert op.Z0 == pytest.approx(50.0)
        assert op.port_report.power_wave_full_scale == pytest.approx(2.0)

    def test_parallel_rlc_scales_like_series(self):
        mesh = _mesh(zmin=("SymmetryPEC", 0.0))
        op = _port(
            mesh,
            (MID, MID, -2 * H),
            (MID, MID, 2 * H),
            element=ParallelRLC(R=100.0, C=4e-12),
        )
        assert isinstance(op.element, ParallelRLC)
        assert op.element.R == pytest.approx(50.0)
        assert op.element.C == pytest.approx(8e-12)
        assert op.element.L is None


class TestPlumbing:
    """The shared port_report consumers pick the scales up unchanged."""

    def test_report_scale_properties(self):
        r = LumpedPortReport(
            symmetry_faces=(("zmin", "PEC", "crossing"), ("xmin", "PMC", "containment")),
        )
        assert r.z_internal_scale == pytest.approx(1.0)
        assert r.z_full_scale == pytest.approx(1.0)
        assert r.power_wave_full_scale == pytest.approx(2.0)
        single = LumpedPortReport(symmetry_faces=(("zmin", "PEC", "crossing"),))
        assert single.z_internal_scale == pytest.approx(0.5)
        assert single.z_full_scale == pytest.approx(2.0)

    def test_excitation_scale_is_the_inverse_wave_scale(self):
        from magnelio.analysis.time_domain import _excitation_scale

        mesh = _mesh(zmin=("SymmetryPEC", 0.0))
        op = _port(mesh, (MID, MID, -2 * H), (MID, MID, 2 * H))
        assert _excitation_scale(op) == pytest.approx(1.0 / math.sqrt(2.0))

    def test_recorder_applies_the_full_model_scale(self):
        from magnelio.ports.recorder import PortSignalRecorder

        mesh = _mesh(zmin=("SymmetryPEC", 0.0))
        op = _port(mesh, (MID, MID, -2 * H), (MID, MID, 2 * H))
        rec = PortSignalRecorder(dt=1e-12, ports=[op])
        assert rec._scales[0] == pytest.approx(math.sqrt(2.0))

    def test_recorder_keeps_none_without_symmetry(self):
        from magnelio.ports.recorder import PortSignalRecorder

        mesh = _mesh()
        op = _port(mesh, (MID, MID, 2 * H), (MID, MID, 4 * H))
        rec = PortSignalRecorder(dt=1e-12, ports=[op])
        assert rec._scales[0] is None
