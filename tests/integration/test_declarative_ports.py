"""Integration tests for the declarative high-level ports (WP4.1).

``PortWaveguide`` / ``PortAnalytical`` resolve into concrete component-level
specs inside ``AnalysisScatteringTD.__post_init__`` — after BC-PEC
consolidation, so boundary-condition walls count as conductors.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio import AnalysisScatteringTD, Material, Mesh
from magnelio.boundaries import PECBoundary
from magnelio.mesh import BoxFace
from magnelio.mesh.grid import GridLines
from magnelio.ports import (
    PortAnalytical,
    PortSpecCoax,
    PortSpecMultiConductor,
    PortSpecNumerical,
    PortSpecRectWG,
    PortWaveguide,
)
from magnelio.ports._modal import ModeType
from magnelio.ports.declarative import normalize_box_face

WR90_A = 22.86e-3
WR90_B = 10.16e-3
ETA0 = 376.730313668


def _wr90_mesh() -> Mesh:
    return Mesh.from_grid(
        GridLines(
            x=np.linspace(0.0, 30e-3, 31),
            y=np.linspace(0.0, WR90_A, 24),
            z=np.linspace(0.0, WR90_B, 11),
        )
    )


def _parallel_plate_analysis(ports) -> AnalysisScatteringTD:
    grid = GridLines(
        x=np.linspace(-5e-3, 5e-3, 11),
        y=np.linspace(-2.5e-3, 2.5e-3, 6),
        z=np.linspace(-10e-3, 10e-3, 41),
    )
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
        ports=ports,
        f_max=10e9,
        verbose=False,
    )


def test_normalize_box_face_forms():
    assert normalize_box_face(BoxFace.Z_MIN) is BoxFace.Z_MIN
    assert normalize_box_face("zmin") is BoxFace.Z_MIN
    assert normalize_box_face("z_min") is BoxFace.Z_MIN
    assert normalize_box_face("X_MAX") is BoxFace.X_MAX
    with pytest.raises(ValueError, match="unknown port plane"):
        normalize_box_face("front")
    with pytest.raises(TypeError, match="BoxFace"):
        normalize_box_face(3)


def test_hollow_face_resolves_to_te_numerical_spec():
    """WR-90 with BC-PEC walls: one wall ring → unified TE/TM port.

    ``mode_type=None`` is the WP-R3 unified multi-mode port: both
    families solved, n_modes lowest cut-offs kept — for one mode on
    WR-90 that is TE10, asserted on the solved report below."""
    analysis = AnalysisScatteringTD(
        mesh=_wr90_mesh().with_boundary_conditions(
            {f: PECBoundary(f) for f in ("ymin", "ymax", "zmin", "zmax")}
        ),
        ports=[
            PortWaveguide(name="port1", plane="xmin"),
            PortWaveguide(name="port2", plane=BoxFace.X_MAX),
        ],
        f_max=12.4e9,
        f_min=8.2e9,
        verbose=False,
    )
    spec = analysis.ports[0]
    assert isinstance(spec, PortSpecNumerical)
    assert spec.mode_type is None
    assert spec.epsilon_r == 1.0
    assert spec.plane is BoxFace.X_MIN

    # The resolved analysis solves the same TE10 as the explicit spec.
    rep = analysis.solve_ports()["port1"]
    f_c_ref = 299_792_458.0 / (2.0 * WR90_A)
    np.testing.assert_allclose(rep.cutoff_num, f_c_ref, rtol=2e-3)
    assert rep.modes[0].mode_type is ModeType.TE


def test_two_conductor_homogeneous_resolves_to_tem():
    """Parallel plate (BC-PEC plates, vacuum): TEM Laplace with scalar ε."""
    analysis = _parallel_plate_analysis(
        [PortWaveguide(name="port1", plane="zmin"), PortWaveguide(name="port2", plane="zmax")],
    )
    spec = analysis.ports[0]
    assert isinstance(spec, PortSpecMultiConductor)
    assert spec.epsilon_r == 1.0  # homogeneous vacuum → TEM path
    assert spec.conductors is None  # auto-detected groups

    rep = analysis.solve_ports()["port1"]
    # Effective width w + dx = 11 mm: from_grid keeps the outermost
    # lines, the magnetic walls sit half an x-cell beyond them.
    np.testing.assert_allclose(
        rep.z_line_num,
        ETA0 * 5e-3 / 11e-3,
        rtol=1e-6,
    )


def test_inhomogeneous_two_conductor_resolves_to_qtem():
    """Half-filled parallel plate: inhomogeneous ε → QTEM (epsilon_r=None)."""
    grid = GridLines(
        x=np.linspace(-5e-3, 5e-3, 11),
        y=np.linspace(-2.5e-3, 2.5e-3, 6),
        z=np.linspace(-10e-3, 10e-3, 21),
    )
    diel = Material.from_isotropic("diel", epsilon=4.0)
    mesh = Mesh.from_grid(
        grid,
        regions=[(diel, (-5e-3, -2.5e-3, -10e-3, 5e-3, 0.0, 10e-3))],
    )
    analysis = AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions(
            {
                "xmin": "PMC",
                "xmax": "PMC",
                "ymin": "PEC",
                "ymax": "PEC",
                "zmin": "PEC",
                "zmax": "PEC",
            }
        ),
        ports=[
            PortWaveguide(name="port1", plane="zmin"),
            PortWaveguide(name="port2", plane="zmax"),
        ],
        f_max=10e9,
        verbose=False,
    )
    spec = analysis.ports[0]
    assert isinstance(spec, PortSpecMultiConductor)
    assert spec.epsilon_r is None  # QTEM dual-Laplace dispatch

    # QTEM z_line lands between the vacuum and the fully-filled value.
    rep = analysis.solve_ports()["port1"]
    z_vac = ETA0 * 0.5
    assert z_vac / 2.0 < rep.z_line_num < z_vac


def test_port_analytical_maps_to_closed_form_specs():
    coax = PortAnalytical(
        name="p",
        plane="zmin",
        family="coax",
        inner_radius=0.5e-3,
        outer_radius=2.5e-3,
        epsilon_r=9.0,
        n_modes=1,
    )
    from magnelio.ports.declarative import resolve_declarative_port

    mesh = _wr90_mesh()  # mesh is irrelevant for the analytical mapping
    spec = resolve_declarative_port(coax, mesh)
    assert isinstance(spec, PortSpecCoax)
    assert (spec.inner_radius, spec.outer_radius) == (0.5e-3, 2.5e-3)
    assert spec.epsilon_r == 9.0 and spec.plane is BoxFace.Z_MIN

    rect = PortAnalytical(
        name="p",
        plane="xmin",
        family="rect_wg",
        width=WR90_A,
        height=WR90_B,
    )
    spec = resolve_declarative_port(rect, mesh)
    assert isinstance(spec, PortSpecRectWG)
    assert (spec.width_a, spec.height_b) == (WR90_A, WR90_B)


def test_declarative_validation():
    with pytest.raises(ValueError, match="inner_radius"):
        PortAnalytical(name="p", plane="zmin", family="coax")
    with pytest.raises(ValueError, match="width"):
        PortAnalytical(name="p", plane="zmin", family="rect_wg")
    with pytest.raises(ValueError, match="unknown analytical port family"):
        PortAnalytical(name="p", plane="zmin", family="microstrip")
    with pytest.raises(ValueError, match="unknown port plane"):
        PortWaveguide(name="p", plane="zmid")
    with pytest.raises(ValueError, match="n_modes"):
        PortWaveguide(name="p", plane="zmin", n_modes=0)

    # A malformed corners spec (flat 4-tuple instead of two corner points)
    with pytest.raises(ValueError, match="two opposite 3D corner points"):
        PortWaveguide(
            name="p",
            plane="zmin",
            corners=(0.0, 0.0, 1e-3, 1e-3),
        )


def test_declarative_end_to_end_sparams():
    """Full run through PortWaveguide: matched TEM line, |S11| < −60 dB."""
    analysis = _parallel_plate_analysis(
        [PortWaveguide(name="port1", plane="zmin"), PortWaveguide(name="port2", plane="zmax")],
    )
    f_axis = np.linspace(0.25e9, 10e9, 41)
    result = analysis.run(f_axis=f_axis, excited=["port1"])
    s11_db = result.db("port1", "port1")
    s21_db = result.db("port2", "port1")
    assert np.nanmax(s11_db) < -60.0
    assert np.nanmax(np.abs(s21_db)) < 0.1
