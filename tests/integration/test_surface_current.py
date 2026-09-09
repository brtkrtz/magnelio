"""The surface current against what it must agree with (DD-273).

Two claims, both on a coaxial line whose answers are known: the current
that circles a conductor is the line current, and the loss that current
dissipates is the one :class:`MonitorWallLoss` integrates during the
march — the two come from one booking, so the second is exact and the
first is as accurate as the wall sampling itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio import AnalysisScatteringTD, Material, MeshControl
from magnelio.geo import Cylinder, Difference, GeometryModel
from magnelio.mesh.mesher import Mesh
from magnelio.monitors import MonitorFieldFrequency, MonitorWallLoss
from magnelio.ports import PortWaveguide
from magnelio.post.wall_loss import surface_resistance

A, B, LZ = 1.0e-3, 3.45e-3, 20e-3  # air coax, eta0/2pi*ln(b/a) = 74.3 ohm
F = np.array([6e9])
F_MAX = 12e9
SIGMA = 5.8e7  # copper


@pytest.fixture(scope="module")
def coax_run():
    air, pec = Material.air(), Material.pec()
    outer = Cylinder(origin=(0, 0, 0), radius=B, height=LZ, axis=(0, 0, 1), material=air)
    inner = Cylinder(origin=(0, 0, 0), radius=A, height=LZ, axis=(0, 0, 1), material=pec)
    model = GeometryModel(background=pec)
    model.add(Difference(outer, inner))
    model.add(inner)
    mesh = Mesh.from_geometry(
        model,
        MeshControl(min_nodes_per_wavelength=12, max_cell_size=0.35e-3),
        f_max=F_MAX,
    )
    fields = MonitorFieldFrequency(name="H", freqs=F, fields=["H"])
    wall = MonitorWallLoss(freqs=F, normal="z", position=LZ / 2, sigma=SIGMA, name="wall")
    analysis = AnalysisScatteringTD(
        mesh=mesh,
        ports=[
            PortWaveguide(name="p1", plane="zmin", n_modes=1),
            PortWaveguide(name="p2", plane="zmax", n_modes=1),
        ],
        f_max=F_MAX,
        verbose=False,
        backend="numpy",
        geometry=model,
    )
    analysis.monitors = (fields, wall)
    result = analysis.run(f_axis=F)
    return mesh, fields, wall, result


def _split(js):
    """Inner and outer conductor, by which way the normal faces the axis."""
    r = np.linalg.norm(js.positions[:, :2], axis=1)
    radial = js.positions[:, :2] / np.maximum(r, 1e-30)[:, None]
    outward = np.sum(js.normals[:, :2] * radial, axis=1)
    return outward > 0.5, outward < -0.5


def test_the_current_around_a_conductor_is_the_line_current(coax_run):
    """1 W into the line's own reference impedance sets the current exactly."""
    mesh, fields, _wall, result = coax_run
    z0 = float(np.real(result.s_params.reference_impedance("p1")[0]))
    assert 70.0 < z0 < 78.0, "the discrete line should land near eta0/2pi*ln(b/a) = 74.3 ohm"
    exact = np.sqrt(1.0 / z0)
    js = fields.spectrum.surface_current(mesh, exclude_faces=("zmin", "zmax"))
    inner, outer = _split(js)
    for name, sel, tol in (("inner", inner, 0.06), ("outer", outer, 0.09)):
        # The patches tile the whole length, so the area integral is I*LZ.
        current = float(np.sum(js.magnitude(0)[sel] * js.areas[sel])) / LZ
        assert abs(current - exact) / exact < tol, (
            f"{name} conductor: {current:.6f} A against {exact:.6f} A"
        )


def test_the_loss_it_dissipates_is_the_wall_monitor_s(coax_run):
    """One booking, two readings — this one is exact, not approximate."""
    mesh, fields, wall, _result = coax_run
    # The port planes carry the feed's cross-section, not wall: leaving
    # them in overstates the loss by 9.5 % on this fixture.
    js = fields.spectrum.surface_current(mesh, exclude_faces=("zmin", "zmax"))
    r_s = float(surface_resistance(F, SIGMA)[0])
    from_current = js.power_loss(r_s)
    from_monitor = float(np.atleast_1d(wall.power_loss(P_in=1.0)["total"])[0])
    assert from_monitor > 0.0
    np.testing.assert_allclose(from_current, from_monitor, rtol=2e-3)


def test_the_two_conductors_carry_opposite_currents(coax_run):
    """A coax returns its current on the shield: the z components oppose."""
    mesh, fields, _wall, _result = coax_run
    js = fields.spectrum.surface_current(mesh, exclude_faces=("zmin", "zmax"))
    inner, outer = _split(js)
    jz_in = np.real(js.vector(0)[inner][:, 2]).mean()
    jz_out = np.real(js.vector(0)[outer][:, 2]).mean()
    assert jz_in * jz_out < 0.0


def test_the_normals_are_the_conformal_ones(coax_run):
    """Sub-cell normals, not staircase ones.

    On this fixture the inner conductor spans only about six cells, so
    the normals are coarse; the sharp statement (0.99 alignment on a
    resolved cylinder) lives in the unit test.
    """
    mesh, fields, _wall, _result = coax_run
    js = fields.spectrum.surface_current(mesh, exclude_faces=("zmin", "zmax"))
    assert js.conformal
    inner, _outer = _split(js)
    r = np.linalg.norm(js.positions[inner][:, :2], axis=1)
    radial = js.positions[inner][:, :2] / r[:, None]
    alignment = np.sum(js.normals[inner][:, :2] * radial, axis=1)
    assert alignment.mean() > 0.95


def test_the_question_is_asked_once(coax_run):
    """The trap: a feed's cross-section is not a wall, and nothing else says so."""
    mesh, fields, _wall, _result = coax_run
    with pytest.warns(UserWarning, match="meets the domain boundary"):
        loose = fields.spectrum.surface_current(mesh)
    # An explicit empty answer settles it without a warning.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fields.spectrum.surface_current(mesh, exclude_faces=())
    tight = fields.spectrum.surface_current(mesh, exclude_faces=("zmin", "zmax"))
    assert loose.area > tight.area
    r_s = float(surface_resistance(F, SIGMA)[0])
    assert loose.power_loss(r_s) > 1.05 * tight.power_loss(r_s)
