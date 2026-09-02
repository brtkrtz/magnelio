"""The dispersive reference of a quasi-TEM port on the modal pipeline (DD-244).

The tutorial-09 microstrip through the public path: what the result's
reference impedances are, what the port's dispersion sweep says, that
the de-embedding of a quasi-TEM feed removes the line's physical
dispersion (KB-027), and that all of it survives the project store.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import magnelio as mio
from magnelio import geo, ports
from magnelio.post.deembed import deembed_s_params

H_SUB, W_STRIP, T_STRIP = 0.8e-3, 1.2e-3, 0.2e-3
W_BOX, H_BOX, L = 8.0e-3, 5.0e-3, 20.0e-3
F_MAX = 15.0e9


def _microstrip():
    fr4 = mio.Material.from_isotropic(name="FR4", epsilon=4.3)
    substrate = geo.Brick(origin=(-W_BOX / 2, 0.0, 0.0), size=(W_BOX, H_SUB, L), material=fr4)
    air_cap = geo.Brick(
        origin=(-W_BOX / 2, H_SUB, 0.0), size=(W_BOX, H_BOX - H_SUB, L), material="air"
    )
    strip = geo.Brick(origin=(-W_STRIP / 2, H_SUB, 0.0), size=(W_STRIP, T_STRIP, L), material="pec")
    model = mio.GeometryModel(boundary_conditions={"xmin": "SymmetryPMC"})
    model.add(substrate)
    model.add(geo.Difference(air_cap, strip))
    model.add(strip)
    model.add_port(ports.PortWaveguide(name="port1", plane="zmin", n_modes=1))
    model.add_port(ports.PortWaveguide(name="port2", plane="zmax", n_modes=1))
    control = mio.MeshControl(min_nodes_per_wavelength=16)
    mesh = mio.Mesh.from_geometry(model, control, f_max=F_MAX)
    return model, control, mesh


@pytest.fixture(scope="module")
def line():
    model, control, mesh = _microstrip()
    analysis = mio.AnalysisScatteringTD(mesh=mesh, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = analysis.run(excited=["port1", "port2"])
    return model, control, mesh, analysis, result


def test_reference_impedance_is_the_quasi_static_line_impedance(line):
    _, _, _, analysis, result = line
    z_qs = analysis.solve_ports()["port1"].z_line_num
    for port in ("port1", "port2"):
        z = result.reference_impedance(port)
        assert np.allclose(z, z_qs, rtol=1e-12)
    assert result.s_params.reference_impedances is not None


def test_dispersion_sweep_of_the_port(line):
    _, _, _, analysis, _ = line
    report = analysis.solve_ports()["port1"]
    d = report.dispersion(np.array([0.3e9, 5e9, 10e9, 15e9]))
    assert d.n_modes == 1
    # Static limit meets the quasi-static value (full-model scaled).
    assert abs(d.z_line[0, 0] / report.z_line_num - 1.0) < 3e-3
    assert abs(d.epsilon_eff[0, 0] / report.modes[0].epsilon_eff - 1.0) < 3e-3
    # Normal dispersion of the microstrip: ε_eff rises with frequency
    # and the impedance ends above where it started.
    assert np.all(np.diff(d.epsilon_eff[0]) > 0.0)
    assert d.z_line[0, -1] > d.z_line[0, 0]
    assert np.all(np.abs(d.alpha[0]) < 1e-6)
    assert "dispersion" in str(d)


def test_deembedding_removes_the_physical_dispersion(line):
    """KB-027: the quasi-static γ left −22° at 15 GHz on this line."""
    _, _, _, _, result = line
    f = result.f_axis
    k = int(np.argmin(np.abs(f - 15e9)))
    de = result.deembed({"port1": L / 2, "port2": L / 2})
    phase = np.degrees(np.angle(de.S("port2", "port1")))
    assert abs(phase[k]) < 4.0
    qs = deembed_s_params(
        result.s_params,
        {"port1": L / 2, "port2": L / 2},
        dt=result.dt,
        port_line_params=result.port_line_params,
        port_normal_dx=result.port_normal_dx,
        port_modes=result.port_modes,
    )
    phase_qs = np.degrees(np.angle(qs.S("port2", "port1")))
    assert abs(phase_qs[k]) > 12.0
    assert de.reference_impedances is not None


def test_renormalisation_round_trip(line):
    _, _, _, _, result = line
    z_qs = float(result.reference_impedance("port1")[0])
    r50 = result.renormalize(50.0)
    back = r50.renormalize(z_qs)
    assert np.allclose(back.matrix, result.s_params.matrix, atol=1e-12)
    assert np.allclose(r50.reference_impedance("port2"), 50.0)


def test_store_round_trip(line, tmp_path):
    model, control, mesh, _, result = line
    path = tmp_path / "proj"
    analysis = mio.AnalysisScatteringTD(mesh=mesh, verbose=False, project=str(path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        analysis.run(excited=["port1", "port2"])
    project = mio.open_project(path)
    assert np.allclose(project.s_params.matrix, result.s_params.matrix, atol=1e-12)
    assert np.allclose(
        project.reference_impedance("port1"), result.reference_impedance("port1"), rtol=1e-12
    )
    de_mem = result.deembed({"port1": L / 2})
    de_store = project.deembed({"port1": L / 2})
    assert np.allclose(de_store.matrix, de_mem.matrix, atol=1e-10)


def test_port_plane_refinement_reproduces_the_users_grid(line):
    model, control, mesh, analysis, _ = line
    z_user = analysis.solve_ports()["port1"].z_line_num
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ladder = ports.refine_port_modes(model, control, mesh, "port1", levels=2)
    assert ladder.levels[0].value == pytest.approx(z_user, rel=1e-12)
    assert ladder.levels[0].n_cells_port_plane == mesh.Nx * mesh.Ny
    assert ladder.levels[1].n_cells_port_plane == 4 * mesh.Nx * mesh.Ny
    # The coarse tutorial grid reads the line low; refinement moves it up.
    assert ladder.levels[1].value > z_user
