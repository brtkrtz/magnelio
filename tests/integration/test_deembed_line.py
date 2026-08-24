"""De-embedding certificate on a uniform TEM line.

A matched parallel-plate line (PMC side walls) between two waveguide
ports carries nothing but grid propagation, so shifting port p1's
reference plane by the full line length must cancel the transmission
phase down to the run's own accuracy floor — the discrete-dispersion
claim of DD-187, measured in
``investigations/port-deembedding/MEASUREMENTS.md`` (internal record)
and reproduced by ``validation/deembed_uniform_line.py``.
"""

import numpy as np
import pytest

import magnelio as mio
from magnelio import AnalysisScatteringTD, Material, Mesh, MeshControl
from magnelio.geo import Brick, GeometryModel
from magnelio.ports import PortWaveguide

pytest.importorskip("OCC.Core.BRepPrimAPI")

F_MAX = 20e9
L = 20e-3


@pytest.fixture(scope="module")
def result():
    a, b = 10e-3, 2e-3
    model = GeometryModel(
        boundary_conditions=mio.BoundaryConditions(xmin="PMC", xmax="PMC"),
    )
    model.add(
        Brick(
            origin=(-a / 2, -b / 2, -L / 2),
            size=(a, b, L),
            material=Material.from_isotropic(name="air", epsilon=1.0),
        )
    )
    model.add_port(PortWaveguide(name="p1", plane="zmin"))
    model.add_port(PortWaveguide(name="p2", plane="zmax"))
    mesh = Mesh.from_geometry(
        model,
        MeshControl(min_nodes_per_wavelength=8),
        f_max=F_MAX,
    )
    analysis = AnalysisScatteringTD(
        mesh=mesh,
        f_max=F_MAX,
        verbose=False,
        backend="numpy",
    )
    return analysis.run(excited=["p1"], energy_stop_db=60.0)


def _band(result):
    return np.asarray(result.f_axis) <= F_MAX


def test_full_length_shift_removes_line_phase(result):
    de = result.deembed({"p1": L})
    residual = np.abs(de.phase("p2", "p1", unwrap=False))[_band(result)]
    # The raw line phase spans thousands of degrees; the discrete
    # shift must cancel it to far below any physical port floor.
    assert residual.max() < 1e-3


def test_continuum_would_leave_grid_dispersion(result):
    # Reference for the discrete claim: cancelling with the continuum
    # exp(-j w L / c) leaves the grid dispersion — degrees of phase on
    # this lambda/8 mesh, orders of magnitude above the discrete
    # residual.  Guards against the discrete path silently degrading
    # into the continuum one.
    f = np.asarray(result.f_axis)
    w = 2.0 * np.pi * f
    s21 = result.S("p2", "p1")
    residual = np.abs(np.angle(s21 * np.exp(1j * w * L / 299792458.0)))[_band(result)]
    assert np.degrees(residual.max()) > 1.0


def test_magnitudes_invariant_on_propagating_line(result):
    de = result.deembed({"p1": L})
    band = _band(result)
    np.testing.assert_allclose(
        np.abs(de.S("p2", "p1"))[band],
        np.abs(result.S("p2", "p1"))[band],
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        np.abs(de.S("p1", "p1"))[band],
        np.abs(result.S("p1", "p1"))[band],
        rtol=1e-9,
    )


def test_split_shift_matches_single_on_transmission(result):
    single = result.deembed({"p1": L})
    split = result.deembed({"p1": L / 2, "p2": L / 2})
    band = _band(result)
    np.testing.assert_allclose(
        split.S("p2", "p1")[band],
        single.S("p2", "p1")[band],
        rtol=1e-9,
    )


def test_deembedded_result_answers_accessors(result, tmp_path):
    de = result.deembed({"p1": L})
    assert de.phase("p2", "p1").shape == np.asarray(de.f_axis).shape
    fig, _ax = de.plot_s(("p2", "p1"))
    import matplotlib.pyplot as plt

    plt.close(fig)
    de.to_touchstone(tmp_path / "de.s1p", channels=["p1"])
    assert (tmp_path / "de.s1p").exists()
