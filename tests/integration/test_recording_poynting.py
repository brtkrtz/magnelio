"""The Poynting vector against the flux identity it must agree with (DD-270).

A matched TEM line carries its power along +z.  Two ways to state it:
:meth:`FieldSpectrum.flux`, the exact FIT pairing ``Re Σ e·h*`` on the
samples, and the cell-centred vector field summed over the
cross-section.  They must agree — and where they seem not to, the
reason is the patch of the boundary cells, not the physics: at a PMC
face the wall lies half an outer cell beyond the outermost grid line,
so ``dx·dy`` is not the physical area there.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio import AnalysisScatteringTD, Material, MeshControl
from magnelio.geo import Brick, GeometryModel
from magnelio.mesh.mesher import Mesh
from magnelio.monitors import MonitorFieldFrequency
from magnelio.ports import PortWaveguide

A, B, LZ = 10.0e-3, 5.0e-3, 20.0e-3
F_MAX = 12.0e9
FREQS = np.array([6e9, 9e9])


@pytest.fixture(scope="module")
def tem_line():
    """A parallel-plate TEM two-port with a one-cell field monitor at z = 0."""
    model = GeometryModel(
        boundary_conditions={
            "xmin": "PMC",
            "xmax": "PMC",
            "ymin": "PEC",
            "ymax": "PEC",
            "zmin": "PEC",
            "zmax": "PEC",
        }
    )
    model.add(
        Brick(
            origin=(-A / 2, -B / 2, -LZ / 2),
            size=(A, B, LZ),
            material=Material.from_isotropic(name="air", epsilon=1.0),
        )
    )
    mesh = Mesh.from_geometry(model, MeshControl(min_nodes_per_wavelength=10), f_max=F_MAX)
    k = int(np.argmin(np.abs(np.asarray(mesh.grid.z))))
    # The flux pairing at node k takes the magnetic samples of cell k
    # (DD-085), so the layer that can state it is the cell above the
    # plane: the monitor is declared at that cell's centre.
    zc = 0.5 * (mesh.grid.z[k] + mesh.grid.z[k + 1])
    monitor = MonitorFieldFrequency(
        name="layer",
        corners=((None, None, zc), (None, None, zc)),
        freqs=FREQS,
        fields=["E", "H"],
    )
    analysis = AnalysisScatteringTD(
        mesh=mesh,
        ports=[
            PortWaveguide(name="port1", plane="zmin", n_modes=1),
            PortWaveguide(name="port2", plane="zmax", n_modes=1),
        ],
        f_max=F_MAX,
        verbose=False,
        backend="numpy",
        geometry=model,
    )
    analysis.monitors = (monitor,)
    result = analysis.run()
    return mesh, monitor.spectrum, result, float(mesh.grid.z[k])


def _cross_section_areas(grid):
    """The physical patch of every cell of a z cross-section.

    The PMC walls at xmin/xmax sit half an outer cell beyond the
    outermost grid line, so the boundary cells own that half as well —
    the booking :class:`~magnelio.monitors.MonitorFluxTime` uses.
    """
    dx = np.asarray(grid.dx, dtype=float).copy()
    dy = np.asarray(grid.dy, dtype=float)
    dx[0] += 0.5 * dx[0]
    dx[-1] += 0.5 * dx[-1]
    assert np.isclose(dx.sum(), A)
    return dx[:, None] * dy[None, :]


def test_the_integral_is_the_flux_identity(tem_line):
    mesh, spectrum, _result, z_node = tem_line
    identity = spectrum.flux("z", z_node)
    s = spectrum.poynting()
    area = _cross_section_areas(mesh.grid)
    integral = np.array([float(np.sum(s[i, :, :, 0, 2] * area)) for i in range(spectrum.n_frames)])
    # A TEM mode is uniform across the section, so the cell-centre
    # average is exact there and only the booking is under test.
    np.testing.assert_allclose(integral, identity, rtol=1e-6)


def test_a_matched_line_carries_its_watt_forward(tem_line):
    _mesh, spectrum, result, _z = tem_line
    s = spectrum.poynting()
    assert np.all(s[..., 2] > 0.0), "the power flows from port 1 towards port 2"
    transmitted = np.abs(result.S("port2", "port1")) ** 2
    assert np.all(transmitted > 0.95), "the line is matched, so nearly all of it arrives"


def test_the_naive_area_is_short_by_the_pmc_half_cells(tem_line):
    """The trap the doc warns about: dx·dy is not the patch at a magnetic wall."""
    mesh, spectrum, _result, z_node = tem_line
    s = spectrum.poynting()
    dx = np.asarray(mesh.grid.dx, dtype=float)
    dy = np.asarray(mesh.grid.dy, dtype=float)
    naive = np.array(
        [float(np.sum(s[i, :, :, 0, 2] * (dx[:, None] * dy[None, :]))) for i in range(2)]
    )
    identity = spectrum.flux("z", z_node)
    # Short by exactly the two half cells the grid lines do not span,
    # the same factor at every frequency — a booking, not a discretisation error.
    np.testing.assert_allclose(naive / identity, dx.sum() / A, rtol=1e-6)
