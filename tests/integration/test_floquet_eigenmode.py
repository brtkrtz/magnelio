"""
Bloch-periodic (Floquet) eigenmodes — DD-182.

Two references hold the implementation:

* the *discrete* dispersion relation of the empty periodic box on a
  uniform grid, which the FIT eigenproblem must reproduce to solver
  tolerance for any phase advance (no discretisation error to hide
  behind);
* the classic half-cell band-edge calculations of a mirror-symmetric
  cell: the phi = 0 spectrum is the union of the {PEC, PEC} and
  {PMC, PMC} half-cell spectra, phi = pi the union of the mixed pairs.
"""

import math
import warnings

import numpy as np
import pytest

from magnelio.boundaries.boundary_conditions import BoundaryConditions
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.solver._eigenmode_3d import EigenmodeSolver3D

C0 = 299_792_458.0
A, B, L = 40e-3, 20e-3, 30e-3
NX, NY, NZ = 8, 4, 6
PERIODIC_Z = {"zmin": "Periodic", "zmax": "Periodic"}


def _box_mesh():
    grid = GridLines(
        x=np.linspace(0.0, A, NX + 1),
        y=np.linspace(0.0, B, NY + 1),
        z=np.linspace(0.0, L, NZ + 1),
    )
    return Mesh.from_grid(grid, boundary_conditions=BoundaryConditions(**PERIODIC_Z))


def _discrete_dispersion(phi: float, count: int) -> np.ndarray:
    """Lowest *count* FIT eigenfrequencies of the PEC-walled box, Bloch in z.

    On a uniform grid each wavenumber enters through the discrete
    ``(2/h) sin(k h / 2)``; TE_mn needs (m, n) != (0, 0), TM_mn needs
    m, n >= 1, and the Bloch wavenumber is ``(phi + 2 pi p) / L``.
    """
    hx, hy, hz = A / NX, B / NY, L / NZ

    def kd(k, h):
        return 2.0 / h * math.sin(0.5 * k * h)

    out = []
    for m in range(3):
        for n in range(3):
            for p in (-1, 0, 1):
                kz = (phi + 2.0 * math.pi * p) / L
                k2 = kd(m * math.pi / A, hx) ** 2 + kd(n * math.pi / B, hy) ** 2 + kd(kz, hz) ** 2
                f = C0 * math.sqrt(k2) / (2.0 * math.pi)
                out += [f] * ((1 if (m, n) != (0, 0) else 0) + (1 if m >= 1 and n >= 1 else 0))
    return np.array(sorted(out)[:count])


def _solve(mesh, n, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return EigenmodeSolver3D(n_modes=n, verbose=False, **kw).solve(mesh)


class TestEmptyBoxDispersion:
    @pytest.mark.parametrize("deg", [0.0, 60.0, 90.0, 150.0, 180.0])
    def test_matches_discrete_dispersion(self, deg):
        mesh = _box_mesh()
        f, E, _ = _solve(
            mesh,
            4,
            boundary_conditions=PERIODIC_Z,
            phase_advance_deg=deg,
            sigma=(2.0 * math.pi * 7.5e9) ** 2,
        )
        ref = _discrete_dispersion(math.radians(deg), 4)
        assert f.size == 4
        np.testing.assert_allclose(f, ref, rtol=1e-8)
        assert np.iscomplexobj(E) == (deg not in (0.0, 180.0))

    def test_returned_field_obeys_the_bloch_condition(self):
        """The far plane of the returned E is the near plane times exp(-i phi)."""
        deg = 60.0
        mesh = _box_mesh()
        _, E, H = _solve(
            mesh,
            1,
            boundary_conditions=PERIODIC_Z,
            phase_advance_deg=deg,
            sigma=(2.0 * math.pi * 4e9) ** 2,
        )
        # The lowest mode is TE10: Ey and Hx, Hz carry it.
        n_Ex, n_Ey = NX * (NY + 1) * (NZ + 1), (NX + 1) * NY * (NZ + 1)
        Ey = E[n_Ex : n_Ex + n_Ey, 0].reshape(NX + 1, NY, NZ + 1)
        factor = np.exp(-1j * math.radians(deg))
        assert abs(Ey).max() > 0.0
        np.testing.assert_allclose(Ey[:, :, NZ], factor * Ey[:, :, 0], atol=1e-12 * abs(E).max())
        n_Hx, n_Hy = (NX + 1) * NY * NZ, NX * (NY + 1) * NZ
        Hz = H[n_Hx + n_Hy :, 0].reshape(NX, NY, NZ + 1)
        assert abs(Hz).max() > 0.0
        np.testing.assert_allclose(Hz[:, :, NZ], factor * Hz[:, :, 0], atol=1e-12 * abs(H).max())

    def test_phase_zero_equals_unspecified(self):
        mesh = _box_mesh()
        kw = dict(boundary_conditions=PERIODIC_Z, sigma=(2.0 * math.pi * 7.5e9) ** 2)
        f_none, _, _ = _solve(mesh, 3, **kw)
        f_zero, _, _ = _solve(mesh, 3, phase_advance_deg=0.0, **kw)
        np.testing.assert_allclose(f_none, f_zero, rtol=1e-10)


class TestDeclarationGuards:
    def test_periodic_faces_must_pair(self):
        with pytest.raises(ValueError, match="come in pairs"):
            BoundaryConditions(zmin="Periodic")

    def test_cpml_is_rejected_by_the_eigensolver(self):
        mesh = _box_mesh()
        with pytest.raises(ValueError, match="CPML"):
            _solve(mesh, 2, boundary_conditions={"xmin": "CPML", **PERIODIC_Z})

    def test_phase_without_periodic_axis_is_rejected(self):
        grid = GridLines(
            x=np.linspace(0.0, A, 5), y=np.linspace(0.0, B, 3), z=np.linspace(0.0, L, 4)
        )
        mesh = Mesh.from_grid(grid)
        with pytest.raises(ValueError, match="no face pair is declared 'Periodic'"):
            _solve(mesh, 2, phase_advance_deg=30.0)

    def test_dict_must_name_periodic_axes(self):
        mesh = _box_mesh()
        with pytest.raises(ValueError, match="not periodic"):
            _solve(mesh, 2, boundary_conditions=PERIODIC_Z, phase_advance_deg={"x": 30.0})

    def test_complex_phase_needs_superlu(self):
        mesh = _box_mesh()
        with pytest.raises(NotImplementedError, match="complex Hermitian"):
            _solve(
                mesh,
                2,
                boundary_conditions=PERIODIC_Z,
                phase_advance_deg=45.0,
                solver="arpack-cholmod",
            )
        with pytest.raises(NotImplementedError, match="periodic"):
            _solve(
                mesh,
                2,
                boundary_conditions=PERIODIC_Z,
                phase_advance_deg=180.0,
                solver="lobpcg",
            )


@pytest.fixture(scope="module")
def iris_cells():
    """A pillbox with an iris, full period and its four half-cell closures."""
    pytest.importorskip("OCC")
    import magnelio as mio
    from magnelio import geo

    pec, air = mio.Material.pec(), mio.Material.air()
    R, Lc, Ri, t = 50e-3, 40e-3, 20e-3, 10e-3
    period = Lc + t

    def build(bcs):
        model = mio.GeometryModel(background=pec, boundary_conditions=bcs)
        model.add(geo.Cylinder(origin=(0, 0, t / 2), radius=R, height=Lc, axis="z", material=air))
        model.add(geo.Cylinder(origin=(0, 0, 0), radius=Ri, height=period, axis="z", material=air))
        return mio.Mesh.from_geometry(model, mio.MeshControl(max_cell_size=6e-3), f_max=4e9)

    full = build({"zmin": "Periodic", "zmax": "Periodic"})
    halves = {
        (lo, hi): build({"zmin": lo, "zmax": (f"Symmetry{hi}", period / 2)})
        for lo in ("PEC", "PMC")
        for hi in ("PEC", "PMC")
    }
    return full, halves


def _analysis(mesh, n, **kw):
    import magnelio as mio

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return mio.AnalysisEigenmode(mesh=mesh, n_modes=n, verbose=False, **kw).run().frequencies


class TestBandEdgesAgainstHalfCells:
    """phi = 0 and phi = pi are the classic wall-type half-cell problems.

    The mixed-wall half cells deviate by ~2e-4 because a PMC mid-plane
    is never an exact grid restriction of the full period (DD-164
    pull-in); the electric-wall pairs agree to solver tolerance.
    """

    def _union(self, halves, pairs, n):
        return np.sort(np.concatenate([_analysis(halves[p], n)[:n] for p in pairs]))[:n]

    def test_phase_zero_is_the_union_of_like_walls(self, iris_cells):
        full, halves = iris_cells
        f = _analysis(full, 5, phase_advance_deg=0.0)
        ref = self._union(halves, [("PEC", "PEC"), ("PMC", "PMC")], 5)
        np.testing.assert_allclose(f, ref, rtol=1e-3)

    def test_phase_pi_is_the_union_of_mixed_walls(self, iris_cells):
        full, halves = iris_cells
        f = _analysis(full, 5, phase_advance_deg=180.0)
        ref = self._union(halves, [("PEC", "PMC"), ("PMC", "PEC")], 5)
        np.testing.assert_allclose(f, ref, rtol=1e-3)

    def test_the_passband_rises_monotonically(self, iris_cells):
        """Iris-coupled TM010 cells: electric coupling, f grows with phi."""
        full, _ = iris_cells
        f = [_analysis(full, 1, phase_advance_deg=deg)[0] for deg in (0.0, 90.0, 180.0)]
        assert f[0] < f[1] < f[2]
        assert np.iscomplexobj(_analysis(full, 1, phase_advance_deg=90.0)) is False  # frequencies
