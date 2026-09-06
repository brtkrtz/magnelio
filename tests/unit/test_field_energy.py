"""Energy and flux from a recorded field (DD-260).

A monitor's frames are grid quantities; with the region's material
operators they state their stored energy (the leapfrog-conserved
pairing of DD-225) and the Poynting flux through a cross-section (the
identity ``MonitorFluxTime`` records).  The gates here are algebraic:
against the quadratic forms on the full diagonals, bit for bit against
the flux monitor, additivity over a cut, invariance under mirroring.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio._fields.field_arrays import FieldArrays
from magnelio._operators.curl import build_curl_matrix
from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.fields import FieldRecording, FieldSpectrum, FieldState
from magnelio.fields._interp import _region_dual, _region_slices
from magnelio.fields._operators import RegionOperators, region_operators
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.monitors import MonitorFieldFrequency, MonitorFieldTime, MonitorFluxTime
from magnelio.monitors.base import region_grid, resolve_mirrors, resolve_region
from magnelio.monitors.field_time import _take_raw

E_NAMES = ("Ex", "Ey", "Ez")
H_NAMES = ("Hx", "Hy", "Hz")
SIX = E_NAMES + H_NAMES
DT = 1.7e-12


def _grid() -> GridLines:
    """Non-uniform, so the dual widths differ from the cells."""
    return GridLines(
        x=np.array([0.0, 1.0, 2.5, 3.5, 5.0, 7.0]) * 1e-3,
        y=np.array([0.0, 1.5, 2.5, 4.0, 5.0]) * 1e-3,
        z=np.array([0.0, 0.8, 2.0, 2.8, 4.0, 5.5, 6.5]) * 1e-3,
    )


def _mesh(bc=None) -> Mesh:
    mesh = Mesh.from_grid(_grid())
    return mesh.with_boundary_conditions(bc) if bc is not None else mesh


def _fields(grid: GridLines, seed: int = 3, complex_part: bool = False) -> FieldArrays:
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    rng = np.random.default_rng(seed)
    shapes = {
        "Ex": (Nx, Ny + 1, Nz + 1),
        "Ey": (Nx + 1, Ny, Nz + 1),
        "Ez": (Nx + 1, Ny + 1, Nz),
        "Hx": (Nx + 1, Ny, Nz),
        "Hy": (Nx, Ny + 1, Nz),
        "Hz": (Nx, Ny, Nz + 1),
    }
    arrays = {}
    for c, shp in shapes.items():
        a = rng.standard_normal(shp)
        if complex_part:
            a = a + 1j * rng.standard_normal(shp)
        arrays[c] = a
    return FieldArrays(**arrays)


def _diagonals(mesh: Mesh):
    return build_M_eps(mesh), build_M_mu(mesh)


def _whole(mesh: Mesh, raw: FieldArrays, h_lead: float = 0.0) -> FieldState:
    region = resolve_region(None, mesh.grid)
    ops = region_operators(mesh, region, *_diagonals(mesh))
    return FieldState._from_raw(mesh.grid, raw, ops=ops, h_lead=h_lead)


def _part(mesh: Mesh, raw: FieldArrays, corners, h_lead: float = 0.0) -> FieldState:
    """The region's own samples, dual widths and operators — what a monitor keeps."""
    region = resolve_region(corners, mesh.grid)
    r = region
    cut = _take_raw(raw, list(SIX), {c: _region_slices(r.ix, r.iy, r.iz, c) for c in SIX})
    ops = region_operators(mesh, region, *_diagonals(mesh))
    return FieldState._from_raw(
        region_grid(mesh.grid, region),
        FieldArrays(**{c: np.asarray(cut[c]) for c in SIX}),
        dual=_region_dual(mesh.grid, r.ix, r.iy, r.iz),
        ops=ops,
        h_lead=h_lead,
    )


def _quadratic(mesh: Mesh, raw: FieldArrays) -> tuple[float, float]:
    m_eps, m_mu = _diagonals(mesh)
    e, h = np.asarray(raw.e_flat), np.asarray(raw.h_flat)
    return 0.5 * float(m_eps @ (e * e)), 0.5 * float(m_mu @ (h * h))


class TestRegionOperators:
    def test_ends_and_symmetry(self):
        mesh = _mesh({"xmin": "PMC", "zmax": "ForceSymmetryPMC"})
        ops = region_operators(mesh, resolve_region(None, mesh.grid), *_diagonals(mesh))
        assert ops.ends == (("pmc", "wall"), ("wall", "wall"), ("wall", "pmc"))
        assert ops.symmetry == ("zmax",)
        assert ops.energy_factor() == 2.0
        assert ops.flux_factor(2) == 1.0 and ops.flux_factor(0) == 2.0
        g = mesh.grid
        sub = resolve_region(((1e-3, None, None), (3.5e-3, None, None)), g)
        ops = region_operators(mesh, sub, *_diagonals(mesh))
        assert ops.ends[0] == ("cut", "cut")
        assert ops.m_eps["Ex"].shape == (2, g.Ny + 1, g.Nz + 1)
        assert ops.m_mu["Hx"].shape == (3, g.Ny, g.Nz)

    def test_edge_weights(self):
        mesh = _mesh({"xmin": "PMC"})
        g = mesh.grid
        ops = region_operators(mesh, resolve_region(None, g), *_diagonals(mesh))
        wx, wy, wz = ops.edge_weights(g, None, purpose="energy")
        assert np.all(wx == 1.0) and np.all(wy == 1.0) and np.all(wz == 1.0)
        wx, wy, wz = ops.edge_weights(g, None, purpose="flux")
        assert wx[0] == 1.0 and wx[-1] == 0.5 and wy[0] == 0.5 and np.all(wx[1:-1] == 1.0)
        sub = resolve_region(((1e-3, None, None), (3.5e-3, None, None)), g)
        ops = region_operators(mesh, sub, *_diagonals(mesh))
        sg = region_grid(g, sub)
        dual = _region_dual(g, sub.ix, sub.iy, sub.iz)
        for purpose in ("energy", "flux"):
            wx = ops.edge_weights(sg, dual, purpose=purpose)[0]
            np.testing.assert_allclose(wx[0], 0.5 * sg.dx[0] / dual[0][0])
            np.testing.assert_allclose(wx[-1], 0.5 * sg.dx[-1] / dual[0][-1])
            assert np.all(wx[1:-1] == 1.0)
        with pytest.raises(ValueError, match="purpose"):
            ops.edge_weights(sg, dual, purpose="power")

    def test_store_encoding_round_trip(self):
        mesh = _mesh({"ymax": "PMC", "xmin": "ForceSymmetryPEC"})
        sub = resolve_region(((None, 1.5e-3, None), (None, 4.5e-3, None)), mesh.grid)
        ops = region_operators(mesh, sub, *_diagonals(mesh))
        back = RegionOperators.from_arrays(
            ops.m_eps, ops.m_mu, ops.ends_codes(), ops.symmetry_flags()
        )
        assert back.ends == ops.ends and back.symmetry == ops.symmetry
        for c in E_NAMES:
            np.testing.assert_array_equal(back.m_eps[c], ops.m_eps[c])
        for c in H_NAMES:
            np.testing.assert_array_equal(back.m_mu[c], ops.m_mu[c])

    def test_validation(self):
        with pytest.raises(KeyError, match="m_eps"):
            RegionOperators(m_eps={}, m_mu={}, ends=(), symmetry=())
        z = {c: np.zeros(1) for c in SIX}
        with pytest.raises(ValueError, match="ends"):
            RegionOperators(m_eps=z, m_mu=z, ends=(("cut", "cut"),), symmetry=())
        with pytest.raises(ValueError, match="symmetry"):
            RegionOperators(m_eps=z, m_mu=z, ends=(("cut", "cut"),) * 3, symmetry=("top",))


class TestEnergy:
    def test_matches_the_quadratic_forms(self):
        mesh = _mesh()
        raw = _fields(mesh.grid)
        w_e, w_h = _quadratic(mesh, raw)
        np.testing.assert_allclose(_whole(mesh, raw).energy(), w_e + w_h, rtol=1e-13)

    def test_march_frame_pairs_the_half_steps(self):
        """h_lead = dt/2: the magnetic term is h(t−dt/2)·Mμ·h(t+dt/2) (DD-225)."""
        mesh = _mesh()
        raw = _fields(mesh.grid)
        m_eps, m_mu = _diagonals(mesh)
        e, h = np.asarray(raw.e_flat), np.asarray(raw.h_flat)
        curl = build_curl_matrix(mesh.grid) @ e
        h_prev = h + (DT / m_mu) * curl
        expected = 0.5 * float(m_eps @ (e * e)) + 0.5 * float(m_mu @ (h_prev * h))
        np.testing.assert_allclose(
            _whole(mesh, raw, h_lead=0.5 * DT).energy(), expected, rtol=1e-12
        )

    @pytest.mark.parametrize("h_lead", [0.0, 0.5 * DT])
    def test_two_regions_add_up_to_their_union(self, h_lead):
        mesh = _mesh({"zmin": "PMC"})
        raw = _fields(mesh.grid)
        whole = _whole(mesh, raw, h_lead).energy()
        left = _part(mesh, raw, ((None, None, None), (2.5e-3, None, None)), h_lead).energy()
        right = _part(mesh, raw, ((2.5e-3, None, None), (None, None, None)), h_lead).energy()
        np.testing.assert_allclose(left + right, whole, rtol=1e-12)
        # and a cut along a second axis
        corner = _part(mesh, raw, ((2.5e-3, 2.5e-3, None), (None, None, None)), h_lead).energy()
        rest = _part(mesh, raw, ((2.5e-3, None, None), (None, 2.5e-3, None)), h_lead).energy()
        np.testing.assert_allclose(corner + rest, right, rtol=1e-12)

    def test_symmetry_plane_doubles_the_joules(self):
        raw = _fields(_grid())
        plain = _whole(_mesh({"xmin": "PMC"}), raw).energy()
        sym = _whole(_mesh({"xmin": "ForceSymmetryPMC"}), raw).energy()
        assert sym == 2.0 * plain

    def test_complex_frame_is_the_time_average(self):
        mesh = _mesh()
        raw = _fields(mesh.grid, complex_part=True)
        m_eps, m_mu = _diagonals(mesh)
        e, h = np.asarray(raw.e_flat), np.asarray(raw.h_flat)
        # RMS phasors: no further factor beyond the ½ of the quadratic form.
        expected = 0.5 * float(m_eps @ np.abs(e) ** 2) + 0.5 * float(m_mu @ np.abs(h) ** 2)
        np.testing.assert_allclose(
            _whole(mesh, raw, h_lead=0.5 * DT).energy(), expected, rtol=1e-13
        )

    def test_without_operators_it_says_so(self):
        with pytest.raises(RuntimeError, match="operators"):
            FieldState.zeros(_grid()).energy()
        with pytest.raises(RuntimeError, match="operators"):
            FieldState.zeros(_grid()).flux("z", 0.0)


class TestFlux:
    @pytest.mark.parametrize(
        "bc",
        [None, {"xmin": "PMC", "ymax": "PMC"}, {"xmin": "ForceSymmetryPMC", "zmax": "PMC"}],
    )
    @pytest.mark.parametrize("normal", ["x", "y", "z"])
    def test_matches_the_flux_monitor_bit_for_bit(self, bc, normal):
        mesh = _mesh(bc)
        raw = _fields(mesh.grid)
        state = _whole(mesh, raw)
        nodes = getattr(mesh.grid, normal)
        for position in (nodes[0], nodes[2], nodes[-1] + 1e-4):
            mon = MonitorFluxTime(normal=normal, position=position, name="p")
            mon.attach(mesh)
            mon.record(raw, 0, 0.0, DT)
            assert state.flux(normal, position) == float(mon.power[0])

    def test_two_regions_add_up_to_the_cross_section(self):
        mesh = _mesh({"xmin": "PMC"})
        raw = _fields(mesh.grid)
        z0 = mesh.grid.z[3]
        whole = _whole(mesh, raw).flux("z", z0)
        left = _part(mesh, raw, ((None, None, None), (2.5e-3, None, None))).flux("z", z0)
        right = _part(mesh, raw, ((2.5e-3, None, None), (None, None, None))).flux("z", z0)
        np.testing.assert_allclose(left + right, whole, rtol=1e-12)

    def test_plane_region_holds_its_own_flux(self):
        """A one-cell-thick region (a plane monitor) carries the plane's pairing."""
        mesh = _mesh()
        raw = _fields(mesh.grid)
        z0 = mesh.grid.z[2]
        whole = _whole(mesh, raw).flux("z", z0)
        layer = _part(mesh, raw, ((None, None, z0), (None, None, z0)))
        assert layer.grid.Nz == 1
        assert layer.flux("z", z0) == whole

    def test_complex_frame_gives_the_real_power(self):
        mesh = _mesh()
        raw = _fields(mesh.grid, complex_part=True)
        state = _whole(mesh, raw)
        k = 2
        ex, hy = np.asarray(raw.Ex)[:, :, k], np.asarray(raw.Hy)[:, :, k]
        ey, hx = np.asarray(raw.Ey)[:, :, k], np.asarray(raw.Hx)[:, :, k]
        wy = np.ones(mesh.grid.Ny + 1)
        wy[0] = wy[-1] = 0.5
        wx = np.ones(mesh.grid.Nx + 1)
        wx[0] = wx[-1] = 0.5
        expected = float(
            np.real(np.sum(ex * np.conj(hy) * wy[None, :]) - np.sum(ey * np.conj(hx) * wx[:, None]))
        )
        np.testing.assert_allclose(state.flux("z", mesh.grid.z[k]), expected, rtol=1e-13)

    def test_normal_checked(self):
        mesh = _mesh()
        with pytest.raises(ValueError, match="normal"):
            _whole(mesh, _fields(mesh.grid)).flux("n", 0.0)


class TestMirrored:
    @pytest.mark.parametrize("kind", ["ForceSymmetryPEC", "ForceSymmetryPMC"])
    @pytest.mark.parametrize("face", ["xmin", "ymax"])
    def test_energy_and_flux_survive_the_mirror(self, kind, face):
        mesh = _mesh({face: kind})
        raw = _fields(mesh.grid)
        if kind.endswith("PEC"):
            # A half-model state has no tangential E on an electric
            # plane (the solver's PEC mask); a random field must be told.
            axis = "xyz".index(face[0])
            end = 0 if face.endswith("min") else -1
            for name in E_NAMES:
                if "xyz".index(name[1]) == axis:
                    continue
                a = np.asarray(getattr(raw, name))
                idx = [slice(None)] * 3
                idx[axis] = end
                a[tuple(idx)] = 0.0
                setattr(raw, name, a)
        field = _whole(mesh, raw, h_lead=0.5 * DT)
        (spec,) = resolve_mirrors(resolve_region(None, mesh.grid), mesh)
        full = field.mirrored(spec)
        assert full._ops is not None and full._ops.symmetry == ()
        np.testing.assert_allclose(full.energy(), field.energy(), rtol=1e-12)
        z0 = mesh.grid.z[3]
        np.testing.assert_allclose(full.flux("z", z0), field.flux("z", z0), rtol=1e-12)
        axis = "xyz".index(face[0])
        assert (
            full.grid.Nx + full.grid.Ny + full.grid.Nz > mesh.grid.Nx + mesh.grid.Ny + mesh.grid.Nz
        )
        assert full._ops.ends[axis][0] == full._ops.ends[axis][1]

    def test_mirroring_a_field_without_operators_stays_plain(self):
        mesh = _mesh({"xmin": "ForceSymmetryPEC"})
        field = FieldState._from_raw(mesh.grid, _fields(mesh.grid))
        (spec,) = resolve_mirrors(resolve_region(None, mesh.grid), mesh)
        assert field.mirrored(spec)._ops is None


class TestSeries:
    def test_recording_and_spectrum_go_frame_by_frame(self):
        mesh = _mesh()
        g = mesh.grid
        ops = region_operators(mesh, resolve_region(None, g), *_diagonals(mesh))
        raws = [_fields(g, seed=s) for s in (1, 2, 3)]
        stack = {c: np.stack([np.asarray(getattr(r, c)) for r in raws]) for c in SIX}
        rec = FieldRecording._from_raw(g, DT * np.arange(1, 4), stack, ops=ops, dt=DT)
        w = rec.energy()
        assert w.shape == (3,)
        for i, r in enumerate(raws):
            assert w[i] == FieldState._from_raw(g, r, ops=ops, h_lead=0.5 * DT).energy()
        p = rec.flux("y", g.y[2])
        assert p[1] == FieldState._from_raw(g, raws[1], ops=ops).flux("y", g.y[2])
        craws = [_fields(g, seed=s, complex_part=True) for s in (4, 5)]
        cstack = {c: np.stack([np.asarray(getattr(r, c)) for r in craws]) for c in SIX}
        spec = FieldSpectrum._from_raw(g, [1e9, 2e9], cstack, ops=ops)
        assert spec.energy().shape == (2,)
        assert spec.flux("z", 0.0)[0] == FieldState._from_raw(g, craws[0], ops=ops).flux("z", 0.0)

    def test_series_without_operators_says_so(self):
        g = _grid()
        stack = {c: np.stack([np.asarray(getattr(_fields(g), c))]) for c in SIX}
        with pytest.raises(RuntimeError, match="operators"):
            FieldRecording._from_raw(g, [1e-12], stack, dt=DT).energy()


class TestMonitorsCarryTheOperators:
    def test_time_monitor(self):
        mesh = _mesh({"xmin": "PMC"})
        raw = _fields(mesh.grid)
        mon = MonitorFieldTime(
            name="v",
            corners=((1e-3, None, None), (5.5e-3, None, None)),
            fields=["E", "H"],
            times=[0.0],
        )
        with pytest.raises(RuntimeError, match="attach"):
            mon.attach_operators(mesh, *_diagonals(mesh))
        mon.attach(mesh)
        assert mon._ops is None
        mon.attach_operators(mesh, *_diagonals(mesh))
        mon.record(raw, 0, 0.0, DT)
        rec = mon.recording
        assert rec._ops.ends[0] == ("cut", "cut")
        expected = _part(mesh, raw, ((1e-3, None, None), (5.5e-3, None, None)), 0.5 * DT).energy()
        np.testing.assert_allclose(rec.energy(), [expected], rtol=1e-13)

    def test_frequency_monitor_dump_round_trip(self):
        mesh = _mesh({"ymin": "ForceSymmetryPMC"})
        raw = _fields(mesh.grid)
        mon = MonitorFieldFrequency(
            name="f",
            corners=((None, None, 2e-3), (None, None, 2e-3)),
            freqs=[1e9],
            fields=["E", "H"],
        )
        mon.attach(mesh)
        mon.attach_operators(mesh, *_diagonals(mesh))
        mon.record(raw, 0, 0.0, DT)
        dump = mon.result_dump()
        assert dump["operators"]["ends"].shape == (3, 2)
        twin = MonitorFieldFrequency(
            name="f",
            corners=((None, None, 2e-3), (None, None, 2e-3)),
            freqs=[1e9],
            fields=["E", "H"],
        )
        twin.attach(mesh)
        assert twin._ops is None
        twin.load_result_dump(dump)
        assert twin._ops.symmetry == ("ymin",) and twin._ops.ends == mon._ops.ends
        np.testing.assert_array_equal(twin.spectrum_raw.energy(), mon.spectrum_raw.energy())
        assert twin.spectrum_raw.flux("z", 2e-3)[0] == mon.spectrum_raw.flux("z", 2e-3)[0]
