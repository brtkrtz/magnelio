"""Unit tests for the Galerkin band-subspace DTBC (WP-R4b, DD-057).

Covers the boundary-paired period blocks (the e_t-only exterior
coupling that makes the projected boundary period own only the
port-plane trace), the homogeneous p=1 cross-check against the exact
scalar DTBC kernel, the palindromic-symmetry certificate of the
Galerkin exterior, kernel auto-extension invariance, the dense
a-priori reflection certificate, and the factory/operator contracts
(band excitation synthesis and its compactness gate).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from magnelio._operators.curl import build_curl_matrix
from magnelio._operators.material_matrices import (
    build_M_eps,
    build_M_mu,
    flatten_port_plane_mass,
    flatten_port_plane_pec_mask,
)
from magnelio.geo import Brick, GeometryModel
from magnelio.materials.material import Material
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh, MeshControl
from magnelio.ports._modal import (
    BoxFace,
    PortSpecMultiConductor,
    band_dtbc,
    build_band_dtbc_port,
)
from magnelio.ports._modal.band_dtbc import (
    BandDTBCBoundary,
    band_apriori_reflection,
    band_subspace,
    galerkin_exterior,
    matrix_dtbc_kernel,
    track_band_families,
)
from magnelio.ports._modal.dtbc import dtbc_kernel
from magnelio.ports._modal.port_plane import PortPlane
from magnelio.ports._modal.zeta_pencil import (
    build_period_blocks,
    find_propagating_modes,
)
from magnelio.solver.stability import courant_dt

# DD-103: the closure these fixtures always assumed.  A face
# with no BC used to evolve under the free curl operator —
# which IS the natural magnetic wall, hence "PMC".
_BC_PEC_Y = {
    "ymin": "PEC",
    "ymax": "PEC",
    "xmin": "PMC",
    "xmax": "PMC",
    "zmin": "PMC",
    "zmax": "PMC",
}

C0 = 299_792_458.0


def _segments(*breaks_and_counts):
    out = []
    for lo, hi, n in breaks_and_counts:
        seg = np.linspace(lo, hi, n + 1)
        out.extend(seg if not out else seg[1:])
    return [float(v) for v in out]


def _layered_mesh(nz=12):
    w, hy, h_if, dz = 10.0e-3, 8.0e-3, 4.0e-3, 1.0e-3
    length = nz * dz
    diel = Material(name="diel", epsilon=(4.0,) * 3)
    model = GeometryModel()
    model.add(Brick(origin=(0, 0, 0), size=(w, h_if, length), material=diel))
    model.add(Brick(origin=(0, h_if, 0), size=(w, hy - h_if, length), material=Material.air()))
    control = MeshControl(
        min_nodes_per_wavelength=4,
        min_cells_per_feature=0,
        max_cell_size=5.1e-3,
        forced_planes={
            "x": _segments((0.0, w, 2)),
            "y": _segments((0.0, h_if, 4), (h_if, hy, 4)),
            "z": _segments((0.0, length, nz)),
        },
    )
    mesh = Mesh.from_geometry(model, control, f_max=8.0e9)
    return mesh.with_boundary_conditions(
        {
            "ymin": "PEC",
            "ymax": "PEC",
            "xmin": "PMC",
            "xmax": "PMC",
            "zmin": "PMC",
            "zmax": "PMC",
        }
    )


def _chains_for(mesh, dt, face=BoxFace.Z_MIN):
    m_eps = flatten_port_plane_mass(build_M_eps(mesh), mesh, face)
    object.__setattr__(
        mesh,
        "pec_mask_edges",
        flatten_port_plane_pec_mask(mesh.pec_mask_edges, mesh, face),
    )
    m_mu = build_M_mu(mesh)
    plane = PortPlane.from_mesh(face, mesh)
    c_3d = build_curl_matrix(mesh.grid)
    chain_b = build_period_blocks(plane, mesh, m_eps, m_mu, c_3d, dt, pairing="boundary")
    return chain_b, plane


@pytest.fixture(scope="module")
def layered_boundary_chain():
    mesh = _layered_mesh()
    dt = courant_dt(mesh.grid, "normal")
    chain, plane = _chains_for(mesh, dt)
    return chain, plane, dt


@pytest.fixture(scope="module")
def layered_families(layered_boundary_chain):
    chain, plane, dt = layered_boundary_chain
    dz = float(plane.normal_dx)
    w_dt0 = 2.0 * math.pi * 1.0e9 * dt
    theta0 = 2.0 * math.pi * 1.0e9 * math.sqrt(2.5) / C0 * dz
    _, ps = find_propagating_modes(chain, w_dt0, 1.3 * theta0)
    track_t = ps[: chain.n_t, 0].real
    f_grid = np.linspace(1.0e9, 7.8e9, 7)
    return track_band_families(chain, dt, f_grid, track_t, 1.6, dz)


class TestBoundaryPairing:
    def test_exterior_coupling_is_et_only(self, layered_boundary_chain):
        """D_m1 columns live on e_t alone in the boundary pairing.

        This is the structural requirement of the projected boundary
        period: the interior touches it only through the port-plane
        tangential trace, and the period's e_z half-plane is virtual.
        """
        chain, _, _ = layered_boundary_chain
        assert chain.pairing == "boundary"
        ez_cols = np.abs(chain.D_m1)[:, chain.n_t :]
        assert ez_cols.max() == 0.0
        # The inward-facing block has full columns.
        assert np.abs(chain.D_p1)[:, chain.n_t :].max() > 0.0

    def test_period_one_is_interior(self, layered_boundary_chain):
        chain, _, _ = layered_boundary_chain
        idx = chain.period(1)
        assert idx.min() >= 0

    def test_palindromic_w_symmetry(self, layered_boundary_chain):
        chain, _, _ = layered_boundary_chain
        w = chain.w_period
        lhs = (chain.D_p1.T.multiply(w[None, :])).toarray()
        rhs = (chain.D_m1.multiply(w[:, None])).toarray()
        ref = np.abs(lhs).max()
        assert np.abs(lhs - rhs).max() < 1e-12 * ref


class TestHomogeneousCrossCheck:
    def test_p1_kernel_equals_scalar_dtbc_kernel(self):
        """Uniform vacuum plate: the p=1 Galerkin kernel IS the exact
        scalar TEM DTBC kernel (both directions)."""
        grid = GridLines(
            x=np.array([0.0, 5.0e-3, 10.0e-3]),
            y=np.arange(7) * 1.0e-3,
            z=np.arange(13) * 1.0e-3,
        )
        mesh = Mesh.from_grid(grid, boundary_conditions=_BC_PEC_Y)
        dt = courant_dt(grid, "normal")
        chain, plane = _chains_for(mesh, dt)
        dz = float(plane.normal_dx)
        w_dt0 = 2.0 * math.pi * 2.0e9 * dt
        _, ps = find_propagating_modes(chain, w_dt0, 1.3 * 2.0 * math.pi * 2.0e9 / C0 * dz)
        track_t = ps[: chain.n_t, 0].real
        fams = track_band_families(chain, dt, np.linspace(1.0e9, 8.0e9, 5), track_t, 1.0, dz)
        V, sv = band_subspace(fams, chain.w_period, svd_tol=1e-8)
        # The TEM profile is frequency-flat: rank 1.
        assert V.shape[1] == 1
        ext = galerkin_exterior(chain, V)
        r = C0 * dt / dz
        assert ext.Dt_p1[0, 0] == pytest.approx(-r * r, rel=1e-10)
        assert ext.Dt_0[0, 0] == pytest.approx(2.0 * r * r, rel=1e-10)
        n_k = 64
        ref = dtbc_kernel(r, 0.0, n_k)
        L_out, cert = matrix_dtbc_kernel(ext.Dt_p1, ext.Dt_0, ext.Dt_m1, n_k)
        L_in, _ = matrix_dtbc_kernel(ext.Dt_m1, ext.Dt_0, ext.Dt_p1, n_k)
        assert cert["l0"] < 1e-14
        # atol covers the rho^m-amplified contour roundoff of the two
        # independent contour integrations (a 1e-9 kernel gap sits at
        # -180 dB in the reflection bound).
        np.testing.assert_allclose(L_out[:, 0, 0], ref, atol=1e-9)
        np.testing.assert_allclose(L_in[:, 0, 0], ref, atol=1e-9)


class TestGalerkinExterior:
    def test_symmetry_certificate_and_enforcement(self, layered_boundary_chain, layered_families):
        chain, _, _ = layered_boundary_chain
        V, sv = band_subspace(layered_families, chain.w_period)
        assert sv[0] == 1.0
        ext = galerkin_exterior(chain, V)
        assert ext.sym_residual < 1e-12
        np.testing.assert_array_equal(ext.Dt_m1, ext.Dt_p1.T)
        np.testing.assert_array_equal(ext.Dt_0, ext.Dt_0.T)

    def test_apriori_reflection_below_criterion(self, layered_boundary_chain, layered_families):
        """Exact a-priori |Gamma| of the built boundary < -100 dB on
        the tracked fundamental points (the WP-R4b gate formula)."""
        chain, _, dt = layered_boundary_chain
        V, _ = band_subspace(layered_families, chain.w_period)
        ext = galerkin_exterior(chain, V)
        fam = layered_families[0]
        pts = [
            (float(fam.freqs[i]), complex(fam.zetas[i]), fam.traces[:, i])
            for i in (0, fam.freqs.size // 2, fam.freqs.size - 1)
        ]
        g = band_apriori_reflection(chain, ext, dt, pts)
        assert max(g) < 1e-5


class TestBoundaryStateMachine:
    @staticmethod
    def _small_exterior():
        grid = GridLines(
            x=np.array([0.0, 5.0e-3, 10.0e-3]),
            y=np.arange(5) * 1.0e-3,
            z=np.arange(13) * 1.0e-3,
        )
        mesh = Mesh.from_grid(grid, boundary_conditions=_BC_PEC_Y)
        dt = courant_dt(grid, "normal")
        chain, plane = _chains_for(mesh, dt)
        dz = float(plane.normal_dx)
        w_dt0 = 2.0 * math.pi * 2.0e9 * dt
        _, ps = find_propagating_modes(chain, w_dt0, 1.3 * 2.0 * math.pi * 2.0e9 / C0 * dz)
        fams = track_band_families(
            chain, dt, np.linspace(1.0e9, 6.0e9, 3), ps[: chain.n_t, 0].real, 1.0, dz
        )
        V, _ = band_subspace(fams, chain.w_period)
        return galerkin_exterior(chain, V)

    def test_auto_extension_is_invariant(self):
        """A run outliving the kernel gives bit-identical states to a
        run with a long kernel from the start (exactness within the
        run is preserved by the extension)."""
        ext = self._small_exterior()
        rng = np.random.default_rng(3)
        n_steps = 40
        coups = 1e-3 * rng.standard_normal((n_steps, ext.p))
        short = BandDTBCBoundary(ext, n_kernel_init=16)
        long = BandDTBCBoundary(ext, n_kernel_init=64)
        for n in range(n_steps):
            xs = short.advance(coups[n])
            xl = long.advance(coups[n])
        # The extension regenerates the kernel on a new contour
        # radius (rho depends on n_kernel), so agreement holds to
        # roundoff, not bitwise.
        np.testing.assert_allclose(xs, xl, rtol=1e-12, atol=1e-16)
        assert short.step_count == n_steps

    def test_source_requires_in_kernel(self):
        ext = self._small_exterior()
        b = BandDTBCBoundary(ext, n_kernel_init=16)
        with pytest.raises(RuntimeError, match="incoming kernel"):
            b.advance(np.zeros(ext.p), np.ones(ext.p))
        b.require_in_kernel()
        b.advance(np.zeros(ext.p), np.ones(ext.p))


class TestContourParallelism:
    """The contour loop may run in processes, and must not change a number.

    The points are independent and each is solved by the same call, so
    splitting them is a wall-clock change only — the kernel has to come
    out bit-identical, or the boundary a resumed run rebuilds would
    differ from the one its state was recorded in.
    """

    def test_threshold_keeps_small_loops_sequential(self, monkeypatch):
        # Below the measured break-even (~3 s of serial work) a spawned
        # pool costs more than it saves.
        assert band_dtbc._contour_workers(4097, 4) == 1
        assert band_dtbc._contour_workers(16385, 4) == 1
        # A production-scale loop is worth splitting — on a machine with
        # cores to spare (a two-core CI runner keeps it sequential).
        import os

        monkeypatch.setattr(os, "cpu_count", lambda: 8)
        assert band_dtbc._contour_workers(131073, 15) > 1
        monkeypatch.setattr(os, "cpu_count", lambda: 2)
        assert band_dtbc._contour_workers(131073, 15) == 1

    def test_parallel_kernel_is_bit_identical(self):
        ext = TestBoundaryStateMachine._small_exterior()
        n_kernel = 512
        saved = band_dtbc._CONTOUR_PARALLEL_MIN_WORK
        try:
            band_dtbc._CONTOUR_PARALLEL_MIN_WORK = float("inf")
            L_seq, cert_seq = matrix_dtbc_kernel(ext.Dt_p1, ext.Dt_0, ext.Dt_m1, n_kernel)
            band_dtbc._CONTOUR_PARALLEL_MIN_WORK = 0.0
            L_par, cert_par = matrix_dtbc_kernel(ext.Dt_p1, ext.Dt_0, ext.Dt_m1, n_kernel)
        finally:
            band_dtbc._CONTOUR_PARALLEL_MIN_WORK = saved
        np.testing.assert_array_equal(L_seq, L_par)
        assert cert_seq["residual"] == cert_par["residual"]


class TestBoundaryCheckpoint:
    """The band boundary's convolution reaches over the whole record, so
    its checkpoint is its two histories plus the projected state — and a
    restore has to be bit-exact, not merely close, because a resumed run
    is contractually identical to an uninterrupted one (DD-070 D4)."""

    @staticmethod
    def _exterior():
        return TestBoundaryStateMachine._small_exterior()

    def test_restore_continues_bit_exactly(self):
        ext = self._exterior()
        rng = np.random.default_rng(11)
        n_a, n_b = 25, 20
        coups = 1e-3 * rng.standard_normal((n_a + n_b, ext.p))

        straight = BandDTBCBoundary(ext, n_kernel_init=64)
        for n in range(n_a + n_b):
            xs = straight.advance(coups[n])

        halted = BandDTBCBoundary(ext, n_kernel_init=64)
        for n in range(n_a):
            halted.advance(coups[n])
        sd = halted.state_dict()

        restored = BandDTBCBoundary(ext, n_kernel_init=64)
        restored.load_state_dict(sd)
        assert restored.step_count == n_a
        for n in range(n_a, n_a + n_b):
            xr = restored.advance(coups[n])

        # Bit-exact: the kernels are rebuilt from the same blocks and the
        # history is restored verbatim, so no seam of any size is allowed.
        np.testing.assert_array_equal(xr, xs)

    def test_restore_carries_the_source_history(self):
        ext = self._exterior()
        rng = np.random.default_rng(12)
        n_a, n_b = 18, 14
        coups = 1e-3 * rng.standard_normal((n_a + n_b, ext.p))
        srcs = 1e-2 * rng.standard_normal((n_a + n_b, ext.p))

        straight = BandDTBCBoundary(ext, n_kernel_init=64)
        straight.require_in_kernel()
        for n in range(n_a + n_b):
            xs = straight.advance(coups[n], srcs[n])

        halted = BandDTBCBoundary(ext, n_kernel_init=64)
        halted.require_in_kernel()
        for n in range(n_a):
            halted.advance(coups[n], srcs[n])

        restored = BandDTBCBoundary(ext, n_kernel_init=64)
        restored.require_in_kernel()
        restored.load_state_dict(halted.state_dict())
        for n in range(n_a, n_a + n_b):
            xr = restored.advance(coups[n], srcs[n])

        np.testing.assert_array_equal(xr, xs)

    def test_source_history_without_the_incoming_kernel_is_refused(self):
        ext = self._exterior()
        driven = BandDTBCBoundary(ext, n_kernel_init=16)
        driven.require_in_kernel()
        driven.advance(np.zeros(ext.p), np.ones(ext.p))
        passive = BandDTBCBoundary(ext, n_kernel_init=16)
        with pytest.raises(RuntimeError, match="incoming kernel"):
            passive.load_state_dict(driven.state_dict())

    def test_rank_mismatch_is_refused(self):
        ext = self._exterior()
        b = BandDTBCBoundary(ext, n_kernel_init=16)
        b.advance(np.zeros(ext.p))
        sd = b.state_dict()
        sd["xt"] = np.zeros(ext.p + 1)
        with pytest.raises(ValueError, match="rank"):
            b.load_state_dict(sd)

    def test_checkpoint_holds_only_the_filled_prefix(self):
        ext = self._exterior()
        b = BandDTBCBoundary(ext, n_kernel_init=256)
        for _ in range(7):
            b.advance(np.zeros(ext.p))
        sd = b.state_dict()
        assert sd["w_hist"].shape == (7, ext.p)
        assert sd["s_hist"].shape == (7, ext.p)


class TestFactoryAndOperator:
    @pytest.fixture(scope="class")
    def port(self):
        mesh = _layered_mesh(nz=12)
        dt = courant_dt(mesh.grid, "normal")
        m_eps = build_M_eps(mesh)
        m_mu = build_M_mu(mesh)
        op = build_band_dtbc_port(
            PortSpecMultiConductor(name="port1", plane=BoxFace.Z_MIN, epsilon_r=None),
            mesh,
            m_eps,
            m_mu,
            dt=dt,
            f_band=(1.0e9, 7.8e9),
            n_grid=5,
            n_kernel_init=64,
        )
        return op, dt

    @pytest.fixture(scope="class")
    def port_no_anchor(self):
        mesh = _layered_mesh(nz=12)
        dt = courant_dt(mesh.grid, "normal")
        op = build_band_dtbc_port(
            PortSpecMultiConductor(name="port1", plane=BoxFace.Z_MIN, epsilon_r=None),
            mesh,
            build_M_eps(mesh),
            build_M_mu(mesh),
            dt=dt,
            f_band=(1.0e9, 7.8e9),
            n_grid=5,
            n_kernel_init=64,
            dc_anchor=False,
        )
        return op, dt

    def test_band_data_contract(self, port):
        op, dt = port
        bd = op.band_data
        assert op.n_modes == len(bd.families) >= 1
        assert bd.p == op.subspace_rank
        assert bd.chain_inward.pairing == "inward"
        assert bd.chain_boundary.pairing == "boundary"
        assert bd.singular_values[0] == 1.0
        assert len(bd.dual_e_profiles) == op.n_modes

    def test_band_excitation_synthesis(self, port):
        op, dt = port
        op.set_excitation_band(0, (2.0e9, 6.5e9), n_syn=8192)
        s = op._src_series
        assert s.shape == (8192, op.subspace_rank)
        peak = np.abs(s).max()
        assert np.abs(s[-256:]).max() < 1e-6 * peak
        # Above the tracked band the direction table ends, so the
        # spectrum must stop there.  Below the span it need not: the
        # DC anchor tabulates the direction down to 0, and the flat
        # low side is what keeps the pulse short.
        sp = np.abs(np.fft.rfft(s[:, 0]))
        fr = np.fft.rfftfreq(8192, dt)
        assert sp[fr > 8.0e9].max() < 1e-6 * sp.max()
        assert sp[fr < 0.5e9].max() > 0.1 * sp.max()
        op.clear_excitation()
        assert op._src_series is None

    def test_dc_anchor_reaches_zero_and_stays_accurate(self, port):
        """The anchored table starts at 0 and interpolates cleanly.

        The anchor makes the direction grid non-uniform (a short first
        interval below the tracked band), which is where a cubic
        spline could ring.  Checked against the closed-form DC limit:
        the first column must be the static Laplace direction.
        """
        op, _ = port
        freqs, U = op._src_directions[0]
        assert op.channel_band(0)[0] == 0.0
        assert freqs[0] == 0.0
        assert np.all(np.diff(freqs) > 0.0)
        # The spline through the table reproduces its own DC column.
        from scipy.interpolate import CubicSpline

        u0 = CubicSpline(freqs, U, axis=0)(0.0)
        assert np.linalg.norm(u0 - U[0]) <= 1e-12 * np.linalg.norm(U[0])

    def test_dc_anchor_can_be_switched_off(self, port_no_anchor):
        """Without the anchor the table is the tracked band alone."""
        op, dt = port_no_anchor
        freqs, _ = op._src_directions[0]
        assert freqs[0] > 0.0
        assert op.channel_band(0)[0] == pytest.approx(freqs[0])
        op.set_excitation_band(0, (2.0e9, 6.5e9), n_syn=8192)
        sp = np.abs(np.fft.rfft(op._src_series[:, 0]))
        fr = np.fft.rfftfreq(8192, dt)
        out = (fr < 0.8e9) | (fr > 8.0e9)
        assert sp[out].max() < 1e-6 * sp.max()

    def test_span_must_fit_subspace_band(self, port):
        op, _ = port
        with pytest.raises(ValueError, match="subspace band"):
            op.set_excitation_band(0, (2.0e9, 9.0e9))

    def test_span_below_the_tracked_band_needs_the_anchor(self, port, port_no_anchor):
        """A span reaching under the tracked band is the anchor's point."""
        op, _ = port
        op.set_excitation_band(0, (0.5e9, 6.0e9), n_syn=8192)
        op_plain, _ = port_no_anchor
        with pytest.raises(ValueError, match="subspace band"):
            op_plain.set_excitation_band(0, (0.5e9, 6.0e9))

    def test_wide_pulse_fails_compactness_gate(self, port):
        op, dt = port

        def wide_pulse(t):
            t0, sig = 0.4e-9, 80e-12
            return math.exp(-0.5 * ((t - t0) / sig) ** 2) * math.sin(2.0 * math.pi * 4.4e9 * t)

        with pytest.raises(ValueError, match="not compact"):
            op.set_excitation(0, wide_pulse, n_syn=2048)
