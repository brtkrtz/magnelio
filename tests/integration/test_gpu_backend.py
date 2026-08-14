"""End-to-end gates for the CuPy GPU backend (Workstream 3).

Skipped when CuPy or a CUDA device is unavailable (the ``cupy``
backend request raises) — NOT skipped unconditionally: on a CUDA
machine these run for real and pin GPU-vs-CPU agreement of the full
pipeline.

Measured on the reference machine (RTX 4070 SUPER): the coax S-params
agree exactly (max|ΔS| = 0) — the FIT updates are element-wise with a
fixed per-element operation order and the modal V/I recursion runs
host-side either way.  The gates use 1e-12 to stay robust across
driver/hardware generations without admitting real defects.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio import AnalysisScatteringTD, Material, Mesh, open_project, resume
from magnelio._backend.array_api import resolve_backend
from magnelio.boundaries.cpml import CPMLBoundary
from magnelio.boundaries.pec import PECBoundary
from magnelio.materials import DispersionModel
from magnelio.materials.surface_impedance import fit_wall_impedances
from magnelio.mesh import BoxFace
from magnelio.mesh._surfaces import (
    enumerate_sibc_surfaces,
    resolve_wall_conductors,
)
from magnelio.mesh.grid import GridLines
from magnelio.monitors.wall_loss import MonitorWallLoss
from magnelio.ports import PortSpecMultiConductor, PortWaveguide
from magnelio.ports.recorder import PortSignalRecorder
from magnelio.solver._sibc import SIBCSpec
from magnelio.solver.fit_td import FITTimeDomainSolver

try:
    resolve_backend("cupy")
    HAS_GPU = True
except Exception:
    HAS_GPU = False

gpu = pytest.mark.skipif(not HAS_GPU, reason="no usable CuPy/CUDA device")


def _coax_mesh():
    """PTFE-filled rectangular coax on a regular grid (no OCC)."""
    a, b, L = 2e-3, 10e-3, 30e-3
    nx = 10
    lin = np.linspace(-b / 2, b / 2, nx + 1)
    z = np.linspace(0.0, L, 31)
    mesh = Mesh.from_grid(
        GridLines(x=lin, y=lin, z=z),
        background=Material.from_isotropic("PTFE", epsilon=2.1),
        regions=[(Material.pec(), (-a / 2, -a / 2, 0.0, a / 2, a / 2, L))],
    )
    return mesh


def _run_coax_res(
    backend: str, *, project=None, total_time_steps=1500, checkpoint_interval=None, precision=None
):
    analysis = AnalysisScatteringTD(
        mesh=_coax_mesh().with_boundary_conditions(
            {
                "xmin": "PEC",
                "xmax": "PEC",
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
        backend=backend,
        precision=precision,
        project=project,
    )
    kwargs = {}
    if checkpoint_interval is not None:
        kwargs["checkpoint_interval"] = checkpoint_interval
    return analysis.run(
        excited=[("port1", 0)], energy_stop_db=None, total_time_steps=total_time_steps, **kwargs
    )


def _run_coax(backend: str):
    res = _run_coax_res(backend)
    return res.S("port1", "port1"), res.S("port2", "port1")


@gpu
class TestGPUBackend:
    def test_coax_sparams_match_cpu(self):
        s11_g, s21_g = _run_coax("cupy")
        s11_c, s21_c = _run_coax("numpy")
        assert np.max(np.abs(s11_g - s11_c)) < 1e-12
        assert np.max(np.abs(s21_g - s21_c)) < 1e-12
        assert np.max(np.abs(s21_c)) > 0.5  # sane transmission

    def test_cpml_march_matches_cpu(self):
        """CPML ψ recursion + PEC-in-PML masks on the GPU backend."""
        lin = np.linspace(0.0, 10e-3, 11)
        mesh = Mesh.from_grid(
            GridLines(x=lin, y=lin, z=lin),
            regions=[(Material.pec(), (3e-3, 3e-3, 2e-3, 5e-3, 5e-3, 8e-3))],
        )
        mesh = mesh.with_boundary_conditions(
            {
                "xmin": "PEC",
                "xmax": "PEC",
                "ymin": "PEC",
                "ymax": "PEC",
                "zmin": "PMC",
                "zmax": "PMC",
            }
        )
        grid = mesh.grid

        def march(backend):
            bcs = {
                "zmin": CPMLBoundary(face="zmin", grid=grid, thickness_cells=3),
                "zmax": CPMLBoundary(face="zmax", grid=grid, thickness_cells=3),
                **{f: PECBoundary(f) for f in ("xmin", "xmax", "ymin", "ymax")},
            }
            s = FITTimeDomainSolver(
                mesh=mesh,
                boundary_conditions=bcs,
                total_time_steps=200,
                verbose=False,
                backend=backend,
            )
            s.setup()
            rng = np.random.default_rng(4)
            e0 = rng.standard_normal(s._fields.e_flat.size) * 1e-3
            xp = s._xp
            s._fields.e_flat[:] = xp.asarray(e0)
            s.run()
            e = s._fields.e_flat
            h = s._fields.h_flat
            to_np = (lambda a: a.get()) if hasattr(e, "get") else np.asarray
            return to_np(e), to_np(h)

        e_g, h_g = march("cupy")
        e_c, h_c = march("numpy")
        scale = np.abs(e_c).max()
        assert np.max(np.abs(e_g - e_c)) < 1e-12 * scale
        assert np.max(np.abs(h_g - h_c)) < 1e-12 * np.abs(h_c).max()
        assert np.abs(e_c).max() < 1.0  # PML absorbed, stable


# ----------------------------------------------------------------------
# WP-G1: device-staged V/I recording
# ----------------------------------------------------------------------


def _disable_staging(monkeypatch):
    """Force the recorder onto the pre-WP-G1 immediate path (the A/B)."""

    def _no_stage(self, e, h):
        self._stage_list = [None] * len(self._ports)
        self._staged = False

    monkeypatch.setattr(PortSignalRecorder, "_init_stages", _no_stage)


def _vi(signals_for_excitation):
    return {k: (v[0].values.copy(), v[1].values.copy()) for k, v in signals_for_excitation.items()}


def _assert_vi_bit_exact(ref_vi, got_vi, tag):
    assert set(ref_vi) == set(got_vi), f"{tag}: channel set differs"
    for chan, (rv, ri) in ref_vi.items():
        gv, gi = got_vi[chan]
        assert gv.shape == rv.shape, f"{tag} {chan}: length {gv.shape} != {rv.shape}"
        assert np.array_equal(rv, gv), (
            f"{tag} {chan}: V not bit-exact, max|Δ|={float(np.max(np.abs(rv - gv))):.3e}"
        )
        assert np.array_equal(ri, gi), (
            f"{tag} {chan}: I not bit-exact, max|Δ|={float(np.max(np.abs(ri - gi))):.3e}"
        )


@gpu
class TestRecorderStagingGPU:
    """WP-G1 gates: the device ring buffer changes WHEN samples cross
    the bus, never the numbers — staged vs immediate must be bit-exact
    on the same GPU run, through the store, and across a resume seam."""

    def test_vi_bit_identical_vs_immediate(self, monkeypatch):
        res_staged = _run_coax_res("cupy")
        with monkeypatch.context() as m:
            _disable_staging(m)
            res_imm = _run_coax_res("cupy")
        key = ("port1", 0)
        _assert_vi_bit_exact(
            _vi(res_imm.signals[key]),
            _vi(res_staged.signals[key]),
            "staged-vs-immediate",
        )
        assert np.array_equal(res_staged.S("port1", "port1"), res_imm.S("port1", "port1"))
        assert np.array_equal(res_staged.S("port2", "port1"), res_imm.S("port2", "port1"))

    def test_streamed_store_bit_identical(self, tmp_path, monkeypatch):
        """The store reader sees identical results.h5 content: every
        flush drains the ring buffer through the identical host dots."""
        _run_coax_res(
            "cupy", project=tmp_path / "staged", total_time_steps=600, checkpoint_interval=100
        )
        with monkeypatch.context() as m:
            _disable_staging(m)
            _run_coax_res(
                "cupy", project=tmp_path / "imm", total_time_steps=600, checkpoint_interval=100
            )
        proj_s = open_project(tmp_path / "staged")
        proj_i = open_project(tmp_path / "imm")
        key = ("port1", 0)
        _assert_vi_bit_exact(
            _vi(proj_i.signals[key]),
            _vi(proj_s.signals[key]),
            "streamed-store",
        )

    def test_resume_bit_exact_across_drain_seam(self, tmp_path, monkeypatch):
        """Stop mid-run (drain + checkpoint), resume on the GPU, and
        match one uninterrupted staged GPU run bit for bit."""
        monkeypatch.setenv("MAGNELIO_BACKEND", "cupy")  # resume → GPU
        ref = _run_coax_res("cupy", total_time_steps=600)
        p = tmp_path / "pp"
        _run_coax_res("cupy", project=p, total_time_steps=250, checkpoint_interval=50)
        proj = resume(p, excited=("port1", 0), total_time_steps=600, verbose=False)
        key = ("port1", 0)
        _assert_vi_bit_exact(
            _vi(ref.signals[key]),
            _vi(proj.signals[key]),
            "resume-drain-seam",
        )


# ----------------------------------------------------------------------
# WP-G3: CUDA-graph capture of the device phases
# ----------------------------------------------------------------------


def _noise_march(mesh, bc_builder, *, steps=150, sibc=None, seed=4):
    """March ``steps`` on the CuPy backend from a seeded noise IC.

    Returns ``(e_host, h_host, solver)`` — the caller compares marched
    fields between the graph and eager paths bit for bit.
    """
    s = FITTimeDomainSolver(
        mesh=mesh,
        boundary_conditions=bc_builder(mesh),
        total_time_steps=steps,
        verbose=False,
        backend="cupy",
        sibc=sibc,
    )
    s.setup()
    rng = np.random.default_rng(seed)
    e0 = rng.standard_normal(s._fields.e_flat.size) * 1e-3
    s._fields.e_flat[:] = s._xp.asarray(e0)
    s.run()
    return (s._fields.e_flat.get(), s._fields.h_flat.get(), s)


def _cpml_box_mesh():
    lin = np.linspace(0.0, 10e-3, 11)
    mesh = Mesh.from_grid(
        GridLines(x=lin, y=lin, z=lin),
        regions=[(Material.pec(), (3e-3, 3e-3, 2e-3, 5e-3, 5e-3, 8e-3))],
    )
    return mesh.with_boundary_conditions(
        {
            "xmin": "PEC",
            "xmax": "PEC",
            "ymin": "PEC",
            "ymax": "PEC",
            "zmin": "PMC",
            "zmax": "PMC",
        }
    )


def _cpml_bcs(mesh):
    grid = mesh.grid
    return {
        "zmin": CPMLBoundary(face="zmin", grid=grid, thickness_cells=3),
        "zmax": CPMLBoundary(face="zmax", grid=grid, thickness_cells=3),
        **{f: PECBoundary(f) for f in ("xmin", "xmax", "ymin", "ymax")},
    }


def _pec_bcs(mesh):
    return {f: PECBoundary(f) for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}


@gpu
class TestCudaGraphsGPU:
    """WP-G3 gates: the two device phases replay as CUDA graphs — the
    identical kernels on the identical pointers, so a graph march must
    be bit-identical to the eager GPU march; capture failure falls back
    to eager with one warning, results unchanged."""

    def test_graphs_engage_and_match_eager_cpml(self, monkeypatch):
        monkeypatch.delenv("MAGNELIO_GPU_GRAPHS", raising=False)
        e_g, h_g, s_g = _noise_march(_cpml_box_mesh(), _cpml_bcs)
        assert s_g._gpu_graphs is not None
        assert s_g._gpu_graphs.ready  # captured, no fallback
        assert not s_g._gpu_graphs.failed

        monkeypatch.setenv("MAGNELIO_GPU_GRAPHS", "0")
        e_e, h_e, s_e = _noise_march(_cpml_box_mesh(), _cpml_bcs)
        assert s_e._gpu_graphs is None  # the deterministic anchor
        assert np.array_equal(e_g, e_e)
        assert np.array_equal(h_g, h_e)
        assert np.abs(e_e).max() > 0.0  # marched something real

    def test_dispersive_march_graph_vs_eager(self, monkeypatch):
        lin = np.linspace(0.0, 8e-3, 9)
        model = DispersionModel.debye(eps_inf=2.0, delta_eps=[0.4], tau=[5.0e-11])

        def mesh():
            return Mesh.from_grid(
                GridLines(x=lin, y=lin, z=lin),
                regions=[
                    (Material.dispersive("debye", model), (2e-3, 2e-3, 2e-3, 6e-3, 6e-3, 6e-3))
                ],
            )

        monkeypatch.delenv("MAGNELIO_GPU_GRAPHS", raising=False)
        e_g, h_g, s_g = _noise_march(mesh(), _pec_bcs, seed=7)
        assert s_g._dispersion is not None  # ADE really in the loop
        assert s_g._gpu_graphs.ready

        monkeypatch.setenv("MAGNELIO_GPU_GRAPHS", "0")
        e_e, h_e, _ = _noise_march(mesh(), _pec_bcs, seed=7)
        assert np.array_equal(e_g, e_e)
        assert np.array_equal(h_g, h_e)

    def test_sibc_march_graph_vs_eager(self, monkeypatch):
        d = 1e-3
        lin = np.arange(9) * d
        faces = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")

        def build():
            metal = Material.lossy_metal("cu", sigma=5.8e7)
            mesh = Mesh.from_grid(
                GridLines(x=lin, y=lin, z=lin),
                regions=[(metal, (2 * d, 2 * d, 1 * d, 6 * d, 5 * d, 3 * d))],
            )
            surfs = enumerate_sibc_surfaces(mesh, bc_pec_faces=faces)
            resolved = resolve_wall_conductors(mesh, surfs, sigma=5.8e7)
            fits = fit_wall_impedances(resolved, 1e9, 1e11)
            return mesh, SIBCSpec(surfaces=tuple(surfs), fits=fits)

        monkeypatch.delenv("MAGNELIO_GPU_GRAPHS", raising=False)
        mesh, spec = build()
        e_g, h_g, s_g = _noise_march(mesh, _pec_bcs, sibc=spec, seed=9)
        assert s_g._sibc is not None  # SIBC really in the loop
        assert s_g._gpu_graphs.ready

        monkeypatch.setenv("MAGNELIO_GPU_GRAPHS", "0")
        mesh, spec = build()
        e_e, h_e, _ = _noise_march(mesh, _pec_bcs, sibc=spec, seed=9)
        assert np.array_equal(e_g, e_e)
        assert np.array_equal(h_g, h_e)

    def test_capture_failure_falls_back_eager(self, monkeypatch):
        import cupy

        monkeypatch.delenv("MAGNELIO_GPU_GRAPHS", raising=False)

        def boom(self, *a, **k):
            raise RuntimeError("forced capture failure (gate)")

        monkeypatch.setattr(cupy.cuda.Stream, "begin_capture", boom)
        with pytest.warns(UserWarning, match="CUDA-graph capture"):
            e_g, h_g, s_g = _noise_march(_cpml_box_mesh(), _cpml_bcs)
        assert s_g._gpu_graphs is not None
        assert s_g._gpu_graphs.failed
        assert not s_g._gpu_graphs.ready

        monkeypatch.undo()
        monkeypatch.setenv("MAGNELIO_GPU_GRAPHS", "0")
        e_e, h_e, _ = _noise_march(_cpml_box_mesh(), _cpml_bcs)
        assert np.array_equal(e_g, e_e)  # behaviour, not results
        assert np.array_equal(h_g, h_e)


@gpu
class TestGPUSinglePrecision:
    """The float32 CUDA kernel (DD-094 / plan WP2): accurate and engaged.

    Single differs from double at the ~1e-6 field floor, so a nonzero-but-
    tiny |ΔS| both proves the physical result is unaffected AND that the
    float32 path genuinely ran — a silent fallback to the double kernel
    would give an *exactly* zero delta and fail the lower bound.  Both
    precisions are requested explicitly, which wins over the suite's
    MAGNELIO_PRECISION=double pin.
    """

    def test_single_kernel_accurate_and_engaged(self):
        r_s = _run_coax_res("cupy", precision="single")
        r_d = _run_coax_res("cupy", precision="double")
        s11_s, s21_s = r_s.S("port1", "port1"), r_s.S("port2", "port1")
        s11_d, s21_d = r_d.S("port1", "port1"), r_d.S("port2", "port1")

        dS = max(np.max(np.abs(s11_s - s11_d)), np.max(np.abs(s21_s - s21_d)))
        assert dS < 1e-4, f"single departs from double by {dS:.2e} (> 1e-4)"
        assert dS > 0.0, (
            "single is bit-identical to double — the float32 kernel did not "
            "engage (silent fallback to the double path)"
        )
        assert np.max(np.abs(s21_d)) > 0.5  # sane transmission

    def test_single_field_store_is_float32_on_gpu(self):
        """precision='single' allocates a float32 device field store."""
        solver = FITTimeDomainSolver(
            mesh=_coax_mesh(),
            boundary_conditions={
                f"{ax}{s}": PECBoundary(f"{ax}{s}")
                for ax in ("x", "y", "z")
                for s in ("min", "max")
            },
            total_time_steps=50,
            verbose=False,
            backend="cupy",
            precision="single",
        )
        solver.setup()
        assert solver._fields.e_flat.dtype == np.float32
        assert solver._alpha_E.dtype == np.float32
        solver.run()
        assert np.isfinite(float(solver._fields.e_flat.get().max()))


# ----------------------------------------------------------------------
# KB-006: MonitorWallLoss on the GPU backend
# ----------------------------------------------------------------------


@gpu
class TestWallLossMonitorGPU:
    """KB-006 gate: the perturbative wall-loss monitor must run on the
    CuPy backend (it used to crash at the first recorded step —
    ``np.asarray`` on device arrays plus host-side fancy indexing) and
    report the same fractions as the NumPy backend: the FIT march is
    element-wise identical across backends and the monitor's DFT
    accumulators live host-side either way."""

    def test_fraction_matches_cpu(self):
        w_a, gap_b, length = 10e-3, 5e-3, 30e-3
        freqs = np.linspace(2e9, 10e9, 5)

        def run(backend):
            mon = MonitorWallLoss(
                freqs=freqs,
                reference_plane=("z", 2e-3),
                sigma=5.8e7,
                bc_faces=("ymin", "ymax"),
            )
            ana = AnalysisScatteringTD(
                mesh=Mesh.from_grid(
                    GridLines(
                        x=np.linspace(0, w_a, 11),
                        y=np.linspace(0, gap_b, 6),
                        z=np.linspace(0, length, 61),
                    ),
                    boundary_conditions={
                        "xmin": "PMC",
                        "xmax": "PMC",
                        "ymin": "PEC",
                        "ymax": "PEC",
                        "zmin": "PEC",
                        "zmax": "PEC",
                    },
                ),
                ports=[
                    PortSpecMultiConductor(name="p1", plane=BoxFace.Z_MIN, n_modes=1),
                    PortSpecMultiConductor(name="p2", plane=BoxFace.Z_MAX, n_modes=1),
                ],
                f_max=10e9,
                monitors=(mon,),
                backend=backend,
                verbose=False,
            )
            ana.run(f_axis=freqs, excited=["p1"], energy_stop_db=None, total_time_steps=800)
            return mon

        mon_g = run("cupy")
        mon_c = run("numpy")
        frac_g = mon_g.dissipated_fraction
        frac_c = mon_c.dissipated_fraction
        assert set(frac_g) == set(frac_c)
        for tag in frac_c:
            np.testing.assert_allclose(frac_g[tag], frac_c[tag], rtol=1e-12)
        # sane physics, not just agreement: both plates lossy and equal
        np.testing.assert_allclose(frac_c["ymin"], frac_c["ymax"], rtol=1e-9)
        assert np.all(frac_c["total"] > 0)
