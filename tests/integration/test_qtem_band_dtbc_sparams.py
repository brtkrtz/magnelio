"""Integration: broadband band-subspace DTBC port on a QTEM line.

WP-R4b-impl acceptance (DD-057), scaled down for CI: a short
half-filled layered parallel plate, ONE pulsed run through the full
production chain (``build_band_dtbc_port`` ->
``set_excitation_band`` -> ``FITTimeDomainSolver`` ->
``PortSignalRecorder`` -> ``compute_band_s_parameters``) and the
per-frequency true-mode |S11| across the measurement span.  The
benchmark (``validation/qtem_band_dtbc_port_floors.py``)
measures below -155 dB on the full-size lines; the bounds here are
generous, and the kernel grid is resolved well enough that they
actually are -- see the ``n_grid`` note below.

A second run of exactly twice the length watches the *length law* of
the floor rather than its absolute value; see
``TestBandDTBCLengthLaw``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from magnelio._backend.array_api import resolve_backend
from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.geo import Brick, GeometryModel
from magnelio.materials.material import Material
from magnelio.mesh.mesher import Mesh, MeshControl
from magnelio.ports._modal import (
    BoxFace,
    PortSpecMultiConductor,
    build_band_dtbc_port,
)
from magnelio.ports.recorder import PortSignalRecorder
from magnelio.post import compute_band_s_parameters
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import courant_dt

#: Base record length.  The doubled run below is exactly 2x this.
N_STEPS_BASE = 4064

#: Kernel length, held FIXED across both record lengths on purpose:
#: KB-038 established that the length law is a property of the
#: convolution length and not of the kernel sizing (holding
#: n_kernel = 65536 while varying only the record reproduced the same
#: floors to 0.1 dB), so scaling the kernel with the record would only
#: mix a second variable into the rate measured here.
N_KERNEL_INIT = 4096


def _segments(*breaks_and_counts):
    out = []
    for lo, hi, n in breaks_and_counts:
        seg = np.linspace(lo, hi, n + 1)
        out.extend(seg if not out else seg[1:])
    return [float(v) for v in out]


@pytest.fixture(scope="module")
def band_mesh():
    w, hy, h_if, nz, dz = 10.0e-3, 8.0e-3, 4.0e-3, 20, 1.0e-3
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


def _band_run(mesh, n_steps, precision=None, backend=None):
    """Run the production chain once and return ``(signals, S, f_axis)``.

    ``precision=None`` follows the suite-wide pin (double, see
    ``tests/conftest.py``); the length-law fixtures below pass
    ``"single"`` explicitly, which is the production default.
    ``backend=None`` likewise follows the suite pin (NumPy); the GPU
    gate passes ``"cupy"`` explicitly, which bypasses it.
    """
    dt = courant_dt(mesh.grid, "normal")

    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    ops = []
    for label, face in (("port1", BoxFace.Z_MIN), ("port2", BoxFace.Z_MAX)):
        ops.append(
            build_band_dtbc_port(
                PortSpecMultiConductor(name=label, plane=face, epsilon_r=None),
                mesh,
                m_eps,
                m_mu,
                dt=dt,
                f_band=(0.3e9, 8.3e9),
                # The floor this fixture asserts is a kernel-fit
                # residual, and the fit's resolution dominates it: at
                # n_grid=9 the worst point read -120.06 dB against the
                # -120 dB bound, at 11 it reads -138.9 and at 13
                # -150.8, with individual frequency points moving up to
                # 52 dB.  Nine points left the gate defending 0.06 dB
                # of margin on an under-resolved fit rather than on the
                # port's accuracy.
                n_grid=13,
                n_kernel_init=N_KERNEL_INIT,
            )
        )
    op1, op2 = ops
    op1.set_excitation_band(0, (1.8e9, 6.8e9), n_syn=3072)

    recorder = PortSignalRecorder(dt=dt, ports=[op1, op2])
    solver = FITTimeDomainSolver(
        mesh=mesh,
        boundary_conditions={},
        ports=[op1, op2],
        recorder=recorder,
        total_time_steps=n_steps,
        dt=dt,
        precision=precision,
        verbose=False,
        **({} if backend is None else {"backend": backend}),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*neither a BoundaryCondition.*",
        )
        solver.run()
    signals = recorder.finalize(n_steps_actual=n_steps)

    f_axis = np.array([1.8e9, 3.0e9, 4.3e9, 5.5e9, 6.8e9])
    S = compute_band_s_parameters(
        signals,
        [op1, op2],
        ("port1", 0),
        f_axis,
    )
    return signals, S, f_axis


def _s11_db(S):
    return 20.0 * np.log10(np.abs(S[("port1", 0)]) + 1e-300)


@pytest.fixture(scope="module")
def band_run(band_mesh):
    return _band_run(band_mesh, N_STEPS_BASE)


@pytest.fixture(scope="module")
def band_run_gpu(band_mesh):
    return _band_run(band_mesh, N_STEPS_BASE, backend="cupy")


@pytest.fixture(scope="module")
def band_run_single(band_mesh):
    return _band_run(band_mesh, N_STEPS_BASE, precision="single")


@pytest.fixture(scope="module")
def band_run_single_doubled(band_mesh):
    return _band_run(band_mesh, 2 * N_STEPS_BASE, precision="single")


class TestBandDTBCSParams:
    def test_s11_floor(self, band_run):
        _, S, f_axis = band_run
        s11_db = _s11_db(S)
        assert np.all(np.isfinite(s11_db))
        assert s11_db.max() < -120.0

    def test_s21_flat(self, band_run):
        _, S, _ = band_run
        s21_db = 20.0 * np.log10(np.abs(S[("port2", 0)]) + 1e-300)
        assert np.all(np.abs(s21_db) < 0.05)

    def test_record_decays(self, band_run):
        signals, _, _ = band_run
        v1 = signals[("port1", 0)][0].values
        assert np.abs(v1[-128:]).max() < 1e-9 * np.abs(v1).max()


class TestBandDTBCLengthLaw:
    """Guard on how fast the port floor degrades with the record length.

    The floor of a band-DTBC port is *not* constant in the length of the
    run: in the production default precision it gets worse the longer
    the march continues, long after the excitation has passed.  That is
    a known, open defect (KB-038 in the internal register), so this
    class deliberately does not assert an absolute floor -- that would
    only re-state the defect and would have to be relaxed every time it
    moved.  What is pinned instead is the *rate*: how many dB the floor
    loses per doubling of the record.  The rate is what must not get
    worse, and it is measurable in minutes, while the length at which
    the defect breaks an absolute acceptance line (49152 steps at full
    size) is far outside any CI budget.

    These fixtures run at ``precision="single"`` on purpose.  The rest
    of the suite is pinned to double by ``tests/conftest.py``, and in
    double this defect is invisible -- which is one reason it went
    unseen.  Measured on this fixture, 2026-09-01, worst / median over
    the five measurement frequencies, kernel held at ``N_KERNEL_INIT``:

        precision   steps    end/peak    worst |S11|    median
        single       4064    4.13e-07      -128.72     -136.19
        single       8128    9.65e-07      -123.97     -129.85
        single      16256    2.02e-06      -118.26     -123.40
        double       4064       --         -149.12     -185.09
        double       8128       --         -149.13     -185.81

    (the two lengths this test runs come from the test itself and
    reproduce bit-for-bit outside pytest; the 16256 row is a standalone
    probe of the same fixture.)  Per doubling of the record, single
    loses 4.75 / 6.35 dB then 5.71 / 6.45 dB, while double loses 0.01 dB
    and *gains* 0.72 dB.  The single-precision rate is the same order as
    the one the internal register records for the full-length layered
    line, about 7.5 dB per doubling -- not re-measured here.  The bounds
    below pin the 4064 -> 8128 pair with roughly 1.7 / 2.3 dB of
    margin.  They are one-sided on purpose: a *smaller* rate means the
    defect got better, and must not fail.
    """

    #: dB the median floor may lose per doubling of the record.
    #: Measured 6.35 dB on 2026-09-01; see the class docstring.
    MAX_MEDIAN_RATE_DB = 8.0

    #: Same for the worst measurement point, which scatters more.
    #: Measured 4.75 dB on 2026-09-01.
    MAX_WORST_RATE_DB = 7.0

    def test_floor_degradation_per_doubling(self, band_run_single, band_run_single_doubled):
        _, s_short, _ = band_run_single
        _, s_long, _ = band_run_single_doubled
        db_short = _s11_db(s_short)
        db_long = _s11_db(s_long)
        assert np.all(np.isfinite(db_short))
        assert np.all(np.isfinite(db_long))

        worst_rate = float(db_long.max() - db_short.max())
        median_rate = float(np.median(db_long) - np.median(db_short))

        assert median_rate < self.MAX_MEDIAN_RATE_DB, (
            f"the band port floor now loses {median_rate:.2f} dB (median) per "
            f"doubling of the record, against {self.MAX_MEDIAN_RATE_DB} dB "
            f"pinned: {np.median(db_short):.2f} dB at {N_STEPS_BASE} steps, "
            f"{np.median(db_long):.2f} dB at {2 * N_STEPS_BASE}"
        )
        assert worst_rate < self.MAX_WORST_RATE_DB, (
            f"the band port floor now loses {worst_rate:.2f} dB (worst point) "
            f"per doubling of the record, against {self.MAX_WORST_RATE_DB} dB "
            f"pinned: {db_short.max():.2f} dB at {N_STEPS_BASE} steps, "
            f"{db_long.max():.2f} dB at {2 * N_STEPS_BASE}"
        )

    def test_doubled_run_still_decays(self, band_run_single_doubled):
        # The rate above is only meaningful while the record really has
        # gone quiet -- otherwise it would measure leftover excitation
        # rather than the port.  end/peak measured 9.65e-07 at 8128
        # steps in single precision on 2026-09-01.
        signals, _, _ = band_run_single_doubled
        v1 = signals[("port1", 0)][0].values
        assert np.abs(v1[-128:]).max() < 1e-5 * np.abs(v1).max()


try:
    resolve_backend("cupy")
    HAS_GPU = True
except Exception:
    HAS_GPU = False


@pytest.mark.skipif(not HAS_GPU, reason="no usable CuPy/CUDA device")
class TestBandDTBCOnGPU:
    """The band port must survive the shipped ``backend="auto"``.

    It did not: ``band_dtbc.py`` was written in ``np.`` throughout and,
    unlike the modal operator, had no host gather, so ``project_V``
    multiplied a host mode profile by a device field slice and the run
    died on the first recorder call.  Nothing saw it -- no GPU test
    named the band port, and ``tests/conftest.py`` pins the suite to
    NumPy.

    The port's own work (subspace, kernels, the projected chain step)
    is host-side double either way; only the field round trip changes.
    So the GPU answer is expected to agree with the CPU one to
    round-off, not merely in trend.
    """

    def test_matches_the_cpu_answer(self, band_run, band_run_gpu):
        _, s_cpu, _ = band_run
        _, s_gpu, _ = band_run_gpu
        db_cpu = _s11_db(s_cpu)
        db_gpu = _s11_db(s_gpu)
        assert np.all(np.isfinite(db_gpu))
        assert np.max(np.abs(db_gpu - db_cpu)) < 1e-6, (
            f"GPU |S11| departs from the CPU answer: {db_gpu} vs {db_cpu}"
        )

    def test_s21_flat_on_gpu(self, band_run_gpu):
        _, s_gpu, _ = band_run_gpu
        s21_db = 20.0 * np.log10(np.abs(s_gpu[("port2", 0)]) + 1e-300)
        assert np.all(np.abs(s21_db) < 0.05)
