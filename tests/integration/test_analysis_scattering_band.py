"""Integration: high-level band-subspace DTBC dispatch (DD-063).

An inhomogeneous QTEM line (half-filled layered parallel plate, the
CI-scale DD-057 fixture) driven through ``AnalysisScatteringTD``:
``port_model="auto"`` must detect the failing DTBC uniform-chain
certificate and route the run through the band pipeline
(``build_band_dtbc_port`` → ``set_excitation_band`` →
``compute_band_s_parameters``), reusing the built ports across
excitations via ``reset_state()``.  The component-level acceptance for the
same fixture lives in ``test_qtem_band_dtbc_sparams.py``; the analysis
run must reach the same −100 dB class.
"""

from __future__ import annotations

import tempfile
import warnings

import numpy as np
import pytest

from magnelio import AnalysisScatteringTD, Material, Mesh, MeshControl
from magnelio.geo import Brick, GeometryModel
from magnelio.mesh import BoxFace
from magnelio.ports import PortSpecMultiConductor, PortSpecRectWG


def _segments(*breaks_and_counts):
    out = []
    for lo, hi, n in breaks_and_counts:
        seg = np.linspace(lo, hi, n + 1)
        out.extend(seg if not out else seg[1:])
    return [float(v) for v in out]


def _layered_mesh() -> Mesh:
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
    return Mesh.from_geometry(model, control, f_max=8.0e9)


def _analysis(mesh: Mesh, **kwargs) -> AnalysisScatteringTD:
    return AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions(
            {
                "ymin": "PEC",
                "ymax": "PEC",
                "xmin": "PMC",
                "xmax": "PMC",
                "zmin": "PMC",
                "zmax": "PMC",
            }
        ),
        ports=[
            PortSpecMultiConductor(
                name="port1",
                plane=BoxFace.Z_MIN,
                epsilon_r=None,
            ),
            PortSpecMultiConductor(
                name="port2",
                plane=BoxFace.Z_MAX,
                epsilon_r=None,
            ),
        ],
        f_max=6.8e9,
        f_min=1.8e9,
        n_freq=5,
        verbose=False,
        **kwargs,
    )


@pytest.fixture(scope="module")
def band_result():
    """One two-excitation analysis run at CI scale (ports built once)."""
    analysis = _analysis(
        _layered_mesh(),
        port_model="auto",
        band_options={
            "f_band": (0.3e9, 8.3e9),
            "n_grid": 9,
            "n_syn": 3072,
            "n_kernel_init": 4096,
        },
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*neither a BoundaryCondition.*",
        )
        return analysis.run(
            excited=["port1", "port2"],
            total_time_steps=4064,
        )


def test_auto_dispatch_selects_band_pipeline(band_result):
    assert band_result.port_model_used == "band"


def test_band_s_parameters_reach_reflection_free_class(band_result):
    """|S11| and (via reset_state reuse) |S22| below −100 dB in-band.

    Measured at this fixture scale: |S11| −146…−227 dB, |S22| the
    same class from the *reused* port pair — a state-reset defect
    (stale boundary histories, stale x1 trace) shows up here as a
    broadband floor collapse on the second excitation.
    """
    f_axis = band_result.f_axis
    for out_p, in_p in (("port1", "port1"), ("port2", "port2")):
        s = np.abs(band_result.S(out_p, in_p))
        s_db = 20 * np.log10(s + 1e-300)
        assert s_db.max() < -100.0, (
            f"|S{out_p}{in_p}| max {s_db.max():.1f} dB at {f_axis[np.argmax(s_db)] / 1e9:.2f} GHz"
        )
    for out_p, in_p in (("port2", "port1"), ("port1", "port2")):
        s21_db = 20 * np.log10(np.abs(band_result.S(out_p, in_p)))
        assert np.max(np.abs(s21_db)) < 0.05


def test_band_result_time_domain_waves_raise(band_result):
    with pytest.raises(ValueError, match="band-DTBC"):
        band_result.b("port1")
    # Raw V/I stays inspectable.
    v_sig, i_sig = band_result.signals[("port1", 0)][("port1", 0)]
    assert v_sig.values.size == band_result.n_actual_steps


def test_default_port_model_is_modal():
    """The production default is the modal pipeline (DD-064).

    Developer decision 2026-07-10 after the DD-063 field trial:
    −30 dB-class |S11| with seconds of runtime and time-domain
    power-wave access beats −100 dB floors at kernel-build cost for
    routine QTEM work.  The Mur fallback stays loud (verbose notice)
    and the band pipeline stays one argument away.
    """
    analysis = _analysis(_layered_mesh())
    assert analysis.port_model == "modal"
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*neither a BoundaryCondition.*",
        )
        result = analysis.run(excited=["port1"])
    assert result.port_model_used == "modal"
    # The Mur fallback is orders of magnitude above the band floors;
    # the bound only pins that the default really ran the modal path.
    s11_db = 20 * np.log10(np.abs(result.S("port1", "port1")) + 1e-300)
    assert s11_db.max() > -100.0
    # Time-domain power waves stay available on the default path.
    b1 = result.b("port1")
    assert b1.values.size == result.n_actual_steps


def _near_dc_analysis(**kwargs) -> AnalysisScatteringTD:
    return AnalysisScatteringTD(
        mesh=_layered_mesh().with_boundary_conditions(
            {
                "ymin": "PEC",
                "ymax": "PEC",
                "xmin": "PMC",
                "xmax": "PMC",
                "zmin": "PMC",
                "zmax": "PMC",
            }
        ),
        ports=[
            PortSpecMultiConductor(
                name="port1",
                plane=BoxFace.Z_MIN,
                epsilon_r=None,
            ),
            PortSpecMultiConductor(
                name="port2",
                plane=BoxFace.Z_MAX,
                epsilon_r=None,
            ),
        ],
        f_max=6.8e9,  # f_min left at 0 → axis starts at f_max/201
        port_model="auto",
        verbose=False,
        **kwargs,
    )


def test_near_dc_axis_runs_on_the_default_axis():
    """A default axis (f_min = 0 → first point at f_max/n_freq) runs.

    It did not use to.  The pulsed drive needed spectral roll-off room
    below the first axis point, room ∝ f_axis[0] forced an
    O(1/f_axis[0]) pulse, and the auto-sizing gate refused rather than
    hang in the single-threaded contour-QZ kernel build.  With the
    excitation direction tabulated down to DC there is no lower band
    edge to roll off against, so the pulse follows the measurement
    span and the axis start stops mattering.

    This is the certificate for that claim: the run that used to be
    refused now produces a reflection-free-class S-matrix.
    """
    result = _near_dc_analysis().run(excited=["port1"])
    assert result.port_model_used == "band"
    f_axis = np.asarray(result.f_axis)
    s11_db = 20 * np.log10(np.abs(result.S("port1", "port1")) + 1e-300)
    s21_db = 20 * np.log10(np.abs(result.S("port2", "port1")) + 1e-300)
    # Skip the lowest decade: the a-priori floor of the Galerkin
    # boundary degrades monotonically toward DC (measured -72 dB at
    # 20 MHz against -113 dB at the band edge on the microstrip
    # fixture of the internal record), and the DFT of a finite record
    # is coarsest there.
    band = f_axis >= 0.5e9
    assert s11_db[band].max() < -80.0, (
        f"|S11| max {s11_db[band].max():.1f} dB at "
        f"{f_axis[band][np.argmax(s11_db[band])] / 1e9:.2f} GHz"
    )
    assert np.max(np.abs(s21_db[band])) < 0.05


def test_kernel_is_sized_to_the_record_not_to_a_margin():
    """The ghost kernel outlives the record and stops there.

    The boundary is exact as long as the kernel outlives the run — the
    convolution never reaches a tap beyond it — so the next power of two
    above the record is the entire requirement.  Every tap past it costs
    contour QZ solves (4*n_kernel + 1 per kernel, three per excited run)
    and buys nothing: measured on the certificate fixture, 16384 taps
    against 4096 on a 4064-step record left the floor at -84.3 dB and
    took three times as long.

    Pinned because the margin is invisible — it costs runtime, never
    accuracy, so nothing else in the suite would notice it growing back.
    """
    from magnelio._operators.material_matrices import build_M_eps, build_M_mu
    from magnelio.solver.stability import spectral_dt

    mesh = _layered_mesh().with_boundary_conditions(
        {"ymin": "PEC", "ymax": "PEC", "xmin": "PMC", "xmax": "PMC", "zmin": "PMC", "zmax": "PMC"}
    )
    analysis = _analysis(mesh, port_model="band")
    dt = spectral_dt(mesh, "normal", m_eps=build_M_eps(mesh), m_mu=build_M_mu(mesh))
    for n_steps in (4064, 8192, 20000):
        cfg = analysis._band_setup(analysis.f_axis, dt, n_steps)
        n_kernel = cfg["n_kernel_init"]
        assert n_kernel >= cfg["n_steps"], (
            f"kernel {n_kernel} shorter than the {cfg['n_steps']}-step record "
            f"— the convolution would run past it and the boundary go active"
        )
        assert n_kernel < 2 * max(1024, cfg["n_steps"]), (
            f"kernel {n_kernel} for a {cfg['n_steps']}-step record is more than "
            f"one doubling of margin; the build cost is linear in it"
        )


def test_near_dc_axis_still_refuses_without_the_dc_anchor():
    """The old constraint stays reachable, and still explains itself."""
    analysis = _near_dc_analysis(band_options={"dc_anchor": False})
    with pytest.raises(ValueError, match="auto-sizing"):
        analysis.run()


def test_port_model_band_requires_multiconductor_everywhere():
    mesh = _layered_mesh()
    with pytest.raises(ValueError, match="PortSpecMultiConductor"):
        AnalysisScatteringTD(
            mesh=mesh.with_boundary_conditions(
                {
                    "ymin": "PEC",
                    "ymax": "PEC",
                    "xmin": "PMC",
                    "xmax": "PMC",
                    "zmin": "PMC",
                    "zmax": "PMC",
                }
            ),
            ports=[
                PortSpecMultiConductor(
                    name="port1",
                    plane=BoxFace.Z_MIN,
                    epsilon_r=None,
                ),
                PortSpecRectWG(
                    name="port2",
                    plane=BoxFace.Z_MAX,
                    width_a=10.0e-3,
                    height_b=8.0e-3,
                    n_modes=1,
                ),
            ],
            f_max=6.8e9,
            port_model="band",
            verbose=False,
        ).run()


# ----------------------------------------------------------------------
# Band pipeline through the project store (DD-230)
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def band_project(tmp_path_factory):
    """The same two-excitation band run, streamed into a project."""
    path = tmp_path_factory.mktemp("band_store") / "proj"
    analysis = _analysis(
        _layered_mesh(),
        port_model="auto",
        project=str(path),
        band_options={
            "f_band": (0.3e9, 8.3e9),
            "n_grid": 9,
            "n_syn": 3072,
            "n_kernel_init": 4096,
        },
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*neither a BoundaryCondition.*")
        analysis.run(excited=["port1", "port2"], total_time_steps=4064)
    from magnelio.io.project import open_project

    return open_project(path)


def test_band_run_streams_to_project(band_project):
    """A band run reaches the store at all — DD-063 refused it outright."""
    assert set(band_project.runs) == {"port1_mode0", "port2_mode0"}
    assert all(info["port_model"] == "band" for info in band_project.runs.values())
    assert band_project.settings.port_model_used == "band"


def test_band_project_s_parameters_are_derived_on_read(band_project):
    """The stored run re-derives the reflection-free floor from disk.

    The S-matrix is not written — the per-port chain, plane, tracked
    family and recording profiles are, and the read rebuilds the mesh
    operators.  Reaching the same −100 dB class as the in-memory run
    is what proves the stored record is complete.
    """
    for port in ("port1", "port2"):
        s_db = 20 * np.log10(np.abs(band_project.S(port, port)) + 1e-300)
        assert s_db.max() < -100.0, f"|S{port}{port}| max {s_db.max():.1f} dB"
    for out_p, in_p in (("port2", "port1"), ("port1", "port2")):
        s21_db = 20 * np.log10(np.abs(band_project.S(out_p, in_p)))
        assert np.max(np.abs(s21_db)) < 0.05


def test_band_project_matches_the_in_memory_run(band_project, band_result):
    """Store and RAM agree to solver run-to-run noise, not better.

    The FIT march is not bit-reproducible (parallel reduction order),
    so two identical runs differ by ~1e-8 in V and ~1e-6 in a −145 dB
    S-parameter.  The store is held to that same tolerance; the exact
    claim is made where it can be: the serialisation round-trip below.
    """
    for out_p in ("port1", "port2"):
        for in_p in ("port1", "port2"):
            delta = np.abs(band_project.S(out_p, in_p) - band_result.S(out_p, in_p))
            assert np.nanmax(delta) < 1e-5


def test_band_project_refuses_time_domain_power_waves(band_project):
    """A stored band run refuses a/b as loudly as an in-memory one."""
    with pytest.raises(ValueError, match="band-DTBC"):
        band_project.b("port1")


def test_a_mode_without_a_channel_is_reported(band_project, monkeypatch):
    """A propagating mode that no channel claims must not pass silently.

    The decomposition writes one least-squares system per frequency
    over every channel, so its right-hand side carries the response of
    every mode at the port.  A channel left without a mode is visible:
    it stays NaN.  A *mode* left without a channel is not -- nothing is
    skipped and the fit hands its content to the modes that remain,
    which reads as a result rather than as the model error it is.

    What produces it physically is a port declaring fewer channels than
    its cross-section carries modes above a cut-on, which is too large
    a fixture to build here.  The search is made to return one
    unclaimable extra mode instead, on the store path where S is
    derived on read, so what this pins is that the count reaches the
    user.
    """
    # The decomposition imports the search inside its own body, so the
    # patch has to land on the module it imports from.
    from magnelio.ports._modal import zeta_pencil

    real_find = zeta_pencil.find_propagating_modes

    def with_an_orphan(chain, w_dt, hint, *args, **kwargs):
        zp, pp = real_find(chain, w_dt, hint, *args, **kwargs)
        if zp.size == 0:
            return zp, pp
        # A profile no channel trace overlaps: a single unit entry
        # leaves the greedy assignment nothing above its 0.5 gate, so
        # this mode is found and then dropped.
        orphan = np.zeros_like(pp[:, :1])
        orphan[0] = 1.0
        return np.concatenate([zp, zp[:1]]), np.concatenate([pp, orphan], axis=1)

    monkeypatch.setattr(zeta_pencil, "find_propagating_modes", with_an_orphan)

    # An explicit axis: S is cached when the axis is left to default,
    # and a cache warmed by an earlier read would never reach the
    # patched search.
    with pytest.warns(UserWarning, match="match no recording channel"):
        band_project.S("port1", "port1", f_axis=band_project.f_axis)


def test_band_decomposition_survives_the_round_trip_exactly():
    """Every stored array comes back bit-identical (DD-230).

    The run-to-run noise above lives in the 3D march; the serialisation
    itself must add nothing, so this is pinned on the written record
    rather than on a second solver run.
    """
    import h5py

    from magnelio._operators.material_matrices import build_M_eps, build_M_mu
    from magnelio.io.project import (
        _read_band_decomposition,
        _write_band_decomposition,
    )
    from magnelio.ports._modal.band_dtbc import BandDecomposition
    from magnelio.ports._modal.factory import build_band_dtbc_port
    from magnelio.solver.stability import spectral_dt

    analysis = _analysis(_layered_mesh(), port_model="auto")
    m_eps = build_M_eps(analysis.mesh)
    m_mu = build_M_mu(analysis.mesh)
    dt = spectral_dt(analysis.mesh, "normal", m_eps=m_eps, m_mu=m_mu)
    spec = analysis.ports[0]
    op = build_band_dtbc_port(
        spec,
        analysis.mesh,
        m_eps,
        m_mu,
        dt=dt,
        f_band=(0.3e9, 8.3e9),
        n_grid=9,
        svd_tol=1e-8,
        n_channels=spec.n_modes,
        n_kernel_init=4096,
    )
    live = BandDecomposition.from_operator(op)

    path = tempfile.mkdtemp()
    with h5py.File(f"{path}/band.h5", "w") as f:
        _write_band_decomposition(f.create_group(spec.name), live)
    with h5py.File(f"{path}/band.h5", "r") as f:
        back = _read_band_decomposition(f[spec.name], spec.name)

    chain_l, chain_b = live.chain_inward, back.chain_inward
    for name in ("D_m1", "D_0", "D_p1"):
        assert np.array_equal(getattr(chain_l, name).toarray(), getattr(chain_b, name).toarray()), (
            name
        )
    for name in ("w_period", "free_u", "free_v", "et_indices", "ez_indices", "et_step"):
        assert np.array_equal(getattr(chain_l, name), getattr(chain_b, name)), name
    assert (chain_l.n_t, chain_l.ez_step, chain_l.dt, chain_l.pairing) == (
        chain_b.n_t,
        chain_b.ez_step,
        chain_b.dt,
        chain_b.pairing,
    )
    # A scalar et_step must not come back as a one-element array — the
    # period() index arithmetic broadcasts differently if it does.
    assert type(chain_b.et_step) is type(chain_l.et_step)

    for name in (
        "e_u_indices",
        "h_v_indices",
        "u_edge_uv",
        "u_edge_lengths",
        "e_v_indices",
        "h_u_indices",
        "v_edge_uv",
        "v_edge_lengths",
        "e_u_indices_interior",
        "e_v_indices_interior",
    ):
        assert np.array_equal(getattr(live.plane, name), getattr(back.plane, name)), name
    assert live.plane.face == back.plane.face
    assert live.plane.normal_dx == back.plane.normal_dx
    assert live.plane.u_bounds == back.plane.u_bounds

    assert np.array_equal(live.family_freqs, back.family_freqs)
    assert np.array_equal(live.family_zetas, back.family_zetas)
    for c in range(live.n_modes):
        for name in ("e_u_profiles", "e_v_profiles", "h_u_profiles", "h_v_profiles"):
            assert np.array_equal(getattr(live, name)[c], getattr(back, name)[c]), name
        assert np.array_equal(live.dual_e_profiles[c][0], back.dual_e_profiles[c][0])
        assert np.array_equal(live.dual_e_profiles[c][1], back.dual_e_profiles[c][1])


def test_band_run_checkpoints_its_boundary(band_project):
    """The solver checkpoint carries the band boundary's whole memory.

    The convolution reaches over the entire record, so the boundary
    state is the projected exterior coordinates plus both histories —
    everything :meth:`reset_state` clears.  A checkpoint missing them
    would restart the boundary from zero mid-record and corrupt the
    decomposition silently, which is why DD-230 withheld checkpoints
    here entirely until the state existed (KB-037).
    """
    for name in band_project.runs:
        ckpt = band_project.checkpoint_state(name)
        assert ckpt is not None, f"run {name} wrote no checkpoint"
        ports = ckpt["ports"]
        assert ports, "checkpoint carries no port state"
        for pname, psd in ports.items():
            assert set(psd) == {"boundary", "x1_prev"}, pname
            assert set(psd["boundary"]) == {"xt", "xt_prev", "n", "w_hist", "s_hist"}


_RESUME_BAND_OPTS = {
    "f_band": (0.3e9, 8.3e9),
    "n_grid": 9,
    "n_syn": 3072,
    "n_kernel_init": 4096,
}


def test_band_run_resumes_bit_exactly(tmp_path):
    """A resumed band run is indistinguishable from an uninterrupted one.

    The band boundary convolves over the whole record, so a resume that
    lost any of its memory would show up immediately in the tail — and
    silently, since the decomposition is derived on read.  The contract
    is therefore bit-exactness, not closeness (DD-230 D4).

    The excitation is safe to compare across the seam: ``n_syn`` is
    sized from the band, the skirt and dt, never from the step count,
    so the shorter first leg synthesises the same pulse as the
    uninterrupted reference.
    """
    n1, n_total = 2432, 3200

    def band_run(path, n_steps, **kw):
        return _analysis(
            _layered_mesh(),
            port_model="auto",
            project=str(path),
            band_options=_RESUME_BAND_OPTS,
        ).run(excited=["port1"], total_time_steps=n_steps, **kw)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*neither a BoundaryCondition.*")
        # Both legs go through the store: the in-memory and streamed
        # paths agree only to a tolerance (compared separately above),
        # while the contract under test here is bit-exactness of the seam.
        ref_path = tmp_path / "uninterrupted"
        band_run(ref_path, n_total)
        path = tmp_path / "resumed"
        band_run(path, n1, checkpoint_interval=608)

        from magnelio import resume
        from magnelio.io.project import open_project

        assert open_project(path).runs["port1_mode0"]["n_steps"] == n1
        proj = resume(path, excited=("port1", 0), total_time_steps=n_total, verbose=False)

    ref = open_project(ref_path)
    assert proj.runs["port1_mode0"]["state"] == "done"
    assert proj.runs["port1_mode0"]["n_steps"] == n_total

    # The recorded waves first: the boundary convolves over the whole
    # record, so memory lost at the seam shows up here before it reaches
    # the decomposition.
    for chan in (("port1", 0), ("port2", 0)):
        for a, b in zip(ref.signals[("port1", 0)][chan], proj.signals[("port1", 0)][chan]):
            assert np.array_equal(np.asarray(a.values), np.asarray(b.values)), chan
    for out_p in ("port1", "port2"):
        assert np.array_equal(ref.S(out_p, "port1"), proj.S(out_p, "port1")), out_p
