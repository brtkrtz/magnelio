"""Integration tests for AnalysisScatteringTD — the high-level S-parameter API.

Drives the same WR-90 TE10 setup as ``test_modal_rectwg_propagation.py``
through the high-level :class:`AnalysisScatteringTD` class instead of
the raw FITTimeDomainSolver / PortSignalRecorder / compute_s_parameters
pipeline.  Verifies that the analysis wrapper produces a sensible
``SParameterResult`` for both single-excitation and multi-excitation
runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio import (
    AnalysisScatteringTD,
    BoundaryConditions,
)
from magnelio.analysis.scattering_td import ScatteringTDResult
from magnelio.boundaries import CPMLBoundary, PECBoundary
from magnelio.mesh import BoxFace
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.ports import PortSpecLumped, PortSpecRectWG
from magnelio.signals import WaveformGaussian, WaveformGaussianModulated

WR90_A = 22.86e-3
WR90_B = 10.16e-3


def _wr90_grid_30mm() -> GridLines:
    L_x = 30e-3
    return GridLines(
        x=np.linspace(0.0, L_x, 31),
        y=np.linspace(0.0, WR90_A, 24),
        z=np.linspace(0.0, WR90_B, 11),
    )


def _lateral_pec_bcs() -> dict:
    """PEC side walls, port faces (x) left magnetic."""
    return {
        "ymin": PECBoundary("ymin"),
        "ymax": PECBoundary("ymax"),
        "zmin": PECBoundary("zmin"),
        "zmax": PECBoundary("zmax"),
        "xmin": "PMC",
        "xmax": "PMC",
    }


def _wr90_specs():
    spec_src = PortSpecRectWG(
        name="port1",
        plane=BoxFace.X_MIN,
        width_a=WR90_A,
        height_b=WR90_B,
        n_modes=1,
    )
    spec_load = PortSpecRectWG(
        name="port2",
        plane=BoxFace.X_MAX,
        width_a=WR90_A,
        height_b=WR90_B,
        n_modes=1,
    )
    return [spec_src, spec_load]


def test_construction_validates_inputs():
    """Empty ports / duplicate names / bad frequency band are rejected."""
    mesh = Mesh.from_grid(_wr90_grid_30mm())

    with pytest.raises(ValueError, match="non-empty"):
        AnalysisScatteringTD(
            mesh=mesh.with_boundary_conditions({}),
            ports=[],
            f_max=12.4e9,
            f_min=8.2e9,
        )

    spec_a = PortSpecRectWG(
        name="dup",
        plane=BoxFace.X_MIN,
        width_a=WR90_A,
        height_b=WR90_B,
        n_modes=1,
    )
    spec_b = PortSpecRectWG(
        name="dup",
        plane=BoxFace.X_MAX,
        width_a=WR90_A,
        height_b=WR90_B,
        n_modes=1,
    )
    with pytest.raises(ValueError, match="unique"):
        AnalysisScatteringTD(
            mesh=mesh.with_boundary_conditions({}),
            ports=[spec_a, spec_b],
            f_max=12.4e9,
            f_min=8.2e9,
        )

    with pytest.raises(ValueError, match="f_max"):
        AnalysisScatteringTD(
            mesh=mesh.with_boundary_conditions({}),
            ports=_wr90_specs(),
            f_max=0.0,
        )
    with pytest.raises(ValueError, match="f_min"):
        AnalysisScatteringTD(
            mesh=mesh.with_boundary_conditions({}),
            ports=_wr90_specs(),
            f_max=1e9,
            f_min=1e9,
        )
    with pytest.raises(ValueError, match="n_freq"):
        AnalysisScatteringTD(
            mesh=mesh.with_boundary_conditions({}),
            ports=_wr90_specs(),
            f_max=1e9,
            n_freq=1,
        )
    # f_calc left the public signature with WP3.1 (internal = f_max).
    with pytest.raises(TypeError, match="f_calc"):
        AnalysisScatteringTD(
            mesh=mesh.with_boundary_conditions({}),
            ports=_wr90_specs(),
            f_max=1e9,
            f_calc=1e9,
        )


def test_default_f_axis_construction():
    """``f_axis`` is ``linspace(max(f_min, f_max/n_freq), f_max, n_freq)``."""
    mesh = Mesh.from_grid(_wr90_grid_30mm())

    # f_min = 0: axis starts at f_max/n_freq (power waves undefined at DC).
    a = AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions({}),
        ports=_wr90_specs(),
        f_max=10e9,
        n_freq=201,
    )
    np.testing.assert_allclose(
        a.f_axis,
        np.linspace(10e9 / 201, 10e9, 201),
    )

    # f_min above f_max/n_freq wins as the start point.
    b = AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions({}),
        ports=_wr90_specs(),
        f_max=12.4e9,
        f_min=8.2e9,
        n_freq=5,
    )
    np.testing.assert_allclose(b.f_axis, np.linspace(8.2e9, 12.4e9, 5))


def test_excitation_auto_derived_per_mode():
    """``waveform=None`` derives the waveform from the excited mode.

    WP3.2 rule (ports the legacy ``waveform_for_mode`` selection): the
    effective lower band edge is ``max(f_cutoff, f_min)``; zero edge →
    DC-inclusive ``gaussian``, positive edge → ``modulated_gaussian``.
    Exercised on operator stubs so every branch is cheap to hit; the
    real-operator path is covered by
    ``test_auto_excitation_te10_uses_real_cutoff``.
    """
    from types import SimpleNamespace

    mesh = Mesh.from_grid(_wr90_grid_30mm())

    def _op(omega_c, name="m0"):
        mode = SimpleNamespace(omega_c=omega_c, name=name)
        return SimpleNamespace(
            name="port1",
            discrete_modes=[SimpleNamespace(mode=mode)],
        )

    a = AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions({}),
        ports=_wr90_specs(),
        f_max=10e9,
    )
    # TEM (omega_c = 0) with f_min = 0 → DC-inclusive gaussian.
    exc = a._resolve_waveform(None, _op(0.0), 0)
    assert isinstance(exc, WaveformGaussian)
    assert exc.f_max == 10e9

    # TE/TM: the cut-off sets the lower band edge even for f_min = 0.
    f_c = 6.5e9
    exc = a._resolve_waveform(None, _op(2 * np.pi * f_c, "TE10"), 0)
    assert isinstance(exc, WaveformGaussianModulated)
    np.testing.assert_allclose(exc.f_min, f_c)
    assert exc.f_max == 10e9

    band = AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions({}),
        ports=_wr90_specs(),
        f_max=12.4e9,
        f_min=8.2e9,
    )
    # Explicit f_min above the cut-off wins.
    exc = band._resolve_waveform(None, _op(2 * np.pi * f_c, "TE10"), 0)
    assert isinstance(exc, WaveformGaussianModulated)
    assert (exc.f_min, exc.f_max) == (8.2e9, 12.4e9)

    # f_min > 0 keeps the band rule for TEM modes (bandpass request).
    exc = band._resolve_waveform(None, _op(0.0), 0)
    assert isinstance(exc, WaveformGaussianModulated)
    assert (exc.f_min, exc.f_max) == (8.2e9, 12.4e9)

    # Explicit excitation stays as override, untouched.
    override = WaveformGaussian(f_max=9e9)
    assert a._resolve_waveform(override, _op(0.0), 0) is override

    # Cut-off at/above f_max: no usable band → clear error.
    with pytest.raises(ValueError, match="cut-off"):
        a._resolve_waveform(None, _op(2 * np.pi * 12e9, "TE30"), 0)

    # Mode index beyond the operator's mode list → clear error.
    with pytest.raises(ValueError, match="out of range"):
        a._resolve_waveform(None, _op(0.0), 1)


def test_auto_excitation_te10_uses_real_cutoff():
    """The auto-waveform picks up the built operator's TE10 cut-off."""
    from magnelio._operators.material_matrices import build_M_eps, build_M_mu

    mesh = Mesh.from_grid(_wr90_grid_30mm())
    analysis = AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions(_lateral_pec_bcs()),
        ports=_wr90_specs(),
        f_max=12.4e9,
        verbose=False,  # f_min stays 0 — cut-off must win
    )
    m_eps = build_M_eps(analysis.mesh)
    m_mu = build_M_mu(analysis.mesh)
    op = analysis._build_operator(
        analysis.ports[0],
        m_eps,
        m_mu,
        dt=1e-13,
        f_calc=12.4e9,
    )
    exc = analysis._resolve_waveform(None, op, 0)
    f_c_te10 = 299_792_458.0 / (2.0 * WR90_A)
    assert isinstance(exc, WaveformGaussianModulated)
    # The operator carries the discrete mode's cut-off, which deviates
    # from the analytic value by the grid dispersion (~0.1 % here).
    np.testing.assert_allclose(exc.f_min, f_c_te10, rtol=2e-3)
    assert exc.f_max == 12.4e9


def test_run_full_band_te10_auto_waveform():
    """``f_min=0`` on a hollow WG now works: the auto-waveform sits above cut-off.

    Under the WP3.1 interim band rule this setup excited a DC-inclusive
    Gaussian whose spectrum was ~half below the TE10 cut-off (total
    reflection, slow Mur ringing).  The per-mode rule derives a
    modulated Gaussian over [f_c, f_max] instead.
    """
    mesh = Mesh.from_grid(_wr90_grid_30mm())
    analysis = AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions(_lateral_pec_bcs()),
        ports=_wr90_specs(),
        f_max=12.4e9,
        verbose=False,  # f_min = 0
    )
    result = analysis.run(
        f_axis=np.linspace(9e9, 11e9, 5),
        excited=["port1"],
    )
    S21 = result.S("port2", "port1")
    assert abs(S21[2]) > 0.5, f"|S21|@10GHz = {abs(S21[2]):.3f}, expected > 0.5 for WR-90 TE10"
    # The sampled reference must be the *modulated* Gaussian: it swings
    # negative, which a plain (DC-inclusive) Gaussian never does.
    assert np.min(result.reference_signal.values) < -0.1


def test_run_returns_single_excitation_result():
    """Single-excitation run produces a 1-column SParameterResult.

    Uses the explicit ``waveform=`` override path; the other run
    tests exercise the band-derived default.
    """
    mesh = Mesh.from_grid(_wr90_grid_30mm())
    waveform = WaveformGaussianModulated(f_min=8.2e9, f_max=12.4e9)

    analysis = AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions(_lateral_pec_bcs()),
        ports=_wr90_specs(),
        f_max=12.4e9,
        f_min=8.2e9,
        waveform=waveform,
        verbose=False,
    )

    f_axis = np.linspace(8.5e9, 12.0e9, 11)
    result = analysis.run(
        f_axis=f_axis,
        excited=["port1"],
        accuracy="normal",
    )

    assert isinstance(result, ScatteringTDResult)
    assert result.s_params.n_excitations == 1
    assert result.excitations == (("port1", 0),)
    assert set(result.channels) == {("port1", 0), ("port2", 0)}

    # Time-domain payload: signals dict keyed by excited (port, mode)
    assert ("port1", 0) in result.signals
    inner = result.signals[("port1", 0)]
    assert ("port1", 0) in inner and ("port2", 0) in inner
    V_src, I_src = inner[("port1", 0)]
    assert len(V_src.values) == result.n_actual_steps
    assert len(I_src.values) == result.n_actual_steps
    assert len(result.reference_signal.values) == result.n_actual_steps

    S11 = result.S("port1", "port1")
    S21 = result.S("port2", "port1")
    assert S11.shape == (len(f_axis),)
    assert S21.shape == (len(f_axis),)

    # Mid-band passband sanity: TE10 passes, return loss is bounded.
    mid = len(f_axis) // 2
    assert abs(S21[mid]) > 0.5, (
        f"|S21|@mid = {abs(S21[mid]):.3f}, expected > 0.5 for hollow WR-90 TE10"
    )
    assert abs(S11[mid]) < 0.7, (
        f"|S11|@mid = {abs(S11[mid]):.3f}, expected < 0.7 (return loss > -3 dB)"
    )


def test_run_without_arguments_uses_internal_defaults():
    """``run()`` needs no arguments: internal f-axis, first port excited."""
    mesh = Mesh.from_grid(_wr90_grid_30mm())
    analysis = AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions(_lateral_pec_bcs()),
        ports=_wr90_specs(),
        f_max=12.4e9,
        f_min=8.2e9,
        n_freq=5,
        verbose=False,
    )
    result = analysis.run()
    assert result.excitations == (("port1", 0),)
    np.testing.assert_allclose(result.f_axis, analysis.f_axis)
    assert len(result.f_axis) == 5


def test_run_two_excitations_produces_full_2x2_matrix():
    """``excited=["port1", "port2"]`` runs twice and merges the columns."""
    mesh = Mesh.from_grid(_wr90_grid_30mm())
    analysis = AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions(_lateral_pec_bcs()),
        ports=_wr90_specs(),
        f_max=12.4e9,
        f_min=8.2e9,
        verbose=False,
    )

    f_axis = np.linspace(9e9, 11e9, 5)
    result = analysis.run(
        f_axis=f_axis,
        excited=["port1", "port2"],
    )
    assert result.s_params.n_excitations == 2
    assert result.s_params.is_complete  # 2x2 full matrix
    # Two signal-sets, one per excited pair
    assert set(result.signals.keys()) == {("port1", 0), ("port2", 0)}

    # Reciprocity: S12 ≈ S21 within FFT noise on a passive lossless WG.
    S12 = result.S("port1", "port2")
    S21 = result.S("port2", "port1")
    mid = len(f_axis) // 2
    assert np.abs(S12[mid] - S21[mid]) / abs(S21[mid]) < 0.2, (
        f"reciprocity broken: |S12 - S21|/|S21| = {np.abs(S12[mid] - S21[mid]) / abs(S21[mid]):.3f}"
    )


def test_run_leaves_the_mesh_untouched_and_is_repeatable():
    """run() must not mutate the caller's mesh — the port-plane PEC
    flatten used to be written back (solver setup AND modal-port
    factory), so every LATER operator build on the same mesh computed
    its 2D port modes against a plane stripped of the wall contour.
    With ports on more than one box face that silently broke
    reciprocity (measured: S12 −26 dB vs S21 −3 dB on a three-port
    divider); on same-axis fixtures it merely re-flattened
    idempotently, which is why the suite never saw it."""
    mesh = Mesh.from_grid(_wr90_grid_30mm()).with_boundary_conditions(_lateral_pec_bcs())
    before = mesh.pec_mask_edges.copy()
    analysis = AnalysisScatteringTD(
        mesh=mesh,
        ports=_wr90_specs(),
        f_max=12.4e9,
        f_min=8.2e9,
        verbose=False,
    )
    f_axis = np.linspace(9e9, 11e9, 3)
    first = analysis.run(f_axis=f_axis, excited=["port1"])
    assert np.array_equal(mesh.pec_mask_edges, before), "run() mutated mesh.pec_mask_edges"

    again = analysis.run(f_axis=f_axis, excited=["port1"])
    np.testing.assert_allclose(
        np.abs(again.S("port2", "port1")),
        np.abs(first.S("port2", "port1")),
        rtol=1e-12,
        err_msg="second run() on the same analysis diverged from the first",
    )


def test_run_with_tuple_excited_and_explicit_mode():
    """``excited=[(name, mode_idx)]`` accepted; mode_idx 0 default elsewhere."""
    mesh = Mesh.from_grid(_wr90_grid_30mm())
    analysis = AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions(_lateral_pec_bcs()),
        ports=_wr90_specs(),
        f_max=12.4e9,
        f_min=8.2e9,
        verbose=False,
    )
    result = analysis.run(
        f_axis=np.linspace(9e9, 11e9, 5),
        excited=[("port1", 0)],
    )
    assert result.excitations == (("port1", 0),)


def test_run_with_high_level_boundary_conditions():
    """``boundary_conditions=BoundaryConditions(...)`` is materialised internally.

    The bbox-PEC entries on the modal-port faces (xmin/xmax) are
    skipped by the solver loop; the modal Mur absorber takes over —
    same physical outcome as passing only the lateral PEC dict.  The
    test only checks that the BoundaryConditions facade plumbs through
    without raising and that the result schema is well-formed.
    """
    mesh = Mesh.from_grid(_wr90_grid_30mm())
    bc = BoundaryConditions()  # all-PEC default
    analysis = AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions(bc),
        ports=_wr90_specs(),
        f_max=12.4e9,
        f_min=8.2e9,
        verbose=False,
    )
    f_axis = np.linspace(9e9, 11e9, 5)
    result = analysis.run(f_axis=f_axis, excited=["port1"])
    assert result.s_params.n_excitations == 1
    assert set(result.channels) == {("port1", 0), ("port2", 0)}
    S11 = result.S("port1", "port1")
    S21 = result.S("port2", "port1")
    assert S11.shape == (len(f_axis),)
    assert S21.shape == (len(f_axis),)
    assert not np.any(np.isnan(S21)), "S21 contains NaN at all frequencies"


def test_run_with_discrete_port_spec():
    """A PortSpecLumped is accepted as a port and produces a finite S11.

    Setup: a 30 mm free-space stub with a 1 mm-long lumped port at z=0
    and a CPML termination at zmax.  The reference impedance for the
    power-wave decomposition is the lumped Z0 (50 Ω) — the analysis
    synthesises a ``_LumpedModeStub`` for compute_s_parameters
    internally, so the user does not have to.
    """
    grid = GridLines(
        x=np.linspace(0, 3e-3, 4),
        y=np.linspace(0, 3e-3, 4),
        z=np.linspace(0, 30e-3, 31),
    )
    mesh = Mesh.from_grid(grid)

    spec = PortSpecLumped(
        name="p1",
        start=(1.5e-3, 1.5e-3, 0.0),
        end=(1.5e-3, 1.5e-3, 1e-3),
        Z0=50.0,
    )
    bcs = {f: PECBoundary(f) for f in ("xmin", "xmax", "ymin", "ymax", "zmin")}
    bcs["zmax"] = CPMLBoundary("zmax", grid, thickness_cells=8)

    analysis = AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions(bcs),
        ports=[spec],
        f_max=5e9,
        verbose=False,
    )
    f_axis = np.linspace(0.5e9, 5e9, 11)
    result = analysis.run(f_axis=f_axis, excited=["p1"])

    assert isinstance(result, ScatteringTDResult)
    assert result.s_params.n_excitations == 1
    assert result.channels == (("p1", 0),)
    S11 = result.S("p1", "p1")
    assert S11.shape == (len(f_axis),)
    assert not np.any(np.isnan(S11)), "S11 contains NaN at all frequencies"

    # Time-domain payload reaches the recorder for the lumped channel
    inner = result.signals[("p1", 0)]
    V_sig, I_sig = inner[("p1", 0)]
    assert len(V_sig.values) == result.n_actual_steps
    assert len(I_sig.values) == result.n_actual_steps


def test_bc_pec_faces_consolidated_into_mesh():
    """BC-PEC faces land in ``mesh.pec_mask_edges`` at construction.

    All three input forms — ``BoundaryConditions``, dict of BC
    instances, dict of strings — must produce the same mask, so the 2D
    mode solvers and the auto conductor detection see BC-PEC walls
    like geometric PEC (DD-050 equivalence).  Non-PEC entries (PMC,
    CPML) must not touch the mask (DD-103: that is what makes a
    symmetry plane a symmetry plane).
    """
    grid = _wr90_grid_30mm()
    lateral_pec = {
        "ymin": "PEC",
        "ymax": "PEC",
        "zmin": "PEC",
        "zmax": "PEC",
        "xmin": "PMC",
        "xmax": "PMC",
    }
    expected = Mesh.from_grid(
        grid,
        boundary_conditions=lateral_pec,
    ).pec_mask_edges

    def _build(bc):
        return AnalysisScatteringTD(
            mesh=Mesh.from_grid(grid, boundary_conditions=bc),
            ports=_wr90_specs(),
            f_max=12.4e9,
            f_min=8.2e9,
            verbose=False,
        )

    # Form 1: dict of BC instances (PEC lateral, ports on xmin/xmax).
    a1 = _build(_lateral_pec_bcs())
    assert np.array_equal(a1.mesh.pec_mask_edges, expected)

    # Form 2: dict of strings; PMC entries contribute nothing.
    a2 = _build(
        {
            "xmin": "PMC",
            "xmax": "PMC",
            "ymin": "PEC",
            "ymax": "PEC",
            "zmin": "PEC",
            "zmax": "PEC",
        }
    )
    assert np.array_equal(a2.mesh.pec_mask_edges, expected)

    # Form 3: high-level BoundaryConditions (CPML on the port faces).
    a3 = _build(BoundaryConditions(xmin="CPML", xmax="CPML"))
    assert np.array_equal(a3.mesh.pec_mask_edges, expected)

    # No PEC anywhere: the mask stays untouched.
    a4 = _build({f: "PMC" for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")})
    assert not a4.mesh.pec_mask_edges.any()


def test_string_bc_dict_materialized_to_instances():
    """String entries in a BC dict become runtime BC instances.

    The solver dispatches on ``apply_E`` / ``apply_H`` attributes, so a
    raw string in its BC dict used to be a silent no-op (WP1.2 side
    finding).  ``_resolve_bc`` must materialise strings and pass BC
    instances through unchanged; unknown strings raise — since DD-103
    already when the closure is declared, not when it is resolved.
    """
    from magnelio.boundaries import PMCBoundary

    grid = _wr90_grid_30mm()
    mesh = Mesh.from_grid(grid)
    passthrough = PECBoundary("zmax")

    analysis = AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions(
            {
                "ymin": "PEC",
                "ymax": "PMC",
                "zmin": "CPML",
                "zmax": passthrough,
                "xmin": "PMC",
                "xmax": "PMC",
            }
        ),
        ports=_wr90_specs(),
        f_max=12.4e9,
        f_min=8.2e9,
        verbose=False,
    )
    bc_objects = analysis._resolve_bc()
    assert isinstance(bc_objects["ymin"], PECBoundary)
    assert isinstance(bc_objects["ymax"], PMCBoundary)
    assert isinstance(bc_objects["zmin"], CPMLBoundary)
    assert bc_objects["zmax"] is passthrough

    with pytest.raises(ValueError, match="PErC"):
        mesh.with_boundary_conditions(
            {
                "ymin": "PErC",
                "ymax": "PMC",
                "xmin": "PMC",
                "xmax": "PMC",
                "zmin": "PMC",
                "zmax": "PMC",
            }
        )
