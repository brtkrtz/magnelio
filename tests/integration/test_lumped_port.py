"""Integration test: discrete (lumped) port via the unified Port protocol.

Runs a small FIT-TD simulation with a Thévenin-loaded discrete port in
a free-space domain.  Exercises:

* :class:`PortSpecLumped` + :func:`build_lumped_port`,
* :meth:`PortOperatorLumped.set_excitation`,
* :class:`PortSignalRecorder` recording V/I,
* :meth:`PortSignalRecorder.finalize` after a full ``solver.run()``.

Quantitative gates (DD-075, the 3a discrete-port audit).  A lumped port
is a **single edge-chain** — a 2-terminal element on one grid edge, not a
sheet spanning a distributed mode.  On a wide TEM line it therefore does
**not** act as a clean matched termination (audit finding F4: a single-
edge load on a distributed cross-section reflects resonantly, |S11|→0 dB
regardless of Z0; measured on Nx=1…8 parallel plates), so the classic
"λ/4-stub S11-null" gate is not achievable until a genuine single-edge
line exists (thin-wire, Cluster 3c).  The two physics gates below need no
distributed-mode fixture:

* ``test_discrete_port_thevenin_invariant`` — the semi-implicit update
  enforces ``V(t) + Z0·I(t) = s(t)`` exactly (F2), which validates the
  √W reference (Z0), the ``Σ e·dl`` convention and the sign at machine
  precision, mesh-independently.
* ``test_discrete_port_cotemporal_decomposition`` — the lumped channel's
  power-wave split is co-temporal (no Yee half-step ``exp(+jω·dt/2)`` on
  I), the DD-075 F3 fix.
"""

import math

import numpy as np


def _build_mesh(Nx=3, Ny=3, Nz=30, L_xy=3e-3, L_z=30e-3):
    from magnelio.mesh.grid import GridLines
    from magnelio.mesh.mesher import Mesh

    grid = GridLines(
        x=np.linspace(0, L_xy, Nx + 1),
        y=np.linspace(0, L_xy, Ny + 1),
        z=np.linspace(0, L_z, Nz + 1),
    )
    return Mesh.from_grid(grid), grid


def _gaussian(f_max: float):
    """Same waveform the legacy DiscretePort used as its built-in source."""
    sigma = 2.0 / (math.pi * f_max)
    t0 = 4.0 / f_max
    return lambda t: math.exp(-(((t - t0) / sigma) ** 2))


def test_discrete_port_smoke():
    """Port injects energy, recorder produces V and I, FFT finite."""
    from magnelio._operators.material_matrices import build_M_eps, build_M_mu
    from magnelio.boundaries.cpml import CPMLBoundary
    from magnelio.boundaries.pec import PECBoundary
    from magnelio.ports._lumped import PortSpecLumped, build_lumped_port
    from magnelio.ports.recorder import PortSignalRecorder
    from magnelio.solver.fit_td import FITTimeDomainSolver
    from magnelio.solver.stability import courant_dt

    mesh, grid = _build_mesh(Nx=3, Ny=3, Nz=30, L_xy=3e-3, L_z=30e-3)

    f_max = 5e9
    dt = courant_dt(grid, accuracy="normal")
    n_steps = 800

    spec = PortSpecLumped(
        name="port1",
        start=(1.5e-3, 1.5e-3, 0.0),
        end=(1.5e-3, 1.5e-3, 1e-3),
        Z0=50.0,
    )
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    op = build_lumped_port(spec, mesh, m_eps, m_mu, dt=dt)
    op.set_excitation(0, _gaussian(f_max))

    bcs = {face: PECBoundary(face) for face in ("xmin", "xmax", "ymin", "ymax", "zmin")}
    bcs["zmax"] = CPMLBoundary("zmax", grid, thickness_cells=8)

    recorder = PortSignalRecorder(dt=dt, ports=[op])
    solver = FITTimeDomainSolver(
        mesh=mesh,
        boundary_conditions=bcs,
        ports=[op],
        total_time_steps=n_steps,
        dt=dt,
        verbose=False,
        recorder=recorder,
    )
    solver.run()

    sigs = recorder.finalize()
    assert ("port1", 0) in sigs, f"Missing channel. Got: {list(sigs)}"
    v_signal, i_signal = sigs[("port1", 0)]

    # Sample count = full run length (no early termination)
    assert len(v_signal.values) == n_steps
    assert len(i_signal.values) == n_steps

    # Energy was injected on V and reflected on I
    assert np.max(np.abs(v_signal.values)) > 0.0, "Port V is zero"
    assert np.max(np.abs(i_signal.values)) > 0.0, "Port I is zero"

    # FFT of both signals is finite
    V = np.fft.rfft(v_signal.values)
    I = np.fft.rfft(i_signal.values)
    assert not np.any(np.isnan(V)), "V FFT contains NaN"
    assert not np.any(np.isnan(I)), "I FFT contains NaN"


def test_discrete_port_energy_bounded():
    """Thévenin loading prevents unbounded energy growth."""
    from magnelio._operators.material_matrices import build_M_eps, build_M_mu
    from magnelio.boundaries.pec import PECBoundary
    from magnelio.ports._lumped import PortSpecLumped, build_lumped_port
    from magnelio.ports.recorder import PortSignalRecorder
    from magnelio.solver.fit_td import FITTimeDomainSolver
    from magnelio.solver.stability import courant_dt

    mesh, grid = _build_mesh(Nx=3, Ny=3, Nz=10, L_xy=3e-3, L_z=10e-3)
    dt = courant_dt(grid, accuracy="normal")

    spec = PortSpecLumped(
        name="port1",
        start=(1.5e-3, 1.5e-3, 0.0),
        end=(1.5e-3, 1.5e-3, 1e-3),
        Z0=50.0,
    )
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    op = build_lumped_port(spec, mesh, m_eps, m_mu, dt=dt)
    op.set_excitation(0, _gaussian(5e9))

    bcs = {face: PECBoundary(face) for face in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}

    recorder = PortSignalRecorder(dt=dt, ports=[op])
    solver = FITTimeDomainSolver(
        mesh=mesh,
        boundary_conditions=bcs,
        ports=[op],
        total_time_steps=2000,
        dt=dt,
        verbose=False,
        recorder=recorder,
    )
    solver.run()

    v = recorder.finalize()[("port1", 0)][0].values
    peak_idx = int(np.argmax(np.abs(v)))
    late_rms = float(np.sqrt(np.mean(v[peak_idx + 200 :] ** 2))) if peak_idx + 200 < len(v) else 0.0
    peak_abs = float(np.max(np.abs(v)))
    assert late_rms < peak_abs, "Late-time signal exceeds peak — energy not bounded"


def _run_lumped_source(Z0=50.0, n_steps=400):
    """High-level run of a single excited lumped port in a PEC/CPML box.

    Returns the ``ScatteringTDResult`` (carries ``signals``,
    ``reference_signal``, ``dt``).  ``total_time_steps`` is pinned because
    a weakly-absorbing lumped fixture never reaches ``energy_stop_db`` and
    would march unbounded (audit finding F5).
    """
    from magnelio import AnalysisScatteringTD, Mesh
    from magnelio.boundaries.cpml import CPMLBoundary
    from magnelio.boundaries.pec import PECBoundary
    from magnelio.mesh.grid import GridLines
    from magnelio.ports import PortSpecLumped

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
        Z0=Z0,
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
    result = analysis.run(
        f_axis=f_axis,
        excited=["p1"],
        total_time_steps=n_steps,
        energy_stop_db=None,
    )
    return result, f_axis, Z0


def test_discrete_port_thevenin_invariant():
    """``V(t) + Z0·I(t) = s(t)`` to machine precision (audit F2, DD-075).

    The semi-implicit Thévenin update solves ``i_port = (v_src - v_total)
    / (Z0 + Σβ)`` and injects it so the post-update line integral is
    ``v_total + i_port·Σβ``; hence ``V + Z0·I = v_src`` identically, and
    ``project_V`` re-reads exactly those edges.  The recorded ``V + Z0·I``
    therefore reproduces the source waveform ``s(t)`` — validating the √W
    reference impedance (Z0), the ``Σ e·dl`` convention and the sign with
    no distributed-mode fixture.  The recorder samples V/I at ``t^{n+1}``
    while ``reference_signal`` is on the naive ``n·dt`` axis, so the
    identity holds under a single-step alignment (the smallest of the
    ±1 shifts).
    """
    result, _f_axis, Z0 = _run_lumped_source()

    V, I = result.signals[("p1", 0)][("p1", 0)]
    s = result.reference_signal
    lhs = V.values + Z0 * I.values
    # DD-078: the reference waveform is the incident power-wave amplitude
    # a(t) in √W; the Thévenin source realising it is v_src = 2√Z0·a(t).
    rhs = 2.0 * math.sqrt(Z0) * s.values
    peak = float(np.max(np.abs(rhs)))
    assert peak > 0.0

    n = len(lhs)
    best = min(
        float(np.max(np.abs(lhs[max(0, sh) : n + min(0, sh)] - rhs[max(0, -sh) : n + min(0, -sh)])))
        for sh in (-1, 0, 1)
    )
    assert best / peak < 1e-12, (
        f"Thévenin invariant V + Z0·I = s violated: max residual "
        f"{best:.3e} (rel {best / peak:.2e}); the semi-implicit update or "
        f"the √W/Σe·dl convention is inconsistent."
    )


def test_discrete_port_cotemporal_decomposition():
    """Lumped power-wave split is co-temporal — the DD-075 F3 fix.

    ``compute_s_parameters`` applies the Yee half-step ``exp(+jω·dt/2)``
    correction only to modally-sampled I (V∼e@n+1, I∼h@n+½).  A lumped
    port's ``project_I`` returns the ``t^{n+1}`` Thévenin current (h
    ignored), so V and I are co-temporal and the correction must be
    skipped.  The published ``result.S`` must equal the co-temporal
    ``(V − Z0·I)/(V + Z0·I)`` split, and must differ from the (wrong)
    temporally-corrected split — otherwise a revert to the old unguarded
    path would go undetected.
    """
    result, f_axis, Z0 = _run_lumped_source()

    V, I = result.signals[("p1", 0)][("p1", 0)]
    omega = 2.0 * math.pi * f_axis
    Vf = V.at_frequencies(f_axis)
    If = I.at_frequencies(f_axis)

    S_published = result.S("p1", "p1")
    S_cotemporal = (Vf - Z0 * If) / (Vf + Z0 * If)
    S_temporal = (Vf - Z0 * If * np.exp(+1j * omega * result.dt / 2.0)) / (
        Vf + Z0 * If * np.exp(+1j * omega * result.dt / 2.0)
    )

    finite = np.isfinite(S_published)
    assert finite.any()
    assert np.max(np.abs(S_published[finite] - S_cotemporal[finite])) < 1e-9, (
        "result.S for a lumped port must use the co-temporal power-wave "
        "split (DD-075 F3); it does not."
    )
    # The guard has teeth: the two formulas are meaningfully different, so a
    # regression to the unguarded temporal correction would flip the check.
    # The measured separation moves with the fixture's response (|I|/|V| at
    # the sample frequencies shifts with the step count — 1.50e-3 at
    # dt = 1.83 ps, 8.45e-4 at 2.01 ps), so the floor only needs to clear
    # the 1e-9 equality gate by orders of magnitude, not sit on the
    # measured value.
    assert np.max(np.abs(S_cotemporal[finite] - S_temporal[finite])) > 1e-4
