"""WP-U2/U5 regression: unified multi-mode ports on conductor cross-sections.

``build_modal_port`` on a ``PortSpecMultiConductor`` with homogeneous
scalar ``epsilon_r`` and ``n_modes > K-1`` merges the Laplace TEM
channels with the lowest TE/TM curl-curl channels of the same
cross-section, ordered by ascending cut-off — the WP-R3 unified-port
mechanics with a third family (PORT_MODES_PLAN.md WP-U2; the WP-U1
gate measured the discrete family cross-orthogonality at solver
tolerance).  The QTEM path (``epsilon_r=None``) keeps the cap and
raises with WP-U6 guidance.

Also pinned: the multi-TEM channel basis.  The per-conductor Laplace
solutions are mutually non-orthogonal (32 % on the symmetric
two-wire) and drove the per-channel DTBC feedback unstable (measured
blow-up to 1e64 through the previously untested 2-signal path);
``solve_tem_laplace`` now returns the Gram-matrix eigenbasis of the
TEM subspace — the odd/even pair on the symmetric two-wire, with
distinct line impedances — which is M_eps-orthonormal as the
downstream machinery assumes.  Single-signal cross-sections keep the
historical path bit-identically.

Full measurements: ``validation/merged_port_cw_floors.py``
(CW floors) and the ``examples/straight_waveguide_*.py`` acceptance
scripts.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import erf

from magnelio import AnalysisScatteringTD, Material, Mesh, MeshControl
from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.geo import Brick, Cylinder, Difference, GeometryModel
from magnelio.ports import PortWaveguide
from magnelio.ports._modal import (
    BoxFace,
    ModeType,
    PortSpecMultiConductor,
    build_modal_port,
)
from magnelio.ports._modal.dtbc import destagger_theta, dtbc_wave_impedance
from magnelio.ports.recorder import PortSignalRecorder
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import (
    compute_min_effective_eps,
    compute_min_effective_mu,
    courant_dt,
)

EPS_R_COAX = 2.25


def _coax_mesh(dx=0.12e-3):
    pec = Material.pec()
    diel = Material.from_isotropic(name="pe", epsilon=EPS_R_COAX)
    length = 12.0 * dx
    outer = Cylinder(
        origin=(0.0, 0.0, 0.0), radius=1.475e-3, height=length, axis="z", material=diel
    )
    inner = Cylinder(origin=(0.0, 0.0, 0.0), radius=0.405e-3, height=length, axis="z", material=pec)
    model = GeometryModel(background=pec)
    model.add(Difference(outer, inner))
    model.add(inner)
    return Mesh.from_geometry(
        model,
        MeshControl(min_nodes_per_wavelength=15, max_cell_size=dx),
        f_max=50.0e9,
    )


def _square_coax_mesh(dx=0.12e-3, outer=2.95e-3, inner=0.81e-3):
    """The round coax's dimensions on square conductors: exactly
    four-fold symmetric on the Cartesian grid whatever the section
    path does, so its TE11-like pair is exactly degenerate.  (The round
    coax's tessellated circles split the pair by ~3e-9 in the pencil
    eigenvalue since the section chord budget was unified — above the
    pencil's 1e-9 dedup — and both polarisations then certify as real
    channels; the exact degeneracy this fixture pins was a property of
    the tessellation, not of the coax.)"""
    pec = Material.pec()
    diel = Material.from_isotropic(name="pe", epsilon=EPS_R_COAX)
    length = 12.0 * dx
    o = Brick(origin=(-outer / 2, -outer / 2, 0.0), size=(outer, outer, length), material=diel)
    i = Brick(origin=(-inner / 2, -inner / 2, 0.0), size=(inner, inner, length), material=pec)
    model = GeometryModel(background=pec)
    model.add(Difference(o, i))
    model.add(i)
    return Mesh.from_geometry(
        model,
        MeshControl(min_nodes_per_wavelength=15, max_cell_size=dx),
        f_max=50.0e9,
    )


def _two_wire_mesh(length=6.0e-3):
    w, s, box = 1.0e-3, 3.0e-3, 10.0e-3
    pec = Material.pec()
    air = Material.from_isotropic(name="air", epsilon=1.0)
    domain = Brick(origin=(-box / 2, -box / 2, 0.0), size=(box, box, length), material=air)
    wire1 = Brick(origin=(-s / 2 - w / 2, -w / 2, 0.0), size=(w, w, length), material=pec)
    wire2 = Brick(origin=(s / 2 - w / 2, -w / 2, 0.0), size=(w, w, length), material=pec)
    model = GeometryModel()
    model.add(Difference(domain, wire1, wire2))
    model.add(wire1)
    model.add(wire2)
    return Mesh.from_geometry(
        model,
        MeshControl(min_nodes_per_wavelength=15),
        f_max=35.0e9,
    )


def _build(mesh, spec, f_calc):
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    dt = courant_dt(
        mesh.grid,
        "normal",
        min_effective_eps=compute_min_effective_eps(mesh),
        min_effective_mu=compute_min_effective_mu(mesh),
    )
    return build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=f_calc)


class TestMergedPortComposition:
    def test_coax_tem_plus_te11(self):
        mesh = _coax_mesh()
        with pytest.warns(UserWarning, match="degenerate"):
            op = _build(
                mesh,
                PortSpecMultiConductor(
                    name="p", plane=BoxFace.Z_MIN, epsilon_r=EPS_R_COAX, n_modes=3
                ),
                f_calc=45.0e9,
            )
        assert op.termination_kinds == ["dtbc", "dtbc", "dtbc"]
        types = [dm.mode.mode_type for dm in op.discrete_modes]
        assert types == [ModeType.TEM, ModeType.TE, ModeType.TE]
        labels = [dm.mode.name for dm in op.discrete_modes]
        assert labels[0].startswith("TEM_lap")
        assert labels[1].startswith("TE_num")
        cutoffs = [dm.mode.omega_c for dm in op.discrete_modes]
        assert cutoffs[0] == 0.0
        assert cutoffs == sorted(cutoffs)

    def test_two_wire_2tem_plus_2te(self):
        mesh = _two_wire_mesh().with_boundary_conditions(
            {
                "xmin": "PEC",
                "xmax": "PEC",
                "ymin": "PEC",
                "ymax": "PEC",
                "zmin": "PMC",
                "zmax": "PMC",
            }
        )
        op = _build(
            mesh,
            PortSpecMultiConductor(name="p", plane=BoxFace.Z_MIN, epsilon_r=1.0, n_modes=4),
            f_calc=25.0e9,
        )
        assert op.termination_kinds == ["dtbc"] * 4
        types = [dm.mode.mode_type for dm in op.discrete_modes]
        assert types == [ModeType.TEM, ModeType.TEM, ModeType.TE, ModeType.TE]
        # The odd/even Gram-eigenbasis has distinct line impedances.
        z0 = op.discrete_modes[0].mode.z_line
        z1 = op.discrete_modes[1].mode.z_line
        assert abs(z0 - z1) / max(z0, z1) > 0.2

    def test_two_wire_tem_basis_is_orthonormal(self):
        """The pre-fix per-conductor basis overlapped by 32 % — the
        source of the DTBC feedback blow-up (1e64)."""
        mesh = _two_wire_mesh().with_boundary_conditions(
            {
                "xmin": "PEC",
                "xmax": "PEC",
                "ymin": "PEC",
                "ymax": "PEC",
                "zmin": "PMC",
                "zmax": "PMC",
            }
        )
        op = _build(
            mesh,
            PortSpecMultiConductor(name="p", plane=BoxFace.Z_MIN, epsilon_r=1.0, n_modes=2),
            f_calc=25.0e9,
        )
        n_e = build_M_eps(mesh).size
        pl = op.plane
        for j, dm in enumerate(op.discrete_modes):
            e = np.zeros(n_e)
            e[pl.e_u_indices] = dm.e_u_profile
            e[pl.e_v_indices] = dm.e_v_profile
            v = op.project_V(e)
            expected = np.zeros(len(op.discrete_modes))
            expected[j] = 1.0
            np.testing.assert_allclose(v, expected, atol=1e-10)

    @staticmethod
    def _microstrip_mesh():
        h_sub, w_strip, t_strip = 0.8e-3, 1.5e-3, 0.2e-3
        w_box, h_box, length = 8.0e-3, 5.0e-3, 12.0e-3
        pec = Material.pec()
        air = Material.from_isotropic(name="air", epsilon=1.0)
        diel = Material.from_isotropic(name="FR4", epsilon=4.3)
        model = GeometryModel()
        model.add(Brick(origin=(-w_box / 2, 0.0, 0.0), size=(w_box, h_sub, length), material=diel))
        air_cap = Brick(
            origin=(-w_box / 2, h_sub, 0.0), size=(w_box, h_box - h_sub, length), material=air
        )
        strip = Brick(
            origin=(-w_strip / 2, h_sub, 0.0), size=(w_strip, t_strip, length), material=pec
        )
        model.add(Difference(air_cap, strip))
        model.add(strip)
        mesh = Mesh.from_geometry(
            model,
            MeshControl(min_nodes_per_wavelength=15),
            f_max=25.0e9,
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

    def test_quasi_tem_impedance_is_labelled_quasi_static(self):
        """The report labels a quasi-TEM line impedance for what it is:
        the frequency-flat value of the Laplace mode, a few percent
        below what the discrete wave carries at the top of the band
        (DD-239).  A homogeneous line keeps the plain grid label."""
        from magnelio.ports._modal.mode_report import PortReport

        op = _build(
            self._microstrip_mesh(),
            PortSpecMultiConductor(name="p", plane=BoxFace.Z_MIN, epsilon_r=None, n_modes=1),
            f_calc=25.0e9,
        )
        assert op.port_report.quasi_static
        assert "(quasi-static, on this grid)" in PortReport.from_operator(op).summary()

        coax = _build(
            _coax_mesh(),
            PortSpecMultiConductor(name="p", plane=BoxFace.Z_MIN, epsilon_r=1.0, n_modes=1),
            f_calc=25.0e9,
        )
        assert not coax.port_report.quasi_static
        assert "Ω (on this grid)" in PortReport.from_operator(coax).summary()

    def test_qtem_multimode_via_zeta_pencil(self):
        """WP-U6: epsilon_r=None with n_modes > K-1 serves the true
        hybrid eigenpairs of the zeta pencil at f_calc (Mur per
        DD-064), with dual-basis projections over the non-orthogonal
        channel set."""
        mesh = self._microstrip_mesh()
        op = _build(
            mesh,
            PortSpecMultiConductor(name="p", plane=BoxFace.Z_MIN, epsilon_r=None, n_modes=3),
            f_calc=25.0e9,
        )
        labels = [dm.mode.name for dm in op.discrete_modes]
        assert labels[0].startswith("QTEM_lap")
        assert labels[1].startswith("HYB_zp")
        assert op.termination_kinds == ["mur"] * 3
        cutoffs = [dm.mode.omega_c for dm in op.discrete_modes]
        assert cutoffs[0] == 0.0
        assert cutoffs == sorted(cutoffs)
        # Dual-basis projections: exact unit self-response despite
        # the non-orthogonal hybrid profiles.
        n_e = build_M_eps(mesh).size
        pl = op.plane
        for j, dm in enumerate(op.discrete_modes):
            e = np.zeros(n_e)
            e[pl.e_u_indices] = dm.e_u_profile
            e[pl.e_v_indices] = dm.e_v_profile
            expected = np.zeros(3)
            expected[j] = 1.0
            np.testing.assert_allclose(op.project_V(e), expected, atol=1e-9)

        # Too few propagating modes at f_calc: loud guidance.
        with pytest.raises(ValueError, match="propagate at f_calc"):
            _build(
                mesh,
                PortSpecMultiConductor(name="p", plane=BoxFace.Z_MIN, epsilon_r=None, n_modes=3),
                f_calc=10.0e9,
            )

    def test_qtem_degenerate_hybrid_pair_not_certified(self):
        """Known WP-U6 limit: on a square coax forced through the QTEM
        path the first hybrid is one polarisation of the exactly
        degenerate TE11-like pair — the pencil dedup collapses the
        eigenvalue to one representative whose tangential profile is
        not real in the DD-056 gauge; the factory refuses loudly
        instead of shipping an uncertified channel."""
        mesh = _square_coax_mesh()
        with pytest.raises(ValueError, match="degenerate or complex"):
            _build(
                mesh,
                PortSpecMultiConductor(name="p", plane=BoxFace.Z_MIN, epsilon_r=None, n_modes=2),
                f_calc=45.0e9,
            )


class TestMergedPortScattering:
    """High-level S-parameters through the merged two-wire port — the
    stability regression for the multi-TEM DTBC feedback (pre-fix:
    signals to 1e64, |S11| -17 dB; post-fix: TEM at the port floor)."""

    def test_two_wire_4mode_sparams(self):
        analysis = AnalysisScatteringTD(
            mesh=_two_wire_mesh().with_boundary_conditions(
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
                PortWaveguide(name="port1", plane="zmin", n_modes=4),
                PortWaveguide(name="port2", plane="zmax", n_modes=4),
            ],
            f_max=35.0e9,
            verbose=False,
        )
        report = analysis.solve_ports()["port1"]
        assert [m.f_cutoff for m in report.modes][:2] == [0.0, 0.0]

        # The TE (mode-2) excitation sits near cut-off, where the stored
        # energy rings down only algebraically (band-edge decay) and never
        # reaches the default −70 dB energy stop — under the unbounded
        # default this run would march forever.  Pin the former auto-sized
        # step count (the pre-DD-070 default) so the run is bounded: the
        # TEM channel still energy-stops early and the TE-channel
        # assertions below are truncation-limited by design.
        result = analysis.run(
            excited=[("port1", 0), ("port1", 2)],
            total_time_steps=6254,
        )

        # TEM channel: port floor.
        s11_tem = result.db("port1", "port1", mode_out=0, mode_in=0)
        assert np.max(s11_tem) < -120.0, f"TEM |S11| max {np.max(s11_tem):.1f} dB (pre-fix: -17 dB)"
        p21_tem = sum(
            np.abs(result.S("port2", "port1", mode_out=k, mode_in=0)) ** 2 for k in range(4)
        )
        assert np.max(np.abs(10 * np.log10(p21_tem))) < 0.1

        # TE channel above 1.2 f_c: pulsed band-edge class, and the
        # total transmitted power must be passive and near-unity.
        f_c = report.modes[2].f_cutoff
        band = result.f_axis >= 1.2 * f_c
        s11_te = result.db("port1", "port1", mode_out=2, mode_in=2)[band]
        assert np.max(s11_te) < -25.0
        p21_te = sum(
            np.abs(result.S("port2", "port1", mode_out=k, mode_in=2)[band]) ** 2 for k in range(4)
        )
        assert np.max(p21_te) < 1.06
        assert np.min(10 * np.log10(p21_te)) > -1.0


class TestMergedPortCWFloor:
    """WP-U5 acceptance pin: CW lock-in TE floor of the merged
    two-wire port on its grid-aligned (certified) chain — measured
    -154.6 dB at 1.2 f_c_hat (acceptance line -100 dB; the conformal
    coax TE11 open item is documented in PORT_MODES_PLAN.md WP-U5 and
    merged_port_cw_floors.py, not pinned here)."""

    def test_two_wire_te_channel_floor(self):
        mesh = _two_wire_mesh().with_boundary_conditions(
            {
                "xmin": "PEC",
                "xmax": "PEC",
                "ymin": "PEC",
                "ymax": "PEC",
                "zmin": "PMC",
                "zmax": "PMC",
            }
        )
        m_eps = build_M_eps(mesh)
        m_mu = build_M_mu(mesh)
        dt = courant_dt(
            mesh.grid,
            "normal",
            min_effective_eps=compute_min_effective_eps(mesh),
            min_effective_mu=compute_min_effective_mu(mesh),
        )
        ops = [
            build_modal_port(
                PortSpecMultiConductor(name=lbl, plane=face, epsilon_r=1.0, n_modes=4),
                mesh,
                m_eps,
                m_mu,
                dt=dt,
                f_calc=25.0e9,
            )
            for lbl, face in (("port1", BoxFace.Z_MIN), ("port2", BoxFace.Z_MAX))
        ]
        channel, ratio = 2, 1.2
        assert ops[0].termination_kinds[channel] == "dtbc"
        r, q, z0 = ops[0].dtbc_line_params[channel]

        w_dt = ratio * q
        period = 2.0 * math.pi / w_dt
        sigma = max(6.0 / ((ratio - 1.0) * q), 8.0 * period)
        s_hat = math.sin(w_dt / 2.0)
        sin_b2 = math.sqrt(max(s_hat**2 - (q / 2.0) ** 2, 1e-30)) / r
        v_g = r * r * math.sin(2.0 * math.asin(min(sin_b2, 1.0))) / math.sin(w_dt)
        n_win = int(30 * period)
        n_meas0 = int(10.0 * sigma + 40.0 * period + 3.0 * mesh.Nz / max(v_g, 1e-3))
        n_steps = n_meas0 + n_win + 2
        t0, sig_t, w_phys = 5.0 * sigma * dt, sigma * dt, w_dt / dt

        def waveform(t: float) -> float:
            amp = 0.5 * (1.0 + float(erf((t - t0) / (math.sqrt(2.0) * sig_t))))
            return amp * math.sin(w_phys * t)

        ops[0].set_excitation(channel, waveform)
        recorder = PortSignalRecorder(dt=dt, ports=ops)
        solver = FITTimeDomainSolver(
            mesh=mesh,
            boundary_conditions={},
            ports=ops,
            recorder=recorder,
            total_time_steps=n_steps,
            dt=dt,
            verbose=False,
        )
        solver.run()
        V_sig, I_sig = recorder.finalize(n_steps_actual=n_steps)[("port1", channel)]
        n_grid = np.arange(n_meas0, n_meas0 + n_win)
        basis = np.column_stack([np.cos(w_dt * n_grid), np.sin(w_dt * n_grid)])
        cv, *_ = np.linalg.lstsq(basis, V_sig.values[n_grid], rcond=None)
        ci, *_ = np.linalg.lstsq(basis, I_sig.values[n_grid], rcond=None)
        V = cv[0] - 1j * cv[1]
        I = (ci[0] - 1j * ci[1]) * np.exp(1j * w_dt / 2.0)
        theta = destagger_theta(np.array([w_dt]), r, q)[0]
        Z = dtbc_wave_impedance(np.array([w_dt]), q, z0, "TE")[0]
        sz = np.sqrt(Z)
        ep, em = np.exp(theta), np.exp(-theta)
        a = (V / sz * ep + sz * I) / (ep + em)
        b = (V / sz * em - sz * I) / (ep + em)
        s11_db = 20.0 * math.log10(max(abs(b / a), 1e-300))
        assert s11_db < -120.0, (
            f"merged two-wire TE CW floor {s11_db:.1f} dB (measured -154.6; acceptance -100)"
        )


def _coupled_microstrip_mesh(s=0.5e-3):
    """Edge-coupled microstrip pair: ground + two strips (K = 3), FR4."""
    h_sub, w_strip, t_strip = 0.8e-3, 1.5e-3, 0.2e-3
    w_box, h_box, length = 12.0e-3, 5.0e-3, 12.0e-3
    pec = Material.pec()
    air = Material.from_isotropic(name="air", epsilon=1.0)
    diel = Material.from_isotropic(name="FR4", epsilon=4.3)
    model = GeometryModel()
    model.add(Brick(origin=(-w_box / 2, 0.0, 0.0), size=(w_box, h_sub, length), material=diel))
    air_cap = Brick(
        origin=(-w_box / 2, h_sub, 0.0), size=(w_box, h_box - h_sub, length), material=air
    )
    strips = [
        Brick(origin=(xc - w_strip / 2, h_sub, 0.0), size=(w_strip, t_strip, length), material=pec)
        for xc in (-(w_strip + s) / 2, (w_strip + s) / 2)
    ]
    model.add(Difference(air_cap, *strips))
    for strip in strips:
        model.add(strip)
    model.add_port(PortWaveguide(name="pair", plane="zmin", n_modes=2))
    mesh = Mesh.from_geometry(model, MeshControl(min_nodes_per_wavelength=15), f_max=25.0e9)
    return mesh.with_boundary_conditions(
        {"xmin": "PEC", "xmax": "PEC", "ymin": "PEC", "ymax": "PEC", "zmin": "PMC", "zmax": "PMC"}
    )


class TestCoupledMicrostripModalPort:
    """DD-196: a ``PortWaveguide(n_modes=2)`` on ground + two strips
    resolves to the QTEM path and returns the even/odd modal pair —
    distinct ε_eff, distinct impedances, an orthonormal channel set
    through the production projections."""

    def test_declarative_pair_yields_even_and_odd_qtem_modes(self):
        from magnelio.ports.declarative import resolve_declarative_port

        mesh = _coupled_microstrip_mesh()
        spec = resolve_declarative_port(mesh.ports[0], mesh)
        assert isinstance(spec, PortSpecMultiConductor)
        assert spec.epsilon_r is None and spec.n_modes == 2
        op = _build(mesh, spec, f_calc=25.0e9)
        even, odd = (dm.mode for dm in op.discrete_modes)
        assert [even.name, odd.name] == ["QTEM_lap00", "QTEM_lap01"]
        assert even.mode_type is ModeType.TEM and odd.mode_type is ModeType.TEM
        assert 1.0 < odd.epsilon_r < even.epsilon_r < 4.3
        assert even.z_line > odd.z_line > 0.0
        assert (even.z_line - odd.z_line) / even.z_line > 0.15
        # Unit self-response through the operator's projections.
        n_e = build_M_eps(mesh).size
        pl = op.plane
        for j, dm in enumerate(op.discrete_modes):
            e = np.zeros(n_e)
            e[pl.e_u_indices] = dm.e_u_profile
            e[pl.e_v_indices] = dm.e_v_profile
            expected = np.zeros(2)
            expected[j] = 1.0
            np.testing.assert_allclose(op.project_V(e), expected, atol=1e-9)

    def test_coupling_grows_as_the_gap_closes(self):
        """Coupled-line coupling C = (Z_e − Z_o)/(Z_e + Z_o) rises for a
        narrower gap — the knob the coupler how-to turns."""
        from magnelio.ports.declarative import resolve_declarative_port

        coupling = []
        for s in (1.5e-3, 0.5e-3):
            mesh = _coupled_microstrip_mesh(s=s)
            op = _build(mesh, resolve_declarative_port(mesh.ports[0], mesh), f_calc=25.0e9)
            z_e, z_o = (dm.mode.z_line for dm in op.discrete_modes)
            coupling.append((z_e - z_o) / (z_e + z_o))
        assert 0.0 < coupling[0] < coupling[1] < 1.0
