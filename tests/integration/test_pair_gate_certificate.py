"""DD-228 regression: the uniform-chain pair gate (stage 1) reports itself.

Stage 2 of the DTBC certificate — the feed-chain slab defect — has
warned since DD-067.  Stage 1, the transversal pair-product gate, made
the same decision quietly: a channel whose feed cross-section is not a
uniform discrete chain keeps running, on modal Mur-1st, trading a
1e-14 termination for a -30 dB-class reflection floor, while its
geometric twin on the same model keeps the exact one (the KB-022
failure mode, measured on the DD-165 stripline coupler).

Pinned here: the gate stays silent on a uniform feed, warns when the
transversal masses are non-uniform *without* a slab defect (so stage 2
cannot fire and mask it), publishes the decision on the mode report,
and holds its tongue where a fallback is not a surprise — an explicit
``termination="mur"``, or a cross-section that is genuinely
inhomogeneous rather than jittered (a QTEM line deviates at the
material-contrast level and was never eligible for the scalar chain).

The meshing-time half — which pair-coupled masses rest on a ladder the
pairing accepted without pinning it — is pinned in
``tests/unit/test_operators.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio import Material, Mesh
from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.mesh.grid import GridLines
from magnelio.ports import PortSpecMultiConductor
from magnelio.ports._modal import BoxFace, build_modal_port
from magnelio.ports._modal.mode_report import PortReport
from magnelio.solver.stability import courant_dt


def _rect_coax_mesh():
    """Rectangular coax, uniform along z — an exact discrete chain."""
    grid = GridLines(
        x=np.linspace(-5e-3, 5e-3, 21),
        y=np.linspace(-5e-3, 5e-3, 21),
        z=np.linspace(0.0, 10e-3, 21),
    )
    mesh = Mesh.from_grid(
        grid,
        regions=[(Material.pec(), (-1e-3, -1e-3, grid.z[0], 1e-3, 1e-3, grid.z[-1]))],
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


def _tilt_transversal_mu(mesh, m_mu, factor):
    """Scale one Hx column by ``factor``, uniformly along the port normal.

    The perturbation has to be translation-invariant along z, or the
    stage-2 slab check sees it first and the stage-1 gate never gets to
    speak.  That masking is not a test artefact: it is why the coupler
    of KB-022 could not be used to measure stage 1 directly.

    The column sits in the dielectric gap (``j = 1``), not on the
    centre conductor — a column inside the PEC carries no modal weight
    and the gate rightly ignores it.
    """
    grid = mesh.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_Hx = (Nx + 1) * Ny * Nz
    out = m_mu.copy()
    hx = out[:n_Hx].reshape(Nx + 1, Ny, Nz)
    hx[Nx // 2, 1, :] *= factor
    out[:n_Hx] = hx.ravel()
    return out


def _build(mesh, m_mu, **kwargs):
    m_eps = build_M_eps(mesh)
    dt = courant_dt(mesh.grid, "normal")
    spec = PortSpecMultiConductor(
        name="p",
        plane=BoxFace.Z_MIN,
        epsilon_r=1.0,
        n_modes=1,
    )
    return build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=8.0e9, **kwargs)


class TestPairGateCertificate:
    def test_uniform_chain_is_silent_and_exact(self, recwarn):
        mesh = _rect_coax_mesh()
        op = _build(mesh, build_M_mu(mesh))
        assert op.termination_kinds == ["dtbc"]
        assert op._dtbc_pair_spread[0] < 1e-8
        assert not [w for w in recwarn if "uniform discrete chain" in str(w.message)]

    def test_non_uniform_cross_section_falls_back_loudly(self):
        mesh = _rect_coax_mesh()
        m_mu = _tilt_transversal_mu(mesh, build_M_mu(mesh), 1.0 + 1e-4)
        with pytest.warns(UserWarning, match="uniform discrete chain"):
            op = _build(mesh, m_mu)
        assert op.termination_kinds == ["mur"]
        assert op._dtbc_pair_spread[0] > 1e-8

    def test_report_publishes_the_decision(self):
        mesh = _rect_coax_mesh()
        exact = PortReport.from_operator(_build(mesh, build_M_mu(mesh)))
        assert exact.modes[0].termination == "dtbc"
        assert exact.modes[0].chain_spread < 1e-8
        assert "termination = dtbc" in exact.summary()

        m_mu = _tilt_transversal_mu(mesh, build_M_mu(mesh), 1.0 + 1e-4)
        with pytest.warns(UserWarning, match="uniform discrete chain"):
            fallback = PortReport.from_operator(_build(mesh, m_mu))
        assert fallback.modes[0].termination == "mur"
        assert fallback.modes[0].chain_spread > 1e-8

    def test_a_genuinely_inhomogeneous_cross_section_is_not_a_defect(self, recwarn):
        """Far above the gate, the fallback is the model, not a jitter.

        An inhomogeneous line deviates at the material-contrast level;
        it never qualified for the scalar chain, and the answer is a
        different port model, not a mesh fix.  Warning about it every
        run would be a model judgement — the decision stays readable
        on the report instead.
        """
        mesh = _rect_coax_mesh()
        m_mu = _tilt_transversal_mu(mesh, build_M_mu(mesh), 1.5)
        op = _build(mesh, m_mu)
        assert op.termination_kinds == ["mur"]
        assert op._dtbc_pair_spread[0] > 1e-4
        assert not [w for w in recwarn if "uniform discrete chain" in str(w.message)]
        assert PortReport.from_operator(op).modes[0].termination == "mur"

    def test_requested_mur_is_not_a_withheld_certificate(self, recwarn):
        """``termination="mur"`` is the caller's own choice, not a veto.

        The factory has no such argument today (only a direct
        ``PortOperatorModal`` construction reaches it), so the guard is
        exercised on a built operator.
        """
        mesh = _rect_coax_mesh()
        m_mu = _tilt_transversal_mu(mesh, build_M_mu(mesh), 1.0 + 1e-4)
        with pytest.warns(UserWarning, match="uniform discrete chain"):
            op = _build(mesh, m_mu)
        recwarn.clear()
        op._termination = "mur"
        op._report_withheld_dtbc()
        assert not [w for w in recwarn if "uniform discrete chain" in str(w.message)]


class TestMurFallbackNotice:
    """The fallback notice states the trade, and its advice runs.

    The gate's warning (above) fires only inside the marginal band,
    because a genuinely inhomogeneous cross-section is a model, not a
    defect.  That leaves the commonest production case — a microstrip
    — with no word at all about the termination it got, so the modal
    pipeline explains itself instead: which channels fell back, what
    that costs, what the run keeps in exchange, and what the
    reflection-free alternative would cost *on this model*.  The last
    part is the one that has to be computed rather than asserted: the
    band pipeline needs a frequency axis clear of DC, and a notice
    that recommends a switch which would raise is worse than none.
    """

    @staticmethod
    def _spec():
        return PortSpecMultiConductor(
            name="p",
            plane=BoxFace.Z_MIN,
            epsilon_r=1.0,
            n_modes=1,
        )

    def _analysis(self, mesh, **kwargs):
        from magnelio import AnalysisScatteringTD

        return AnalysisScatteringTD(
            mesh=mesh,
            ports=[self._spec()],
            f_max=8.0e9,
            verbose=True,
            **kwargs,
        )

    def test_certified_port_says_nothing(self):
        mesh = _rect_coax_mesh()
        op = _build(mesh, build_M_mu(mesh))
        notice = self._analysis(mesh)._mur_fallback_notice([op], 1e-13)
        assert notice is None

    def test_inhomogeneous_channel_is_named_as_a_model_not_a_defect(self):
        mesh = _rect_coax_mesh()
        op = _build(mesh, _tilt_transversal_mu(mesh, build_M_mu(mesh), 1.5))
        assert op.termination_kinds == ["mur"]
        notice = self._analysis(mesh)._mur_fallback_notice([op], 1e-13)
        assert "inhomogeneous cross-section" in notice
        assert "chain spread" in notice
        # The balance sheet: what is given up, and what is kept.
        assert "-30 dB" in notice
        assert "a()/b()" in notice

    def test_marginal_channel_points_back_at_the_port_warning(self):
        mesh = _rect_coax_mesh()
        with pytest.warns(UserWarning, match="uniform discrete chain"):
            op = _build(mesh, _tilt_transversal_mu(mesh, build_M_mu(mesh), 1.0 + 1e-4))
        notice = self._analysis(mesh)._mur_fallback_notice([op], 1e-13)
        assert "uniform-chain gate" in notice
        assert "inhomogeneous cross-section" not in notice

    def test_the_band_advice_is_only_offered_when_it_would_run(self):
        """The quoted axis start is the one ``_band_setup`` accepts.

        Both numbers come from the same inversion of the compactness
        sizing, so the notice cannot recommend an axis the band
        pipeline would then refuse.
        """
        mesh = _rect_coax_mesh()
        op = _build(mesh, _tilt_transversal_mu(mesh, build_M_mu(mesh), 1.5))
        dt = 1e-13
        f_rec = self._analysis(mesh)._band_min_axis_start(dt)

        low = self._analysis(mesh, f_min=0.0)
        assert low.f_axis[0] < f_rec
        notice = low._mur_fallback_notice([op], dt)
        assert "would also need a frequency axis" in notice
        with pytest.raises(ValueError, match="auto-sizing"):
            low._band_setup(low.f_axis, dt, None)

        high = self._analysis(mesh, f_min=2.0 * f_rec)
        notice = high._mur_fallback_notice([op], dt)
        assert "already clears" in notice
        high._band_setup(high.f_axis, dt, None)

    def test_base_class_points_at_the_class_that_owns_the_switch(self):
        """``AnalysisTD`` has no ``port_model``, so it must not suggest one."""
        from magnelio import AnalysisTD

        mesh = _rect_coax_mesh()
        op = _build(mesh, _tilt_transversal_mu(mesh, build_M_mu(mesh), 1.5))
        analysis = AnalysisTD(mesh=mesh, ports=[self._spec()], f_max=8.0e9, verbose=True)
        notice = analysis._mur_fallback_notice([op], 1e-13)
        assert "AnalysisScatteringTD(port_model=" in notice
        assert "frequency axis starting at" not in notice
