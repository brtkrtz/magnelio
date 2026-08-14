"""Regression test for rect-coax S-parameter quality (WP1, finding F6).

A 50 mm rectangular coaxial line (2 mm square inner PEC conductor,
10 mm square PEC shell via ``background=pec``, PTFE ε_r = 2.1) with one
TEM MultiConductor port per z-face must show a broadband match well
below −30 dB — the analysis pipeline used to sit at −21.5 dB because the
power-wave decomposition ignored the spatial half-cell stagger of the I
sampling plane (see ``modal_sparameters.py`` concern 1b; full WP1
narrative in the archived ``REWORK_PLAN.md``, git history).

The band edge is the sharp regression detector: without the de-stagger
the artefact is ``β·dz/4`` (−24 dB at 10 GHz for the 0.8 mm mesh used
here, −22 dB for any λ/20 mesh); with it the measured value is −46 dB.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio import AnalysisScatteringTD, Material, Mesh, MeshControl
from magnelio.geo import Brick, Difference, GeometryModel
from magnelio.mesh import BoxFace
from magnelio.ports import PortSpecMultiConductor

A_INNER = 2e-3
B_OUTER = 10e-3
EPS_R = 2.1
LENGTH = 50e-3
F_MAX = 10e9


def _build_analysis() -> AnalysisScatteringTD:
    ptfe = Material.from_isotropic("PTFE", epsilon=EPS_R)
    pec = Material.pec()

    model = GeometryModel(background=pec)
    ptfe_full = Brick(
        origin=(-B_OUTER / 2, -B_OUTER / 2, 0),
        size=(B_OUTER, B_OUTER, LENGTH),
        material=ptfe,
    )
    inner = Brick(
        origin=(-A_INNER / 2, -A_INNER / 2, 0),
        size=(A_INNER, A_INNER, LENGTH),
        material=pec,
    )
    model.add(Difference(ptfe_full, inner, material=ptfe, name="PTFE"))
    model.add(inner)

    control = MeshControl(min_nodes_per_wavelength=20, max_cell_size=0.8e-3)
    mesh = Mesh.from_geometry(model, control, f_max=F_MAX)

    specs = [
        PortSpecMultiConductor(
            name="port1",
            plane=BoxFace.Z_MIN,
            epsilon_r=EPS_R,
            n_modes=1,
        ),
        PortSpecMultiConductor(
            name="port2",
            plane=BoxFace.Z_MAX,
            epsilon_r=EPS_R,
            n_modes=1,
        ),
    ]
    return AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions(
            {
                "xmin": "PEC",
                "xmax": "PEC",
                "ymin": "PEC",
                "ymax": "PEC",
                "zmin": "PEC",
                "zmax": "PEC",
            }
        ),
        ports=specs,
        f_max=F_MAX,
        verbose=False,
    )


@pytest.fixture(scope="module")
def rect_coax_result():
    analysis = _build_analysis()
    f_axis = np.linspace(F_MAX / 40, F_MAX, 81)
    return analysis.run(f_axis=f_axis, excited=["port1"]), f_axis


def test_rect_coax_broadband_match(rect_coax_result):
    """|S11| stays below −130 dB across 0.25–10 GHz, |S21| flat at 0 dB.

    History on the 0.8 mm mesh (16×16×63): −34.6 dB max after the WP1
    de-stagger fix, −39.8 dB after the DD-052 travelling-wave
    profiles.  WP-R2 (exact DTBC termination + discrete de-stagger):
    max in-band |S11| = −159.3 dB.  The bound guards the −100 dB
    reflection-free acceptance criterion with ~30 dB margin.
    """
    result, f_axis = rect_coax_result

    S11 = result.S("port1", "port1")
    S21 = result.S("port2", "port1")
    s11_db = 20 * np.log10(np.abs(S11) + 1e-30)
    s21_db = 20 * np.log10(np.abs(S21) + 1e-30)

    assert np.all(np.isfinite(s11_db))
    assert s11_db.max() < -130.0, (
        f"broadband |S11| regression: max in band = {s11_db.max():.2f} dB "
        f"at f = {f_axis[np.argmax(s11_db)] / 1e9:.2f} GHz (bound: -130 dB; "
        f"reflection-free acceptance line is -100 dB)"
    )
    assert np.max(np.abs(s21_db)) < 0.1, (
        f"|S21| deviates from 0 dB by {np.max(np.abs(s21_db)):.3f} dB (bound: 0.1 dB)"
    )


def test_destaggered_time_domain_power_waves(rect_coax_result):
    """``result.b(destagger=True)`` removes the incident leak entirely.

    The co-located time-domain split leaks ``β·dz/4`` of the incident
    pulse into ``b1`` — a derivative-of-pulse ghost measured at
    −37.8 dB relative to the ``a1`` peak on this mesh, five orders of
    magnitude above the −159 dB port floor.  The frequency-domain
    de-stagger (default since the DD-063 analysis consolidation) must
    push the time-domain ``b1`` down to that floor: measured
    max|b1|/max|a1| = 1.07e-8 (−159.4 dB), gated at −140 dB.  The
    incident wave and the transmission must be preserved.
    """
    result, _ = rect_coax_result

    a1_raw = result.a("port1", destagger=False)
    b1_raw = result.b("port1", destagger=False)
    a1 = result.a("port1")
    b1 = result.b("port1")
    b2 = result.b("port2")

    pk_a = float(np.abs(a1_raw.values).max())
    leak_raw = float(np.abs(b1_raw.values).max()) / pk_a
    leak_ds = float(np.abs(b1.values).max()) / pk_a

    # The raw leak is the regression detector for the *test itself*:
    # if it disappears, the leak mechanism changed and the gate below
    # no longer measures what it claims to.
    assert 3e-3 < leak_raw < 5e-2, (
        f"raw co-located b1 leak {leak_raw:.3e} outside the expected "
        f"β·dz/4 class — test premise changed"
    )
    assert leak_ds < 1e-7, (
        f"de-staggered time-domain b1 at {leak_ds:.3e} of the a1 peak "
        f"({20 * np.log10(leak_ds):.1f} dB; bound −140 dB) — the "
        f"frequency-domain de-stagger no longer reaches the port floor"
    )

    # Incident wave essentially unchanged (the correction is a
    # per-bin phase/impedance fix, not a re-normalisation) ...
    assert np.abs(a1.values - a1_raw.values).max() / pk_a < 5e-2
    # ... and lossless-line energy conservation a1 -> b2 holds to
    # first order on the de-staggered pair.
    e_a1 = float(np.sum(a1.values**2))
    e_b2 = float(np.sum(b2.values**2))
    assert abs(e_a1 - e_b2) / e_a1 < 1e-4
