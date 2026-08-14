"""Tests for ScatteringTDResult.a / .b time-domain power waves (WP5.2).

``a(t) = (V/√Z + √Z·I)/2`` and ``b(t) = (V/√Z − √Z·I)/2`` with the
recorder's temporal Yee half-step corrected by midpoint-averaging the
I samples.  This replaces the ``V − s(t)`` notebook pattern, which is
wrong by design (V is the total modal voltage; s(t) is the source
waveform, not the launched wave).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from magnelio import (
    AnalysisScatteringTD,
    Mesh,
)
from magnelio.analysis.scattering_td import (
    ScatteringTDResult,
    _LumpedModeStub,
)
from magnelio.mesh import BoxFace
from magnelio.mesh.grid import GridLines
from magnelio.ports import PortSpecMultiConductor
from magnelio.post import SParameterResult
from magnelio.signals import Signal1D


def _synthetic_result(Z0: float = 50.0) -> tuple[ScatteringTDResult, float]:
    """Matched forward wave into a LUMPED port (co-temporal V/I).

    A discrete port samples I co-temporally with V (DD-075 F3): both live
    at ``t^{n+1}``, so a matched outgoing wave is ``V[n] = s((n+1)·dt)``,
    ``I[n] = V[n]/Z0``.  The ``_LumpedModeStub`` reports ``i_cotemporal``,
    so the a/b split skips the Yee temporal half-step and ``b(t) ≡ 0``
    *exactly* (``V = Z0·I`` ⇒ ``V/√Z − √Z·I = 0``), not merely to the
    O((ω·dt)²) midpoint-average accuracy of the staggered modal case.
    """
    dt = 1e-12
    n = 400
    f0 = 5e9

    def s(t):
        env = np.exp(-(((t - 100e-12) / 30e-12) ** 2))
        return env * np.cos(2.0 * math.pi * f0 * (t - 100e-12))

    t_axis = np.arange(n) * dt
    V = s(t_axis + dt)
    I = s(t_axis + dt) / Z0  # co-temporal with V (lumped port)

    sigs = {
        ("p1", 0): (
            Signal1D(t=t_axis, values=V, dt=dt, label="V"),
            Signal1D(t=t_axis, values=I, dt=dt, label="I"),
        ),
    }
    f_axis = np.linspace(1e9, 10e9, 5)
    s_params = SParameterResult.from_single_excitation(
        {("p1", 0): np.zeros(5, dtype=complex)},
        ("p1", 0),
        f_axis,
    )
    result = ScatteringTDResult(
        s_params=s_params,
        signals={("p1", 0): sigs},
        reference_signal=Signal1D(
            t=t_axis,
            values=s(t_axis),
            dt=dt,
            label="ref",
        ),
        dt=dt,
        n_actual_steps=n,
        port_modes={"p1": [_LumpedModeStub(z0=Z0)]},
    )
    return result, Z0


def test_matched_forward_wave_has_vanishing_b():
    result, Z0 = _synthetic_result()
    a = result.a("p1")
    b = result.b("p1")

    # a carries the whole wave: peak = V_peak/sqrt(Z) (exact for the
    # co-temporal lumped split — no temporal correction to blur it).
    v_peak = np.abs(result.signals[("p1", 0)][("p1", 0)][0].values).max()
    np.testing.assert_allclose(
        np.abs(a.values).max(),
        v_peak / math.sqrt(Z0),
        rtol=1e-9,
    )
    # b vanishes exactly: co-temporal lumped I (DD-075 F3) means no Yee
    # temporal rotation, and V = Z0·I for a matched wave ⇒ b ≡ 0.
    assert np.abs(b.values).max() < 1e-12 * np.abs(a.values).max()
    # Same time axis and metadata as the recorded V.
    assert a.dt == result.dt and len(a.values) == result.n_actual_steps
    assert a.label == "a(p1,0)" and b.label == "b(p1,0)"


def test_power_wave_errors():
    result, _ = _synthetic_result()
    with pytest.raises(KeyError, match="channel"):
        result.a("p2")
    with pytest.raises(KeyError, match="channel"):
        result.a("p1", 3)  # unrecorded mode index is a missing channel
    with pytest.raises(KeyError, match="excitation"):
        result.a("p1", excited="p2")

    bare = ScatteringTDResult(
        s_params=result.s_params,
        signals=result.signals,
        reference_signal=result.reference_signal,
        dt=result.dt,
        n_actual_steps=result.n_actual_steps,
    )
    with pytest.raises(ValueError, match="port_modes"):
        bare.a("p1")


def _parallel_plate_analysis() -> AnalysisScatteringTD:
    grid = GridLines(
        x=np.linspace(-5e-3, 5e-3, 11),
        y=np.linspace(-2.5e-3, 2.5e-3, 6),
        z=np.linspace(-10e-3, 10e-3, 41),
    )
    specs = [
        PortSpecMultiConductor(name="port1", plane=BoxFace.Z_MIN, n_modes=1),
        PortSpecMultiConductor(name="port2", plane=BoxFace.Z_MAX, n_modes=1),
    ]
    return AnalysisScatteringTD(
        mesh=Mesh.from_grid(
            grid,
            boundary_conditions={
                "xmin": "PMC",
                "xmax": "PMC",
                "ymin": "PEC",
                "ymax": "PEC",
                "zmin": "PEC",
                "zmax": "PEC",
            },
        ),
        ports=specs,
        f_max=10e9,
        verbose=False,
    )


def test_parallel_plate_power_wave_split():
    """Matched TEM line: a₁ launches, b₂ receives, b₁ stays small."""
    analysis = _parallel_plate_analysis()
    result = analysis.run(excited=["port1"])

    a1 = result.a("port1")
    b1 = result.b("port1")
    b2 = result.b("port2")

    a1_peak = np.abs(a1.values).max()
    assert a1_peak > 0.0
    # Transmission: the outgoing wave at port 2 carries the launched
    # amplitude (|S21| ≈ 0 dB on this line).
    np.testing.assert_allclose(
        np.abs(b2.values).max(),
        a1_peak,
        rtol=0.05,
    )
    # Reflection stays far below the launch (measured |S11| < −60 dB in
    # frequency domain; the TD view keeps the spatial-stagger leak, so
    # only a loose 10 % bound is asserted).
    assert np.abs(b1.values).max() < 0.1 * a1_peak
    # b2 arrives after a1 peaks (one line length of travel).
    t_a1 = a1.t[np.argmax(np.abs(a1.values))]
    t_b2 = b2.t[np.argmax(np.abs(b2.values))]
    assert t_b2 > t_a1


def test_multi_excitation_requires_selector():
    analysis = _parallel_plate_analysis()
    result = analysis.run(excited=["port1", "port2"])

    with pytest.raises(ValueError, match="pass excited="):
        result.a("port1")

    a1 = result.a("port1", excited="port1")
    a2 = result.a("port2", excited=("port2", 0))
    np.testing.assert_allclose(
        np.abs(a1.values).max(),
        np.abs(a2.values).max(),
        rtol=0.02,
    )
