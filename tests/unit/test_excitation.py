"""Unit tests for the core Excitation (DD-224)."""

from __future__ import annotations

import pytest

import magnelio as mio
from magnelio import Excitation
from magnelio.signals import WaveformGaussian, WaveformGaussianModulated, WaveformSine


def test_core_export():
    assert mio.Excitation is Excitation
    assert "Excitation" in mio.__all__


def test_defaults():
    exc = Excitation("port1")
    assert (exc.source, exc.mode, exc.waveform) == ("port1", 0, None)
    assert (exc.amplitude, exc.delay, exc.phase) == (1.0, 0.0, 0.0)
    assert exc.effective_delay() == 0.0


def test_frozen():
    exc = Excitation("port1")
    with pytest.raises(AttributeError):
        exc.mode = 1  # type: ignore[misc]


def test_coerce_shorthands():
    assert Excitation.coerce("port1") == Excitation("port1")
    assert Excitation.coerce(("port1", 1)) == Excitation("port1", mode=1)
    assert Excitation.coerce(["port1", 2]) == Excitation("port1", mode=2)
    exc = Excitation("pw", amplitude=3.0)
    assert Excitation.coerce(exc) is exc
    with pytest.raises(TypeError, match="name, mode"):
        Excitation.coerce(3)
    with pytest.raises(TypeError, match="name, mode"):
        Excitation.coerce(("port1", 1, 2))


def test_validation():
    with pytest.raises(TypeError, match="name of a port or source"):
        Excitation("")
    with pytest.raises(TypeError, match="name of a port or source"):
        Excitation(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mode"):
        Excitation("p", mode=-1)
    with pytest.raises(ValueError, match="mode"):
        Excitation("p", mode=True)
    with pytest.raises(TypeError, match="Waveform"):
        Excitation("p", waveform="gaussian")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="amplitude"):
        Excitation("p", amplitude=float("inf"))
    with pytest.raises(ValueError, match="delay"):
        Excitation("p", delay=-1e-9)
    with pytest.raises(ValueError, match="phase"):
        Excitation("p", phase=float("nan"))


def test_phase_needs_a_carrier():
    with pytest.raises(ValueError, match="carrier"):
        Excitation("p", waveform=WaveformGaussian(f_max=10e9), phase=90.0)
    # No waveform yet: the check is deferred to the run's default waveform.
    exc = Excitation("p", phase=90.0)
    with pytest.raises(ValueError, match="carrier"):
        exc.effective_delay()


def test_phase_becomes_a_delay_on_carriers():
    w = WaveformGaussianModulated(f_min=8e9, f_max=12e9)  # f_center = 10 GHz
    exc = Excitation("p", waveform=w, phase=90.0, delay=1e-9)
    assert exc.effective_delay() == pytest.approx(1e-9 + 0.25 / 10e9)
    cw = Excitation("p", waveform=WaveformSine(f=2e9), phase=-180.0)
    assert cw.effective_delay() == pytest.approx(-0.5 / 2e9)


def test_numeric_fields_are_floats():
    exc = Excitation("p", amplitude=2, delay=0, phase=0)
    assert isinstance(exc.amplitude, float)
    assert isinstance(exc.delay, float)
    assert isinstance(exc.phase, float)
