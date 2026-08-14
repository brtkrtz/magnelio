"""Tests for magnelio.post.sparameter_result.SParameterResult.

Phase 2c steps 11 + 12 of the modal-port architecture
(`reference_architecture_phase2_mode_solver.md` §5).

Six layers of validation:

1. **Construction invariants.**  Shape mismatches, empty inputs,
   duplicate channels / excitations, excitations not in channels are
   all rejected at ``__post_init__``.
2. **Single-excitation wrapping.**  ``from_single_excitation`` wraps
   a :func:`compute_s_parameters` dict; the resulting matrix is
   ``(Nf, n_channels, 1)`` and the channel-order argument lets the
   caller fix a canonical order.
3. **Multi-excitation aggregation.**  ``from_multiple_excitations``
   and ``merge`` produce identical N×N matrices from K single-
   excitation runs.  Mismatched f_axis or channel orderings are
   rejected.
4. **Named-port accessors.**  ``S(out, in)`` and ``db(out, in)`` look
   up by string name; mode_in / mode_out are optional and default to
   0.  ``KeyError`` for unknown ports / unrequested excitations.
5. **dB convention.**  ``floor_db`` floors the magnitude before
   ``20·log10`` to avoid ``-inf``; NaN inputs (from
   ``compute_s_parameters`` under-threshold guard) flow through to the
   floor too.
6. **Multi-mode per port.**  Two mode indices on the same port name
   are addressed by ``mode_out`` / ``mode_in``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from magnelio.ports._modal.mode import Mode, ModeType
from magnelio.post import SParameterResult, compute_s_parameters
from magnelio.signals.signal_1d import Signal1D

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _stub_field(u, v):
    z = np.zeros_like(u)
    return z, z, z, z


def _tem_mode(z_line: float = 50.0) -> Mode:
    return Mode(
        name="TEM",
        mode_type=ModeType.TEM,
        omega_c=0.0,
        epsilon_r=1.0,
        field_evaluator=_stub_field,
        z_line=z_line,
    )


def _two_port_dict(f: np.ndarray, s11: complex, s21: complex):
    """Build a 2-port S-parameter dict (uniform spectra)."""
    return {
        ("port1", 0): np.full(f.size, s11, dtype=complex),
        ("port2", 0): np.full(f.size, s21, dtype=complex),
    }


# ---------------------------------------------------------------------
# 1) Construction invariants
# ---------------------------------------------------------------------


class TestConstructionInvariants:
    """Direct ``SParameterResult(...)`` construction validation."""

    def test_empty_f_axis_rejected(self):
        with pytest.raises(ValueError, match="f_axis must be 1D"):
            SParameterResult(
                f_axis=np.array([]),
                channels=(("p", 0),),
                excitations=(("p", 0),),
                matrix=np.zeros((0, 1, 1), dtype=complex),
            )

    def test_negative_frequency_rejected(self):
        with pytest.raises(ValueError, match="positive frequencies"):
            SParameterResult(
                f_axis=np.array([1e9, -2e9]),
                channels=(("p", 0),),
                excitations=(("p", 0),),
                matrix=np.zeros((2, 1, 1), dtype=complex),
            )

    def test_empty_channels_rejected(self):
        with pytest.raises(ValueError, match="channels must be non-empty"):
            SParameterResult(
                f_axis=np.array([1e9]),
                channels=(),
                excitations=(("p", 0),),
                matrix=np.zeros((1, 0, 1), dtype=complex),
            )

    def test_duplicate_channels_rejected(self):
        with pytest.raises(ValueError, match="channels must be unique"):
            SParameterResult(
                f_axis=np.array([1e9]),
                channels=(("p", 0), ("p", 0)),
                excitations=(("p", 0),),
                matrix=np.zeros((1, 2, 1), dtype=complex),
            )

    def test_excitation_not_in_channels_rejected(self):
        with pytest.raises(ValueError, match="not in channels"):
            SParameterResult(
                f_axis=np.array([1e9]),
                channels=(("p1", 0),),
                excitations=(("p2", 0),),
                matrix=np.zeros((1, 1, 1), dtype=complex),
            )

    def test_matrix_shape_mismatch_rejected(self):
        with pytest.raises(ValueError, match="matrix shape"):
            SParameterResult(
                f_axis=np.array([1e9, 2e9]),
                channels=(("p", 0),),
                excitations=(("p", 0),),
                matrix=np.zeros((1, 1, 1), dtype=complex),  # wrong Nf
            )

    def test_valid_construction(self):
        f = np.array([1e9, 2e9, 3e9])
        m = np.zeros((3, 2, 1), dtype=complex)
        m[:, 0, 0] = 0.1
        m[:, 1, 0] = 0.95
        r = SParameterResult(
            f_axis=f,
            channels=(("a", 0), ("b", 0)),
            excitations=(("a", 0),),
            matrix=m,
        )
        assert r.n_frequencies == 3
        assert r.n_channels == 2
        assert r.n_excitations == 1


# ---------------------------------------------------------------------
# 2) from_single_excitation
# ---------------------------------------------------------------------


class TestFromSingleExcitation:
    @pytest.fixture
    def setup(self):
        f = np.linspace(1e9, 5e9, 4)
        s_dict = _two_port_dict(f, s11=0.05 + 0j, s21=0.95 + 0j)
        return f, s_dict

    def test_default_channel_order_follows_dict_keys(self, setup):
        f, s_dict = setup
        r = SParameterResult.from_single_excitation(
            s_dict,
            ("port1", 0),
            f,
        )
        assert r.channels == (("port1", 0), ("port2", 0))
        assert r.excitations == (("port1", 0),)
        assert r.matrix.shape == (4, 2, 1)

    def test_explicit_channel_order(self, setup):
        f, s_dict = setup
        r = SParameterResult.from_single_excitation(
            s_dict,
            ("port1", 0),
            f,
            channel_order=(("port2", 0), ("port1", 0)),
        )
        assert r.channels == (("port2", 0), ("port1", 0))
        # port2 is now row 0 → S(port2, port1) reads matrix[:, 0, 0].
        np.testing.assert_array_equal(
            r.S("port2", "port1"),
            s_dict[("port2", 0)],
        )

    def test_empty_dict_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            SParameterResult.from_single_excitation(
                {},
                ("p", 0),
                np.array([1e9]),
            )

    def test_excited_not_in_dict_rejected(self, setup):
        f, s_dict = setup
        with pytest.raises(KeyError, match="not in s_dict"):
            SParameterResult.from_single_excitation(
                s_dict,
                ("portX", 0),
                f,
            )

    def test_spectrum_shape_mismatch_rejected(self, setup):
        f, _ = setup
        bad = {("port1", 0): np.zeros(7, dtype=complex)}  # wrong Nf
        with pytest.raises(ValueError, match="expected"):
            SParameterResult.from_single_excitation(
                bad,
                ("port1", 0),
                f,
            )

    def test_channel_order_must_cover_all_keys(self, setup):
        f, s_dict = setup
        with pytest.raises(ValueError, match="enumerate every key"):
            SParameterResult.from_single_excitation(
                s_dict,
                ("port1", 0),
                f,
                channel_order=(("port1", 0),),  # missing port2
            )


# ---------------------------------------------------------------------
# 3) Multi-excitation aggregation
# ---------------------------------------------------------------------


class TestMultiExcitationAggregation:
    @pytest.fixture
    def two_port_runs(self):
        f = np.linspace(1e9, 5e9, 4)
        # excite port1 → S11, S21
        run_p1 = _two_port_dict(f, s11=0.05 + 0j, s21=0.95 + 0j)
        # excite port2 → S12, S22
        run_p2 = _two_port_dict(f, s11=0.92 + 0j, s21=0.04 + 0j)
        # In run_p2 the keys are still (port1, 0) and (port2, 0); 'port1'
        # spectrum *now* represents S12 (effect on port1 from exciting
        # port2), and 'port2' represents S22.
        return f, [
            (("port1", 0), run_p1),
            (("port2", 0), run_p2),
        ]

    def test_from_multiple_yields_full_matrix(self, two_port_runs):
        f, runs = two_port_runs
        r = SParameterResult.from_multiple_excitations(runs, f)
        assert r.matrix.shape == (4, 2, 2)
        assert r.is_complete is True
        assert r.excitations == (("port1", 0), ("port2", 0))

    def test_S11_S12_S21_S22(self, two_port_runs):
        f, runs = two_port_runs
        r = SParameterResult.from_multiple_excitations(runs, f)
        np.testing.assert_array_equal(
            r.S("port1", "port1"),
            np.full(4, 0.05, dtype=complex),
        )
        np.testing.assert_array_equal(
            r.S("port2", "port1"),
            np.full(4, 0.95, dtype=complex),
        )
        np.testing.assert_array_equal(
            r.S("port1", "port2"),
            np.full(4, 0.92, dtype=complex),
        )
        np.testing.assert_array_equal(
            r.S("port2", "port2"),
            np.full(4, 0.04, dtype=complex),
        )

    def test_merge_equivalent_to_from_multiple(self, two_port_runs):
        f, runs = two_port_runs
        via_multi = SParameterResult.from_multiple_excitations(runs, f)
        via_merge = SParameterResult.merge(
            [SParameterResult.from_single_excitation(d, exc, f) for exc, d in runs]
        )
        assert via_multi.channels == via_merge.channels
        assert via_multi.excitations == via_merge.excitations
        np.testing.assert_array_equal(via_multi.matrix, via_merge.matrix)

    def test_mismatched_channel_sets_rejected(self, two_port_runs):
        f, runs = two_port_runs
        # Mutate run 2 to drop a channel.
        runs[1] = (
            runs[1][0],
            {k: v for k, v in runs[1][1].items() if k != ("port2", 0)},
        )
        with pytest.raises(ValueError, match="must observe the same channels"):
            SParameterResult.from_multiple_excitations(runs, f)

    def test_duplicate_excitation_rejected(self):
        f = np.linspace(1e9, 5e9, 4)
        d = _two_port_dict(f, s11=0.05 + 0j, s21=0.95 + 0j)
        runs = [(("port1", 0), d), (("port1", 0), d)]
        with pytest.raises(ValueError, match="duplicate excitations"):
            SParameterResult.from_multiple_excitations(runs, f)

    def test_merge_mismatched_f_axis_rejected(self, two_port_runs):
        f, runs = two_port_runs
        r1 = SParameterResult.from_single_excitation(runs[0][1], runs[0][0], f)
        f_other = np.linspace(2e9, 6e9, 4)
        r2 = SParameterResult.from_single_excitation(runs[1][1], runs[1][0], f_other)
        with pytest.raises(ValueError, match="f_axis differs"):
            SParameterResult.merge([r1, r2])

    def test_merge_mismatched_channels_rejected(self, two_port_runs):
        f, runs = two_port_runs
        r1 = SParameterResult.from_single_excitation(runs[0][1], runs[0][0], f)
        r2 = SParameterResult.from_single_excitation(
            runs[1][1],
            runs[1][0],
            f,
            channel_order=(("port2", 0), ("port1", 0)),  # reversed
        )
        with pytest.raises(ValueError, match="channels"):
            SParameterResult.merge([r1, r2])

    def test_merge_empty_list_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            SParameterResult.merge([])


# ---------------------------------------------------------------------
# 4) Named-port accessors
# ---------------------------------------------------------------------


class TestAccessors:
    @pytest.fixture
    def two_port(self):
        f = np.linspace(1e9, 5e9, 4)
        s_dict = _two_port_dict(f, s11=0.05 + 0j, s21=0.95 + 0j)
        return SParameterResult.from_single_excitation(
            s_dict,
            ("port1", 0),
            f,
        )

    def test_S_returns_correct_spectrum(self, two_port):
        r = two_port
        np.testing.assert_array_equal(
            r.S("port2", "port1"),
            np.full(4, 0.95, dtype=complex),
        )

    def test_S_unknown_observed_raises(self, two_port):
        r = two_port
        with pytest.raises(KeyError, match="not in result"):
            r.S("portX", "port1")

    def test_S_unrequested_excitation_raises(self, two_port):
        """``port2`` is observed but was *not* excited in this single-
        excitation result, so ``S('?, 'port2')`` raises."""
        r = two_port
        with pytest.raises(KeyError, match="only carries excitations"):
            r.S("port1", "port2")

    def test_S_returns_independent_copy(self, two_port):
        """Mutating the returned array must not bleed back into matrix."""
        r = two_port
        s = r.S("port1", "port1")
        s[0] = 999 + 999j
        # Re-read; matrix unchanged.
        s2 = r.S("port1", "port1")
        assert s2[0] == complex(0.05)


# ---------------------------------------------------------------------
# 5) dB conversion + floor + NaN handling
# ---------------------------------------------------------------------


class TestDb:
    def test_db_basic(self):
        f = np.array([1e9, 2e9])
        d = {("p1", 0): np.array([0.1 + 0j, 1.0 + 0j])}
        r = SParameterResult.from_single_excitation(d, ("p1", 0), f)
        np.testing.assert_allclose(
            r.db("p1", "p1"),
            np.array([-20.0, 0.0]),
        )

    def test_db_floors_below_threshold(self):
        f = np.array([1e9, 2e9])
        # 1e-30 is well below the default -200 dB floor (1e-10).
        d = {("p1", 0): np.array([1e-30 + 0j, 1.0 + 0j])}
        r = SParameterResult.from_single_excitation(d, ("p1", 0), f)
        out = r.db("p1", "p1")
        assert out[0] == -200.0  # floored
        assert out[1] == 0.0

    def test_db_handles_nan(self):
        """NaN entries (from compute_s_parameters under-threshold) flow
        through to the floor rather than producing more NaNs."""
        f = np.array([1e9, 2e9])
        d = {("p1", 0): np.array([np.nan + 1j * np.nan, 0.5 + 0j])}
        r = SParameterResult.from_single_excitation(d, ("p1", 0), f)
        out = r.db("p1", "p1")
        assert out[0] == -200.0
        np.testing.assert_allclose(out[1], 20 * math.log10(0.5))

    def test_db_custom_floor(self):
        f = np.array([1e9])
        d = {("p1", 0): np.array([1e-10 + 0j])}
        r = SParameterResult.from_single_excitation(d, ("p1", 0), f)
        np.testing.assert_allclose(r.db("p1", "p1", floor_db=-150.0), -150.0)


# ---------------------------------------------------------------------
# 6) Multi-mode per port
# ---------------------------------------------------------------------


class TestMultiModePerPort:
    def test_two_modes_one_port(self):
        f = np.linspace(1e9, 5e9, 3)
        d = {
            ("p", 0): np.full(3, 0.1, dtype=complex),
            ("p", 1): np.full(3, 0.5, dtype=complex),
        }
        r = SParameterResult.from_single_excitation(d, ("p", 0), f)
        np.testing.assert_array_equal(
            r.S("p", "p", mode_out=0, mode_in=0),
            np.full(3, 0.1, dtype=complex),
        )
        np.testing.assert_array_equal(
            r.S("p", "p", mode_out=1, mode_in=0),
            np.full(3, 0.5, dtype=complex),
        )


# ---------------------------------------------------------------------
# 7) Properties
# ---------------------------------------------------------------------


class TestProperties:
    def test_port_names_preserves_first_occurrence_order(self):
        f = np.array([1e9])
        d = {
            ("portB", 0): np.array([0.1 + 0j]),
            ("portA", 0): np.array([0.2 + 0j]),
        }
        # channel_order = dict iteration → ("portB", "portA")
        r = SParameterResult.from_single_excitation(d, ("portB", 0), f)
        assert r.port_names == ("portB", "portA")

    def test_is_complete_single_excitation_two_channels(self):
        f = np.array([1e9])
        d = _two_port_dict(f, s11=0.05 + 0j, s21=0.95 + 0j)
        r = SParameterResult.from_single_excitation(d, ("port1", 0), f)
        assert r.is_complete is False

    def test_is_complete_full_2x2(self):
        f = np.array([1e9])
        d1 = _two_port_dict(f, s11=0.05 + 0j, s21=0.95 + 0j)
        d2 = _two_port_dict(f, s11=0.94 + 0j, s21=0.06 + 0j)
        r = SParameterResult.from_multiple_excitations(
            [(("port1", 0), d1), (("port2", 0), d2)],
            f,
        )
        assert r.is_complete is True


# ---------------------------------------------------------------------
# 8) Round-trip with compute_s_parameters
# ---------------------------------------------------------------------


class TestRoundTripWithComputeSParameters:
    """End-to-end: ``compute_s_parameters`` → wrap → access pipeline."""

    def test_open_circuit_S11_equals_one(self):
        N, dt = 256, 5e-12
        t = np.arange(N) * dt
        V_vals = np.exp(-((t - 50 * dt) ** 2) / (2.0 * (20 * dt) ** 2))
        V = Signal1D(t=t, values=V_vals, dt=dt, label="V")
        I = Signal1D(t=t, values=np.zeros(N), dt=dt, label="I")
        recorder = {("p", 0): (V, I)}
        port_modes = {"p": [_tem_mode()]}
        f = np.linspace(1e9, 20e9, 21)

        s_dict = compute_s_parameters(
            recorder,
            port_modes,
            ("p", 0),
            reference_signal=V,
            f_axis=f,
        )
        result = SParameterResult.from_single_excitation(
            s_dict,
            ("p", 0),
            f,
        )
        S = result.S("p", "p")
        np.testing.assert_allclose(S.real, 1.0, atol=1e-9)
        np.testing.assert_allclose(S.imag, 0.0, atol=1e-9)
        # dB equivalent of S = 1 is 0 dB (with the default floor).
        np.testing.assert_allclose(
            result.db("p", "p"),
            np.zeros_like(f),
            atol=1e-9,
        )
