"""Reference impedances, renormalisation and export references (DD-244)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from magnelio.post.sparameter_result import SParameterResult

F = np.array([1e9, 2e9, 3e9, 4e9])
CH = (("p1", 0), ("p2", 0))


def _lossless_two_port(seed: int = 0) -> np.ndarray:
    """A random lossless reciprocal two-port per frequency (S unitary, symmetric)."""
    rng = np.random.default_rng(seed)
    out = np.empty((F.size, 2, 2), dtype=complex)
    for k in range(F.size):
        th, ph1, ph2 = rng.uniform(0, 2 * np.pi, 3)
        r, t = np.cos(th), np.sin(th)
        # Symmetric unitary: [[r e^{iφ1}, i t e^{i(φ1+φ2)/2}], [i t ..., r e^{iφ2}]]
        out[k] = np.array(
            [
                [r * np.exp(1j * ph1), 1j * t * np.exp(0.5j * (ph1 + ph2))],
                [1j * t * np.exp(0.5j * (ph1 + ph2)), r * np.exp(1j * ph2)],
            ]
        )
    return out


def _result(matrix=None, refs=None) -> SParameterResult:
    if matrix is None:
        matrix = _lossless_two_port()
    if refs is None:
        refs = {CH[0]: np.full(F.size, 50.0), CH[1]: np.full(F.size, 50.0)}
    return SParameterResult(
        f_axis=F, channels=CH, excitations=CH, matrix=matrix, reference_impedances=refs
    )


def _is_unitary(m: np.ndarray) -> bool:
    return all(np.allclose(np.conj(mk.T) @ mk, np.eye(mk.shape[0]), atol=1e-12) for mk in m)


class TestReferenceImpedances:
    def test_accessor_and_validation(self):
        res = _result()
        assert np.allclose(res.reference_impedance("p1"), 50.0)
        with pytest.raises(KeyError):
            res.reference_impedance("p9")
        bare = SParameterResult(f_axis=F, channels=CH, excitations=CH, matrix=_lossless_two_port())
        with pytest.raises(ValueError, match="no reference impedances"):
            bare.reference_impedance("p1")
        with pytest.raises(ValueError, match="lacks channels"):
            SParameterResult(
                f_axis=F,
                channels=CH,
                excitations=CH,
                matrix=_lossless_two_port(),
                reference_impedances={CH[0]: np.full(F.size, 50.0)},
            )
        with pytest.raises(ValueError, match="shape"):
            SParameterResult(
                f_axis=F,
                channels=CH,
                excitations=CH,
                matrix=_lossless_two_port(),
                reference_impedances={CH[0]: np.ones(2), CH[1]: np.ones(2)},
            )

    def test_merge_carries_and_checks_them(self):
        full = _result()
        cols = [
            SParameterResult(
                f_axis=F,
                channels=CH,
                excitations=(c,),
                matrix=full.matrix[:, :, [j]],
                reference_impedances=full.reference_impedances,
            )
            for j, c in enumerate(CH)
        ]
        merged = SParameterResult.merge(cols)
        assert merged.reference_impedances is not None
        assert np.allclose(merged.matrix, full.matrix)
        other = SParameterResult(
            f_axis=F,
            channels=CH,
            excitations=(CH[1],),
            matrix=full.matrix[:, :, [1]],
            reference_impedances={CH[0]: np.full(F.size, 60.0), CH[1]: np.full(F.size, 50.0)},
        )
        with pytest.raises(ValueError, match="different"):
            SParameterResult.merge([cols[0], other])


class TestRenormalize:
    def test_identity(self):
        res = _result()
        same = res.renormalize(50.0)
        assert np.allclose(same.matrix, res.matrix)
        assert np.allclose(same.reference_impedance("p2"), 50.0)

    def test_matched_load_seen_from_another_reference(self):
        # A one-port matched to 50 Ω reads Γ = (50 − 100)/(50 + 100) from 100 Ω.
        one = SParameterResult(
            f_axis=F,
            channels=(CH[0],),
            excitations=(CH[0],),
            matrix=np.zeros((F.size, 1, 1), dtype=complex),
            reference_impedances={CH[0]: np.full(F.size, 50.0)},
        )
        r = one.renormalize(100.0)
        assert np.allclose(r.matrix[:, 0, 0], (50.0 - 100.0) / (50.0 + 100.0))

    def test_round_trip_and_losslessness(self):
        res = _result()
        moved = res.renormalize({"p1": 75.0, "p2": np.linspace(30.0, 40.0, F.size)})
        assert _is_unitary(moved.matrix)
        assert np.allclose(moved.matrix, moved.matrix.transpose(0, 2, 1))
        back = moved.renormalize(50.0)
        assert np.allclose(back.matrix, res.matrix, atol=1e-12)
        assert np.allclose(moved.reference_impedance("p1"), 75.0)
        assert np.allclose(moved.reference_impedance("p2")[-1], 40.0)

    def test_tuple_keys_and_untouched_channels(self):
        res = _result()
        moved = res.renormalize({("p1", 0): 25.0})
        assert np.allclose(moved.reference_impedance("p1"), 25.0)
        assert np.allclose(moved.reference_impedance("p2"), 50.0)
        assert not np.allclose(moved.matrix[:, 1, 1], res.matrix[:, 1, 1])

    def test_agrees_with_the_closed_form_for_a_line(self):
        # A lossless line of impedance Z between two Z-referenced ports
        # is matched (S11 = 0, |S21| = 1); against Z' it reads the
        # textbook Γ = (Z − Z')/(Z + Z') of the line's input when the
        # far end is terminated in Z' — for a quarter-wave line
        # Z_in = Z²/Z', so Γ = (Z² − Z'²)/(Z² + Z'²).
        Z, Zp = 50.0, 100.0
        m = np.zeros((F.size, 2, 2), dtype=complex)
        m[:, 0, 1] = m[:, 1, 0] = np.exp(-1j * np.pi / 2)  # quarter wave
        res = _result(matrix=m, refs={CH[0]: np.full(F.size, Z), CH[1]: np.full(F.size, Z)})
        r = res.renormalize(Zp)
        expected = (Z**2 - Zp**2) / (Z**2 + Zp**2)
        assert np.allclose(r.matrix[:, 0, 0], expected)

    def test_rejections(self):
        res = _result()
        partial = SParameterResult(
            f_axis=F,
            channels=CH,
            excitations=(CH[0],),
            matrix=res.matrix[:, :, [0]],
            reference_impedances=res.reference_impedances,
        )
        # An incomplete result renormalises its excited sub-network.
        sub = partial.renormalize(75.0)
        assert sub.channels == (CH[0],) and sub.excitations == (CH[0],)
        assert np.allclose(sub.reference_impedance("p1"), 75.0)
        bare = SParameterResult(f_axis=F, channels=CH, excitations=CH, matrix=res.matrix)
        with pytest.raises(ValueError, match="no reference"):
            bare.renormalize(50.0)
        with pytest.raises(KeyError):
            res.renormalize({"p9": 50.0})
        with pytest.raises(ValueError, match="positive"):
            res.renormalize(-5.0)


class TestExports:
    def test_touchstone_states_the_common_reference(self, tmp_path):
        res = _result(refs={CH[0]: np.full(F.size, 49.3), CH[1]: np.full(F.size, 49.3)})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res.to_touchstone(tmp_path / "a.s2p")
        text = (tmp_path / "a.s2p").read_text()
        assert "# Hz S RI R 49.3\n" in text
        assert "! port 1 reference impedance: 49.3 Ohm (constant)" in text

    def test_touchstone_renormalises_on_request(self, tmp_path):
        res = _result(refs={CH[0]: np.full(F.size, 49.3), CH[1]: np.full(F.size, 49.3)})
        res.to_touchstone(tmp_path / "b.s2p", z_ref=50)
        text = (tmp_path / "b.s2p").read_text()
        assert "# Hz S RI R 50\n" in text
        body = [ln for ln in text.splitlines() if ln and not ln.startswith(("!", "#"))]
        s11 = complex(float(body[0].split()[1]), float(body[0].split()[2]))
        assert s11 == pytest.approx(res.renormalize(50.0).matrix[0, 0, 0])

    def test_touchstone_warns_on_a_varying_reference(self, tmp_path):
        res = _result(refs={CH[0]: np.linspace(300.0, 400.0, F.size), CH[1]: np.full(F.size, 50.0)})
        with pytest.warns(UserWarning, match="vary with frequency"):
            res.to_touchstone(tmp_path / "c.s2p")
        text = (tmp_path / "c.s2p").read_text()
        assert "# Hz S RI R 50\n" in text
        assert "frequency dependent" in text
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res.to_touchstone(tmp_path / "d.s2p", z_ref=50)

    def test_touchstone_without_references_keeps_the_nominal_line(self, tmp_path):
        bare = SParameterResult(f_axis=F, channels=CH, excitations=CH, matrix=_lossless_two_port())
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            bare.to_touchstone(tmp_path / "e.s2p")
        text = (tmp_path / "e.s2p").read_text()
        assert "# Hz S RI R 50\n" in text
        assert "not recorded" in text

    def test_skrf_carries_z0(self):
        skrf = pytest.importorskip("skrf")
        res = _result(refs={CH[0]: np.full(F.size, 49.3), CH[1]: np.linspace(30.0, 40.0, F.size)})
        ntw = res.to_skrf()
        assert isinstance(ntw, skrf.Network)
        assert np.allclose(ntw.z0[:, 0], 49.3)
        assert np.allclose(ntw.z0[:, 1], np.linspace(30.0, 40.0, F.size))
        ntw50 = res.to_skrf(z_ref=50)
        assert np.allclose(ntw50.z0, 50.0)
        assert np.allclose(ntw50.s, res.renormalize(50.0).matrix)
