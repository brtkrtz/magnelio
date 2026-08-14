"""Unit gates for the pole-residue dispersion model layer (DD-083).

Every constructor is checked against its closed-form permittivity to
machine precision; the mandatory passivity check is exercised on each
rejection branch; the Material integration and the store serialisation
round-trip are gated.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio import Material
from magnelio.io.project import _material_from_dict, _material_to_dict
from magnelio.materials import DispersionModel

W_TEST = 2.0 * np.pi * np.logspace(8, 11, 200)


class TestConstructorsClosedForm:
    def test_debye(self):
        m = DispersionModel.debye(1.8, 79.2, 9.4e-12)
        ref = 1.8 + 79.2 / (1.0 + 1j * W_TEST * 9.4e-12)
        np.testing.assert_allclose(m.evaluate(W_TEST), ref, rtol=1e-12)

    def test_debye_multi_term(self):
        m = DispersionModel.debye(2.0, [1.0, 0.5], [1e-11, 1e-9])
        ref = 2.0 + 1.0 / (1 + 1j * W_TEST * 1e-11) + 0.5 / (1 + 1j * W_TEST * 1e-9)
        np.testing.assert_allclose(m.evaluate(W_TEST), ref, rtol=1e-12)

    def test_lorentz_underdamped(self):
        w0, dl = 2 * np.pi * 5e9, 2 * np.pi * 0.25e9
        m = DispersionModel.lorentz(2.0, 0.5, w0, dl)
        ref = 2.0 + 0.5 * w0**2 / (w0**2 + 2j * W_TEST * dl - W_TEST**2)
        np.testing.assert_allclose(m.evaluate(W_TEST), ref, rtol=1e-12)
        assert len(m.poles) == 1 and m.poles[0][0].imag > 0

    def test_lorentz_overdamped(self):
        w0 = 2 * np.pi * 5e9
        m = DispersionModel.lorentz(2.0, 0.5, w0, 3 * w0)
        ref = 2.0 + 0.5 * w0**2 / (w0**2 + 6j * W_TEST * w0 - W_TEST**2)
        np.testing.assert_allclose(m.evaluate(W_TEST), ref, rtol=1e-9)
        assert len(m.poles) == 2

    def test_drude(self):
        wp, g = 2 * np.pi * 5e9, 2 * np.pi * 0.1e9
        m = DispersionModel.drude(1.0, wp, g)
        ref = 1.0 - wp**2 / (W_TEST**2 - 1j * W_TEST * g)
        np.testing.assert_allclose(m.evaluate(W_TEST), ref, rtol=1e-9)
        # Partial fractions: the DC pole plus one relaxation pole.
        assert sorted(a.real for a, _ in m.poles) == [-g, 0.0]

    def test_djordjevic_sarkar_pins_reference(self):
        m = DispersionModel.djordjevic_sarkar(4.3, 0.02, 5e9, 1e6, 1e12)
        e = m.evaluate(np.array([2 * np.pi * 5e9]))[0]
        assert abs(e.real - 4.3) < 1e-12
        assert abs(-e.imag / e.real - 0.02) < 1e-12
        assert 0.0 < m.eps_inf < 4.3

    def test_djordjevic_sarkar_tan_delta_flat(self):
        """tan delta stays flat to the model's inherent causal slope:
        within 2 % over +-0.5 decade around f_ref and within 6 % over
        +-1.5 decades (measured 0.017 / 0.052; densifying the comb does
        NOT shrink this — it is the Kramers-Kronig drift, not ripple)."""
        m = DispersionModel.djordjevic_sarkar(4.3, 0.02, 1e9, 1e6, 1e12)

        def drift(dec):
            f = np.logspace(9 - dec, 9 + dec, 30)
            e = m.evaluate(2 * np.pi * f)
            return np.abs((-e.imag / e.real) / 0.02 - 1.0).max()

        assert drift(0.5) < 0.02
        assert drift(1.5) < 0.06

    def test_pole_normalisation_conjugates_lower_half(self):
        a, r = complex(-1e9, -5e9), complex(1.0, 2.0)
        m = DispersionModel(2.0, ((a, r),), (1e8, 1e10))
        assert m.poles[0] == (a.conjugate(), r.conjugate())


class TestPassivityRejection:
    def test_unstable_pole(self):
        with pytest.raises(ValueError, match="unstable"):
            DispersionModel(1.0, ((complex(1e6), complex(1e6)),), (1e8, 1e10))

    def test_undamped_oscillatory_pole(self):
        with pytest.raises(ValueError, match="undamped"):
            DispersionModel(
                1.0,
                ((complex(0, 1e9), complex(0, -1e9)),),
                (1e8, 1e10),
            )

    def test_real_pole_complex_residue(self):
        with pytest.raises(ValueError, match="complex residue"):
            DispersionModel(
                1.0,
                ((complex(-1e9), complex(1e9, 1e9)),),
                (1e8, 1e10),
            )

    def test_dc_pole_negative_residue(self):
        with pytest.raises(ValueError, match="DC pole"):
            DispersionModel(1.0, ((complex(0), complex(-1e9)),), (1e8, 1e10))

    def test_band_sampled_gain(self):
        # A negative Debye strength is an active medium: eps'' < 0.
        with pytest.raises(ValueError, match="non-passive"):
            DispersionModel.debye(2.0, -1.0, 1e-10)

    def test_bad_band(self):
        with pytest.raises(ValueError, match="f_band"):
            DispersionModel(1.0, (), (1e10, 1e8))

    def test_lorentz_critical_damping(self):
        with pytest.raises(ValueError, match="critical"):
            DispersionModel.lorentz(1.0, 0.5, 1e9, 1e9)

    def test_bad_eps_inf(self):
        with pytest.raises(ValueError, match="eps_inf"):
            DispersionModel(-1.0, (), (1e8, 1e10))


class TestMaterialIntegration:
    def test_dispersive_factory(self):
        m = DispersionModel.debye(2.0, 1.0, 1e-11)
        mat = Material.dispersive("sub", m, sigma=0.01)
        assert mat.epsilon == (2.0, 2.0, 2.0)
        assert mat.sigma == (0.01, 0.01, 0.01)
        assert mat.dispersion is m
        assert not mat.is_lossless
        assert "dispersive" in repr(mat)

    def test_epsilon_must_match_eps_inf(self):
        m = DispersionModel.debye(2.0, 1.0, 1e-11)
        with pytest.raises(ValueError, match="eps_inf"):
            Material("bad", epsilon=(4.0, 4.0, 4.0), dispersion=m)

    def test_pec_excludes_dispersion(self):
        m = DispersionModel.debye(2.0, 1.0, 1e-11)
        with pytest.raises(ValueError, match="mutually exclusive"):
            Material("bad", epsilon=(2.0,) * 3, is_pec=True, dispersion=m)

    def test_dispersion_in_equality(self):
        m1 = DispersionModel.debye(2.0, 1.0, 1e-11)
        m2 = DispersionModel.debye(2.0, 1.5, 1e-11)
        a = Material.dispersive("s", m1)
        assert a == Material.dispersive("s", m1)
        assert a != Material.dispersive("s", m2)


class TestStoreRoundTrip:
    def test_round_trip(self):
        m = DispersionModel.lorentz(2.0, 0.5, 2 * np.pi * 5e9, 2 * np.pi * 2e8)
        mat = Material.dispersive("sub", m, mu=1.5, sigma=0.02)
        got = _material_from_dict(_material_to_dict(mat))
        assert got == mat
        assert got.dispersion.poles == mat.dispersion.poles
        assert got.dispersion.f_band == mat.dispersion.f_band

    def test_legacy_dict_without_key(self):
        d = _material_to_dict(Material.from_isotropic("air"))
        assert "dispersion" not in d
        assert _material_from_dict(d).dispersion is None
