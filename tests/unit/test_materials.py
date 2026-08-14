"""Unit tests for Material dataclass."""

import pytest

from magnelio.materials.material import Material


class TestMaterialDefaults:
    def test_air_factory(self):
        m = Material.air()
        assert m.name == "air"
        assert m.epsilon == (1.0, 1.0, 1.0)
        assert m.mu == (1.0, 1.0, 1.0)
        assert m.sigma == (0.0, 0.0, 0.0)
        assert not m.is_pec

    def test_pec_factory(self):
        m = Material.pec()
        assert m.is_pec
        # PEC is realised by edge masking; sigma stays at its default
        assert m.sigma == (0.0, 0.0, 0.0)
        assert not m.is_lossy_metal

    def test_vacuum_alias(self):
        m = Material.vacuum()
        assert m.name == "vacuum"

    def test_from_isotropic(self):
        m = Material.from_isotropic("FR4", epsilon=4.4, sigma=0.001)
        assert m.epsilon == (4.4, 4.4, 4.4)
        assert m.sigma == (0.001, 0.001, 0.001)


class TestLossyMetal:
    def test_factory(self):
        m = Material.lossy_metal("copper", sigma=5.8e7)
        assert m.is_pec
        assert m.is_lossy_metal
        assert m.sigma == (5.8e7, 5.8e7, 5.8e7)
        assert m.mu == (1.0, 1.0, 1.0)
        assert not m.is_lossless

    def test_mu_enters(self):
        m = Material.lossy_metal("steel", sigma=1.4e6, mu=100.0)
        assert m.mu == (100.0, 100.0, 100.0)

    def test_invalid_sigma_rejected(self):
        with pytest.raises(ValueError):
            Material.lossy_metal("bad", sigma=0.0)
        with pytest.raises(ValueError):
            Material.lossy_metal("bad", sigma=-1.0)
        with pytest.raises(ValueError):
            Material.lossy_metal("bad", sigma=float("inf"))

    def test_legacy_inf_sigma_is_not_lossy(self):
        # Stores written before DD-081 carry sigma=inf on plain PEC
        m = Material(name="PEC", sigma=(float("inf"),) * 3, is_pec=True)
        assert m.is_pec
        assert not m.is_lossy_metal

    def test_lossy_dielectric_is_not_lossy_metal(self):
        m = Material.from_isotropic("lossy", epsilon=4.0, sigma=0.01)
        assert not m.is_lossy_metal


class TestMaterialValidation:
    def test_wrong_tuple_length(self):
        with pytest.raises(ValueError):
            Material(name="bad", epsilon=(1.0, 1.0))  # only 2 elements

    def test_isotropic_property(self):
        iso = Material.from_isotropic("iso", epsilon=4.0)
        assert iso.is_isotropic

        aniso = Material(name="aniso", epsilon=(4.0, 4.0, 2.0))
        assert not aniso.is_isotropic

    def test_lossless_property(self):
        m = Material.air()
        assert m.is_lossless

        m2 = Material(name="lossy", sigma=(0.1, 0.1, 0.1))
        assert not m2.is_lossless


class TestMaterialAnisotropic:
    def test_diagonal_anisotropic(self):
        m = Material(
            name="aniso",
            epsilon=(2.0, 3.0, 4.0),
            mu=(1.0, 1.0, 2.0),
            sigma=(0.0, 0.0, 0.01),
        )
        assert m.epsilon[2] == 4.0
        assert m.mu[2] == 2.0
        assert m.sigma[2] == 0.01
