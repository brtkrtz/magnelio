"""The Poynting vector as a derived view of a field (DD-270).

``S = E × H`` on the cell centres: the plane-wave value and its
convergence, the two readings of a real and a complex frame, the
vocabulary (``"S"``, ``"Sx"``…) the containers and the pictures speak,
and the refusal to state a power density from half a field.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio.constants import ETA0
from magnelio.fields import FieldRecording, FieldSpectrum, FieldState
from magnelio.mesh.grid import GridLines
from magnelio.post._symmetry import mirror_sign

COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _cube(n: int, L: float = 1.0) -> GridLines:
    line = np.linspace(0.0, L, n + 1)
    return GridLines(x=line, y=line, z=line)


def _plane_wave(grid: GridLines, e0: float = 3.0, k: float = 2 * np.pi) -> FieldState:
    """A z-travelling standing pattern: E = x̂ E₀cos(kz), H = ŷ E₀/η cos(kz)."""
    return FieldState.from_function(
        grid,
        E=lambda x, y, z: (e0 * np.cos(k * z), 0 * y, 0 * z),
        H=lambda x, y, z: (0 * x, e0 / ETA0 * np.cos(k * z), 0 * z),
    )


def _series_grid() -> GridLines:
    return GridLines(x=np.linspace(0, 4e-3, 5), y=np.linspace(0, 3e-3, 4), z=[0.0, 1e-3])


def _stack(grid, names, n, dtype=float):
    zero = FieldState.zeros(grid)
    rng = np.random.default_rng(11)
    out = {}
    for c in names:
        shape = (n, *getattr(zero, c).shape)
        a = rng.standard_normal(shape)
        out[c] = a + 1j * rng.standard_normal(shape) if dtype is complex else a
    return out


class TestThePlaneWave:
    def test_value_and_direction(self):
        e0, n = 3.0, 20
        fs = _plane_wave(_cube(n), e0=e0)
        s = fs.poynting()
        assert s.shape == (n, n, n, 3)
        # The wave carries power along +z alone.
        assert np.abs(s[..., 0]).max() == 0.0
        assert np.abs(s[..., 1]).max() == 0.0
        zc = fs.cell_centres[2]
        exact = e0**2 / ETA0 * np.cos(2 * np.pi * zc) ** 2
        np.testing.assert_allclose(s[n // 2, n // 2, :, 2], exact, rtol=0, atol=0.02 * exact.max())

    def test_second_order_in_the_cell_size(self):
        """Halving the cell quarters the error: the cell-centre average is O(h²)."""

        def err(n):
            fs = _plane_wave(_cube(n))
            zc = fs.cell_centres[2]
            exact = 9.0 / ETA0 * np.cos(2 * np.pi * zc) ** 2
            got = fs.poynting()[n // 2, n // 2, :, 2]
            return np.abs(got - exact).max() / exact.max()

        coarse, fine = err(20), err(40)
        assert 3.5 < coarse / fine < 4.5


class TestTheTwoReadings:
    def _phasor(self, grid, h_factor):
        e0 = 2.0
        return FieldState.from_function(
            grid,
            E=lambda x, y, z: (e0 + 0j * x, 0j * y, 0j * z),
            H=lambda x, y, z: (0j * x, h_factor * e0 / ETA0 + 0j * y, 0j * z),
        )

    def test_rms_phasor_is_the_time_average(self):
        """A complex frame is an RMS phasor: Re(E × H*), with no further half."""
        fs = self._phasor(_cube(6), 1.0)
        s = fs.poynting()
        assert not np.iscomplexobj(s)
        np.testing.assert_allclose(s[..., 2], 4.0 / ETA0, rtol=1e-12)

    def test_quadrature_carries_no_active_power(self):
        """E and H in quadrature: no net flow, but stored reactive power."""
        fs = self._phasor(_cube(6), 1j)
        np.testing.assert_allclose(fs.poynting()[..., 2], 0.0, atol=1e-18)
        complex_s = fs.poynting(complex_product=True)
        assert np.iscomplexobj(complex_s)
        np.testing.assert_allclose(np.imag(complex_s[..., 2]), -4.0 / ETA0, rtol=1e-12)

    def test_a_real_frame_ignores_complex_product(self):
        fs = _plane_wave(_cube(6))
        np.testing.assert_array_equal(fs.poynting(), fs.poynting(complex_product=True))


class TestTheVocabulary:
    def test_cell_centred_derives_the_named_components(self):
        fs = _plane_wave(_cube(8))
        cc = fs.cell_centred(["Sz", "Ex"])
        assert sorted(cc) == ["Ex", "Sz"]
        np.testing.assert_array_equal(cc["Sz"], fs.poynting()[..., 2])

    def test_an_unknown_name_is_rejected(self):
        with pytest.raises(KeyError, match="component must be one of"):
            _plane_wave(_cube(4)).cell_centred(["Bx"])

    def test_a_series_derives_per_frame(self):
        grid = _series_grid()
        rec = FieldRecording(grid, [0.0, 1e-12], dt=1e-12, **_stack(grid, COMPONENTS, 2))
        every = rec.poynting()
        assert every.shape == (2, 4, 3, 1, 3)
        np.testing.assert_allclose(every[1], rec.poynting(frame=1))
        np.testing.assert_allclose(every[1], rec.frame(1).poynting())
        np.testing.assert_allclose(rec.poynting(frame=1, squeeze=True), every[1][:, :, 0])
        np.testing.assert_allclose(
            rec.cell_centred(["Sy"], frame=0)["Sy"], every[0][..., 1], rtol=1e-12
        )

    def test_a_layer_derives_too(self):
        grid = _series_grid()
        rec = FieldRecording(grid, [0.0], **_stack(grid, COMPONENTS, 1))
        layer = rec.cell_centred_layer(0, 2, 0, ["Sx", "Ez"])
        assert sorted(layer) == ["Ez", "Sx"]
        np.testing.assert_allclose(layer["Sx"], rec.poynting(frame=0)[:, :, 0, 0], rtol=1e-12)

    def test_corners_cut_the_same_box_as_the_fields(self):
        grid = _series_grid()
        rec = FieldRecording(grid, [0.0], **_stack(grid, COMPONENTS, 1))
        corners = ((0.0, 0.0, 0.0), (2e-3, 3e-3, 1e-3))
        cut = rec.poynting(corners=corners, frame=0)
        assert cut.shape == (2, 3, 1, 3)
        np.testing.assert_allclose(cut, rec.poynting(frame=0)[:2], rtol=1e-12)

    def test_the_viewer_offers_it_only_with_both_fields(self):
        from magnelio.post.field_3d import _available_components

        assert "S" in _available_components(COMPONENTS)
        assert "Sz" in _available_components(COMPONENTS)
        assert "S" not in _available_components(("Ex", "Ey", "Ez"))


class TestHalfAField:
    def test_a_series_without_h_refuses(self):
        grid = _series_grid()
        rec = FieldRecording(grid, [0.0], **_stack(grid, ("Ex", "Ey", "Ez"), 1))
        with pytest.raises(KeyError, match="needs all six"):
            rec.poynting()

    def test_a_frame_of_such_a_series_refuses_too(self):
        """The unrecorded half is zeros in a frame — a silent zero power density."""
        grid = _series_grid()
        rec = FieldRecording(grid, [0.0], **_stack(grid, ("Ex", "Ey", "Ez"), 1))
        with pytest.raises(KeyError, match="needs all six"):
            rec.frame(0).poynting()
        with pytest.raises(KeyError, match="needs all six"):
            rec.frame(0).cell_centred(["Sx"])

    def test_a_hand_built_field_states_it(self):
        """A field assembled by hand records nothing and answers for all six."""
        assert _plane_wave(_cube(4)).poynting().shape == (4, 4, 4, 3)


class TestTheMirrorRule:
    """S = E × H, so its parity is the product — the same on either wall."""

    @pytest.mark.parametrize("kind", ["PEC", "PMC"])
    @pytest.mark.parametrize("mirror_axis", [0, 1, 2])
    @pytest.mark.parametrize("comp_axis", [0, 1, 2])
    def test_parity_is_the_product_of_the_field_parities(self, kind, mirror_axis, comp_axis):
        b, c = (comp_axis + 1) % 3, (comp_axis + 2) % 3
        from_fields = mirror_sign("E", b, mirror_axis, kind) * mirror_sign(
            "H", c, mirror_axis, kind
        )
        # The other term of the cross product must agree, or S would
        # have no parity at all.
        other = mirror_sign("E", c, mirror_axis, kind) * mirror_sign("H", b, mirror_axis, kind)
        assert from_fields == other
        assert mirror_sign("S", comp_axis, mirror_axis, kind) == from_fields

    def test_no_power_crosses_a_symmetry_plane(self):
        for kind in ("PEC", "PMC"):
            assert mirror_sign("S", 1, 1, kind) == -1.0
            assert mirror_sign("S", 0, 1, kind) == 1.0
        assert mirror_sign("S", None, 1, "PEC") == 1.0


class TestThePicture:
    def test_a_slice_draws_the_vector_and_names_its_unit(self):
        import matplotlib

        matplotlib.use("Agg")
        fs = _plane_wave(_cube(10))
        _, ax = fs.plot("S", normal="x")
        assert "Poynting" in ax.get_title()
        assert "W/m²" in ax.figure.axes[-1].get_ylabel()
        _, ax = fs.plot("Sz", normal="x", plot_type="color")
        assert "W/m²" in ax.figure.axes[-1].get_ylabel()

    def test_a_spectrum_plots_it_without_a_phase(self):
        """A power density is real already: the phase slider cannot rotate it."""
        import matplotlib

        matplotlib.use("Agg")
        grid = _series_grid()
        spec = FieldSpectrum(grid, [5e9], **_stack(grid, COMPONENTS, 1, dtype=complex))
        _, ax_0 = spec.plot("Sz", frame=0, plot_type="color", phase=0.0)
        _, ax_90 = spec.plot("Sz", frame=0, plot_type="color", phase=90.0)
        first = ax_0.get_children()[0].get_array()
        second = ax_90.get_children()[0].get_array()
        np.testing.assert_allclose(np.asarray(first), np.asarray(second))
