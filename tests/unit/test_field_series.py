"""Unit tests for ``magnelio.fields.FieldRecording`` / ``FieldSpectrum`` (DD-259 step 1)."""

from __future__ import annotations

import numpy as np
import pytest

from magnelio.fields import FieldRecording, FieldSpectrum, FieldState
from magnelio.mesh.grid import GridLines

COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


@pytest.fixture
def grid():
    return GridLines(
        x=np.linspace(0, 10e-3, 6),
        y=np.array([0, 1e-3, 3e-3, 6e-3, 10e-3]),
        z=np.linspace(0, 4e-3, 4),
    )


@pytest.fixture
def layer_grid():
    """One cell layer along z: what a plane monitor stores (DD-175 taken literally)."""
    return GridLines(x=np.linspace(0, 10e-3, 6), y=np.linspace(0, 8e-3, 5), z=[2e-3, 3e-3])


def _stack(grid, names, n, scale=1.0):
    z = FieldState.zeros(grid)
    rng = np.random.default_rng(7)
    return {
        c: scale
        * rng.standard_normal((n, *getattr(z, c).shape))
        * np.arange(1, n + 1)[(slice(None),) + (None,) * 3]
        for c in names
    }


class TestRecording:
    def test_round_trip_and_frames(self, grid):
        comps = _stack(grid, COMPONENTS, 3)
        rec = FieldRecording(grid, [0.0, 1e-12, 2e-12], dt=1e-12, **comps)
        assert rec.n_frames == len(rec) == 3
        assert rec.components == COMPONENTS
        assert rec.shape == (5, 4, 3)
        assert not rec.is_complex
        for c in COMPONENTS:
            np.testing.assert_allclose(rec.component(c), comps[c], rtol=1e-13)
        f = rec.frame(1)
        assert isinstance(f, FieldState) and f.grid is grid
        for c in COMPONENTS:
            np.testing.assert_allclose(f.component(c), comps[c][1], rtol=1e-13)
        # Negative indices count from the end, like a list.
        np.testing.assert_allclose(rec.frame(-1).Ez, comps["Ez"][2], rtol=1e-13)
        with pytest.raises(IndexError):
            rec.frame(3)

    def test_time_bases(self, grid):
        comps = _stack(grid, ("Ez",), 2)
        rec = FieldRecording(grid, [1e-12, 3e-12], dt=2e-12, **comps)
        np.testing.assert_allclose(rec.times, [1e-12, 3e-12])
        np.testing.assert_allclose(rec.times_h, [2e-12, 4e-12])
        assert rec.index_of(2.9e-12) == 1
        np.testing.assert_allclose(rec.at_time(0.0).Ez, comps["Ez"][0], rtol=1e-13)
        # Without a step the two bases coincide (not a march).
        assert FieldRecording(grid, [0.0], **_stack(grid, ("Ez",), 1)).times_h[0] == 0.0
        with pytest.raises(ValueError, match="increasing"):
            FieldRecording(grid, [1e-12, 1e-12], **comps)

    def test_partial_components(self, grid):
        comps = _stack(grid, ("Ez", "Hx"), 2)
        rec = FieldRecording(grid, [0.0, 1e-12], **comps)
        assert rec.components == ("Ez", "Hx")
        f = rec.frame(0)
        np.testing.assert_allclose(f.Ez, comps["Ez"][0], rtol=1e-13)
        assert np.all(f.Ex == 0.0) and np.all(f.Hy == 0.0)
        with pytest.raises(KeyError, match="not recorded"):
            rec.component("Ex")

    def test_construction_errors(self, grid):
        with pytest.raises(TypeError, match="GridLines"):
            FieldRecording(object(), [0.0], Ez=np.zeros((1, 6, 5, 3)))
        with pytest.raises(ValueError, match="Yee shape"):
            FieldRecording(grid, [0.0, 1e-12], Ez=np.zeros((1, 6, 5, 3)))
        with pytest.raises(KeyError, match="unknown component"):
            FieldRecording(grid, [0.0], Bx=np.zeros((1, 6, 5, 3)))
        with pytest.raises(ValueError, match="at least one recorded"):
            FieldRecording(grid, [0.0])

    def test_cell_centred_matches_the_frames(self, grid):
        comps = _stack(grid, COMPONENTS, 2)
        rec = FieldRecording(grid, [0.0, 1e-12], **comps)
        stacked = rec.cell_centred()
        assert stacked["Ex"].shape == (2, 5, 4, 3)
        for i in range(2):
            single = rec.cell_centred(frame=i)
            for c in COMPONENTS:
                np.testing.assert_allclose(stacked[c][i], single[c])
                np.testing.assert_allclose(single[c], rec.frame(i).cell_centred([c])[c])
        box = rec.cell_centred(["Ez"], corners=((0, 0, 0), (4e-3, 10e-3, 4e-3)))
        assert box["Ez"].shape == (2, 2, 4, 3)

    def test_positions_and_repr(self, grid):
        rec = FieldRecording(grid, [0.0, 2e-12], **_stack(grid, ("Ey",), 2))
        x, y, z = rec.positions("Ey")
        assert x.size == 6 and y.size == 4 and z.size == 4
        assert len(rec.cell_centres[0]) == 5
        text = repr(rec)
        assert text.startswith("FieldRecording(2 frames") and "Ey" in text and "ns" in text

    def test_plane_layer_grid(self, layer_grid):
        """A one-cell-thick recording keeps Ez on one node plane and Ex, Ey on two."""
        comps = _stack(layer_grid, ("Ex", "Ey", "Ez"), 2)
        rec = FieldRecording(layer_grid, [0.0, 1e-12], **comps)
        assert rec.component("Ez").shape == (2, 6, 5, 1)
        assert rec.component("Ex").shape == (2, 5, 5, 2)
        assert rec.cell_centred(["Ez"])["Ez"].shape == (2, 5, 4, 1)

    def test_frame_feeds_an_initial_field(self, grid):
        from magnelio.sources import SourceFieldInitial  # noqa: PLC0415

        rec = FieldRecording(grid, [0.0, 1e-12], **_stack(grid, COMPONENTS, 2))
        src = SourceFieldInitial(name="restart", field=rec.at_time(1e-12))
        np.testing.assert_allclose(src.field.Ez, rec.frame(1).Ez)

    def test_plot_delegates_to_the_frame(self, grid):
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        rec = FieldRecording(grid, [0.0, 1e-12], **_stack(grid, COMPONENTS, 2))
        fig, ax = rec.plot("Ez", t=1e-12, normal="z", position=2e-3, plot_type="color")
        assert "t = 0.001 ns" in ax.get_title()
        plt.close(fig)
        with pytest.raises(ValueError, match="not both"):
            rec.plot("Ez", t=0.0, frame=1, normal="z")


class TestSpectrum:
    def test_complex_frames_and_snapshots(self, grid):
        z = FieldState.zeros(grid)
        ez = np.stack([np.ones(z.Ez.shape), 1j * np.ones(z.Ez.shape)])
        spec = FieldSpectrum(grid, [1e9, 2e9], Ez=ez)
        assert spec.is_complex and spec.components == ("Ez",)
        np.testing.assert_allclose(spec.frequencies, [1e9, 2e9])
        assert spec.f is spec.frequencies
        assert spec.index_of(1.8e9) == 1
        assert spec.at_frequency(2e9).is_complex
        np.testing.assert_allclose(spec.at_frequency(2e9).Ez, 1j)
        # Re(j · e^{+jπ/2}) = -1; the snapshot is real.  The phasors are
        # those of the e^{+jwt} convention, so the phase is w t.
        snap = spec.snapshot(2e9, phase=90.0)
        assert not snap.is_complex
        np.testing.assert_allclose(snap.Ez, -1.0, atol=1e-12)
        np.testing.assert_allclose(spec.snapshot(2e9, phase=-90.0).Ez, 1.0, atol=1e-12)
        np.testing.assert_allclose(spec.snapshot(frame=1).Ez, 0.0, atol=1e-12)
        assert repr(spec).startswith("FieldSpectrum(2 frames") and "GHz" in repr(spec)

    def test_real_input_becomes_complex(self, grid):
        spec = FieldSpectrum(grid, [1e9], **_stack(grid, ("Ex",), 1))
        assert np.iscomplexobj(spec.component("Ex"))

    def test_plot_at_a_phase(self, grid):
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        z = FieldState.zeros(grid)
        spec = FieldSpectrum(grid, [1e9], Ez=np.stack([(1 + 1j) * np.ones(z.Ez.shape)]))
        fig, ax = spec.plot("Ez", f=1e9, phase=0.0, normal="z", position=2e-3, plot_type="color")
        assert "f = 1 GHz" in ax.get_title()
        plt.close(fig)


class TestViewer:
    def test_series_show_lays_the_frame_on_the_cut(self, grid):
        pv = pytest.importorskip("pyvista")
        pv.OFF_SCREEN = True
        comps = _stack(grid, COMPONENTS, 3, scale=1.0)
        rec = FieldRecording(grid, [0.0, 1e-12, 2e-12], dt=1e-12, **comps)
        pl = rec.show("Ez", normal="z", position=2e-3, mode="none", frame=2, size=(200, 150))
        sheet = pl.renderer.actors["field_cut"].mapper.dataset
        k = int(np.argmin(np.abs(rec.cell_centres[2] - 2e-3)))
        expected = rec.cell_centred(["Ez"], frame=2)["Ez"][:, :, k].ravel(order="F")
        np.testing.assert_allclose(sheet.cell_data["field"], expected)
        pl.close()
        spec = FieldSpectrum(grid, [1e9, 2e9], Ez=comps["Ez"][:2] * (1 + 0j))
        pl = spec.show("Ez", mode="none", f=2e9, phase=90.0, size=(200, 150))
        assert "field_cut" in pl.renderer.actors
        pl.close()
