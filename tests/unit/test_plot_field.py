import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from magnelio.post.plot_field import (
    CrossSectionOverlay,
    plot_field_scalar,
    plot_field_vector,
    render_geometry_overlay,
)


@pytest.fixture
def grid_2d():
    xc = np.linspace(0, 10e-3, 20)
    yc = np.linspace(0, 5e-3, 10)
    return xc, yc


@pytest.fixture
def scalar_field(grid_2d):
    xc, yc = grid_2d
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    return np.sin(2 * np.pi * X / 10e-3) * np.cos(2 * np.pi * Y / 5e-3)


@pytest.fixture
def vector_field(grid_2d):
    xc, yc = grid_2d
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    u = -np.sin(2 * np.pi * Y / 5e-3)
    v = np.cos(2 * np.pi * X / 10e-3)
    return u, v


class TestPlotFieldScalar:
    def test_color_returns_fig_ax(self, grid_2d, scalar_field):
        xc, yc = grid_2d
        fig, ax = plot_field_scalar(xc, yc, scalar_field, plot_type="color")
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_contour_returns_fig_ax(self, grid_2d, scalar_field):
        xc, yc = grid_2d
        fig, ax = plot_field_scalar(xc, yc, scalar_field, plot_type="contour")
        assert fig is not None
        plt.close(fig)

    def test_symmetric_limits(self, grid_2d, scalar_field):
        xc, yc = grid_2d
        fig, ax = plot_field_scalar(
            xc,
            yc,
            scalar_field,
            symmetric=True,
            plot_type="color",
        )
        # pcolormesh creates QuadMesh in ax.collections
        qm = [c for c in ax.collections if hasattr(c, "get_clim")]
        assert len(qm) > 0
        clim = qm[0].get_clim()
        assert clim[0] == pytest.approx(-clim[1])
        plt.close(fig)

    def test_explicit_vmin_vmax(self, grid_2d, scalar_field):
        xc, yc = grid_2d
        fig, ax = plot_field_scalar(
            xc,
            yc,
            scalar_field,
            vmin=-0.5,
            vmax=0.5,
            plot_type="color",
        )
        qm = [c for c in ax.collections if hasattr(c, "get_clim")]
        clim = qm[0].get_clim()
        assert clim == pytest.approx((-0.5, 0.5))
        plt.close(fig)

    def test_flip_swaps_labels(self, grid_2d, scalar_field):
        xc, yc = grid_2d
        fig1, ax1 = plot_field_scalar(
            xc,
            yc,
            scalar_field,
            xlabel="x",
            ylabel="y",
            flip=False,
        )
        fig2, ax2 = plot_field_scalar(
            xc,
            yc,
            scalar_field,
            xlabel="x",
            ylabel="y",
            flip=True,
        )
        assert "x" in ax1.get_xlabel()
        assert "y" in ax2.get_xlabel()
        plt.close("all")

    def test_scale_mm_in_labels(self, grid_2d, scalar_field):
        xc, yc = grid_2d
        fig, ax = plot_field_scalar(xc, yc, scalar_field, scale_mm=True)
        assert "mm" in ax.get_xlabel()
        plt.close(fig)

    def test_scale_m_in_labels(self, grid_2d, scalar_field):
        xc, yc = grid_2d
        fig, ax = plot_field_scalar(xc, yc, scalar_field, scale_mm=False)
        assert "[m]" in ax.get_xlabel()
        plt.close(fig)

    def test_existing_ax(self, grid_2d, scalar_field):
        xc, yc = grid_2d
        fig, ax = plt.subplots()
        fig2, ax2 = plot_field_scalar(xc, yc, scalar_field, ax=ax)
        assert ax2 is ax
        assert fig2 is fig
        plt.close(fig)

    def test_invalid_plot_type_raises(self, grid_2d, scalar_field):
        xc, yc = grid_2d
        with pytest.raises(ValueError, match="plot_type"):
            plot_field_scalar(xc, yc, scalar_field, plot_type="invalid")


class TestPlotFieldVector:
    def test_returns_fig_ax(self, grid_2d, vector_field):
        xc, yc = grid_2d
        u, v = vector_field
        fig, ax = plot_field_vector(xc, yc, u, v)
        assert fig is not None
        plt.close(fig)

    def test_normalize_arrows(self, grid_2d, vector_field):
        xc, yc = grid_2d
        u, v = vector_field
        fig, ax = plot_field_vector(
            xc,
            yc,
            u,
            v,
            normalize_arrows=True,
        )
        assert fig is not None
        plt.close(fig)

    def test_threshold_suppresses_arrows(self, grid_2d):
        xc, yc = grid_2d
        u = np.zeros((len(xc), len(yc)))
        v = np.zeros((len(xc), len(yc)))
        u[10, 5] = 1.0
        fig, ax = plot_field_vector(
            xc,
            yc,
            u,
            v,
            threshold=0.5,
        )
        assert fig is not None
        plt.close(fig)

    def test_vmax_clips(self, grid_2d, vector_field):
        xc, yc = grid_2d
        u, v = vector_field
        fig, ax = plot_field_vector(xc, yc, u, v, vmax=0.5)
        assert fig is not None
        plt.close(fig)

    def test_flip(self, grid_2d, vector_field):
        xc, yc = grid_2d
        u, v = vector_field
        fig1, ax1 = plot_field_vector(
            xc,
            yc,
            u,
            v,
            xlabel="x",
            ylabel="y",
            flip=False,
        )
        fig2, ax2 = plot_field_vector(
            xc,
            yc,
            u,
            v,
            xlabel="x",
            ylabel="y",
            flip=True,
        )
        assert "x" in ax1.get_xlabel()
        assert "y" in ax2.get_xlabel()
        plt.close("all")

    def test_quiver_scale_override(self, grid_2d, vector_field):
        xc, yc = grid_2d
        u, v = vector_field
        fig, ax = plot_field_vector(
            xc,
            yc,
            u,
            v,
            quiver_scale=42.0,
        )
        assert fig is not None
        plt.close(fig)


class TestArrowRaster:
    """The arrow positions must describe the picture, not the mesh."""

    @staticmethod
    def _arrow_positions(ax):
        quiv = ax.collections[0]
        return quiv.get_offsets()

    def test_raster_is_isotropic_on_a_graded_grid(self):
        # A grid refined 10x in the middle of x and uniform in y: strided
        # subsampling would cluster arrows in the refined band.
        xc = np.concatenate(
            [
                np.linspace(0.0, 4e-3, 9, endpoint=False),
                np.linspace(4e-3, 6e-3, 40, endpoint=False),
                np.linspace(6e-3, 10e-3, 10),
            ]
        )
        yc = np.linspace(0.0, 10e-3, 12)
        u = np.ones((xc.size, yc.size))
        v = np.zeros((xc.size, yc.size))
        fig, ax = plot_field_vector(xc, yc, u, v, density=16, scale_mm=False)
        pos = self._arrow_positions(ax)
        xs = np.unique(pos[:, 0])
        ys = np.unique(pos[:, 1])
        assert np.allclose(np.diff(xs), np.diff(xs)[0])
        assert np.diff(xs)[0] == pytest.approx(np.diff(ys)[0], rel=1e-9)
        plt.close(fig)

    def test_density_counts_arrows_along_the_longer_axis(self, vector_field):
        xc = np.linspace(0.0, 10e-3, 60)
        yc = np.linspace(0.0, 5e-3, 30)
        u = np.ones((xc.size, yc.size))
        v = np.zeros((xc.size, yc.size))
        fig, ax = plot_field_vector(xc, yc, u, v, density=11, scale_mm=False)
        pos = self._arrow_positions(ax)
        assert np.unique(pos[:, 0]).size == 11
        assert np.unique(pos[:, 1]).size == 6
        plt.close(fig)

    def test_values_are_interpolated_not_sampled(self):
        # A linear ramp: an interpolated raster point between two cell
        # centres must read the intermediate value.
        xc = np.linspace(0.0, 1.0, 5)
        yc = np.linspace(0.0, 1.0, 5)
        X, _ = np.meshgrid(xc, yc, indexing="ij")
        fig, ax = plot_field_vector(xc, yc, X.copy(), np.zeros_like(X), density=9, scale_mm=False)
        quiv = ax.collections[0]
        pos = quiv.get_offsets()
        assert np.allclose(quiv.U, pos[:, 0], atol=1e-12)
        plt.close(fig)


class TestValidMask:
    def test_masked_cells_stay_blank(self):
        xc = np.linspace(0.0, 1.0, 9)
        yc = np.linspace(0.0, 1.0, 9)
        u = np.ones((9, 9))
        v = np.zeros((9, 9))
        valid = np.ones((9, 9), dtype=bool)
        valid[:4, :] = False  # left half buried in a conductor
        fig, ax = plot_field_vector(xc, yc, u, v, valid=valid, density=9, scale_mm=False)
        quiv = ax.collections[0]
        pos = quiv.get_offsets()
        drawn = ~np.asarray(quiv.Umask)
        assert not np.any(pos[drawn, 0] < 0.35)
        assert np.any(drawn)
        plt.close(fig)

    def test_masked_neighbours_do_not_dilute_the_field(self):
        # Without the mask the zeros of the dead half would be averaged
        # into the live cells at the interface.
        xc = np.linspace(0.0, 1.0, 9)
        yc = np.linspace(0.0, 1.0, 9)
        u = np.full((9, 9), 2.0)
        u[:4, :] = 0.0
        v = np.zeros((9, 9))
        valid = np.ones((9, 9), dtype=bool)
        valid[:4, :] = False
        fig, ax = plot_field_vector(xc, yc, u, v, valid=valid, density=9, scale_mm=False)
        quiv = ax.collections[0]
        live = ~np.asarray(quiv.Umask)
        assert np.allclose(np.asarray(quiv.U)[live], 2.0)
        plt.close(fig)


class TestPlotFieldVectorNormalComponent:
    def test_pure_normal_field_draws_markers(self, grid_2d):
        from matplotlib.collections import PathCollection

        xc, yc = grid_2d
        u = np.zeros((len(xc), len(yc)))
        v = np.zeros((len(xc), len(yc)))
        w = np.ones((len(xc), len(yc)))
        w[:5, :] = -1.0
        fig, ax = plot_field_vector(xc, yc, u, v, w=w, wlabel="z")
        scatters = [c for c in ax.collections if isinstance(c, PathCollection)]
        # ⊙ and ⊗ groups, each a coloured-circle + glyph scatter pair
        assert len(scatters) == 4
        assert ax.get_legend() is not None
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any("+z" in s for s in labels)
        assert any("−z" in s for s in labels)
        plt.close(fig)

    def test_colour_encodes_full_magnitude(self, grid_2d):
        xc, yc = grid_2d
        u = np.full((len(xc), len(yc)), 3.0)
        v = np.zeros((len(xc), len(yc)))
        w = np.full((len(xc), len(yc)), 4.0)
        fig, ax = plot_field_vector(xc, yc, u, v, w=w)
        quiv = ax.collections[0]
        assert float(np.max(quiv.get_array())) == pytest.approx(5.0)
        assert quiv.norm.vmax == pytest.approx(5.0)
        plt.close(fig)

    def test_auto_scale_references_full_magnitude_with_w(self, grid_2d):
        xc, yc = grid_2d
        u = np.full((len(xc), len(yc)), 3.0)
        v = np.zeros((len(xc), len(yc)))
        w = np.full((len(xc), len(yc)), 4.0)
        fig, ax = plot_field_vector(xc, yc, u, v, w=w)
        fig2, ax2 = plot_field_vector(xc, yc, u, v)
        q, q2 = ax.collections[0], ax2.collections[0]
        # scale = peak magnitude / arrow spacing: 5/3 ratio with vs without w
        assert q.scale / q2.scale == pytest.approx(5.0 / 3.0)
        plt.close("all")

    def test_no_markers_when_in_plane_dominates(self, grid_2d, vector_field):
        from matplotlib.collections import PathCollection

        xc, yc = grid_2d
        u, v = vector_field
        w = 0.001 * np.ones_like(u)
        fig, ax = plot_field_vector(xc, yc, u, v, w=w)
        scatters = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(scatters) == 0
        plt.close(fig)

    def test_colorbar_label_states_in_plane_without_w(self, grid_2d, vector_field):
        xc, yc = grid_2d
        u, v = vector_field
        fig, ax = plot_field_vector(xc, yc, u, v)
        assert "In-plane" in fig.axes[-1].get_ylabel()
        fig2, ax2 = plot_field_vector(xc, yc, u, v, w=np.ones_like(u))
        assert "In-plane" not in fig2.axes[-1].get_ylabel()
        plt.close("all")

    def test_flip_with_w(self, grid_2d, vector_field):
        xc, yc = grid_2d
        u, v = vector_field
        fig, ax = plot_field_vector(xc, yc, u, v, w=np.ones_like(u), flip=True)
        assert fig is not None
        plt.close(fig)


class TestRenderGeometryOverlay:
    def test_none_overlay_is_noop(self):
        fig, ax = plt.subplots()
        render_geometry_overlay(None, ax=ax)
        plt.close(fig)

    def test_invalid_overlay_silently_skips(self):
        fig, ax = plt.subplots()
        overlay = CrossSectionOverlay(
            geometry="not_real_geometry",
            normal="z",
            position=0.0,
        )
        render_geometry_overlay(overlay, ax=ax)
        plt.close(fig)
