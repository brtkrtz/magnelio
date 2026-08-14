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
