"""Unit tests for GeometryModel.plot_cross_section()."""

import pytest

occ = pytest.importorskip("OCC.Core.BRepPrimAPI")

import matplotlib

matplotlib.use("Agg")

from magnelio.geo import Brick, GeometryModel
from magnelio.materials.material import Material


def _count_patches(ax):
    """Count Polygon patches on an axes (excluding non-Polygon artists)."""
    from matplotlib.patches import Polygon as MplPolygon

    return sum(1 for p in ax.patches if isinstance(p, MplPolygon))


class TestPlotCrossSection:
    def test_brick_creates_patch(self):
        """Brick sliced through its middle produces exactly one patch."""
        mat = Material.from_isotropic("dielectric", epsilon=4.0)
        model = GeometryModel()
        model.add(Brick(origin=(0, 0, 0), size=(4e-3, 3e-3, 2e-3), material=mat))

        fig, ax = model.plot_cross_section("z", 1e-3)
        assert _count_patches(ax) == 1

    def test_invisible_shape_skipped(self):
        """Shape with visible=False produces no patches."""
        mat = Material.from_isotropic("hidden", epsilon=2.0)
        mat.visible = False
        model = GeometryModel()
        model.add(Brick(origin=(0, 0, 0), size=(4e-3, 3e-3, 2e-3), material=mat))

        fig, ax = model.plot_cross_section("z", 1e-3)
        assert _count_patches(ax) == 0

    def test_no_intersection_no_patches(self):
        """Cutting plane outside the shape produces no patches."""
        mat = Material.from_isotropic("dielectric", epsilon=4.0)
        model = GeometryModel()
        model.add(Brick(origin=(0, 0, 0), size=(4e-3, 3e-3, 2e-3), material=mat))

        fig, ax = model.plot_cross_section("z", 5e-3)
        assert _count_patches(ax) == 0

    def test_axis_labels_z_normal(self):
        """Cutting normal='z' → x-label contains 'x', y-label contains 'y'."""
        model = GeometryModel()
        mat = Material.from_isotropic("d", epsilon=2.0)
        model.add(Brick(origin=(0, 0, 0), size=(1e-3, 1e-3, 1e-3), material=mat))

        fig, ax = model.plot_cross_section("z", 0.5e-3)
        assert "x" in ax.get_xlabel().lower()
        assert "y" in ax.get_ylabel().lower()

    def test_scale_meters(self):
        """scale_mm=False → labels contain 'm' (not 'mm')."""
        model = GeometryModel()
        mat = Material.from_isotropic("d", epsilon=2.0)
        model.add(Brick(origin=(0, 0, 0), size=(1e-3, 1e-3, 1e-3), material=mat))

        fig, ax = model.plot_cross_section("z", 0.5e-3, scale_mm=False)
        xlabel = ax.get_xlabel()
        assert "[m]" in xlabel
        assert "mm" not in xlabel

    def test_multiple_shapes(self):
        """Two non-overlapping bricks produce two patches."""
        mat_a = Material.from_isotropic("a", epsilon=2.0)
        mat_b = Material.from_isotropic("b", epsilon=4.0)
        model = GeometryModel()
        model.add(Brick(origin=(0, 0, 0), size=(1e-3, 1e-3, 1e-3), material=mat_a))
        model.add(Brick(origin=(2e-3, 0, 0), size=(1e-3, 1e-3, 1e-3), material=mat_b))

        fig, ax = model.plot_cross_section("z", 0.5e-3)
        assert _count_patches(ax) == 2

    def test_air_drawn_as_outline(self):
        """Air material (transparent) is drawn as an unfilled dashed outline."""
        model = GeometryModel()
        model.add(Brick(origin=(0, 0, 0), size=(1e-3, 1e-3, 1e-3), material=Material.air()))

        fig, ax = model.plot_cross_section("z", 0.5e-3)
        assert _count_patches(ax) == 1
        patch = ax.patches[0]
        assert patch.get_facecolor()[3] == 0.0  # unfilled
        assert patch.get_linestyle() != "solid"

    def test_air_skipped_on_request(self):
        """outline_transparent=False restores the skip behaviour."""
        model = GeometryModel()
        model.add(Brick(origin=(0, 0, 0), size=(1e-3, 1e-3, 1e-3), material=Material.air()))

        fig, ax = model.plot_cross_section("z", 0.5e-3, outline_transparent=False)
        assert _count_patches(ax) == 0

    def test_flip_swaps_axes(self):
        """flip=True swaps axis labels and polygon coordinates."""
        mat = Material.from_isotropic("d", epsilon=2.0)
        model = GeometryModel()
        model.add(Brick(origin=(0, 0, 0), size=(1e-3, 2e-3, 3e-3), material=mat))

        # normal='x' → default: xlabel='y', ylabel='z'; flipped: xlabel='z', ylabel='y'
        _, ax_normal = model.plot_cross_section("x", 0.5e-3)
        _, ax_flipped = model.plot_cross_section("x", 0.5e-3, flip=True)

        assert "y" in ax_normal.get_xlabel().lower()
        assert "z" in ax_normal.get_ylabel().lower()
        assert "z" in ax_flipped.get_xlabel().lower()
        assert "y" in ax_flipped.get_ylabel().lower()


class TestVolumelessFeatures:
    """Wires, ports and lumped elements are part of the model too.

    A plane section returns nothing for any of them — they carry no
    volume — so before this they were simply absent from the picture:
    a wire antenna rendered as an empty air box.
    """

    @staticmethod
    def _monopole_model():
        from magnelio import circuit, ports
        from magnelio.geo import Curve, ThinWire

        model = GeometryModel(background=Material.pec())
        model.add(
            Brick(
                origin=(-20e-3, -20e-3, 0.0),
                size=(40e-3, 40e-3, 40e-3),
                material=Material.air(),
            )
        )
        model.add(
            ThinWire(
                Curve.polyline([(0.0, 0.0, 2e-3), (0.0, 0.0, 25e-3)]),
                radius=0.2e-3,
                name="monopole",
            )
        )
        model.add_port(
            ports.PortLumped(name="feed", start=(0.0, 0.0, 0.0), end=(0.0, 0.0, 2e-3), Z0=50.0)
        )
        model.add_element(
            circuit.LumpedElement(
                name="R1",
                start=(5e-3, 0.0, 4e-3),
                end=(5e-3, 0.0, 8e-3),
                element=circuit.SeriesRLC(R=100.0),
            )
        )
        return model

    def test_longitudinal_cut_draws_each_feature_once(self):
        _, ax = self._monopole_model().plot_cross_section("y", 0.0)
        assert len(ax.lines) == 3
        assert {t.get_text() for t in ax.texts} == {"monopole", "feed", "R1"}

    def test_wire_along_the_cut_is_a_line_across_it_is_a_ring(self):
        from magnelio.post.plot_geometry import _WIRE_COLOR

        model = self._monopole_model()
        _, along = model.plot_cross_section("y", 0.0)
        wire = next(ln for ln in along.lines if ln.get_color() == _WIRE_COLOR)
        assert len(wire.get_xdata()) > 2  # sampled polyline, not a marker
        assert wire.get_marker() in (None, "None", "")

        _, across = model.plot_cross_section("z", 10e-3)
        rings = [ln for ln in across.lines if ln.get_marker() == "o"]
        assert len(rings) == 1
        # Hollow, so a field plot underneath still reads through it.
        assert rings[0].get_markerfacecolor() == "none"
        assert [t.get_text() for t in across.texts] == ["monopole"]

    def test_cut_missing_the_wire_draws_nothing(self):
        # The wire ends at z = 25 mm; a cut above it must stay empty.
        _, ax = self._monopole_model().plot_cross_section("z", 30e-3)
        assert len(ax.lines) == 0
        assert len(ax.texts) == 0

    def test_features_can_be_switched_off(self):
        _, ax = self._monopole_model().plot_cross_section(
            "y", 0.0, show_wires=False, show_ports=False
        )
        assert len(ax.lines) == 0
        assert len(ax.texts) == 0

    def test_each_feature_class_gets_its_own_colour(self):
        # Wire, port and element must be told apart at a glance, and
        # none may borrow a material colour.
        _, ax = self._monopole_model().plot_cross_section("y", 0.0)
        colours = {ln.get_color() for ln in ax.lines}
        assert len(colours) == 3

    def test_face_port_is_drawn_on_a_perpendicular_cut_only(self):
        from magnelio import ports

        model = GeometryModel()
        model.add(
            Brick(
                origin=(0.0, 0.0, 0.0),
                size=(22.86e-3, 10.16e-3, 40e-3),
                material=Material.air(),
            )
        )
        model.add_port(ports.PortWaveguide(name="p1", plane="zmin"))
        model.add_port(ports.PortWaveguide(name="p2", plane="zmax"))

        _, cut = model.plot_cross_section("y", 5e-3)
        assert {t.get_text() for t in cut.texts} == {"p1", "p2"}

        # A cut lying *in* a port plane would cover the whole section;
        # drawing it would hide the geometry rather than explain it.
        _, parallel = model.plot_cross_section("z", 0.0)
        assert len(parallel.lines) == 0

    @staticmethod
    def _two_windows_on_one_face():
        """Two coax-sized ports on the same wall, far apart in z."""
        from magnelio import ports

        model = GeometryModel(background=Material.pec())
        model.add(
            Brick(
                origin=(0.0, 0.0, -20e-3),
                size=(40e-3, 30e-3, 120e-3),
                material=Material.air(),
            )
        )
        model.add_port(
            ports.PortWaveguide(
                name="p1", plane="ymax", corners=((0.0, None, -14e-3), (4e-3, None, -6e-3))
            )
        )
        model.add_port(
            ports.PortWaveguide(
                name="p2", plane="ymax", corners=((0.0, None, 80e-3), (4e-3, None, 88e-3))
            )
        )
        return model

    def test_a_sub_face_port_spans_its_window_not_the_domain(self):
        """The window is the port; the rest of that wall is wall."""
        model = self._two_windows_on_one_face()

        _, ax = model.plot_cross_section("z", -10e-3)

        drawn = [ln for ln in ax.lines if ln.get_linewidth() == 3.0]
        assert len(drawn) == 1
        xs = drawn[0].get_xdata()
        assert min(xs) == pytest.approx(0.0)
        assert max(xs) == pytest.approx(4.0)  # mm — the window, not 40 mm

    def test_a_cut_outside_every_window_draws_no_port(self):
        """Between the two windows that wall carries no port at all."""
        model = self._two_windows_on_one_face()

        _, ax = model.plot_cross_section("z", 40e-3)

        assert [ln for ln in ax.lines if ln.get_linewidth() == 3.0] == []
        assert {t.get_text() for t in ax.texts} == set()

    def test_each_window_is_drawn_on_its_own_cut(self):
        """Both ports sit on ymax; a cut must not draw them on top of each other."""
        model = self._two_windows_on_one_face()

        _, near = model.plot_cross_section("z", -10e-3)
        _, far = model.plot_cross_section("z", 84e-3)

        assert {t.get_text() for t in near.texts} == {"p1"}
        assert {t.get_text() for t in far.texts} == {"p2"}

    def test_a_full_face_port_still_spans_the_whole_edge(self):
        """No corners means the whole face — the pre-existing behaviour."""
        from magnelio import ports

        model = GeometryModel(background=Material.pec())
        model.add(
            Brick(origin=(0.0, 0.0, 0.0), size=(40e-3, 30e-3, 40e-3), material=Material.air())
        )
        model.add_port(ports.PortWaveguide(name="p1", plane="ymax"))

        _, ax = model.plot_cross_section("z", 20e-3)

        drawn = [ln for ln in ax.lines if ln.get_linewidth() == 3.0]
        assert len(drawn) == 1
        assert max(drawn[0].get_xdata()) == pytest.approx(40.0)
