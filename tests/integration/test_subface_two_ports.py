"""WP6.3 integration: two sub-face ports on one bbox face.

Geometry: two parallel rectangular waveguide channels (a = 20 mm,
b = 10 mm, separated by a 10 mm PEC web from ``background=pec``),
running the full domain length along z.  Each channel opening on
Z_MIN / Z_MAX carries its own sub-face ``PortWaveguide(corners=...)`` —
so Z_MIN hosts *two* ports.

Physics checks with channel 1 excited in TE10:

- transmission through channel 1 is essentially lossless in band,
- channel 2 is PEC-separated → its port sees only numerical noise,
- the sub-face mode solver reports the *channel* TE10 cutoff.
"""

from __future__ import annotations

import numpy as np

from magnelio import AnalysisScatteringTD, Material, Mesh, MeshControl
from magnelio.geo import Brick, Difference, GeometryModel
from magnelio.ports import PortWaveguide
from magnelio.ports._modal.factory import PortSpecNumerical

C0 = 2.99792458e8
A_CH = 20e-3  # channel width (x) -> TE10 cutoff 7.495 GHz
B_CH = 10e-3  # channel height (y)
WEB = 10e-3  # PEC web between the channels
LENGTH = 30e-3  # channel length (z)
F_MAX = 11e9
F_C = C0 / (2 * A_CH)

# Channel x-ranges inside the domain (channels touch the y walls and
# the outer x walls; the web between them is background PEC).
CH1_X = (0.0, A_CH)
CH2_X = (A_CH + WEB, 2 * A_CH + WEB)
CH_Y = (0.0, B_CH)


def _build_analysis() -> AnalysisScatteringTD:
    pec = Material.pec()
    vac = Material.air()

    model = GeometryModel(background=pec)
    model.add(
        Brick(
            origin=(CH1_X[0], CH_Y[0], 0.0),
            size=(A_CH, B_CH, LENGTH),
            material=vac,
        )
    )
    model.add(
        Brick(
            origin=(CH2_X[0], CH_Y[0], 0.0),
            size=(A_CH, B_CH, LENGTH),
            material=vac,
        )
    )

    control = MeshControl(min_nodes_per_wavelength=15, max_cell_size=2e-3)
    mesh = Mesh.from_geometry(model, control, f_max=F_MAX)

    ports = [
        PortWaveguide(
            name="p1_in",
            plane="zmin",
            corners=((CH1_X[0], CH_Y[0], None), (CH1_X[1], CH_Y[1], None)),
        ),
        PortWaveguide(
            name="p1_out",
            plane="zmax",
            corners=((CH1_X[0], CH_Y[0], None), (CH1_X[1], CH_Y[1], None)),
        ),
        PortWaveguide(
            name="p2_in",
            plane="zmin",
            corners=((CH2_X[0], CH_Y[0], None), (CH2_X[1], CH_Y[1], None)),
        ),
        PortWaveguide(
            name="p2_out",
            plane="zmax",
            corners=((CH2_X[0], CH_Y[0], None), (CH2_X[1], CH_Y[1], None)),
        ),
    ]
    return AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions(
            {
                "xmin": "PEC",
                "xmax": "PEC",
                "ymin": "PEC",
                "ymax": "PEC",
                "zmin": "PEC",
                "zmax": "PEC",
            }
        ),
        ports=ports,
        f_max=F_MAX,
        n_freq=41,
        verbose=False,
    )


def test_two_subface_ports_on_one_face():
    analysis = _build_analysis()

    # Declarative resolution: hollow windows -> TE PortSpecNumerical
    # with the tangential window attached.
    resolved = {spec.name: spec for spec in analysis.ports}
    assert set(resolved) == {"p1_in", "p1_out", "p2_in", "p2_out"}
    for spec in resolved.values():
        assert isinstance(spec, PortSpecNumerical)
        assert spec.window is not None

    # Port reports carry the *channel* TE10 cutoff, not the face's.
    reports = analysis.solve_ports()
    for label, report in reports.items():
        f_c_num = report.cutoff_num
        assert abs(f_c_num - F_C) / F_C < 2e-2, (
            f"{label}: cutoff {f_c_num / 1e9:.3f} GHz vs channel TE10 {F_C / 1e9:.3f} GHz"
        )

    result = analysis.run(excited=["p1_in"])
    f_axis = result.f_axis
    in_band = f_axis >= 1.15 * F_C  # clear of the Mur near-cutoff peak

    s21 = 20 * np.log10(np.abs(result.S("p1_out", "p1_in")) + 1e-30)
    s11 = 20 * np.log10(np.abs(result.S("p1_in", "p1_in")) + 1e-30)
    iso_in = 20 * np.log10(np.abs(result.S("p2_in", "p1_in")) + 1e-30)
    iso_out = 20 * np.log10(np.abs(result.S("p2_out", "p1_in")) + 1e-30)

    assert np.all(np.isfinite(s21))
    # Lossless straight channel: |S21| ~ 0 dB in band.
    assert s21[in_band].min() > -1.0, (
        f"channel-1 transmission dropped to {s21[in_band].min():.2f} dB"
    )
    assert s11[in_band].max() < -10.0, (
        f"channel-1 match: max in-band |S11| = {s11[in_band].max():.2f} dB"
    )
    # PEC-separated neighbour channel: numerical noise only.
    assert iso_in.max() < -60.0, (
        f"channel isolation broken: |S(p2_in, p1_in)| = {iso_in.max():.1f} dB"
    )
    assert iso_out.max() < -60.0, (
        f"channel isolation broken: |S(p2_out, p1_in)| = {iso_out.max():.1f} dB"
    )


def _rect_coax_mesh(embedded: bool):
    """Rectangular coax cross-section, either embedded as a window in a
    larger PEC block or standalone as the whole domain."""
    pec = Material.pec()
    vac = Material.air()
    ox = 10e-3 if embedded else 0.0
    oy = 5e-3 if embedded else 0.0
    outer_w, outer_h = 20e-3, 10e-3
    inner_w, inner_h = 5e-3, 2.5e-3
    L = 7.5e-3
    model = GeometryModel(background=pec)
    channel = Brick(
        origin=(ox, oy, 0.0),
        size=(outer_w, outer_h, L),
        material=vac,
    )
    inner = Brick(
        origin=(ox + (outer_w - inner_w) / 2, oy + (outer_h - inner_h) / 2, 0.0),
        size=(inner_w, inner_h, L),
        material=pec,
    )
    if embedded:
        frame = Brick(
            origin=(0.0, 0.0, 0.0),
            size=(40e-3, 20e-3, L),
            material=pec,
        )
        model.add(Difference(frame, channel, material=pec, name="frame"))
    model.add(Difference(channel, inner, material=vac, name="channel"))
    model.add(inner)
    control = MeshControl(min_nodes_per_wavelength=15, max_cell_size=1.25e-3)
    mesh = Mesh.from_geometry(model, control, f_max=F_MAX)
    corners = ((ox, oy, None), (ox + outer_w, oy + outer_h, None)) if embedded else None
    return mesh, corners


def test_subface_tem_window_matches_standalone_cross_section():
    """Embedded rect-coax window == the same cross-section as full domain.
    The window's PEC frame (edge-BC rule ring) plays the outer
    conductor; the standalone reference has real domain walls.  The
    TEM z_line of both must agree — same cross-section, same grid
    spacing — which pins the whole sub-face TEM chain (ring conductor
    group, windowed Laplace, boundary mass factors).
    """
    from magnelio.ports import PortSpecMultiConductor

    z_lines = {}
    for embedded in (False, True):
        mesh, corners = _rect_coax_mesh(embedded)
        analysis = AnalysisScatteringTD(
            mesh=mesh.with_boundary_conditions(
                {
                    "xmin": "PEC",
                    "xmax": "PEC",
                    "ymin": "PEC",
                    "ymax": "PEC",
                    "zmin": "PEC",
                    "zmax": "PEC",
                }
            ),
            ports=[PortWaveguide(name="p", plane="zmin", corners=corners)],
            f_max=F_MAX,
            verbose=False,
        )
        spec = analysis.ports[0]
        assert isinstance(spec, PortSpecMultiConductor)
        report = analysis.solve_ports()["p"]
        z_lines[embedded] = report.z_line_num
    assert z_lines[True] > 0.0
    assert abs(z_lines[True] - z_lines[False]) / z_lines[False] < 3e-2, (
        f"embedded window z_line {z_lines[True]:.2f} Ohm vs standalone {z_lines[False]:.2f} Ohm"
    )
