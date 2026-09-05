"""DD-256 gate: a thin wire landing on a thin PEC sheet.

A quarter-wave monopole stands on a plate that spans the domain and
touches the CPML on all four sides (an image plane).  The plate is built
three ways on ONE grid — plane positions, wire nodes and feed edge
identical, only the plate's representation differs:

* ``solid`` — a 3 mm PEC block whose top face is the wire's foot (the
  DD-080 endpoint case (a), the reference);
* ``sheet`` — a 0.2 mm sheet on the DD-059 thin-metallisation path
  (t/cell = 0.2), the wire drawn on the metal's TOP face, i.e. inside
  the thickness band the mesher collapses onto the sheet plane;
* ``sheet_thick`` — the same with t/cell = 0.7, where the cell above the
  sheet plane classifies as metal.

The wire's foot segment is the only place the three differ: on the
sheet its ring faces cross the metal layer and compose ``m`` with the
sub-cell value (DD-256) instead of keeping the bare-grid inductance.
Resonance and feed resistance of the sheet variants are pinned to the
solid's; the thick variant carries the sub-cell representation's own
thickness error on top and gets the wider window.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from magnelio import AnalysisScatteringTD, GeometryModel, Mesh, MeshControl, geo
from magnelio.ports import PortSpecLumped

pytest.importorskip("OCC.Core.BRepPrimAPI")

C0 = 299_792_458.0
D = 1e-3  # cell
H = 15e-3  # monopole height above the plate
A_RADIUS = 0.05 * D
SIDE = 6e-3  # half the plate = half the box; the CPML wraps outside
ZB = 4e-3  # the sheet plane / the solid's top face
Z0_FEED = 73.0
N_STEPS = 8000
THICKNESS = {"solid": 3e-3, "sheet": 0.2e-3, "sheet_thick": 0.7e-3}

_CACHE: dict = {}


def _model(kind):
    t = THICKNESS[kind]
    if kind == "solid":
        z_lo, z_hi, z_foot = ZB - t, ZB, ZB
    else:
        z_lo, z_hi, z_foot = ZB, ZB + t, ZB + t  # wire drawn on the metal top
    z_top = ZB + H + 4 * D
    w = 2 * SIDE
    model = GeometryModel(
        boundary_conditions={f: "CPML" for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    )
    model.add(geo.Brick(origin=(-SIDE, -SIDE, 0.0), size=(w, w, z_lo), material="air"))
    model.add(geo.Brick(origin=(-SIDE, -SIDE, z_lo), size=(w, w, z_hi - z_lo), material="pec"))
    model.add(geo.Brick(origin=(-SIDE, -SIDE, z_hi), size=(w, w, z_top - z_hi), material="air"))
    # Stub from the plate to the feed gap, gap edge, arm: DD-080 (c).
    model.add(
        geo.ThinWire(
            geo.Curve.polyline([(0, 0, z_foot), (0, 0, ZB + 2 * D)]), radius=A_RADIUS, name="stub"
        )
    )
    model.add(
        geo.ThinWire(
            geo.Curve.polyline([(0, 0, ZB + 3 * D), (0, 0, ZB + H)]), radius=A_RADIUS, name="arm"
        )
    )
    return model


def _resonance(kind):
    if kind in _CACHE:
        return _CACHE[kind]
    # 38 nodes per wavelength at 8 GHz = the 1 mm cell, so the PML depth is
    # its nominal cell count and the grid is the same for all three kinds.
    control = MeshControl(min_nodes_per_wavelength=38, max_cell_size=D, min_cell_size=D)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no endpoint, no claimed-stencil warning
        mesh = Mesh.from_geometry(_model(kind), control, f_max=8e9)
    g = mesh.grid
    k_gap = int(np.argmin(np.abs(g.z - (ZB + 2 * D))))
    port = PortSpecLumped(
        name="feed",
        start=(0.0, 0.0, float(g.z[k_gap])),
        end=(0.0, 0.0, float(g.z[k_gap + 1])),
        Z0=Z0_FEED,
    )
    f_axis = np.linspace(3.2e9, 6.2e9, 121)
    ana = AnalysisScatteringTD(mesh=mesh, ports=[port], f_max=8e9, verbose=False)
    res = ana.run(f_axis=f_axis, excited=["feed"], total_time_steps=N_STEPS, energy_stop_db=None)
    s11 = res.S("feed", "feed")
    zin = Z0_FEED * (1 + s11) / (1 - s11)
    im = zin.imag
    idx = np.nonzero((im[:-1] < 0) & (im[1:] >= 0))[0]
    assert idx.size, f"{kind}: no Im Zin zero crossing in the scanned band"
    i = idx[0]
    f0, f1, y0, y1 = f_axis[i], f_axis[i + 1], im[i], im[i + 1]
    f_res = float(f0 - y0 * (f1 - f0) / (y1 - y0))
    r_res = float(np.interp(f_res, f_axis, zin.real))
    _CACHE[kind] = (f_res, r_res, (g.Nx, g.Ny, g.Nz), tuple(np.round(g.z, 9)))
    return _CACHE[kind]


def test_grids_are_identical():
    """The three plates mesh onto one grid — the comparison is the junction alone."""
    shapes = {k: _resonance(k)[2:] for k in THICKNESS}
    assert shapes["sheet"] == shapes["solid"] == shapes["sheet_thick"]
    assert np.any(np.isclose(shapes["sheet"][1], ZB)), "sheet plane is not a grid plane"
    assert not np.any(np.isclose(shapes["sheet"][1], ZB + THICKNESS["sheet"])), (
        "the metal top face re-entered the grid"
    )


def test_monopole_on_solid_is_a_monopole():
    """Plausibility anchor: the reference sits in the T4/T5 textbook window."""
    f_res, r_res, *_ = _resonance("solid")
    f_half = C0 / (2 * (2 * H))  # the equivalent dipole's half-wave frequency
    assert 0.44 * 2 * f_half <= f_res <= 0.50 * 2 * f_half, f"f_res = {f_res / 1e9:.3f} GHz"
    assert 25.0 <= r_res <= 50.0, f"R_in = {r_res:.1f} ohm"


@pytest.mark.parametrize(
    ("kind", "tol_f", "tol_r"),
    [
        ("sheet", 0.015, 0.06),  # measured +0.6 % / -2.2 % (2026-09-05)
        ("sheet_thick", 0.04, 0.10),  # measured +2.6 % / +5.7 %
    ],
)
def test_wire_on_sheet_matches_wire_on_solid(kind, tol_f, tol_r):
    f_ref, r_ref, *_ = _resonance("solid")
    f_res, r_res, *_ = _resonance(kind)
    assert abs(f_res - f_ref) <= tol_f * f_ref, (
        f"{kind}: f_res = {f_res / 1e9:.4f} GHz vs solid {f_ref / 1e9:.4f} GHz"
    )
    assert abs(r_res - r_ref) <= tol_r * r_ref, f"{kind}: R_in = {r_res:.2f} vs solid {r_ref:.2f}"
