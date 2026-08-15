"""Certificate: no section edge is lost between the kernel and the mesh.

Reproduces the measurements DD-168 cites.

``BRepBuilderAPI_MakeWire`` accepts an edge at a vertex that already
joins two, which makes a branched pseudo-wire; ``BRepTools_WireExplorer``
then walks one arm and stops, and the edges left inside are never
tessellated and never reported.  On a stripline coupler this halved the
cross-section of a lofted electrode over a 70 µm band without a single
warning: cells came out air where the solid is metal.

Three checks, each on the geometry that showed the defect:

1.  **Edge conservation.**  Every section edge the kernel produces ends
    up in exactly one traversal chain, on the planes where the old wire
    builder dropped half of them.

2.  **Continuity across the grazing band.**  The area the section
    returns varies smoothly with the plane.  Losing one of two
    contours halves it, which no smooth solid does over 10 µm.

3.  **What the mesh books.**  The coupler meshes with no open-chain
    warning, on an unchanged grid, and the conformal faces and PEC
    cells that the lost contours cost are back.

Run from the repository root::

    python validation/section_chain_completeness_certificate.py
"""

from __future__ import annotations

import warnings

import numpy as np

import magnelio as mio
from magnelio import geo, ports
from magnelio.constants import C0

# --- the coupler, at the parameters the defect was found on -----------
F0 = 0.5e9
F_MAX = 2e9
DIA = 50e-3
LEN_ADD = 1.5 * DIA
LL = C0 / (4 * F0)
W = 24e-3
T = 1e-3
H = 3e-3
RI = 1.52e-3
RA = 3.5e-3
L_COAX = 10e-3
GC = 1.5 * RA
G = 4 * RA
GT = 10e-3
ALPHA = W / DIA * 180 / np.pi
BETA = (W + 2 * GT) / DIA * 180 / np.pi

# Chordal budget and escape of the conformal pass on this model.
H_MIN = 2.5110044771209718e-04
DEFLECTION = H_MIN * 1e-2
NUDGE = H_MIN * 1e-1

# Planes through the electrode's lateral extreme (6.1810 mm), inside the
# near-tangency band where the wire builder branched.
GRAZING_X = (6.110e-3, 6.113e-3, 6.120e-3)
BAND_X = np.arange(6.050e-3, 6.1201e-3, 10e-6)


def _build():
    """The coupler assembly and its model."""
    pec = mio.Material.pec()
    air = mio.Material.air()

    vac = geo.Cylinder(
        radius=DIA / 2, origin=(0, 0, -LEN_ADD), axis="z", height=LL + 2 * LEN_ADD, material=air
    )
    pit = (
        geo.Face(
            normal="x",
            points=((0, -G), (0, LL + G), (DIA / 2 + T + H, LL + G), (DIA / 2 + T + H, -G)),
            material=pec,
        )
        .revolved(axis="z", angle_deg=BETA)
        .rotated(axis="z", angle_deg=-BETA / 2)
        .filleted(edges="all", radius=1e-3)
    )
    coax_vac = geo.Cylinder(
        origin=(0, 0, -GC), axis="y", height=DIA / 2 + L_COAX, radius=RA, material=air
    )
    coax_cond = geo.Cylinder(
        origin=(0, DIA / 2 + T + H, -GC), axis="y", height=-T - H + L_COAX, radius=RI, material=pec
    )
    electrodes = geo.Face(
        normal="x",
        points=((DIA / 2, 0), (DIA / 2, LL), (DIA / 2 + T, LL), (DIA / 2 + T, 0)),
        material=pec,
    ).revolved(axis="z", angle_deg=ALPHA)
    electrodes = electrodes.rotated(axis="z", angle_deg=-ALPHA / 2)
    transition = electrodes.lofted(
        (0, DIA / 2 + T / 2, 0),
        coax_cond,
        (0, DIA / 2 + T + H, -GC),
        material=pec,
        blend="tangent",
        tension=(0.8, 0.2),
    )
    vac += pit
    vac += coax_vac
    vac += coax_vac.mirrored(normal=(0, 0, 1), position=LL / 2)
    electrodes += transition
    electrodes += transition.mirrored(normal=(0, 0, 1), position=LL / 2)
    electrodes += coax_cond
    electrodes += coax_cond.mirrored(normal=(0, 0, 1), position=LL / 2)
    vac += vac.rotated(axis="z", angle_deg=180)
    electrodes += electrodes.rotated(axis="z", angle_deg=180)
    vac -= electrodes

    model = mio.GeometryModel(
        background=pec, boundary_conditions={"xmin": "SymmetryPMC", "ymin": "SymmetryPEC"}
    )
    model.add(vac)
    model.add(electrodes)
    r_port = 1.6 * RA
    model.add_port(
        ports.PortWaveguide(
            name="port1",
            plane="ymax",
            corners=((-r_port, None, -GC - r_port), (r_port, None, -GC + r_port)),
            n_modes=1,
        )
    )
    model.add_port(
        ports.PortWaveguide(
            name="port2",
            plane="ymax",
            corners=((-r_port, None, LL + GC - r_port), (r_port, None, LL + GC + r_port)),
            n_modes=1,
        )
    )
    return electrodes, model


def _section_edges(shape, axis, position):
    """The raw section edges, exactly as the polygon path collects them."""
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
    from OCC.Core.gp import gp_Dir, gp_Pln, gp_Pnt
    from OCC.Core.TopAbs import TopAbs_EDGE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods

    from magnelio.geo._occ_backend import keep_operands_intact

    normal = {"x": gp_Dir(1, 0, 0), "y": gp_Dir(0, 1, 0), "z": gp_Dir(0, 0, 1)}[axis]
    origin = [0.0, 0.0, 0.0]
    origin["xyz".index(axis)] = position
    section = BRepAlgoAPI_Section()
    section.SetRunParallel(True)
    keep_operands_intact(section)
    section.Init1(shape)
    section.Init2(gp_Pln(gp_Pnt(*origin), normal))
    section.Build()
    edges = []
    explorer = TopExp_Explorer(section.Shape(), TopAbs_EDGE)
    while explorer.More():
        edges.append(topods.Edge(explorer.Current()))
        explorer.Next()
    return edges


def check_edge_conservation(shape) -> bool:
    """Check 1 — every section edge reaches exactly one chain."""
    from magnelio.geo._occ_backend import _chain_section_edges

    print("\n1. Edge conservation on the grazing planes")
    print("   plane [mm]   raw edges   chained   chains")
    ok = True
    for position in GRAZING_X:
        edges = _section_edges(shape, "x", position)
        chains = _chain_section_edges(edges, 1, 2)
        members = [id(edge) for chain in chains for edge, _ in chain]
        conserved = len(members) == len(edges) and len(set(members)) == len(edges)
        ok &= conserved and bool(edges)
        print(
            f"   x={position * 1e3:8.4f}   {len(edges):9d}   {len(members):7d}   "
            f"{len(chains):6d}   {'ok' if conserved else 'LOST'}"
        )
    return ok


def check_band_continuity(shape) -> bool:
    """Check 2 — the section area varies smoothly across the band."""
    from magnelio.geo._occ_backend import cross_section_polygons
    from magnelio.geo._polygon_clip import polygon_area

    print("\n2. Continuity of the returned area across the tangency band")
    print("   plane [mm]   contours   area [mm2]   step vs. previous")
    areas = []
    worst = 0.0
    for position in BAND_X:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            polys = cross_section_polygons(shape, "x", float(position), DEFLECTION, nudge=0.0)
        area = sum(abs(polygon_area(p)) for p in polys)
        step = "" if not areas else f"{(area - areas[-1]) / areas[-1] * 100:+8.2f} %"
        if areas:
            worst = max(worst, abs(area - areas[-1]) / areas[-1])
        areas.append(area)
        n_warn = sum(1 for c in caught if "open section chain" in str(c.message))
        print(
            f"   x={position * 1e3:8.4f}   {len(polys):8d}   {area * 1e6:10.4f}   {step:>17s}"
            + ("   (warned)" if n_warn else "")
        )
    # A lost contour is a halving; a smooth solid over 10 um is far below.
    print(f"   largest step between neighbouring planes: {worst * 100:.2f} %  (limit 25 %)")
    return worst < 0.25


def check_mesh(model) -> bool:
    """Check 3 — what the coupler mesh books."""
    print("\n3. The coupler mesh")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mesh = mio.Mesh.from_geometry(
            model, control=mio.MeshControl(min_cell_size=T / 4), f_max=F_MAX
        )
    n_warn = sum(1 for c in caught if "open section chain" in str(c.message))

    nx, ny, nz = mesh.Nx, mesh.Ny, mesh.Nz
    n_hx = (nx + 1) * ny * nz
    faces = mesh.face_material
    cat_hx = faces.category[:n_hx]
    a_pec_hx = faces.A_face_pec[:n_hx]
    pec_ids = [k for k, v in mesh.material_library.items() if v.is_pec]
    pec_cells = int(np.isin(mesh.material_id, pec_ids).sum())

    print(f"   grid                       {nx} x {ny} x {nz}")
    print(f"   open-chain warnings        {n_warn}")
    print(f"   conformal Hx faces         {int((cat_hx == 2).sum())}")
    print(f"   PEC area booked on Hx      {float(np.nansum(a_pec_hx)) * 1e6:.4f} mm2")
    print(f"   cells classified PEC       {pec_cells}")
    # Measured before the rewrite: 11737 conformal Hx faces, 64904 PEC
    # cells, 117843.7094 mm2 booked -- the lost contours cost all three.
    return (
        n_warn == 0
        and int((cat_hx == 2).sum()) > 11737
        and pec_cells > 64904
        and float(np.nansum(a_pec_hx)) * 1e6 > 117843.7094
    )


def main() -> int:
    print("=" * 72)
    print("Section chain completeness certificate (DD-168)")
    print("=" * 72)
    electrodes, model = _build()
    shape = electrodes._occ_shape(1.0)

    results = {
        "edge conservation": check_edge_conservation(shape),
        "band continuity": check_band_continuity(shape),
        "mesh booking": check_mesh(model),
    }

    print("\n" + "=" * 72)
    for name, passed in results.items():
        print(f"  {name:22s} {'PASS' if passed else 'FAIL'}")
    verdict = all(results.values())
    print(f"\n  {'PASS' if verdict else 'FAIL'}")
    print("=" * 72)
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
