"""DD-157 certificate: open section chains never book fantasy coverage.

The smallest body reproducing the mutilated ``BRepAlgoAPI_Section``
edge set is the full stripline-coupler vacuum union (pipe + pit with
fillet + two coax bores + tangent-blend transitions, mirrored and
rotated, minus the electrode union) — every reduced variant sections
cleanly.  Before DD-157 the section at ``x = ri - deflection`` (the
DD-106 shifted side of the inner-conductor tangent plane) came back
as ONE open 13-point chain spanning both coax bores; implicitly
closed, it booked the bore-wall H face at 0.80 free instead of 0.19,
y-layer dependent, which broke the feed-chain slab invariance behind
the coax ports (defect 0.43 → modal Mur fallback).

Checks:

1. Section consistency on the offending plane: every contour closed,
   |signed coverage| of the probe rectangle equals the even-odd
   sampled coverage and the analytic bore-stripe fraction.
2. Mesh-level: the bore-wall ``M_mu(Hx)`` column is y-invariant and
   the declared-symmetry quarter model builds both coax ports on the
   exact DTBC (no feed-chain warning).
"""

import warnings

import numpy as np

import magnelio as mio
from magnelio import geo, ports
from magnelio.constants import C0
from magnelio.geo._occ_backend import cross_section_polygons
from magnelio.geo._polygon_clip import (
    clip_polygon_to_rect,
    point_in_polygon,
    polygon_area,
)


def build_coupler_bodies():
    fmax = 2e9
    dia = 50e-3
    len_add = 100e-3
    length = C0 / (4 * fmax)
    w = 24e-3
    t = 1e-3
    h = 3e-3
    g = w
    gt = 10e-3
    ri = 1.52e-3
    ra = 3.5e-3
    l_coax = 30e-3
    alpha = w / dia * 180 / np.pi
    beta = (w + 2 * gt) / dia * 180 / np.pi

    pec = mio.Material.pec()
    air = mio.Material.air()

    vac = geo.Cylinder(
        radius=dia / 2,
        origin=(0, 0, -len_add),
        axis="z",
        height=length + 2 * len_add,
        material=air,
    )
    pit = (
        geo.Face(
            normal="x",
            points=((0, -g), (0, length + g), (dia / 2 + t + h, length + g), (dia / 2 + t + h, -g)),
            material=pec,
        )
        .revolved(axis="z", angle_deg=beta)
        .rotated(axis="z", angle_deg=-beta / 2)
        .filleted(edges="all", radius=1e-3)
    )
    coax_vac = geo.Cylinder(
        origin=(0, 0, -w / 2), axis="y", height=dia / 2 + l_coax, radius=ra, material=air
    )
    coax_cond = geo.Cylinder(
        origin=(0, dia / 2 + t + h, -w / 2),
        axis="y",
        height=-t - h + l_coax,
        radius=ri,
        material=pec,
    )
    electrodes = geo.Face(
        normal="x",
        points=((dia / 2, 0), (dia / 2, length), (dia / 2 + t, length), (dia / 2 + t, 0)),
        material=pec,
    ).revolved(axis="z", angle_deg=alpha)
    electrodes = electrodes.rotated(axis="z", angle_deg=-alpha / 2)
    transition = electrodes.lofted(
        (0, dia / 2 + t / 2, 0),
        coax_cond,
        (0, dia / 2 + t + h, -w / 2),
        material=pec,
        blend="tangent",
        tension=(0.8, 0.2),
    )
    vac += pit
    vac += coax_vac
    vac += coax_vac.mirrored(normal=(0, 0, 1), position=length / 2)
    electrodes += transition
    electrodes += transition.mirrored(normal=(0, 0, 1), position=length / 2)
    electrodes += coax_cond
    electrodes += coax_cond.mirrored(normal=(0, 0, 1), position=length / 2)
    vac += vac.rotated(axis="z", angle_deg=180)
    electrodes += electrodes.rotated(axis="z", angle_deg=180)
    vac -= electrodes
    return [vac, electrodes], dict(fmax=fmax, w=w, t=t, ri=ri, ra=ra, length=length)


def check_section_consistency(vac, p):
    ri, ra, w = p["ri"], p["ra"], p["w"]
    delta = 2.5e-6
    rect = (51.27e-3, -8.93e-3, 55.0e-3, -8.50e-3)
    area = (rect[2] - rect[0]) * (rect[3] - rect[1])
    polys = cross_section_polygons(vac._occ_shape(1.0), "x", ri - delta, deflection=delta)
    for c in polys:
        gap = float(np.hypot(*(c[0] - c[-1])))
        seg = float(np.hypot(*(c[1:] - c[:-1]).T).sum())
        assert gap <= max(8 * delta, 5e-2 * (seg + gap)), "open contour returned"
    signed = sum(
        polygon_area(c) for c in (clip_polygon_to_rect(q, rect) for q in polys) if len(c) >= 3
    )
    uu, vv = np.meshgrid(np.linspace(rect[0], rect[2], 120), np.linspace(rect[1], rect[3], 120))
    par = np.zeros(uu.size, dtype=int)
    for q in polys:
        par += np.fromiter(
            (point_in_polygon((x, y), q) for x, y in zip(uu.ravel(), vv.ravel())),
            dtype=int,
            count=uu.size,
        )
    even_odd = float(np.mean(par % 2 == 1))
    x = ri - delta
    half = np.sqrt(ra * ra - x * x)
    lo, hi = max(rect[1], -w / 2 - half), min(rect[3], -w / 2 + half)
    frac_an = max(0.0, hi - lo) / (rect[3] - rect[1])
    print(
        f"section coverage: |signed| {abs(signed) / area:.4f}  "
        f"even-odd {even_odd:.4f}  analytic {frac_an:.4f}"
    )
    assert abs(abs(signed) / area - even_odd) < 0.02
    assert abs(even_odd - frac_an) < 0.05


def check_ports_exact_dtbc(bodies, p):
    model = mio.GeometryModel(
        background=mio.Material.pec(),
        boundary_conditions={
            "xmin": "SymmetryPMC",
            "ymin": "SymmetryPEC",
        },
    )
    for b in bodies:
        model.add(b)
    w, t, ra = p["w"], p["t"], p["ra"]
    r_port = 1.6 * ra
    model.add_port(
        ports.PortWaveguide(
            name="port1",
            plane="ymax",
            corners=((-r_port, None, -w / 2 - r_port), (r_port, None, -w / 2 + r_port)),
            n_modes=1,
        )
    )
    model.add_port(
        ports.PortWaveguide(
            name="port2",
            plane="ymax",
            corners=(
                (-r_port, None, p["length"] + w / 2 - r_port),
                (r_port, None, p["length"] + w / 2 + r_port),
            ),
            n_modes=1,
        )
    )
    mesh = mio.Mesh.from_geometry(
        model, control=mio.MeshControl(min_cell_size=t / 4), f_max=p["fmax"]
    )
    analysis = mio.AnalysisScatteringTD(mesh=mesh, f_max=p["fmax"], verbose=False)
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        reports = analysis.solve_ports()
    feed_warnings = [w_ for w_ in wlist if "feed-chain mass slabs" in str(w_.message)]
    for name, rep in reports.items():
        print(rep)
    assert not feed_warnings, "coax ports fell back to Mur — slab invariance broken"
    print("both coax ports on the exact DTBC — certificate PASSED")


if __name__ == "__main__":
    bodies, p = build_coupler_bodies()
    check_section_consistency(bodies[0], p)
    check_ports_exact_dtbc(bodies, p)
