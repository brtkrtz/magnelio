"""DD-167 certificate: the degeneracy escape has a length of its own.

The DD-157 retry steps off a degenerate section plane to find a clean
one.  Its step length used to be the tessellation chord deflection,
which is a chordal-accuracy budget and nothing else.  The two mesh
passes deliberately tessellate an order apart — cell classification
needs point-in-polygon fidelity, the conformal-area sites integrate
over the polygon — so the finer pass inherited an escape reach ten
times shorter, and could no longer leave near-tangency bands the
coarser pass cleared without trouble.  The two then disagreed about
where the material is.

The fixture is the stripline-coupler assembly of DD-157, meshed at
``min_cell_size = t/4``.  A grid line is anchored on the electrode's
lateral extreme (x = 6.1804 mm), so the neighbouring cell-centre plane
at 6.0124 mm grazes the electrode's own side face by construction; the
tangency band there is 0.238 mm wide, against an escape reach that had
shrunk to 20 µm.  Refining the mesh moves that cell centre *closer* to
the extreme, so the collision is systematic, not a coincidence of one
mesh.

Checks:

1. Section-level A/B on the offending plane: escape tied to the chord
   warns and returns no contour for the conductor union; escape at a
   tenth of the cell returns the sliver.
2. Both mesh passes escape by the same distance, so the cells and the
   material matrices cannot disagree.
3. Mesh-level: the full coupler meshes without a single open-chain
   warning, and the conformal PEC area on the previously collapsing
   H-face plane is no longer a hole between its neighbours.

Measured (2026-08-15, 33 x 32 x 108 = 114048 cells):

    open-chain warnings                    4  ->  0
    electrodes section at x=6.0124 mm      0.00  ->  43.81 mm^2
    electrodes section at y=25.303 mm      0.00  ->  1744.02 mm^2
    conformal H faces on y=25.303 mm     378  ->  942
    A_face_pec on y=25.303 mm         5357.7  ->  6183.1 mm^2
      (neighbours 6293.7 / 5933.8: a 12.4 % dip became -1.1 %)
    cell classification                  unchanged (64904 PEC cells)

Internal record: `investigations/section-open-chains/MEASUREMENTS.md`.
"""

import warnings

import numpy as np

import magnelio as mio
from magnelio import geo
from magnelio.constants import C0
from magnelio.geo._filling import (
    CLASSIFY_DEFLECTION_FRACTION,
    SECTION_DEFLECTION_FRACTION,
    SECTION_NUDGE_FRACTION,
)
from magnelio.geo._occ_backend import cross_section_polygons
from magnelio.geo._polygon_clip import polygon_area

# The plane the mesher lands on, and the mesh it lands on it with.
PLANE_X = 6.0124254e-3
PLANE_Y = 25.30303e-3
H_MIN = 2.5110044771209718e-4
MAX_DIP = 0.05  # the electrode area varies smoothly across this plane


def build_model():
    """The coupler of DD-157, declared on its two symmetry planes."""
    from magnelio import ports

    dia = 50e-3
    len_add = 1.5 * dia
    length = C0 / (4 * 0.5e9)
    w = 24e-3
    t = 1e-3
    h = 3e-3
    ri = 1.52e-3
    ra = 3.5e-3
    l_coax = 10e-3
    gc = 1.5 * ra
    g = 4 * ra
    gt = 10e-3
    alpha = w / dia * 180 / np.pi
    beta = (w + 2 * gt) / dia * 180 / np.pi

    pec = mio.Material.pec()
    air = mio.Material.air()

    vac = geo.Cylinder(
        radius=dia / 2, origin=(0, 0, -len_add), axis="z", height=length + 2 * len_add, material=air
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
        origin=(0, 0, -gc), axis="y", height=dia / 2 + l_coax, radius=ra, material=air
    )
    coax_cond = geo.Cylinder(
        origin=(0, dia / 2 + t + h, -gc), axis="y", height=-t - h + l_coax, radius=ri, material=pec
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
        (0, dia / 2 + t + h, -gc),
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

    model = mio.GeometryModel(
        background=pec, boundary_conditions={"xmin": "SymmetryPMC", "ymin": "SymmetryPEC"}
    )
    for body in (vac, electrodes):
        model.add(body)
    r_port = 1.6 * ra
    for name, z0 in (("port1", -gc), ("port2", length + gc)):
        model.add_port(
            ports.PortWaveguide(
                name=name,
                plane="ymax",
                corners=((-r_port, None, z0 - r_port), (r_port, None, z0 + r_port)),
                n_modes=1,
            )
        )
    return model, (vac, electrodes), t


def area(polys):
    return sum(abs(polygon_area(p)) for p in polys)


def check_section_ab(conductor):
    """1. Same plane, same tessellation, two escape lengths."""
    print("\n[1] section A/B on the degenerate planes")
    defl = H_MIN * SECTION_DEFLECTION_FRACTION
    ok = True
    for axis, pos in (("x", PLANE_X), ("y", PLANE_Y)):
        shape = conductor._occ_shape(1.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tied = cross_section_polygons(shape, axis, pos, deflection=defl, nudge=defl)
        own = cross_section_polygons(
            shape, axis, pos, deflection=defl, nudge=H_MIN * SECTION_NUDGE_FRACTION
        )
        warned = any("open section chain" in str(c.message) for c in caught)
        print(
            f"    {axis}={pos * 1e3:8.4f} mm  escape=chord: {len(tied):2d} contour(s), "
            f"{area(tied) * 1e6:9.4f} mm^2, warned={warned}   "
            f"escape=h/10: {len(own):2d} contour(s), {area(own) * 1e6:9.4f} mm^2"
        )
        ok &= warned and not tied and bool(own) and area(own) > 0.0
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_shared_escape():
    """2. The two mesh passes must escape by the same distance."""
    print("\n[2] escape shared between the cell and the material pass")
    classify = H_MIN * CLASSIFY_DEFLECTION_FRACTION
    conformal = H_MIN * SECTION_DEFLECTION_FRACTION
    escape = H_MIN * SECTION_NUDGE_FRACTION
    print(f"    chordal budget   classification {classify:.4g} m   conformal {conformal:.4g} m")
    print(f"    escape step      both {escape:.4g} m  (reach {8 * escape / H_MIN:.2f} cells)")
    ok = classify != conformal and 8.0 * escape < H_MIN
    print(f"    -> {'PASS' if ok else 'FAIL'} (different chords, one escape, reach under a cell)")
    return ok


def check_mesh(model, t):
    """3. The coupler meshes clean and the H-face plane is no longer a hole."""
    print("\n[3] full coupler mesh")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mesh = mio.Mesh.from_geometry(
            model, control=mio.MeshControl(min_cell_size=t / 4), f_max=2e9
        )
    n_open = sum(1 for c in caught if "open section chain" in str(c.message))
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    print(f"    mesh {Nx} x {Ny} x {Nz} = {Nx * Ny * Nz}, open-chain warnings = {n_open}")

    n_Hx = (Nx + 1) * Ny * Nz
    hy = mesh.face_material.A_face_pec[n_Hx : n_Hx + Nx * (Ny + 1) * Nz].reshape(Nx, Ny + 1, Nz)
    cat = mesh.face_material.category[n_Hx : n_Hx + Nx * (Ny + 1) * Nz].reshape(Nx, Ny + 1, Nz)
    j0 = int(np.argmin(np.abs(mesh.grid.y - PLANE_Y)))
    a = [float(np.nansum(hy[:, j])) * 1e6 for j in (j0 - 1, j0, j0 + 1)]
    dip = 1.0 - a[1] / (0.5 * (a[0] + a[2]))
    print(
        f"    A_face_pec on y-planes {j0 - 1}/{j0}/{j0 + 1}: "
        f"{a[0]:.1f} / {a[1]:.1f} / {a[2]:.1f} mm^2   dip = {dip * 100:+.2f} %"
    )
    print(f"    conformal H faces on the plane: {int((cat[:, j0] == 2).sum())}")
    ok = n_open == 0 and dip < MAX_DIP
    print(f"    -> {'PASS' if ok else 'FAIL'} (no warning, dip below {MAX_DIP * 100:.0f} %)")
    return ok


def main():
    model, (_vac, conductor), t = build_model()
    results = [
        check_section_ab(conductor),
        check_shared_escape(),
        check_mesh(model, t),
    ]
    print("\n" + "=" * 64)
    print("DD-167 CERTIFICATE:", "PASS" if all(results) else "FAIL")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
