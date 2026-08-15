"""Certificate: a mirrored copy keeps its port's exact termination.

``couple_face_material_pairs`` offers every H face two candidate ladders
and defines the face mass through the pair identity when they agree.
Agreement is tested at a relative ``rtol``, so two agreeing ladders can
still differ by up to that much -- and the DTBC uniform-chain gate that
consumes the result is two orders tighter.  Anything landing in that
band produces a certified-looking mass that still costs the port its
exact termination.

This certificate pins the choice: of two valid, agreeing ladders the
one whose own two partners disagree less supplies the target.

The fixture is a stripline coupler fed by two coaxial stubs, the second
of which is a *mirrored copy* of the first.  Mirroring inflates the
tolerance of the unioned solid, and the resulting section jitter lands
squarely in the band above.  Both stubs are otherwise identical, so the
port on the original is the control: it certifies either way, and any
difference between the two ports is the defect.

Reducing the model does not reproduce it -- the same lesson as the
section-contour certificate next to this one: a small mirrored coax
sections cleanly and both of its ports certify.  The full union is the
fixture.

Measured (weighted pair spread of the single TEM channel; the DTBC
gate admits 1e-8):

    port          axis-order choice        conditioning choice
    port1 (orig)      7.2e-15                   7.1e-15
    port2 (mirror)    1.7e-08  -> Mur           6.3e-14  -> DTBC

Run from ``magnelio/`` (about two minutes, most of it meshing):

    CUPY_ACCELERATORS= python validation/pair_ladder_choice_certificate.py
"""

from __future__ import annotations

import sys
import warnings

import numpy as np

import magnelio as mio
from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.constants import C0
from magnelio.geo import Cylinder, Face
from magnelio.ports import PortWaveguide
from magnelio.solver.stability import spectral_dt

F0 = 1e9
F_MAX = 2e9
DIA = 50e-3
LEN_ADD = 1.5 * DIA
ELL = C0 / (4 * F0)
W = 12e-3
T = 1e-3
H = 3e-3
GT = 10e-3
RI = 1.52e-3
RA = 3.5e-3
L_COAX = 10e-3

# The gate the pairing feeds, and the floor the control port holds.
GATE = 1e-8
FLOOR = 1e-12

PEC = mio.Material.pec()
AIR = mio.Material.air()


def coupler_mesh() -> mio.Mesh:
    """The full union — every reduced variant of it certifies anyway."""
    alpha = W / DIA * 180 / np.pi
    beta = (W + 2 * GT) / DIA * 180 / np.pi

    vac = Cylinder(
        radius=DIA / 2, origin=(0, 0, -LEN_ADD), axis="z", height=ELL + 2 * LEN_ADD, material=AIR
    )
    pit = (
        Face(
            normal="x",
            points=((0, -W), (0, ELL + W), (DIA / 2 + T + H, ELL + W), (DIA / 2 + T + H, -W)),
            material=PEC,
        )
        .revolved(axis="z", angle_deg=beta)
        .rotated(axis="z", angle_deg=-beta / 2)
        .filleted(edges="all", radius=1e-3)
    )
    coax_vac = Cylinder(
        origin=(0, 0, -W / 2), axis="y", height=DIA / 2 + L_COAX, radius=RA, material=AIR
    )
    coax_cond = Cylinder(
        origin=(0, DIA / 2 + T + H, -W / 2),
        axis="y",
        height=-T - H + L_COAX,
        radius=RI,
        material=PEC,
    )
    electrodes = Face(
        normal="x",
        points=((DIA / 2, 0), (DIA / 2, ELL), (DIA / 2 + T, ELL), (DIA / 2 + T, 0)),
        material=PEC,
    ).revolved(axis="z", angle_deg=alpha)
    electrodes = electrodes.rotated(axis="z", angle_deg=-alpha / 2)
    transition = electrodes.lofted(
        (0, DIA / 2 + T / 2, 0),
        coax_cond,
        (0, DIA / 2 + T + H, -W / 2),
        material=PEC,
        blend="tangent",
        tension=(0.8, 0.2),
    )
    vac += pit
    vac += coax_vac
    vac += coax_vac.mirrored(normal=(0, 0, 1), position=ELL / 2)
    electrodes += transition
    electrodes += transition.mirrored(normal=(0, 0, 1), position=ELL / 2)
    electrodes += coax_cond
    electrodes += coax_cond.mirrored(normal=(0, 0, 1), position=ELL / 2)
    vac += vac.rotated(axis="z", angle_deg=180)
    electrodes += electrodes.rotated(axis="z", angle_deg=180)
    vac -= electrodes

    model = mio.GeometryModel(
        background=PEC,
        boundary_conditions={"xmin": "SymmetryPMC", "ymin": "SymmetryPEC"},
    )
    for body in (vac, electrodes):
        model.add(body)
    r_port = 1.6 * RA
    model.add_port(
        PortWaveguide(
            name="port1",
            plane="ymax",
            corners=((-r_port, None, -W / 2 - r_port), (r_port, None, -W / 2 + r_port)),
            n_modes=1,
        )
    )
    model.add_port(
        PortWaveguide(
            name="port2",
            plane="ymax",
            corners=(
                (-r_port, None, ELL + W / 2 - r_port),
                (r_port, None, ELL + W / 2 + r_port),
            ),
            n_modes=1,
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        control = mio.MeshControl(min_cell_size=T / 4)
        return mio.Mesh.from_geometry(model, control=control, f_max=F_MAX)


def measure(mesh: mio.Mesh) -> list[dict]:
    analysis = mio.AnalysisScatteringTD(mesh=mesh, f_max=F_MAX, verbose=False, backend="numpy")
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    dt = spectral_dt(mesh, "normal", m_eps=m_eps, m_mu=m_mu)
    rows = []
    for spec in analysis.ports:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            op = analysis._build_operator(spec, m_eps, m_mu, dt, analysis.f_max)
        dm = op.discrete_modes[0]
        pair = np.concatenate([op._me_u_port * op._mh_v, op._me_v_port * op._mh_u])
        weight = np.concatenate(
            [op._me_u_port * dm.e_u_profile**2, op._me_v_port * dm.e_v_profile**2]
        )
        active = weight > 1e-12 * weight.max()
        r_pairs = dt / np.sqrt(pair[active])
        w = weight[active]
        r_mean = float(np.dot(w, r_pairs) / w.sum())
        rows.append(
            {
                "name": spec.name,
                "kinds": list(op.termination_kinds),
                "spread": float(op._dtbc_pair_spread[0]),
                "worst": float(np.abs((r_pairs - r_mean) / r_mean).max()),
                "z_line": op.port_report.z_line_num,
            }
        )
    return rows


def main() -> None:
    mesh = coupler_mesh()
    print(f"coupler mesh {mesh.Nx} x {mesh.Ny} x {mesh.Nz}")
    print(
        f"{'port':8s} {'termination':13s} {'pair spread':>12s} "
        f"{'worst edge':>12s} {'z_line [Ω]':>12s}"
    )
    rows = measure(mesh)
    for r in rows:
        print(
            f"{r['name']:8s} {','.join(r['kinds']):13s} {r['spread']:12.4e} "
            f"{r['worst']:12.3e} {r['z_line']:12.4f}"
        )

    for r in rows:
        assert r["kinds"] == ["dtbc"], (
            f"{r['name']}: termination {r['kinds']} — its channel lost the exact "
            f"discrete boundary (pair spread {r['spread']:.3e} against the {GATE:.0e} gate; "
            f"axis-order choice read 1.7e-08 on port2)"
        )
        assert r["spread"] < FLOOR, (
            f"{r['name']}: pair spread {r['spread']:.3e} exceeds {FLOOR:.0e} — "
            f"certified, but no longer at the floor the control port holds"
        )
    # The two stubs are geometrically identical, so their line impedances
    # must agree; a difference would mean the fixture stopped being a
    # controlled pair and the comparison above proves nothing.
    z1, z2 = rows[0]["z_line"], rows[1]["z_line"]
    assert abs(z2 / z1 - 1.0) < 1e-9, f"the two stubs no longer match: {z1} vs {z2}"
    print("CERTIFICATE PASSED")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
