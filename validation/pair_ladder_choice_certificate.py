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

Measured 2026-08-15 at ``ebf7c89`` (weighted pair spread of the single
TEM channel; the DTBC pair gate admits 1e-8):

    port          axis-order choice        conditioning choice
    port1 (orig)      7.2e-15                   7.1e-15
    port2 (mirror)    1.7e-08  -> Mur           6.3e-14  -> DTBC

Re-measured 2026-09-01 (KB-039): the fixture stopped certifying on
either stage, and was repaired in the mesher on the same day (the
paragraph after the controls).  The transparent-boundary certificate
has two gates, and the one that runs first is not the one this file is
about; both are quoted here at their broken values because the
bisection below is what found the mesher defect:

    stage 2  ``_port_chain_slab_defect`` -- the masses feeding the 2D
             mode solve must continue unchanged into the first feed
             cells.  Both ports read 8.4165e-02 against the 1e-8 gate,
             so every channel falls back to modal Mur-1st and stage 1
             is never evaluated; the operator then reports its pair
             spread as ``None``, by design.
    stage 1  the pair-ladder spread this certificate is about.  It is
             recomputed below from the operator's own port masses so
             the table stays complete either way, and it fails too:
             1.1902e-02 (port1) and 2.1978e-02 (port2), worst single
             edge 1.322e-01 / 3.192e-01, against a pair gate that is
             2e-6 since DD-229.  The port cross-section itself is no
             longer uniform, four orders past that gate and twelve
             past the 2026-08-15 pins.

The stage-2 defect is a drift in what the fixture *builds*, not in what
the gate *measures*: ``_port_chain_slab_defect``, its call site and the
1e-8 tolerance are byte-identical to the initial public tree, while the
mesh built from this unchanged script moved underneath them.  Bisected
2026-09-01 over the first-parent history, reading the slab defect the
factory itself measures (port-plane masses flattened first), port1:

    ebf7c89 (2026-08-15)  6.6135e-11  DTBC   the pins above
    6ca4049 (DD-191)      6.6135e-11  DTBC   grid 94 -> 103 cells in z
    a188229 (DD-192/193)  2.7740e-10  DTBC
    67707a8 (DD-198)      2.7740e-10  DTBC
    955bc97 (2026-08-26)  2.7740e-10  DTBC   last good, stage 1 at
                                             6.98e-15 / 1.07e-13
    acd8417 (DD-199)      8.4165e-02  Mur    facet section engine
    HEAD    (2026-09-01)  8.4165e-02  Mur

``955bc97`` is the first parent of the ``acd8417`` merge, so the flip is
that merge and nothing between.

The trigger is the tangent-blend loft between the electrode and the
coaxial inner conductor: dropping that one body, and nothing else,
restores the exact termination.  DD-199 is the merge that put free-form
faces on a lifted triangulation and a B-spline blend is the union's
free-form body, which fits -- but which face the engine actually treats
differently was not traced.  Controls measured 2026-09-01 on the same
tree, slab defect as the factory sees it and the operator's own pair
spreads:

    fixture as built          8.4165e-02  Mur / Mur
    loft dropped              1.4784e-10  DTBC  5.7e-15 / 8.9e-14
    coax stubs only           4.7536e-11  DTBC  4.8e-15 / 4.9e-14
    coax 10 mm longer, so
      the loft sits further
      from the port           4.2646e-01  Mur / Mur
    loft as its own body,
      not fused into the
      electrode               8.4165e-02  Mur / Mur

The longer coax also regrids (30 x 35 x 103), so it is not a controlled
change of distance alone -- but it does not help, and neither does
keeping the fuse out of it.  The cells the gate compares hold no loft
material at all: they sit above y = 31.3 mm while the loft ends at
y = 29 mm, and the entries that deviate lie inside the two coaxial
bores.

Resolved 2026-09-01 in the section engine, with the fixture untouched.
A shape with one free-form face is represented by its triangulation as
a whole, so the loft took the coaxial cylinders off their own geometry
too, and a triangulated cylinder cuts differently depending on where
the plane falls between its node rows -- which is exactly the
uniformity the exact termination consumes.  Cylindrical faces are now
answered from the cylinder itself on that path, and the fixture reads

    both ports  dtbc  slab defect 1.4795e-10, pair spreads
                      5.2e-15 (port1) and 1.3e-13 (port2)

against the 1.4784e-10 / 5.7e-15 / 8.9e-14 of the loft-dropped control
and the 7.1e-15 / 6.3e-14 pinned on 2026-08-15.  The original/mirror
asymmetry the certificate exists for is back at 25x.

Run from ``magnelio/`` (about 20 s, 15 s of it meshing):

    CUPY_ACCELERATORS= python validation/pair_ladder_choice_certificate.py
"""

from __future__ import annotations

import math
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
# The gate was 1e-8 when this fixture was pinned and is 2e-6 since
# DD-229 raised it out of the reflection budget
# (ports/_modal/operator.py::_DTBC_PAIR_SPREAD_TOL).
GATE = 2e-6
FLOOR = 1e-12
# Certificate stage 2 (DD-067), the gate that runs before the pairing
# one: ports/_modal/operator.py::_DTBC_SLAB_DEFECT_TOL.
SLAB_GATE = 1e-8

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
    # The only free-form surface in the union, and the one the DD-199
    # facet section path costs the feed its z-invariance through
    # (KB-039).  Dropping it restores the exact termination.
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


def _stage_one(op, dm) -> tuple[float | None, float | None]:
    """Recompute the pair-ladder statistics from the operator's masses.

    The operator reports its own spread only when the pairing gate
    actually ran; the earlier chain-slab gate withholds it (``None``)
    without ever looking at the pairs.  Recomputing here — the same
    modal-weighted RMS the operator forms — keeps the table complete
    in both cases.

    Returns ``(spread, worst)``: the weighted RMS deviation of the
    per-pair chain ratio relative to its weighted mean, and the worst
    single edge.  ``(None, None)`` when no pair carries weight or a
    pair product is non-positive, i.e. when the statistic itself is
    undefined rather than merely withheld.
    """
    pair = np.concatenate([op._me_u_port * op._mh_v, op._me_v_port * op._mh_u])
    weight = np.concatenate([op._me_u_port * dm.e_u_profile**2, op._me_v_port * dm.e_v_profile**2])
    w_max = float(weight.max()) if weight.size else 0.0
    if w_max <= 0.0:
        return None, None
    active = weight > 1e-12 * w_max
    if not np.any(active) or np.any(pair[active] <= 0.0):
        return None, None
    r_pairs = op._dt / np.sqrt(pair[active])
    w = weight[active]
    r_mean = float(np.dot(w, r_pairs) / w.sum())
    spread = float(math.sqrt(np.dot(w, (r_pairs - r_mean) ** 2) / w.sum()) / r_mean)
    worst = float(np.abs((r_pairs - r_mean) / r_mean).max())
    return spread, worst


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
        spread_local, worst = _stage_one(op, dm)
        reported = op._dtbc_pair_spread[0]
        rows.append(
            {
                "name": spec.name,
                "kinds": list(op.termination_kinds),
                # What the operator itself decided on: None whenever a
                # gate withheld the exact boundary before the pairing
                # one ran.
                "spread": None if reported is None else float(reported),
                # The same statistic, recomputed here, always defined.
                "spread_local": spread_local,
                "worst": worst,
                "slab_defect": op._chain_slab_defect,
                "z_line": op.port_report.z_line_num,
            }
        )
    return rows


def _fmt(value: float | None, spec: str, withheld: str = "withheld") -> str:
    """Right-align ``value`` in ``spec``'s width, or a word in its place."""
    head = spec.split(".")[0]
    width = int(head) if head else 0
    return f"{withheld:>{width}s}" if value is None else format(value, spec)


def main() -> None:
    mesh = coupler_mesh()
    print(f"coupler mesh {mesh.Nx} x {mesh.Ny} x {mesh.Nz}")
    print(
        f"{'port':8s} {'termination':12s} {'pair spread':>12s} {'stage 1 here':>12s} "
        f"{'worst edge':>12s} {'slab defect':>12s} {'z_line [Ω]':>12s}"
    )
    rows = measure(mesh)
    for r in rows:
        print(
            f"{r['name']:8s} {','.join(r['kinds']):12s} {_fmt(r['spread'], '12.4e')} "
            f"{_fmt(r['spread_local'], '12.4e', 'undefined')} "
            f"{_fmt(r['worst'], '12.3e', 'undefined')} "
            f"{_fmt(r['slab_defect'], '12.4e', 'unmeasured')} {r['z_line']:12.4f}"
        )
    print(
        "  pair spread: what the operator decided on — 'withheld' when an "
        "earlier gate\n  stopped it.  stage 1 here: the same statistic "
        "recomputed from the operator's\n  own port masses, so the table "
        "carries a number either way."
    )
    for r in rows:
        if r["spread"] is None:
            print(
                f"{r['name']}: the exact boundary was withheld before the pair "
                f"gate ran — feed-chain slab defect "
                f"{_fmt(r['slab_defect'], '.4e', 'unmeasured')} against the "
                f"{SLAB_GATE:.0e} gate, so every channel runs modal Mur-1st."
            )

    for r in rows:
        assert r["kinds"] == ["dtbc"], (
            f"{r['name']}: termination {r['kinds']} — its channel lost the exact "
            f"discrete boundary (pair spread {_fmt(r['spread'], '.3e')}, "
            f"recomputed {_fmt(r['spread_local'], '.3e', 'undefined')} against the "
            f"{GATE:.0e} gate; feed-chain slab defect "
            f"{_fmt(r['slab_defect'], '.3e', 'unmeasured')} against the "
            f"{SLAB_GATE:.0e} one.  Pinned 2026-08-15: 7.1e-15 / 6.3e-14, both "
            f"on the exact boundary)"
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
