"""Carrier lines of the edge pass: compiled candidates, in-house planar hits.

Every carrier line of a structured grid is axis-aligned and meets a
planar axis-aligned face in one exactly known point.  The compiled
candidate search must reproduce the general slab test, and the in-house
hit test must report what the kernel's intersector reports — with the
geometric ``ON`` where the kernel's classifier loses a boundary point.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio import geo
from magnelio.geo import _occ_backend as backend
from magnelio.geo._line_kernels import axis_line_candidates, planar_point_state

pytest.importorskip("OCC")

MM = 1e-3
TOL = 1e-8


def _comb(teeth: int):
    parts = [geo.Brick(origin=(0, 0, 0), size=(teeth * 2 * MM, MM, 0.1 * MM), material="pec")]
    parts += [
        geo.Brick(origin=(i * 2 * MM, MM, 0), size=(MM, 3 * MM, 0.1 * MM), material="pec")
        for i in range(teeth)
    ]
    return geo.Union(*parts, material="pec")


def _grid_edges(x, y, z, axis):
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    start = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    step = np.zeros(3)
    step[axis] = (x[1] - x[0], y[1] - y[0], z[1] - z[0])[axis]
    return np.stack([start, start + step], axis=1)


def _general_candidates(flo, fhi, tol, p0, direction):
    """The general slab test of ``_line_candidates``, as a reference."""
    w_in = np.full(len(flo), -np.inf)
    w_out = np.full(len(flo), np.inf)
    for ax in range(3):
        lo = flo[:, ax] - tol
        hi = fhi[:, ax] + tol
        d = direction[ax]
        if abs(d) > 1e-300:
            t1 = (lo - p0[ax]) / d
            t2 = (hi - p0[ax]) / d
            np.maximum(w_in, np.minimum(t1, t2), out=w_in)
            np.minimum(w_out, np.maximum(t1, t2), out=w_out)
        else:
            inside = (lo <= p0[ax]) & (p0[ax] <= hi)
            w_in[~inside] = np.inf
            w_out[~inside] = -np.inf
    keep = np.nonzero(w_in <= w_out)[0]
    order = np.argsort(w_in[keep], kind="stable")
    idx = keep[order]
    return idx, w_in[idx], w_out[idx]


def test_compiled_candidates_reproduce_the_general_slab_test():
    rng = np.random.default_rng(3)
    lo = rng.uniform(0, 1, size=(400, 3))
    hi = lo + rng.uniform(0, 0.3, size=(400, 3))
    lo[:50, 0] = 0.25  # boxes sharing one slab bound, for the tie order
    for _ in range(60):
        ax = int(rng.integers(3))
        d = float(rng.choice([-1.0, 1.0]))
        p0 = rng.uniform(-0.1, 1.1, size=3)
        if rng.random() < 0.3:
            p0[(ax + 1) % 3] = 0.25 - 1e-9  # a tolerance-wide coincidence
        direction = np.zeros(3)
        direction[ax] = d
        got = axis_line_candidates(lo, hi, 1e-9, p0, ax, d)
        ref = _general_candidates(lo, hi, 1e-9, p0, direction)
        assert np.array_equal(got[0], ref[0])
        assert np.array_equal(got[1], ref[1])
        assert np.array_equal(got[2], ref[2])


def test_planar_point_state_with_a_hole_and_a_tolerance_band():
    outer = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
    hole = np.array([[1.0, 1.0], [1.0, 2.0], [2.0, 2.0], [2.0, 1.0]])
    verts = np.vstack([outer, hole])
    offsets = np.array([0, 4, 8])
    state = lambda u, v: planar_point_state(u, v, verts, offsets, 1e-6)  # noqa: E731
    assert state(3.0, 3.0) == 1
    assert state(1.5, 1.5) == 0  # in the hole
    assert state(5.0, 1.0) == 0
    assert state(-1e-7, 2.0) == 2  # within tolerance of the outline
    assert state(1.0 + 5e-7, 1.5) == 2  # within tolerance of the hole
    assert state(2.0 + 5e-6, 1.5) == 1  # beyond the band, in the material
    assert state(4.0, 4.0) == 2  # a corner


def test_planar_row_of_faces():
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods

    def faces(shape):
        out = []
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            out.append(topods.Face(exp.Current()))
            exp.Next()
        return out

    brick = geo.Brick(origin=(0, 0, 0), size=(2 * MM, MM, 0.5 * MM))._occ_shape(1.0)
    rows = [backend._planar_row(f) for f in faces(brick)]
    assert all(r is not None for r in rows)
    top = [r for r in rows if r[0] == 2 and r[1] == pytest.approx(0.5 * MM)]
    assert len(top) == 1 and top[0][2] == 1.0 and top[0][4].tolist() == [0, 4]
    bottom = [r for r in rows if r[0] == 2 and r[1] == 0.0]
    assert bottom[0][2] == -1.0

    plate = geo.Difference(
        geo.Brick(origin=(0, 0, 0), size=(4 * MM, 4 * MM, 0.2 * MM)),
        geo.Cylinder(origin=(2 * MM, 2 * MM, -MM), radius=MM, height=3 * MM, axis="z"),
    )._occ_shape(1.0)
    rows = [backend._planar_row(f) for f in faces(plate)]
    # The bore, and the two faces whose outline carries its circle.
    assert sum(r is None for r in rows) == 3
    assert all(r[0] in (0, 1) for r in rows if r is not None)  # the four straight sides

    cyl = geo.Cylinder(origin=(0, 0, 0), radius=MM, height=MM, axis="z")._occ_shape(1.0)
    assert all(backend._planar_row(f) is None for f in faces(cyl))  # arcs everywhere


def _line(p0, ax, d=1.0):
    from OCC.Core.gp import gp_Dir, gp_Lin, gp_Pnt

    dvec = np.zeros(3)
    dvec[ax] = d
    return gp_Lin(gp_Pnt(*map(float, p0)), gp_Dir(*dvec)), dvec


def _hits(solid, p0, ax, in_house, monkeypatch):
    monkeypatch.setattr(backend, "_PLANAR_ROW_HITS", in_house)
    line, dvec = _line(p0, ax)
    cand = solid._line_candidates(np.asarray(p0, float), dvec)
    return sorted(solid.flagged_hits(cand[0], line, -np.inf, np.inf, np.asarray(p0, float), dvec))


def test_in_house_hits_match_the_kernel(monkeypatch):
    teeth = 12
    occ = _comb(teeth)._occ_shape(1.0)
    solid = backend._PrefilteredLineSolid(occ, TOL)
    assert any(r is not None for r in solid._row_planar)
    rng = np.random.default_rng(5)
    # Lines through cell centres: no coincidence with an outline.
    for _ in range(150):
        ax = int(rng.integers(3))
        p0 = rng.uniform((-MM, -MM, -0.1 * MM), ((teeth * 2 + 1) * MM, 5 * MM, 0.3 * MM))
        p0[ax] = 0.0
        got = _hits(solid, p0, ax, True, monkeypatch)
        ref = _hits(solid, p0, ax, False, monkeypatch)
        assert len(got) == len(ref), (p0, ax, got, ref)
        for (w1, s1, u1), (w2, s2, u2) in zip(got, ref):
            assert abs(w1 - w2) < 1e-12 and s1 == s2 and u1 == u2, (p0, ax, got, ref)
    # Lines on the copper's own outlines: the geometric ON is at least
    # as conservative as the kernel's classifier — every kernel hit is
    # reproduced, any extra in-house hit is an untrusted boundary hit.
    for i in range(teeth):
        for p0, ax in (
            ((i * 2 * MM, 2 * MM, 0.0), 2),  # on a tooth's side plane
            ((i * 2 * MM, 0.0, 0.05 * MM), 1),  # along a tooth's side, mid-thickness
            ((0.0, MM, 0.05 * MM), 0),  # along the spine/tooth junction
            ((i * 2 * MM + 0.5 * MM, 0.0, 0.1 * MM), 1),  # in the top plane
        ):
            got = _hits(solid, p0, ax, True, monkeypatch)
            ref = _hits(solid, p0, ax, False, monkeypatch)
            for w2, s2, u2 in ref:
                assert any(abs(w1 - w2) < 1e-12 and s1 == s2 and u1 == u2 for w1, s1, u1 in got)
            extra = [h for h in got if not any(abs(h[0] - w2) < 1e-12 for w2, _, _ in ref)]
            assert all(u for _, _, u in extra), (p0, ax, extra)


def test_edge_fractions_with_and_without_in_house_hits(monkeypatch):
    teeth = 12
    occ = _comb(teeth)._occ_shape(1.0)
    x = np.concatenate([np.arange(0, teeth * 2 * MM + 1e-9, 0.5 * MM), [0.25 * MM, 1.75 * MM]])
    x = np.unique(np.round(x, 12))
    y = np.array([-0.5 * MM, 0.0, 0.5 * MM, MM, 2.5 * MM, 4 * MM, 4.5 * MM])
    z = np.array([-0.1 * MM, 0.0, 0.05 * MM, 0.1 * MM, 0.2 * MM])
    edges = np.concatenate([_grid_edges(x, y, z, axis) for axis in range(3)])
    monkeypatch.setattr(backend, "_PLANAR_ROW_HITS", True)
    in_house = backend.compute_edge_pec_fractions([occ], edges, TOL)
    monkeypatch.setattr(backend, "_PLANAR_ROW_HITS", False)
    kernel = backend.compute_edge_pec_fractions([occ], edges, TOL)
    assert np.allclose(in_house, kernel, atol=1e-12, rtol=0)
    assert 0.0 < in_house.mean() < 1.0


def test_curved_rows_stay_with_the_kernel(monkeypatch):
    post = geo.Cylinder(origin=(MM, MM, 0), radius=0.4 * MM, height=MM, axis="z", material="pec")
    occ = post._occ_shape(1.0)
    solid = backend._PrefilteredLineSolid(occ, TOL)
    assert all(r is None for r in solid._row_planar)
    x = np.linspace(0, 2 * MM, 9)
    z = np.array([-0.2 * MM, 0.3 * MM, 0.8 * MM, 1.2 * MM])
    edges = np.concatenate([_grid_edges(x, x, z, axis) for axis in range(3)])
    in_house = backend.compute_edge_pec_fractions([occ], edges, TOL)
    monkeypatch.setattr(backend, "_PLANAR_ROW_HITS", False)
    kernel = backend.compute_edge_pec_fractions([occ], edges, TOL)
    assert np.allclose(in_house, kernel, atol=1e-12, rtol=0)
    assert 0.0 < in_house.mean() < 1.0


def test_planar_tiles_keep_the_parity_of_the_face():
    """Tiles of a comb cap: away from tile borders, the even-odd state
    over a tile's clipped rings equals the state over the whole cap."""
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods

    teeth = 12
    occ = _comb(teeth)._occ_shape(1.0)
    exp = TopExp_Explorer(occ, TopAbs_FACE)
    cap = None
    while exp.More():
        face = topods.Face(exp.Current())
        planar = backend._planar_row(face)
        if planar is not None and planar[0] == 2 and planar[1] > 0.0 and len(planar[3]) > 20:
            cap = planar
        exp.Next()
    assert cap is not None
    axis, level, outward, verts, offsets = cap
    lo = np.array([verts[:, 0].min(), verts[:, 1].min(), level])
    hi = np.array([verts[:, 0].max(), verts[:, 1].max(), level])
    tiles = backend._planar_tiles(cap, lo, hi)
    assert tiles is not None and len(tiles) > 1
    rng = np.random.default_rng(11)
    pts = rng.uniform(lo[:2], hi[:2], size=(2000, 2))
    checked = 0
    for pu, pv in pts:
        whole = planar_point_state(pu, pv, verts, offsets, 1e-9)
        hosts = [
            t
            for t, t_lo, t_hi in tiles
            if t_lo[0] - 1e-9 <= pu <= t_hi[0] + 1e-9 and t_lo[1] - 1e-9 <= pv <= t_hi[1] + 1e-9
        ]
        states = {planar_point_state(pu, pv, t[3], t[4], 1e-9) for t in hosts}
        if whole == 2 or 2 in states:
            continue  # on an outline or a tile border: the fallback's call
        checked += 1
        if whole == 1:
            assert 1 in states, (pu, pv)
        else:
            assert states <= {0}, (pu, pv, states)
    assert checked > 1500
    solid = backend._PrefilteredLineSolid(occ, TOL)
    assert len(solid._row_face) > len(solid._faces)
    assert len(solid._ints) == 0  # nothing built until a kernel row is needed


def test_oblique_lines_are_decided_in_house_on_planar_rows():
    """An oblique line meets a planar row at one exactly known point:
    the whole face's kernel hits, without a kernel intersector."""
    from OCC.Core.gp import gp_Dir, gp_Lin, gp_Pnt
    from OCC.Core.IntCurvesFace import IntCurvesFace_Intersector

    teeth = 12
    occ = _comb(teeth)._occ_shape(1.0)
    solid = backend._PrefilteredLineSolid(occ, TOL)
    assert len(solid._row_face) > len(solid._faces)  # the cap is tiled
    rng = np.random.default_rng(2)
    for _ in range(40):
        p0 = rng.uniform((-MM, -MM, -MM), ((teeth * 2 + 1) * MM, 5 * MM, -0.5 * MM))
        direction = rng.normal(size=3)
        direction[2] = abs(direction[2]) + 0.5
        direction /= np.linalg.norm(direction)
        line = gp_Lin(gp_Pnt(*p0), gp_Dir(*direction))
        cand = solid._line_candidates(p0, direction)
        hits = sorted(solid.flagged_hits(cand[0], line, -np.inf, np.inf, p0, direction))
        reference = []
        for fi in sorted({int(solid._row_face[r]) for r in cand[0]}):
            it = IntCurvesFace_Intersector(solid._faces[fi], TOL)
            it.Perform(line, -1e100, 1e100)
            reference += [it.WParameter(k) for k in range(1, it.NbPnt() + 1)]
        assert sorted(round(w, 12) for w, _, _ in hits) == sorted(round(w, 12) for w in reference)
    assert len(solid._ints) == 0  # no kernel intersector was ever built


# ── Batch formulation: line table, compiled pairs, per-edge bookkeeping ────

from magnelio.geo._line_kernels import (  # noqa: E402
    axis_line_pairs,
    classify_on_lines,
    line_flags,
)


def test_line_pairs_reproduce_the_single_line_candidates():
    rng = np.random.default_rng(7)
    lo = rng.uniform(0, 1, size=(300, 3))
    hi = lo + rng.uniform(0, 0.3, size=(300, 3))
    lo[:40, 1] = 0.5  # boxes sharing one slab bound
    n = 500
    ax = rng.integers(3, size=n)
    pts = rng.uniform(-0.1, 1.1, size=(n, 3))
    pts[::7, 0] = 0.5 - 1e-9  # a tolerance-wide coincidence
    u = backend._TRANSVERSE_AXES[0][ax]
    v = backend._TRANSVERSE_AXES[1][ax]
    rows = np.arange(n)
    pair_line, pair_row = axis_line_pairs(ax, pts[rows, u], pts[rows, v], lo, hi, 1e-9)
    got = {(int(a), int(b)) for a, b in zip(pair_line, pair_row, strict=True)}
    ref = set()
    for i in range(n):
        idx, _, _ = axis_line_candidates(lo, hi, 1e-9, pts[i], int(ax[i]), 1.0)
        ref.update((i, int(r)) for r in idx)
    assert got == ref


def test_line_flags_and_classification_follow_the_sequential_rules():
    # line 0: clean pair; line 1: untrusted hit; line 2: leaves before it
    # enters; line 3: two entries in a row; line 4: no hit at all.
    offsets = np.array([0, 2, 4, 6, 8, 8])
    step = np.array([1, -1, 1, 0, -1, 1, 1, 1])
    untrusted = np.array([False, False, False, True, False, False, False, False])
    assert line_flags(offsets, step, untrusted).tolist() == [True, False, False, False, True]
    cross_offsets = np.array([0, 2, 2])
    cross_w = np.array([1.0, 3.0])
    ok = np.array([True, False])
    pts = np.array([0, 0, 0, 0, 0, 1])
    w = np.array([0.5, 2.0, 3.5, 1.0 + 5e-10, 3.0 - 5e-10, 0.0])
    state = classify_on_lines(pts, w, cross_offsets, cross_w, ok, 1e-9)
    # outside, inside, outside, ON → inside, ON → inside, undecided
    assert state.tolist() == [0, 1, 0, 1, 1, 2]


def test_line_table_keeps_the_first_origin_and_the_ids():
    occ = geo.Brick(origin=(0, 0, 0), size=(MM, MM, MM), material="pec")._occ_shape(1.0)
    table = backend._LineTable(backend._PrefilteredLineSolid(occ, TOL), TOL)
    ax = np.array([2, 2, 0, 2])
    pu = np.array([0.5 * MM, 0.5 * MM, 0.5 * MM, 0.2 * MM])
    pv = np.array([0.5 * MM, 0.5 * MM, 0.5 * MM, 0.2 * MM])
    origin = np.array([-MM, -2 * MM, -3 * MM, -MM])
    ids = table.resolve(ax, pu, pv, origin)
    assert ids.tolist() == [0, 0, 1, 2]
    assert table.origin.tolist() == [-MM, -3 * MM, -MM]  # the first asker's
    # The z line through the cube's centre enters at z = 0 and leaves at
    # z = 1 mm, parameters relative to its origin.
    sl = slice(table.hit_offsets[0], table.hit_offsets[1])
    assert table.ok[0]
    assert table.hit_w[sl].tolist() == [MM, 2 * MM]
    assert table.hit_step[sl].tolist() == [1, -1]
    # A second round: one known key (keeps its id and origin), one new.
    ids = table.resolve(
        np.array([2, 1]),
        np.array([0.5 * MM, 0.5 * MM]),
        np.array([0.5 * MM, 0.5 * MM]),
        np.array([7.0, 0.0]),
    )
    assert ids.tolist() == [0, 3] and table.origin[0] == -MM and table.ax.size == 4
    # The single-point classifier: centre inside, outside beyond a face,
    # on the boundary → inside.
    assert table.classify_point(np.array([0.5 * MM, 0.5 * MM, 0.5 * MM]), [2]) is False
    assert table.classify_point(np.array([0.5 * MM, 0.5 * MM, 1.5 * MM]), [2]) is True
    assert table.classify_point(np.array([0.5 * MM, 0.5 * MM, MM]), [2]) is False


def _sequential_fractions(solid, occ, edges, tol):
    """The per-edge bookkeeping of the edge pass on single-line queries
    (the pass before the batch formulation), as a reference: windows of
    one line query per carrier line, the fallback classifying every
    sub-segment midpoint on its own probe lines, the oblique probes and
    the solid classifier as the last resort."""
    import bisect

    from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
    from OCC.Core.gp import gp_Dir, gp_Lin, gp_Pnt
    from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON

    lines = {}
    classifier = BRepClass3d_SolidClassifier()
    classifier.Load(occ)

    def line_entry(ax, point):
        key = (ax, float(point[(ax + 1) % 3]), float(point[(ax + 2) % 3]))
        if key in lines:
            return lines[key]
        origin = np.array([float(point[0]), float(point[1]), float(point[2])])
        dvec = np.zeros(3)
        dvec[ax] = 1.0
        cand = solid._line_candidates(origin, dvec)
        gline = gp_Lin(gp_Pnt(*origin), gp_Dir(*dvec))
        hits = sorted(solid.flagged_hits(cand[0], gline, -np.inf, np.inf, origin, dvec))
        crossings = [(w, s) for w, s, _ in hits if s != 0]
        ok = not any(u for _, _, u in hits)
        prev = 0
        for _, s in crossings:
            if s == prev:
                ok = False
                break
            prev = s
        if crossings and crossings[0][1] != 1:
            ok = False
        entry = (origin[ax], hits, [w for w, _, _ in hits], [w for w, _ in crossings], ok)
        lines[key] = entry
        return entry

    def classify(point, axes):
        for ax in axes:
            origin_ax, _, _, ws, ok = line_entry(int(ax), point)
            if not ok:
                continue
            w_rel = float(point[int(ax)]) - origin_ax
            pos = bisect.bisect_left(ws, w_rel - tol)
            if pos < len(ws) and ws[pos] <= w_rel + tol:
                return False
            return bisect.bisect(ws, w_rel) % 2 == 0
        state = solid.point_state(point, backend._OBLIQUE_PROBES)
        if state is None:
            classifier.Perform(gp_Pnt(*map(float, point)), tol)
            return classifier.State() not in (TopAbs_IN, TopAbs_ON)
        return state

    f_L = np.ones(len(edges))
    for i, (p0, p1) in enumerate(edges):
        dx = p1 - p0
        length = float(np.linalg.norm(dx))
        ax = int(np.nonzero(dx)[0][0])
        origin_ax, hits_all, ws_all, ws, ok = line_entry(ax, p0)
        offset = float(p0[ax]) - origin_ax
        lo = bisect.bisect_left(ws_all, offset - tol)
        hi = bisect.bisect_right(ws_all, offset + length + tol)
        window = hits_all[lo:hi]
        hits = [(w - offset, s) for w, s, _ in window]
        clean = not any(u for _, _, u in window)
        params = []
        for p in sorted(max(0.0, min(w, length)) for w, _ in hits):
            if not params or p - params[-1] > tol:
                params.append(p)
        bounds = [0.0]
        for p in params:
            if p - bounds[-1] > tol:
                bounds.append(p)
        if length - bounds[-1] > tol:
            bounds.append(length)
        fallback = not clean
        trans_ws, first_outside = [], True
        if not fallback:
            trans = sorted((w, s) for w, s in hits if s != 0)
            if trans:
                fallback = any(trans[k][1] != -trans[k - 1][1] for k in range(1, len(trans)))
                trans_ws = [w for w, _ in trans]
                first_outside = trans[0][1] > 0
            elif ok:
                first_outside = bisect.bisect(ws, offset + length / 2.0) % 2 == 0
            else:
                fallback = True
        direction = dx / length
        probe_axes = np.argsort(np.abs(direction), kind="stable")
        outside_len = 0.0
        for a, b in zip(bounds[:-1], bounds[1:], strict=True):
            if b - a < tol:
                continue
            t_mid = (a + b) / 2
            if fallback:
                outside = classify(p0 + t_mid * direction, probe_axes)
            else:
                k = bisect.bisect(trans_ws, t_mid)
                outside = first_outside if k % 2 == 0 else not first_outside
            if outside:
                outside_len += b - a
        f_L[i] = max(0.0, min(outside_len / length, 1.0))
    return f_L


@pytest.mark.parametrize("body", ["comb", "post"])
def test_batch_fractions_equal_the_sequential_bookkeeping(body):
    teeth = 8
    if body == "comb":
        shape = _comb(teeth)
        x = np.concatenate([np.arange(0, teeth * 2 * MM + 1e-9, 0.5 * MM), [0.25 * MM, 1.75 * MM]])
        x = np.unique(np.round(x, 12))
        y = np.array([-0.5 * MM, 0.0, 0.5 * MM, MM, 2.5 * MM, 4 * MM, 4.5 * MM])
        z = np.array([-0.1 * MM, 0.0, 0.05 * MM, 0.1 * MM, 0.2 * MM])
    else:
        shape = geo.Union(
            geo.Cylinder(origin=(MM, MM, 0), radius=0.4 * MM, height=MM, axis="z", material="pec"),
            geo.Brick(
                origin=(0.8 * MM, 0.8 * MM, 0), size=(0.4 * MM, 1.6 * MM, MM), material="pec"
            ),
            material="pec",
        )
        x = y = np.linspace(0, 2 * MM, 9)
        z = np.array([-0.2 * MM, 0.0, 0.3 * MM, 0.8 * MM, MM, 1.2 * MM])
    occ = shape._occ_shape(1.0)
    edges = np.concatenate([_grid_edges(x, y, z, axis) for axis in range(3)])
    got = backend.compute_edge_pec_fractions([occ], edges, TOL)
    assert backend._EDGE_FRACTION_STATS["fallback_edges"] > 0  # the rounds are exercised
    ref = _sequential_fractions(backend._PrefilteredLineSolid(occ, TOL), occ, edges, TOL)
    assert np.array_equal(got, ref)
    assert 0.0 < got.mean() < 1.0
