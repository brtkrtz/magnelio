"""Slab-restricted OCC sections and the side step of degenerate planes.

A section over the faces a plane can reach must return the contours of
the whole body; the side step of a degenerate plane must clear the
kernel's tolerance, or the kernel reports the face the plane was meant
to leave on both sides.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio.geo import Brick, Cylinder, Difference, Sphere, Union
from magnelio.geo import _occ_backend as ob
from magnelio.materials import Material

pytest.importorskip("OCC")

AIR = Material.from_isotropic(name="air", epsilon=1.0)
PEC = Material.pec()
DEFLECTION = 6e-8  # a 6 µm cell's tessellation budget — below the kernel tolerance
NUDGE = 6e-7


def _pocketed(n_pockets=12, size=(0.2e-3, 0.2e-3, 0.2e-3), z=0.8e-3):
    body = Brick(origin=(0, 0, 0), size=(4e-3, 2e-3, 1e-3), material=AIR)
    pockets = [
        Brick(
            origin=(0.2e-3 + 0.3e-3 * i, 0.4e-3 + 0.5e-3 * (i % 3), z),
            size=size,
            material=PEC,
        )
        for i in range(n_pockets)
    ]
    return Difference(body, *pockets), pockets


def _canonical(polys):
    return sorted((tuple(map(tuple, p)) for p in polys), key=lambda t: (len(t), t))


def _sections(shape, axis, positions, slab):
    return [
        _canonical(
            ob.cross_section_polygons(
                shape, axis, pos, deflection=DEFLECTION, scale=1.0, nudge=NUDGE, slab=slab
            )
        )
        for pos in positions
    ]


def test_restricted_sections_equal_whole_body_sections():
    pocketed, pockets = _pocketed()
    occ = pocketed._occ_shape(1.0)
    slab = ob._FaceSlabIndex(occ)
    assert slab.n_faces > 6
    ends = sorted({p.origin[0] for p in pockets} | {p.origin[0] + p.size[0] for p in pockets})
    positions = [e + s for e in ends for s in (6e-7, -6e-7)] + list(np.linspace(0.1e-3, 3.9e-3, 17))
    assert any(slab.restrict(0, pos) is not occ for pos in positions)
    assert _sections(occ, "x", positions, slab) == _sections(occ, "x", positions, None)
    zs = [0.7e-3, 0.9e-3, 0.8e-3 + 6e-7, 0.8e-3 - 6e-7]
    assert _sections(occ, "z", zs, slab) == _sections(occ, "z", zs, None)


def test_restricted_sections_on_a_curved_union():
    shape = Union(
        Cylinder(origin=(1e-3, 1e-3, 0), radius=0.5e-3, height=1e-3, axis="z", material=PEC),
        Brick(origin=(1e-3, 0.8e-3, 0), size=(2e-3, 0.4e-3, 1e-3), material=PEC),
    )
    occ = shape._occ_shape(1.0)
    slab = ob._FaceSlabIndex(occ)
    positions = list(np.linspace(0.6e-3, 2.9e-3, 12))
    assert any(slab.restrict(0, pos) is not occ for pos in positions)
    assert _sections(occ, "x", positions, slab) == _sections(occ, "x", positions, None)


def test_restrict_returns_the_shape_when_every_face_reaches_the_plane():
    occ = Sphere(center=(0, 0, 0), radius=1e-3, material=AIR)._occ_shape(1.0)
    slab = ob._FaceSlabIndex(occ)
    assert slab.n_faces == 1
    assert slab.restrict(0, 0.0) is occ
    assert slab.tolerance >= 0.0


def _floor_with_holes(n_holes=30):
    """An air slab whose floor carries the outline of every post pocket."""
    body = Brick(origin=(0, 0, 0), size=(12e-3, 2e-3, 1e-3), material=AIR)
    posts = [
        Cylinder(
            origin=(0.4e-3 + 0.38e-3 * i, 1e-3, 0),
            radius=0.15e-3,
            height=0.6e-3,
            axis="z",
            material=PEC,
        )
        for i in range(n_holes)
    ]
    return Difference(body, *posts), posts


def _n_edges(shape):
    from OCC.Core.TopAbs import TopAbs_EDGE
    from OCC.Core.TopExp import TopExp_Explorer

    n = 0
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer.More():
        n += 1
        explorer.Next()
    return n


def _faces_of(compound):
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer

    faces = []
    explorer = TopExp_Explorer(compound, TopAbs_FACE)
    while explorer.More():
        faces.append(explorer.Current())
        explorer.Next()
    return faces


class TestTiledFaces:
    """A heavy planar face enters a section as the one tile the plane
    crosses; every other situation takes the whole face."""

    def test_the_heavy_floor_is_tiled_on_demand_only(self):
        pocketed, _ = _floor_with_holes()
        slab = ob._FaceSlabIndex(pocketed._occ_shape(1.0))
        assert slab._tiles == {}
        floor = max(range(slab.n_faces), key=lambda i: _n_edges(slab.faces[i]))
        assert _n_edges(slab.faces[floor]) >= ob._SLAB_TILE_MIN_EDGES
        tiles = slab.tiles(floor)
        assert tiles is not None and tiles.normal_axis == 2
        assert len(tiles.pieces) >= 2 and len(tiles.cuts[0]) == len(tiles.pieces) - 1
        assert len(tiles.cuts[1]) == 0
        simple = min(range(slab.n_faces), key=lambda i: _n_edges(slab.faces[i]))
        assert slab.tiles(simple) is None
        assert slab.tiles(floor) is tiles

    def test_sections_over_tiles_match_the_whole_body(self):
        pocketed, _ = _floor_with_holes()
        occ = pocketed._occ_shape(1.0)
        slab = ob._FaceSlabIndex(occ)
        floor = max(range(slab.n_faces), key=lambda i: _n_edges(slab.faces[i]))
        cuts = slab.tiles(floor).cuts[0]
        rng = np.random.default_rng(11)
        positions = list(rng.uniform(0.05e-3, 11.95e-3, 60))
        positions += [c + s for c in cuts for s in (0.0, 0.5e-7, -3e-7, 1e-5, -1e-5)]
        assert _sections(occ, "x", positions, slab) == _sections(occ, "x", positions, None)
        ys = list(np.linspace(0.2e-3, 1.8e-3, 5))
        assert _sections(occ, "y", ys, slab) == _sections(occ, "y", ys, None)
        zs = [0.3e-3, 0.6e-3 + 6e-7, 0.6e-3 - 6e-7]
        assert _sections(occ, "z", zs, slab) == _sections(occ, "z", zs, None)

    def test_a_plane_inside_one_tile_takes_the_tile_and_no_other_case_does(self):
        pocketed, _ = _floor_with_holes()
        occ = pocketed._occ_shape(1.0)
        slab = ob._FaceSlabIndex(occ)
        floor = max(range(slab.n_faces), key=lambda i: _n_edges(slab.faces[i]))
        face = slab.faces[floor]
        tiles = slab.tiles(floor)
        cut = float(tiles.cuts[0][0])
        inside = 0.5 * (cut + float(tiles.cuts[0][1]))

        def carries_the_floor(compound):
            return any(f.IsSame(face) for f in _faces_of(compound))

        def carries_a_tile(compound):
            return any(any(f.IsSame(p) for p in tiles.pieces) for f in _faces_of(compound))

        assert carries_a_tile(slab.restrict(0, inside))
        assert not carries_the_floor(slab.restrict(0, inside))
        on_cut = slab.restrict(0, cut)
        assert carries_the_floor(on_cut) and not carries_a_tile(on_cut)
        across = slab.restrict(1, 1e-3)  # a y-plane crosses every strip
        assert carries_the_floor(across) and not carries_a_tile(across)
        coplanar = slab.restrict(2, 0.0)
        assert carries_the_floor(coplanar) and not carries_a_tile(coplanar)


def test_pool_worker_sections_with_its_own_slab():
    from OCC.Core.BRepTools import breptools

    pocketed, _ = _pocketed()
    occ = pocketed._occ_shape(1.0)
    blob = breptools.WriteToString(occ)
    ob._section_worker_init([(7, blob)])
    try:
        assert ob._SECTION_WORKER_SLABS[7].n_faces == ob._FaceSlabIndex(occ).n_faces
        direct = _sections(occ, "x", [0.3e-3], None)[0]
        via_worker = _canonical(ob._section_worker(("x", 0.3e-3, 7, DEFLECTION, 1.0, NUDGE, "")))
        assert via_worker == direct
    finally:
        ob._SECTION_WORKER_SHAPES.pop(7, None)
        ob._SECTION_WORKER_SLABS.pop(7, None)


def _face_pass(shapes_with_material, face_specs, face_axes, library=None, **kwargs):
    n = len(face_specs)
    outs = dict(
        pec_area_out=np.zeros(n),
        pec_area_geom_out=np.zeros(n),
        pec_area_jump_out=np.zeros(n),
    )
    values = ob.compute_face_material_areas(
        shapes_with_material,
        library or {0: AIR, 1: PEC},
        np.asarray(face_specs, dtype=float),
        np.asarray(face_axes, dtype=int),
        prop="mu",
        deflection=DEFLECTION,
        nudge=NUDGE,
        scale=1.0,
        **outs,
        **kwargs,
    )
    return values, outs


def test_side_step_clears_the_kernel_tolerance_on_a_conductor_end_wall():
    """A face lying in a conductor's end wall is free (min-convention)
    and books the wall's jump.

    The wall is the end of a 12.6 µm × 5 µm pocket inside an air body
    with a conducting background — the Lange coupler's finger.  With the
    side step at this grid's deflection (60 nm, below the kernel's
    tolerance) the section Boolean reported the pocket on both sides of
    its end wall; the spurious opening fell to the background, the face
    read fully blocked and the wall jump zero.
    """
    pocketed, pockets = _pocketed(6, size=(0.5e-3, 12.6e-6, 5e-6), z=0.5e-3)
    p = pockets[2]
    x = p.origin[0] + p.size[0]
    y, z = p.origin[1], p.origin[2]
    w, t = p.size[1], p.size[2]
    rect = (x, y + 0.2 * w, z + 0.2 * t, y + 0.8 * w, z + 0.8 * t)
    area = 0.6 * w * 0.6 * t
    values, outs = _face_pass(
        [(pocketed, 1), (p, 2)], [rect], [0], library={0: PEC, 1: AIR, 2: PEC}
    )
    assert outs["pec_area_out"][0] == pytest.approx(0.0, abs=1e-30)
    assert outs["pec_area_geom_out"][0] == pytest.approx(area, rel=1e-9)
    assert outs["pec_area_jump_out"][0] == pytest.approx(-area, rel=1e-9)
    assert values[0] == pytest.approx(1.0)


def test_shifted_planes_are_answered_without_a_boolean(monkeypatch):
    pocketed, pockets = _pocketed(6)
    calls = []
    original = ob.cross_section_polygons

    def counting(*args, **kwargs):
        calls.append(args[2])
        return original(*args, **kwargs)

    monkeypatch.setattr(ob, "cross_section_polygons", counting)
    face_specs, face_axes = [], []
    for p in pockets:
        for x in (p.origin[0], p.origin[0] + p.size[0]):
            y, z = p.origin[1], p.origin[2]
            face_specs.append((x, y + 0.05e-3, z + 0.05e-3, y + 0.15e-3, z + 0.15e-3))
            face_axes.append(0)
    values, outs = _face_pass([(pocketed, 0), *((p, 1) for p in pockets)], face_specs, face_axes)
    assert calls == []
    assert np.all(np.isfinite(values))
    area = 0.1e-3 * 0.1e-3
    assert np.allclose(outs["pec_area_out"], 0.0, atol=1e-30)
    assert np.allclose(outs["pec_area_geom_out"], area, rtol=1e-9)
    assert np.allclose(np.abs(outs["pec_area_jump_out"]), area, rtol=1e-9)
