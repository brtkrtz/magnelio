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
