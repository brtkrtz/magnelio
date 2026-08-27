"""Classification pieces of large planar faces in the edge pass.

``_PrefilteredLineSolid`` cuts a large axis-aligned planar face into
pieces for the kernel's line-face intersector, whose classification
costs O(edges of the face).  The pieces must report what the whole face
reports — same parameters, states and transitions — so the edge
fractions are bit-identical with and without them.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio import geo
from magnelio.geo import _occ_backend as backend
from magnelio.geo._prism_fuse import prism_candidates

MM = 1e-3


def _area(face) -> float:
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps

    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    return props.Mass()


def _comb(teeth: int):
    """A spine with *teeth* teeth: the cap carries 4 + 4 × teeth edges."""
    parts = [geo.Brick(origin=(0, 0, 0), size=(teeth * 2 * MM, MM, 0.1 * MM), material="pec")]
    parts += [
        geo.Brick(origin=(i * 2 * MM, MM, 0), size=(MM, 3 * MM, 0.1 * MM), material="pec")
        for i in range(teeth)
    ]
    return geo.Union(*parts, material="pec")


def _grid_edges(x, y, z, axis):
    """Axis-aligned unit segments of a grid, ``(N, 2, 3)``."""
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    start = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    step = np.zeros(3)
    step[axis] = (x[1] - x[0], y[1] - y[0], z[1] - z[0])[axis]
    return np.stack([start, start + step], axis=1)


class TestClassificationPieces:
    def test_simple_face_is_left_whole(self):
        (cap,) = prism_candidates(geo.Brick(origin=(0, 0, 0), size=(MM, MM, MM))._occ_shape(1.0))[
            2
        ][2]
        assert backend._classification_pieces(cap, np.zeros(3), np.array([MM, MM, 0.0])) is None

    def test_curved_face_is_left_whole(self):
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.GeomAbs import GeomAbs_Cylinder
        from OCC.Core.TopAbs import TopAbs_FACE
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopoDS import topods

        shape = geo.Cylinder(origin=(0, 0, 0), radius=MM, height=MM)._occ_shape(1.0)
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = topods.Face(explorer.Current())
            explorer.Next()
            if BRepAdaptor_Surface(face).GetType() == GeomAbs_Cylinder:
                break
        lo, hi = np.array([-MM, -MM, 0.0]), np.array([MM, MM, MM])
        assert backend._classification_pieces(face, lo, hi) is None

    def test_pieces_cover_the_face_and_keep_its_normal(self):
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.TopAbs import TopAbs_REVERSED

        def effective_normal(f):
            d = BRepAdaptor_Surface(f).Plane().Axis().Direction()
            n = np.array((d.X(), d.Y(), d.Z()))
            return -n if f.Orientation() == TopAbs_REVERSED else n

        teeth = 12
        cap = prism_candidates(_comb(teeth)._occ_shape(1.0))[2][2][0]
        lo, hi = np.array([0.0, 0.0, 0.0]), np.array([teeth * 2 * MM, 4 * MM, 0.0])
        pieces = backend._classification_pieces(cap, lo, hi)
        assert pieces is not None and len(pieces) > 1
        assert sum(_area(p) for p, _, _ in pieces) == pytest.approx(_area(cap), rel=1e-12)
        for piece, p_lo, p_hi in pieces:
            assert np.allclose(effective_normal(piece), effective_normal(cap))
            # Boxes carry the kernel's tolerance gap (1e-7).
            assert np.all(np.asarray(p_lo) >= lo - 1e-6) and np.all(np.asarray(p_hi) <= hi + 1e-6)

    def test_edge_fractions_are_identical_with_and_without_pieces(self, monkeypatch):
        """Grid lines through and along a comb: f_L must not change by a bit."""
        teeth = 12
        shape = _comb(teeth)
        occ = shape._occ_shape(1.0)
        # Grid planes on the copper's own edges (as the mesher would put
        # them) and between: grazing and clean crossings alike.
        x = np.concatenate([np.arange(0, teeth * 2 * MM + 1e-9, 0.5 * MM), [0.25 * MM, 1.75 * MM]])
        x = np.unique(np.round(x, 12))
        y = np.array([-0.5 * MM, 0.0, 0.5 * MM, MM, 2.5 * MM, 4 * MM, 4.5 * MM])
        z = np.array([-0.1 * MM, 0.0, 0.05 * MM, 0.1 * MM, 0.2 * MM])
        edges = np.concatenate([_grid_edges(x, y, z, axis) for axis in range(3)])

        with_pieces = backend.compute_edge_pec_fractions([occ], edges, 1e-8)
        monkeypatch.setattr(backend, "_PIECE_MIN_EDGES", 10**9)
        without = backend.compute_edge_pec_fractions([occ], edges, 1e-8)

        assert np.array_equal(with_pieces, without)
        assert 0.0 < with_pieces.mean() < 1.0  # inside, outside and crossing edges all present

    def test_large_cap_gets_rows_beyond_its_face(self):
        occ = _comb(12)._occ_shape(1.0)
        solid = backend._PrefilteredLineSolid(occ, 1e-8)
        assert len(solid._row_face) > len(solid._faces)
        assert set(solid._row_face.tolist()) == set(range(len(solid._faces)))
