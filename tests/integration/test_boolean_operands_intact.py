"""A mesh build must leave the geometry it meshed unchanged (DD-146).

OCCT's Boolean kernel defaults to ``NonDestructive = false``: it may
raise the tolerance of its *argument* shapes, insert p-curves and shift
vertices in place.  Magnelio caches the built solid per shape, and a
Boolean result shares its unmodified sub-shapes with the operands, so
those edits are permanent and they reach back into the user's own
bodies.  A second model built from the same bodies then meshes a
different — silently degraded — geometry.
"""

import pytest

import magnelio as mio
from magnelio import geo

QUARTER = dict(x1=0, x2=10, y1=0, y2=10, z1=-10, z2=10)


def _occ():
    return pytest.importorskip("OCC.Core.BRepPrimAPI")


def _max_edge_tolerance(occ_shape):
    """Largest edge tolerance in *occ_shape* — the quantity BOP inflates."""
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopAbs import TopAbs_EDGE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods

    tolerances = []
    explorer = TopExp_Explorer(occ_shape, TopAbs_EDGE)
    while explorer.More():
        tolerances.append(BRep_Tool.Tolerance(topods.Edge(explorer.Current())))
        explorer.Next()
    return max(tolerances)


def _coupler_bodies():
    """A stripline coupler cut down to what still triggers the failure.

    Curved Boolean seams are the point: a revolved recess filleted into
    a pipe, two transverse coax stubs and a revolved electrode cut back
    out.  Plain bricks and lone cylinders survive a destructive kernel
    untouched — the tolerance only creeps on seams like these.
    """
    pec, air = mio.Material.pec(), mio.Material.air()
    dia, t, h, w, length = 90e-3, 1e-3, 3e-3, 20e-3, 75e-3

    pipe = geo.Cylinder(
        radius=dia / 2, origin=(0, 0, -20e-3), axis="z", height=length + 40e-3, material=air
    )
    recess = (
        geo.Face(
            normal="x",
            points=((0, -w), (0, length + w), (dia / 2 + t + h, length + w), (dia / 2 + t + h, -w)),
            material=pec,
        )
        .revolved(axis="z", angle_deg=51.0)
        .rotated(axis="z", angle_deg=-25.5)
        .filleted(edges="all", radius=1e-3)
    )
    coax = geo.Cylinder(
        origin=(0, 0, -w / 2), axis="y", height=dia / 2 + 10e-3, radius=4e-3, material=air
    )
    electrode = (
        geo.Face(
            normal="x",
            points=((dia / 2, 0), (dia / 2, length), (dia / 2 + t, length), (dia / 2 + t, 0)),
            material=pec,
        )
        .revolved(axis="z", angle_deg=25.7)
        .rotated(axis="z", angle_deg=-12.85)
    )

    vacuum = pipe + recess + coax + coax.mirrored(normal=(0, 0, 1), position=length / 2)
    vacuum = vacuum + vacuum.rotated(axis="z", angle_deg=180)
    electrode = electrode + electrode.rotated(axis="z", angle_deg=180)
    return [vacuum - electrode, electrode]


def _mesh_the_quarter(bodies):
    """Mesh the x>0, y>0 quarter of *bodies* — the symmetry-cut workflow."""
    model = mio.GeometryModel(background=mio.Material.pec())
    for body in bodies:
        model.add(geo.Intersection(body, geo.Brick.from_ranges(**QUARTER)))
    return mio.Mesh.from_geometry(model, mio.MeshControl(), f_max=1e9)


class TestMeshLeavesGeometryIntact:
    def test_edge_tolerances_do_not_move(self):
        """The invariant: meshing writes nothing back into the operands."""
        _occ()
        bodies = _coupler_bodies()
        before = [_max_edge_tolerance(b._occ_shape(1.0)) for b in bodies]

        _mesh_the_quarter(bodies)

        after = [_max_edge_tolerance(b._occ_shape(1.0)) for b in bodies]
        assert after == before

    def test_a_second_model_gets_the_same_geometry(self):
        """The consequence users hit: the same cut, twice, cuts the same.

        This body is small enough to survive one destructive mesh build
        — the tolerance creep it suffers stays below what the seams can
        absorb, so this test alone does not fail without the guard.  It
        guards the end state the tolerance invariant only implies: on
        the full coupler the second quarter-space cut returned a
        fragment and the mesh collapsed to one cell in x (internal
        record: ``investigations/boolean-operand-mutation``).
        """
        _occ()
        bodies = _coupler_bodies()
        first = _mesh_the_quarter(bodies)
        second = _mesh_the_quarter(bodies)

        assert (second.Nx, second.Ny, second.Nz) == (first.Nx, first.Ny, first.Nz)
        assert second.Nx > 1
