"""DD-185: built-in materials may be named by string across the API.

``"air"``, ``"vacuum"`` and ``"pec"`` resolve to canonical instances of
the parameter-free factories wherever the public API expects a material;
resolution happens at the call site, so nothing downstream ever sees a
string.
"""

import numpy as np
import pytest

from magnelio import GeometryModel, Material, Mesh
from magnelio.geo import Brick, Cylinder, Difference, Union
from magnelio.materials.material import resolve_material
from magnelio.mesh.grid import GridLines


class TestResolveMaterial:
    def test_air_resolves_to_factory_instance(self):
        assert resolve_material("air") == Material.air()

    def test_vacuum_resolves(self):
        assert resolve_material("vacuum") == Material.vacuum()

    def test_case_insensitive(self):
        assert resolve_material("PEC").is_pec
        assert resolve_material("Air") == Material.air()

    def test_canonical_instance_is_shared(self):
        # Identity matters: the mesher's material bookkeeping is
        # id-based, so every "air" must be the same object.
        assert resolve_material("air") is resolve_material("AIR")

    def test_material_passes_through(self):
        m = Material.from_isotropic("diel", epsilon=2.0)
        assert resolve_material(m) is m

    def test_none_passes_through(self):
        assert resolve_material(None) is None

    def test_unknown_name_raises_with_choices(self):
        with pytest.raises(ValueError, match='"air", "vacuum", "pec"'):
            resolve_material("copper", "Brick.material")

    def test_unknown_name_names_argument(self):
        with pytest.raises(ValueError, match="Brick.material"):
            resolve_material("cooper", "Brick.material")

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError, match="takes a Material"):
            resolve_material(1.0, "Brick.material")


class TestStringAcceptance:
    def test_brick_material_string(self):
        assert Brick(material="pec").material.is_pec

    def test_cylinder_material_string(self):
        c = Cylinder(radius=1e-3, height=1e-3, material="air")
        assert c.material == Material.air()

    def test_geometry_model_background_string(self):
        assert GeometryModel(background="pec").background.is_pec

    def test_geometry_model_background_default_air(self):
        assert GeometryModel().background == Material.air()

    def test_geometry_model_background_invalid_type(self):
        with pytest.raises(TypeError, match="GeometryModel"):
            GeometryModel(background=1.0)

    def test_geometry_model_background_unknown_name(self):
        with pytest.raises(ValueError, match="not a built-in material name"):
            GeometryModel(background="copper")

    def test_boolean_material_string(self):
        a = Brick(material="air")
        b = Brick(origin=(0.2, 0.2, 0.2), size=(0.5, 0.5, 0.5))
        assert Difference(a, b, material="pec").material.is_pec
        assert Union(a, material="pec").material.is_pec

    def test_boolean_inherits_resolved_operand_material(self):
        a = Brick(material="air")
        b = Brick(origin=(0.2, 0.2, 0.2), size=(0.5, 0.5, 0.5))
        assert Difference(a, b).material == Material.air()

    def test_construction_body_stays_material_less(self):
        # No material remains the construction-body marker; the string
        # feature must not blur that distinction.
        assert Brick().material is None


class TestFromGrid:
    def _grid(self) -> GridLines:
        return GridLines(
            x=np.linspace(0, 4e-3, 5),
            y=np.linspace(0, 4e-3, 5),
            z=np.linspace(0, 4e-3, 5),
        )

    def test_background_string(self):
        mesh = Mesh.from_grid(self._grid(), background="pec")
        assert mesh.material_library[0].is_pec

    def test_region_string(self):
        mesh = Mesh.from_grid(
            self._grid(),
            regions=[("pec", (0, 0, 0, 4e-3, 4e-3, 2e-3))],
        )
        assert mesh.material_library[1].is_pec

    def test_repeated_region_string_shares_one_id(self):
        # Both regions name the same built-in: the id-based library
        # bookkeeping must see one shared instance, not two.
        mesh = Mesh.from_grid(
            self._grid(),
            regions=[
                ("pec", (0, 0, 0, 4e-3, 4e-3, 1e-3)),
                ("pec", (0, 0, 3e-3, 4e-3, 4e-3, 4e-3)),
            ],
        )
        assert len(mesh.material_library) == 2  # background + one PEC


class TestCadImportMapping:
    def test_broadcast_string(self):
        from magnelio.io.cad import _resolve_materials

        mats = _resolve_materials(["a", "b"], "pec")
        assert all(m.is_pec for m in mats)

    def test_dict_value_string_reaches_imported_solid(self):
        # Dict values pass through untouched here; ImportedSolid
        # resolves them at construction.
        from magnelio.io.cad import _resolve_materials

        assert _resolve_materials(["pin"], {"pin": "pec"}) == ["pec"]


class TestLoftConstructor:
    def test_material_string_resolves(self):
        # Regression: Loft is a public class constructor, not routed
        # through the loft() factory — its material must resolve too
        # (tutorial 14 hit the raw string in plot_cross_section).
        from magnelio.geo import Face, Loft

        throat = Face(normal="z", points=((0, 0), (1e-3, 0), (0, 1e-3)))
        mouth = Face(normal="z", points=((0, 0), (2e-3, 0), (0, 2e-3)), position=5e-3)
        assert Loft(throat, mouth, material="pec").material.is_pec
