"""Public meshing policy scope and independent concurrent builds."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from magnelio import geo
from magnelio.geo import _occ_backend as backend
from magnelio.geo._section_policy import robust_sections_enabled, section_policy
from magnelio.mesh.mesher import Mesh, MeshControl


@pytest.mark.parametrize("enabled", [False, True])
def test_mesh_build_applies_control_and_restores_context(monkeypatch, enabled):
    seen = []
    original = backend.batch_cross_sections

    def observe(*args, **kwargs):
        seen.append(robust_sections_enabled())
        return original(*args, **kwargs)

    monkeypatch.setattr(backend, "batch_cross_sections", observe)
    monkeypatch.setenv("MAGNELIO_SURFACE_SECTIONS", "1")
    model = geo.GeometryModel(background="air")
    model.add(geo.Cylinder(axis="z", radius=0.002, height=0.004, material="pec"))
    with section_policy(False):
        Mesh.from_geometry(
            model, MeshControl(robust_sections=enabled, min_cells_per_feature=0), f_max=1e9
        )
        assert not robust_sections_enabled()
    assert seen and all(value is enabled for value in seen)


def test_failed_build_restores_policy():
    with section_policy(False):
        with pytest.raises((AttributeError, TypeError)):
            Mesh.from_geometry(None, MeshControl(robust_sections=True), f_max=1e9)
        assert not robust_sections_enabled()


def test_policy_is_independent_between_threads():
    with section_policy(True), ThreadPoolExecutor(max_workers=1) as pool:
        assert robust_sections_enabled()
        assert not pool.submit(robust_sections_enabled).result()


def test_robust_sections_is_disabled_by_default():
    assert MeshControl().robust_sections is False
    with pytest.raises(ValueError, match="robust_sections"):
        MeshControl(robust_sections="yes")
