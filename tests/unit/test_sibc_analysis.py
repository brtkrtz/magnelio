"""Unit gates for the high-level SIBC wall model (WP-D5).

The ``wall_model="sibc"`` switch on ``AnalysisScatteringTD``: input
validation, the cached spec build (band, tag set, port-face exclusion),
the recipe round-trip (incl. pre-WP-D5 recipes), and the
``MonitorWallLoss`` SIBC accounting (spec faces + ``Re Z_fit`` instead
of the perturbative enumeration + ``R_s``).  The end-to-end physics
(alpha vs the closed form, monitor vs power balance) lives in
``tests/integration/test_sibc_analysis.py``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from magnelio import AnalysisScatteringTD, Mesh
from magnelio._fields.field_arrays import FieldState
from magnelio.analysis._recipe import build_scattering_recipe, recipe_kwargs
from magnelio.materials.material import Material
from magnelio.materials.roughness import Hammerstad
from magnelio.materials.surface_impedance import SurfaceImpedanceFit
from magnelio.mesh import BoxFace
from magnelio.mesh._surfaces import enumerate_sibc_surfaces
from magnelio.mesh.grid import GridLines
from magnelio.monitors.wall_loss import MonitorWallLoss
from magnelio.ports import PortSpecMultiConductor
from magnelio.solver._sibc import SIBCSpec

# DD-103: the closure these fixtures always assumed.  A face
# with no BC used to evolve under the free curl operator —
# which IS the natural magnetic wall, hence "PMC".
_BC_OPEN = {
    "xmin": "PMC",
    "xmax": "PMC",
    "ymin": "PMC",
    "ymax": "PMC",
    "zmin": "PMC",
    "zmax": "PMC",
}

SIG = 5.8e3


def _plate_grid():
    return GridLines(
        x=np.linspace(0, 10e-3, 11),
        y=np.linspace(0, 5e-3, 6),
        z=np.linspace(0, 30e-3, 31),
    )


def _plate_analysis(**kw):
    defaults = dict(
        mesh=Mesh.from_grid(
            _plate_grid(),
            boundary_conditions={
                "xmin": "PMC",
                "xmax": "PMC",
                "ymin": "PEC",
                "ymax": "PEC",
                "zmin": "PEC",
                "zmax": "PEC",
            },
        ),
        ports=[
            PortSpecMultiConductor(name="p1", plane=BoxFace.Z_MIN, n_modes=1),
            PortSpecMultiConductor(name="p2", plane=BoxFace.Z_MAX, n_modes=1),
        ],
        f_max=10e9,
        f_min=2e9,
        verbose=False,
    )
    defaults.update(kw)
    return AnalysisScatteringTD(**defaults)


class TestValidation:
    def test_bad_wall_model_raises(self):
        with pytest.raises(ValueError, match="wall_model"):
            _plate_analysis(wall_model="leontovich")

    def test_sibc_without_any_conductor_source_raises(self):
        """The plan's loud error: no lossy metal, no override — nothing
        to build a Z_s from."""
        with pytest.raises(ValueError, match="needs a conductor"):
            _plate_analysis(wall_model="sibc")

    def test_sibc_with_override_constructs(self):
        ana = _plate_analysis(wall_model="sibc", wall_sigma=SIG)
        assert ana.wall_model == "sibc"

    def test_sibc_with_lossy_metal_constructs_without_override(self):
        """A lossy-metal solid brings its own sigma — no override
        needed, as long as no plain-PEC wall is left without one (side
        walls PMC here; the port faces zmin/zmax never carry walls)."""
        metal = Material.lossy_metal("cu", sigma=5.8e7)
        mesh = Mesh.from_grid(
            _plate_grid(),
            regions=[(metal, (3e-3, 2e-3, 10e-3, 7e-3, 3e-3, 20e-3))],
            boundary_conditions={
                "xmin": "PMC",
                "xmax": "PMC",
                "ymin": "PMC",
                "ymax": "PMC",
                "zmin": "PEC",
                "zmax": "PEC",
            },
        )
        ana = _plate_analysis(mesh=mesh, wall_model="sibc")
        spec = ana._sibc_spec()
        assert {s.tag for s in spec.surfaces} == {1}
        assert spec.fits[1].sigma == 5.8e7

    def test_default_is_perturbative_and_spec_none(self):
        ana = _plate_analysis()
        assert ana.wall_model == "perturbative"
        assert ana._sibc_spec() is None


class TestSpecBuild:
    def test_port_faces_excluded_and_band_from_axis(self):
        """PEC BC walls become SIBC walls EXCEPT the faces hosting
        ports (port planes stay lossless); the fit band is the analysis
        band [f_axis[0], f_max]."""
        ana = _plate_analysis(wall_model="sibc", wall_sigma=SIG)
        spec = ana._sibc_spec()
        assert sorted(s.tag for s in spec.surfaces) == ["ymax", "ymin"]
        fit = spec.fits["ymin"]
        assert fit.f_lo == pytest.approx(float(ana.f_axis[0]))
        assert fit.f_hi == pytest.approx(10e9)
        assert fit.sigma == SIG

    def test_spec_is_cached(self):
        ana = _plate_analysis(wall_model="sibc", wall_sigma=SIG)
        assert ana._sibc_spec() is ana._sibc_spec()

    def test_wire_wall_monitors_sets_spec(self):
        mon = MonitorWallLoss(
            freqs=np.linspace(2e9, 10e9, 5),
            normal="z",
            position=2e-3,
            sigma=SIG,
            bc_faces=("ymin", "ymax"),
        )
        ana = _plate_analysis(
            wall_model="sibc",
            wall_sigma=SIG,
            monitors=(mon,),
        )
        ana._wire_wall_monitors()
        assert mon.sibc is ana._sibc_spec()

    def test_perturbative_leaves_monitors_untouched(self):
        mon = MonitorWallLoss(
            freqs=np.linspace(2e9, 10e9, 5),
            normal="z",
            position=2e-3,
            sigma=SIG,
            bc_faces=("ymin", "ymax"),
        )
        ana = _plate_analysis(monitors=(mon,))
        ana._wire_wall_monitors()
        assert mon.sibc is None


class TestRecipe:
    def test_wall_model_round_trips_through_json(self):
        ana = _plate_analysis(
            wall_model="sibc",
            wall_sigma=SIG,
            wall_mu=2.0,
            wall_roughness=Hammerstad(rms_height=1e-6),
        )
        recipe = json.loads(json.dumps(build_scattering_recipe(ana)))
        kw = recipe_kwargs(recipe)
        assert kw["wall_model"] == "sibc"
        assert kw["wall_sigma"] == SIG
        assert kw["wall_mu"] == 2.0
        assert kw["wall_roughness"] == Hammerstad(rms_height=1e-6)

    def test_pre_wp_d5_recipe_defaults_to_perturbative(self):
        """Older recipes carry no wall keys — the rebuilt analysis must
        be the unchanged perturbative one, never an error."""
        ana = _plate_analysis()
        recipe = build_scattering_recipe(ana)
        for key in ("wall_model", "wall_sigma", "wall_mu", "wall_roughness"):
            recipe.pop(key)
        kw = recipe_kwargs(recipe)
        assert kw["wall_model"] == "perturbative"
        assert kw["wall_sigma"] is None
        assert kw["wall_roughness"] is None


class TestMonitorSIBCAccounting:
    def _fit(self, c0, branches=()):
        return SurfaceImpedanceFit(
            sigma=SIG,
            mu=1.0,
            roughness=None,
            f_lo=1e9,
            f_hi=1e10,
            c0=c0,
            branches=tuple(branches),
            rel_err_re=0.0,
            rel_err_cplx=0.0,
        )

    def _spec_and_mesh(self):
        mesh = Mesh.from_grid(_plate_grid(), boundary_conditions=_BC_OPEN)
        surfs = enumerate_sibc_surfaces(mesh, bc_pec_faces=("ymin", "ymax"))
        fits = {
            "ymin": self._fit(0.4, ((2e10, 0.05),)),
            "ymax": self._fit(0.7),
        }
        return SIBCSpec(surfaces=tuple(surfs), fits=fits), mesh

    def test_raw_power_loss_uses_re_z_fit_on_spec_faces(self):
        """SIBC mode: loss = 1/2 Re Z_fit(f) * sum(w |H|^2) over the
        spec's own rows — hand-checked against the formula with
        synthetic fields."""
        spec, mesh = self._spec_and_mesh()
        freqs = np.array([2e9, 5e9, 1e10])
        mon = MonitorWallLoss(
            freqs=freqs,
            normal="z",
            position=2e-3,
            sibc=spec,
        )
        mon.attach(mesh)
        fields = FieldState.zeros(mesh.Nx, mesh.Ny, mesh.Nz, xp=np)
        rng = np.random.default_rng(4)
        fields.Hx[...] = rng.standard_normal(fields.Hx.shape)
        fields.Hz[...] = rng.standard_normal(fields.Hz.shape)
        dt = 1e-12
        mon.record(fields, 0, 0.0, dt)

        loss = mon.raw_power_loss()
        h_arrays = (fields.Hx, fields.Hy, fields.Hz)
        for surf in spec.surfaces:
            vals = np.empty(surf.comp.size)
            for c in range(3):
                sel = surf.comp == c
                if sel.any():
                    vals[sel] = h_arrays[c].reshape(-1)[surf.flat_idx[sel]]
            h_phys2 = (dt * np.abs(vals) * surf.inv_l_dual) ** 2
            re_z = spec.fits[surf.tag].impedance(freqs).real
            expected = 0.5 * re_z * np.sum(surf.weight * h_phys2)
            np.testing.assert_allclose(loss[surf.tag], expected, rtol=1e-12)

    def test_result_dump_shape_unchanged(self):
        """Same reader interface / wall_loss.h5 shape as the
        perturbative monitor (WP-D5 contract)."""
        spec, mesh = self._spec_and_mesh()
        mon = MonitorWallLoss(
            freqs=np.array([2e9, 1e10]),
            normal="z",
            position=2e-3,
            sibc=spec,
        )
        mon.attach(mesh)
        fields = FieldState.zeros(mesh.Nx, mesh.Ny, mesh.Nz, xp=np)
        fields.Hx[...] = 1.0
        fields.Ex[...] = 1.0
        mon.record(fields, 0, 0.0, 1e-12)
        dump = mon.result_dump()
        assert set(dump) == {
            "freqs",
            "tags",
            "fraction",
            "total",
            "h_bins",
            "ref_bins",
        }
        assert sorted(dump["tags"]) == ["ymax", "ymin"]

    def test_missing_fit_raises_at_attach(self):
        spec, mesh = self._spec_and_mesh()
        broken = SIBCSpec(surfaces=spec.surfaces, fits={"ymin": spec.fits["ymin"]})
        mon = MonitorWallLoss(
            freqs=np.array([2e9]),
            normal="z",
            position=2e-3,
            sibc=broken,
        )
        with pytest.raises(ValueError, match="no impedance fit"):
            mon.attach(mesh)


class TestCheckpointDisk:
    def test_sibc_state_survives_checkpoint_h5(self, tmp_path):
        """The ``"sibc"`` state group (string-keyed tags, per-branch u
        arrays) round-trips through the DD-070 checkpoint serialiser
        bit-exactly and loads back into the solver."""
        from magnelio.boundaries.pec import PECBoundary
        from magnelio.io.project import (
            _read_state_dict_h5,
            _write_state_dict_h5,
        )
        from magnelio.materials.surface_impedance import fit_wall_impedances
        from magnelio.mesh._surfaces import resolve_wall_conductors
        from magnelio.solver.fit_td import FITTimeDomainSolver

        mesh = Mesh.from_grid(_plate_grid(), boundary_conditions=_BC_OPEN)
        faces = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
        surfs = enumerate_sibc_surfaces(mesh, bc_pec_faces=faces)
        fits = fit_wall_impedances(
            resolve_wall_conductors(mesh, surfs, sigma=SIG),
            1e9,
            1e10,
        )
        spec = SIBCSpec(surfaces=tuple(surfs), fits=fits)
        s = FITTimeDomainSolver(
            mesh=mesh,
            boundary_conditions={f: PECBoundary(f) for f in faces},
            dt=1e-12,
            total_time_steps=30,
            verbose=False,
            sibc=spec,
        )
        s.setup()
        rng = np.random.default_rng(6)
        s._fields.e_flat[:] = rng.standard_normal(s._fields.e_flat.size)
        s.run()

        sd = s.state_dict()
        assert "sibc" in sd and any(v for v in sd["sibc"].values())
        path = tmp_path / "checkpoint.h5"
        _write_state_dict_h5(path, sd)
        rd = _read_state_dict_h5(path)
        for tag, branches in sd["sibc"].items():
            for key, val in branches.items():
                np.testing.assert_array_equal(rd["sibc"][tag][key], val)
                assert np.abs(val).max() > 0.0
        s.load_state_dict(rd)  # accepts the disk form
