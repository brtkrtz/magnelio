"""Certificates for the rank-r dispersive modal source (DD-248).

The synthesis has two failure modes that are invisible in any norm on
the profiles themselves and only show up as a run that never decays:
truncating the coefficients at the edge of the solved band, and a gauge
that jumps with frequency.  Both are gated here on the localisation of
the synthesised waveforms, which is the property the time-domain source
actually needs.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.geo import Brick, GeometryModel
from magnelio.materials.material import Material
from magnelio.mesh.mesher import Mesh, MeshControl
from magnelio.ports._modal import BoxFace, PortSpecMultiConductor, build_modal_port
from magnelio.ports._modal.dispersive_source import (
    DEFAULT_MAX_RANK,
    synthesise_dispersive_source,
)
from magnelio.ports._modal.factory import build_port_dispersion_record
from magnelio.solver.stability import courant_dt

F_MAX = 6.0e9
F_BAND = (0.5e9, 6.0e9)


def _segments(*spec):
    out = []
    for lo, hi, n in spec:
        out.append(np.linspace(lo, hi, n + 1))
    return np.unique(np.concatenate(out))


@pytest.fixture(scope="module")
def port_and_record():
    w, hy, h_if, n_len, d_len = 10.0e-3, 8.0e-3, 4.0e-3, 12, 1.0e-3
    length = n_len * d_len
    model = GeometryModel()
    model.add(
        Brick(
            origin=(0, 0, 0),
            size=(w, h_if, length),
            material=Material(name="lower", epsilon=(4.0,) * 3),
        )
    )
    model.add(Brick(origin=(0, h_if, 0), size=(w, hy - h_if, length), material=Material.air()))
    control = MeshControl(
        min_nodes_per_wavelength=4,
        min_cells_per_feature=0,
        max_cell_size=5.1e-3,
        forced_planes={
            "x": _segments((0.0, w, 2)),
            "y": _segments((0.0, h_if, 4), (h_if, hy, 4)),
            "z": _segments((0.0, length, n_len)),
        },
    )
    mesh = Mesh.from_geometry(model, control, f_max=F_MAX).with_boundary_conditions(
        {
            "ymin": "PEC",
            "ymax": "PEC",
            "xmin": "PMC",
            "xmax": "PMC",
            "zmin": "PMC",
            "zmax": "PMC",
        }
    )
    m_eps, m_mu = build_M_eps(mesh), build_M_mu(mesh)
    dt = courant_dt(mesh.grid, "normal")
    op = build_modal_port(
        PortSpecMultiConductor(name="p", plane=BoxFace.Z_MIN, epsilon_r=None),
        mesh,
        m_eps,
        m_mu,
        dt=dt,
        f_calc=F_MAX,
    )
    rec = build_port_dispersion_record(op, mesh, m_eps, m_mu, F_MAX)
    return op, rec, m_eps, m_mu, dt


def _drive(n_t, dt, f_c=3.0e9, bw=2.5e9):
    sig_t = 1.0 / (np.pi * bw)
    t0 = 4.0 * sig_t
    t = np.arange(n_t) * dt
    return np.exp(-(((t - t0) / sig_t) ** 2)) * np.cos(2.0 * np.pi * f_c * (t - t0))


def _synth(port_and_record, n_t=2048, **kw):
    op, rec, m_eps, m_mu, dt = port_and_record
    return synthesise_dispersive_source(
        rec,
        0,
        _drive(n_t, dt),
        dt,
        F_BAND,
        m_eps=m_eps,
        m_mu=m_mu,
        n_band_points=41,
        dual_projector=op.dual_projection_of,
        **kw,
    )


class TestSynthesis:
    def test_rank_reaches_the_target(self, port_and_record):
        terms = _synth(port_and_record, target_db=-40.0)
        assert terms.rank <= DEFAULT_MAX_RANK
        assert terms.profile_error_db <= -40.0

    def test_a_looser_target_costs_no_more_rank(self, port_and_record):
        loose = _synth(port_and_record, target_db=-20.0)
        tight = _synth(port_and_record, target_db=-70.0)
        assert loose.rank <= tight.rank

    def test_rank_can_be_pinned(self, port_and_record):
        terms = _synth(port_and_record, rank=1)
        assert terms.rank == 1
        assert terms.waveform.shape[0] == 1
        assert terms.profiles_u.shape[0] == 1

    def test_waveforms_are_localised(self, port_and_record):
        """The synthesis must not ring.

        Truncating the coefficients where the dispersion was solved, or
        letting the gauge jump with frequency, produces a source that is
        still moving at the end of the window — the run then never
        decays and its S-parameters are read off a moving signal.  Both
        are caught here.
        """
        terms = _synth(port_and_record, rank=2)
        for series in (terms.projected, terms.projected_interior):
            peak = np.abs(series).max()
            assert peak > 0.0
            assert np.abs(series[-128:]).max() / peak < 1e-3
            assert np.abs(series[:16]).max() / peak < 1e-3

    def test_projection_is_the_dual_of_what_is_imprinted(self, port_and_record):
        """``projected`` must be the dual projection of the family.

        The TF/SF subtraction is exact only if the scalar it removes is
        the projection of the field actually written on the plane.
        """
        op, *_ = port_and_record
        terms = _synth(port_and_record, rank=2)
        for n in (0, 200, 900):
            e_u = terms.profiles_u.T @ terms.waveform[:, n]
            e_v = terms.profiles_v.T @ terms.waveform[:, n]
            v_port, _ = op.dual_projection_of(e_u, e_v)
            assert v_port == pytest.approx(terms.projected[n], rel=1e-9, abs=1e-12)

    def test_interior_is_not_the_port_series_delayed(self, port_and_record):
        """The two planes differ by dispersion, not by one delay."""
        terms = _synth(port_and_record, rank=2)
        assert not np.allclose(terms.projected, terms.projected_interior)

    def test_band_must_be_positive_and_ordered(self, port_and_record):
        op, rec, m_eps, m_mu, dt = port_and_record
        with pytest.raises(ValueError, match="f_band"):
            synthesise_dispersive_source(
                rec,
                0,
                _drive(256, dt),
                dt,
                (6.0e9, 0.5e9),
                m_eps=m_eps,
                m_mu=m_mu,
                dual_projector=op.dual_projection_of,
            )

    def test_dual_projector_is_required(self, port_and_record):
        _op, rec, m_eps, m_mu, dt = port_and_record
        with pytest.raises(ValueError, match="dual_projector"):
            synthesise_dispersive_source(
                rec, 0, _drive(256, dt), dt, F_BAND, m_eps=m_eps, m_mu=m_mu
            )


class TestOperatorBinding:
    def test_refused_without_a_scalar_excitation(self, port_and_record):
        op, *_ = port_and_record
        terms = _synth(port_and_record, rank=1)
        op.clear_excitation()
        with pytest.raises(ValueError, match="no excitation"):
            op.set_excitation_dispersive(0, terms)

    def test_clear_excitation_drops_the_dispersive_terms(self, port_and_record):
        op, *_ = port_and_record
        terms = _synth(port_and_record, rank=1)
        op.set_excitation(0, lambda t: 0.0)
        op.set_excitation_dispersive(0, terms)
        assert op._disp_sources
        op.clear_excitation()
        assert not op._disp_sources

    def test_step_index_is_clamped_and_time_based(self, port_and_record):
        op, *_ = port_and_record
        terms = _synth(port_and_record, rank=1)
        n_last = terms.projected.size - 1
        assert op._disp_step(-1.0, terms) == 0
        assert op._disp_step(0.0, terms) == 0
        assert op._disp_step(5.0 * op._dt, terms) == 5
        assert op._disp_step(1e9 * op._dt, terms) == n_last

    def test_a_port_without_a_dispersive_source_keeps_the_frozen_path(self, port_and_record):
        op, *_ = port_and_record
        op.clear_excitation()
        assert op._disp_sources == {}
