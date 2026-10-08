"""Tests for the canonical curve rasteriser + integrate_E (Cluster 3, DD-076).

The rasteriser turns a Curve into an ordered, directed chain of primary-grid
E-edges; integrate_E sums the signed edge voltages ``Σ sign·E·dl``.  The
gates below check that this reproduces the analytic line integral ``∫E·dl``:

* a uniform field over any path (incl. an oblique helix) integrates to
  ``E·(B − A)`` — the staircase's net displacement is exact;
* a non-uniform conservative field (φ = x² ⇒ E_x = −2x, midpoint-sampled)
  integrates to ``φ(A) − φ(B)`` exactly, even on a graded grid;
* the EdgePath structure (signs, lengths, flat indices) is self-consistent
  and reverses with the curve direction.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio._fields.field_arrays import FieldArrays
from magnelio.circuit import EdgePath, integrate_E, rasterize_curve
from magnelio.fields import FieldSpectrum, FieldState
from magnelio.geo import Curve
from magnelio.mesh.grid import GridLines

pytestmark = pytest.mark.filterwarnings("error::numpy.exceptions.ComplexWarning")


def _uniform_grid(n: int = 20, L: float = 20e-3) -> GridLines:
    ax = np.linspace(0.0, L, n + 1)
    return GridLines(x=ax.copy(), y=ax.copy(), z=ax.copy())


def _graded_axis(lo: float, hi: float, n: int, growth: float = 1.18) -> np.ndarray:
    d = growth ** np.arange(n, dtype=float)
    d = d / d.sum() * (hi - lo)
    return lo + np.concatenate([[0.0], np.cumsum(d)])


def _set_uniform(field: FieldArrays, E0) -> None:
    field.Ex[:] = E0[0]
    field.Ey[:] = E0[1]
    field.Ez[:] = E0[2]


def test_uniform_field_diagonal_polyline():
    """∫E·dl of a uniform field over a diagonal polyline = E·(B − A)."""
    grid = _uniform_grid()
    field = FieldArrays.zeros(grid.Nx, grid.Ny, grid.Nz)
    E0 = (3.0, -2.0, 5.0)
    _set_uniform(field, E0)

    A = (grid.x[2], grid.y[3], grid.z[1])
    B = (grid.x[15], grid.y[12], grid.z[9])
    V = integrate_E(field, Curve.polyline([A, B]), grid)
    expected = sum(E0[k] * (B[k] - A[k]) for k in range(3))
    assert abs(V - expected) < 1e-12 * abs(expected)


def test_uniform_field_helix_is_path_independent():
    """An oblique helix in a uniform field integrates to the net displacement.

    The staircase visits many edges (its length far exceeds the straight-line
    distance), yet the signed sum depends only on the snapped endpoints — the
    property every downstream consumer relies on.
    """
    grid = _uniform_grid()
    field = FieldArrays.zeros(grid.Nx, grid.Ny, grid.Nz)
    E0 = (3.0, -2.0, 5.0)
    _set_uniform(field, E0)

    center = (grid.x[10], grid.y[10], grid.z[2])
    helix = Curve.helix(radius=4e-3, pitch=4e-3, turns=3.0, origin=center, axis="z")
    path = rasterize_curve(helix, grid)

    # The staircase is far longer than the 12 mm axial rise (obliqueness):
    # 3 turns of radius 4 mm span ~75 mm of circumference.
    assert path.length > 40e-3

    from magnelio.geo._occ_backend import sample_wire

    pts = sample_wire(helix._occ_shape(), 1e-4)
    A, B = pts[0], pts[-1]
    V = integrate_E(field, helix, grid)
    expected = sum(E0[k] * (B[k] - A[k]) for k in range(3))
    assert abs(V - expected) < 1e-9 * max(abs(expected), 1e-30)


@pytest.mark.parametrize("graded", [False, True])
def test_quadratic_potential_exact(graded):
    """φ = x² (E_x = −2x, midpoint-sampled) ⇒ ∫E·dl = φ(A) − φ(B), exactly.

    The cell-midpoint value of a linear ``E_x = −2x`` integrates exactly over
    each cell (``−(x_{i+1}² − x_i²)``), so the chain telescopes to
    ``x_A² − x_B²`` on any grid — uniform or graded.
    """
    if graded:
        grid = GridLines(
            x=_graded_axis(0.0, 20e-3, 20),
            y=np.linspace(0.0, 20e-3, 21),
            z=np.linspace(0.0, 20e-3, 21),
        )
    else:
        grid = _uniform_grid()

    field = FieldArrays.zeros(grid.Nx, grid.Ny, grid.Nz)
    xm = 0.5 * (grid.x[:-1] + grid.x[1:])  # x-edge midpoints
    field.Ex[:] = (-2.0 * xm)[:, None, None]  # E_x = −2x

    A = (grid.x[1], grid.y[2], grid.z[1])
    B = (grid.x[17], grid.y[14], grid.z[13])
    V = integrate_E(field, Curve.polyline([A, B]), grid)
    expected = A[0] ** 2 - B[0] ** 2  # φ(A) − φ(B)
    assert abs(V - expected) < 1e-12 * abs(expected)


def test_axis_aligned_segment_edges():
    """A straight +z segment yields all-z edges, +1 signs, contiguous nodes."""
    grid = _uniform_grid()
    A = (grid.x[5], grid.y[5], grid.z[2])
    B = (grid.x[5], grid.y[5], grid.z[11])
    path = rasterize_curve(Curve.polyline([A, B]), grid)

    assert len(path) == 9  # z index 2 → 11
    assert all(a == "z" for a in path.axes)
    assert all(s == 1 for s in path.signs)
    assert [ijk[2] for ijk in path.ijk] == list(range(2, 11))
    # flat indices land in the Ez block of the flat E layout
    n_Ex = grid.Nx * (grid.Ny + 1) * (grid.Nz + 1)
    n_Ey = (grid.Nx + 1) * grid.Ny * (grid.Nz + 1)
    assert all(fi >= n_Ex + n_Ey for fi in path.flat_indices)


def test_reversing_curve_flips_signs():
    """Reversing endpoints negates every sign and hence the integral."""
    grid = _uniform_grid()
    field = FieldArrays.zeros(grid.Nx, grid.Ny, grid.Nz)
    _set_uniform(field, (1.0, 2.0, -3.0))

    A = (grid.x[3], grid.y[4], grid.z[2])
    B = (grid.x[12], grid.y[9], grid.z[14])
    V_fwd = integrate_E(field, Curve.polyline([A, B]), grid)
    V_rev = integrate_E(field, Curve.polyline([B, A]), grid)
    assert abs(V_fwd + V_rev) < 1e-12 * abs(V_fwd)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_flat_indices_match_field_layout(dtype):
    """Integrating via flat_indices equals integrating via field arrays."""
    grid = _uniform_grid()
    field = FieldArrays.zeros(grid.Nx, grid.Ny, grid.Nz, dtype=dtype)
    rng = np.random.default_rng(0)
    field.e_flat[:] = rng.standard_normal(field.e_flat.shape)

    curve = Curve.polyline(
        [
            (grid.x[2], grid.y[2], grid.z[2]),
            (grid.x[9], grid.y[2], grid.z[2]),
            (grid.x[9], grid.y[13], grid.z[10]),
        ]
    )
    path = rasterize_curve(curve, grid)
    v_flat = sum(
        s * float(field.e_flat[fi]) * dl
        for s, fi, dl in zip(path.signs, path.flat_indices, path.dls)
    )
    v_field = integrate_E(field, curve, grid)
    assert abs(v_flat - v_field) < 1e-15 * max(abs(v_field), 1e-30)
    assert isinstance(path, EdgePath)
    assert abs(path.length - sum(path.dls)) < 1e-18


def test_rasterize_errors():
    """Too-coarse grid (single-node curve) and bad sampling raise."""
    grid = _uniform_grid(n=4, L=20e-3)  # 5-mm cells
    tiny = Curve.polyline(
        [(grid.x[2], grid.y[2], grid.z[2]), (grid.x[2] + 1e-4, grid.y[2] + 1e-4, grid.z[2])]
    )
    with pytest.raises(ValueError, match="single grid node"):
        rasterize_curve(tiny, grid)

    good = Curve.polyline([(grid.x[1], grid.y[1], grid.z[1]), (grid.x[3], grid.y[1], grid.z[1])])
    with pytest.raises(ValueError, match="samples_per_cell"):
        rasterize_curve(good, grid, samples_per_cell=1)


@pytest.mark.parametrize("value", [3.0, 3 + 4j, 4j, 3 + 0j])
def test_physical_field_integral_preserves_scalar_kind_and_phase(value):
    grid = _uniform_grid()
    field = FieldState.from_function(grid, E=lambda x, y, z: (value, 0, 0))
    curve = Curve.polyline([(grid.x[2], grid.y[3], grid.z[4]), (grid.x[9], grid.y[3], grid.z[4])])
    voltage = integrate_E(field, curve, grid)
    assert voltage == pytest.approx(value * 7e-3)
    assert type(voltage) is (complex if np.iscomplexobj(value) else float)


@pytest.mark.parametrize("graded", [False, True])
def test_complex_conservative_field_integrates_on_all_axes_and_reverses(graded):
    axes = [_graded_axis(0, 20e-3, 20, growth=g) for g in (1.12, 1.17, 1.21)]
    grid = GridLines(*axes) if graded else _uniform_grid()
    coefficients = (2 + 3j, -4 + 1j, 5 - 2j)
    field = FieldState.from_function(
        grid, E=lambda x, y, z: tuple(-2 * c * q for c, q in zip(coefficients, (x, y, z)))
    )
    a = (grid.x[2], grid.y[3], grid.z[4])
    b = (grid.x[15], grid.y[12], grid.z[13])
    expected = sum(c * (a_i**2 - b_i**2) for c, a_i, b_i in zip(coefficients, a, b))
    forward = integrate_E(field, Curve.polyline([a, b]), grid)
    reverse = integrate_E(field, Curve.polyline([b, a]), grid)
    assert forward == pytest.approx(expected, rel=1e-12)
    assert reverse == pytest.approx(-expected, rel=1e-12)


def test_frequency_frame_integral_matches_real_snapshots_at_each_phase():
    grid = _uniform_grid()
    shape = FieldState.zeros(grid).Ex.shape
    spectrum = FieldSpectrum(grid, [5e9], Ex=np.full((1, *shape), 3 + 4j))
    curve = Curve.polyline([(grid.x[2], grid.y[3], grid.z[4]), (grid.x[9], grid.y[3], grid.z[4])])
    voltage = integrate_E(spectrum.at_frequency(5e9), curve, grid)
    assert voltage == pytest.approx(0.021 + 0.028j)
    for phase in (0, 37, 90, 180):
        snapshot = spectrum.snapshot(5e9, phase_deg=phase)
        expected = (voltage * np.exp(1j * np.deg2rad(phase))).real
        assert integrate_E(snapshot, curve, grid) == pytest.approx(expected, abs=1e-14)


def test_complex_circulation_on_a_closed_path():
    grid = _uniform_grid()
    coefficient = 3 + 4j
    field = FieldState.from_function(grid, E=lambda x, y, z: (0, coefficient * x, 0))
    x0, x1 = grid.x[2], grid.x[12]
    y0, y1 = grid.y[3], grid.y[15]
    z = grid.z[4]
    curve = Curve.polyline([(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z), (x0, y0, z)])
    assert integrate_E(field, curve, grid) == pytest.approx(coefficient * (x1 - x0) * (y1 - y0))
