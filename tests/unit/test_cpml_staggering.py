"""Independent staggered-grid and mirror checks for CPML profiles."""

import numpy as np
import pytest

from magnelio._fields.field_arrays import FieldArrays
from magnelio.boundaries.cpml import CPMLBoundary
from magnelio.constants import C0, EPS0
from magnelio.mesh.grid import GridLines


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("side", ["min", "max"])
@pytest.mark.parametrize("alpha", [0.0, 0.02])
def test_e_coefficients_sample_physical_nodes(axis, side, alpha):
    widths = np.array([0.5, 0.7, 1.1, 1.4, 1.4, 1.1, 0.7, 0.5]) * 1e-3
    nodes = np.concatenate(([0.0], np.cumsum(widths)))
    axes = [np.linspace(0, 1e-2, 9)] * 3
    axes[axis] = nodes
    grid = GridLines(**dict(zip("xyz", axes)))
    bc = CPMLBoundary("xyz"[axis] + side, grid, thickness_cells=3, alpha_max=alpha)
    dt = 1e-12
    with np.errstate(divide="raise", invalid="raise"):
        bc.initialize(dt, xp=np)
    indices = bc.pml_axis_indices
    interface = nodes[3] if side == "min" else nodes[-4]
    thickness = abs(interface - (nodes[0] if side == "min" else nodes[-1]))
    rho = abs(nodes[indices] - interface) / thickness
    sigma = -4 * C0 * EPS0 * np.log(bc.R_target) / (2 * thickness) * rho**3
    kappa = 1 + 6 * rho**3
    shift = alpha * (1 - rho)
    expected_b = np.exp(-(sigma / kappa + shift) * dt / EPS0)
    expected_c = np.zeros_like(sigma)
    active = sigma > 0
    expected_c[active] = (
        sigma[active]
        * (1 - expected_b[active])
        / (kappa[active] * sigma[active] + kappa[active] ** 2 * shift[active])
    )
    np.testing.assert_allclose(bc._b_E, expected_b, rtol=2e-14, atol=0)
    np.testing.assert_allclose(bc._c_E, expected_c, rtol=2e-14, atol=0)
    np.testing.assert_allclose(bc._ck_E, 1 - 1 / kappa, rtol=2e-14, atol=0)


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_updates_commute_with_spatial_reflection(axis, graded, dtype):
    widths = np.array([0.5, 0.7, 1.1, 1.4, 1.4, 1.1, 0.7, 0.5]) * 1e-3
    if not graded:
        widths[:] = 1e-3
    axes = [np.concatenate(([0.0], np.cumsum(widths)))] * 3
    grid = GridLines(**dict(zip("xyz", axes)))
    a = CPMLBoundary("xyz"[axis] + "min", grid, thickness_cells=3)
    b = CPMLBoundary("xyz"[axis] + "max", grid, thickness_cells=3)
    for bc in (a, b):
        bc.initialize(1e-12, xp=np, dtype=dtype)
    left = FieldArrays.zeros(8, 8, 8, xp=np, dtype=dtype)
    right = FieldArrays.zeros(8, 8, 8, xp=np, dtype=dtype)
    rng = np.random.default_rng(23)
    names = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    signs = {}
    for name in names:
        signs[name] = (-1 if "xyz".index(name[1].lower()) == axis else 1) * (
            1 if name[0] == "E" else -1
        )
        values = getattr(left, name)
        values[:] = rng.standard_normal(values.shape)
    a.apply(left)
    for name in names:
        getattr(right, name)[:] = signs[name] * np.flip(getattr(left, name), axis=axis)
    beta_e = np.ones(left.e_flat.size, dtype=dtype)
    beta_h = np.ones(left.h_flat.size, dtype=dtype)
    for _ in range(3):
        a.update_E(left, beta_e)
        b.update_E(right, beta_e)
        a.apply(left)
        b.apply(right)
        a.update_H(left, beta_h)
        b.update_H(right, beta_h)
        for name in names:
            np.testing.assert_allclose(
                getattr(right, name),
                signs[name] * np.flip(getattr(left, name), axis=axis),
                rtol=0,
                atol=1e-6 if dtype == np.float32 else 1e-14,
            )


def test_old_checkpoint_reinstates_cell_sampled_e_and_window_masks():
    grid = GridLines(**{axis: np.linspace(0, 1e-2, 9) for axis in "xyz"})
    bc = CPMLBoundary("xmin", grid, thickness_cells=3)
    bc.initialize(1e-12, xp=np)
    bc.set_port_windows([{1: (2, 5), 2: (2, 5)}])
    modern = bc.state_dict()
    old = {name: value for name, value in modern.items() if name != "profile_sampling"}
    bc.load_state_dict(old)
    assert bc.state_dict()["profile_sampling"] == 0
    np.testing.assert_array_equal(bc._b_E, bc._b)
    np.testing.assert_array_equal(bc._c_E, bc._c)
    assert np.all(bc._c_E1[:, 2:5, 2:6] == 0)
    bc.load_state_dict(modern)
    assert bc.state_dict()["profile_sampling"] == 1
    assert np.any(bc._c_E != bc._c)
    assert np.all(bc._c_E1[:, 2:5, 2:6] == 0)
