"""The ARPACK request grows at one shift when null-space artefacts crowd it (DD-195).

At a user-pinned shift the shift ladder has one rung, so a Krylov
request in which most vectors converge on the curl-curl null space
used to hand back fewer modes than asked for — no retry.  Now the
request grows at the same shift by the artefact count (twice at
most), sharing one SuperLU factorisation.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from magnelio.analysis.eigenmode import AnalysisEigenmode
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.solver import _eigenmode_3d as eig3d

C0 = 299_792_458.0
A, B, C = 30e-3, 20e-3, 15e-3


def _cavity():
    grid = GridLines(x=np.linspace(0, A, 11), y=np.linspace(0, B, 9), z=np.linspace(0, C, 7))
    return Mesh.from_grid(grid)


def _f110():
    return (C0 / 2) * math.sqrt((1 / A) ** 2 + (1 / B) ** 2)


def test_request_grows_until_the_modes_are_there(monkeypatch):
    mesh = _cavity()
    sigma = (2 * math.pi * _f110()) ** 2
    reference = AnalysisEigenmode(mesh=mesh, n_modes=3, sigma=sigma, verbose=False).run()
    f_ref = np.asarray(reference.frequencies)
    assert f_ref.size == 3

    calls: list[int] = []
    nulls: list[int] = []
    real = eig3d.EigenmodeSolver3D._solve_arpack
    floor = (2 * math.pi * eig3d._F_PHYSICAL_MIN) ** 2

    def crowded(self, A_f, B_f, sigma, n_free, k_request, op_inv=None):
        """First call: all but one physical vector replaced by artefacts."""
        calls.append(k_request)
        vals, vecs = real(self, A_f, B_f, sigma, n_free, k_request, op_inv=op_inv)
        if len(calls) == 1:
            vals = np.array(vals, copy=True)
            vals[:-1] = 0.0  # below the 1 MHz floor: null-space artefacts
            nulls.append(int(np.count_nonzero(vals < floor)))
        return vals, vecs

    monkeypatch.setattr(eig3d.EigenmodeSolver3D, "_solve_arpack", crowded)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = AnalysisEigenmode(mesh=mesh, n_modes=3, sigma=sigma, verbose=False).run()
    assert len(calls) == 2 and calls[1] > calls[0]
    assert calls[1] == calls[0] + nulls[0] + 2
    np.testing.assert_allclose(np.asarray(result.frequencies), f_ref, rtol=1e-9)


def test_growth_is_bounded_and_the_shortfall_stays_loud(monkeypatch):
    mesh = _cavity()
    sigma = (2 * math.pi * _f110()) ** 2
    calls: list[int] = []
    real = eig3d.EigenmodeSolver3D._solve_arpack

    floor = (2 * math.pi * eig3d._F_PHYSICAL_MIN) ** 2

    def always_crowded(self, A_f, B_f, sigma, n_free, k_request, op_inv=None):
        """Every call: only the lowest physical vector survives."""
        calls.append(k_request)
        vals, vecs = real(self, A_f, B_f, sigma, n_free, k_request, op_inv=op_inv)
        vals = np.array(vals, copy=True)
        keep = int(np.argmax(vals > floor))
        vals[np.arange(vals.size) != keep] = 0.0
        return vals, vecs

    monkeypatch.setattr(eig3d.EigenmodeSolver3D, "_solve_arpack", always_crowded)
    with pytest.warns(RuntimeWarning, match="returned 1 of 3"):
        result = AnalysisEigenmode(mesh=mesh, n_modes=3, sigma=sigma, verbose=False).run()
    assert len(calls) == eig3d._NULL_GROW_RETRIES + 1
    assert all(b > a for a, b in zip(calls, calls[1:]))
    assert len(result.frequencies) == 1


def test_shared_factorisation_reproduces_eigsh():
    from scipy.sparse.linalg import eigsh

    mesh = _cavity()
    solver = eig3d.EigenmodeSolver3D(n_modes=3, verbose=False)
    # Assemble the free-DOF problem the way the solver does, via one
    # unpatched solve that we intercept.
    captured = {}
    real = eig3d.EigenmodeSolver3D._solve_arpack

    def grab(self, A_f, B_f, sigma, n_free, k_request, op_inv=None):
        captured.update(A_f=A_f, B_f=B_f, sigma=sigma, k=k_request)
        return real(self, A_f, B_f, sigma, n_free, k_request, op_inv=op_inv)

    eig3d.EigenmodeSolver3D._solve_arpack = grab
    try:
        solver.solve(mesh)
    finally:
        eig3d.EigenmodeSolver3D._solve_arpack = real
    A_f, B_f, sigma, k = captured["A_f"], captured["B_f"], captured["sigma"], captured["k"]
    plain, _ = eigsh(A_f, M=B_f, k=k, which="LM", sigma=sigma)
    shared, _ = eigsh(
        A_f,
        M=B_f,
        k=k,
        which="LM",
        sigma=sigma,
        OPinv=eig3d.EigenmodeSolver3D._arpack_op_inv(A_f, B_f, sigma),
    )
    floor = (2 * math.pi * eig3d._F_PHYSICAL_MIN) ** 2
    plain_phys = np.sort(plain[plain > floor])
    shared_phys = np.sort(shared[shared > floor])
    # The null-space residues (|λ| ~ 1e7 against physical 1e21) are
    # numerical noise on both paths; the physical pairs agree.
    np.testing.assert_allclose(shared_phys, plain_phys, rtol=1e-8)
