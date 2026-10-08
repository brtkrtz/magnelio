"""CPML pulse-reflection and opposing-face certificate (DD-286).

Split a transverse Gaussian field into counter-propagating vacuum pulses.
Compare with a larger domain whose walls cannot return during the recording.
Use the same analytic time step and binary-exact grid spacing in both domains.
The 1--12 GHz spectral gate measures the reflected signal after the incident
pulse has passed; it is a fixture certificate, not a universal R_target bound.

Run from the repository root with the mio interpreter.
"""

import math

import numpy as np

from magnelio.boundaries.cpml import CPMLBoundary
from magnelio.constants import C0
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.solver.fit_td import FITTimeDomainSolver


class _Trace:
    def __init__(self, axis, component, indices):
        self.axis = axis
        self.component = component
        self.indices = indices
        self.values = []

    def record(self, fields, n, t, dt):
        index = [0, 0, 0]
        index[self.axis] = self.indices
        self.values.append(np.asarray(getattr(fields, self.component)[tuple(index)]).copy())


def _pulse(axis, cells, thickness, precision="double"):
    d = 2.0**-10
    electric_axis = (axis + 1) % 3
    component = "E" + "xyz"[electric_axis]
    axes = [np.array([0.0, d]) for _ in range(3)]
    axes[axis] = np.arange(cells + 1) * d
    grid = GridLines(**dict(zip("xyz", axes)))
    bc = {}
    for a in range(3):
        kind = "PEC" if a == electric_axis or a == axis else "PMC"
        for side in ("min", "max"):
            face = "xyz"[a] + side
            bc[face] = (
                CPMLBoundary(face, grid, thickness_cells=thickness)
                if a == axis and thickness
                else kind
            )
    dt = 0.9 * d / (C0 * np.sqrt(3))
    trace = _Trace(axis, component, [cells // 2 - 40, cells // 2 + 40])
    solver = FITTimeDomainSolver(
        mesh=Mesh.from_grid(grid, boundary_conditions=bc),
        boundary_conditions=bc,
        dt=dt,
        total_time_steps=math.ceil(0.95e-9 / dt),
        monitors=[trace],
        precision=precision,
        backend="numpy",
        verbose=False,
    )
    solver.setup()
    shape = [1, 1, 1]
    shape[axis] = cells + 1
    pulse = np.exp(-(((axes[axis] - cells * d / 2) / (6 * d)) ** 2))
    getattr(solver._fields, component)[:] = pulse.reshape(shape) * d
    solver.run()
    return np.asarray(trace.values), dt


def certify():
    for axis in range(3):
        reference, dt = _pulse(axis, 600, 0)
        t = (np.arange(len(reference)) + 1) * dt
        incident = t < 0.3e-9
        kernel = np.exp(-2j * np.pi * np.linspace(1e9, 12e9, 111)[:, None] * t[None, :])
        incident_spectrum = kernel[:, incident] @ reference[incident]
        peak = float(np.abs(reference).max())
        assert peak > 0
        for thickness, limit_db in ((8, -65.0), (16, -90.0), (24, -100.0)):
            values, _ = _pulse(axis, 240, thickness)
            reflected = kernel[:, ~incident] @ (values - reference)[~incident]
            reflection_db = 20 * np.log10(np.abs(reflected / incident_spectrum).max())
            mirror_error = float(np.abs(values[:, 0] - values[:, 1]).max()) / peak
            print(
                f"{'xyz'[axis]}: {thickness:2d} cells, "
                f"worst reflection {reflection_db:.2f} dB, mirror defect {mirror_error:.3e}"
            )
            assert reflection_db < limit_db
            assert mirror_error < 2e-6


if __name__ == "__main__":
    certify()
