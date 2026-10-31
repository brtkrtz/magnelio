# Implementation engineering (non-method)

The items in this chapter are software engineering, not numerical
research methods; they are listed for completeness and to delimit the
citation apparatus of the previous chapters.

## Kernel dispatch and backends

Three-tier kernel dispatch: NumPy reference kernels,
Numba-JIT CPU stencil kernels {cite}`numba2015`, and
CUDA kernels via CuPy {cite}`cupy2017` with
`backend="auto"` GPU selection.  GPU step orchestration
(device-resident recorder staging, fused port-plane transfers, CUDA
graph capture of the device phases) is performance
engineering. Dedicated CPU/GPU comparisons cover selected workloads;
general device-port coverage remains limited.
The CPU kernels sweep the grid plane by plane, updating all three
field components per plane so every field array streams from memory
once per half-step; at production mesh sizes they move about 85 % of
the bandwidth a STREAM triad reaches from Numba on the same CPU,
which is the practical ceiling short of temporal blocking.

## Precision

Selectable single/double precision for the whole time-loop state.
The switch, its cost and its accuracy consequences have
their own chapter — see [numerical precision](precision.md).

## Parallel mesh building

The CSG/section pipeline parallelises cross-section extraction and
face accounting (process pool with cost-aware scheduling, Numba
polygon kernels).  Engineering only.

## Dependencies with numerical relevance

- NumPy/SciPy: sparse matrices, `eigsh` (ARPACK), `spsolve` (SuperLU),
  `nnls` (Lawson–Hanson {cite}`lawsonhanson1974`).
- pythonocc-core / Open CASCADE: geometry kernel.
- h5py/HDF5, VTK: storage and visualisation formats.
