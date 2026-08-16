# Implementation engineering (non-method)

The items in this chapter are software engineering, not numerical
research methods; they are listed for completeness and to delimit the
citation apparatus of the previous chapters.

## Kernel dispatch and backends

Three-tier kernel dispatch (DD-032): NumPy reference kernels,
Numba-JIT CPU stencil kernels {cite}`numba2015`, and
CUDA kernels via CuPy {cite}`cupy2017` with
`backend="auto"` GPU selection (DD-090).  GPU step orchestration
(device-resident recorder staging, fused port-plane transfers, CUDA
graph capture of the device phases, DD-092) is performance
engineering.  S-parameters on GPU are gated bit-exact against CPU.

## Precision

Selectable single/double precision for the whole time-loop state
(DD-094); see the [discretisation chapter](fit-discretization.md)
for the numerical argument.

## Parallel mesh building

The CSG/section pipeline parallelises cross-section extraction and
face accounting (process pool with cost-aware scheduling, Numba
polygon kernels).  Engineering only.

## Dependencies with numerical relevance

- NumPy/SciPy: sparse matrices, `eigsh` (ARPACK), `spsolve` (SuperLU),
  `nnls` (Lawson–Hanson {cite}`lawsonhanson1974`).
- pythonocc-core / Open CASCADE: geometry kernel.
- h5py/HDF5, VTK, XDMF: storage and visualisation formats.
