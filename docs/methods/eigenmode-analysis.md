# Eigenmode analysis

## 3D cavity eigenmode solver

Resonant modes are computed from the discrete curl-curl generalised
eigenvalue problem

$$
\mathbf C^{\mathsf T} \mathbf M_\mu^{-1} \mathbf C\, \hat e
 = \omega^2\, \mathbf M_\varepsilon\, \hat e
$$

(`solver/eigenmode_3d.py`), the standard FIT eigenformulation
{cite}`weiland1996,clemensweiland2001`.  PEC walls
are imposed by degree-of-freedom elimination, which also removes the
gradient null space for all-PEC cavities (DD-009); PMC walls are the
natural boundary condition (DD-065).

The default backend is **ARPACK shift-invert Lanczos**
(`scipy.sparse.linalg.eigsh` with a SuperLU factorisation of
$A - \sigma B$; DD-007), i.e. the implicitly restarted Arnoldi/Lanczos
method of Lehoucq, Sorensen and Yang {cite}`arpack1998`.
The shift $\sigma$ is auto-estimated
boundary-condition-aware, an in-house heuristic (DD-010).

Two experimental backends exist (DD-033):

- a CHOLMOD Cholesky path with **tree-cotree gauging** to eliminate
  the gradient null space; tree-cotree/spanning-tree gauging of
  curl-curl systems is established FEM practice, e.g. Albanese and
  Rubinacci {cite}`albaneserubinacci1990` and Manges
  and Cendes {cite}`mangescendes1995`; CHOLMOD is
  {cite}`chen2008cholmod`;
- an AMG-preconditioned path via pyamg {cite}`pyamg2023`,
  documented as not recommended (scalar
  smoothed-aggregation AMG does not achieve mesh-independent
  convergence on the vector curl-curl operator — a known limitation
  in the literature on AMG for Maxwell problems).

### Periodic structures: Bloch boundaries and the dispersion diagram

A face pair declared `"Periodic"` turns the cavity problem into a
unit-cell problem of an infinite periodic structure.  The field in
one period leads the next by a **phase advance** $\varphi$ (Floquet's
theorem, in the periodic-waveguide form given by Collin
{cite}`collin1991`): on the far plane of the period the tangential
electric edge voltages are those of the near plane times
$e^{-\mathrm j\varphi}$.  The solver imposes this by a congruence
transformation — a projector $\mathbf P$ identifies every far-plane
edge with its near-plane image (times the phase factor), and the
reduced operators $\mathbf P^{\mathsf H} \mathbf A \mathbf P$,
$\mathbf P^{\mathsf H} \mathbf B \mathbf P$ are solved with the
same shift-invert machinery.  The far plane contributes no material
metric of its own: the FIT material matrices book a full dual cell on
every domain face, and that full cell stands for the identified pair.

For $\varphi = 0$ and $\varphi = \pi$ the projector is real and the
problem stays real symmetric; in between it is complex Hermitian and
the eigenvectors are travelling Bloch modes, which only the SuperLU
backend solves.  Sweeping $\varphi$ from $0$ to $\pi$ traces the
**dispersion (Brillouin) diagram** $f(\varphi)$ of the structure; the
band edges $\varphi = 0$ and $\varphi = \pi$ coincide with the classic
half-cell calculations (electric or magnetic wall at the cell
boundary), which is the check the implementation is held to.
Verified against the discrete dispersion relation of the empty
periodic box (exact to solver tolerance for $\varphi$ between 0 and
180 degrees, `tests/integration/test_floquet_eigenmode.py`).

Quality factors of eigenmodes are evaluated with the perturbative
wall-loss route (see [conductor losses](conductor-losses.md))
{cite}`pozar2012,jackson1999`.

## 2D mode solver

The port-plane 2D eigenmode machinery (curl-curl restriction, TEM/QTEM
Laplace) is described in the [ports chapter](ports.md); it shares the
matrices and the ARPACK backend with the 3D solver.
