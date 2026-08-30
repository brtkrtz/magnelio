"""Deterministic ARPACK start vector, shared by every sparse eigensolve.

Left to itself ARPACK starts from a random vector, so the converged
eigenpairs carry a residual that differs from one run to the next.  Two
independent findings in this codebase trace back to that:

* the TE/TM-vs-TEM crosstalk on the coax fixture wandered over
  3.1e-16 ... 1.1e-13 across rebuilds of the *same* port (KB-010), which
  is physically zero either way but occasionally crossed a 1e-12
  assertion;
* the stability time step must be bit-identical across rebuilds of the
  same mesh, because a project store resumes bit-exactly (DD-142).

Both were fixed in place, with the same three lines, in two different
modules.  This is the third caller, so the vector lives here.

The direction is arbitrary but must not be *structured*: a vector of
ones can sit orthogonal to a mode of interest and starve it.
"""

from __future__ import annotations

import numpy as np


def arpack_v0(n: int) -> np.ndarray:
    """Fixed generic ARPACK start vector for an ``n``-dimensional solve."""
    return np.random.default_rng(0).standard_normal(n)
