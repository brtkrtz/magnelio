"""Vector fitting of tabulated frequency responses.

In-repo implementation of the Gustavsen/Semlyen vector-fitting
iteration [Gustavsen & Semlyen, IEEE Trans. Power Delivery 14(3), 1999]
specialised to the scalar pole-residue form used by
:class:`~magnelio.materials.dispersion.DispersionModel`,

    f(s) = d + sum_p r_p / (s - a_p),        s = j*omega,

with a real constant term ``d`` (the eps_inf of a permittivity fit) and
poles that are real or come in complex-conjugate pairs.  Only NumPy
linear algebra is used — no new dependency.

The fit itself enforces *stability* (unstable poles are flipped into
the left half-plane each iteration, the standard VF rule).  It does NOT
enforce *passivity* — that is the job of the ``DispersionModel``
constructor, which acts as the mandatory acceptance filter for fits
from measured data.
"""

# Design: DD-086 (in-repo vector fitting), DD-083 (DispersionModel passivity
# filter).

from __future__ import annotations

import numpy as np

# Relative pole movement below which the pole-relocation iteration is
# considered converged.  VF typically converges in < 10 iterations on
# smooth (Kramers-Kronig-consistent) permittivity data.
_POLE_TOL = 1e-8
_MAX_ITER = 30


def _basis(s: np.ndarray, poles: np.ndarray) -> np.ndarray:
    """Real-coefficient partial-fraction basis matrix (complex valued).

    For a real pole ``a``: one column ``1/(s - a)``.
    For a conjugate pair stored as ``a`` with Im(a) > 0: two columns
    ``1/(s-a) + 1/(s-a*)`` and ``j/(s-a) - j/(s-a*)``, so that real
    coefficients ``(c', c'')`` represent the residue ``r = c' + j c''``
    (with ``r*`` implied on the conjugate pole).
    """
    cols = []
    for a in poles:
        if a.imag == 0.0:
            cols.append(1.0 / (s - a))
        else:
            p = 1.0 / (s - a)
            q = 1.0 / (s - np.conj(a))
            cols.append(p + q)
            cols.append(1j * (p - q))
    return np.column_stack(cols)


def _coeffs_to_residues(poles: np.ndarray, x: np.ndarray) -> list[complex]:
    """Map the real coefficient vector back to complex residues."""
    res = []
    k = 0
    for a in poles:
        if a.imag == 0.0:
            res.append(complex(x[k]))
            k += 1
        else:
            res.append(complex(x[k], x[k + 1]))
            k += 2
    return res


def _n_coeffs(poles: np.ndarray) -> int:
    return int(sum(1 if a.imag == 0.0 else 2 for a in poles))


def _relocated_poles(poles: np.ndarray, c_hat: np.ndarray) -> np.ndarray:
    """Zeros of ``sigma(s) = 1 + sum c_hat_p phi_p(s)`` (new pole set).

    Standard VF: eigenvalues of ``A - b c^T`` with the real block form
    for conjugate pairs.
    """
    n = _n_coeffs(poles)
    A = np.zeros((n, n))
    b = np.zeros(n)
    k = 0
    for a in poles:
        if a.imag == 0.0:
            A[k, k] = a.real
            b[k] = 1.0
            k += 1
        else:
            A[k, k] = a.real
            A[k, k + 1] = a.imag
            A[k + 1, k] = -a.imag
            A[k + 1, k + 1] = a.real
            b[k] = 2.0
            k += 2
    new = np.linalg.eigvals(A - np.outer(b, c_hat))
    # flip unstable poles into the left half-plane (VF rule), fold onto
    # the upper half-plane
    new = np.where(new.real > 0.0, -new.real + 1j * new.imag, new)
    new = np.where(new.imag < 0.0, np.conj(new), new)
    # A real state matrix has eigenvalues in exact conjugate duos: after
    # the fold each complex pair appears twice.  Keep every second of
    # the sorted complex ones (duo partners are FP-adjacent), all reals.
    is_real = np.abs(new.imag) <= 1e-9 * np.abs(new)
    out = [complex(p.real) for p in sorted(new[is_real], key=lambda p: p.real)]
    cplx = sorted(new[~is_real], key=lambda p: (p.imag, p.real))
    out.extend(complex(p) for p in cplx[::2])
    return np.asarray(out, dtype=complex)


def _solve_real_ls(M: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Least squares on the [Re; Im]-stacked real system."""
    A = np.vstack([M.real, M.imag])
    y = np.concatenate([rhs.real, rhs.imag])
    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    return x


def vector_fit(
    omega: np.ndarray,
    values: np.ndarray,
    n_poles: int,
    start: str = "complex",
) -> tuple[float, list[tuple[complex, complex]], float]:
    """Fit ``values(j*omega)`` to ``d + sum r_p/(s - a_p)``.

    Parameters
    ----------
    omega : np.ndarray
        Angular frequencies [rad/s], ascending, > 0.
    values : np.ndarray
        Complex samples of the target function at *omega*.
    n_poles : int
        Model order counting conjugate partners individually (an
        underdamped pair counts as 2).
    start : str
        Starting pole set (the standard Gustavsen recommendation):
        ``"complex"`` — log-spaced weakly damped conjugate pairs
        (resonant data); ``"real"`` — log-spaced real poles (smooth
        relaxation data).  The iteration relocates freely between real
        and paired poles either way; the start only steers convergence.

    Returns
    -------
    (d, poles, rel_err)
        Constant term, ``(a_p, r_p)`` list (conjugate pairs stored once
        with Im(a) > 0), and the maximum relative deviation
        ``max |fit - values| / max |values|`` over the table.
    """
    omega = np.asarray(omega, dtype=float)
    values = np.asarray(values, dtype=complex)
    s = 1j * omega

    if start == "real":
        beta = np.logspace(
            np.log10(omega[0]),
            np.log10(omega[-1]),
            int(n_poles),
        )
        poles = [complex(-b) for b in beta]
    elif start == "complex":
        # weakly damped pairs across the span (+ one real for odd n)
        n_pairs, n_real = divmod(int(n_poles), 2)
        beta = np.logspace(
            np.log10(omega[0]),
            np.log10(omega[-1]),
            max(n_pairs, 1),
        )[:n_pairs]
        poles = [complex(-b / 100.0, b) for b in beta]
        if n_real:
            poles.append(complex(-np.sqrt(omega[0] * omega[-1])))
    else:
        raise ValueError(f"vector_fit: unknown start {start!r}")
    poles = np.asarray(poles, dtype=complex)

    for _ in range(_MAX_ITER):
        phi = _basis(s, poles)
        n = phi.shape[1]
        # columns: [c (n), d (1), c_hat (n)];  rows: one per sample
        M = np.hstack([phi, np.ones((s.size, 1)), -values[:, None] * phi])
        x = _solve_real_ls(M, values)
        c_hat = x[n + 1 :]
        new_poles = _relocated_poles(poles, c_hat)
        if new_poles.size == poles.size:
            move = np.max(np.abs(np.sort_complex(new_poles) - np.sort_complex(poles)))
            scale = float(np.max(np.abs(poles)))
            poles = new_poles
            if move <= _POLE_TOL * scale:
                break
        else:  # pair split/merged — keep iterating
            poles = new_poles

    # Final residue fit with the relocated poles held fixed.
    phi = _basis(s, poles)
    M = np.hstack([phi, np.ones((s.size, 1))])
    x = _solve_real_ls(M, values)
    n = phi.shape[1]
    d = float(x[n])
    residues = _coeffs_to_residues(poles, x[:n])

    fit = np.full(s.shape, d, dtype=complex)
    for a, r in zip(poles, residues):
        fit += r / (s - a)
        if a.imag != 0.0:
            fit += np.conj(r) / (s - np.conj(a))
    rel_err = float(np.max(np.abs(fit - values)) / np.max(np.abs(values)))

    return d, list(zip((complex(a) for a in poles), residues)), rel_err
