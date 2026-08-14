"""Causal rational surface impedance for the TD-SIBC.

Realises the Leontovich surface impedance as a Foster/Stieltjes ladder

    Z(s) = c0 + sum_p c_p * s / (s + b_p),      c_p >= 0, b_p > 0,

fitted by non-negative least squares (NNLS) on fixed log-spaced pole
frequencies spanning the working band plus a guard factor beyond it on
both sides.  Every ladder branch is elementarily passive (a series-R +
parallel-RL element), so the time-domain recursion built on this form
is dissipative by construction — no numerical positive-real test is
load-bearing for stability (internal dossier `investigations/sibc/DERIVATION.md` §2/§5,
with the measured accuracy record in ``fit_feasibility_probe.py``).

Targets:

* **Smooth metal** — the closed form ``Z_s = sqrt(j w mu / sigma)``,
  fitted directly (no completion needed).
* **Causal roughness** — the surface-roughness models supply only the
  REAL part ``R_rough = K(f) * R_s,smooth``; the real factor is
  non-causal as a TD impedance.  The causal reactance is completed by
  a subtracted Kramers-Kronig quadrature of the roughness EXCESS
  ``(K - 1) * R_s`` (better-behaved tails than the full R: the excess
  vanishes as f^{3/2} at low frequency), added to the exact smooth
  reactance ``X_s = R_s``.

Numerical guard rails, measured in the fitting probe (internal
record, see above) and binding here:
all fitting runs in normalised coordinates (``w/w0``, ``Z/Z0``) — the
raw physical scales lose 3 orders of magnitude of accuracy to column
conditioning alone — and the NNLS iteration cap is raised above the
scipy default, which was measured to trip on rough targets.
"""

# Design: DD-088 (roughness models and their non-causality finding);
# work packages WP-D1 (fit feasibility probe) / WP-D2 (this module).

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import nnls

from magnelio.constants import MU0  # noqa: E402
from magnelio.materials.roughness import SurfaceRoughness


def smooth_surface_impedance(f, sigma: float, mu: float = 1.0) -> np.ndarray:
    """Exact smooth-conductor surface impedance ``sqrt(j w mu / sigma)``.

    Parameters
    ----------
    f : array_like
        Frequency [Hz].
    sigma : float
        Conductivity [S/m].
    mu : float, optional
        Relative permeability of the conductor.

    Returns
    -------
    np.ndarray
        Complex ``Z_s`` [Ohm]; ``Re Z_s == Im Z_s == R_s(f)``.
    """
    w = 2.0 * np.pi * np.asarray(f, dtype=float)
    return np.sqrt(1j * w * MU0 * mu / sigma)


def kk_reactance(
    f_eval,
    r_of_f,
    f_lo: float,
    f_hi: float,
    guard_decades: float = 8.0,
    n_grid: int = 200_000,
) -> np.ndarray:
    """Hilbert-transform a surface-resistance real part into its
    causal reactance (subtracted Kramers-Kronig quadrature).

        X(w) = (2w/pi) PV int_0^inf [R(t) - R(w)] / (t^2 - w^2) dt

    on a dense log grid spanning ``guard_decades`` beyond ``[f_lo,
    f_hi]`` on both sides.  After the subtraction the integrand is
    smooth at ``t = w`` (removable singularity); the node closest to
    the pole is replaced by the analytic limit ``R'(w) / (2w)`` with a
    central-difference derivative.  Accuracy is gated against the
    smooth closed form (``X = R`` exactly) in the unit suite.

    Parameters
    ----------
    f_eval : array_like
        Frequencies [Hz] at which the reactance is needed.
    r_of_f : callable
        Vectorised real part ``R(f)`` [Ohm].  Must be evaluable on the
        whole guard-extended grid.
    f_lo, f_hi : float
        Working band [Hz] — sets the grid centre.
    guard_decades : float, optional
        Log-decades of grid on each side beyond the band.
    n_grid : int, optional
        Quadrature nodes.

    Returns
    -------
    np.ndarray
        ``X(f_eval)`` [Ohm].
    """
    f_eval = np.asarray(f_eval, dtype=float)
    t = (
        2.0
        * np.pi
        * np.logspace(
            np.log10(f_lo) - guard_decades,
            np.log10(f_hi) + guard_decades,
            int(n_grid),
        )
    )
    r_t = np.asarray(r_of_f(t / (2.0 * np.pi)), dtype=float)

    x = np.empty(f_eval.shape, dtype=float)
    for i, f in enumerate(f_eval.ravel()):
        w = 2.0 * np.pi * f
        r_w = float(r_of_f(np.array([f]))[0])
        den = (t - w) * (t + w)
        num = r_t - r_w
        j = int(np.argmin(np.abs(t - w)))
        integrand = num / np.where(np.arange(t.size) == j, 1.0, den)
        # removable-singularity node: limit [R(t)-R(w)]/(t^2-w^2) ->
        # R'(w)/(2w), central difference on the angular axis
        dr = (float(r_of_f(np.array([1.001 * f]))[0]) - float(r_of_f(np.array([0.999 * f]))[0])) / (
            0.002 * w
        )
        integrand[j] = dr / (2.0 * w)
        x.ravel()[i] = (2.0 * w / np.pi) * np.trapezoid(integrand, t)
    return x


def _foster_nnls(w_band, z_target, n_branches: int, guard: float):
    """One NNLS ladder fit in normalised coordinates.

    Returns ``(c0, b, c)`` on PHYSICAL scales.  Normalisation is exact:
    ``s/(s + b)`` is invariant under ``s -> s/w0, b -> b/w0`` and the
    coefficients scale linearly with ``Z0`` (DERIVATION.md §2,
    conditioning trap).
    """
    w0 = float(np.sqrt(w_band[0] * w_band[-1]))
    z0 = float(np.max(np.abs(z_target)))
    sn = 1j * w_band / w0
    bn = (2.0 * np.pi / w0) * np.logspace(
        np.log10(w_band[0] / (2.0 * np.pi) / guard),
        np.log10(w_band[-1] / (2.0 * np.pi) * guard),
        int(n_branches),
    )
    zn = z_target / z0
    weight = 1.0 / np.abs(zn)
    cols = [sn / (sn + b) for b in bn] + [np.ones_like(sn)]
    m = np.column_stack(cols) * weight[:, None]
    m_r = np.vstack([m.real, m.imag])
    rhs = np.concatenate([(zn * weight).real, (zn * weight).imag])
    coeff, _ = nnls(m_r, rhs, maxiter=200 * m_r.shape[1])
    return z0 * coeff[-1], bn * w0, z0 * coeff[:-1]


@dataclass(frozen=True)
class SurfaceImpedanceFit:
    """Passive rational surface impedance ``c0 + sum c_p s/(s+b_p)``.

    Frozen (joins ``Material`` equality and the store the way the
    roughness models do); built by :func:`fit_surface_impedance`.

    Parameters
    ----------
    sigma : float
        Conductivity [S/m] of the fitted conductor.
    mu : float
        Relative permeability.
    roughness : SurfaceRoughness or None
        Roughness model whose causal completion this fit carries.
    f_lo, f_hi : float
        Working band [Hz]; accuracy statements hold on this band.
    c0 : float
        Instantaneous (high-frequency) resistance term [Ohm].
    branches : tuple of (float, float)
        ``(b_p [rad/s], c_p [Ohm])`` ladder branches, all positive.
    rel_err_re : float
        Achieved max relative deviation of ``Re Z`` from the target
        surface resistance over the band.
    rel_err_cplx : float
        Achieved max relative deviation of the complex fit from the
        (causally completed) target over the band.
    """

    sigma: float
    mu: float
    roughness: SurfaceRoughness | None
    f_lo: float
    f_hi: float
    c0: float
    branches: tuple[tuple[float, float], ...]
    rel_err_re: float = field(compare=False)
    rel_err_cplx: float = field(compare=False)

    def impedance(self, f) -> np.ndarray:
        """Evaluate the rational ``Z(j 2 pi f)`` [Ohm]."""
        s = 2j * np.pi * np.asarray(f, dtype=float)
        z = np.full(s.shape, complex(self.c0))
        for b_p, c_p in self.branches:
            z = z + c_p * s / (s + b_p)
        return z

    @property
    def r_instantaneous(self) -> float:
        """High-frequency limit ``Z(inf) = c0 + sum c_p`` [Ohm]."""
        return self.c0 + sum(c_p for _, c_p in self.branches)


def fit_surface_impedance(
    sigma: float,
    mu: float = 1.0,
    roughness: SurfaceRoughness | None = None,
    f_lo: float = 1e8,
    f_hi: float = 1e11,
    tol: float = 1e-3,
    guard: float = 10.0,
    max_branches: int = 32,
) -> SurfaceImpedanceFit:
    """Fit a passive Foster ladder to a conductor's surface impedance.

    The branch count is chosen by an acceptance loop: the smallest
    ladder whose ``Re Z`` matches the target surface resistance within
    ``tol`` (max relative, on a denser check grid than the fit grid)
    is returned; failure to reach ``tol`` at ``max_branches`` raises.

    Parameters
    ----------
    sigma : float
        Conductivity [S/m].
    mu : float, optional
        Relative permeability of the conductor.
    roughness : SurfaceRoughness, optional
        Roughness model; its real factor ``K(f)`` is causally completed
        via :func:`kk_reactance` on the roughness excess.
    f_lo, f_hi : float, optional
        Working band [Hz].
    tol : float, optional
        Acceptance bound on the band's ``Re Z`` relative error.
    guard : float, optional
        Pole-placement guard factor beyond the band (measured default;
        the top branches need room above the band to carry the sqrt
        growth).
    max_branches : int, optional
        Acceptance-loop cap.

    Returns
    -------
    SurfaceImpedanceFit

    Raises
    ------
    ValueError
        If ``tol`` is not reached at ``max_branches`` branches.
    """
    if sigma <= 0.0:
        raise ValueError(f"fit_surface_impedance: sigma must be > 0, got {sigma!r}")
    if not f_hi > f_lo > 0.0:
        raise ValueError(f"fit_surface_impedance: need 0 < f_lo < f_hi, got {f_lo!r}, {f_hi!r}")

    decades = np.log10(f_hi / f_lo)
    n_fit = max(60, int(100 * decades))
    f_band = np.logspace(np.log10(f_lo), np.log10(f_hi), n_fit)
    f_check = np.logspace(np.log10(f_lo), np.log10(f_hi), 2 * n_fit + 1)

    z_smooth_band = smooth_surface_impedance(f_band, sigma, mu)
    if roughness is None:
        z_band = z_smooth_band
        r_check = smooth_surface_impedance(f_check, sigma, mu).real
    else:

        def r_excess(f):
            k = roughness.factor(f, sigma, mu)
            return (k - 1.0) * smooth_surface_impedance(f, sigma, mu).real

        def r_full(f):
            k = roughness.factor(f, sigma, mu)
            return k * smooth_surface_impedance(f, sigma, mu).real

        x_band = z_smooth_band.imag + kk_reactance(
            f_band,
            r_excess,
            f_lo,
            f_hi,
        )
        z_band = r_full(f_band) + 1j * x_band
        r_check = r_full(f_check)

    for n_branches in range(4, max_branches + 1, 2):
        c0, b, c = _foster_nnls(2.0 * np.pi * f_band, z_band, n_branches, guard)
        fit = SurfaceImpedanceFit(
            sigma=float(sigma),
            mu=float(mu),
            roughness=roughness,
            f_lo=float(f_lo),
            f_hi=float(f_hi),
            c0=float(c0),
            branches=tuple((float(b_p), float(c_p)) for b_p, c_p in zip(b, c) if c_p > 0.0),
            rel_err_re=0.0,
            rel_err_cplx=0.0,
        )
        z_chk = fit.impedance(f_check)
        err_re = float(np.max(np.abs(z_chk.real - r_check) / r_check))
        z_fit_band = fit.impedance(f_band)
        err_cplx = float(np.max(np.abs(z_fit_band - z_band) / np.abs(z_band)))
        if err_re <= tol:
            return SurfaceImpedanceFit(
                sigma=fit.sigma,
                mu=fit.mu,
                roughness=fit.roughness,
                f_lo=fit.f_lo,
                f_hi=fit.f_hi,
                c0=fit.c0,
                branches=fit.branches,
                rel_err_re=err_re,
                rel_err_cplx=err_cplx,
            )
    raise ValueError(
        f"fit_surface_impedance: Re-part tolerance {tol:g} not reached "
        f"with {max_branches} branches over {f_lo:g}-{f_hi:g} Hz "
        f"(last error {err_re:.2e}); widen tol or raise max_branches"
    )


def fit_wall_impedances(
    resolved: dict,
    f_lo: float,
    f_hi: float,
    *,
    tol: float = 1e-3,
    guard: float = 10.0,
    max_branches: int = 32,
) -> dict:
    """Fit one surface impedance per wall tag, sharing identical fits.

    Completes the wall-tag resolution: takes the conductor properties
    of ``mesh.surfaces.resolve_wall_conductors`` and returns
    ``tag -> SurfaceImpedanceFit`` over the caller band.  Tags with
    identical ``(sigma, mu, roughness)`` — the common case of many
    walls in one metal — share ONE fit object (the roughness models are
    frozen/hashable, so the triple is a dict key), so the NNLS
    acceptance loop runs once per distinct conductor, not once per tag.

    Parameters
    ----------
    resolved : dict
        ``tag -> (sigma, mu, roughness)`` per
        ``resolve_wall_conductors``.
    f_lo, f_hi : float
        Working band [Hz] passed to :func:`fit_surface_impedance`.
    tol, guard, max_branches : optional
        Forwarded to :func:`fit_surface_impedance`.

    Returns
    -------
    dict
        ``tag -> SurfaceImpedanceFit``.
    """
    cache: dict = {}
    fits: dict = {}
    for tag, (sigma, mu, roughness) in resolved.items():
        key = (float(sigma), float(mu), roughness)
        if key not in cache:
            cache[key] = fit_surface_impedance(
                sigma,
                mu,
                roughness,
                f_lo=f_lo,
                f_hi=f_hi,
                tol=tol,
                guard=guard,
                max_branches=max_branches,
            )
        fits[tag] = cache[key]
    return fits
