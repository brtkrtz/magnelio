"""Pole-residue dispersion models for frequency-dependent permittivity.

ONE general mechanism covers every supported dispersive material:

    eps(omega) = eps_inf + sum_p  r_p / (j*omega - a_p)

with real poles (a_p, r_p real) and complex-conjugate pole pairs (stored
once with Im(a_p) > 0; the conjugate partner is implied).  Debye, Lorentz,
Drude and Djordjevic-Sarkar are *constructors* on this single form, not
separate models — the solver runs the same trapezoidal auxiliary
differential equation for all of them.

Passivity is checked at construction and is mandatory: an unstable run
from a non-passive pole set is the classic dispersive-FDTD failure mode,
and fits from measured data (C4) must be rejected here, not diagnosed
from a diverging field.  The check requires Re(a_p) <= 0 for every pole
(Re(a_p) = 0 only for the real Drude DC pole, which the trapezoidal
update integrates exactly as a conductivity) and eps''(omega) >= 0
sampled over the declared validity band ``f_band``.
"""

# Design: DD-083 (pole-residue dispersion form), DD-084 (trapezoidal ADE
# solver).

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_N_PASSIVITY_SAMPLES = 128
# Roundoff allowance for the band-sampled eps'' >= 0 check, relative to
# |eps(omega)|: large cancelling residues can leave eps'' at the
# floating-point floor of the partial-fraction sum.
_PASSIVITY_RTOL = 1e-9


@dataclass(frozen=True)
class DispersionModel:
    """Pole-residue permittivity model ``eps_inf + sum r_p/(j*omega - a_p)``.

    Parameters
    ----------
    eps_inf : float
        Relative permittivity in the high-frequency limit.  This is the
        value the host :class:`Material` must carry in ``epsilon`` — it
        drives the mass matrix and the CFL limit.
    poles : tuple of (complex, complex)
        ``(a_p, r_p)`` pairs in rad/s.  A pole with ``Im(a_p) != 0``
        represents a complex-conjugate *pair* stored once (poles with
        ``Im(a_p) < 0`` are conjugated on input so the stored member has
        ``Im(a_p) > 0``); a real pole must carry a real residue.
    f_band : tuple of (float, float)
        Declared validity band ``(f_min, f_max)`` in Hz.  Passivity
        (``eps'' >= 0``) is enforced on log-spaced samples across this
        band at construction.

    Raises
    ------
    ValueError
        If any pole is unstable (``Re(a_p) > 0``), marginally stable but
        oscillatory (``Re(a_p) = 0`` with ``Im(a_p) != 0``), a real pole
        carries a complex residue, the DC pole (``a_p = 0``) has a
        non-positive residue, or ``eps''(omega) < 0`` anywhere on the
        sampled band.
    """

    eps_inf: float
    poles: tuple[tuple[complex, complex], ...]
    f_band: tuple[float, float]

    def __post_init__(self) -> None:
        if not (self.eps_inf > 0.0):
            raise ValueError(f"DispersionModel: eps_inf must be > 0, got {self.eps_inf!r}")
        f1, f2 = self.f_band
        if not (0.0 < f1 < f2):
            raise ValueError(
                f"DispersionModel: f_band must satisfy 0 < f_min < f_max, got {self.f_band!r}"
            )
        norm = []
        for a, r in self.poles:
            a, r = complex(a), complex(r)
            if a.imag < 0.0:  # store the upper-half-plane member
                a, r = a.conjugate(), r.conjugate()
            if a.real > 0.0:
                raise ValueError(f"DispersionModel: unstable pole a = {a!r} (requires Re(a) <= 0)")
            if a.real == 0.0 and a.imag != 0.0:
                raise ValueError(
                    f"DispersionModel: undamped oscillatory pole a = {a!r} "
                    f"(Re(a) = 0 is only allowed for the real Drude DC pole)"
                )
            if a.imag == 0.0 and r.imag != 0.0:
                raise ValueError(
                    f"DispersionModel: real pole a = {a.real!r} carries a complex residue r = {r!r}"
                )
            if a == 0.0 and not (r.real > 0.0):
                raise ValueError(
                    f"DispersionModel: the DC pole (a = 0) needs a residue "
                    f"r > 0 (an effective DC conductivity), got {r!r}"
                )
            norm.append((a, r))
        object.__setattr__(self, "poles", tuple(norm))

        # Mandatory band-sampled passivity: eps''(omega) >= 0.
        f = np.logspace(math.log10(f1), math.log10(f2), _N_PASSIVITY_SAMPLES)
        eps = self.evaluate(2.0 * np.pi * f)
        eps_im = -eps.imag  # eps = eps' - j*eps''
        tol = _PASSIVITY_RTOL * np.abs(eps)
        bad = eps_im < -tol
        if bad.any():
            i = int(np.argmin(eps_im))
            raise ValueError(
                f"DispersionModel: non-passive — eps''({f[i]:.4g} Hz) = "
                f"{eps_im[i]:.4g} < 0 on the declared band "
                f"[{f1:.4g}, {f2:.4g}] Hz"
            )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, omega) -> np.ndarray:
        """Complex relative permittivity at angular frequency *omega* [rad/s].

        Conjugate pairs contribute both members; the result follows the
        engineering ``e^{+j omega t}`` convention (``eps = eps' - j eps''``
        with ``eps'' >= 0`` for a passive material).
        """
        jw = 1j * np.asarray(omega, dtype=float)
        eps = np.full(jw.shape, self.eps_inf, dtype=complex)
        for a, r in self.poles:
            eps += r / (jw - a)
            if a.imag != 0.0:
                eps += r.conjugate() / (jw - a.conjugate())
        return eps

    # ------------------------------------------------------------------
    # Constructors — industry-standard parameterisations
    # ------------------------------------------------------------------

    @classmethod
    def from_table(
        cls,
        f,
        eps,
        n_poles: int | None = None,
        f_band: tuple[float, float] | None = None,
        tol: float = 1e-3,
        max_poles: int = 30,
    ) -> "DispersionModel":
        """Fit a tabulated permittivity onto the pole-residue form.

        In-repo Gustavsen/Semlyen vector fitting
        (:mod:`magnelio.materials.vector_fit`); the fit enforces pole
        stability, and the :class:`DispersionModel` constructor acts as
        the mandatory passivity acceptance filter — a fit that violates
        it raises with the offending frequency, an active material is
        never silently shipped.

        Parameters
        ----------
        f : array_like
            Table frequencies [Hz], ascending, > 0 (>= 4 points).
        eps : array_like or (array_like, array_like)
            Complex relative permittivity ``eps' - j*eps''`` per table
            point, or a ``(eps_prime, tan_delta)`` pair of real arrays.
        n_poles : int, optional
            Model order (conjugate partners counted individually).
            Default ``None``: automatic — the order grows until the
            maximum relative fit error over the table beats *tol* and
            the result passes the passivity filter, capped at
            *max_poles* with a clear error.  With an explicit order the
            fit is accepted at whatever error it reaches (noisy
            measured tables) — only passivity is enforced.
        f_band : (float, float), optional
            Declared validity band [Hz].  Default: the table span.
        tol : float, optional
            Automatic-order acceptance threshold on
            ``max |fit - table| / max |table|`` (default 1e-3).
        max_poles : int, optional
            Automatic-order cap (default 30).

        Raises
        ------
        ValueError
            If the table itself is non-passive (``eps'' < 0``), the
            automatic order search hits *max_poles* without an
            acceptable passive fit, or the fitted model violates the
            passivity/stability rules.
        """
        from magnelio.materials.vector_fit import vector_fit  # noqa: PLC0415

        f = np.asarray(f, dtype=float)
        if f.ndim != 1 or f.size < 4:
            raise ValueError(
                f"from_table: need a 1D table with >= 4 frequencies, got shape {f.shape}"
            )
        if not ((f > 0.0).all() and (np.diff(f) > 0.0).all()):
            raise ValueError("from_table: frequencies must be ascending and > 0")
        if isinstance(eps, (tuple, list)) and len(eps) == 2:
            eps_prime = np.asarray(eps[0], dtype=float)
            tan_delta = np.asarray(eps[1], dtype=float)
            values = eps_prime * (1.0 - 1j * tan_delta)
        else:
            values = np.asarray(eps, dtype=complex)
        if values.shape != f.shape:
            raise ValueError(
                f"from_table: eps shape {values.shape} does not match f shape {f.shape}"
            )
        if (-values.imag < -_PASSIVITY_RTOL * np.abs(values)).any():
            i = int(np.argmin(-values.imag))
            raise ValueError(
                f"from_table: non-passive table — eps''({f[i]:.4g} Hz) = "
                f"{-values.imag[i]:.4g} < 0 (gain data cannot be "
                f"represented by a passive material)"
            )
        if f_band is None:
            f_band = (float(f[0]), float(f[-1]))

        omega = 2.0 * np.pi * f

        def candidates(n: int):
            """Both Gustavsen start sets, best (passive) fit first."""
            fits = []
            for start in ("real", "complex"):
                try:
                    d, poles, rel_err = vector_fit(
                        omega,
                        values,
                        n,
                        start=start,
                    )
                except np.linalg.LinAlgError:
                    continue
                fits.append((rel_err, d, poles))
            fits.sort(key=lambda t: t[0])
            return fits

        if n_poles is not None:
            last_exc: Exception | None = None
            for rel_err, d, poles in candidates(int(n_poles)):
                try:
                    return cls(eps_inf=d, poles=tuple(poles), f_band=f_band)
                except ValueError as exc:
                    last_exc = exc
            raise (
                last_exc
                if last_exc is not None
                else ValueError(f"from_table: vector fit failed at n_poles={n_poles}")
            )

        last_exc = None
        last_err = np.inf
        for n in range(1, int(max_poles) + 1):
            for rel_err, d, poles in candidates(n):
                last_err = min(last_err, rel_err)
                if rel_err > tol:
                    continue
                try:
                    return cls(eps_inf=d, poles=tuple(poles), f_band=f_band)
                except ValueError as exc:
                    last_exc = exc  # accurate but non-passive candidate
        detail = (
            f"best relative error {last_err:.3g}"
            if np.isfinite(last_err)
            else "vector fit produced no candidate"
        )
        if last_exc is not None:
            detail += f"; last passivity rejection: {last_exc}"
        raise ValueError(
            f"from_table: no acceptable passive fit up to {max_poles} "
            f"poles (tol {tol:.3g}; {detail}).  Noisy measured tables "
            f"need an explicit n_poles (accepted at its own error) or a "
            f"larger tol/max_poles."
        )

    @classmethod
    def debye(
        cls,
        eps_inf: float,
        delta_eps,
        tau,
        f_band: tuple[float, float] | None = None,
    ) -> "DispersionModel":
        """Multi-term Debye relaxation ``sum delta_eps_k / (1 + j*omega*tau_k)``.

        Parameters
        ----------
        eps_inf : float
            High-frequency relative permittivity.
        delta_eps : float or sequence of float
            Relaxation strength per term (``eps_s - eps_inf`` for one term).
        tau : float or sequence of float
            Relaxation time per term [s]; same length as *delta_eps*.
        f_band : (float, float), optional
            Validity band [Hz].  Default: two decades around the
            relaxation frequencies ``1/(2 pi tau)``.
        """
        de = np.atleast_1d(np.asarray(delta_eps, dtype=float))
        t = np.atleast_1d(np.asarray(tau, dtype=float))
        if de.shape != t.shape:
            raise ValueError(
                f"debye: delta_eps and tau need matching lengths, got {de.shape} vs {t.shape}"
            )
        if not (t > 0.0).all():
            raise ValueError(f"debye: tau must be > 0, got {tau!r}")
        poles = tuple((complex(-1.0 / tk), complex(dk / tk)) for dk, tk in zip(de, t))
        if f_band is None:
            f_relax = 1.0 / (2.0 * np.pi * t)
            f_band = (float(f_relax.min()) / 100.0, float(f_relax.max()) * 100.0)
        return cls(eps_inf=eps_inf, poles=poles, f_band=f_band)

    @classmethod
    def lorentz(
        cls,
        eps_inf: float,
        delta_eps: float,
        omega0: float,
        delta: float,
        f_band: tuple[float, float] | None = None,
    ) -> "DispersionModel":
        """Lorentz resonance ``delta_eps * omega0^2 / (omega0^2 + 2j*omega*delta - omega^2)``.

        Parameters
        ----------
        eps_inf : float
            High-frequency relative permittivity.
        delta_eps : float
            Resonance strength (``eps_s - eps_inf``).
        omega0 : float
            Resonance angular frequency [rad/s].
        delta : float
            Damping rate [rad/s] (must be > 0 and != omega0; the
            critically damped double pole is not representable in the
            simple-pole form — perturb delta instead).
        f_band : (float, float), optional
            Validity band [Hz].  Default: two decades around
            ``omega0 / (2 pi)``.
        """
        if not (omega0 > 0.0 and delta > 0.0):
            raise ValueError(
                f"lorentz: omega0 and delta must be > 0, got omega0={omega0!r}, delta={delta!r}"
            )
        s = delta_eps * omega0**2
        if delta < omega0:  # underdamped: one conjugate pair
            wd = math.sqrt(omega0**2 - delta**2)
            poles = ((complex(-delta, wd), complex(0.0, -s / (2.0 * wd))),)
        elif delta > omega0:  # overdamped: two real poles
            wd = math.sqrt(delta**2 - omega0**2)
            poles = (
                (complex(-delta + wd), complex(+s / (2.0 * wd))),
                (complex(-delta - wd), complex(-s / (2.0 * wd))),
            )
        else:
            raise ValueError(
                "lorentz: delta == omega0 (critical damping) is a double "
                "pole the simple-pole form cannot represent; perturb delta"
            )
        if f_band is None:
            f0 = omega0 / (2.0 * np.pi)
            f_band = (f0 / 100.0, f0 * 100.0)
        return cls(eps_inf=eps_inf, poles=poles, f_band=f_band)

    @classmethod
    def drude(
        cls,
        eps_inf: float,
        omega_p: float,
        gamma: float,
        f_band: tuple[float, float] | None = None,
    ) -> "DispersionModel":
        """Drude free-carrier model ``eps_inf - omega_p^2 / (omega^2 - j*omega*gamma)``.

        Partial fractions give one real relaxation pole at ``-gamma`` and
        the real DC pole at 0 whose trapezoidal update is exactly the
        semi-implicit conductor with ``sigma = eps0 * omega_p^2 / gamma``.

        Parameters
        ----------
        eps_inf : float
            High-frequency relative permittivity.
        omega_p : float
            Plasma angular frequency [rad/s].
        gamma : float
            Collision rate [rad/s] (must be > 0).
        f_band : (float, float), optional
            Validity band [Hz].  Default: two decades around
            ``omega_p / (2 pi)``.
        """
        if not (omega_p > 0.0 and gamma > 0.0):
            raise ValueError(
                f"drude: omega_p and gamma must be > 0, got omega_p={omega_p!r}, gamma={gamma!r}"
            )
        s = omega_p**2 / gamma
        poles = (
            (complex(0.0), complex(s)),
            (complex(-gamma), complex(-s)),
        )
        if f_band is None:
            fp = omega_p / (2.0 * np.pi)
            f_band = (fp / 100.0, fp * 100.0)
        return cls(eps_inf=eps_inf, poles=poles, f_band=f_band)

    @classmethod
    def djordjevic_sarkar(
        cls,
        eps_r: float,
        tan_delta: float,
        f_ref: float,
        f1: float = 1e3,
        f2: float = 1e12,
    ) -> "DispersionModel":
        """Causal wideband constant-loss-tangent substrate model.

        The Djordjevic–Sarkar continuum (log-uniform Debye distribution
        between ``f1`` and ``f2``) is discretised at two poles per decade
        — enough that the comb ripple is negligible against the model's
        *inherent* tan-delta slope (eps'' is near-constant while eps'
        falls logarithmically, so tan delta drifts ~5 % per 1.5 decades
        from f_ref at eps_r=4.3/tan_delta=0.02; that drift IS the causal
        Kramers–Kronig behaviour, not a discretisation artefact).
        ``eps_inf`` and the total relaxation strength are solved so the
        model reproduces exactly ``eps' = eps_r`` and
        ``eps''/eps' = tan_delta`` at ``f_ref``.  A truly constant
        tan delta over all frequencies violates Kramers–Kronig — this is
        the standard causal approximation every time-domain solver uses.

        Parameters
        ----------
        eps_r : float
            Real relative permittivity at *f_ref*.
        tan_delta : float
            Loss tangent at *f_ref* (>= 0; 0 returns a pole-free model).
        f_ref : float
            Reference frequency [Hz]; must lie inside ``(f1, f2)``.
        f1, f2 : float, optional
            Debye-distribution corner frequencies [Hz] — also the
            declared validity band.  Defaults 1 kHz / 1 THz.
        """
        if not (0.0 < f1 < f_ref < f2):
            raise ValueError(
                f"djordjevic_sarkar: need 0 < f1 < f_ref < f2, "
                f"got f1={f1!r}, f_ref={f_ref!r}, f2={f2!r}"
            )
        if tan_delta < 0.0:
            raise ValueError(f"djordjevic_sarkar: tan_delta must be >= 0, got {tan_delta!r}")
        if tan_delta == 0.0:
            return cls(eps_inf=eps_r, poles=(), f_band=(f1, f2))

        n_dec = math.log10(f2 / f1)
        n_poles = max(2, int(round(2.0 * n_dec)) + 1)
        w_k = (
            2.0
            * np.pi
            * np.logspace(
                math.log10(f1),
                math.log10(f2),
                n_poles,
            )
        )
        # Unit-strength comb chi_d(omega) = (1/N) sum 1/(1 + j*omega/w_k):
        # solve eps_inf, delta_eps from the two conditions at f_ref.
        w_ref = 2.0 * np.pi * f_ref
        chi_d = np.mean(1.0 / (1.0 + 1j * w_ref / w_k))
        delta_eps = tan_delta * eps_r / (-chi_d.imag)
        eps_inf = eps_r - delta_eps * chi_d.real
        if not (eps_inf > 0.0):
            raise ValueError(
                f"djordjevic_sarkar: tan_delta={tan_delta!r} over "
                f"{n_dec:.1f} decades needs delta_eps={delta_eps:.3g} > "
                f"permitted by eps_r={eps_r!r} (eps_inf would be "
                f"{eps_inf:.3g} <= 0); narrow [f1, f2] or reduce tan_delta"
            )
        poles = tuple((complex(-wk), complex(delta_eps / n_poles * wk)) for wk in w_k)
        return cls(eps_inf=eps_inf, poles=poles, f_band=(f1, f2))
