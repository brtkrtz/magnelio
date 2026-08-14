"""Exact discrete transparent boundary condition (DTBC) for modal ports.

Method core (reflection-free plan WP-R1, gate passed session 84; the
plan REFLECTION_FREE_PLAN.md is retired to git history — see DD-054):
per port mode on a uniform feed line the longitudinal dynamics of the
modal amplitude ``u_k^n`` is the exact 1D leapfrog Klein-Gordon chain

    u^{n+1}_k = 2 u^n_k - u^{n-1}_k
                + r^2 (u^n_{k+1} - 2 u^n_k + u^n_{k-1}) - q^2 u^n_k

with ``r`` the modal Courant number and ``q`` the discrete cut-off
times ``dt`` (TEM: ``q = 0``).  Z-transform in time turns the update
into the spatial two-term recurrence with characteristic root

    lambda(z) = A - sqrt(A^2 - 1),
    A(z) = 1 + (z - 2 + 1/z + q^2) / (2 r^2),

on the branch ``|lambda| <= 1`` for ``|z| > 1`` (outgoing solution of
the semi-infinite uniform continuation).  The exact transparent
boundary at plane ``K`` is the ghost relation
``u_hat_{K+1} = lambda(z) * u_hat_K``, in the time domain the causal
convolution ``u^n_{K+1} = sum_{m>=1} l_m u^{n-m}_K`` with kernel
``l_m`` = Laurent coefficients of ``lambda`` around infinity
(``l_0 = 0``, so the ghost depends on *past* boundary samples only and
slots into the explicit leapfrog).

Exactness within a finite run
-----------------------------

At step ``n`` the ghost convolution reaches at most ``n`` samples into
the past, so a kernel of length ``> n`` renders the boundary **exact**
— no truncation, no passivity question (the exact half-lattice symbol
is passive).  :class:`DTBCTermination` therefore auto-extends its
kernel (powers of two, contour-integration regeneration is O(n log n))
whenever a run outlives the current length; the truncation-error and
passivity analysis of the R1 gate only becomes relevant for a future
compressed (sum-of-exponentials) production form.

Incident-wave injection through the boundary
--------------------------------------------

An incident wave entering the domain through the port is prescribed as
the amplitude ``s^n`` *at the ghost plane* ``K+1``.  The exact discrete
incoming wave at the boundary plane is then the same kernel applied to
the source history, ``u_inc,K^n = sum_{m>=1} l_m s^{n-m}``, and the
transparent condition acts on the scattered part only:

    u^n_{K+1} = s^n + sum_{m>=1} l_m (u_K - u_inc,K)^{n-m}.

Because the interior of the domain reduces to the same chain, this
launches the *exact discrete* incoming wave — no fractional-delay
interpolation and no velocity approximation, for propagating and
evanescent spectral content alike.

De-stagger symbol
-----------------

For the discrete travelling wave ``u_k^n = z^n lambda^k`` the co-located
H sample half a cell and half a step away obeys, on the leapfrog pair,

    h_{k+1/2}^{n+1/2} / e_k^n
        = z^{1/2} lambda^{1/2} * dt sin(beta_hat dz/2) /
          (M_mu sin(w dt/2)),

whose magnitude is frequency-independent (``= dt/(M_mu r)`` by the
discrete dispersion ``sin(w dt/2) = r sin(beta_hat dz/2)`` for TEM) —
that is why the static travelling-wave V/I calibration is exact at all
frequencies.  The frequency-dependent part is purely the propagation
factor ``lambda^{1/2}`` (plus the temporal ``z^{1/2}`` already
compensated in postprocessing).  :func:`destagger_theta` exposes
``theta = -log(lambda)/2`` so the two-plane a/b decomposition can
de-stagger the I sampling plane with the *discrete* propagation factor
instead of the continuum ``gamma * dz/2`` (whose dispersion gap
``~ (beta dz/2)^3 (1 - r^2)/6`` caps measured floors near -70 dB on
lambda/20 meshes).
"""

from __future__ import annotations

import math

import numpy as np

# Contour offset for evaluations "on" the unit circle: far enough off
# the circle to keep the outgoing-root selection unambiguous against
# roundoff, close enough that the induced bias stays below -130 dB
# even at a 1.01*f_c evaluation edge (R1 spike measurement).
_EDGE_OFFSET = 1e-8


def lambda_symbol(
    z: np.ndarray | complex,
    r: float,
    q: float = 0.0,
) -> np.ndarray:
    """Outgoing characteristic root ``lambda(z)`` with ``|lambda| <= 1``.

    Parameters
    ----------
    z : np.ndarray or complex
        Evaluation points, ``|z| > 1`` (use a small offset such as
        ``(1 + 1e-8) * exp(1j*w*dt)`` for on-circle evaluations).
    r : float
        Modal Courant number ``v * dt / dz`` of the uniform feed chain.
    q : float, default 0.0
        Discrete cut-off ``omega_c_hat * dt`` (TEM: 0).

    Returns
    -------
    np.ndarray
        ``lambda(z)``, complex, same shape as ``z``.
    """
    z = np.asarray(z, dtype=complex)
    A = 1.0 + (z - 2.0 + 1.0 / z + q * q) / (2.0 * r * r)
    root = np.sqrt(A * A - 1.0)
    lam_minus = A - root
    lam_plus = A + root
    return np.where(
        np.abs(lam_minus) <= np.abs(lam_plus),
        lam_minus,
        lam_plus,
    )


def dtbc_kernel(r: float, q: float, n_kernel: int) -> np.ndarray:
    """Convolution kernel ``l_m``, ``m = 0 .. n_kernel-1``.

    Laurent coefficients of ``lambda(z)`` around infinity via contour
    integration: on ``z = rho e^{i theta}`` the ``l_m`` are ``rho^m``
    times the Fourier coefficients of ``lambda`` along the circle.
    ``rho > 1`` keeps the branch selection unambiguous; it is chosen so
    that ``rho^{n_kernel}`` stays O(e^4) (bounded roundoff
    amplification).  ``l_0 = 0`` analytically (``lambda ~ r^2/z`` at
    infinity).

    Parameters
    ----------
    r, q : float
        Chain parameters, see :func:`lambda_symbol`.
    n_kernel : int
        Number of kernel samples to return.

    Returns
    -------
    np.ndarray
        Real kernel of shape ``(n_kernel,)``.
    """
    if n_kernel < 2:
        raise ValueError("n_kernel must be >= 2")
    n_fft = 8 * n_kernel
    rho = math.exp(4.0 / n_kernel)
    theta = 2.0 * np.pi * np.arange(n_fft) / n_fft
    z = rho * np.exp(1j * theta)
    coeff = np.fft.ifft(lambda_symbol(z, r, q))
    m = np.arange(n_kernel)
    return np.real(coeff[:n_kernel]) * rho**m


def reflection_bound(
    r: float,
    q: float,
    kernel: np.ndarray,
    w_dt: np.ndarray,
) -> np.ndarray:
    """Exact modal reflection ``|Gamma(w)|`` of the truncated kernel.

    For a boundary implementing the approximated symbol
    ``lambda~(z) = sum_m l_m z^{-m}`` instead of ``lambda(z)``,

        Gamma(w) = (lambda~ - lambda) / (1/lambda - lambda~)

    at ``z = e^{i w dt}``.  This is the a-priori gate of the R1 method
    note: the achievable floor of a *steady-state* (run longer than the
    kernel) boundary is computable before any TD run.  A run shorter
    than the kernel sees the exact boundary and this bound does not
    apply (it is then -inf in exact arithmetic).

    Parameters
    ----------
    r, q : float
        Chain parameters.
    kernel : np.ndarray
        Kernel samples ``l_0 .. l_{n-1}`` (e.g. from
        :func:`dtbc_kernel`).
    w_dt : np.ndarray
        Evaluation frequencies as ``omega * dt`` [rad].

    Returns
    -------
    np.ndarray
        ``|Gamma|`` at each ``w_dt``.
    """
    w_dt = np.asarray(w_dt, dtype=float)
    z = np.exp(1j * w_dt) * (1.0 + _EDGE_OFFSET)
    lam = lambda_symbol(z, r, q)
    m = np.arange(kernel.size)
    lam_t = np.array([np.sum(kernel * zz ** (-m)) for zz in z])
    return np.abs(lam_t - lam) / np.abs(1.0 / lam - lam_t)


def destagger_theta(
    w_dt: np.ndarray,
    r: float,
    q: float = 0.0,
) -> np.ndarray:
    """Discrete half-cell de-stagger exponent ``theta = -log(lambda)/2``.

    Defined so that ``e^{-theta} = lambda^{1/2}`` is the exact
    propagation factor of the discrete travelling wave from the port
    plane to the I sampling plane half a cell inside.  In the passband
    ``theta = +i beta_hat dz / 2`` (purely imaginary); below cut-off
    ``lambda`` is real in (0, 1) and ``theta`` is real positive —
    the same convention as the continuum ``gamma * dz / 2`` it
    replaces in :func:`compute_s_parameters`.

    Parameters
    ----------
    w_dt : np.ndarray
        Evaluation frequencies as ``omega * dt`` [rad].
    r, q : float
        Chain parameters, see :func:`lambda_symbol`.

    Returns
    -------
    np.ndarray
        Complex ``theta`` per frequency.
    """
    w_dt = np.asarray(w_dt, dtype=float)
    z = np.exp(1j * w_dt) * (1.0 + _EDGE_OFFSET)
    lam = lambda_symbol(z, r, q)
    return -0.5 * np.log(lam)


def dtbc_wave_impedance(
    w_dt: np.ndarray,
    q: float,
    z0: float,
    mode_kind: str,
) -> np.ndarray:
    """Exact wave impedance of the discrete travelling wave.

    For a Klein-Gordon port mode the V/I ratio of the *discrete*
    outgoing wave — measured through the production projections at
    the port plane — is (WP-R3 pre-check,
    ``validation/kg_dtbc_precheck_spike.py``)

        Z_TE(w) = z0 * s / rad,     Z_TM(w) = z0 * rad / s,
        s = sin(w dt / 2),          rad = sqrt(s^2 - (q/2)^2),

    the continuum relations under ``omega -> (2/dt) sin(omega dt/2)``
    and ``beta -> (2/dz) sin(beta_hat dz/2)``.  The continuum
    ``z_wave(omega)`` misses this by O((omega dt)^2, (beta dz)^2) — a
    -40 to -60 dB measured-|S11| cap on lambda/20 meshes.  Below the
    discrete cut-off (``s < q/2``) the branch ``rad = -j sqrt(...)``
    continues the outgoing (decaying) root: Z_TE inductive, Z_TM
    capacitive, matching the continuum reactances.

    Parameters
    ----------
    w_dt : np.ndarray
        Evaluation frequencies as ``omega * dt`` [rad].
    q : float
        Discrete cut-off times dt of the certified chain.
    z0 : float
        Static impedance constant of the channel
        (``PortOperatorModal``: ``z0 = r * nV * c_pair / (dt * nI)``
        from the stored discrete profiles).
    mode_kind : str
        ``"TE"`` or ``"TM"``.

    Returns
    -------
    np.ndarray
        Complex ``Z(w)`` per frequency.
    """
    s = np.sin(np.asarray(w_dt, dtype=float) / 2.0)
    d = s * s - (q / 2.0) ** 2
    rad = np.where(
        d >= 0.0,
        np.sqrt(np.maximum(d, 0.0)) + 0.0j,
        -1j * np.sqrt(np.maximum(-d, 0.0)),
    )
    if mode_kind == "TE":
        return z0 * s / rad
    if mode_kind == "TM":
        return z0 * rad / s
    raise ValueError(f"mode_kind must be 'TE' or 'TM', got {mode_kind!r}")


class DTBCTermination:
    """Exact DTBC boundary state for one port mode.

    Owns the modal chain state at the boundary plane (``u_K``), the
    scattered-field and source histories, and the ghost-relation
    kernel.  One :meth:`advance` call performs one leapfrog step of the
    boundary plane:

        u_K^{n+1} = 2 u_K^n - u_K^{n-1}
                    + r^2 (ghost^n - 2 u_K^n + u_int^n) - q^2 u_K^n

    with ``ghost^n = s^n + sum_{m>=1} l_m (u_K - u_inc,K)^{n-m}`` and
    ``u_inc,K^n = sum_{m>=1} l_m s^{n-m}`` (see module docstring).

    The kernel is auto-extended (powers of two) so that its length
    always exceeds the number of completed steps — within a run the
    boundary is the *exact* discrete transparent condition, not a
    truncation.

    Parameters
    ----------
    r : float
        Modal Courant number of the uniform feed chain.  Must satisfy
        ``0 < r <= 1`` (the interior update is unstable otherwise).
    q : float, default 0.0
        Discrete cut-off times ``dt`` (TEM: 0).
    n_kernel_init : int, default 4096
        Initial kernel length; grows on demand.
    """

    def __init__(
        self,
        r: float,
        q: float = 0.0,
        *,
        n_kernel_init: int = 4096,
    ) -> None:
        if not (0.0 < r <= 1.0):
            raise ValueError(f"r must be in (0, 1], got {r}")
        if q < 0.0:
            raise ValueError(f"q must be >= 0, got {q}")
        self.r = float(r)
        self.q = float(q)
        self._r2 = self.r * self.r
        self._q2 = self.q * self.q

        self._n_kernel = int(n_kernel_init)
        self._rebuild_kernel()

        self._u = 0.0  # u_K^n
        self._u_prev = 0.0  # u_K^{n-1}
        self._n = 0  # completed steps
        self._w_hist = np.zeros(self._n_kernel)
        self._s_hist = np.zeros(self._n_kernel)

    def _rebuild_kernel(self) -> None:
        self._kernel = dtbc_kernel(self.r, self.q, self._n_kernel)
        # Reversed tail l_{n_k-1} .. l_1 for contiguous dot products:
        # kflip[-n:] . hist[:n] = sum_{m=1..n} l_m hist[n-m].
        self._kflip = self._kernel[1:][::-1].copy()

    def _ensure_capacity(self, n: int) -> None:
        """Guarantee kernel length > n and history capacity >= n + 1."""
        if n + 1 > self._n_kernel:
            while self._n_kernel < n + 1:
                self._n_kernel *= 2
            self._rebuild_kernel()
        if n + 1 > self._w_hist.size:
            grow = max(2 * self._w_hist.size, n + 1)
            for name in ("_w_hist", "_s_hist"):
                old = getattr(self, name)
                new = np.zeros(grow)
                new[: old.size] = old
                setattr(self, name, new)

    def state_dict(self) -> dict:
        """Checkpoint the exact boundary state (DD-070, WP-S6).

        The transparent condition is a convolution over the *entire*
        boundary history, so a bit-exact resume must carry ``w``/``s``
        up to the completed step count ``n`` plus the two boundary
        amplitudes.  The kernel is a pure function of ``(r, q)`` and is
        re-derived on load.
        """
        n = self._n
        return {
            "u": float(self._u),
            "u_prev": float(self._u_prev),
            "n": int(n),
            "w_hist": self._w_hist[:n].copy(),
            "s_hist": self._s_hist[:n].copy(),
        }

    def load_state_dict(self, sd: dict) -> None:
        """Restore state written by :meth:`state_dict` (bit-exact resume)."""
        n = int(sd["n"])
        self._ensure_capacity(n)
        self._u = float(sd["u"])
        self._u_prev = float(sd["u_prev"])
        self._n = n
        self._w_hist[:] = 0.0
        self._s_hist[:] = 0.0
        self._w_hist[:n] = np.asarray(sd["w_hist"], dtype=float)
        self._s_hist[:n] = np.asarray(sd["s_hist"], dtype=float)

    @property
    def u_boundary(self) -> float:
        """Current boundary amplitude ``u_K^n`` (last value returned)."""
        return self._u

    def initialize(self, u0: float) -> None:
        """Set a non-zero initial boundary amplitude.

        Analogous to ``PortOperatorModal.initialize_state``: for a
        non-trivial initial condition, ``u_K^0 = u_K^{-1} = u0``
        (zero-velocity start).  Exactness of the transparent condition
        assumes the *exterior* is quiescent at ``t = 0``; an interior
        IC that has not yet reached the port satisfies this by
        construction.
        """
        if self._n != 0:
            raise RuntimeError(
                "initialize() must be called before the first advance()",
            )
        self._u = float(u0)
        self._u_prev = float(u0)

    def advance(self, u_interior: float, src: float = 0.0) -> float:
        """One boundary leapfrog step; returns ``u_K^{n+1}``.

        Parameters
        ----------
        u_interior : float
            Modal amplitude ``u_{K-1}^n`` at the first interior plane,
            *at the same time level as the current boundary value*
            (i.e. projected from the field state one solver step ago).
        src : float, default 0.0
            Incident amplitude ``s^n`` prescribed at the ghost plane.
            Zero for a passive (absorbing-only) port.

        Returns
        -------
        float
            The new boundary amplitude ``u_K^{n+1}`` to write onto the
            port plane.
        """
        n = self._n
        self._ensure_capacity(n)

        if n == 0:
            u_inc = 0.0
            conv_w = 0.0
        else:
            kf_tail = self._kflip[self._kflip.size - n :]
            # einsum, not np.dot: BLAS ddot multithreads past ~16k
            # elements and its per-call thread-team overhead (ms-class,
            # all cores spinning) dwarfs the actual reduction; einsum
            # stays single-threaded SIMD (~us-class at these lengths).
            u_inc = float(np.einsum("i,i->", kf_tail, self._s_hist[:n]))
            conv_w = float(np.einsum("i,i->", kf_tail, self._w_hist[:n]))

        ghost = src + conv_w
        self._w_hist[n] = self._u - u_inc
        self._s_hist[n] = src

        u_new = (
            (2.0 - 2.0 * self._r2 - self._q2) * self._u
            - self._u_prev
            + self._r2 * (ghost + u_interior)
        )
        self._u_prev = self._u
        self._u = u_new
        self._n = n + 1
        return u_new
