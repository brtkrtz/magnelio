"""Post-hoc reference-plane shift (de-embedding) of an S-matrix.

:func:`deembed_s_params` moves the reference plane of selected ports
along their feed lines and returns the S-matrix referenced at the
shifted planes.  The shift multiplies every S-parameter touching a
shifted channel by the inverse of the line propagation factor over the
shift distance — reflections collect the factor twice, transmissions
once per shifted end.

The propagation factor comes from the *discrete* dispersion of the
port's uniform feed chain wherever the run certified its line
parameters: the same characteristic root ``lambda(z)`` the transparent
port boundary is built from, so de-embedding removes exactly the
propagation the grid applied — including the numerical-dispersion part
that the continuum ``exp(-γd)`` would leave behind on coarse meshes.
Channels without certified line parameters fall back to the mode's
continuum ``γ(ω)``.

Users reach this through ``result.deembed(...)`` on any scattering
result; this module is the shared implementation behind it.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from magnelio.post.sparameter_result import SParameterResult


def _chain_lambda_log(w_dt: np.ndarray, r: float, q: float) -> np.ndarray:
    """``log(lambda)`` of the outgoing chain root, exactly on the circle.

    Same characteristic root as
    :func:`~magnelio.ports._modal.dtbc.lambda_symbol`, but evaluated
    at real frequencies without that function's off-circle offset: on
    the unit circle ``A(ω)`` is real, so the branch is decided by
    ``A`` alone — passband ``|A| ≤ 1`` gives the pure phase
    ``-j·arccos(A) = -j·beta_hat·dz`` (``|lambda| = 1`` exactly, so
    de-embedding never touches propagating magnitudes), ``A > 1``
    (below cut-off) the real decay ``log(A - sqrt(A²-1))``, and
    ``A < -1`` (beyond the chain passband) continues the ``A = -1``
    limit ``-jπ``.  The offset's ``O(d/dz · 1e-8)`` magnitude bias
    would otherwise grow with the shift distance.
    """
    A = 1.0 + (2.0 * np.cos(w_dt) - 2.0 + q * q) / (2.0 * r * r)
    out = np.empty(A.shape, dtype=complex)
    prop = np.abs(A) <= 1.0
    out[prop] = -1j * np.arccos(A[prop])
    low = A > 1.0
    out[low] = np.log(A[low] - np.sqrt(A[low] ** 2 - 1.0))
    high = A < -1.0
    out[high] = np.log(-A[high] - np.sqrt(A[high] ** 2 - 1.0)) - 1j * np.pi
    return out


def _channel_shift_factor(
    omega: np.ndarray,
    dt: float,
    distance: float,
    line_params: tuple | None,
    dz: float | None,
    mode,
) -> np.ndarray | None:
    """Inverse propagation factor of one channel over ``distance``.

    Discrete path: ``lambda^(-d/dz)`` — in the passband a pure phase
    advance ``exp(+j·beta_hat·d)``, below cut-off a real growth
    ``exp(+alpha_hat·d)``.  Continuum fallback: ``exp(+γ(ω)·d)``, the
    same convention.  Returns ``None`` when the channel carries no
    line dispersion at all (lumped ports).
    """
    if line_params is not None and dz:
        r, q = float(line_params[0]), float(line_params[1])
        log_lam = _chain_lambda_log(omega * dt, r, q)
        return np.exp(-(distance / float(dz)) * log_lam)
    gamma = getattr(mode, "gamma", None)
    if gamma is not None:
        g = np.array([gamma(float(w)) for w in omega], dtype=complex)
        return np.exp(g * distance)
    return None


def deembed_s_params(
    s_params: SParameterResult,
    distances: Mapping[str, float],
    *,
    dt: float,
    port_line_params: dict | None = None,
    port_normal_dx: dict | None = None,
    port_modes: dict | None = None,
) -> SParameterResult:
    """Return ``s_params`` referenced at shifted port planes.

    Parameters
    ----------
    s_params : SParameterResult
        The S-matrix to de-embed; not modified.
    distances : mapping of str to float
        Per-port shift distance [m].  Positive moves the reference
        plane from the port plane *into* the domain; negative moves it
        outward (adds line length).  Ports not named keep their plane.
    dt : float
        Solver time step [s] the run used (the discrete factors
        evaluate at ``ω·dt``).
    port_line_params : dict, optional
        Per-channel certified discrete line parameters
        ``{(port, mode): (r, q, z0)}`` as recorded on the result.
    port_normal_dx : dict, optional
        Per-port feed cell size along the port normal [m].
    port_modes : dict, optional
        Per-port ordered mode list, for the continuum ``γ(ω)``
        fallback on channels without certified line parameters.

    Returns
    -------
    SParameterResult
        A new result on the same frequency axis and channels.

    Raises
    ------
    ValueError
        If a named port has no channels in the result, or a shifted
        channel carries no line dispersion (lumped ports).
    """
    known = set(s_params.port_names)
    unknown = [p for p in distances if p not in known]
    if unknown:
        raise ValueError(
            f"cannot de-embed unknown port(s) {sorted(unknown)}; "
            f"this result has ports {sorted(known)}."
        )
    for port, d in distances.items():
        if not np.isfinite(d):
            raise ValueError(f"de-embed distance for port '{port}' must be finite; got {d!r}.")

    omega = 2.0 * np.pi * np.asarray(s_params.f_axis, dtype=float)
    ones = np.ones_like(omega, dtype=complex)

    factors: dict[tuple[str, int], np.ndarray] = {}
    for key in dict.fromkeys(s_params.channels + s_params.excitations):
        port, mode_idx = key
        if port not in distances:
            factors[key] = ones
            continue
        modes = (port_modes or {}).get(port) or []
        factor = _channel_shift_factor(
            omega,
            dt,
            float(distances[port]),
            (port_line_params or {}).get(key),
            (port_normal_dx or {}).get(port),
            modes[mode_idx] if mode_idx < len(modes) else None,
        )
        if factor is None:
            raise ValueError(
                f"channel {key} carries no feed-line dispersion; "
                "de-embedding shifts the reference plane along a "
                "waveguide port's feed line and does not apply to "
                "lumped ports."
            )
        factors[key] = factor

    row = np.stack([factors[key] for key in s_params.channels], axis=1)
    col = np.stack([factors[key] for key in s_params.excitations], axis=1)
    matrix = s_params.matrix * row[:, :, None] * col[:, None, :]
    return SParameterResult(
        f_axis=np.asarray(s_params.f_axis, dtype=float),
        channels=s_params.channels,
        excitations=s_params.excitations,
        matrix=matrix,
    )
