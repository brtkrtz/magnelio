"""
Backend abstraction layer.

All numerical modules use ``xp = get_xp()`` instead of importing numpy directly.
This allows transparent switching between NumPy (CPU) and CuPy (GPU, v1.1).

See design-decisions.md DD-006.
"""

import os

import numpy as np

_BACKEND: str = "numpy"


def get_xp():
    """Return the active array module (numpy or cupy).

    Usage in numerical code::

        from magnelio._backend.array_api import get_xp

        def compute(data):
            xp = get_xp()
            return xp.zeros_like(data)
    """
    if _BACKEND == "numpy":
        return np
    elif _BACKEND == "cupy":
        try:
            import cupy  # noqa: PLC0415

            return cupy
        except ImportError as exc:
            raise ImportError(
                "CuPy is not installed. Install it with: pip install cupy-cuda12x"
            ) from exc
    raise ValueError(f"Unknown backend: {_BACKEND!r}. Valid options: 'numpy', 'cupy'")


_AUTO_XP = None  # cached result of the "auto" device probe
_AUTO_FALLBACK_REASON: str | None = None


def auto_fallback_reason() -> str | None:
    """Why the ``"auto"`` device probe fell back to NumPy.

    ``None`` when the probe found a usable GPU or has not run yet
    (e.g. ``MAGNELIO_BACKEND`` short-circuited it).  The CPU fallback
    is the documented behaviour of ``"auto"``, so it is not a warning;
    solvers surface this string in their verbose banner instead.
    """
    return _AUTO_FALLBACK_REASON


def backend_summary(xp) -> str:
    """Human-readable one-liner for a resolved backend module."""
    if xp is np:
        return "NumPy (CPU)"
    try:
        dev = xp.cuda.runtime.getDevice()
        name = xp.cuda.runtime.getDeviceProperties(dev)["name"].decode()
        return f"CuPy (GPU: {name})"
    except Exception:
        return "CuPy (GPU)"


def resolve_backend(backend: str = "auto"):
    """Resolve a per-solver backend request to an array module.

    Parameters
    ----------
    backend : {"auto", "numpy", "cupy"}
        ``"numpy"`` returns NumPy.  ``"cupy"`` returns CuPy or raises
        with a clear message when CuPy or a CUDA device is unavailable.
        ``"auto"`` (default) first honours the ``MAGNELIO_BACKEND``
        environment variable ("numpy"/"cupy" — the deterministic
        override for test suites and batch farms), then probes once
        per process for a usable CuPy + CUDA device and falls back to
        NumPy silently — the fallback is the documented behaviour, not
        an anomaly; :func:`auto_fallback_reason` exposes the probe
        failure for verbose solver banners.

    Notes
    -----
    This is the public per-solver mechanism (``FITTimeDomainSolver`` /
    ``AnalysisScatteringTD`` ``backend=``); the module-global
    ``set_backend``/``get_xp`` pair remains for internal consumers that
    are not wired to a solver instance.
    """
    if backend == "numpy":
        return np
    if backend == "cupy":
        try:
            import cupy  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "backend='cupy' requested but CuPy is not installed. "
                "Install it with: pip install cupy-cuda12x"
            ) from exc
        try:
            if cupy.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA device reported")
            cupy.zeros(1)  # forces context creation — the real probe
        except Exception as exc:
            raise RuntimeError(
                "backend='cupy' requested but no usable CUDA device was "
                f"found ({exc}). Use backend='auto' for a CPU fallback."
            ) from exc
        return cupy
    if backend == "auto":
        env = os.environ.get("MAGNELIO_BACKEND", "").strip().lower()
        if env:
            if env == "auto":
                pass  # explicit "auto" = probe below
            elif env in ("numpy", "cupy"):
                return resolve_backend(env)
            else:
                raise ValueError(
                    f"MAGNELIO_BACKEND={env!r} is invalid; use 'auto', 'numpy' or 'cupy'"
                )
        global _AUTO_XP, _AUTO_FALLBACK_REASON
        if _AUTO_XP is None:
            try:
                _AUTO_XP = resolve_backend("cupy")
            except Exception as exc:
                # The chained cause carries the bare probe failure
                # ("No module named 'cupy'", "no CUDA device reported");
                # the wrapper text addresses explicit backend="cupy"
                # callers and would be misleading here.
                _AUTO_FALLBACK_REASON = str(exc.__cause__ or exc)
                _AUTO_XP = np
        return _AUTO_XP
    raise ValueError(f"Unknown backend: {backend!r}. Valid options: 'auto', 'numpy', 'cupy'")


def resolve_precision(precision: str | None = None) -> tuple[np.dtype, np.dtype]:
    """Resolve a per-solver precision request to (real, complex) dtypes.

    Parameters
    ----------
    precision : {"single", "double", None}
        ``"single"`` returns ``(float32, complex64)``; ``"double"`` returns
        ``(float64, complex128)``.  ``None`` (the unspecified default) first
        honours the ``MAGNELIO_PRECISION`` environment variable
        ("single"/"double") — the deterministic override for test suites and
        batch farms — and falls back to ``"single"`` (the production
        default).  An explicit ``"single"``/``"double"`` argument always
        wins and does *not* consult the environment, exactly as an explicit
        ``backend="cupy"`` bypasses ``MAGNELIO_BACKEND`` in
        :func:`resolve_backend`.  Any other value raises.

    Returns
    -------
    (real_dtype, complex_dtype) : tuple of numpy dtypes
        The dtype of the time-loop **field and coefficient** arrays.  The
        DFT/Freq accumulators, the modal-port solve and the geometry pipeline
        are pinned to double in code and are *not* driven by this value — see
        design-decisions.md DD-094.

    Notes
    -----
    This is the public per-solver mechanism (``FITTimeDomainSolver`` /
    ``AnalysisScatteringTD`` ``precision=``).  The precision axis is
    orthogonal to the backend axis (:func:`resolve_backend`): any precision
    runs on any backend.
    """
    if precision is None:
        env = os.environ.get("MAGNELIO_PRECISION", "").strip().lower()
        precision = env or "single"
    if precision == "single":
        return np.dtype(np.float32), np.dtype(np.complex64)
    if precision == "double":
        return np.dtype(np.float64), np.dtype(np.complex128)
    raise ValueError(f"Unknown precision: {precision!r}. Valid options: 'single', 'double'")


def copy_into(dst, src) -> None:
    """``dst[:] = src`` across backends.

    CuPy rejects slice-assignment from a host ``numpy.ndarray``
    ("non-scalar numpy.ndarray cannot be used for fill"), so checkpoint
    state loaded from HDF5 (always host arrays, DD-070) must be staged
    through ``cupy.asarray`` when the destination lives on the device.
    NumPy destinations take the plain assignment.
    """
    if hasattr(dst, "get") and isinstance(src, np.ndarray):
        import cupy  # noqa: PLC0415 — only reachable with device arrays

        dst[:] = cupy.asarray(src)
    else:
        dst[:] = src


def set_backend(name: str) -> None:
    """Switch the global array backend.

    Args:
        name: One of ``'numpy'`` or ``'cupy'``.

    Note:
        CuPy support is planned for v1.1. Setting ``'cupy'`` in v1.0 will raise
        ImportError unless CuPy is manually installed.
    """
    global _BACKEND
    if name not in ("numpy", "cupy"):
        raise ValueError(f"Unknown backend: {name!r}. Valid options: 'numpy', 'cupy'")
    _BACKEND = name


def get_backend() -> str:
    """Return the name of the currently active backend."""
    return _BACKEND
