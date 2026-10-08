"""Per-build selection of the optional bounded-surface section route."""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

_ROBUST_SECTIONS = ContextVar("magnelio_robust_sections", default=None)


def robust_sections_enabled() -> bool:
    selected = _ROBUST_SECTIONS.get()
    if selected is not None:
        return selected
    # Internal A/B probes outside a mesh build; the public control is authoritative.
    return os.environ.get("MAGNELIO_SURFACE_SECTIONS", "").strip() == "1"


@contextmanager
def section_policy(enabled: bool):
    token = _ROBUST_SECTIONS.set(enabled)
    try:
        yield
    finally:
        _ROBUST_SECTIONS.reset(token)


def mesh_section_policy(build):
    @wraps(build)
    def selected(cls, geometry, control, *args, **kwargs):
        with section_policy(control.robust_sections):
            return build(cls, geometry, control, *args, **kwargs)

    return selected


def worker_section_policy(enabled: bool) -> None:
    _ROBUST_SECTIONS.set(enabled)
