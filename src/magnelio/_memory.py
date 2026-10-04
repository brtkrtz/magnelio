"""Allocation accounting without reading or copying array contents."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass

import numpy as np


def array_bytes(value: object) -> int:
    """Count distinct NumPy backing allocations in a data structure.

    Views retain their entire backing allocation. Shared arrays, cycles,
    and aliased buffers are counted once. Python objects and native CAD
    allocations are outside this accounting.
    """
    seen: set[int] = set()
    buffers: set[int] = set()

    def visit(item: object) -> int:
        if id(item) in seen:
            return 0
        seen.add(id(item))
        if isinstance(item, np.ndarray):
            owner = item
            while isinstance(owner.base, np.ndarray):
                owner = owner.base
            backing = owner.base if owner.base is not None else owner
            if isinstance(backing, memoryview):
                backing = backing.obj
            if id(backing) in buffers:
                return 0
            buffers.add(id(backing))
            if isinstance(backing, np.ndarray):
                return backing.nbytes
            try:
                return memoryview(backing).nbytes
            except TypeError:
                return owner.nbytes
        if isinstance(item, Mapping):
            return sum(visit(v) for v in item.values())
        if isinstance(item, (tuple, list)):
            return sum(visit(v) for v in item)
        if is_dataclass(item) and not isinstance(item, type):
            return sum(visit(getattr(item, f.name)) for f in fields(item))
        return 0

    return visit(value)


def format_bytes(value: int) -> str:
    """Format binary storage units for compact human-readable reports."""
    for unit, scale in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if value >= scale:
            number = value / scale
            return f"{number:.1f} {unit}" if number < 10 else f"{number:.0f} {unit}"
    return f"{value} B"
