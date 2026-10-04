"""Shared backing allocations must not inflate the memory report."""

from dataclasses import dataclass

import numpy as np

from magnelio._memory import array_bytes


def test_views_retain_one_whole_backing_allocation():
    backing = np.empty(100, dtype=np.float64)
    separate = np.empty(20, dtype=np.int32)
    assert array_bytes([backing[:2], backing[50:], separate]) == 880


def test_external_buffer_aliases_and_cycles():
    buffer = bytearray(100)
    arrays = [np.frombuffer(buffer, dtype=np.uint8), np.frombuffer(buffer, dtype=np.int32)]
    arrays.append(arrays)
    assert array_bytes(arrays) == 100


def test_nested_dataclass_array_aliases():
    @dataclass
    class Data:
        values: object
        other: object

    backing = np.empty(7, dtype=np.float64)
    assert array_bytes(Data({"array": backing}, (backing,))) == 56
