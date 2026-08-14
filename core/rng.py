from __future__ import annotations

import math
from numbers import Integral

import numpy as np
from numba import njit


RNG_ALGORITHM_VERSION = "counter-splitmix64-v1"
_GOLDEN_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_MIX_MULTIPLIER_1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX_MULTIPLIER_2 = np.uint64(0x94D049BB133111EB)
_STREAM_TAG = np.uint64(0xD2B74407B1CE6E93)
_NORMAL_STREAM_TAG = np.uint64(0xCA5A826395121157)
_U32_MASK = np.uint64(0xFFFFFFFF)
_MIN_UNIFORM = 1.0 / float(1 << 53)
_U64_MAX = (1 << 64) - 1


@njit("uint64(uint64,uint64)", inline="always")
def _add_u64(left: np.uint64, right: np.uint64) -> np.uint64:
    low = (left & _U32_MASK) + (right & _U32_MASK)
    carry = low >> np.uint64(32)
    high = (left >> np.uint64(32)) + (right >> np.uint64(32)) + carry
    return (low & _U32_MASK) | ((high & _U32_MASK) << np.uint64(32))


@njit("uint64(uint64,uint64)", inline="always")
def _multiply_u64(left: np.uint64, right: np.uint64) -> np.uint64:
    left_low = left & _U32_MASK
    left_high = left >> np.uint64(32)
    right_low = right & _U32_MASK
    right_high = right >> np.uint64(32)
    low_product = left_low * right_low
    high = (
        (low_product >> np.uint64(32))
        + ((left_high * right_low) & _U32_MASK)
        + ((left_low * right_high) & _U32_MASK)
    )
    return (low_product & _U32_MASK) | ((high & _U32_MASK) << np.uint64(32))


@njit("uint64(uint64)", inline="always")
def _splitmix64_u64(value: np.uint64) -> np.uint64:
    value = _add_u64(value, _GOLDEN_GAMMA)
    value = _multiply_u64(
        value ^ (value >> np.uint64(30)),
        _MIX_MULTIPLIER_1,
    )
    value = _multiply_u64(
        value ^ (value >> np.uint64(27)),
        _MIX_MULTIPLIER_2,
    )
    return value ^ (value >> np.uint64(31))


@njit(inline="always")
def splitmix64(value: np.uint64) -> np.uint64:
    """Give a 64-bit hash value."""
    return _splitmix64_u64(np.uint64(value))


@njit("uint64(uint64,uint64,uint64)", inline="always")
def _counter_u64_u64(
    seed: np.uint64,
    counter: np.uint64,
    stream: np.uint64,
) -> np.uint64:
    value = _splitmix64_u64(seed)
    value = _splitmix64_u64(value ^ counter)
    return _splitmix64_u64(value ^ _add_u64(stream, _STREAM_TAG))


@njit(inline="always")
def counter_u64(
    seed: np.uint64,
    counter: np.uint64,
    stream: np.uint64 = np.uint64(0),
) -> np.uint64:
    """Give a deterministic 64-bit value."""
    return _counter_u64_u64(
        np.uint64(seed),
        np.uint64(counter),
        np.uint64(stream),
    )


@njit(inline="always")
def counter_uniform(
    seed: np.uint64,
    counter: np.uint64,
    stream: np.uint64 = np.uint64(0),
) -> float:
    """Give a deterministic uniform value."""
    value = counter_u64(seed, counter, stream)
    return float(value >> np.uint64(11)) * _MIN_UNIFORM


@njit(inline="always")
def counter_normal(
    seed: np.uint64,
    counter: np.uint64,
    stream: np.uint64 = np.uint64(0),
) -> float:
    """Give a deterministic standard normal value."""
    stream_u64 = np.uint64(stream)
    u1 = counter_uniform(seed, counter, stream_u64)
    u2 = counter_uniform(seed, counter, _add_u64(stream_u64, _NORMAL_STREAM_TAG))
    if u1 < _MIN_UNIFORM:
        u1 = _MIN_UNIFORM
    radius = math.sqrt(-2.0 * math.log(u1))
    return radius * math.cos(2.0 * math.pi * u2)


def _as_u64(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"Set {name} to an integer.")
    number = int(value)
    if number < 0 or number > _U64_MAX:
        raise ValueError(f"Set {name} to an integer from 0 through 2^64 - 1.")
    return number


def derive_seed(root_seed: int, *components: int) -> int:
    """Calculate a deterministic child seed."""
    value = _as_u64("root_seed", root_seed)
    for index, component in enumerate(components, start=1):
        item = _as_u64(f"component_{index}", component)
        value = int(counter_u64(np.uint64(value), np.uint64(item), np.uint64(index)))
    return value
