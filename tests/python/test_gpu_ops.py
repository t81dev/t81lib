# tests/python/test_gpu_ops.py — Smoke tests for the new GPU-friendly helpers.

import array
import math
import time

import pytest

try:
    import t81lib
except ImportError as exc:
    raise SystemExit(
        "Failed to import t81lib when running GPU helper tests."
    ) from exc


def make_float_array(values):
    return array.array("f", values)


def test_where_basic():
    cond = make_float_array([1.0, 0.0, -0.5, 2.0])
    x = make_float_array([10.0, 10.0, 10.0, 10.0])
    y = make_float_array([-1.0, -1.0, -1.0, -1.0])
    result = t81lib.where(cond, x, y)
    assert result.tolist() == [10.0, -1.0, -1.0, 10.0]


def test_clamp_basic():
    values = make_float_array([-2.0, -0.5, 0.5, 3.0])
    result = t81lib.clamp(values, -1.0, 2.0)
    assert result.tolist() == [-1.0, -0.5, 0.5, 2.0]


def test_lerp_basic():
    start = make_float_array([0.0, 1.0])
    end = make_float_array([1.0, 3.0])
    weight = make_float_array([0.25, 0.5])
    result = t81lib.lerp(start, end, weight)
    assert result.tolist() == [0.25, 2.0]


def test_addcmul_basic():
    base = make_float_array([1.0, 2.0])
    tensor1 = make_float_array([1.0, 0.5])
    tensor2 = make_float_array([2.0, 4.0])
    result = t81lib.addcmul(base, tensor1, tensor2, value=0.5)
    assert result.tolist() == [2.0, 6.0]


@pytest.mark.skipif(
    not getattr(t81lib, "HAS_CUDA_BACKEND", False),
    reason="CUDA backend not enabled",
)
def test_where_cuda_backend_available():
    cond = make_float_array([1.0])
    x = make_float_array([5.0])
    y = make_float_array([0.0])
    result = t81lib.where(cond, x, y)
    assert result.tolist() == [5.0]


@pytest.mark.skipif(
    not getattr(t81lib, "HAS_CUDA_BACKEND", False),
    reason="CUDA backend not enabled",
)
def test_where_cuda_latency_and_accuracy():
    length = 1 << 20
    cond = array.array("f", (1.0 if index % 2 == 0 else 0.0 for index in range(length)))
    x = array.array("f", (float(index) for index in range(length)))
    y = array.array("f", (float(length + index) / 2.0 for index in range(length)))
    start_ts = time.perf_counter()
    result = t81lib.where(cond, x, y)
    elapsed = time.perf_counter() - start_ts
    assert elapsed < 1.5, "GPU where should stay responsive for large buffers"
    for index, value in enumerate(result):
        expected = x[index] if cond[index] != 0.0 else y[index]
        assert math.isclose(value, expected, abs_tol=1e-6)


@pytest.mark.skipif(
    not getattr(t81lib, "HAS_ROCM_BACKEND", False),
    reason="ROCm backend not enabled",
)
def test_clamp_rocm_latency_and_accuracy():
    length = 1 << 14
    values = array.array("f", ((index % 5) - 2.5 for index in range(length)))
    start_ts = time.perf_counter()
    result = t81lib.clamp(values, -1.0, 1.0)
    elapsed = time.perf_counter() - start_ts
    assert elapsed < 0.5, "ROCm clamp should stay responsive"
    for index, value in enumerate(result):
        expected = min(1.0, max(-1.0, values[index]))
        assert math.isclose(value, expected, abs_tol=1e-6)
