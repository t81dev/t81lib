import warnings

import pytest


torch = pytest.importorskip("torch")


def _best_device() -> torch.device | None:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return None


def test_ternary_tensor_cpu_roundtrip():
    import t81.torch as t81_torch

    weight = torch.linspace(-1.0, 1.0, steps=48, dtype=torch.float32).reshape(3, 16)
    ternary = t81_torch.TernaryTensor.from_float(weight, threshold=0.45)
    rhs = torch.randn(16, 4, dtype=torch.float32)
    out = torch.matmul(ternary, rhs)
    assert out.shape == (3, 4)
    assert out.device.type == "cpu"


def test_ternary_tensor_gpu_fallback_warning():
    device = _best_device()
    if device is None:
        pytest.skip("No GPU/MPS device available for fallback test.")

    import t81.torch as t81_torch

    weight = torch.linspace(-1.0, 1.0, steps=48, dtype=torch.float32, device=device).reshape(3, 16)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        ternary = t81_torch.TernaryTensor.from_float(weight, threshold=0.45)
    assert any("moving tensor" in str(item.message) for item in caught)
    rhs = torch.randn(16, 4, dtype=torch.float32, device=device)
    out = torch.matmul(ternary, rhs)
    assert out.device == device
