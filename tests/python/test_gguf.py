"""
tests/python/test_gguf.py — Regression coverage for GGUF read/write helpers.

This script ensures ``t81.gguf.write_gguf`` and ``t81.gguf.read_gguf`` can round-trip
balanced ternary tensors and exposes the feature as a lightweight regression guard.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import t81
from t81 import gguf


def main() -> int:
    try:
        import torch
        from transformers import PreTrainedModel, PretrainedConfig
    except ImportError:
        print(
            "Skipping GGUF regression test (requires torch + transformers)",
            file=sys.stderr,
        )
        return 0

    class _DummyConfig(PretrainedConfig):
        model_type = "dummy"
        architectures = ("DummyModel",)

    class _DummyModel(PreTrainedModel):
        config_class = _DummyConfig
        base_model_prefix = "dummy"

        def __init__(self, config: PretrainedConfig | None = None):
            config = config or _DummyConfig()
            super().__init__(config)
            self.linear = t81.Linear(4, 3)

        def forward(self, tensor: torch.Tensor) -> torch.Tensor:  # pragma: no cover - not executed
            return self.linear(tensor)

    def run_roundtrip(tmp_path: Path) -> bool:
        model = _DummyModel()
        with torch.no_grad():
            values = torch.linspace(
                -1.0, 0.8, steps=model.linear.weight.numel(), dtype=torch.float32
            ).reshape(model.linear.weight.shape)
            model.linear.weight.copy_(values)
        output_path = tmp_path / "dummy.gguf"
        gguf.write_gguf(model, output_path, quant="TQ1_0", threshold=0.45)
        decoded = gguf.read_gguf(output_path, dequantize=True)
        tensor = decoded.get("linear.weight")
        if not isinstance(tensor, torch.Tensor):
            return False
        if tensor.shape != model.linear.weight.shape:
            return False
        if tensor.isnan().any():
            return False
        if tensor.abs().max() > model.linear.weight.abs().max() + 1e-3:
            return False
        raw_payload = gguf.read_gguf(output_path, dequantize=False).get("linear.weight")
        if not isinstance(raw_payload, (bytes, bytearray)):
            return False
        return True

    with tempfile.TemporaryDirectory() as tempdir:
        if not run_roundtrip(Path(tempdir)):
            print("GGUF round-trip regression failed", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
