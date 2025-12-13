"""Regression guard that validates GGUF output via llama.cpp tooling when it is available."""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import torch
    import transformers  # noqa: F401
except ImportError:
    pytest.skip(
        "Skipping llama.cpp integration (requires torch + transformers)", allow_module_level=True
    )

from t81 import gguf
from t81 import cli_validator
from tests.python._models import _DummyModel, _populate_linear


def _write_dummy_gguf(tmp_path: Path) -> Path:
    model = _DummyModel()
    _populate_linear(model, -1.0, 1.0)
    output_path = tmp_path / "dummy-llama-compat.gguf"
    gguf.write_gguf(model, output_path, quant="TQ1_0", threshold=0.45)
    return output_path


@pytest.mark.slow
def test_llama_cpp_validator_roundtrip(tmp_path: Path) -> None:
    """Run llama.cpp's validator (if installed) against a small GGUF export."""
    validator = cli_validator._llama_cpp_validator()
    if validator is None:
        pytest.skip("llama.cpp gguf validator not available")
    gguf_path = _write_dummy_gguf(tmp_path)
    cli_validator.validate_with_llama_cpp(gguf_path)
    valid_path = gguf_path.with_suffix(".valid.gguf")
    if valid_path.exists():
        valid_path.unlink()
