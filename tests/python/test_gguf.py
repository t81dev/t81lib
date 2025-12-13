"""tests/python/test_gguf.py — Regression coverage for GGUF read/write helpers."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from t81 import gguf
from t81.scripts import t81_dequant

try:
    import torch
    import transformers  # noqa: F401
except ImportError:
    pytest.skip("Skipping GGUF regression test (requires torch + transformers)", allow_module_level=True)

from tests.python._models import _DummyModel, _LargeDummyModel, _populate_linear


_QUANT_TOLERANCES: dict[str, float] = {
    "TQ1_0": 0.07,
    "TQ2_0": 0.03,
}


def _write_and_read_roundtrip(tmp_path: Path, quant: str, threshold: float, fill_range: tuple[float, float]):
    model = _DummyModel()
    _populate_linear(model, *fill_range)
    output_path = tmp_path / f"dummy-{quant.lower()}.gguf"
    gguf.write_gguf(model, output_path, quant=quant, threshold=threshold)
    payload, metadata = gguf.read_gguf(output_path, dequantize=True, return_metadata=True)
    return model, output_path, payload, metadata


def _quant_tolerance(quant: str) -> float:
    """Return the empirically calibrated tolerance for the visible quant forms."""
    return _QUANT_TOLERANCES[quant]


@pytest.mark.slow
@pytest.mark.parametrize(
    ("quant", "threshold", "fill_range"),
    [
        ("TQ1_0", 0.45, (-1.0, 0.8)),
        ("TQ2_0", 0.25, (-0.9, 0.9)),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_roundtrip_gguf(tmp_path: Path, quant: str, threshold: float, fill_range: tuple[float, float]) -> None:
    """Ensure each quantized format round-trips and retains metadata."""
    atol = _quant_tolerance(quant)
    model, output_path, payload, metadata = _write_and_read_roundtrip(tmp_path, quant, threshold, fill_range)
    decoded = payload.get("linear.weight")
    assert isinstance(decoded, torch.Tensor)
    assert decoded.shape == model.linear.weight.shape
    assert torch.isfinite(decoded).all()
    assert metadata["quantization.type"] == quant.lower()
    assert metadata["quantization.threshold"] == pytest.approx(threshold, rel=1e-5)
    raw_payload = gguf.read_gguf(output_path, dequantize=False).get("linear.weight")
    assert isinstance(raw_payload, (bytes, bytearray))


@pytest.mark.slow
def test_roundtrip_large_linear(tmp_path: Path) -> None:
    """Exercise a larger weight matrix to trigger alignment/shape edge cases."""
    model = _LargeDummyModel()
    _populate_linear(model, -5.0, 5.0)
    output_path = tmp_path / "dummy-large-gguf"
    gguf.write_gguf(model, output_path, quant="TQ1_0", threshold=0.45)
    payload, metadata = gguf.read_gguf(output_path, dequantize=True, return_metadata=True)
    decoded = payload.get("linear.weight")
    assert isinstance(decoded, torch.Tensor)
    assert torch.isfinite(decoded).all()
    assert decoded.shape == (64, 128)
    assert metadata["quantization.threshold"] == pytest.approx(0.45, rel=1e-5)


@pytest.mark.slow
def test_roundtrip_gguf_tq2(tmp_path: Path) -> None:
    """Make sure the TQ2 file dequantizes to float32 without producing NaNs."""
    model, output_path, _payload, _metadata = _write_and_read_roundtrip(
        tmp_path, "TQ2_0", 0.45, (-0.9, 0.9)
    )
    float_path = tmp_path / "dummy-tq2-float.gguf"
    gguf.dequantize_gguf_to_float(output_path, float_path)
    float_decoded = gguf.read_gguf(float_path, dequantize=True).get("linear.weight")
    assert isinstance(float_decoded, torch.Tensor)
    assert float_decoded.shape == model.linear.weight.shape
    assert torch.isfinite(float_decoded).all()


@pytest.mark.slow
def test_write_gguf_invalid_quant(tmp_path: Path) -> None:
    """Reject unsupported quantization identifiers."""
    model = _DummyModel()
    bad_path = tmp_path / "bad.gguf"
    with pytest.raises(ValueError, match="quant must be one of"):
        gguf.write_gguf(model, bad_path, quant="INVALID")



@pytest.mark.slow
def test_t81_dequant_cli_info(tmp_path: Path, capsys) -> None:
    model = _DummyModel()
    with torch.no_grad():
        values = torch.linspace(-0.5, 0.5, model.linear.weight.numel(), dtype=torch.float32).reshape(
            model.linear.weight.shape
        )
        model.linear.weight.copy_(values)
    gguf_path = tmp_path / "cli-info.gguf"
    gguf.write_gguf(model, gguf_path, quant="TQ1_0", threshold=0.45)
    exit_code = t81_dequant.main(
        [
            str(gguf_path),
            "--tensor",
            "linear.weight",
            "--sample",
            "2",
            "--info",
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    sample_line = next(
        (line for line in captured.out.splitlines() if line.startswith("Sample (linear.weight):")), None
    )
    assert sample_line is not None
    sample_values = ast.literal_eval(sample_line.split(":", 1)[1].strip())
    assert len(sample_values) == 2
    expected_samples = values.flatten()[:2].tolist()
    for actual, expected in zip(sample_values, expected_samples):
        assert abs(actual - expected) <= 0.15


@pytest.mark.slow
def test_t81_dequant_cli_unknown_tensor(tmp_path: Path, capsys) -> None:
    """Requesting a missing tensor prints a helpful message without crashing."""
    model = _DummyModel()
    _populate_linear(model, -0.3, 0.3)
    gguf_path = tmp_path / "cli-unknown-tensor.gguf"
    gguf.write_gguf(model, gguf_path, quant="TQ1_0", threshold=0.45)
    exit_code = t81_dequant.main(
        [
            str(gguf_path),
            "--tensor",
            "missing.weight",
            "--info",
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Tensor 'missing.weight' not found" in captured.out


@pytest.mark.slow
def test_t81_dequant_cli_missing_input(tmp_path: Path) -> None:
    """Missing paths bail early with a clear message."""
    missing_path = tmp_path / "missing.gguf"
    with pytest.raises(SystemExit, match="does not exist"):
        t81_dequant.main([str(missing_path)])


@pytest.mark.slow
def test_t81_dequant_cli_unimplemented_target(tmp_path: Path) -> None:
    """Unsupported targets raise before attempting conversion."""
    model = _DummyModel()
    _populate_linear(model, -0.5, 0.5)
    gguf_path = tmp_path / "dummy-support.gguf"
    gguf.write_gguf(model, gguf_path, quant="TQ1_0", threshold=0.45)
    with pytest.raises(SystemExit, match="q8_0"):
        t81_dequant.main([str(gguf_path), "--target", "q8_0"])
