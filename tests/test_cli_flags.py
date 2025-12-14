"""Smoke tests that cover `t81-convert`/`t81-gguf` flag combinations."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from transformers import AutoModelForSequenceClassification, BertConfig


def _build_dummy_hf_model(directory: Path) -> None:
    config = BertConfig(
        vocab_size=64,
        hidden_size=32,
        num_attention_heads=4,
        intermediate_size=64,
        num_hidden_layers=1,
        num_labels=2,
    )
    model = AutoModelForSequenceClassification.from_config(config)
    model.save_pretrained(directory)


def _run_convert(source: Path, target: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "t81",
        "convert",
        str(source),
        str(target),
        "--device-map",
        "none",
        "--force-cpu-device-map",
        "--torch-dtype",
        "float16",
    ]
    subprocess.run(cmd, check=True)


def _run_gguf(source: Path, gguf_path: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "t81",
        "gguf",
        str(gguf_path),
        "--from-hf",
        str(source),
        "--device-map",
        "none",
        "--force-cpu-device-map",
        "--torch-dtype",
        "float16",
        "--quant",
        "TQ2_0",
    ]
    subprocess.run(cmd, check=True)


def _run_info(path: Path) -> None:
    cmd = [sys.executable, "-m", "t81", "info", str(path)]
    subprocess.run(cmd, check=True)


def test_convert_respects_flags(tmp_path: Path) -> None:
    source = tmp_path / "hf"
    target = tmp_path / "converted"
    source.mkdir()
    _build_dummy_hf_model(source)
    _run_convert(source, target)
    assert (target / "t81_metadata.json").exists()
    _run_info(target)


def test_gguf_respects_device_map(tmp_path: Path) -> None:
    source = tmp_path / "hf"
    gguf_file = tmp_path / "model.gguf"
    source.mkdir()
    _build_dummy_hf_model(source)
    _run_gguf(source, gguf_file)
    assert gguf_file.exists() and gguf_file.stat().st_size > 0
    _run_info(gguf_file)
