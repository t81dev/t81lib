"""Helpers that verify GGUF files with llama.cpp tooling or the Python reader."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _llama_cpp_validator() -> list[str] | None:
    for name in ("gguf_validate", "gguf_to_gguf"):
        exe = shutil.which(name)
        if exe is not None:
            return [exe]
    return None


def validate_with_llama_cpp(gguf_path: Path) -> None:
    """Use llama.cpp tooling when available to validate a GGUF file."""

    validator = _llama_cpp_validator()
    if validator is None:
        return
    command = [*validator, "-i", str(gguf_path)]
    if validator[0].endswith("gguf_to_gguf"):
        command.extend(["-o", str(gguf_path.with_suffix(".valid.gguf"))])
    _run_command(command)


def validate_with_reader(gguf_path: Path) -> None:
    """Fall back to the `t81.gguf` reader if llama.cpp tooling is unavailable."""

    try:
        from t81 import gguf
    except ImportError as exc:
        raise RuntimeError("gguf reader validation requires t81lib in Python") from exc
    decoded = gguf.read_gguf(str(gguf_path), dequantize=False)
    if not decoded:
        raise RuntimeError("GGUF reader returned no tensors")


def validate_gguf_file(gguf_path: Path) -> None:
    """Ensure the GGUF file loads via llama.cpp tooling or the Python reader."""

    gguf_path = gguf_path.resolve()
    if not gguf_path.exists():
        raise FileNotFoundError(f"{gguf_path} not found for validation")
    validate_with_llama_cpp(gguf_path)
    validate_with_reader(gguf_path)
