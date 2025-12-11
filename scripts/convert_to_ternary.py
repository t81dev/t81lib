#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CLI helper that converts a model to use ternary linear layers."""

from __future__ import annotations

import argparse
import pathlib
import tempfile
from typing import Optional

import torch
from torch import nn

from t81.nn import Linear as TernaryLinear

try:
    from transformers import AutoModelForCausalLM
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None  # type: ignore[assignment]

try:
    from safetensors.torch import save_file as save_safetensors
except ImportError:  # pragma: no cover
    save_safetensors = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """

    parser = argparse.ArgumentParser(
        description="Convert a PyTorch/HuggingFace model to use ternary Linear."
    )
    parser.add_argument(
        "model",
        help="Hugging Face repo id or local .pt/.pth/.safetensors file to convert.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Balanced ternary quantization threshold (0 < threshold <= 1).",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output path (defaults to '<input>-ternary.<ext>').",
    )
    return parser.parse_args()


def load_model(source: str) -> tuple[nn.Module, str, Optional[pathlib.Path]]:
    """Load the model from disk or the Hugging Face Hub.

    Args:
        source: Model identifier.

    Returns:
        tuple[nn.Module, str, Optional[pathlib.Path]]: Loaded module, suffix,
            and original path when available.
    """

    path = pathlib.Path(source)
    if path.exists():
        return _load_local_model(path)
    if AutoModelForCausalLM is None:
        raise RuntimeError(
            "transformers is required to load Hugging Face models; "
            "install it via 'pip install transformers'."
        )
    model = AutoModelForCausalLM.from_pretrained(source)
    return model, ".pt", None


def _load_local_model(path: pathlib.Path) -> tuple[nn.Module, str, pathlib.Path]:
    """Load a module that was serialized to a local file.

    Args:
        path: Local path to the saved module.

    Returns:
        tuple[nn.Module, str, pathlib.Path]: Module, suffix, and path.
    """

    suffix = path.suffix.lower()
    if suffix not in {".pt", ".pth", ".safetensors"}:
        raise ValueError("Model file must be .pt, .pth, or .safetensors")
    model = torch.load(path)
    return model, suffix, path


def build_output_path(
    override: Optional[pathlib.Path], source: str, suffix: str
) -> pathlib.Path:
    """Compute the output path for the converted weights.

    Args:
        override: Custom output path provided by the user.
        source: Original source string.
        suffix: File suffix to use for the converted model.

    Returns:
        pathlib.Path: Path to write the converted module.
    """

    if override is not None:
        return override
    source_path = pathlib.Path(source)
    if source_path.exists():
        stem = source_path.with_suffix("").name
    else:
        stem = source.replace("/", "_").replace(":", "_")
    return pathlib.Path(f"{stem}-ternary{suffix}")


def replace_linears(module: nn.Module, threshold: float) -> int:
    """Recursively swap Linear modules for ternary equivalents.

    Args:
        module: Module tree to inspect.
        threshold: Quantization threshold for ternary weights.

    Returns:
        int: Number of replaced Linear modules.
    """

    replaced = 0
    for name, child in list(module._modules.items()):
        if isinstance(child, nn.Linear) and not isinstance(child, TernaryLinear):
            bias = child.bias is not None
            new_linear = TernaryLinear(
                child.in_features,
                child.out_features,
                bias=bias,
                device=child.weight.device,
                dtype=child.weight.dtype,
                threshold=threshold,
            )
            with torch.no_grad():
                new_linear.weight.copy_(child.weight)
                new_linear.weight.requires_grad = child.weight.requires_grad
                if bias and child.bias is not None and new_linear.bias is not None:
                    new_linear.bias.copy_(child.bias)
                    new_linear.bias.requires_grad = child.bias.requires_grad
            new_linear.training = child.training
            module._modules[name] = new_linear
            replaced += 1
        else:
            replaced += replace_linears(child, threshold)
    return replaced


def save_model(model: nn.Module, output_path: pathlib.Path, suffix: str) -> None:
    """Persist the converted model in the requested serialization format.

    Args:
        model: Module with ternary Linear layers.
        output_path: Destination file path.
        suffix: Serialization suffix (.pt/.pth/.safetensors).
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if suffix in {".pt", ".pth"}:
        torch.save(model, output_path)
        return
    if suffix == ".safetensors":
        if save_safetensors is None:
            raise RuntimeError("safetensors is required to save .safetensors files")
        save_safetensors(model.state_dict(), str(output_path))
        return
    raise ValueError("Unsupported output suffix")


def measure_serialized_size(model: nn.Module, suffix: str) -> int:
    """Estimate the serialized size of a module.

    Args:
        model: Module to serialize.
        suffix: Serialization suffix to use.

    Returns:
        int: Size in bytes of the serialized module.
    """

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = pathlib.Path(tmp.name)
    try:
        save_model(model, tmp_path, suffix)
        return tmp_path.stat().st_size
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> None:
    """Main entry point for the conversion CLI.

    This function parses CLI arguments, loads the model, swaps its Linear
    layers for ternary equivalents, and saves the converted artifact.
    """

    args = parse_args()
    if not 0 < args.threshold <= 1:
        raise SystemExit("Threshold must satisfy 0 < threshold <= 1")
    model, suffix, source_path = load_model(args.model)
    output_path = build_output_path(args.output, args.model, suffix)
    if output_path.suffix.lower() != suffix:
        output_path = output_path.with_suffix(suffix)
    if source_path and source_path.exists():
        original_size = source_path.stat().st_size
    else:
        original_size = measure_serialized_size(model, suffix)
    replaced = replace_linears(model, args.threshold)
    if replaced == 0:
        print("No Linear modules were converted; the model already uses ternary layers.")
    save_model(model, output_path, suffix)
    new_size = output_path.stat().st_size
    converted_base = (
        f"Converted {args.model!r} -> {output_path} with {replaced} "
        "ternary Linear(s). "
    )
    if original_size is not None:
        reduction = original_size / new_size if new_size else float("inf")
        stats = (
            f"Original size={original_size}B, new size={new_size}B, "
            f"reduction={reduction:.2f}x."
        )
    else:
        stats = f"New size={new_size}B."
    print(converted_base + stats)


if __name__ == "__main__":
    main()
