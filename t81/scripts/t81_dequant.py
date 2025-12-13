"""
t81-dequant — convert TQ1_0/TQ2_0 GGUF bundles back to float32 so stock runtimes can load them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from t81 import gguf


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dequantize a TQ1_0/TQ2_0 GGUF bundle into float-compatible tensors."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the TQ1_0/TQ2_0 GGUF bundle produced by t81-convert.",
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        help="Optional destination GGUF path for the rewritten bundle.",
    )
    parser.add_argument(
        "--target",
        choices=("f16", "f32", "q8_0"),
        default="f16",
        help="Output tensor type (f16 yields best compression, q8_0 is not implemented yet).",
    )
    parser.add_argument(
        "--tensor",
        type=str,
        help="Optional tensor name to print metadata or sample values for (defaults to first tensor).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="When used with `--info`, print the first N dequantized values for the selected tensor.",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print metadata/tensor info without writing a new file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without writing any files.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational logging.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="After writing the GGUF bundle, reload it to ensure it loads cleanly.",
    )
    parser.add_argument(
        "--list-tensors",
        action="store_true",
        help="List all tensor names (similar to --info) and exit.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="t81-dequant",
        help="Show the version of the t81-dequant helper.",
    )
    return parser.parse_args(argv)


def _default_output(input_path: Path, target: str) -> Path:
    suffix = f"-{target}.gguf"
    return input_path.with_name(f"{input_path.stem}{suffix}")


def _target_config(target: str) -> tuple[np.dtype, int]:
    if target == "f16":
        return np.float16, 101
    if target == "f32":
        return np.float32, 100
    raise SystemExit("--target q8_0 currently not implemented (planned in a follow-up)")


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.input.exists():
        raise SystemExit(f"{args.input!s} does not exist")

    output_path = args.output or _default_output(args.input, args.target)
    info_msg = f"Dequantizing {args.input.name} → {output_path.name} (target={args.target})"
    if not args.quiet:
        print(info_msg)

    dtype, ggml_type = _target_config(args.target)

    quantized_payload: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None
    decoded_payload: Mapping[str, torch.Tensor | bytes] | None = None
    tensor_names_cache: list[str] | None = None

    def _ensure_quantized():
        nonlocal quantized_payload, metadata
        if quantized_payload is None:
            quantized_payload, metadata = gguf.read_gguf(args.input, dequantize=False, return_metadata=True)
        return quantized_payload, metadata

    def _ensure_decoded():
        nonlocal decoded_payload
        if decoded_payload is None:
            decoded_payload = gguf.read_gguf(args.input, dequantize=True)
        return decoded_payload

    def _tensor_names():
        nonlocal tensor_names_cache
        if tensor_names_cache is None:
            payload, _ = _ensure_quantized()
            tensor_names_cache = sorted(payload.keys())
        return tensor_names_cache

    def _print_sample(selected_tensor: str, payload: Mapping[str, Any]) -> None:
        decoded = _ensure_decoded()
        sample_tensor = decoded.get(selected_tensor)
        if isinstance(sample_tensor, torch.Tensor):
            arr = sample_tensor.flatten()[: args.sample].cpu().numpy()
            formatted = np.array2string(arr, threshold=10, edgeitems=5, floatmode="fixed", precision=4)
            print(f"Sample ({selected_tensor}): {arr.tolist()}")
            print(f"Sample ({selected_tensor}) [{arr.shape}]:")
            print(formatted)
            raw_chunk = payload.get(selected_tensor)
            if isinstance(raw_chunk, (bytes, bytearray)) and len(raw_chunk) >= 2:
                first_scale = float(np.frombuffer(raw_chunk[:2], dtype=np.float16)[0])
                print(f"First block scale ≈ {first_scale:.4f}")
        else:
            print(f"Sample request ignored: tensor {selected_tensor!r} is not numeric")

    if args.info:
        payload, metadata = _ensure_quantized()
        alignment = metadata.get("general.alignment", 32)
        print(f"Alignment={alignment}, tensors={len(payload)} (quantized), metadata keys={len(metadata)}")
        quant_type = metadata.get("quantization.type")
        threshold = metadata.get("quantization.threshold", "unknown")
        block_size = metadata.get("quantization.block_size", "unknown")
        if quant_type is not None:
            print(f"Quantization: {quant_type.upper()} (threshold={threshold}, block_size={block_size})")
        for key, value in sorted(metadata.items()):
            print(f"  {key} = {value!r}")
        tensor_names = _tensor_names()
        if tensor_names:
            print("Tensors:")
            for name in tensor_names:
                print(f"  - {name}")
        selected_tensor = args.tensor or (tensor_names[0] if tensor_names else None)
        if selected_tensor and args.sample > 0:
            _print_sample(selected_tensor, payload)
        elif args.tensor and selected_tensor not in payload:
            print(f"Tensor {args.tensor!r} not found in the file")

    if args.list_tensors and not args.info:
        tensor_names = _tensor_names()
        if tensor_names:
            print("Tensors:")
            for name in tensor_names:
                print(f"  - {name}")
        else:
            print("No tensors found in the file.")
        return 0

    if args.sample > 0 and not args.info:
        payload, _ = _ensure_quantized()
        tensor_names = _tensor_names()
        selected_tensor = args.tensor or (tensor_names[0] if tensor_names else None)
        if not selected_tensor:
            print("No tensors available to sample.")
        elif selected_tensor not in payload:
            print(f"Tensor {selected_tensor!r} not found in the file")
        else:
            _print_sample(selected_tensor, payload)
    if args.dry_run:
        print("Dry run completed.")
        return 0

    gguf.dequantize_gguf(args.input, output_path, dtype=dtype, ggml_type=ggml_type)
    if args.validate:
        gguf.read_gguf(output_path, dequantize=True)
    if not args.quiet:
        print("Conversion complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
