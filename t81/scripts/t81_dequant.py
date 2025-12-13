"""
t81-dequant — convert TQ1_0/TQ2_0 GGUF bundles back to float32 so stock runtimes can load them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

from t81 import dequantize_gguf


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
    return parser.parse_args(argv)


def _default_output(input_path: Path, target: str) -> Path:
    suffix = f"-{target}.gguf"
    return input_path.with_name(f"{input_path.stem}{suffix}")


def _target_config(target: str) -> tuple[np.dtype, int]:
    if target == "f16":
        return np.float16, 101
    if target == "f32":
        return np.float32, 100
    raise SystemExit("--target q8_0 currently not implemented")


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.input.exists():
        raise SystemExit(f"{args.input!s} does not exist")

    output_path = args.output or _default_output(args.input, args.target)
    if args.info or not args.quiet:
        info_msg = f"Dequantizing {args.input.name} → {output_path.name} (target={args.target})"
        print(info_msg)

    dtype, ggml_type = _target_config(args.target)
    if args.info:
        metadata, _ = gguf.read_gguf(args.input, dequantize=False, return_metadata=True)
        alignment = metadata.get("general.alignment", 32)
        print(f"Alignment={alignment}, tensors={len(metadata)} keys, target dtype={dtype}")
    if args.dry_run:
        print("Dry run completed.")
        return 0

    dequantize_gguf(args.input, output_path, dtype=dtype, ggml_type=ggml_type)
    if not args.quiet:
        print("Conversion complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
