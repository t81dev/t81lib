"""
CLI for exporting transformers models as ternary GGUF files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from t81 import convert
from t81 import gguf


def _parse_dtype(value: str):
    return convert._parse_dtype(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a HF model to t81 and export it as a ternary GGUF file."
    )
    parser.add_argument("output", help="Output GGUF file path.")
    parser.add_argument(
        "--from-hf",
        help="Hugging Face model identifier or directory to convert.",
    )
    parser.add_argument(
        "--from-t81",
        help="Existing t81-converted directory to re-export.",
    )
    parser.add_argument(
        "--quant",
        choices=["TQ1_0", "TQ2_0"],
        default="TQ1_0",
        help="Ternary GGUF format to emit.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Threshold used during ternary quantization.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map forwarded to transformers during conversion.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=_parse_dtype,
        help="Optional torch dtype for the conversion.",
    )
    parser.add_argument(
        "--keep-biases-bf16",
        dest="keep_biases_bf16",
        action="store_true",
        default=True,
        help="Keep biases in BF16/FP32 when available.",
    )
    parser.add_argument(
        "--no-keep-biases-bf16",
        dest="keep_biases_bf16",
        action="store_false",
        help="Force all biases to float32.",
    )
    args = parser.parse_args()

    if bool(args.from_hf) == bool(args.from_t81):
        parser.error("provide exactly one of --from-hf or --from-t81")

    source = args.from_hf or args.from_t81 or ""
    model = convert.convert(
        source,
        threshold=args.threshold,
        keep_biases_bf16=args.keep_biases_bf16,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )

    gguf.write_gguf(
        model,
        Path(args.output),
        quant=args.quant,
        threshold=args.threshold,
    )

    return 0
