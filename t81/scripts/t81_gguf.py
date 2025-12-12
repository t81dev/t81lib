"""
CLI for exporting transformers models as ternary GGUF files.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from t81.cli_progress import CLIProgress
from t81.cli_validator import validate_gguf_file


def _handle_missing_dependency(exc: ImportError) -> None:
    names = getattr(exc, "name", None)
    print(
        "t81-gguf depends on the torch + transformers extras from t81lib.",
        file=sys.stderr,
    )
    print(
        "Install them with `pip install .[torch]` (or `pip install t81lib[torch]`) and retry.",
        file=sys.stderr,
    )
    if names:
        print(f"ImportError: No module named {names}", file=sys.stderr)
    else:
        print(f"ImportError: {exc}", file=sys.stderr)


def main() -> int:
    try:
        from t81 import gguf
    except ImportError as exc:
        _handle_missing_dependency(exc)
        return 1
    try:
        convert = importlib.import_module("t81.convert")
    except ImportError as exc:
        _handle_missing_dependency(exc)
        return 1

    def _parse_dtype(value: str):
        return convert._parse_dtype(value)

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
        "--force-cpu-device-map",
        action="store_true",
        dest="force_cpu_device_map",
        help="Force `device_map=None` so the converted weights stay on CPU/disk instead of using accelerate offloading.",
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
    parser.add_argument(
        "--validate",
        action="store_true",
        help="After exporting the GGUF bundle, run llama.cpp's validator (or the Python reader) against it.",
    )
    args = parser.parse_args()

    if bool(args.from_hf) == bool(args.from_t81):
        parser.error("provide exactly one of --from-hf or --from-t81")

    source = args.from_hf or args.from_t81 or ""
    progress = CLIProgress("t81-gguf", total_steps=2)
    model = convert.convert(
        source,
        threshold=args.threshold,
        keep_biases_bf16=args.keep_biases_bf16,
        device_map=convert._normalize_device_map_arg(args.device_map),
        torch_dtype=args.torch_dtype,
        force_cpu_device_map=args.force_cpu_device_map,
    )
    progress.step("converted ternary model")

    gguf.write_gguf(
        model,
        Path(args.output),
        quant=args.quant,
        threshold=args.threshold,
    )
    progress.step("wrote GGUF bundle")
    if args.validate:
        validate_gguf_file(Path(args.output))

    return 0
