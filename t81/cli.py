"""
Unified CLI dispatcher for t81lib.

The standalone ``t81`` entry point now exposes ``t81 convert`` and ``t81 gguf`` subcommands.
Compatibility wrappers still allow the legacy ``t81-convert``/``t81-gguf`` console scripts
to work while sharing the same parsing logic.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Sequence

from .cli_progress import CLIProgress
from .cli_validator import validate_gguf_file
from .convert import _METADATA_FILENAME
from .gguf import HEADER_MAGIC, HEADER_SIZE, HEADER_STRUCT, _parse_metadata_from_file


def _handle_missing_dependency(exc: ImportError) -> None:
    names = getattr(exc, "name", None)
    print(
        "The t81 CLI depends on the torch + transformers extras from t81lib.",
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


def _import_convert_module():
    try:
        from . import convert

        return convert
    except ImportError as exc:  # pragma: no cover - only triggered when extras missing
        _handle_missing_dependency(exc)
        raise SystemExit(1) from exc


def _import_gguf_module():
    try:
        from . import gguf

        return gguf
    except ImportError as exc:  # pragma: no cover - only triggered when extras missing
        _handle_missing_dependency(exc)
        raise SystemExit(1) from exc


def _add_convert_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[arg-type]
    parser = subparsers.add_parser(
        "convert",
        help="Convert a Hugging Face checkpoint to a ternary-aware HF directory.",
        description="Convert a Hugging Face model to `t81.nn.Linear` layers and optionally export GGUF.",
    )
    parser.add_argument("model_id_or_path", help="Pretrained model identifier or local directory.")
    parser.add_argument("output_dir", help="Destination directory for the converted model.")
    parser.add_argument("--threshold", type=float, default=0.45, help="Ternary quantization threshold.")
    parser.add_argument(
        "--keep-biases-bf16",
        action="store_true",
        dest="keep_biases_bf16",
        default=True,
        help="Keep bias tensors in BF16/FP32 when possible.",
    )
    parser.add_argument(
        "--no-keep-biases-bf16",
        action="store_false",
        dest="keep_biases_bf16",
        help="Force biases to float32.",
    )
    parser.add_argument("--device-map", default="auto", help="Device map passed to transformers.")
    parser.add_argument(
        "--force-cpu-device-map",
        action="store_true",
        dest="force_cpu_device_map",
        help="Avoid accelerate offloading so tensors stay on CPU for saving.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=lambda value: _import_convert_module()._parse_dtype(value),
        help="Optional torch dtype for the floating-point buffer used during conversion.",
    )
    parser.add_argument(
        "--output-gguf",
        metavar="PATH",
        help="Also write a GGUF file at the given path.",
    )
    parser.add_argument(
        "--gguf-quant",
        choices=("TQ1_0", "TQ2_0"),
        default="TQ1_0",
        help="GGUF quantization format to emit (when --output-gguf is provided).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the GGUF bundle (if written) before exiting.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress bars and stats.",
    )
    parser.set_defaults(func=_handle_convert)


def _add_gguf_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[arg-type]
    parser = subparsers.add_parser(
        "gguf",
        help="Convert a HF checkpoint or t81 directory directly to GGUF.",
        description="Run conversion + GGUF serialization in one step.",
    )
    parser.add_argument("output", help="Output GGUF file path.")
    parser.add_argument(
        "--from-hf",
        help="HF model identifier or directory to convert before writing GGUF.",
        dest="from_hf",
    )
    parser.add_argument(
        "--from-t81",
        help="Existing t81-converted directory to re-export.",
        dest="from_t81",
    )
    parser.add_argument(
        "--quant",
        choices=["TQ1_0", "TQ2_0"],
        default="TQ1_0",
        help="Ternary GGUF format to emit.",
    )
    parser.add_argument("--threshold", type=float, default=0.45, help="Threshold for ternary conversion.")
    parser.add_argument("--device-map", default="auto", help="Device map forwarded to transformers.")
    parser.add_argument(
        "--force-cpu-device-map",
        action="store_true",
        dest="force_cpu_device_map",
        help="Disable accelerate so tensors remain on CPU for saving.",
    )
    parser.add_argument(
        "--keep-biases-bf16",
        action="store_true",
        dest="keep_biases_bf16",
        default=True,
        help="Keep bias tensors in BF16/FP32 when available.",
    )
    parser.add_argument(
        "--no-keep-biases-bf16",
        action="store_false",
        dest="keep_biases_bf16",
        help="Force all biases to float32.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=lambda value: _import_convert_module()._parse_dtype(value),
        help="Optional torch dtype for conversion before GGUF.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run llama.cpp's GGUF validator (or the Python reader) after writing the bundle.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress bars and stats.",
    )
    parser.set_defaults(func=_handle_gguf)


def _add_info_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[arg-type]
    parser = subparsers.add_parser(
        "info",
        help="Display metadata for a converted directory or GGUF bundle.",
        description="Show threshold, quant type, and metadata for t81 outputs.",
    )
    parser.add_argument("source", help="Directory (converted checkpoint) or .gguf file.")
    parser.set_defaults(func=_handle_info)


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="t81", description="t81lib unified CLI.")
    parser.add_argument(
        "--version",
        action="version",
        version=f"t81lib {importlib.metadata.version('t81lib')}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_convert_parser(subparsers)
    _add_gguf_parser(subparsers)
    _add_info_parser(subparsers)
    return parser


def _handle_convert(args: argparse.Namespace) -> int:
    convert = _import_convert_module()
    gguf_module = _import_gguf_module()
    total_steps = 2 + (1 if args.output_gguf else 0)
    progress = CLIProgress("t81 convert", total_steps=total_steps, quiet=args.quiet)
    model = convert.convert(
        args.model_id_or_path,
        threshold=args.threshold,
        keep_biases_bf16=args.keep_biases_bf16,
        device_map=convert._normalize_device_map_arg(args.device_map),
        torch_dtype=args.torch_dtype,
        force_cpu_device_map=args.force_cpu_device_map,
    )
    progress.step("converted HF checkpoint")
    model.save_pretrained_t81(args.output_dir)
    progress.step("saved ternary checkpoint")
    if args.output_gguf:
        gguf_module.write_gguf(
            model,
            Path(args.output_gguf),
            quant=args.gguf_quant,
            threshold=args.threshold,
        )
        progress.step("wrote GGUF bundle")
        if args.validate:
            validate_gguf_file(Path(args.output_gguf))
    return 0


def _handle_gguf(args: argparse.Namespace) -> int:
    convert = _import_convert_module()
    gguf_module = _import_gguf_module()
    if bool(args.from_hf) == bool(args.from_t81):
        raise SystemExit("provide exactly one of --from-hf or --from-t81")
    source = args.from_hf or args.from_t81  # type: ignore[assignment]
    progress = CLIProgress("t81 gguf", total_steps=2, quiet=args.quiet)
    model = convert.convert(
        source,
        threshold=args.threshold,
        keep_biases_bf16=args.keep_biases_bf16,
        device_map=convert._normalize_device_map_arg(args.device_map),
        torch_dtype=args.torch_dtype,
        force_cpu_device_map=args.force_cpu_device_map,
    )
    progress.step("converted ternary model")
    gguf_module.write_gguf(
        model,
        Path(args.output),
        quant=args.quant,
        threshold=args.threshold,
    )
    progress.step("wrote GGUF bundle")
    if args.validate:
        validate_gguf_file(Path(args.output))
    return 0


def _print_dir_info(directory: Path) -> int:
    metadata_path = directory / _METADATA_FILENAME
    if not metadata_path.exists():
        print(f"Missing {metadata_path.name} in {directory}", file=sys.stderr)
        return 1
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    threshold = metadata.get("threshold")
    keep_biases = metadata.get("keep_biases_bf16")
    print(f"t81 converted directory: {directory}")
    print(f"  threshold: {threshold}")
    print(f"  keep_biases_bf16: {keep_biases}")
    return 0


def _print_gguf_info(path: Path) -> int:
    with path.open("rb") as handle:
        header = handle.read(HEADER_SIZE)
        if len(header) < HEADER_SIZE:
            print("file too short to be a GGUF bundle", file=sys.stderr)
            return 1
        magic, version, num_tensors, metadata_count = HEADER_STRUCT.unpack_from(header, 0)
        if magic != HEADER_MAGIC:
            print("invalid GGUF header", file=sys.stderr)
            return 1
        metadata, _ = _parse_metadata_from_file(handle, metadata_count)
    print(f"GGUF bundle: {path}")
    print(f"  version: {version}")
    print(f"  tensors: {num_tensors}")
    print(f"  architecture: {metadata.get('general.architecture')}")
    print(f"  name: {metadata.get('general.name')}")
    threshold = metadata.get("quantization.threshold")
    if threshold is not None:
        print(f"  quant threshold: {threshold}")
    quant_type = metadata.get("quantization.type")
    if quant_type:
        print(f"  quant type: {quant_type}")
    print(f"  block size: {metadata.get('quantization.block_size')}")
    return 0


def _handle_info(args: argparse.Namespace) -> int:
    source = Path(args.source)
    if source.is_dir():
        return _print_dir_info(source)
    if source.is_file():
        return _print_gguf_info(source)
    print(f"{source} does not exist", file=sys.stderr)
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the unified ``t81`` CLI."""
    parser = _create_parser()
    args = parser.parse_args(None if argv is None else list(argv))
    return args.func(args)


def main_convert(argv: Sequence[str] | None = None) -> int:
    args = ["convert"] + (list(argv) if argv is not None else sys.argv[1:])
    return main(args)


def main_gguf(argv: Sequence[str] | None = None) -> int:
    args = ["gguf"] + (list(argv) if argv is not None else sys.argv[1:])
    return main(args)
