#!/usr/bin/env python3
"""Convert a model via the CLI and measure its ternary inference latency."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoModel

from t81.nn import Linear as TernaryLinear
import t81.convert  # ensure AutoModel.from_pretrained_t81 is registered


def _timeit(func, iterations: int = 32) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    return (time.perf_counter() - start) / iterations


def run_convert(args: argparse.Namespace) -> None:
    """Invoke the `t81-convert` CLI from the current environment."""

    cmd = [
        sys.executable,
        "-m",
        "t81.convert",
        args.model_id,
        args.output_dir,
        "--threshold",
        str(args.threshold),
        "--device-map",
        args.device_map,
    ]
    if args.force_cpu:
        cmd.append("--force-cpu-device-map")
    if args.torch_dtype:
        cmd.extend(["--torch-dtype", args.torch_dtype])
    subprocess.run(cmd, check=True)


def find_first_ternary_linear(model: torch.nn.Module) -> TernaryLinear | None:
    for module in model.modules():
        if isinstance(module, TernaryLinear):
            return module
    return None


def measure_linear_latency(
    linear: TernaryLinear,
    iterations: int = 32,
    batch_size: int = 16,
) -> Tuple[float, float]:
    """Return (float32, ternary) GEMM latencies using cached training weights."""

    device = linear.weight.device
    weight = linear.weight.detach().to(device)
    bias = linear.bias.detach().to(device) if linear.bias is not None else None
    sample = torch.randn(batch_size, weight.shape[1], device=device)

    def float_matmul() -> None:
        with torch.no_grad():
            torch.nn.functional.linear(sample, weight, bias=bias)

    linear.eval()

    def ternary_matmul() -> None:
        with torch.no_grad():
            linear(sample)

    float_latency = _timeit(float_matmul, iterations)
    ternary_latency = _timeit(ternary_matmul, iterations)
    return float_latency, ternary_latency


def compression_summary(linear: TernaryLinear) -> Tuple[int, int, float]:
    rows, cols = linear.weight.shape
    limbs = (cols + 47) // 48
    ternary_bytes = rows * limbs * 16
    float_bytes = linear.weight.numel() * linear.weight.element_size()
    ratio = float_bytes / ternary_bytes if ternary_bytes else float("inf")
    return float_bytes, ternary_bytes, ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chain t81-convert + latency measurement.")
    parser.add_argument("model_id", help="HF model ID or local directory to convert.")
    parser.add_argument("output_dir", help="Directory to write the converted checkpoint.")
    parser.add_argument("--threshold", type=float, default=0.45, help="Ternary quantization threshold.")
    parser.add_argument("--device-map", default="auto", help="Device map forwarded to transformers.")
    parser.add_argument("--force-cpu", action="store_true", help="Keep the converted model on CPU.")
    parser.add_argument("--torch-dtype", help="Optional torch dtype (e.g., float16, bfloat16).")
    parser.add_argument("--iterations", type=int, default=32, help="Number of timing iterations.")
    parser.add_argument("--batch-size", type=int, default=16, help="Synthetic batch size for inference.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    run_convert(args)
    print(f"t81-convert finished, loading {args.output_dir} via AutoModel.from_pretrained_t81")
    model = AutoModel.from_pretrained_t81(args.output_dir)
    linear = find_first_ternary_linear(model)
    if linear is None:
        raise SystemExit("no ternary Linear module found inside the converted model")
    float_latency, ternary_latency = measure_linear_latency(
        linear,
        iterations=args.iterations,
        batch_size=args.batch_size,
    )
    float_bytes, ternary_bytes, ratio = compression_summary(linear)
    print("inference timing (ms/call):")
    print(f"  float32 F.linear : {float_latency * 1e3:.3f} ms")
    print(f"  ternary GEMM    : {ternary_latency * 1e3:.3f} ms")
    print(
        f"storage (per linear): float={float_bytes / 1024:.2f} KiB, "
        f"ternary={ternary_bytes / 1024:.2f} KiB, ratio={ratio:.2f}",
    )


if __name__ == "__main__":
    main()
