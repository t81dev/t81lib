#!/usr/bin/env python3
"""Benchmark quantize→latency+energy for a converted model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel

import t81lib
import t81.convert  # registers AutoModel.from_pretrained_t81
from t81.hardware import TernaryEmulator
from t81.nn import Linear as TernaryLinear
from scripts.quantize_measure import (
    compression_summary,
    find_first_ternary_linear,
    measure_linear_latency,
    run_convert,
)


def pack_trits(linear: TernaryLinear) -> np.ndarray:
    """Pack a linear weight into trits so the emulator can trace transitions."""

    weight = linear.weight.detach().cpu().to(dtype=torch.float32).numpy()
    packed = t81lib.pack_dense_matrix(weight, threshold=linear.ternary_threshold)
    trits = t81lib.unpack_packed_limbs(packed, rows=weight.shape[0], cols=weight.shape[1])
    return np.asarray(trits, dtype=np.int8)


def simulate_energy(
    trits: np.ndarray,
    power_model: dict[str, float],
    fuzzy_threshold: float,
) -> dict[str, Any]:
    """Run the emulator over the trit matrix and return energy summaries."""

    emulator = TernaryEmulator(power_model=power_model, fuzzy_threshold=fuzzy_threshold)
    for row_idx, row in enumerate(trits):
        row_list = row.tolist()
        for trit_idx, trit in enumerate(row_list):
            emulator.set_wire(f"row-{row_idx}-{trit_idx}", int(trit))
        emulator.ripple_adder(row_list, row_list[::-1], name_prefix=f"sum-{row_idx}")
    emulator.clock_tick()
    return {
        "energy_consumed": emulator.energy_consumed,
        "transitions": dict(emulator.transition_counter),
        "power_trace_len": len(emulator.power_trace),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automated quantize → latency/energy benchmark.")
    parser.add_argument("model_id", help="Hugging Face model ID or local dir to convert.")
    parser.add_argument("output_dir", help="Directory where the converted checkpoint is stored.")
    parser.add_argument("--threshold", type=float, default=0.45, help="Ternary quantization threshold.")
    parser.add_argument("--device-map", default="auto", help="`t81-convert` device map.")
    parser.add_argument("--force-cpu", action="store_true", help="Keep the converted checkpoint on CPU.")
    parser.add_argument("--torch-dtype", help="Optional torch dtype for the float copy.")
    parser.add_argument("--iterations", type=int, default=32, help="Timing iterations for latencies.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for latency runs.")
    parser.add_argument(
        "--power-model",
        nargs=2,
        action="append",
        metavar=("MODE", "SCALE"),
        help="Optional power model entries for the emulator (mode scale).",
    )
    parser.add_argument("--output", type=Path, help="CSV/JSON file to append benchmark rows.")
    return parser.parse_args()


def build_power_model(entries: list[tuple[str, str]] | None) -> dict[str, float]:
    """Turn CLI key/value entries into a power model dict."""

    defaults = {"ternary": 1.0, "binary": 0.6}
    if not entries:
        return defaults
    model = {}
    for key, val in entries:
        model[key] = float(val)
    return {**defaults, **model}


def write_output(output: Path, row: dict[str, Any]) -> None:
    """Append benchmark metadata to CSV or write JSON based on extension."""

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".json":
        data = []
        if output.exists():
            data = json.loads(output.read_text())
            if not isinstance(data, list):
                data = []
        data.append(row)
        output.write_text(json.dumps(data, indent=2))
        return
    fieldnames = list(row.keys())
    existed = output.exists()
    with output.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not existed:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    run_convert(args)
    print(f"Loaded ternary checkpoint from {args.output_dir} via AutoModel.from_pretrained_t81")
    model = AutoModel.from_pretrained_t81(args.output_dir)
    linear = find_first_ternary_linear(model)
    if linear is None:
        raise SystemExit("converted checkpoint contains no ternary Linear modules")
    float_latency, ternary_latency = measure_linear_latency(
        linear,
        iterations=args.iterations,
        batch_size=args.batch_size,
    )
    float_bytes, ternary_bytes, ratio = compression_summary(linear)
    trits = pack_trits(linear)
    energy = simulate_energy(
        trits,
        build_power_model(args.power_model),
        fuzzy_threshold=0.3,
    )
    experiment = {
        "model_id": args.model_id,
        "threshold": args.threshold,
        "float_latency_ms": float_latency * 1e3,
        "ternary_latency_ms": ternary_latency * 1e3,
        "compression_ratio": ratio,
        "ternary_bytes": ternary_bytes,
        "float_bytes": float_bytes,
        "energy_consumed": energy["energy_consumed"],
        "power_trace_len": energy["power_trace_len"],
        "transitions_total": sum(energy["transitions"].values()),
        "transitions": energy["transitions"],
    }
    print("Benchmark results:")
    for key, value in experiment.items():
        print(f"  {key}: {value}")
    if args.output:
        write_output(args.output, experiment)


if __name__ == "__main__":
    main()
