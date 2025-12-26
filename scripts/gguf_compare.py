#!/usr/bin/env python3
"""
Compare two GGUF benchmark JSON reports and print deltas.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_report(path: Path) -> dict[str, float | int | str | None]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _fmt_delta(label: str, base: float | None, cand: float | None, unit: str) -> str:
    if base is None or cand is None:
        return f"{label}: unavailable"
    delta = cand - base
    sign = "+" if delta >= 0 else ""
    return f"{label}: {cand:.3f}{unit} ({sign}{delta:.3f}{unit})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare GGUF benchmark JSON reports.")
    parser.add_argument("--baseline", required=True, help="Baseline JSON report.")
    parser.add_argument("--candidate", required=True, help="Candidate JSON report.")
    parser.add_argument("--baseline-label", default="baseline", help="Label for the baseline report.")
    parser.add_argument("--candidate-label", default="candidate", help="Label for the candidate report.")
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    baseline = _load_report(baseline_path)
    candidate = _load_report(candidate_path)

    print(f"GGUF benchmark comparison ({args.baseline_label} -> {args.candidate_label})")
    print(
        _fmt_delta(
            "model_size_mib",
            baseline.get("model_size_mib"),
            candidate.get("model_size_mib"),
            " MiB",
        )
    )
    print(
        _fmt_delta(
            "peak_rss_mib",
            baseline.get("peak_rss_mib"),
            candidate.get("peak_rss_mib"),
            " MiB",
        )
    )
    print(
        _fmt_delta(
            "eval_ms_per_token",
            baseline.get("eval_ms_per_token"),
            candidate.get("eval_ms_per_token"),
            " ms/token",
        )
    )
    print(
        _fmt_delta(
            "eval_tokens_per_sec",
            baseline.get("eval_tokens_per_sec"),
            candidate.get("eval_tokens_per_sec"),
            " tok/s",
        )
    )
    print(
        _fmt_delta(
            "prompt_ms_per_token",
            baseline.get("prompt_ms_per_token"),
            candidate.get("prompt_ms_per_token"),
            " ms/token",
        )
    )
    print(
        _fmt_delta(
            "prompt_tokens_per_sec",
            baseline.get("prompt_tokens_per_sec"),
            candidate.get("prompt_tokens_per_sec"),
            " tok/s",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
