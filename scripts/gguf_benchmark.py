#!/usr/bin/env python3
"""
Benchmark a GGUF bundle with llama.cpp and report size/RAM/latency metrics.

This script is intentionally dependency-free so it can run anywhere Python is
available. It expects a llama.cpp-compatible `llama-cli` binary.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import resource
import shutil
import subprocess
import sys
from pathlib import Path


_TIMING_LINE_RE = re.compile(r"(llama_print_timings:|common_perf_print:).*")
_MS_PER_TOKEN_RE = re.compile(r"([0-9.]+) ms per token")
_TOKENS_PER_SEC_RE = re.compile(r"([0-9.]+) tokens per second")
_COMMON_PROMPT_RE = re.compile(r"prompt eval time = .*\\(([^)]+)\\)")
_COMMON_EVAL_RE = re.compile(r"eval time = .*\\(([^)]+)\\)")


def _find_llama_cli(path: str) -> str:
    candidate = Path(path)
    if candidate.exists():
        return str(candidate)
    resolved = shutil.which(path)
    if resolved:
        return resolved
    raise SystemExit(f"unable to locate llama.cpp binary: {path}")


def _child_maxrss() -> int:
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss


def _rss_to_mib(rss: int) -> float | None:
    if rss <= 0:
        return None
    if platform.system() == "Darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def _extract_timings(output: str) -> dict[str, float | None]:
    timing_lines = [line for line in output.splitlines() if _TIMING_LINE_RE.search(line)]
    prompt_line = next((line for line in timing_lines if "prompt eval time" in line), "")
    eval_line = next((line for line in timing_lines if "eval time" in line and "prompt" not in line), "")

    def _parse_line(line: str) -> tuple[float | None, float | None]:
        ms_match = _MS_PER_TOKEN_RE.search(line)
        tok_match = _TOKENS_PER_SEC_RE.search(line)
        ms_per_token = float(ms_match.group(1)) if ms_match else None
        tokens_per_sec = float(tok_match.group(1)) if tok_match else None
        return ms_per_token, tokens_per_sec

    prompt_ms, prompt_tps = _parse_line(prompt_line)
    eval_ms, eval_tps = _parse_line(eval_line)
    if prompt_ms is None and prompt_line:
        match = _COMMON_PROMPT_RE.search(prompt_line)
        if match:
            prompt_ms, prompt_tps = _parse_line(match.group(1))
    if eval_ms is None and eval_line:
        match = _COMMON_EVAL_RE.search(eval_line)
        if match:
            eval_ms, eval_tps = _parse_line(match.group(1))
    return {
        "prompt_ms_per_token": prompt_ms,
        "prompt_tokens_per_sec": prompt_tps,
        "eval_ms_per_token": eval_ms,
        "eval_tokens_per_sec": eval_tps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark a GGUF bundle with llama.cpp and report size/RAM/latency.",
    )
    parser.add_argument("--gguf", required=True, help="Path to the GGUF file.")
    parser.add_argument(
        "--llama-cli",
        default="llama-cli",
        help="Path or name of the llama.cpp llama-cli binary.",
    )
    parser.add_argument(
        "--prompt",
        default="In ternary we trust.",
        help="Prompt used for batch=1 timing.",
    )
    parser.add_argument("--n-predict", type=int, default=64, help="Tokens to generate.")
    parser.add_argument("--ctx", type=int, default=2048, help="Context length.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size.")
    parser.add_argument("--threads", type=int, help="Thread count (llama.cpp -t).")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    parser.add_argument(
        "--json-output",
        help="Optional path to write a JSON summary report.",
    )
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to llama-cli after --extra.",
    )
    args = parser.parse_args()

    gguf_path = Path(args.gguf)
    if not gguf_path.exists():
        raise SystemExit(f"GGUF path does not exist: {gguf_path}")
    llama_cli = _find_llama_cli(args.llama_cli)

    cmd = [
        llama_cli,
        "-m",
        str(gguf_path),
        "-p",
        args.prompt,
        "-n",
        str(args.n_predict),
        "-c",
        str(args.ctx),
        "-b",
        str(args.batch),
    ]
    if args.threads:
        cmd.extend(["-t", str(args.threads)])
    if args.extra:
        cmd.extend(args.extra)

    before_rss = _child_maxrss()
    result = subprocess.run(cmd, capture_output=True, text=True)
    after_rss = _child_maxrss()

    output = (result.stdout or "") + "\n" + (result.stderr or "")
    if result.returncode != 0:
        print(output, file=sys.stderr)
        raise SystemExit(result.returncode)

    max_rss_delta = max(0, after_rss - before_rss)
    report = {
        "gguf_path": str(gguf_path),
        "model_size_bytes": gguf_path.stat().st_size,
        "model_size_mib": gguf_path.stat().st_size / (1024 * 1024),
        "peak_rss_mib": _rss_to_mib(max_rss_delta),
        "batch": args.batch,
        "ctx": args.ctx,
        "n_predict": args.n_predict,
        "prompt": args.prompt,
    }
    report.update(_extract_timings(output))

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"GGUF benchmark (batch={args.batch})")
        print(f"  model: {report['gguf_path']} ({report['model_size_mib']:.2f} MiB)")
        if report["peak_rss_mib"] is not None:
            print(f"  peak RSS: {report['peak_rss_mib']:.2f} MiB")
        else:
            print("  peak RSS: unavailable")
        eval_ms = report.get("eval_ms_per_token")
        eval_tps = report.get("eval_tokens_per_sec")
        if eval_ms is not None or eval_tps is not None:
            print(f"  eval: {eval_ms} ms/token, {eval_tps} tok/s")
        else:
            print("  eval: timing not found in llama.cpp output")
        prompt_ms = report.get("prompt_ms_per_token")
        prompt_tps = report.get("prompt_tokens_per_sec")
        if prompt_ms is not None or prompt_tps is not None:
            print(f"  prompt: {prompt_ms} ms/token, {prompt_tps} tok/s")
        else:
            print("  prompt: timing not found in llama.cpp output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
