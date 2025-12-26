<!--
docs/references/cli-usage.md — Usage notes for the console scripts.
-->

# CLI helpers

The `t81lib` package exposes multiple CLI helpers once you install the `torch`/`transformers` extras:

```bash
pipx install .[torch]  # or pip install t81lib[torch] if you prefer a shared venv
```

After that the unified `t81` CLI (with `convert`/`gguf` subcommands), `t81-qat`, and `t81-dequant` land in `~/.local/bin` (or pipx’s `bin` directory). The old `t81-convert`/`t81-gguf` names still exist as wrappers for backward compatibility.

## `t81 convert` (legacy `t81-convert`)

Convert any Hugging Face checkpoint into the ternary-aware runtime that powers `t81.torch`/`t81.nn`.

```bash
t81 convert meta-llama/Llama-3.2-3B-Instruct path/to/converted --quant TQ1_0 --torch-dtype bfloat16
```

`t81 convert` outputs a lightweight progress line (bar + percentage) for conversion, checkpointing, and optional GGUF export so you can monitor long-lived runs without missing other stderr logs. The legacy `t81-convert` name still invokes this codepath.

After you emit a GGUF bundle with `--output-gguf`, the new `--validate` flag reads the file via `gguf.read_gguf` and, if llama.cpp’s `gguf_validate`/`gguf_to_gguf` is installed, passes the file through that tool to ensure compatibility before the command exits.

Key flags:

- `--threshold`: the ternary cutoff (default `0.45`). Adjust this to trade sparsity vs. activation range.
- `--device-map`: forwards a `transformers` device map string; use `auto` (default) for Accelerate offloading or `none`/`cpu` to keep everything on host RAM.
- `--force-cpu-device-map`: overrides `transformers` so the conversion stays on CPU, which prevents `NotImplementedError: Cannot copy out of meta tensor` when saving and guarantees the `t81_metadata.json` footprint is serializable.
- `--keep-biases-bf16` / `--no-keep-biases-bf16`: control whether bias tensors stay in BF16 or are promoted to FP32.
- `--torch-dtype`: optional dtype for the floating-point buffer used during quantization.
- `--output-gguf`: pipe the converted model straight into `gguf.write_gguf` (see `t81 gguf` below) with a `--gguf-quant` choice of `TQ1_0` or `TQ2_0`.
- `--gguf-profile`: apply a named GGUF export profile (e.g. `compression-first`), overriding `--gguf-quant` and the conversion threshold.
- `--validate`: after writing the GGUF bundle, run llama.cpp’s GGUF validator (or the Python reader) so you can detect incompatible exports before the command succeeds.

The command rewrites every `nn.Linear` into `t81.nn.Linear`, stores metadata in `t81_metadata.json`, and emits compression stats so you can see how much VRAM you save.

## `t81 gguf` (legacy `t81-gguf`)

Write a ternary GGUF bundle that works with `llama.cpp`, Ollama, or LM Studio without rerunning the conversion step.

```bash
t81 gguf out.3.t81.gguf --from-hf meta-llama/Llama-3.2-3B-Instruct --quant TQ2_0
```

Alternatively, re-export an existing t81-converted directory:

```bash
t81 gguf out.3.t81.gguf --from-t81 path/to/converted
```

`t81 gguf` also prints a brief progress line tied to its conversion and GGUF serialization stages. The older `t81-gguf` name remains available as a wrapper.

Pass `--validate` when you want the fresh GGUF bundle checked by both the Python reader and llama.cpp’s validator so incompatibilities are caught before you ship the file.

Use the same `--threshold`, `--device-map`, `--torch-dtype`, and `--force-cpu-device-map` knobs as `t81 convert` because `t81 gguf` delegates to that CLI internally.

Use `--profile compression-first` to force the compression-first profile (TQ1_0 + default threshold) and stamp profile metadata into the bundle. Use `--profile tq1_1-draft` with `T81_ENABLE_TQ1_1=1` to write the experimental TQ1_1 payloads.

### Compression-first wedge (FP16 to ternary GGUF)

If you want a single compression-first export profile that loads in llama.cpp without extra flags, use the `compression-first` profile. It stamps `t81.profile=compression-first` in the GGUF metadata and pins the quantization scheme to TQ1_0 with the default threshold.

1) Export a baseline FP16 GGUF with llama.cpp (or reuse an existing FP16 GGUF bundle):

```bash
python llama.cpp/convert.py meta-llama/Llama-3.2-3B-Instruct --outtype f16 --outfile llama3.2-3b-f16.gguf
```

2) Export the ternary GGUF via the compression-first profile:

```bash
t81 gguf llama3.2-3b-tq1.gguf --from-hf meta-llama/Llama-3.2-3B-Instruct --profile compression-first
```

3) Benchmark both bundles for size, peak RAM, and batch=1 latency:

```bash
python scripts/gguf_benchmark.py --gguf llama3.2-3b-f16.gguf --llama-cli /path/to/llama-cli --n-predict 128
python scripts/gguf_benchmark.py --gguf llama3.2-3b-tq1.gguf --llama-cli /path/to/llama-cli --n-predict 128
```

Capture the reported MiB on disk, peak RSS, and `eval` ms/token so the before/after comparison is repeatable.

Example before/after (TinyLlama-1.1B Q4_0 to TQ1_0, batch=1, CPU-only):

```bash
python scripts/gguf_benchmark.py --gguf tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  --llama-cli /opt/homebrew/bin/llama-cli --n-predict 128 \
  --prompt "In ternary we trust. The goal is compression-first inference on today’s binary machines. We compare GGUF exports for size, RAM, and latency at batch=1. Use a fixed prompt to measure prompt eval and token eval timings." \
  --extra --device none --n-gpu-layers 0

python scripts/gguf_benchmark.py --gguf tinyllama-1.1b-tq1.gguf \
  --llama-cli /opt/homebrew/bin/llama-cli --n-predict 128 \
  --prompt "In ternary we trust. The goal is compression-first inference on today’s binary machines. We compare GGUF exports for size, RAM, and latency at batch=1. Use a fixed prompt to measure prompt eval and token eval timings." \
  --extra --device none --n-gpu-layers 0
```

Observed numbers (Apple M2, llama.cpp brew build):

```
Baseline Q4_0: size 608.16 MiB, peak RSS 1190.02 MiB, eval 55.94 ms/token (17.88 tok/s)
TQ1_0:         size 696.89 MiB, peak RSS  760.58 MiB, eval 55.51 ms/token (18.01 tok/s)
```

Tensor/metadata sanity check (same model, same tensor names/shapes):

```
Baseline types: Q4_0 (155), F32 (45), Q6_K (1), file_type=2, kv_count=23
TQ1_0 types:    TQ1_0 (154), F32 (47), file_type=36, kv_count=31
```

### TQ1_1 draft header layout (size-only sketch)

Candidate layout for a tighter on-disk TQ1_0 variant without changing math:

- Current TQ1_0 block: 48-byte qs + 4-byte qh + 2-byte FP16 scale = 54 bytes.
- Draft TQ1_1: keep qs/qh, store FP8 (e4m3) scale per block.
  - Scale payload per block: 1 byte.
  - Estimated block size: 48 + 4 + 1 = 53 bytes.
  - Estimated savings: 54 -> 53 bytes (~1.9% per block).

This is a header-size sketch only; `--profile tq1_1-draft` requires `T81_ENABLE_TQ1_1=1` and writes experimental payloads that llama.cpp will not load today.

## Compression-first report template

Use this template to keep FP16 ↔ TQ1_0 comparisons consistent:

```
Model:
Baseline GGUF:
Ternary GGUF:
Profile:
llama.cpp commit:
llama-cli path:
Load flags:
Prompt:
n_predict / ctx / batch:

Baseline metrics:
  size_mib:
  peak_rss_mib:
  eval_ms_per_token:
  eval_tokens_per_sec:
  prompt_ms_per_token:
  prompt_tokens_per_sec:

Ternary metrics:
  size_mib:
  peak_rss_mib:
  eval_ms_per_token:
  eval_tokens_per_sec:
  prompt_ms_per_token:
  prompt_tokens_per_sec:

Accuracy sanity:
  dataset:
  metric:
  baseline_score:
  ternary_score:
  delta:

Size audit:
  script:
  total_delta_bytes:
```

You can automate report capture with JSON summaries:

```bash
python scripts/gguf_benchmark.py --gguf llama3.2-3b-f16.gguf \
  --llama-cli /path/to/llama-cli --n-predict 128 \
  --json-output reports/llama3.2-3b-f16.json

python scripts/gguf_benchmark.py --gguf llama3.2-3b-tq1.gguf \
  --llama-cli /path/to/llama-cli --n-predict 128 \
  --json-output reports/llama3.2-3b-tq1.json

python scripts/gguf_compare.py \
  --baseline reports/llama3.2-3b-f16.json \
  --candidate reports/llama3.2-3b-tq1.json \
  --baseline-label fp16 --candidate-label tq1

python scripts/gguf_size_audit.py --gguf llama3.2-3b-tq1.gguf \
  > reports/llama3.2-3b-tq1-size.csv
```

For auditability, run a quick accuracy sanity check for the `compression-first` profile and record the delta:

```bash
t81 gguf llama3.2-3b-tq1.gguf --from-hf meta-llama/Llama-3.2-3B-Instruct \
  --profile compression-first

# Example: plug in your existing eval harness to score both bundles and record
# the dataset + metric used (e.g., wikitext-2 perplexity or a tiny MMLU subset).
```

## GGUF compatibility matrix

Record known-good loader environments so "loads natively" stays objective.

| Runtime | Version / commit | Load flags | Notes |
| --- | --- | --- | --- |
| llama.cpp | `TODO` | `TODO` | Update with the exact commit hash you validated. |
| Ollama | `TODO` | `TODO` | Include the app version or build number. |
| LM Studio | `TODO` | `TODO` | Note the model format setting if applicable. |

## `t81-qat`

Run quantization-aware training (QAT) on a Hugging Face model with datasets.

```bash
t81-qat gpt2 --dataset-name wikitext --output-dir ternary-gpt2 \
  --per-device-train-batch-size 4 --learning-rate 5e-5 --max-train-samples 1000
```

`--ternary-threshold`, `--ternary-stochastic-rounding`, and `--ternary-warmup-steps` mirror the batch quantization helpers in `t81.trainer`. The CLI expects `datasets` + `transformers` to be installed alongside `torch`; if they are missing the command explains how to satisfy the dependencies (`pip install .[torch]` or `pip install t81lib[torch]`).

## `t81 info`

Inspect a converted directory or GGUF bundle without loading a model.

```bash
t81 info path/to/converted
t81 info model.t81.gguf
```

The command prints the saved threshold/keep-bias metadata for converted checkpoints and echoes the GGUF metadata (architecture, quant type/threshold, tensor count, version) so you can verify a bundle before distribution.

## Troubleshooting

- On macOS/Metal devices with limited GPU memory (e.g., 8 GB M2s) keep conversions on the CPU with `--force-cpu-device-map` or pass `--device-map none`. This avoids accelerated offloading paths that can exceed the working set and trigger Metal kills.
- If you see `NotImplementedError: Cannot copy out of meta tensor` while saving or exporting, rerun `t81 convert`/`t81 gguf` (or the legacy `t81-convert`/`t81-gguf` scripts) with `--force-cpu-device-map` to pin the tensors to host memory before writing them.
- When re-using an existing converted directory, `t81 gguf` (and `t81-gguf`) preserves `t81_metadata.json`, so converting a new HF checkpoint isn’t required if you only need a GGUF bundle.
- Handling multi-gigabyte files: `t81 gguf`, `t81-dequant`, and `t81 convert` (plus the legacy wrappers) now stream metadata/tensors off disk, but you still need to keep every tensor on CPU. Export with `--force-cpu-device-map`, set `ACCELERATE_DISABLE=1`/`HF_ACCELERATE_DISABLE=1`, and optionally configure `MPLCONFIGDIR` + `FONTCONFIG_PATH` cache dirs before running those helpers; see [docs/troubleshooting.md#large-gguf-conversions](../troubleshooting.md#large-gguf-conversions) for the full checklist.

For advanced control (custom thresholds, bias strategies, or dtype hooks) the same APIs are available programmatically via `t81.convert.convert`, `t81.convert.save_pretrained_t81`, and `t81.gguf.write_gguf`.

## Python ↔ CLI crosswalk

- `t81-qat` mirrors `t81.trainer.TernaryTrainer` + `t81.nn.Linear`. The new `examples/ternary_qat_inference_comparison.py` script shows a mini QAT loop, logs the warmup threshold schedule, and exercises the cached ternary GEMM path so you can experiment interactively before hitting the CLI; the CLI now prints a three-stage progress line (`dataset prepared`, `training complete`, `checkpoint saved`) to highlight where dataset prep or saving waits occur.
- `t81 convert` (a drop-in replacement for `t81-convert`) wraps `t81.convert.convert`/`t81.convert.save_pretrained_t81`. Use the same threshold, dtype, bias, and device-map knobs if you need fine-grained control while scripting.
- `t81 gguf` (and its `t81-gguf` alias) delegates to `t81.gguf.write_gguf`; reuse the same quantization/device map knobs when you need GGUF bundles for llama.cpp, Ollama, or LM Studio.

## Serialization hooks

When you persist checkpoints for AI workloads, `t81::core::bigint::to_bytes()`/`from_bytes()` already emit a compact payload (negative flag + limb count header + canonical limb bytes) that plays nicely with binary formats. The new `docs/references/serialization.md` describes FlatBuffers/msgpack/JSON adapters, shows how to treat the byte blob as a `std::span<const std::uint8_t>`, and keeps the `std::hash<t81::core::bigint>` compatibility guarantees intact so you can sanity-check round-trips after deserializing from custom containers.

## Benchmark notes

Before deploying ternary checkpoints, collect a few reference metrics and record them alongside your CLI metadata:

- **Serialization round-trip** — measure `bigint::to_bytes()`/`from_bytes()` latency + blob size so you can catalog how long a FlatBuffers/msgpack/JSON adapter will take to persist or reload a checkpoint.
- **GEMM throughput** — compare `t81lib.gemm_ternary` (or the cached `t81.torch.TernaryTensor` path) against `torch.matmul` on representative layer shapes; note bytes per second and latency per call so you can quote the speedup in your deployment notes.
- **QAT convergence schedule** — log the warmup threshold curve from `t81.trainer.TernaryTrainer` (or scripts like `examples/ternary_qat_inference_comparison.py`) plus loss/accuracy over the first few epochs so you can validate the threshold ramps that `t81-qat` mirrors.

Link these benchmarks back to the CLI workflows (e.g., mention the dataset/model used for `t81-qat` runs and the limb size for GEMM measurements) so AI teams can cite the numbers when proposing ternary deployments.

Consider extending `docs/diagrams/cli-workflows-mermaid.md` (or adding a complementary diagram) that overlays these targets/metrics on the `t81 convert`, `t81 gguf`, and `t81-qat` workflows (the legacy `t81-convert`/`t81-gguf` wrappers remain) so researchers can visualize where to insert serialization and GEMM benchmarks in their pipelines.

These crosswalk notes make it easy to prototype in Python (with the t81.torch/t81.nn stack) and later translate the workflow to the CLI scripts once you validate accuracy/latency trade-offs.

## Regression coverage

- `tests/test_cli_flags.py` now runs `t81 convert` and `t81 gguf` (instead of the legacy wrappers) against a tiny saved Hugging Face classifier, exercising `--device-map none`, `--torch-dtype float16`, and `--force-cpu-device-map` so changes to the conversion/gguf code paths trigger a test failure before the CLI hits CI.
- When you adjust device maps or dtype logic, rerun the new test to make sure the metadata (`t81_metadata.json`) still exists, and the GGUF bundle writes without triggering `NotImplementedError: Cannot copy out of meta tensor`.
