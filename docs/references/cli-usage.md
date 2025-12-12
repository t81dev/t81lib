<!--
docs/references/cli-usage.md — Usage notes for the console scripts.
-->

# CLI helpers

The `t81lib` package exposes three console scripts once you install the `torch`/`transformers` extras:

```bash
pipx install .[torch]  # or pip install t81lib[torch] if you prefer a shared venv
```

After that `t81-convert`, `t81-gguf`, and `t81-qat` land in `~/.local/bin` (or pipx’s `bin` directory) so you can run the tooling without dropping into Python.

## `t81-convert`

Convert any Hugging Face checkpoint into the ternary-aware runtime that powers `t81.torch`/`t81.nn`.

```bash
t81-convert meta-llama/Llama-3.2-3B-Instruct path/to/converted --quant TQ1_0 --torch-dtype bfloat16
```

Key flags:

- `--threshold`: the ternary cutoff (default `0.45`). Adjust this to trade sparsity vs. activation range.
- `--device-map`: forwards a `transformers` device map string; use `auto` (default) for Accelerate offloading or `none`/`cpu` to keep everything on host RAM.
- `--force-cpu-device-map`: overrides `transformers` so the conversion stays on CPU, which prevents `NotImplementedError: Cannot copy out of meta tensor` when saving and guarantees the `t81_metadata.json` footprint is serializable.
- `--keep-biases-bf16` / `--no-keep-biases-bf16`: control whether bias tensors stay in BF16 or are promoted to FP32.
- `--torch-dtype`: optional dtype for the floating-point buffer used during quantization.
- `--output-gguf`: pipe the converted model straight into `gguf.write_gguf` (see `t81-gguf` below) with a `--gguf-quant` choice of `TQ1_0` or `TQ2_0`.

The command rewrites every `nn.Linear` into `t81.nn.Linear`, stores metadata in `t81_metadata.json`, and emits compression stats so you can see how much VRAM you save.

## `t81-gguf`

Write a ternary GGUF bundle that works with `llama.cpp`, Ollama, or LM Studio without rerunning the conversion step.

```bash
t81-gguf out.3.t81.gguf --from-hf meta-llama/Llama-3.2-3B-Instruct --quant TQ2_0
```

Alternately, re-export an existing t81-converted directory:

```bash
t81-gguf out.3.t81.gguf --from-t81 path/to/converted
```

Use the same `--threshold`, `--device-map`, `--torch-dtype`, and `--force-cpu-device-map` knobs as `t81-convert` because `t81-gguf` delegates to that CLI internally.

## `t81-qat`

Run quantization-aware training (QAT) on a Hugging Face model with datasets.

```bash
t81-qat gpt2 --dataset-name wikitext --output-dir ternary-gpt2 \
  --per-device-train-batch-size 4 --learning-rate 5e-5 --max-train-samples 1000
```

`--ternary-threshold`, `--ternary-stochastic-rounding`, and `--ternary-warmup-steps` mirror the batch quantization helpers in `t81.trainer`. The CLI expects `datasets` + `transformers` to be installed alongside `torch`; if they are missing the command explains how to satisfy the dependencies (`pip install .[torch]` or `pip install t81lib[torch]`).

## Troubleshooting

- On macOS/Metal devices with limited GPU memory (e.g., 8 GB M2s) keep conversions on the CPU with `--force-cpu-device-map` or pass `--device-map none`. This avoids accelerated offloading paths that can exceed the working set and trigger Metal kills.
- If you see `NotImplementedError: Cannot copy out of meta tensor` while saving or exporting, rerun `t81-convert`/`t81-gguf` with `--force-cpu-device-map` to pin the tensors to host memory before writing them.
- When re-using an existing converted directory, `t81-gguf` preserves `t81_metadata.json`, so converting a new HF checkpoint isn’t required if you only need a GGUF bundle.

For advanced control (custom thresholds, bias strategies, or dtype hooks) the same APIs are available programmatically via `t81.convert.convert`, `t81.convert.save_pretrained_t81`, and `t81.gguf.write_gguf`.

## Python ↔ CLI crosswalk

- `t81-qat` mirrors `t81.trainer.TernaryTrainer` + `t81.nn.Linear`. The new `examples/ternary_qat_inference_comparison.py` script shows a mini QAT loop, logs the warmup threshold schedule, and exercises the cached ternary GEMM path so you can experiment interactively before hitting the CLI.
- `t81-convert` wraps `t81.convert.convert`/`t81.convert.save_pretrained_t81`. Use the same threshold, dtype, bias, and device-map knobs if you need fine-grained control while scripting.
- `t81-gguf` delegates to `t81.gguf.write_gguf`; reuse the same quantization/device map knobs when you need GGUF bundles for llama.cpp, Ollama, or LM Studio.

## Serialization hooks

When you persist checkpoints for AI workloads, `t81::core::bigint::to_bytes()`/`from_bytes()` already emit a compact payload (negative flag + limb count header + canonical limb bytes) that plays nicely with binary formats. The new `docs/references/serialization.md` describes FlatBuffers/msgpack/JSON adapters, shows how to treat the byte blob as a `std::span<const std::uint8_t>`, and keeps the `std::hash<t81::core::bigint>` compatibility guarantees intact so you can sanity-check round-trips after deserializing from custom containers.

## Benchmark notes

Before deploying ternary checkpoints, collect a few reference metrics and record them alongside your CLI metadata:

- **Serialization round-trip** — measure `bigint::to_bytes()`/`from_bytes()` latency + blob size so you can catalog how long a FlatBuffers/msgpack/JSON adapter will take to persist or reload a checkpoint.
- **GEMM throughput** — compare `t81lib.gemm_ternary` (or the cached `t81.torch.TernaryTensor` path) against `torch.matmul` on representative layer shapes; note bytes per second and latency per call so you can quote the speedup in your deployment notes.
- **QAT convergence schedule** — log the warmup threshold curve from `t81.trainer.TernaryTrainer` (or scripts like `examples/ternary_qat_inference_comparison.py`) plus loss/accuracy over the first few epochs so you can validate the threshold ramps that `t81-qat` mirrors.

Link these benchmarks back to the CLI workflows (e.g., mention the dataset/model used for `t81-qat` runs and the limb size for GEMM measurements) so AI teams can cite the numbers when proposing ternary deployments.

Consider extending `docs/diagrams/cli-workflows-mermaid.md` (or adding a complementary diagram) that overlays these targets/metrics on the `t81-convert`, `t81-gguf`, and `t81-qat` workflows so researchers can visualize where to insert serialization and GEMM benchmarks in their pipelines.

These crosswalk notes make it easy to prototype in Python (with the t81.torch/t81.nn stack) and later translate the workflow to the CLI scripts once you validate accuracy/latency trade-offs.
