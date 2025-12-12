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
