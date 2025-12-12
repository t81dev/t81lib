<!--
examples/cli-examples.md — Ready-to-run CLI snippets showing the `t81-*` helpers.
-->

# CLI examples

These annotated snippets show how to run the three console scripts once you have
installed `t81lib[torch]` (or `. [torch]`) so the `t81-convert`, `t81-gguf`, and
`t81-qat` entry points land in your `PATH`.

## Prerequisites

```bash
pipx install .[torch]
```

If you prefer a shared environment:

```bash
pip install t81lib[torch]
```

`pipx` installs the scripts under `~/.local/bin` (or pipx’s `bin` directory); make
sure that directory is on your shell `PATH`.

## Convert a checkpoint

```bash
t81-convert meta-llama/Llama-3.2-3B-Instruct \
  /tmp/converted-llama3.2-3b --quant TQ1_0 \
  --torch-dtype bfloat16 --force-cpu-device-map
```

- `--quant TQ1_0` forces the ternary quantization lookup we ship with the
  runtimes.
- `--torch-dtype bfloat16` keeps FP16 buffers while quantizing.
- `--force-cpu-device-map` ensures Accelerate pins tensors to CPU so the new
  `t81_metadata.json` stays serializable.
- The command rewrites every `nn.Linear` to `t81.nn.Linear` before producing the
  converted directory with ternary tensors, `t81_metadata.json`, and the stats
  log.

To emit a GGUF bundle in the same run, add `--output-gguf --gguf-quant TQ1_0`
while pointing `--output-dir` (or the last positional argument) at where you
want the converted directory to stay.

## Generate a GGUF bundle

```bash
t81-gguf \
  converted-llama3.2-3b/out.3.t81.gguf \
  --from-t81 /tmp/converted-llama3.2-3b \
  --quant TQ2_0 --device-map none --force-cpu-device-map
```

- `--from-t81` reuses the metadata + helpers already stored in the converted
  directory so you can skip re-converting the HF checkpoint.
- `--device-map none` keeps everything on the host, and `--force-cpu-device-map`
  makes the CLI idempotent on macOS/Metal devices with limited RAM.
- The resulting `out.3.t81.gguf` bundle works with `llama.cpp`, Ollama, or LM
  Studio.

If you need to re-run a conversion from scratch before exporting, swap
`--from-t81` for `--from-hf meta-llama/...` and reuse the same threshold, dtype,
and force-cpu knobs noted above.

## Quantization-aware training (QAT)

```bash
t81-qat gpt2 \
  --dataset-name wikitext \
  --output-dir ~/ternary-gpt2 \
  --per-device-train-batch-size 4 \
  --learning-rate 5e-5 \
  --max-train-samples 1000 \
  --ternary-threshold 0.45 \
  --ternary-stochastic-rounding \
  --ternary-warmup-steps 500
```

- The CLI mirrors the `t81.trainer` helpers, so you can sweep the ternary
  threshold, stochastic rounding, and warmup steps just like in the Python API.
- Install `datasets` + `transformers` alongside `torch`; `t81-qat` shows the
  missing-dependency message when those extras are unavailable.
- Training snapshots and logs land under `~/ternary-gpt2` so you can later convert
  them with `t81-convert`.
