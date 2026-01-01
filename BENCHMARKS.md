# Quantization Benchmark Suite

This repository now provides a reproducible benchmark that compares FP32,
post-training ternary quantization (PTQ), and quantization-aware training (QAT)
through a small Fashion-MNIST classifier. The script is located at
`scripts/ternary_quantization_benchmark.py` and is designed to log accuracy,
latency, and storage so you can understand the benefits of moving from float32
weights to ternary-trained representations.

## Benchmark matrix

Use these baseline runs to keep results comparable across machines. The “Expected output” snippets show the fields or files to look for, not exact numbers.

### 1) Fashion-MNIST FP32/PTQ/QAT

```bash
python scripts/ternary_quantization_benchmark.py \
  --data-dir ~/data/fashion-mnist \
  --batch-size 128 \
  --fp32-epochs 3 \
  --qat-epochs 3 \
  --threshold 0.45 \
  --device cpu \
  --output benchmarks/fashion_mnist_quantization.csv
```

Expected output:
- `benchmarks/fashion_mnist_quantization.csv` with `mode`, `accuracy`, `loss`, `latency_s`, `bytes`.

### 2) GPT-2 PTQ/QAT micro-run (sanity check)

```bash
python scripts/phi3_ptq_qat_benchmark.py \
  --model-id gpt2 \
  --device cpu \
  --dtype float32 \
  --max-eval-tokens 32 \
  --eval-texts 8 \
  --max-new-tokens 4 \
  --run-qat \
  --qat-steps 5 \
  --train-split 'train[:20]'
```

Expected output:
- Console summary with size, compression ratio, perplexity, and tok/s for baseline/PTQ/QAT.

### 3) ViT CIFAR-10 PTQ/QAT baseline (quick)

```bash
python scripts/vit_ptq_qat_benchmark.py \
  --model-id google/vit-base-patch16-224 \
  --device cpu \
  --threshold 0.45 \
  --batch-size 32 \
  --max-train-samples 2048 \
  --max-eval-samples 512 \
  --eval-batches 16 \
  --run-qat \
  --qat-steps 50 \
  --json-output benchmarks/vit_cifar10_baseline.json
```

Expected output:
- Console summary with size, accuracy/loss, and images/s for baseline/PTQ/QAT.
- `benchmarks/vit_cifar10_baseline.json` with stage metrics and model metadata.

### Fast-mode recipes (quick baselines)

Use these when you want a low-latency run to confirm the pipeline without
waiting for full PTQ/QAT loops.

ViT size + accuracy baseline (skip throughput, minimal eval):

```bash
python scripts/vit_ptq_qat_benchmark.py \
  --model-id google/vit-base-patch16-224 \
  --device cpu \
  --threshold 0.45 \
  --batch-size 16 \
  --max-train-samples 256 \
  --max-eval-samples 128 \
  --eval-batches 1 \
  --max-eval-batches 1 \
  --skip-throughput \
  --json-output benchmarks/vit_cifar10_quick.json
```

Observed output (CPU, size-only run with `--max-eval-batches 0` + `--skip-throughput`):
- baseline size: 0.32 GiB
- PTQ size: 0.03 GiB
- accuracy/loss/images_per_s: 0.0 (skipped)

Phi-3 baseline PPL only (skip latency + PTQ PPL/QAT):

```bash
python scripts/phi3_ptq_qat_benchmark.py \
  --model-id microsoft/Phi-3-mini-4k-instruct \
  --device cpu \
  --dtype float32 \
  --max-eval-tokens 512 \
  --eval-texts 16 \
  --max-new-tokens 16 \
  --skip-latency \
  --skip-ptq-ppl \
  --json-output benchmarks/phi3_baseline_ppl.json
```

Status: PTQ PPL + short QAT pending (CPU-only PTQ conversion exceeded 2h locally). Resume on GPU:

```bash
python scripts/phi3_ptq_qat_benchmark.py \
  --model-id microsoft/Phi-3-mini-4k-instruct \
  --device auto \
  --dtype bfloat16 \
  --threshold 0.45 \
  --max-eval-tokens 128 \
  --eval-texts 2 \
  --max-new-tokens 0 \
  --skip-latency \
  --run-qat \
  --qat-steps 5 \
  --train-split 'train[:10]' \
  --json-output benchmarks/phi3_ptq_qat_fast.json
```

Note: PTQ still runs on CPU (t81.torch fallback), so keep enough host RAM available.

### 4) GGUF export + load check

```bash
t81 convert microsoft/Phi-3-mini-4k-instruct phi3-t81 --threshold 0.45 --force-cpu-device-map
t81 gguf phi3-tq1.gguf --from-t81 phi3-t81 --quant TQ1_0 --validate
python scripts/gguf_benchmark.py --gguf phi3-tq1.gguf --llama-cli /path/to/llama-cli --n-predict 128
```

Expected output:
- `t81 gguf` prints validation success.
- `gguf_benchmark.py` prints size, peak RSS, and eval ms/token.

Phi-3 GGUF baseline (TQ1_0, CPU-only, llama.cpp build 7340):

```bash
python scripts/gguf_benchmark.py \
  --gguf phi3-tq1-fixed12.gguf \
  --llama-cli /opt/homebrew/bin/llama-cli \
  --n-predict 64 --extra --device none --n-gpu-layers 0
```

Observed output:
- size: 1481.96 MiB
- peak RSS: 2260.02 MiB
- prompt: 54.35 ms/token (18.4 tok/s)
- eval: 56.22 ms/token (17.79 tok/s)

### 5) GEMM throughput (CPU)

```bash
python - <<'PY'
import numpy as np
import t81lib

m, n, k = 1024, 1024, 1024
weights = np.random.randn(m, k).astype(np.float32)
packed = t81lib.pack_dense_matrix(weights, threshold=0.45)
rhs = np.random.randn(k, n).astype(np.float32)
out = np.zeros((m, n), dtype=np.float32)
t81lib.gemm_ternary(packed, packed, out, m, n, k)
print("gemm_ternary OK", out.shape)
PY
```

Expected output:
- Console prints `gemm_ternary OK (1024, 1024)`.

## Script overview

1. The benchmark builds a `TinyClassifier` (a single `nn.Linear` head on flattened Fashion-MNIST images) and trains it in FP32 for a few epochs.
2. It then quantizes the trained weights with `t81lib.quantize_to_trits`, measures accuracy/latency over the validation split, and estimates ternary storage.
3. Finally it runs `TernaryTrainer` on the same dataset to perform QAT, evaluates the resulting ternary model, and writes all three modes into a CSV row (default `benchmarks/quantization_runs.csv`).

## Running the benchmark

Install the extras that provide the training stack:

```bash
pip install -e .[torch]
```

Then, execute the benchmark:

```bash
python scripts/ternary_quantization_benchmark.py \
  --data-dir ~/data/fashion-mnist \
  --batch-size 128 \
  --fp32-epochs 3 \
  --qat-epochs 3 \
  --threshold 0.45 \
  --device cuda \
  --output benchmarks/fashion_mnist_quantization.csv
```

The script downloads Fashion-MNIST, trains the FP32 model, applies PTQ, and runs the QAT loop. After each mode it prints accuracy, latency, and storage figures, then appends them to the specified CSV file for later comparison.

## Interpreting the CSV

Each row contains:

* `mode` – one of `fp32`, `ptq`, or `qat`.
* `accuracy`, `loss` – validation metrics from the benchmark.
* `latency_s` – average per-batch inference latency (seconds).
* `bytes` – estimated storage for the weight tensors (ternary storage for PTQ/QAT, raw float bytes for FP32).

Use this CSV to plot accuracy vs. storage or compare latency across the three modes.

## JSON artifact schema (ViT + Phi-3)

The ViT and Phi-3 scripts emit JSON when you pass `--json-output`. These files
are intended to be committed alongside baseline numbers.

ViT JSON keys (from `scripts/vit_ptq_qat_benchmark.py`):

```json
{
  "model_id": "google/vit-base-patch16-224",
  "dataset": "cifar10",
  "device": "cpu",
  "threshold": 0.45,
  "baseline": {"size_gib": 0.00, "accuracy": 0.0, "loss": 0.0, "images_per_s": 0.0},
  "ptq": {"size_gib": 0.00, "accuracy": 0.0, "loss": 0.0, "images_per_s": 0.0},
  "qat": null
}
```

Phi-3 JSON keys (from `scripts/phi3_ptq_qat_benchmark.py`):

```json
{
  "model_id": "microsoft/Phi-3-mini-4k-instruct",
  "dataset": "wikitext-2-raw-v1",
  "device": "cpu",
  "dtype": "float32",
  "threshold": 0.45,
  "max_eval_tokens": 1024,
  "eval_texts": 32,
  "max_new_tokens": 64,
  "skip_latency": true,
  "skip_ptq_ppl": false,
  "run_qat": false,
  "qat_steps": 5,
  "train_split": "train[:1%]",
  "learning_rate": 5e-5,
  "compression_ratio": 0.0,
  "baseline": {"size_gib": 0.00, "ppl": 0.0, "tok_s": 0.0},
  "ptq": {"size_gib": 0.00, "ppl": null, "tok_s": 0.0},
  "qat": null
}
```

## Diagrams

View the [benchmark comparison diagram](docs/diagrams/benchmarks.mermaid.md) for a quick latency/storage summary that highlights the 15–22× wins.

## Results sharing

When opening a pull request, add the latest benchmark rows (or a summary table) to this file or reference the CSV as part of your performance discussion so reviewers can reproduce the numbers.
