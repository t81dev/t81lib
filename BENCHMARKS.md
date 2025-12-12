#Quantization Benchmark Suite

This repository now provides a reproducible benchmark that compares FP32,
    post - training ternary quantization(PTQ), and quantization - aware training(QAT) through a small Fashion-MNIST classifier. The script is located at `scripts/ternary_quantization_benchmark.py` and is designed to log accuracy, latency, and storage so you can understand the benefits of moving from float32 weights to ternary-trained representations.

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

## Diagrams

View the [benchmark comparison diagram](docs/diagrams/benchmarks.mermaid.md) for a quick latency/storage summary that highlights the 15–22× wins.

## Results sharing

When opening a pull request, add the latest benchmark rows (or a summary table) to this file or reference the CSV as part of your performance discussion so reviewers can reproduce the numbers.
