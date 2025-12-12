# Examples

This file is the canonical list of runnable scripts and notebooks maintained under `examples/`.

## PyTorch + quantization demos

- `examples/demo_llama_conversion.py` — Convert a Hugging Face Llama checkpoint, swap every `torch.nn.Linear` for `t81.nn.Linear`, and inspect the cached ternary weights.
- `examples/scaling_laws_ternary.py` — Compare ternary vs float scaling laws across RMSNorm, RoPE, and throughput axes.
- `examples/ternary_sparse_preview.py` — Explore hybrid sparsity, GEMM packing, and quantized transformer inference with notebook-friendly visuals.
- `examples/ternary_quantization_demo.ipynb` — Tutorial notebook showing packed GEMMs, quantized trits, and dequantization.
- `examples/ternary_transformer_demo.ipynb` — Micro GPT stack with cached ternary projections and packed GEMM profiling.
- `examples/ternary_mnist_demo.ipynb` — Quantize an MNIST classifier, pack weight buffers, and route inference through `t81lib.gemm_ternary`.
- `examples/ternary_qat_inference_comparison.py` — Run a miniature QAT loop, log ternary threshold schedules, and compare latency between `torch.matmul` and cached `TernaryTensor`.

## Hardware & CLI demos

- `examples/ternary_hardware_sim_demo.ipynb` — Build a ternary adder, trace virtual power/latency, and compare energy vs binary hardware using `t81.hardware.TernaryEmulator`.
- `examples/cli-examples.md` — Copy/paste-ready snippets for `t81-convert`, `t81-gguf`, and `t81-qat` workflows.

Refer to [docs/use-cases.md](docs/use-cases.md) for details on how these examples tie into broader quantization, scaling-law, and hardware experiments.
