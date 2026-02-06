# Use cases & demos

`t81lib` already powers ternary-aware workflows that explore performance, energy, and accuracy trade-offs beyond conventional FP16/int8 tooling:

### Extreme compression for on-device inference

Ternary PTQ/QAT on Phi-3-mini-4k-instruct: measure size (FP16 -> ternary), latency, and perplexity in a single workflow.
Start here: [`examples/ternary_phi3_ptq_qat_demo.ipynb`](../examples/ternary_phi3_ptq_qat_demo.ipynb).

1. Convert existing large language models to ternary weights while keeping `torch.nn.Module` semantics (`examples/demo_llama_conversion.py`).
2. Study ternary scaling laws for RMSNorm, RoPE, and ternary softmax to compare precision versus throughput (`examples/scaling_laws_ternary.py`).
3. Preview ternary-sparse transformers with custom `SparseTriangular` layers and quantized attention (`examples/ternary_sparse_preview.py`).
4. Enumerate synthetic quantization configurations (dim/threshold/samples) to log accuracy, latency, and storage before touching real checkpoints (`examples/quantization_config_report.py`).

## Python demos & notebooks

These scripts and notebooks mirror the CLI workflows while keeping you inside Python so you can iterate on thresholds, datasets, or schedulers:

- `examples/demo_llama_conversion.py`: convert a Hugging Face Llama checkpoint, swap `torch.nn.Linear` → `t81.nn.Linear`, and inspect the ternary cache.
- `examples/scaling_laws_ternary.py`: compare ternary and float scaling laws across precision, sparsity, and throughput axes.
- `examples/ternary_sparse_preview.py`, `examples/ternary_quantization_demo.ipynb`, and `examples/ternary_transformer_demo.ipynb`: explore hybrid sparsity, GEMM packing, and quantized transformer inference with notebook support.
- `examples/ternary_phi3_ptq_qat_demo.ipynb`: PTQ/QAT on Phi-3-mini with size, latency, and perplexity tracking in one compact notebook.
- `examples/ternary_qat_inference_comparison.py`: run a miniature QAT loop, dump the ternary threshold schedule, and mask latency between `torch.matmul` and the cached `t81.torch.TernaryTensor`/`t81lib.gemm_ternary` path.
- `examples/ternary_mnist_demo.ipynb`: quantize a compact MNIST classifier, pack weight buffers with `t81lib.pack_dense_matrix`, and route inference through `t81lib.gemm_ternary` to compare accuracy, latency, and memory versus float32 and 1-bit baselines.

## Additional references

Mentioned demos also appear in `docs/index.md` and `docs/references/cli-usage.md` so you can toggle between CLI helpers and Python stories without guessing. The [quantization workflow diagram](diagrams/quantization-workflow.mermaid.md) ties the PyTorch conversion path to the CLI export/inference steps above.
