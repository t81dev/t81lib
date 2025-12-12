# Python Cookbook

These recipes show practical ways to combine the `t81lib` bindings, the `t81.torch` helpers, and the CLI utilities so you can prototype, test, and ship balanced ternary quantization pipelines from Python.

## 1. Quantize a dense weight matrix and run `gemm_ternary`

```python
import numpy as np
import t81lib

weights = np.random.randn(16, 48).astype(np.float32)
packed = t81lib.pack_dense_matrix(weights, threshold=0.45)
# allocate an accumulator for the result
c = np.zeros((16, 16), dtype=np.float32)
t81lib.gemm_ternary(packed, packed, c, 16, 16, 48)
```

This shows how to drive the low-level binding (`t81lib.pack_dense_matrix`) together with `gemm_ternary` without needing PyTorch.

## 2. Drop in `t81.torch.TernaryTensor` during training

```python
import torch
import t81

tensor = t81.torch.TernaryTensor.from_float(
    torch.nn.functional.linear(weight, bias=None),
    threshold=0.5,
)
output = t81.torch.TernaryTensor.matmul_input(tensor, input_tensor)
```

Use the `t81.torch` bridge when you want packed ternary GEMMs to interoperate with `torch.matmul` or `torch.nn.Linear`. The helper keeps the accumulator in float32/BF16 while the weights live in a packed ternary buffer.

## 3. Combine Python preprocessing with CLI automation

Run the CLI helpers after preparing a model in Python to ensure quantized checkpoints stay compatible.

```bash
PYTHONPATH=build-python python examples/ternary_qat_inference_comparison.py

# Use CLI to convert the trained checkpoint to GGUF
t81-convert --input model.pt --output model.t81 --threshold 0.47
# Validate and export a gguf bundle
t81-gguf --input model.t81 --validate
```

This recipe shows how Python experiments (scripts, notebooks) complement the CLI docs in `docs/references/cli-usage.md`.
