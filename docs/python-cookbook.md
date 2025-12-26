# Python Cookbook

These recipes show practical ways to combine the `t81lib` bindings, the `t81.torch` helpers, and the CLI utilities so you can prototype, test, and ship balanced ternary quantization pipelines from Python.

## Python entry points

| Entry point | Use it when you want to... | Start here |
| --- | --- | --- |
| `t81lib` | access the low-level pybind11 bindings (BigInt, Limb, packing, GGUF helpers) | `import t81lib` |
| `t81` | use the high-level re-exported API without worrying about missing bindings | `import t81` |
| `t81.torch` | run ternary tensor ops or integrate with Torch tensors | `from t81 import torch as t81torch` |
| `t81.nn` | swap in ternary-aware layers (e.g., `Linear`) | `from t81 import nn as t81nn` |
| `t81.convert` / `t81.gguf` | call the conversion/GGUF helpers programmatically | `from t81 import convert, gguf` |
| `t81.hardware` | explore ternary hardware emulation helpers | `from t81 import hardware` |

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

### Keep packed buffers on the GPU

When CUDA/ROCm support is enabled, `t81lib.gemm_ternary` accepts GPU-backed metadata directly. The dispatcher expects A/B to describe `ScalarType::TernaryLimb` rows with `TRYTES_PER_LIMB` packed trytes (e.g., NumPy `dtype('V16')` or `torch.uint8` views shaped `(M, k_limbs, 16)` after calling `torch.from_numpy(packed).reshape(...)`). The accumulator `C` stays float32 contiguous, and with `Backend::Auto` the binding routes the work to the compiled GPU kernel if the necessary backend is available.

Use `t81.torch.TernaryTensor` to keep limbs on the GPU and let the binding generate the required metadata, or copy a packed NumPy buffer onto CUDA/ROCm if you need to interface with other tooling. Because the binding now shares the same `TensorMetadata` flow as `t81lib.where`/`clamp`/`lerp`, no extra copying or manual span conversions are required when the inputs already live on a compatible device.

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
