# PyTorch integration

The Python bindings expose `t81.torch`, `t81.nn`, and the `t81.trit` dtype so you can quantize/execute ternary GEMMs directly inside PyTorch code.

## `t81.torch`

- `TernaryTensor.from_float(...)` quantizes a float tensor into balanced ternary weights and caches the packed limbs.
- `TernaryTensor.matmul_input(...)` generates packed GEMMs by driving `t81lib.gemm_ternary` on AVX/NEON hosts while keeping accumulators in FP32/BF16.
- `t81.trit` keeps the canonical ternary dtype visible to Torch modules.

Reuse `t81lib.pack_dense_matrix`, `t81lib.unpack_packed_limbs`, and `t81lib.dequantize_trits` for preprocessing, debugging, or bridging to NumPy.

## `t81.nn`

This module keeps biases in FP32/BF16 while quantizing weights lazily, so you can swap in `t81.nn.Linear` for `torch.nn.Linear` and still rely on `torch.compile()`/FSDP. Use `model.to(dtype=t81.trit)` when you want the entire network to follow the ternary path that shares the same cached `TernaryTensor` / `gemm_ternary` backend.

## Conversion helpers

- `scripts/convert_to_ternary.py` walks a checkpoint tree, swaps every `torch.nn.Linear` for `t81.nn.Linear`, reports size reductions, and can force CPU device maps for reliable saves.
- `t81-convert`, `t81-gguf`, and `examples/cli-examples.md` provide one-click scripts that mirror the Python flows while also exporting GGUF bundles for llama.cpp and Hugging Face runtimes.

See [examples/README.md](../examples/README.md) and [use-cases.md](use-cases.md) for runnable demos, scaling-law studies, and QAT comparisons.
