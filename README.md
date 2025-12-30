<!--
README.md — Visitor-facing overview, focused onboarding, and first-steps guidance.
-->

# t81 — Balanced Ternary for AI & Numerics



![CI](https://github.com/t81dev/t81lib/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-blue)
![Docs](https://img.shields.io/badge/docs-via%20GitHub%20Pages-brightgreen)

`t81lib` is a modern, header-first C++20 and Python library that brings balanced ternary arithmetic,
packed ternary GEMMs, Python bindings, and quantization helpers to deterministic numerics and ternary-aware
AI workflows.

**Featured demo** — [Ternary Quantization on Phi-3-mini: PTQ + QAT (8-10x compression, <10% PPL degradation)](examples/ternary_phi3_ptq_qat_demo.ipynb)

## Getting started for Torch users

If you are arriving from PyTorch or Hugging Face, start here and treat `t81` as your single entry point:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ".[torch]"
```

```python
import t81 as t8

tensor = t8.torch.TernaryTensor.from_float(weight, threshold=0.45)
output = tensor.matmul_input(input_tensor, bias=bias)
```

# -> Runs 2-4x faster on CPU than FP16 with packed ternary kernels

Next steps: `t8.nn.Linear` for drop-in layers, `t8.convert`/`t8.gguf` for programmatic conversion, and `t81 convert`/`t81 gguf` for CLI flows.

## Minimum viable success

```cpp
#include <t81/t81lib.hpp>

int main() {
  using t81::Int;
  Int sum = Int::from_int(1) + Int::from_int(2);
  return (sum == Int::from_int(3)) ? 0 : 1;
}
```

```python
import t81lib

print(t81lib.BigInt(3) * t81lib.BigInt(7))
```

## Who is this for?

`t81lib` is for:

- Systems and numerics engineers exploring deterministic or non-binary arithmetic primitives.
- AI practitioners experimenting with ternary quantization, packed GEMMs, and energy-aware inference.
- Researchers evaluating size/latency trade-offs beyond FP16/int8 or whose experiments benefit from exact, tryte-precise math.

It is **not** a drop-in replacement for PyTorch or NumPy, but a focused toolkit for ternary-aware systems.

## Choose your path

- **C++ limb/bigint & numerics** — build locally, include `<t81/t81lib.hpp>`, and verify the `tests/unit/` suite. Starts: [Quick start](#quick-start) & [docs/api-overview.md](docs/api-overview.md).
- **Python quantization & helpers** — `pip install .[torch]` unlocks `t81lib`/`t81`, NumPy wrappers, and `t81.torch`/`t81.nn`. See [docs/python-install.md](docs/python-install.md) & [docs/python-api.md](docs/python-api.md).
- **CLI & GGUF/QAT workflows** — `t81 convert`, `t81 gguf`, and `t81-qat` (the legacy `t81-convert`/`t81-gguf` aliases still work) automate quantize→export→train flows. Follow [docs/references/cli-usage.md](docs/references/cli-usage.md).

## Highlights

- **Balanced-ternary core**: `t81::Int` (an alias for `t81::core::limb`) ships overflow-aware arithmetic, canonical I/O, and deterministic hashing.
- **Ternary-friendly GEMMs**: `t81::linalg::gemm_ternary` packs balanced ternary matrices into AVX/NEON-accelerated kernels with alpha/beta semantics mirrored in the Python binding.
- **Python, CLI, and Torch helpers**: Pybind11 bindings expose quantize/dequantize utilities and `t81.torch`/`t81.nn`, while `t81 convert`, `t81 gguf`, and `t81-qat` (and their legacy `t81-convert`/`t81-gguf` aliases) automate quantize/export/train workflows.
- **Normative docs & demos**: Architecture notes, CLI references, and runnable demos live under `docs/`, `examples/`, and `bench/`.

## Quick start

### 1. Build & test locally

```bash
git clone https://github.com/t81dev/t81lib.git
cmake -S . -B build -DT81LIB_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### 2. Install for Python consumers (optional)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ".[torch]"
```

`pip install` builds the Python bindings, exposes `t81lib`/`t81`, and pulls the optional `torch` helpers when you request `[torch]`. See [docs/python-install.md](docs/python-install.md) for pipx, CLI helpers, and verification scripts.

On macOS or other PEP 668-enforced environments, activate a virtualenv before running `pip install ".[torch]"` (or use `python3 -m pip install --user ".[torch]" --break-system-packages` if you understand the risks) so pip can install the extra dependencies without hitting the “externally managed environment” error.

### 2a. CLI-friendly Pipx install

If you prefer shell-level access to the unified `t81` CLI (with `convert`/`gguf` subcommands) plus `t81-qat` and `t81-dequant`, pipx can install the repo and then inject the torch extras:

```bash
pipx install --python python3 /Users/t81dev/Desktop/t81lib
pipx inject t81lib torch transformers accelerate datasets safetensors
```

Pipx doesn’t understand `.[torch]` when pointing at a local directory, so we first install the package from source and then inject the optional dependencies you need (torch, transformers, accelerate, datasets, safetensors). Once that completes, the CLI helpers will run from `~/.local/bin` with the same requirements as `pip install ".[torch]"`. Continue running `pipx uninstall t81lib` and reinject if you upgrade the repo checkout.

### 3. Consume as a subproject

```cmake
add_subdirectory(external/t81lib)
target_link_libraries(my_app PRIVATE t81::t81lib)
```

```cpp
#include <t81/t81lib.hpp>
```

### 4. Use through vcpkg

```bash
vcpkg install t81lib[tests,benchmarks]:x64-windows
```

```cmake
find_package(t81lib REQUIRED)
target_link_libraries(... t81::t81lib)
```

## Python & PyTorch

`pip install .[torch]` unlocks the `t81lib`/`t81` namespace, NumPy quantization helpers, and the `t81.torch`/`t81.nn` layers that mix ternary weights with FP32/BF16 biases. Jump deeper via [docs/python-api.md](docs/python-api.md), [docs/python-cookbook.md](docs/python-cookbook.md), and [docs/torch.md](docs/torch.md).

## GPU backends

Optional CUDA/ROCm backends can be enabled with `-DUSE_CUDA=ON` / `-DUSE_ROCM=ON` so the Python bindings link against the GPU kernels. `t81lib` exposes a compact `TensorMetadata` ABI that carries device, dtype, shape, and stride info, allowing `where`, `clamp`, `lerp`, and `addcmul` to work directly on NumPy arrays or Torch tensors. See [docs/gpu.md](docs/gpu.md), [docs/torch.md](docs/torch.md), and the [GPU dispatch diagram](docs/diagrams/gpu-dispatch.mermaid.md) for build flags, device routing, supported ops, and lifetime details.

## CLI helpers

`t81 convert`, `t81 gguf`, `t81 info`, and `t81-qat` automate quantize→export→train flows with progress reporting and validation hooks (the legacy `t81-convert`/`t81-gguf` names still work). Note: both `t81 convert` (new) and `t81-convert` (legacy) are available for backward compatibility. Browse [docs/references/cli-usage.md](docs/references/cli-usage.md), [docs/diagrams/cli-workflows-mermaid.md](docs/diagrams/cli-workflows-mermaid.md), and [examples/cli-examples.md](examples/cli-examples.md) for recipes.

### Large models & GGUF streaming

When targeting multi-gigabyte models (Llama 3.x or Gemma 3.x checkpoints) you can still run the CLI helpers without triggering macOS’s OOM killer, but you need to pin everything to CPU memory:

- Set `ACCELERATE_DISABLE=1` (and `HF_ACCELERATE_DISABLE=1` when you launch a `transformers` command) so Accelerate never offloads tensors to `meta`/disk and the helpers can call `.to("cpu")`.
- Prefer `--force-cpu-device-map` or `--device-map none/cpu` so `t81 convert`/`t81 gguf` (and the legacy `t81-convert`/`t81-gguf` wrappers) keep checkpoint shards on host RAM.
- The Python GGUF reader now streams metadata/tensor infos directly from a file handle, seeks to each tensor block, and only buffers one tensor at a time, so `t81 gguf`/`t81-dequant` (and `t81-gguf`/`t81-dequant` scripts for compatibility) can handle the resulting bundles without reading the entire file into memory.

If you still see `NotImplementedError: Cannot copy out of meta tensor` or a kernel that dies while building Matplotlib’s font cache, repeat the cache setup from [docs/troubleshooting.md](docs/troubleshooting.md#large-gguf-conversions) (`MPLCONFIGDIR`, `FONTCONFIG_PATH`, etc.) before rerunning the CLI.

### Dequantizing for downstream runtimes

Use the new `t81-dequant` helper (backed by `t81.dequantize_gguf_to_float`) to rewrite a TQ1_0 or TQ2_0 bundle into float32 before handing it to stock llama.cpp, Ollama, or LM Studio builds that lack ternary support:

```bash
t81-dequant model-tq1.gguf model-compatible-f16.gguf
```

That command rewrites the tensors in place while preserving the standard GGUF metadata so the resulting file works with existing loaders. Keep the original `model-tq1.gguf`/`model-tq2.gguf` around for runtimes that already understand TQ tensors, and only run `t81-dequant` when you need immediate compatibility.

For a zero-disk workaround you can also dequantize on the fly (via `t81.dequantize_gguf_to_float` or a small loader patch) before instantiating `llama_cpp.Llama`; see the docs for an example monkey patch if you want to load `model-tq1.gguf` or `model-tq2.gguf` directly without producing an intermediate copy.


## Use cases

- Ternary LLM weight quantization and GGUF exports for Hugging Face + `llama.cpp`.
- Packed ternary GEMMs for CPU inference and `torch.matmul` comparisons.
- Deterministic numerics and research-grade arithmetic built atop the same core.
- Ternary hardware simulation and energy modeling (see [docs/hardware.md](docs/hardware.md)).

See [Use Cases & Demos](docs/use-cases.md) for real-world workflows, including the Phi-3-mini ternary notebook.

## Examples

See [examples/README.md](examples/README.md) for the canonical scripts/notebooks (LLM conversion, GEMM packing, quantization demos, hardware previews, etc.).

## Why ternary?

- Balanced ternary offers ≈1.58 bits of entropy per digit with symmetric overflow semantics and deterministic hashing for `t81::Int`.
- Balanced ternary trades representational symmetry for determinism and composability, making it easy to reason about ternary algebra while keeping model state portable.
- Refer to [BENCHMARKS.md](BENCHMARKS.md) for the 15–22× storage/latency wins versus FP16 and [docs/hardware.md](docs/hardware.md) for AVX/NEON/AVX-512 kernel sketches.

## Numeric building blocks

`t81lib` exposes fixed-width ternary integers, big integers, rationals, floats, vectors, and Montgomery helpers that share a single canonical core.
See [docs/api-overview.md](docs/api-overview.md) for the full surface described in the umbrella header.

## Stability & compatibility

- Supported toolchains: C++20-capable Clang/GCC/MSVC (or `pip install`’s compatible CPython builds) with CMake ≥ 3.22; the build auto-detects AVX2/AVX-512/NEON and falls back to portable kernels when those SIMD targets are unavailable.
- We track the ABI/API surface via `include/t81/t81lib.hpp`; expect the core headers to evolve until we reach a stable v1 release and consult [CHANGELOG.md](CHANGELOG.md) for migration notes.

## Docs & resources

### Getting started

- [docs/index.md](docs/index.md) — Docs portal (great for GitHub Pages).
- [docs/python-install.md](docs/python-install.md) — Install python bindings, pipx, and CLI helpers.
- [docs/python-api.md](docs/python-api.md) & [docs/python-cookbook.md](docs/python-cookbook.md) — Python recipes.
- [docs/torch.md](docs/torch.md) — PyTorch integration, `t81.torch`, and `t81.nn`.
- [docs/references/cli-usage.md](docs/references/cli-usage.md) — CLI workflows for convert/gguf/qat.

### Specs & design

- [docs/t81lib-spec-v1.0.0.md](docs/t81lib-spec-v1.0.0.md) — Normative contract.
- [docs/design/](docs/design/) — Deep dives on `limb`, `bigint`, and `montgomery`.
- [docs/api-overview.md](docs/api-overview.md) — Umbrella helper catalog.
- [ARCHITECTURE.md](ARCHITECTURE.md) & [docs/index.md](docs/index.md) — Architecture stories and portal.

### Examples & testing

- [examples/README.md](examples/README.md) — Canonical scripts and notebooks.
- [tests/](tests/) & [bench/](bench/) — Regression suites and throughput gauges.
- [BENCHMARKS.md](BENCHMARKS.md) — Fashion-MNIST FP32/PTQ/QAT comparison.
- [CONTRIBUTING.md](CONTRIBUTING.md) & [CHANGELOG.md](CHANGELOG.md) — Contribution guidance and history.

## Benchmarks

See [BENCHMARKS.md](BENCHMARKS.md) for the Fashion-MNIST FP32/PTQ/QAT comparison.
Phi-3 GGUF baseline (TQ1_0, CPU-only, llama.cpp): size 1481.96 MiB, peak RSS 2260.02 MiB, prompt 54.35 ms/token (18.4 tok/s), eval 56.22 ms/token (17.79 tok/s).

```bash
cmake -S . -B build -DT81LIB_BUILD_BENCH=ON -DT81LIB_USE_GOOGLE_BENCH=ON
cmake --build build -j
./build/bench/*
```

## Contribution & support

- Open an issue to discuss architecture or API needs.
- Follow [CONTRIBUTING.md](CONTRIBUTING.md) for branching, formatting, and testing workflows.
- Reference [docs/design/](docs/design/) when proposing optimizations to ensure semantic safety.
- Track progress in [CHANGELOG.md](CHANGELOG.md).
- Security concerns can be reported privately; the repo currently follows a standard disclosure cadence.

## License

`t81lib` is MIT-licensed; see [LICENSE](LICENSE).
