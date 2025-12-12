<!--
README.md — Visitor-facing overview, focused onboarding, and first-steps guidance.
-->

# t81lib — Balanced ternary quantization & core numerics

![CI](https://github.com/t81dev/t81lib/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-blue)
![Docs](https://img.shields.io/badge/docs-via%20GitHub%20Pages-brightgreen)
![≈1.58-bit but balanced ternary](https://img.shields.io/badge/%E2%89%A51.58-bit%20but%20balanced%20ternary-purple)
![15–22× smaller than FP16](https://img.shields.io/badge/15%E2%80%9322%C3%97%20smaller%20than%20FP16-orange)
![AVX-512 ternary GEMM](https://img.shields.io/badge/AVX-512%20ternary%20GEMM-lightgrey)

`t81lib` is a modern, header-first C++20 and Python library that brings balanced ternary arithmetic,
packed ternary GEMMs, Python bindings, and quantization helpers to deterministic numerics and ternary-aware
AI workflows.

## Who is this for?

`t81lib` is for:

- Systems and numerics engineers exploring deterministic or non-binary arithmetic primitives.
- AI practitioners experimenting with ternary quantization, packed GEMMs, and energy-aware inference.
- Researchers evaluating size/latency trade-offs beyond FP16/int8 or whose experiments benefit from exact, tryte-precise math.

It is **not** a drop-in replacement for PyTorch or NumPy, but a focused toolkit for ternary-aware systems.

## Highlights

- **Balanced-ternary core**: `t81::Int` (an alias for `t81::core::limb`) ships overflow-aware arithmetic, canonical I/O, and deterministic hashing.
- **Ternary-friendly GEMMs**: `t81::linalg::gemm_ternary` packs balanced ternary matrices into AVX/NEON-accelerated kernels with alpha/beta semantics mirrored in the Python binding.
- **Python, CLI, and Torch helpers**: Pybind11 bindings expose quantize/dequantize utilities and `t81.torch`/`t81.nn`, while `t81-convert`, `t81-gguf`, and `t81-qat` automate quantize/export/train workflows.
- **Normative docs & demos**: Architecture notes, CLI references, and runnable demos live under `docs/`, `examples/`, and `bench/`.

## Start here

- **C++** → Quick start → *Build & test locally*
- **Python** → Quick start → *Python install* (pip + docs)
- **CLI workflows** → [docs/references/cli-usage.md](docs/references/cli-usage.md)
- **PyTorch integration** → [docs/torch.md](docs/torch.md)
- **Architecture & specs** → [ARCHITECTURE.md](ARCHITECTURE.md) & [docs/index.md](docs/index.md)

## Quick start

### 1. Build & test locally

```bash
git clone https://github.com/t81dev/t81lib.git
cmake -S . -B build -DT81LIB_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### 2. Install for Python consumers

```bash
pip install .[torch]
```

`pip install` builds the Python bindings, exposes `t81lib`/`t81`, and pulls the optional `torch` helpers when you request `[torch]`.
See [docs/python-install.md](docs/python-install.md) for pipx, CLI helpers, and verification scripts.

### 3. Build the Python bindings (optional)

```bash
cmake -S . -B build -DT81LIB_BUILD_TESTS=ON -DT81LIB_BUILD_PYTHON_BINDINGS=ON
cmake --build build -j
PYTHONPATH=build python tests/python/test_bindings.py
```

### 4. Consume as a subproject

```cmake
add_subdirectory(external/t81lib)
target_link_libraries(my_app PRIVATE t81::t81lib)
```

Include the umbrella header in your sources:

```cpp
#include <t81/t81lib.hpp>
```

### 5. Use through vcpkg

```bash
vcpkg install t81lib[tests,benchmarks]:x64-windows
find_package(t81lib REQUIRED)
target_link_libraries(... t81::t81lib)
```

## Python & PyTorch (quick overview)

```python
import torch
import t81.torch as t81_torch
import t81lib

weights = t81_torch.TernaryTensor.from_float(torch.randn(128, 128))
outputs = weights.matmul_input(torch.randn(32, 128))
packed = t81lib.pack_dense_matrix(weights, threshold=0.45)
```

The same handlers power the `t81.nn` layers that keep biases in FP32/BF16 while quantizing weights.
See [docs/python-api.md](docs/python-api.md), [docs/python-cookbook.md](docs/python-cookbook.md), [docs/python-install.md](docs/python-install.md), and
[docs/torch.md](docs/torch.md) for the full story.

## CLI helpers

`t81-convert`, `t81-gguf`, and `t81-qat` help you quantize, export GGUF bundles, and run quantization-aware training with alpha/beta controls.
See [docs/references/cli-usage.md](docs/references/cli-usage.md),
[docs/diagrams/cli-workflows-mermaid.md](docs/diagrams/cli-workflows-mermaid.md), and
[examples/cli-examples.md](examples/cli-examples.md) for walkthroughs and ready-to-run snippets.

## Use cases (see docs)

- Ternary LLM weight quantization and GGUF exports for Hugging Face + `llama.cpp`.
- Packed ternary GEMMs for CPU inference and `torch.matmul` comparisons.
- Deterministic numerics and research-grade arithmetic built atop the same core.
- Ternary hardware simulation and energy modeling (see [docs/hardware.md](docs/hardware.md)).

See [docs/use-cases.md](docs/use-cases.md) for demos, notebooks, and experiments that spotlight these flows.

## Examples

See [examples/README.md](examples/README.md) for the canonical scripts/notebooks (LLM conversion, GEMM packing, quantization demos, hardware previews, etc.).

## Numeric building blocks

`t81lib` exposes fixed-width ternary integers, big integers, rationals, floats, vectors, and Montgomery helpers that share a single canonical core.
See [docs/api-overview.md](docs/api-overview.md) for the full surface described in the umbrella header.

## Docs & resources

- [docs/index.md](docs/index.md) — Docs portal (great for GitHub Pages).
- [docs/t81lib-spec-v1.0.0.md](docs/t81lib-spec-v1.0.0.md) — Normative contract for consumers.
- [docs/design/](docs/design/) — Deep dives on `limb`, `bigint`, and `montgomery` internals.
- [docs/python-install.md](docs/python-install.md) — Detailed install flow for Python, pipx, and CLI helpers.
- [docs/python-api.md](docs/python-api.md) & [docs/python-cookbook.md](docs/python-cookbook.md).
- [docs/torch.md](docs/torch.md) — PyTorch integration, `t81.torch`, and `t81.nn`.
- [docs/use-cases.md](docs/use-cases.md), [docs/hardware.md](docs/hardware.md), [docs/api-overview.md](docs/api-overview.md).
- [examples/README.md](examples/README.md) — Canonical scripts/notebooks.
- [tests/](tests/) & [bench/](bench/) — Regression suites and throughput gauges.
- [CONTRIBUTING.md](CONTRIBUTING.md) & [CHANGELOG.md](CHANGELOG.md).

## Benchmarks

See [BENCHMARKS.md](BENCHMARKS.md) for the Fashion-MNIST FP32/PTQ/QAT comparison.

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
