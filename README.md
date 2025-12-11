<!--
README.md — Repository overview, setup, and quickstart instructions.
-->

# t81lib — Balanced Ternary Core Library

![CI](https://github.com/t81dev/t81lib/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-blue)
![Docs](https://img.shields.io/badge/docs-via%20GitHub%20Pages-brightgreen)

`t81lib` is a modern, header-first C++20 library that brings **balanced ternary** arithmetic
to production-grade software. At the heart sits `t81::core::limb`—a canonical 48-trit scalar with a
deterministic 16-byte encoding—and a carefully layered ecosystem for high-level math,
modular helpers, and deterministic utilities.

## Highlights

- **Balanced ternary scalar**: `t81::Int` (alias of `t81::core::limb`) exposes safe, overflow-aware arithmetic, canonical I/O,
  and deterministic hashing when you need ternary determinism in a binary world.
- **Arbitrary-precision math**: `t81::core::bigint` layers on top of limbs with sign-plus-magnitude,
  Karatsuba-aware multiplication, canonical normalization, and full conversion helpers.
- **SIMD accelerations**: The SIMD helpers under `include/t81/core/detail/simd_impl.hpp` contain AVX-512, AVX2, and NEON implementations for tryte addition/multiplication so the core limb arithmetic can leverage vector hardware when available.
- **Base81 I/O**: Formatting/parsing now supports bases 2..81 with the playful base-3⁴ alphabet (0-9, a-z, A-Z, and punctuation) so you can round-trip balanced-ternary-friendly strings without extra glue.
- **Concrete helpers**: Montgomery contexts, I/O formatters, random tooling, and utility guards
  keep reusable patterns consistent and testable.
- **High-level helpers**: The umbrella header now also exposes `t81::Float`, `t81::Ratio`,
  `t81::Complex`, `t81::Polynomial`, `t81::F2m`, and `t81::Fixed<N>` along with modular
  helpers like `t81::Modulus`/`t81::MontgomeryInt` for quick prototyping of ternary-aware
  algebra and number-theoretic math.
- **Specs & architecture**: Normative coverage under [doc/](doc/), plus a human-friendly
  [ARCHITECTURE.md](ARCHITECTURE.md) walkthrough (new!) and a docs portal at [docs/index.md](docs/index.md) for quick orientation.
- **Examples & proofs**: `examples/` shows runnable use cases while `tests/` and
  `bench/` capture behavioral regression suites and throughput gauges.

## Quick start

### 1. Build & test locally

```bash
git clone https://github.com/t81dev/t81lib.git
cmake -S . -B build -DT81LIB_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### Python bindings

#### Build from source (CMake)

Enable the Python bindings by configuring with `-DT81LIB_BUILD_PYTHON_BINDINGS=ON`, then build and point `PYTHONPATH` at that build directory:

```bash
cmake -S . -B build -DT81LIB_BUILD_TESTS=ON -DT81LIB_BUILD_PYTHON_BINDINGS=ON
cmake --build build -j
PYTHONPATH=build python tests/python/test_bindings.py
```

After a successful build `import t81lib` exposes the `Limb`, `BigInt`, and Montgomery helpers plus bundled numerical utilities:

```python
import t81lib

value = t81lib.BigInt(42)
value = value * t81lib.one()
print(str(value))

ctx = t81lib.BigIntMontgomeryContext(t81lib.BigInt(17))
print(ctx.mod_pow(value, t81lib.BigInt(3)))
```

Use `tests/python/test_bindings.py` as a sanity-check script or reference when you need to inspect the pybind11 glue.

#### Install via pip

The library ships with `setup.py`/`pyproject.toml`, so `pip` will call CMake, build the same extension, and install the `t81lib` module together with the `t81` Python helpers defined at the repo root.

```bash
pip install .
pip install .[torch]   # optional PyTorch helpers, pulls in torch>=2.0
```

The bindings expose balanced ternary primitives and helpers for integrating NumPy or buffer-compatible tensors (including Torch tensors that live on CPU). Highlights include:

* `t81lib.Limb` / `t81lib.BigInt` along with `t81lib.LimbMontgomeryContext` and `t81lib.BigIntMontgomeryContext`.
* `t81lib.quantize_to_trits` / `t81lib.dequantize_trits` for converting between floats and the \{-1,0,1\} digit stream.
* `t81lib.pack_dense_matrix`, which clamps a NumPy float32 matrix, quantizes each row to balanced ternary, and returns a `(rows, limbs, 16)` buffer of packed limb bytes suitable for `gemm_ternary`.
* `t81lib.unpack_packed_limbs`, which reconstructs the trits stored inside `pack_dense_matrix` when you need debugging data or logistic checks.

For example:

```python
import numpy as np
import t81lib

weights = np.random.randn(16, 64).astype(np.float32)
packed = t81lib.pack_dense_matrix(weights, threshold=0.45)
trits = t81lib.unpack_packed_limbs(packed, rows=16, cols=64)
dequantized = t81lib.dequantize_trits(trits)
```

Pair `pack_dense_matrix` with `t81lib.gemm_ternary` (or the PyTorch helpers under `t81.torch`) to keep packed ternary GEMMs on the host while your float32 accumulators gather the results. See `examples/ternary_quantization_demo.ipynb` for a ready-to-run walkthrough that turns a dense layer matrix into packed limbs, feeds it through `gemm_ternary`, and inspects the quantized trits.

### 2. Consume as a subproject

```cmake
add_subdirectory(external/t81lib)
target_link_libraries(my_app PRIVATE t81::t81lib)
```

Include the umbrella header:

```cpp
#include <t81/t81lib.hpp>
```

### 3. Use through vcpkg

```bash
vcpkg install t81lib[tests,benchmarks]:x64-windows
```

Once installed: `find_package(t81lib REQUIRED)` + `target_link_libraries(... t81::t81lib)`.

## PyTorch helpers

`import t81` now pulls in the PyTorch bridge under `t81.torch`, which repacks `t81::linalg::gemm_ternary` into a `TernaryTensor` that plays nicely with `torch.matmul`/`torch.mm` and exposes the custom `t81.trit` dtype. Call `TernaryTensor.from_float(...)` to quantize ternary weights and `TernaryTensor.matmul_input(...)` to fire off packed GEMMs inside your training/inference loops:

```python
import torch
import t81.torch as t81_torch

weights = t81_torch.TernaryTensor.from_float(torch.randn(128, 128))
outputs = weights.matmul_input(torch.randn(32, 128))
```

The companion `t81.nn` module keeps scalars exact (e.g., `Ratio`-based RMSNorm, RoPE, and softmax) while integrating with the same ternary GEMM plumbing, so you can simply do `model.to(dtype=t81.trit)` and let `t81.torch`/`t81.nn` handle the quantization, dispatch, and gradient flow. See `examples/demo_llama_conversion.py`, `examples/scaling_laws_ternary.py`, and `examples/ternary_sparse_preview.py` for runnable PyTorch + ternary GEMM demos.

## AI use cases

`t81lib` is already powering ternary-aware workflows that push the performance envelope in real-world AI systems. For rapid experimentation, the Python bindings let you swap a dense floating-point tensor for a `TernaryTensor`, quantize activations on the fly, and keep GEMMs on the host using the AVX/NEON-packed kernels. The demos above show how to:

1. Convert an existing large language model to ternary weights while preserving `torch.nn.Module` semantics (`examples/demo_llama_conversion.py`).
2. Study scaling-law behavior in ternary networks, comparing sparsity, precision, and model size trade-offs using `examples/scaling_laws_ternary.py`.
3. Preview ternary-sparse transformers with custom `SparseTriangular` layers and quantized attention using `examples/ternary_sparse_preview.py`.

For more advanced pipelines, the `t81.nn` helpers expose RMSNorm, softmax, and RoPE in exact rational or fixed-width representations so you can build ternary-friendly training/inference loops without re-implementing numerics. When performance matters, point profiling tools at `t81::linalg::gemm_ternary` (or `t81lib.gemm_ternary` in Python) to compare packed ternary GEMMs against baseline `torch.matmul` runs; the alpha/beta semantics make it easy to blend ternary updates with FP32 accumulators for mixed-precision schedules.

## AI Examples

- [`examples/ternary_mnist_demo.ipynb`](examples/ternary_mnist_demo.ipynb) walks through quantizing a compact MNIST classifier, packing weight buffers with `t81lib.pack_dense_matrix`, and routing inference through `t81lib.gemm_ternary` for a compact comparison of accuracy, latency, and memory versus float32 and 1-bit baselines.
- [`examples/ternary_transformer_demo.ipynb`](examples/ternary_transformer_demo.ipynb) builds a micro GPT stack, caches ternary projections via column-aware packing, profiles generation with `t81lib.gemm_ternary`, and highlights sparse attention multiplications via `t81lib.spmm_simple`.

## Usage at a glance

```cpp
#include <t81/t81lib.hpp>
using t81::Int;

Int a = Int::from_int(42);
Int b = Int::from_int(-7);
Int sum = a + b; //... deterministically balanced ternary
```

```cpp
#include <t81/t81lib.hpp>
using t81::core::bigint;

bigint x{42};
bigint y{1};
for (int i = 0; i < 10; ++i) {
    y *= x;
}
```

Montgomery helpers live in `<t81/core/montgomery_helpers.hpp>`:

```cpp
using namespace t81::core;
using namespace t81::core::montgomery;
const limb modulus = limb::from_value(97);
auto ctx = make_limb_context(modulus);
auto product = modular_multiply(ctx, limb::from_value(5), limb::from_value(7));
```

Callers can also rely on guard classes for const-time exponent limits.

## High-level numeric helpers

Beyond the core limb and bigint foundations, the umbrella header now exposes
convenient containers for common algebraic patterns:

- `t81::Int` is the umbrella alias for `t81::core::limb`, letting you grab the balanced
  ternary scalar without importing `core` names.

- `t81::Float` stores a ternary mantissa/exponent pair, keeps the mantissa normalized,
  and supports multiplying while trimming trailing zero trits; `Float::from_string`
  now understands scientific-style `e`/`E` trit exponents plus the `₃` tryte exponent
  marker, and `t81::to_string` accepts a `max_fractional_digits` hint so you can round
  or truncate the printed fractional portion.
- `t81::FloatN<N>` mirrors `Float` but pins the mantissa to `N` trits via `Fixed<N>` so you can reason about fixed precision floats at compile time.
- `t81::Ratio` keeps normalized, sign-aware rational numbers powered by `core::bigint` numerators
  and denominators, complete with arithmetic and comparison helpers; use `Ratio::to_float()`
  to get a rounded ternary `Float` when you need a floating-point view.
- `t81::Complex` and `t81::Polynomial` model simple complex arithmetic and polynomial math
  on any coefficient that satisfies the usual operators.
- `t81::Vector<T>` holds a length-`N` collection of coefficients, and `t81::Matrix<Element, R, C>`
  exposes resize-fixed matrices with element-wise arithmetic, scalar/Matrix multiplication, and
  vector products over the same ternary-aware scalars.
- `t81::Vector<T>` holds a length-`N` collection of coefficients, exposes element-wise add/subtract,
  scaling, and dot-product helpers, and reuses the same balanced-ternary value types as the numeric helpers.
- `t81::linalg::gemm_ternary` multiplies packed balanced-ternary matrices (`A` shaped `[M, K/48]`,
  `B` shaped `[K/48, N]`) into an FP32 accumulation buffer, exposes alpha/beta scaling, and is
  mirrored in Python via `t81lib.gemm_ternary` so tensor-heavy workloads can run on AVX/NEON hosts.
- `t81::F2m` wraps extension-field arithmetic over binary polynomials using a chosen modulus,
  providing reduction, addition, multiplication, and exponentiation.
- `t81::Fixed<N>` represents fixed-width signed ternary values with modular normalization
  and arithmetic in the range `-(3^N-1)/2` .. `(3^N-1)/2`, and the new three-way comparison
  lets you put `Fixed<N>` (and the `Int81` alias) into ordered containers.
- `t81::Modulus` and `t81::MontgomeryInt` let you declaratively build Montgomery contexts,
  cache powers of three, and multiply/add in Montgomery space with consistent modular safety.
- `t81::ntt` exposes a radix-3 NTT pipeline for `Polynomial<core::limb>` multiplication when the modulus supports a primitive third root of unity.
- `t81::literals::_t3` is a user-defined literal for ternary strings (overlines included), and the new `std::formatter` specializations for `Float` and `FloatN` make `std::format`/`std::print` render the canonical ternary notation.

These helpers make it easy to prototype higher-level systems without leaving the umbrella header.

## Architecture snapshot

- **limb**: 48 trits (trytes), stable 16-byte packing, explicit overflow, deterministic hashing, and
  conversion helpers for floating point and integer views.
- **bigint**: sign bit + limbs, normalization, magnitude add/sub/mul/div, and trit/tryte helpers for shifting,
  rotating, and bitwise operations.
- **Montgomery & helpers**: contexts, modular multiply/pow, context guards, and bridging helpers in
  `t81::core::montgomery`.
- **I/O, util, docs**: `t81::io` for formatting/parsing, `t81::util` for randomness & invariants,
  and mirrored design docs under [doc/](doc/).
- **New architecture guide**: See [ARCHITECTURE.md](ARCHITECTURE.md) for a concise narrative plus ASCII call flow of the core layers.

## Docs & resources

- [doc/t81lib-spec-v1.0.0.md](doc/t81lib-spec-v1.0.0.md) — Normative library contract for consumers.
- [doc/design/](doc/design/) — Deep dives on `limb`/`bigint`/`montgomery` internals.
- [docs/index.md](docs/index.md) — A friendly docs portal pointing to specs, examples, and tutorials (great for GitHub Pages).
- [examples/](examples/) — Compileable demos that exercise `limb`, `bigint`, and Montgomery helpers.
- [tests/](tests/) & [bench/](bench/) — Regression/property suites and microbenchmarks for throughput decisions.
- [CONTRIBUTING.md](CONTRIBUTING.md) & [CHANGELOG.md](CHANGELOG.md) — Expectations, workflows, and the evolution log.

## Testing & benchmarks

- `cmake -S . -B build -DT81LIB_BUILD_TESTS=ON`
- `cmake --build build -j`
- `ctest --test-dir build --output-on-failure`
- `cmake -S . -B build -DT81LIB_BUILD_BENCH=ON -DT81LIB_USE_GOOGLE_BENCH=ON && cmake --build build -j`
- Execute `./build/bench/*` to compare trit-run performance.

## Contribution & support

- Open an issue to discuss architecture or API needs.
- Follow [CONTRIBUTING.md](CONTRIBUTING.md) for branching, formatting, and testing workflows.
- Reference [doc/design/](doc/design/) when proposing optimizations to ensure semantic safety.
- Track progress in [CHANGELOG.md](CHANGELOG.md) for breaking changes or release notes.
- Security concerns can be reported privately; the repo currently follows a standard disclosure cadence.

## License

`t81lib` is MIT-licensed; see [LICENSE](LICENSE).
