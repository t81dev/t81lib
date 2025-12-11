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

Enable the Python bindings by configuring with `-DT81LIB_BUILD_PYTHON_BINDINGS=ON`. Then build normally and point `PYTHONPATH` at the build directory before importing the module:

```bash
cmake -S . -B build -DT81LIB_BUILD_TESTS=ON -DT81LIB_BUILD_PYTHON_BINDINGS=ON
cmake --build build -j
PYTHONPATH=build python tests/python/test_bindings.py
```

After a successful build `import t81lib` exposes the `BigInt` class plus helper functions such as `zero()`, `one()`, `gcd()`, and `mod_pow()`:

```python
import t81lib

value = t81lib.BigInt(42)
value = value * t81lib.one()
print(str(value))
```

Use `tests/python/test_bindings.py` as a simple sanity-check script and reference for building larger Python workflows.

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
  and supports multiplying while trimming trailing zero trits.
- `t81::FloatN<N>` mirrors `Float` but pins the mantissa to `N` trits via `Fixed<N>` so you can reason about fixed precision floats at compile time.
- `t81::Ratio` keeps normalized, sign-aware rational numbers powered by `core::bigint` numerators
  and denominators, complete with arithmetic and comparison helpers.
- `t81::Complex` and `t81::Polynomial` model simple complex arithmetic and polynomial math
  on any coefficient that satisfies the usual operators.
- `t81::F2m` wraps extension-field arithmetic over binary polynomials using a chosen modulus,
  providing reduction, addition, multiplication, and exponentiation.
- `t81::Fixed<N>` represents fixed-width signed ternary values with modular normalization
  and arithmetic in the range `-(3^N-1)/2` .. `(3^N-1)/2`.
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
