# t81lib — Balanced Ternary Core Library

![CI](https://github.com/t81dev/t81lib/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-blue)
![Docs](https://img.shields.io/badge/docs-via%20GitHub%20Pages-brightgreen)

`t81lib` is a modern, header-first C++20 library that brings **balanced ternary** arithmetic
to production-grade software. At the heart sits `t81::core::limb`—a canonical 48-trit scalar with a
deterministic 16-byte encoding—and a carefully layered ecosystem for high-level math,
modular helpers, and deterministic utilities.\n

## Highlights

- **Balanced ternary scalar**: `limb` exposes safe, overflow-aware arithmetic, canonical I/O,
  and deterministic hashing when you need ternary determinism in a binary world.
- **Arbitrary-precision math**: `t81::core::bigint` layers on top of limbs with sign-plus-magnitude,
  Karatsuba-aware multiplication, canonical normalization, and full conversion helpers.
- **Concrete helpers**: Montgomery contexts, I/O formatters, random tooling, and utility guards
  keep reusable patterns consistent and testable.
- **Specs & architecture**: Normative coverage under `doc/`, plus a human-friendly
  `ARCHITECTURE.md` walkthrough (new!) and a docs portal at `docs/index.md` for quick orientation.
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
using t81::core::limb;

limb a = limb::from_int(42);
limb b = limb::from_int(-7);
limb sum = a + b; //... deterministically balanced ternary
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

## Architecture snapshot

- **limb**: 48 trits (trytes), stable 16-byte packing, explicit overflow, deterministic hashing, and
  conversion helpers for floating point and integer views.
- **bigint**: sign bit + limbs, normalization, magnitude add/sub/mul/div, and trit/tryte helpers for shifting,
  rotating, and bitwise operations.
- **Montgomery & helpers**: contexts, modular multiply/pow, context guards, and bridging helpers in
  `t81::core::montgomery`.
- **I/O, util, docs**: `t81::io` for formatting/parsing, `t81::util` for randomness & invariants,
  and mirrored design docs under `doc/`.
- **New architecture guide**: See `ARCHITECTURE.md` for a concise narrative plus ASCII call flow of the core layers.

## Docs & resources

- `doc/t81lib-spec-v1.0.0.md` — Normative library contract for consumers.
- `doc/design/` — Deep dives on `limb`/`bigint`/`montgomery` internals.
- `docs/index.md` — A friendly docs portal pointing to specs, examples, and tutorials (great for GitHub Pages).
- `examples/` — Compileable demos that exercise `limb`, `bigint`, and Montgomery helpers.
- `tests/` & `bench/` — Regression/property suites and microbenchmarks for throughput decisions.
- `CONTRIBUTING.md` & `CHANGELOG.md` — Expectations, workflows, and the evolution log.

## Testing & benchmarks

- `cmake -S . -B build -DT81LIB_BUILD_TESTS=ON`
- `cmake --build build -j`
- `ctest --test-dir build --output-on-failure`
- `cmake -S . -B build -DT81LIB_BUILD_BENCH=ON -DT81LIB_USE_GOOGLE_BENCH=ON && cmake --build build -j`
- Execute `./build/bench/*` to compare trit-run performance.

## Contribution & support

- Open an issue to discuss architecture or API needs.
- Follow `CONTRIBUTING.md` for branching, formatting, and testing workflows.
- Reference `doc/design/` when proposing optimizations to ensure semantic safety.
- Track progress in `CHANGELOG.md` for breaking changes or release notes.
- Security concerns can be reported privately; the repo currently follows a standard disclosure cadence.

## License

`t81lib` is MIT-licensed; see `LICENSE`.
