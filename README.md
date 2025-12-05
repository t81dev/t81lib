# t81lib

t81lib is a small foundation for balanced ternary arithmetic. It primarily exposes:

- `t81::core::T81Limb`: packed 48-trit limbs with Kogge-Stone addition, Booth/Karatsuba multiplication, and comparison helpers.
- `t81::core::packing`: conversions between trits and compact binary states to measure the real cost of storage.
- Benchmarks under `bench/` that include ASCII reporting, GMP/Boost/TTMath comparisons, and limb math smoke tests.

## Documentation

- `doc/overview.md` summarizes the numeric philosophy, key types, and the optional benchmark surface.
- `CHANGELOG.md` tracks the latest progress.
- `CODE_OF_CONDUCT.md` and `CONTRIBUTING.md` explain how to collaborate respectfully.
- API docs may be generated with Doxygen by adding a Doxyfile and run `cmake --build build --target doc`.
- `doc/formatting.md` captures the preferred `clang-format`/`clang-tidy` workflow so contributions stay consistent.
## Automation

- **CI**: GitHub Actions builds the project (`cmake --build`, `ctest`) on pushes/PRs to `main`.
- **Performance suite**: Scheduled GitHub Actions rebuild with `BUILD_BENCHMARKS=ON` and runs `./build-bench/bench/t81lib-bench`.
- **Releases**: Tag `vX.Y.Z` to trigger the release workflow that packages the headers/docs and publishes a GitHub release referencing `CHANGELOG.md`.

## Examples & pkg-config

`examples/basic` demonstrates how to `find_package(t81lib)` and consume the header-only target from another CMake project. Build it with:

```bash
cmake -S examples/basic -B examples/basic/build
cmake --build examples/basic/build
./examples/basic/build/example_basic
```

After installing t81lib (`cmake --install build`), a `lib/pkgconfig/t81lib.pc` file is generated so legacy build systems can incorporate the headers via `pkg-config --cflags t81lib`.

## Building

```bash
cmake -S . -B build
cmake --build build
cmake --build build --target test
```

The default build produces the header-only `t81lib` interface plus `tests/basic.cpp`.

## Benchmarks (optional)

```bash
cmake -S . -B build-bench -DBUILD_BENCHMARKS=ON
cmake --build build-bench
./build-bench/bench/t81lib-bench
```

Benchmarks run only when `BUILD_BENCHMARKS` is enabled; otherwise the ASCII dashboard and competitors stay out of the standard workflow.

## Layout

- `include/` — public headers for `t81::core`.
- `src/` — packaging helpers mirrored for embedders.
- `tests/` — lightweight smoke tests.
- `bench/` — optional representative benchmarks (GMP/Boost/TTMath comparisons, ASCII plots).
- `doc/` — reference overview.

## Configuration header

Include `<t81/t81lib_config.hpp>` to inspect the compiled version numbers, stringified version, and feature detection macros. The generated file exposes `T81LIB_VERSION_MAJOR`, `T81LIB_VERSION_MINOR`, `T81LIB_VERSION_PATCH`, `T81LIB_VERSION_STRING`, `T81LIB_VERSION_NUMBER`, and `T81LIB_HAS_BENCHMARKS` so downstream code can adapt to build-time decisions.

## License

MIT. See `LICENSE`.

## Installation

After building, run `cmake --install build` (or point to your install prefix) to copy the headers and install files. Downstream projects can then `find_package(t81lib)` to consume the exported `t81::t81lib` target and include headers from the installed `include/` tree.
