# API overview

This page captures the high-level helpers exposed by the umbrella header so you can understand the building blocks without diving into every header. Review the [core architecture diagram](diagrams/core-architecture.mermaid.md) for an inheritance/data-flow sketch of the same helpers.

## Core numerics

- `t81::Int` aliases `t81::core::limb`, the balanced ternary scalar with deterministic encoding, canonical I/O, and overflow-aware arithmetic.
- `t81::Float` and `t81::FloatN<N>` represent ternary floating-point numbers with normalized mantissas; `Float::from_string` and `std::formatter` keep the canonical notation intact.
- `t81::Ratio`, `t81::Complex`, and `t81::Polynomial` layer higher-level algebra atop the core numerics while reusing the same canonical types.

## Containers & linear algebra

- `t81::Vector<T>` and `t81::Matrix<Element, R, C>` provide fixed-length coefficient storage with element-wise arithmetic, scalar multipliers, and dot products over ternary-aware elements.
- `t81::linalg::gemm_ternary` multiplies packed ternary matrices into FP32/BF16 accumulators, exposes alpha/beta scaling, and powers both the C++ core and the Python binding `t81lib.gemm_ternary`.

## Specialized helpers

- `t81::F2m` models GF(2^m) arithmetic for extension-field use cases.
- `t81::Fixed<N>` keeps fixed-width signed ternary values, now with balanced `/` and `%` helpers plus three-way comparisons for ordered containers.
- `t81::Modulus` and `t81::MontgomeryInt` build and operate Montgomery contexts with cached powers of three and modular safety guards.
- `t81::ntt` exposes radix-3 NTT pipelines for `Polynomial<core::limb>` multiplication when the modulus admits a primitive third root.
- `t81::literals::_t3` provides a ternary literal with overline support so you can author canonical strings directly in user code.

This page is the shortcut for maintainers who want to understand the API surface without hunting through every header.
