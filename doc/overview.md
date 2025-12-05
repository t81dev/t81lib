# t81lib Overview

t81lib is a header-first library for high-throughput balanced ternary arithmetic. The core surface exposes:

- `t81::core::T81Limb`: a 48-trit limb with packed trytes, Kogge-Stone adders, and Karatsuba multiplication.
- `t81::core::packing`: utilities for packing/unpacking trits into compact bit patterns.
- Helpers such as `T81Limb27/54` and `T81Limb54::karatsuba` support higher precision arithmetic and conversions.

Highlights:

- Zero-cost addition, subtraction, negation, and comparison paths.
- Benchmarks that compare against GMP/Boost/TTMath and showcase ASCII dashboards.
- Optional research-grade routines (e.g., `mul_wide_fast`, Booth multipliers) contained in `include/t81/core`.

## Configuration header

The generated `<t81/t81lib_config.hpp>` records the current project version and toggles such as `T81LIB_HAS_BENCHMARKS`. Consumers can include it to gate behavior on version numbers or request the same feature set that was configured for t81lib itself.

Grab the headers from `include/` and link your project against the `t81lib` interface target to reuse the limb arithmetic.
