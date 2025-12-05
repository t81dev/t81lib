# Design Rationale

Balanced ternary arithmetic is the core motivation for t81lib. This document explains why the radix and the chosen limb layout were favored and how `T81Limb` exposes that design to users.

## Why balanced ternary?

- **Symmetry**: Balanced ternary digits (`-1, 0, +1`) center zero, eliminate bias, and support negation without a sign bit.
- **Minimal carries**: The three-valued logic reduces the probability of long carry chains, which the Kogge-Stone adder in `T81Limb` exploits to deliver consistent addition latency.
- **Storage density**: Each trit encodes 3 states, so a 48-trit limb (16 trytes) fits comfortably into existing CPU registers while representing 243⁴³ states.
- **Ternary-native algorithms** (e.g., Booth, Karatsuba) are easier to map onto this balanced representation than to simulate on binary limbs of the same bit width.

## Limb structure

`T81Limb` packs 16 trytes (3 trits each) into a 48-trit value. Every tryte is stored with the range `[-13, +13]`, providing exactly 27 states per tryte and enabling Kogge-Stone-style carry propagation tables (`detail::ADD_TABLE`, `detail::COMPOSITION_TABLE`) that operate on small LUTs.

The limb exposes:

- `operator+`/`addc` — constexpr parallel-prefix addition using LUT composition tables.
- `operator-`/`operator-()` — zero-cost negation via `-trytes` + 1, matching the balanced representation.
- `mul_booth_karatsuba` (via `operator*`) — wide `mul_wide` plus Karatsuba decomposition into `T81Limb27`/`T81Limb54`.
- `mul_wide_fast` — a Karatsuba-optimized path guarded by canonical tests (`mul_wide_canonical`).

## Extending precision

To build larger bigints, stack `T81Limb` values (e.g., `std::array<T81Limb, N>`). Use `mul_wide` to emit low/high halves for schoolbook algorithms and rely on `T81Limb27`/`T81Limb54` for Karatsuba splits when you need sub-limb optimizations.

T81 limbs also provide packing helpers (`t81::core::packing`) so you can compare storage cost to binary encodings, and the configuration header (`t81/t81lib_config.hpp`) exposes feature flags/versioning that keep extensions compatible across releases.
