# API Reference & Recipes

Use this guide to understand the public-facing APIs that compose `t81lib` and learn how to apply them to wide multiplies, canonical conversions, and packing combinations.

## High-level surface

- `t81::core::T81Limb` — 48-trit limbs stored as 16 packed trytes (0..16). Exposes constexpr Kogge-Stone addition (`operator+`/`addc`), Booth/Karatsuba multiplication (`operator*`, `mul_wide`, `mul_wide_fast`), zero-cost negation/subtraction, and raw comparison helpers.
- `t81::core::T81Limb27` / `T81Limb54` — integer compositions used internally for Karatsuba; you can rely on them if you need smaller limb views with the same interface.
- `t81::core::packing` — helper functions for encoding/decoding trits to compact binary states plus utility constants that expose how many bits represent `N` trits.

## Usage recipes

### Wide multiply (mul_wide)

1. Convert the operands to `T81Limb`.
2. Call `auto [lo, hi] = T81Limb::mul_wide(a, b);` for a full 96-trit result.
3. The low and high limbs are ready to be stored or fed into higher precision arithmetic.

```cpp
using t81::core::T81Limb;

T81Limb lhs = T81Limb::from_int(42);
T81Limb rhs = T81Limb::from_int(-7);

auto [lo, hi] = T81Limb::mul_wide(lhs, rhs);
```

`mul_wide_fast` applies the Karatsuba path and is guarded by build-time assertions in debug builds for bit-for-bit parity with the canonical multiplication.

### Conversions

- `T81Limb::from_int(int64_t)` — converts a C++ integer in range `[-121, +121]` into a limb. Throws if the value falls outside the supported range.
- `T81Limb::to_int()` — converts the limb back into a native integer.
- Tryte-level helpers (`t81::encode_tryte`, `t81::decode_tryte`) expose the 3-trit bundles that compose each limb.

Use these conversions whenever you cross the boundary from native numerics to balanced ternary data structures.

### Packing trits

`t81::core::packing` gives you the ability to measure density:

```cpp
using namespace t81::core::packing;
std::array<Trit, 8> trits{Trit::P, Trit::Z, Trit::M, Trit::P, Trit::P, Trit::Z, Trit::Z, Trit::M};
auto packed = pack_trits(trits);
auto roundtrip = unpack_trits<8>(packed);
```

`states_for_trits(N)` and `packed_bits(N)` show how many binary bits are required to cover the entire ternary state space for `N` trits.

## Generating docs

We maintain a `Doxyfile` in the repository root. Generate the full API HTML reference with:

```bash
doxygen Doxyfile
```

The generated output lands under `docs/html` by default and makes it easy to browse the public headers plus `doc/api.md` itself.
