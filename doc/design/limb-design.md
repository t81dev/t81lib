# limb-design.md — t81::core::limb Design Notes

**Status:** Informative (non-normative)  
**Target:** `t81::core::limb` in `include/t81/core/limb.hpp`  
**Related spec:** `t81lib-spec-v1.0.0.md` (normative scalar spec)

This document explains how the canonical 48-trit balanced-ternary scalar `t81::core::limb` is implemented inside `t81lib`. It is a design and implementation guide, not a formal spec; if it ever disagrees with the spec, the spec wins.

---

## 1. Role in the Library

`t81::core::limb` is the canonical fixed-width balanced-ternary scalar:

- All higher-level numerics in `t81lib` (e.g. `t81::core::bigint`) are defined in terms of `limb` values.
- Binary serialization, hashing, and cross-process interoperability are specified at the limb level.
- SIMD and “fancy” algorithms may be added, but they must be observationally identical to the scalar semantics defined in the spec.

In short: **limb is the anchor**; everything else is built on it.

---

## 2. Design Objectives

Key design goals:

1. **Deterministic semantics:**  
   No UB, no implementation-defined overflow. All behavior is specified.

2. **Canonical encoding:**  
   - Exactly one 16-tryte representation for each mathematical value.
   - Representation is stable across builds and platforms.

3. **Ternary-native operations:**  
   - Arithmetic works in balanced ternary, not binary emulation.
   - Tritwise logical ops (`&`, `|`, `^`, `consensus`) are exposed as first-class operations.

4. **Reference-friendly implementation:**  
   - Algorithms are defined in terms of clear reference models:
     - LUT-driven Kogge–Stone-style add.
     - Schoolbook + normalized `mul_wide`.
   - Makes conformance testing and alternative implementations straightforward.

5. **Composable building block:**  
   - Simple, predictable APIs so higher layers (`bigint`, vectors, VMs) can treat `limb` like a well-behaved “ternary int128”.

---

## 3. Representation

### 3.1 Mathematical model

A `limb` represents:

```text
x = ∑_{i=0}^{47} d_i · 3^i,  where d_i ∈ {−1, 0, +1}
````

with range:

```text
min() = −(3^48 − 1)/2
max() = +(3^48 − 1)/2
```

These bounds are chosen so that the value set is symmetric around zero and closed under ± and most typical operations used by higher layers (up to explicit overflow checks).

### 3.2 Tryte packing

Internally:

* 48 trits are grouped into 16 **trytes** of 3 trits each.
* Each tryte is encoded as a byte `tryte_t` with range `0..26`.
* Storage:

  ```cpp
  class limb {
  public:
      static constexpr int TRITS  = 48;
      static constexpr int TRYTES = 16;
      static constexpr int BYTES  = 16;
      using tryte_t = std::uint8_t;

  private:
      alignas(16) tryte_t trytes_[TRYTES];
  };
  ```

Layout:

* Little-endian in **trytes**:

  * `trytes_[0]` → trits 0..2 (least-significant trits).
  * `trytes_[1]` → trits 3..5.
  * …
  * `trytes_[15]` → trits 45..47 (most-significant trits).

This layout is the canonical 16-byte serialization used by `from_bytes` / `to_bytes`.

### 3.3 Trit ↔ tryte LUTs

To avoid ambiguous encodings, we fix a single mapping:

* `tryte_t` ∈ `{0..26}` ↔ `(t0, t1, t2)` ∈ `{−1,0,+1}³`.

This mapping is materialized as:

```cpp
namespace t81::core::detail {

extern const std::array<std::array<std::int8_t, 3>, 27> TRYTE_TO_TRITS;
// and a matching inverse mapping (conceptual TRITS_TO_TRYTE)

} // namespace t81::core::detail
```

All decoding/encoding of trits is done strictly through this pair of tables (or an equivalent implementation that is provably identical). This guarantees:

* Unique encoding for every trit triple.
* Cross-implementation compatibility.

---

## 4. Public Interface (Summary)

The formal contract lives in the spec. Implementation-wise, `limb` exposes:

* Constants and introspection:

  * `zero()`, `one()`, `min()`, `max()`, `is_zero()`, `is_negative()`, `signum()`, etc.
* Indexed access:

  * `get_trit(i)`, `set_trit(i, value ∈ {−1,0,+1})`.
  * `get_tryte(i)`, `set_tryte(i, tryte_t)` (low-level).
* Arithmetic:

  * `+`, `-`, unary `-`, `*`, `/`, `%`, `div_mod`, `pow_mod`.
* Comparison:

  * `compare`, full relational operators, `operator<=>`.
* Tritwise logical:

  * `&`, `|`, `^`, `~`, `consensus`.
* Shifts and “rotations”:

  * Tryte shifts (`<<`, `>>`) and trit-level “rotations” that are actually zero-filling shifts (per spec).
* Conversions:

  * To/from native integer types, and to/from strings, plus `from_bytes` / `to_bytes`.

This design doc focuses on how those operations are *implemented* in terms of the underlying trit/tryte structure.

---

## 5. Internal Architecture

### 5.1 Helper views

Internally we rely on two conceptual helpers:

1. **Trit view**:

   A temporary `std::array<std::int8_t, 48>` where each entry is in `{−1,0,+1}`.

   * `to_trits()` decodes all 16 trytes using `TRYTE_TO_TRITS`.
   * `from_trits()` encodes trits back into trytes using the inverse LUT.

2. **Wide accumulator** (for multiplication):

   * `std::array<int, 96>` for intermediate convolution and carry/borrow normalization.

These views are not part of the public API but are used in the reference algorithms.

### 5.2 Core LUTs

In addition to `TRYTE_TO_TRITS` / inverse, we use:

* **Addition tables**:

  * `ADD_TABLE[a_idx][b_idx]` → precomputed result for each tryte pair under possible carry-ins.
  * Encodes both result tryte and carry pattern.

* **Carry composition tables**:

  * `COMPOSITION_TABLE[x][y]` → function composition of carry maps, used for Kogge–Stone-style prefix computation.

These tables let us:

* Perform addition in O(TRYTES * log TRYTES) with minimal branches.
* Ensure the implementation is both fast and amenable to a clear reference model.

---

## 6. Addition Design

### 6.1 Conceptual algorithm

The limb spec defines addition as exact integer addition over the 48-trit value domain. The implementation is organized as:

1. **Decode trytes to “wide” digit units**
   Each tryte is mapped to a balanced digit in `[-13, +13]`, which is effectively the signed sum of its three trits in base-3.

2. **Precompute per-tryte combine results**
   A table `ADD_TABLE` stores, for each pair of input trytes `(a,b)`:

   * How they transform a carry-in {−1,0,+1}.
   * What result tryte and carry-out they produce.

3. **Parallel carry propagation**
   Construct an array of “carry maps” for each position, then use a Kogge–Stone-style prefix operation, powered by `COMPOSITION_TABLE`, to compute the carry-in at every tryte index in O(log n) steps.

4. **Finalize sums**
   Using the per-position carry-in, look up the final sum tryte for each position, yielding a canonical tryte array.

5. **Re-encode as `limb`**
   Store the resulting trytes back into `trytes_`.

### 6.2 Properties

* Fully specified: given tables and composition rules, the result is deterministic.
* No overflow: addition is closed over the representable 48-trit range.
* Conservative implementation: if SIMD is available, the Kogge–Stone structure can be vectorized, but the scalar path is clear and testable.

---

## 7. Multiplication Design (`mul_wide` and `operator*`)

### 7.1 `mul_wide` as ground truth

`mul_wide(a, b)` is the **reference** wide product:

* Output: `std::pair<limb, limb> { low, high }`
  representing a full 96-trit product split into low and high halves.

The algorithm:

1. **Decode inputs to trits**

   * `a_trits[0..47]`, `b_trits[0..47]` in `{−1,0,+1}`.

2. **Schoolbook convolution**

   * Use `acc[0..95]` as integer accumulators (can be `int`).
   * For all `i,j` in `[0,47]`:

     ```cpp
     acc[i + j] += a_trits[i] * b_trits[j];
     ```

3. **Normalization passes**

   We perform a fixed sequence of passes to bring each `acc[k]` into `{−1,0,+1}` while pushing balanced carries forward/backward:

   * Six left-to-right passes:

     * At each `k = 0..94`, choose a balanced carry `c` so that `acc[k] - 3c ∈ {−1,0,+1}`, then:

       * `acc[k]  -= 3 * c;`
       * `acc[k+1] += c;`
   * One optional right-to-left pass to remove any residual out-of-range digits at the high end.

   This exact pattern (six forward, one backward) defines the reference behavior. Optimized implementations must match the final result.

4. **Clamp and re-encode**

   * For each `k`, clamp `acc[k]` into `{−1,0,+1}` (should already be in range if normalization is correct).
   * First 48 trits → `low` via `from_trits(trits[0..47])`.
   * Last 48 trits → `high` via `from_trits(trits[48..95])`.

### 7.2 `operator*` and overflow

`operator*` and `operator*=` are defined in terms of `mul_wide`:

* Compute `{low, high} = mul_wide(a, b)`.
* If `high` is **not** zero (any non-zero trit), this is considered overflow:

  * `operator*` MUST throw `std::overflow_error`.
* Otherwise:

  * Return `low` as the product.

This gives a clear, testable contract:

* `mul_wide` defines the exact 96-trit product.
* `operator*` is “safe” truncated multiplication with precise overflow signaling.

---

## 8. Other Operations

### 8.1 Division and remainder

`div_mod` is conceptually implemented via:

* A long-division algorithm operating on the integer value represented by the limb.
* Implemented in terms of repeated shifts and subtracts in ternary, or via a conversion to an internal wide integer representation.

Constraints:

* For any non-zero divisor `d`, we must have:

  ```text
  x = q * d + r
  0 <= |r| < |d|
  sign(r) == sign(x) or r == 0
  ```

* Division by zero throws `std::domain_error`.

The specific algorithm can be chosen for performance; the observable behavior is fixed.

### 8.2 Tritwise logical operators

These operators work **per trit**, using the decoded `{−1,0,+1}` digits, and then re-encode:

* `&` → per-trit `min(a_i, b_i)`.
* `|` → per-trit `max(a_i, b_i)`.
* `^` → per-trit “carry-free sum” with local normalization.
* `~` → per-trit negation.
* `consensus` → keep trit only when both operands agree, else 0.

Implementation:

* Most naturally done by decoding to trits, computing each trit, and re-encoding via the LUTs.
* For performance, these can be partially table-driven at the tryte level, but the tritwise semantics are authoritative.

### 8.3 Shifts and “rotations”

Two families:

1. **Tryte shifts** (`<<`, `>>`):

   * Operate in units of 3 trits (1 tryte).
   * For `n > 0`:

     * `<< n` shifts trytes towards more significant positions, zero-filling from the low side.
     * `>> n` shifts trytes towards less significant positions, zero-filling from the high side.
   * For `n >= TRYTES`, result is zero.
   * For `n < 0`, shift is a no-op (design choice to avoid UB).

2. **Trit “rotations”** (`rotate_left_tbits`, `rotate_right_tbits`):

   * Despite the name, these are **zero-filled trit shifts** with specified behavior:

     * Shift by `n` trits left/right.
     * Fill vacated positions with zero trits.
     * Negative `n` is treated as zero.
   * The spec intentionally preserves these semantics for compatibility with existing tests.

Implementation:

* Tryte shifts: operate directly on `trytes_` with memmove/memcpy and LUT zero tryte.
* Trit “rotations”: most simply implemented via trit array buffer and LUT re-encode.

---

## 9. Conversions and I/O

### 9.1 Native integer conversions

To convert `limb → native integer`:

1. Decode to an internal wide integer (e.g. 128-bit) via balanced ternary evaluation.
2. Check bounds for the target type.
3. Either return the cast value or throw `std::overflow_error`.

To convert `native integer → limb`:

1. Perform repeated division by 3 with balanced adjustment so digits stay in `{−1,0,+1}`.
2. Ensure the integer is within `[min(), max()]`.
3. Encode via LUTs.

These conversions must be:

* Deterministic and stable.
* Explicit for all non-`bool` targets.

### 9.2 String conversions and binary form

String I/O:

* `to_string(base)` uses typical integer formatting rules for bases 2..36, plus a sign.
* `from_string(str, base)` parses sign and digits, then constructs a `limb` with explicit overflow checking.

Binary I/O:

* `from_bytes` / `to_bytes` must operate on the canonical 16-byte layout directly on `trytes_`.
* This is the authoritative binary format for `limb` in t81lib.

---

## 10. Invariants and Testing Strategy

### 10.1 Invariants

The implementation maintains:

* Every public `limb` instance always satisfies:

  * All `trytes_[i]` ∈ `[0,26]`.
  * The `trytes_` array is a valid encoding of some `x ∈ [min(), max()]`.
* No operation leaves a `limb` in a non-canonical encoding.

### 10.2 Tests

Recommended test layers:

1. **Unit tests**:

   * Check each operation on known values and edge cases: `min()`, `max()`, `−1`, `0`, `1`.
   * Validate tryte/trit decoding/encoding round-trips.

2. **Property tests** (e.g. RapidCheck):

   * `(a + b) − b == a`.
   * `(a * b)` matches `mul_wide(a, b).first` when `mul_wide(a, b).second == 0`.
   * `div_mod` identities:

     * `x == q * d + r` and the remainder constraints.
   * Serialization:

     * `from_bytes(a.to_bytes()) == a`.

3. **Reference model cross-checks**:

   * Implement a slow, obviously-correct version using big integers or arbitrary long trit arrays.
   * Compare it exhaustively / fuzzily against the production implementation.

4. **Platform variance**:

   * CI matrix across multiple compilers and architectures.
   * Ensure no UB-triggered variance (e.g., signed shifts, overflow, strict aliasing).

---

## 11. Extension Points

The limb design intentionally leaves room for:

* **SIMD acceleration**:

  * Tryte-wise operations (add, mul) can be vectorized using AVX2/AVX-512 or other ISAs.
  * As long as the result is bit-identical to the scalar path, the spec remains satisfied.

* **Alternative internal encodings**:

  * Internally, the implementation can maintain additional cached representations (e.g. decoded trits, binary int128) as long as:

    * `trytes_` is canonical whenever observed externally.
    * All public operations behave as specified.

* **Additional helper APIs**:

  * Non-normative helpers (`to_trits()`, `from_trits()`, debug dumps) can be exposed behind `t81::util` or `detail` for testing and instrumentation.

---

This document is meant to give future contributors enough context to understand **why** `limb` looks the way it does and how to extend or optimize it without breaking the guarantees that the rest of the T81 stack depends on.
