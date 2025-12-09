# bigint-design.md — t81::core::bigint Design Notes

**Status:** Informative (non-normative, library-level)  
**Target:** `t81::core::bigint` in `include/t81/core/bigint.hpp`  
**Related specs:**  
- `t81lib-spec-v1.0.0.md` (library-level, limb is normative)  
- `limb-design.md` (scalar implementation notes)

This document explains how the arbitrary-precision balanced-ternary integer type
`t81::core::bigint` is implemented in terms of the canonical scalar type
`t81::core::limb`. It is an implementation and architecture guide, not a formal spec;
if it disagrees with the limb spec, the limb spec wins.

`bigint` is allowed to evolve more than `limb` but must always respect limb’s
semantics and canonical encoding.

---

## 1. Role in the Library

`t81::core::bigint` is the “multi-limb” counterpart to `t81::core::limb`:

- Provides arbitrary-precision signed integers in balanced ternary.
- Built as a sequence of canonical `limb` values.
- Serves higher layers: big-number algorithms, cryptographic-style arithmetic,
  VM word types that exceed 48 trits, etc.
- Uses `limb`’s canonical encoding for all cross-boundary serialization and hashing.

The mental model: **`bigint` is to `limb` what `std::bigint` (if it existed) would be to
`std::int128_t`, except the base is 3 instead of 2.**

---

## 2. Design Objectives

Key goals for `bigint`:

1. **Limb-centric design**  
   - `bigint` is defined as a normalized sign + vector of `limb`s.
   - No alternate scalar encoding is allowed; limbs must always be canonical.

2. **Predictable normalization**  
   - Unique, canonical representation for each mathematical integer.
   - No leading zero limbs (except a single zero-limb for the value 0).

3. **Semantic alignment with limb**  
   - Arithmetic operations reduce to repeated limb operations, respecting the same
     overflow-free semantics and error handling.
   - Conversions to/from `limb` and native integers are consistent.

4. **Separation of concerns**  
   - The limb spec carries the heavy “normative” burden.
   - `bigint` is a library module built on top; it can adopt faster algorithms
     (Karatsuba, etc.) without touching limb’s guarantees.

5. **Extensibility**  
   - Easy to add more algorithms (GCD, modular inverse, Montgomery contexts)
     without changing the fundamental representation.

---

## 3. Representation

### 3.1 Mathematical model

A `t81::core::bigint` represents a signed integer in ℤ via:

```text
value = sign * ∑_{k=0}^{n-1} L_k · B^k
````

where:

* `sign ∈ {−1, 0, +1}` (with 0 only if the number is zero),
* `L_k` are `limb` values (each 48-trit balanced ternary),
* `B` is the limb base, i.e. `B = 3^48` (the numeric range of a limb),
* `n` is the number of limbs.

In other words: `bigint` is a base-`3^48` expansion of the integer in canonical limbs.

### 3.2 Concrete C++ layout

A typical layout:

```cpp
namespace t81::core {

class bigint {
public:
    // Constructors, operators, etc.

private:
    std::vector<limb> limbs_; // little-endian: limbs_[0] is least significant.
    bool negative_ = false;
};

} // namespace t81::core
```

Conventions:

* **Limb order:** little-endian in limbs:

  * `limbs_[0]` is the least significant `limb`.
  * `limbs_.back()` is the most significant.

* **Sign flag:**

  * `negative_ == false` and `limbs_.empty()` → the value **0**.
  * `negative_ == false` and `limbs_` non-empty → positive.
  * `negative_ == true` and `limbs_` non-empty → negative.

* **Zero representation:**

  * Preferred normalized form for zero is:

    * `negative_ == false`
    * `limbs_.empty()` (or optionally one zero limb; pick one canonical rule and
      document it).

### 3.3 Normalization invariants

`bigint` MUST maintain the following invariants after any public operation:

1. **No leading zero limbs**

   * `limbs_.empty()` OR `limbs_.back()` MUST be nonzero (`!= limb::zero()`).

2. **Zero sign convention**

   * If `limbs_.empty()`, then `negative_` MUST be `false`.
   * The representation `(negative_ == true, limbs_.empty())` is forbidden.

3. **Canonical limb encoding**

   * Every `limb` in `limbs_` MUST be in canonical form according to the limb spec
     (valid trytes, correct trits, etc.).

A private `normalize()` helper is responsible for enforcing these invariants after
mutating operations.

---

## 4. Public Interface (Conceptual)

The specific header can grow, but a minimal `bigint` surface looks like:

* Construction:

  * Default (zero).
  * From `limb`.
  * From native integers (`intmax_t`, `uintmax_t`).
  * From string (`from_string` style helper or constructor).

* Observers:

  * `is_zero()`, `is_negative()`, `signum()`.
  * `limb_count()`, `limb_at(k)` (read-only).

* Arithmetic:

  * `+`, `-`, unary `-`, `*`, `/`, `%`, `div_mod`.

* Comparison:

  * `compare(const bigint&)`, full relational operators, `operator<=>`.

* Conversions:

  * To/from `limb` (when in range).
  * To/from native integers (with overflow checks).
  * String formats (shared semantics with `limb`’s `to_string` / `from_string`).

* Possibly:

  * GCD, modular inverse, `pow_mod`, etc., which are natural on bigints.

All of these MUST be defined in terms of the mathematical model in §3.1 and the
limb operations described in the limb spec.

---

## 5. Core Algorithms

### 5.1 Normalization (`normalize()`)

`normalize()` is called after any operation that mutates `limbs_`:

1. Remove leading zero limbs:

   ```cpp
   while (!limbs_.empty() && limbs_.back().is_zero()) {
       limbs_.pop_back();
   }
   ```

2. Fix sign for zero:

   ```cpp
   if (limbs_.empty()) {
       negative_ = false;
   }
   ```

This keeps representations canonical and comparison logic simple.

---

### 5.2 Comparison

Comparison is implemented in three levels:

1. **Sign comparison**

   * If `signum(lhs) != signum(rhs)`, sign decides (`−` < `+`).

2. **Magnitude comparison:**

   * Compare `limbs_.size()` first; longer magnitude is larger.
   * If sizes equal, compare limbs from most-significant (`back()`) down to index 0
     using `limb::compare`.

3. **Combine:**

   * If both are non-negative, magnitude comparison is final.
   * If both are negative, reversed magnitude comparison (more negative is smaller).

This logic mirrors typical big-int designs but uses `limb::compare` as the base primitive.

---

### 5.3 Addition and subtraction

We implement **unsigned magnitude** add/sub, then handle signs.

#### 5.3.1 Magnitude addition

Given `|a|` and `|b|` as positive `bigint` magnitudes:

1. Ensure `|a|` has at least as many limbs as `|b|` (swap if needed).
2. Loop over limb indices:

   ```cpp
   limb carry = limb::zero();
   for (std::size_t i = 0; i < max_len; ++i) {
       limb ai = (i < a.size()) ? a[i] : limb::zero();
       limb bi = (i < b.size()) ? b[i] : limb::zero();
       limb sum = ai + bi + carry; // where "carry" is modeled as a limb with small value
       // update carry based on overflow in sum, if you model carry in limb-space
   }
   ```

However, direct “limb + limb + carry-limb” is not ideal; more typical is:

* Work with a `limb` base representation and use `mul_wide` and `div_mod` to extract carry,
  or
* Use a wider intermediate representation (e.g. temporary 96-trit wide accumulation)
  and then split into limb + carry.

The **conceptual** model:

```text
result_value = |a| + |b|
```

represented in base `B = 3^48`. The implementation need not explicitly compute in base `B`,
but must produce the same result as if it had.

The actual implementation can:

* Convert each limb to an internal signed wide integer,
* Do base-`B` addition with a scalar carry,
* Convert back to canonical `limb`s.

This is a design choice; the key invariant is that each resulting limb is canonical and
the overall value is correct.

#### 5.3.2 Magnitude subtraction

For `|a| >= |b|`, compute `|a| − |b|`:

* Similar to addition but with borrow propagation.
* Again, may be implemented either:

  * implicitly in balanced ternary via limb operations, or
  * via an internal wide integer per limb with explicit borrow.

Normalization ensures no leading zero limbs after the operation.

#### 5.3.3 Signed add/sub wrapper

Define:

* `a + b`:

  * If `sign(a) == sign(b)`, result sign = that sign, magnitude = `|a| + |b|`.
  * If `sign(a) != sign(b)`, subtract the smaller magnitude from the larger:

    * If `|a| >= |b|`, result sign = `sign(a)`, magnitude = `|a| − |b|`.
    * Else, sign = `sign(b)`, magnitude = `|b| − |a|`.

* `a − b`:

  * Implement as `a + (-b)`.

---

### 5.4 Multiplication

`bigint` multiplication is based on limb multiplication and the chosen wide product model.

#### 5.4.1 Schoolbook multiplication

Reference algorithm:

1. Let `n = a.limbs_.size()`, `m = b.limbs_.size()`.
2. Allocate `result` with `n + m` limbs initialized to zero.
3. For each `i = 0..n-1` and `j = 0..m-1`:

   * Compute `limb_ij = a.limbs_[i] * b.limbs_[j]`.

     * Use `mul_wide` if you want the full precision and manage carries at the limb level,
       or
     * Use the scalar `*` with overflow checks if you know the product cannot overflow.

   * Accumulate into `result.limbs_[i + j]` with proper carry handling.

Because `limb::operator*` throws on overflow, **it is recommended** that `bigint`
implement multiplication in terms of `mul_wide`:

* Interpret `mul_wide(a_i, b_j)` as:

  ```text
  a_i * b_j = low_ij + high_ij * B
  ```

* Add `low_ij` to `result_limb[i + j]` and `high_ij` to `result_limb[i + j + 1]`,
  propagating carries.

The conceptual base is `B = 3^48`, but internally you can work with limb values and a
small carry window.

#### 5.4.2 Sign handling

* `sign(result) = sign(a) * sign(b)` (with 0 handled separately).
* After computing magnitude, call `normalize()`.

#### 5.4.3 Optional advanced algorithms

The design allows pluggable algorithms:

* For small sizes: schoolbook multiplication.
* For larger sizes: Karatsuba, Toom-Cook, FFT-based NTTs (all in base `B` via limbs).

The only constraint: **the result must be mathematically correct** and expressed as a
canonical `bigint` (normalized limbs + sign).

---

### 5.5 Division and `div_mod`

Division is implemented via **unsigned magnitude long division** in base `B`, then wrapped
for signs.

High-level approach:

1. Work on magnitudes `|a|` and `|b|` with `|b| > 0`.

2. Use a standard base-`B` long division algorithm:

   * Normalize divisor and dividend if needed (e.g. scale to make the leading limb
     “large” for quotient estimation).
   * At each step, estimate the next quotient limb using the top one–two limbs of the
     dividend and divisor.
   * Correct via trial multiplication and subtraction.

3. Produce quotient `q` and remainder `r` in magnitude form.

4. Restore sign:

   * Sign of `q` = `sign(a) * sign(b)`.
   * Remainder `r`:

     * Same sign as `a`, or zero (as per limb spec semantics).

5. Normalize both `q` and `r`.

Division by zero must throw `std::domain_error`.

`/`, `%`, `/=`, `%=` are defined in terms of `div_mod`.

---

## 6. Conversions

### 6.1 To/from limb

* `bigint(limb x)`:

  * If `x.is_zero()`, set `negative_ = false`, `limbs_.clear()`.
  * Else, `negative_ = x.is_negative()`, `limbs_ = { abs(x) }`.

* `explicit operator limb() const`:

  * If the value fits in one limb (i.e., `limbs_.size() <= 1` and magnitude ≤ `limb::max()`),
    convert:

    * For zero: `return limb::zero()`.
    * For one limb: apply sign to the single limb and ensure no overflow.
  * Otherwise, throw `std::overflow_error`.

### 6.2 To/from native integers

Conversions to/from native integer types mirror the limb logic:

* For `bigint → intmax_t`:

  * If too many limbs or magnitude exceeds `intmax_t` range, throw.
  * Otherwise, accumulate in a wide integer and cast as in limb.

* For `intmax_t → bigint`:

  * Decompose the integer into base `B = 3^48` digits, each represented as a `limb`.
  * Apply sign.

Conversions to/from strings can reuse the same algorithm as limb, but on arbitrary-precision integers:

* Repeated division by `base` to extract digits in the requested radix.

---

## 7. Serialization and Hashing

### 7.1 Binary serialization

A reasonable canonical binary format for `bigint` (not mandated by limb spec but
compatible with it):

* Sign byte: `{0, 1, 2}` representing `{0, positive, negative}`.
* Limb count `n` (e.g., variable-length encoding or fixed-width).
* `n` limbs in **little-endian** order, each serialized via `limb::to_bytes()`.

This ensures that:

* Binary encoding is stable and round-trippable.
* `limb` encoding remains the atomic building block.

### 7.2 Hashing

A canonical hash can be defined as:

* Hash of `(sign, limb_count, canonical limb bytes)`,
* Or equivalently, hashing the canonical binary serialization.

The key requirement: hashing must depend only on canonical limb encodings and the sign,
not on capacity or allocation details.

---

## 8. Testing Strategy

`bigint` testing should mirror limb testing but with more focus on cross-layer interactions.

1. **Unit tests**

   * Construction from `limb`, from native ints, from strings.
   * Arithmetic on small values that fit within one limb, cross-checked directly with limb.
   * Edge cases around limb boundaries (just above/below `limb::max()` or `limb::min()`).

2. **Property tests**

   * Verify ring-like identities:

     * `(a + b) − b == a`.
     * `(a * b) / b == a` when `b` divides `a` exactly.
   * `div_mod` identities.
   * Consistency with `limb`: when the result fits in a limb, both paths agree.

3. **Reference model**

   * A slow but simple reference `bigint` implemented via:

     * Large binary integer (e.g. built on `boost::multiprecision::cpp_int`, or an internal test-only type), or
     * Arbitrary-length ternary digit arrays.
   * Use this for fuzzing and cross-checking against production `bigint`.

4. **Interplay with limb**

   * Ensure that every operation composed from `limb` primitives preserves limb invariants.
   * Verify that `bigint` never constructs a non-canonical `limb`.

---

## 9. Extension Points

Planned or possible future extensions:

* **Multi-precision algorithms**:

  * Karatsuba, Toom-Cook, FFT-style multiplication in base `3^48`.
  * Modular arithmetic contexts (Montgomery, Barrett) using `bigint`.

* **Specialized formats**:

  * Compressed encodings for sparse bigints.
  * Fixed-width `N`-limb types (e.g. 96-trit, 192-trit) as thin wrappers over `bigint`.

* **Higher-level protocols**:

  * VM value types (e.g. TISC registers) implemented as `bigint` or fixed-size
    wrappers around it.
  * Cryptographic primitives using `bigint` as the base integer type.

All such extensions must preserve the core constraints:

* `bigint` is a canonical sign + sequence of canonical limbs.
* `limb` remains the only scalar with a stable binary representation and normative spec.

---

This design document should give contributors enough context to implement or optimize
`t81::core::bigint` while keeping it firmly anchored to the `limb` semantics and to the
overall goals of `t81lib`.
