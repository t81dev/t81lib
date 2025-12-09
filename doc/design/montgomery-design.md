# Montgomery-design.md — Montgomery Arithmetic over t81::core Types

**Status:** Informative (non-normative, library-level)  
**Targets:**  
- `t81::core::limb` (48-trit scalar)  
- `t81::core::bigint` (multi-limb arbitrary precision)  
**Related docs:**  
- `t81lib-spec-v1.0.0.md` (library-level spec, limb normative)  
- `limb-design.md` (scalar implementation notes)  
- `bigint-design.md` (multi-limb integer design)

This document describes how Montgomery modular arithmetic is designed on top of the
canonical balanced-ternary types provided by `t81lib`. It is a design and implementation
guide; if it disagrees with the limb spec, the limb spec wins.

Montgomery arithmetic is **not** part of the v1.0.0 core normative surface, but it is a
natural advanced module that higher layers (cryptography, large moduli arithmetic,
VMs) can rely on.

---

## 1. Overview

Montgomery arithmetic provides:

- Efficient modular multiplication `a · b · R⁻¹ mod m` without division,  
- Using a fixed “radix” `R` such that `gcd(m, R) = 1`.

In `t81lib`, the natural choices are:

- For limb-level contexts: `R = 3^48` (one limb base).  
- For bigint-level contexts: `R = 3^(48·k)` for some limb count `k`.

We treat Montgomery as a **context**:

```cpp
namespace t81::core {

template <class Int>
class MontgomeryContext; // Int = limb or bigint

} // namespace t81::core
````

The context holds precomputed constants and exposes:

* Conversion to/from Montgomery domain,
* Fast `montgomery_mul(a, b)` and `montgomery_pow(base, exp)`.

All semantics are defined in terms of the underlying integer type `Int`’s
mathematical value and the limb spec.

---

## 2. Mathematical Model

### 2.1 Base type `Int`

`Int` is either:

* `t81::core::limb` (48-trit integer with fixed width), or
* `t81::core::bigint` (multi-limb integer, base `3^48` internally).

Both types:

* Represent signed integers in balanced ternary.
* Have well-defined addition, subtraction, and multiplication semantics with no UB.

For Montgomery arithmetic we mostly care about **non-negative residues** modulo `m`,
even though the underlying representation is balanced.

### 2.2 Domain parameters

Given a modulus `m` with:

* `m > 0`,
* `gcd(m, 3) = 1` (i.e., `m` is not divisible by 3),

we choose:

* A radix `R` such that:

  * `R` is a power of 3 (`R = 3^t` for some `t`),
  * `R > m`,
  * `gcd(R, m) = 1`.
* Precompute:

  * `R_mod_m = R mod m`,
  * `R2_mod_m = R^2 mod m`,
  * `m'` such that:

    ```text
    m · m' ≡ -1 (mod R)
    ```

This is the Montgomery parameter set.

For limb-level:

* `Int = limb`,
* We fix `R = 3^48` and require `m < R` (i.e. modulus fits in one limb and `m` is
  not divisible by 3).

For bigint-level:

* `Int = bigint`,
* We can choose `R = 3^(48·k)` where `k` is the number of limbs chosen for the
  Montgomery domain (e.g. `k = ceil(bit_length(m) / 48)` plus a safety margin).

### 2.3 Montgomery representation

For an integer `x` with `0 ≤ x < m`, its Montgomery representation is:

```text
x̄ = x · R (mod m)
```

Given `ā` and `b̄` in Montgomery form, we define:

* `montgomery_mul(ā, b̄) = ā · b̄ · R⁻¹ (mod m)`,
* which is equivalent to `((a · R) · (b · R)) · R⁻1 = (a · b · R) (mod m)` in standard form.

Conversion:

* To Montgomery:

  ```text
  to_montgomery(x) = x · R (mod m)
                   = x · R2_mod_m · R⁻1 (mod m)
  ```

  Implementation-friendly form:

  ```cpp
  x_bar = REDC(x * R2_mod_m);  // see §3
  ```

* From Montgomery:

  ```text
  from_montgomery(x̄) = x̄ · R⁻1 (mod m)
                      = REDC(x̄)  // effectively removes one factor of R
  ```

---

## 3. REDC Operation (Montgomery Reduction)

### 3.1 REDC definition

`REDC(T)` is defined for `T` in the range `0 ≤ T < m·R` (limb case) or
`0 ≤ T < m·R^k` (bigint case), and returns:

```text
REDC(T) = T · R⁻1 (mod m)
```

in the range `0 ≤ REDC(T) < m`.

We implement REDC using the standard Montgomery trick:

1. Compute:

   ```text
   u = (T mod R) · m' mod R
   ```

2. Compute:

   ```text
   t = (T + u · m) / R
   ```

3. If `t ≥ m`, return `t − m`, else return `t`.

This uses only:

* Mod `R` (radix operations),
* Division by `R` (simple “drop the low limb(s)”),
* Addition and multiplication modulo `m`.

### 3.2 Limb-level REDC

For `Int = limb` and `R = 3^48`:

* `R` is the numeric range of `limb`, but we only need:

  * `T` as a double-width value of up to `m · R` (~96 trits),
  * `T mod R` is just the **low-limb** of the wide product,
  * `T / R` is the **high-limb**.

We can use `mul_wide` as per `limb-design.md`:

* `mul_wide(a, b)` → `{low, high}` representing:

  ```text
  a * b = low + high · R
  ```

For REDC, we treat `T` as a 2-limb value `(T_low, T_high)`:

1. Compute `u`:

   ```text
   // T_low is limb, m_prime is limb with special property:
   //     m * m_prime ≡ -1 (mod R)
   u = (T_low * m_prime) mod R
   ```

   In practice, since `u` only needs to be computed modulo `R`, we can:

   * Use `mul_wide(T_low, m_prime)` → `{u_low, u_high}`, then let `u = u_low`,
     since `u_low` already represents the result mod `R` (ignoring overflow in
     `u_high`), or
   * Implement a dedicated `mul_mod_R` if necessary.

2. Compute `T + u · m` as a 2-limb value:

   ```text
   (U_low, U_high) = (T_low, T_high) + u * m
   ```

3. Divide by `R`:

   * `t = U_high` (dropping the low limb).

4. If `t ≥ m`, subtract `m`, else keep `t`.

`REDC` returns a single limb `t` in `[0, m)`.

### 3.3 Bigint-level REDC

For `Int = bigint`, `R = 3^(48·k)`:

* `T` is a multi-limb integer with up to `2k` limbs plus headroom.
* `T mod R` = the lower `k` limbs.
* `T / R` = the upper limbs (drop the lowest `k` limbs).

We generalize:

1. Compute:

   ```text
   u = (T mod R) · m' mod R
   ```

   Where:

   * `T mod R` is a `bigint` formed from the lowest `k` limbs,
   * `m'` is also a `bigint` with at most `k` limbs.

2. Compute `T + u · m` (bigint multiplication & addition).

3. Drop the lowest `k` limbs to divide by `R`.

4. If resulting `t ≥ m`, subtract `m`, else keep `t`.

Implementation detail:

* For bigints, we likely store `R` implicitly as “`k` limbs worth of base”.
* REDC becomes a base-`B` algorithm where `B = 3^48` and we manipulate limb vectors.

---

## 4. Montgomery Context API

### 4.1 Template

We use a templated context:

```cpp
namespace t81::core {

template <class Int>
class MontgomeryContext {
public:
    using int_type = Int;

    // Construct with modulus m (Int), throws if invalid (e.g. m <= 0 or divisible by 3).
    explicit MontgomeryContext(const Int& modulus);

    const Int& modulus() const noexcept;

    // Conversion
    Int to_montgomery(const Int& x) const;      // x̄ = x * R mod m
    Int from_montgomery(const Int& x_bar) const;// x = x̄ * R⁻1 mod m

    // Core operation
    Int mul(const Int& a_bar, const Int& b_bar) const;   // Montgomery multiply
    Int square(const Int& a_bar) const;                  // mul(a_bar, a_bar)

    // Exponentiation
    Int pow(const Int& base_bar, const Int& exp) const;  // base_bar^exp (Montgomery domain)

private:
    Int m_;              // modulus
    Int R_mod_m_;        // R mod m
    Int R2_mod_m_;       // R^2 mod m
    Int m_prime_;        // m' such that m * m' ≡ -1 (mod R)
    // internal helper for REDC(T)
    Int redc(const Int& T_low, const Int& T_high) const; // limb case
    // or, for bigint: Int redc(const Int& T) const;
};

} // namespace t81::core
```

Notes:

* For `limb` context, `redc` takes a wide product represented as `(low, high)`.
* For `bigint` context, `redc` takes a large `Int` representing `T`.

### 4.2 Usage pattern

Example (limb):

```cpp
using t81::core::limb;
using t81::core::MontgomeryContext;

limb m = /* some odd (mod 3) positive modulus < 3^48 */;
MontgomeryContext<limb> ctx(m);

limb a = /* 0 <= a < m */;
limb b = /* 0 <= b < m */;

limb a_bar = ctx.to_montgomery(a);
limb b_bar = ctx.to_montgomery(b);

limb c_bar = ctx.mul(a_bar, b_bar);
limb c     = ctx.from_montgomery(c_bar); // c = a * b mod m
```

Same logic for `bigint`, just with bigger numbers.

---

## 5. Balanced Ternary vs Standard Montgomery

Montgomery is usually described in binary (radix `2^w`). Our environment is balanced
ternary, base 3:

* Radix: `R = 3^t` rather than `2^w`.
* Base integer arithmetic is now in balanced ternary, but the **Montgomery algebra**
  stays the same:

  * `m · m' ≡ -1 (mod R)` still defines the reduction parameter.
  * `REDC(T) = (T + u·m) / R` is unchanged structurally.

Implications:

* All modular arithmetic is still on ordinary integers; balanced ternary is just the
  **representation** chosen for `Int`.
* As long as `limb` and `bigint` implement exact integer arithmetic, Montgomery
  arithmetic is simply a composition of those operations.

Balanced ternary only enters via:

* The guarantee that `gcd(m, 3) = 1` (we must avoid moduli divisible by 3),
* The form of `R` (power of 3 in place of power of 2),
* Performance characteristics of limb operations.

---

## 6. Error Handling and Preconditions

### 6.1 Context construction

`MontgomeryContext<Int>::MontgomeryContext(const Int& m)` MUST:

* Throw (e.g. `std::invalid_argument`) if:

  * `m <= 0`,
  * `m` is divisible by 3 (`m mod 3 == 0`),
  * In the limb case, `m >= R` (i.e., not strictly less than the limb base).

It MUST:

* Compute `R_mod_m`, `R2_mod_m`, and `m_prime_` deterministically.
* Ensure that all precomputed values are canonical `Int` values.

### 6.2 Operations

`to_montgomery` and `from_montgomery`:

* Expect `0 ≤ x < m` (or treat larger values as reduced modulo `m` if we choose
  to make them forgiving; this is a documented design choice).
* If they internally rely on base operations that can throw (e.g., `div_mod`,
  arithmetic overflow), such exceptions propagate.

`mul`:

* Expects both inputs already in Montgomery form.
* Returns a Montgomery-form result in `[0, m)`.

`pow`:

* Typical *square-and-multiply* or *square-and-multiply-always* in Montgomery domain.
* Negative exponents should be rejected (e.g. `std::domain_error`) unless we add
  a separate modular inverse API.

---

## 7. Determinism and Constant-Time Concerns

`t81lib` as a whole is determinism-oriented. For Montgomery arithmetic:

* Semantics are deterministic by construction (no UB, no unspecified behavior).
* Reference implementations should avoid data-dependent undefined behavior.

Constant-time behavior (side-channel resistance):

* **Out of scope** for this document, but relevant for cryptographic contexts.
* If we add a “constant-time” variant, it should:

  * Avoid secret-dependent branches and memory accesses,
  * Use fixed-latency operations as much as possible.

The design allows both:

* A straightforward reference (non-constant-time) implementation,
* A constant-time hardened implementation with the same public API.

---

## 8. Testing Strategy

Montgomery code sits on top of `limb` and `bigint`, so the testing approach is layered:

### 8.1 Correctness against slow reference

Implement a slow reference `mod_mul` and `mod_pow` based on:

* Direct big-int arithmetic (`bigint` or a test-only arbitrary precision type),
* `a * b % m` using division / remainder from the core library.

For random `a, b, m`:

* Check:

  ```text
  from_montgomery(mul(to_montgomery(a), to_montgomery(b)))
      == (a * b) mod m
  ```

* And for exponentiation:

  ```text
  from_montgomery(pow(to_montgomery(a), e))
      == (a^e) mod m
  ```

### 8.2 Edge cases

Test:

* Small moduli (e.g., `m = 2`, `m = 4`, `m = 5` but not divisible by 3),
* Moduli close to `R` (in limb case),
* `a = 0`, `a = 1`, `a = m-1`,
* Large random exponents (including 0).

### 8.3 Invariants

* `to_montgomery(0) == 0` and `from_montgomery(0) == 0`.
* `to_montgomery(1)` acts as multiplicative identity in Montgomery domain.
* `mul(x̄, to_montgomery(1)) == x̄`.

---

## 9. Extension Points

The Montgomery design here is deliberately minimal and context-driven. Natural
extensions include:

* **Montgomery contexts over `bigint`** with:

  * Precomputed exponentiation windows,
  * Precomputed `n`-limb `R` choices tuned for fixed modulus sizes.

* **Modular operations:**

  * `mod_add`, `mod_sub`, `mod_neg`, `mod_inv` built on the same context,
  * APIs for “raw” reduction (`reduce(x) = x mod m`).

* **Integration with higher layers:**

  * Cryptographic primitives that use `MontgomeryContext<bigint>` for large
    prime moduli,
  * VM instructions for modular arithmetic expressed in terms of Montgomery
    operations.

All such extended APIs MUST treat the canonical `Int` representations, and the core
Montgomery algebra described above, as the source of truth.

---

This document should give contributors enough structure to implement and optimize
Montgomery arithmetic in `t81lib` without undermining the canonical `limb`/`bigint`
semantics or the library’s determinism goals.