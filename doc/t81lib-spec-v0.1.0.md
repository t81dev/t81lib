<!--
doc/t81lib-spec-v0.1.0.md — Specification of the t81lib API at version 0.1.0.
-->

# t81lib v0.1.0 — Balanced Ternary Core Library

## 0. Status and Notation

* **Status:** v0.1.0 FINAL (normative)
* **Library name:** `t81lib`
* **Umbrella header:** `<t81/t81lib.hpp>`
* **Primary scalar type:** `t81::core::limb` (48-trit balanced-ternary limb)
* **Additional modules:** `t81::core::bigint`, `t81::io`, `t81::util` (see §3)

Key words **MUST**, **MUST NOT**, **SHOULD**, **MAY** are to be interpreted as in RFC 2119.

This document defines:

1. The **overall structure and guarantees** of the `t81lib` balanced-ternary core library.
2. The **normative semantics** for the canonical fixed-width scalar type `t81::core::limb`.
3. The **intended role and stability level** of higher-level modules (`bigint`, I/O, utilities).

The limb semantics are intended to be ABI- and semantically-stable for ≥10 years. Other modules can evolve more freely but MUST remain limb-compatible.

---

## 1. Scope and Goals of t81lib

### 1.1 Scope

`t81lib` is a C++ library providing a minimal, deterministic core for balanced-ternary computation:

* A fixed-width, signed, balanced-ternary scalar type `t81::core::limb`.
* Optional multi-limb arbitrary-precision integer type `t81::core::bigint`.
* String and binary I/O helpers in `t81::io`.
* Utility facilities (random generation, debugging) in `t81::util`.

The core design principles:

* **Canonical representation:** unique encoding per mathematical value.
* **Deterministic semantics:** no hidden UB, no implementation-defined overflow.
* **Interoperability:** stable binary format for scalar limbs, suitable for hashing, storage, and cross-process communication.
* **Layering:** higher-level constructs are explicitly built on `t81::core::limb`.

### 1.2 Non-goals

The following are explicitly **out of scope** for this v1.0.0 core library spec:

* Alternative limb widths (27, 54, 81 trits, etc.) as first-class, stabilized types.
* Advanced multipliers (Karatsuba, Toom-Cook, FFT/NTT) and full cryptographic frameworks.
* SIMD APIs as part of the public interface (AVX2/AVX-512/NEON/SVE).
* Language bindings and FFI layers (Python, Rust, etc.).
* VM integration, tensor libraries, or AI-specific APIs.

These features MAY appear in other layers of the T81 ecosystem, or as experimental/extension headers within this repo, but are not part of the normative `t81lib` v1.0.0 core.

---

## 2. Library Structure and Namespaces

### 2.1 Header layout

Public headers:

```text
include/
└── t81/
    ├── t81lib.hpp          # umbrella
    ├── core/
    │   ├── limb.hpp        # canonical 48-trit scalar (normative)
    │   ├── bigint.hpp      # multi-limb integer (provisional)
    │   └── traits.hpp      # type traits, tagging
    ├── io/
    │   ├── format.hpp      # string/stream formatting helpers
    │   └── parse.hpp       # parsing helpers
    └── util/
        ├── random.hpp      # test-oriented RNG helpers
        └── debug.hpp       # debug dumps and invariants
```

The canonical entry point:

```cpp
#include <t81/t81lib.hpp>
```

In the umbrella header, the library MUST re-export:

```cpp
namespace t81 {

inline constexpr int T81LIB_VERSION_MAJOR = 1;
inline constexpr int T81LIB_VERSION_MINOR = 0;
inline constexpr int T81LIB_VERSION_PATCH = 0;

} // namespace t81
```

and include the key modules:

```cpp
#include <t81/core/limb.hpp>
#include <t81/core/bigint.hpp>   // may be provisional but exists
#include <t81/io/format.hpp>
#include <t81/io/parse.hpp>
#include <t81/util/random.hpp>
#include <t81/util/debug.hpp>
```

### 2.2 Namespaces and stability levels

* `t81::core::limb`

  * **Status:** Normative, long-term stable.
  * This spec (Part I) is binding.

* `t81::core::bigint`

  * **Status:** Library-level module, designed to be stable but allowed to evolve.
  * Must be **logically defined as a sequence of `limb` values** and use limb’s canonical encoding/trits.

* `t81::io::*`

  * **Status:** Convenience; semantics MUST respect limb/bigint specs but formatting details MAY evolve (e.g., additional helper functions, new syntaxes).

* `t81::util::*`

  * **Status:** Utility; provided for testing, fuzzing, and debugging, not a hard ABI surface.

Any symbol in a `detail` namespace or `detail/` header is internal and may change without notice.

---

## 3. Numeric Model

### 3.1 Balanced ternary and canonical limb

The entire library is based on a single scalar model:

* **Base:** 3, with digits (trits) in `{−1, 0, +1}`.
* **Canonical scalar:** `t81::core::limb` with **48 trits** of precision.
* **Tryte packing:** 3 trits per tryte, 16 trytes per limb.
* **Binary serialization:** 16-byte canonical form.

All higher-level constructs:

* MUST be defined in terms of `limb`’s mathematical value and canonical representation.
* MUST NOT introduce alternate encodings for the same scalar mathematical value.

`bigint`, for example, is conceptually a sign plus a sequence of limbs with normalization rules. Its exact API may grow, but its semantics are subordinate to the limb spec.

---

## 4. Part I — `t81::core::limb` (Normative Scalar Spec)

This section is a library-scoped restatement of the limb spec. It is normative for all code that uses `t81lib`.

### 4.1 Value domain

`t81::core::limb` models integers in the range:

* **Trit width:** `TRITS = 48`

* **Value set:**

  ```text
  x = ∑_{i=0}^{47} d_i · 3^i,  where each d_i ∈ {−1, 0, +1}
  ```

* **Range:**

  ```text
  min() = −(3^48 − 1)/2
  max() = +(3^48 − 1)/2
  ```

`min()` and `max()` MUST be provided as `constexpr`.

### 4.2 Storage layout and canonical encoding

`limb` MUST be implemented as:

```cpp
namespace t81::core {

class limb {
public:
    static constexpr int TRITS  = 48;
    static constexpr int TRYTES = 16;
    static constexpr int BYTES  = 16;

    using tryte_t = std::uint8_t; // 0..26

private:
    alignas(16) tryte_t trytes_[TRYTES];
};

} // namespace t81::core
```

Requirements:

1. **Little-endian tryte order**

   * `trytes_[0]` encodes trits 0..2 (LSB).
   * `trytes_[15]` encodes trits 45..47 (MSB).

2. **Tryte range**

   * Each `trytes_[i]` MUST satisfy `0 <= trytes_[i] <= 26`.

3. **Canonical encoding**

   * Exactly one valid encoding per mathematical value.
   * No redundant encodings.

### 4.3 Tryte ↔ trit mapping and LUTs

`limb` relies on a fixed mapping:

* `tryte_t` ∈ `{0..26}` ↔ triple of trits `(t0, t1, t2)` ∈ `{−1,0,+1}³`.

The mapping is defined via:

```cpp
namespace t81::core::detail {

extern const std::array<std::array<std::int8_t, 3>, 27> TRYTE_TO_TRITS;
// plus the inverse mapping, conceptually TRITS_TO_TRYTE

} // namespace t81::core::detail
```

All conformant implementations MUST use the exact same mapping as this reference table and its inverse, even if implemented with different code.

This mapping underpins:

* `from_bytes` / `to_bytes`
* `get_trit` / `set_trit`
* All arithmetic and I/O semantics.

### 4.4 Public API surface for `limb`

At minimum, a conformant implementation MUST provide the following members (or equivalent free functions) with the same semantics:

* Construction, constants, introspection (`zero`, `one`, `min`, `max`, `is_zero`, `signum`, etc.).
* Trit and tryte accessors.
* Arithmetic: `+`, `-`, unary `-`, `*`, `/`, `%`, `div_mod`, `pow_mod`, with no UB and well-defined exceptions.
* Comparison: `compare`, `==`, relational operators, `operator<=>`.
* Tritwise logical operators: `&`, `|`, `^`, `~`, and `consensus`.
* Shift and “rotation” operations on trytes/trits with the specified semantics.
* Native conversions to/from integer and floating-point types with precise overflow/rounding rules.
* String conversions (`to_string`, `from_string`) and binary serialization (`from_bytes`, `to_bytes`).
* Optional utility such as `bit_width()` with the defined semantics.

The detailed semantics, including exception behavior, normalization, overflow conditions, and canonical serialization, are as in the prior limb-specific spec you provided and SHOULD be preserved verbatim within your document as subsections (3.1–3.9, 4–6, etc.).

### 4.5 Hashing and equality

Within the library context:

* Equality for `limb` is equivalent to bytewise equality of the canonical 16-byte encoding.
* Any “canonical” hash MUST be a pure function of those 16 bytes and stable across builds and platforms.

---

## 5. Part II — `t81::core::bigint` (Built on `limb`)

This section defines how `bigint` relates to `limb`. It is **library-level** rather than “limb-level” normative; its surface MAY grow, but certain invariants MUST hold.

### 5.1 Representation constraints

A `t81::core::bigint` implementation:

* MUST conceptually be a sign plus a sequence of `limb` values.

* MUST define a canonical normalization, e.g.:

  * No leading zero limbs, except that zero is represented as a single zero limb and a non-negative sign.

* MUST define addition, subtraction, multiplication, division, and modulus in terms of the exact mathematical integers represented by the limb sequence.

The exact class layout is not mandated, but any exported binary format for bigints SHOULD reference:

* A count of limbs.
* A sequence of canonical `limb` encodings.

### 5.2 Compatibility with limb semantics

All `bigint` operations:

* MUST be observationally equivalent to computing on the underlying integers that each `limb` represents.
* MUST NOT introduce alternate encodings for the same integer once normalized.
* MUST treat `limb`’s serialization format as stable.

That is, if two `bigint` values compare equal, they MUST have canonical limb sequences whose per-limb canonical encodings are bit-identical (ignoring internal allocation or padding).

---

## 6. Part III — I/O and Utility Modules

### 6.1 `t81::io` module

I/O helpers are not themselves a scalar spec, but they have requirements:

* Formatting of `limb` and `bigint` MUST reflect their mathematical value, not internal padding or non-canonical representations.
* Any “binary” I/O helper for `limb` MUST be equivalent to `to_bytes` / `from_bytes`.
* Future extended ternary formats (e.g., custom notations) MUST NOT alter basic `from_string(str, base)` semantics.

### 6.2 `t81::util` module

Utility functions (random generators, debug dumps, invariant checks):

* MUST treat `limb` and `bigint` as opaque values that obey the canonical encoding.
* MAY rely on internal details for debugging, but MUST NOT advertise these details as stable APIs.

Example:

* A `random_limb` helper MAY be provided for tests/fuzzing. It MUST produce valid, canonical `limb` values but is not constrained in distribution or seed behavior beyond what the implementation documents.

---

## 7. Platform Extensions and Optimizations

`t81lib` MAY provide platform-specific accelerated implementations (e.g., specialized `limb` add/mul via AVX2, or batched operations):

* These are implementation details.
* They MUST be **bit-identical** to the scalar reference semantics of `limb` and, transitively, to all operations defined in higher-level modules.

No extension MAY alter:

* The canonical limb encoding.
* The semantics of any limb operation described in Part I.
* The logical relation between `bigint` and `limb`.

---

## 8. Rationale (Informative)

* **Limb as the anchor:**
  The entire library hangs off a single, well-specified scalar, `t81::core::limb`. By freezing its semantics and representation, every higher layer (bigint, tensors, VMs, network protocols) has a solid foundation.

* **Library, not just a type:**
  `t81lib` is a coherent package: it provides a numeric core (`limb` and `bigint`), I/O forms, and utilities, with a single umbrella header and clear structure.

* **Long-term stability:**
  By separating the **normative scalar** (Part I) from the **library modules** (Parts II–III), the project can evolve I/O and bigint capabilities without perturbing the core ABI and semantics.

* **Determinism and reproducibility:**
  Canonical encodings, stable hashes, and deterministic behavior underlie the larger T81 ecosystem’s goals: reproducible computations, verifiable pipelines, and robust cross-implementation interoperability.

---
