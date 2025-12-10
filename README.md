# t81lib — Balanced Ternary Core Library

`t81lib` is a C++20 library for deterministic, **balanced ternary** integer arithmetic.

At its core is `t81::core::limb`, a canonical 48-trit scalar with a **stable 16-byte
encoding**. Higher layers (big integers, modular arithmetic, VMs, etc.) can safely
treat it as the “ternary `int128`” of the T81 ecosystem.

This README focuses on:

- What `t81lib` provides,
- How to build and integrate it,
- How to use the core types,
- Where to find the formal specs and design notes.

---

## 1. Features

**Core scalar**

- `t81::core::limb` — 48-trit, signed balanced-ternary scalar
  - Digits in `{−1, 0, +1}` (“trits”)
  - Canonical packing into 16 trytes → 16 bytes
  - No UB; overflow behavior is explicit and specified
  - `constexpr`-friendly where practical

**Multi-precision**

- `t81::core::bigint` — multi-limb, arbitrary-precision balanced-ternary integer
  - Representation: sign + vector of canonical `limb`s
  - Base `B = 3^48` internally
  - Normalized, canonical representation (no leading zero limbs)

**I/O and utilities**

- `t81::io` — string / stream I/O helpers for `limb` and `bigint`
- `t81::util` — random value generators, debug dumps, invariants

**Design & spec**

- Normative spec for `limb` (long-term stability, ≥10 years)
- Library-level spec and design notes for:
  - `limb` internals
  - `bigint` internals
  - Montgomery modular arithmetic design

**Build system & hygiene**

- CMake-first (‘modern CMake’ style)
- `t81::t81lib` target with proper include directories
- Optional tests and benchmarks
- CI workflow example (GitHub Actions)

---

## 2. Status & Stability

- **Library version:** `t81lib` is pre-1.0 as a whole until APIs stabilize.
- **Scalar (`t81::core::limb`):**  
  Normative and intended to be stable for ≥10 years:
  - Canonical encoding
  - Arithmetic semantics
  - Binary I/O

- **`bigint`, `io`, `util`:**
  - Considered library-level APIs: stable enough for users, but may evolve
  - Always remain compatible with `limb` semantics

You can treat `limb` like a frozen foundation and `bigint`/friends as evolving
layers built on top.

---

## 3. Getting Started

### 3.1 Requirements

- C++20 compiler (GCC, Clang, MSVC with C++20 support)
- CMake ≥ 3.16

### 3.2 Fetch & build (standalone)

```bash
git clone https://github.com/your-org/t81lib.git
cd t81lib
cmake -S . -B build -DT81LIB_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
````

### 3.3 Using as a CMake subproject

If you vendor `t81lib` into your project:

```cmake
add_subdirectory(external/t81lib)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE t81::t81lib)
```

Include headers in your code:

```cpp
#include <t81/t81lib.hpp>
```

### 3.4 Using as an installed package

Once installed:

```cmake
find_package(t81lib REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE t81::t81lib)
```

---

## 4. Basic Usage

All core APIs are accessible via the umbrella header:

```cpp
#include <t81/t81lib.hpp>

using t81::core::limb;
using t81::core::bigint;
```

### 4.1 Working with `t81::core::limb`

```cpp
#include <t81/t81lib.hpp>
#include <iostream>

int main() {
    using t81::core::limb;

    limb a = limb::from_int(42);
    limb b = limb::from_int(-7);

    limb c = a + b;         // exact balanced-ternary addition
    limb d = a * b;         // throws std::overflow_error on overflow

    std::cout << "c = " << c.to_string(10) << "\n";
    std::cout << "d = " << d.to_string(10) << "\n";

    // Binary I/O via canonical 16-byte encoding
    std::uint8_t buf[limb::BYTES] = {};
    d.to_bytes(buf);

    limb e = limb::from_bytes(buf);
    std::cout << "e == d? " << std::boolalpha << (e == d) << "\n";
}
```

Key properties:

* `limb::TRITS == 48`, `TRYTES == 16`, `BYTES == 16`
* Stable tryte/trit mapping (see `doc/t81lib-spec-v1.0.0.md`)
* All arithmetic is deterministic; overflows are signaled, not UB

### 4.2 Working with `t81::core::bigint`

```cpp
#include <t81/t81lib.hpp>
#include <iostream>

int main() {
    using t81::core::bigint;

    bigint x{42};         // construction from int (API may adapt)
    bigint y{1};
    for (int i = 0; i < 10; ++i) {
        y = y * x;        // y = 42^10
    }

    std::cout << "42^10 = " << y.to_string(10) << "\n";

    // Down-conversion when in range:
    try {
        t81::core::limb small = static_cast<t81::core::limb>(y);
        std::cout << "Fits in a limb: " << small.to_string(10) << "\n";
    } catch (const std::overflow_error&) {
        std::cout << "Does not fit in a limb.\n";
    }
}
```

`bigint`:

* Represents an integer as sign + vector of `limb`s in base `3^48`
* Maintains canonical normalization (no leading zero limbs)

### 4.3 Montgomery arithmetic (design-level, optional implementation)

The README does not promise a stable Montgomery API, but the design is:

```cpp
#include <t81/t81lib.hpp>
#include <iostream>

int main() {
    using t81::core::limb;
    using t81::core::MontgomeryContext; // when provided

    limb m = limb::from_int(97);   // modulus (not divisible by 3)
    MontgomeryContext<limb> ctx(m);

    limb a = limb::from_int(5);
    limb b = limb::from_int(7);

    limb a_bar = ctx.to_montgomery(a);
    limb b_bar = ctx.to_montgomery(b);

    limb c_bar = ctx.mul(a_bar, b_bar);
    limb c     = ctx.from_montgomery(c_bar);   // c == (5 * 7) mod 97

    std::cout << "5 * 7 mod 97 = " << c.to_string(10) << "\n";
}
```

See `doc/design/montgomery-design.md` for the full design.

---

## 5. Directory Layout

A typical tree (simplified):

```text
t81lib/
├── CMakeLists.txt
├── README.md
├── LICENSE
├── CHANGELOG.md
├── CONTRIBUTING.md
├── include/
│   └── t81/
│       ├── t81lib.hpp            # umbrella
│       ├── core/
│       │   ├── limb.hpp          # canonical 48-trit scalar
│       │   ├── bigint.hpp        # arbitrary-precision integer
│       │   ├── traits.hpp        # type traits / concepts
│       │   └── detail/
│       │       └── lut.hpp       # trit/tryte tables, internal LUTs
│       ├── io/
│       │   ├── format.hpp        # string/stream I/O
│       │   └── parse.hpp         # parsing helpers
│       └── util/
│           ├── random.hpp        # random generators
│           └── debug.hpp         # debug dumps, invariants
├── src/
│   ├── limb.cpp                  # optional non-header-only parts
│   ├── bigint.cpp
│   └── io.cpp
├── tests/
│   ├── CMakeLists.txt
│   ├── unit/
│   │   ├── test_limb_basic.cpp
│   │   ├── test_bigint_basic.cpp
│   │   └── ...
│   ├── fuzz/
│   │   └── ...
│   └── property/
│       └── ...
├── bench/
│   ├── CMakeLists.txt
│   └── bench_limb_add.cpp
├── examples/
│   ├── CMakeLists.txt
│   └── example_basic.cpp
├── doc/
│   ├── t81lib-spec-v1.0.0.md
│   └── design/
│       ├── limb-design.md
│       ├── bigint-design.md
│       └── montgomery-design.md
└── .github/
    └── workflows/
        └── ci.yml
```

The public API is everything under `include/t81/**` that is **not** in a `detail`
subdirectory.

---

## 6. Specs & Design Documents

Normative and design docs live under `doc/`:

* **Library spec (top-level)**

  * `doc/t81lib-spec-v1.0.0.md`

    * Library-level overview
    * Limb as canonical scalar
    * Stability guarantees

* **Design notes**

  * `doc/design/limb-design.md`

    * Internal representation
    * LUTs & Kogge–Stone add
    * `mul_wide` design and normalization passes
  * `doc/design/bigint-design.md`

    * Representation as sign + limbs
    * Algorithms for add/mul/div
    * Normalization invariants
  * `doc/design/montgomery-design.md`

    * Montgomery reduction in base `3^48`
    * Context API for `limb` and `bigint`

If you are implementing new features or optimizing internals, start with these.

---

## 7. Testing & Benchmarks

### 7.1 Building tests

```bash
cmake -S . -B build -DT81LIB_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Tests typically cover:

* `limb` basics: construction, arithmetic, conversions
* `bigint` magnitude operations, sign handling, conversions
* Serialization and equality invariants
* Property tests (when enabled) comparing against reference models

### 7.2 Benchmarks

If benchmarks are enabled:

```bash
cmake -S . -B build -DT81LIB_BUILD_BENCH=ON
cmake --build build -j
./build/bench/bench_limb_add
./build/bench/bench_limb_ops
```

Benchmarks target:

* `limb` add/mul throughput
* `limb` arithmetic (add/sub/mul/div/mod) comparisons between ternary operators and equivalent binary math
* Effects of different implementations / backends (scalar vs SIMD)

---

## 8. Contributing

Contributions are welcome. The general flow:

1. **Open an issue** describing what you want to change:

   * Bug fix
   * New feature
   * Optimization
   * Documentation or examples

2. **Fork & branch**

   * Create a feature branch in your fork.

3. **Keep it deterministic**

   * Avoid UB and implementation-defined behavior.
   * When in doubt, align with the limb spec.

4. **Add tests**

   * For new features, add unit tests.
   * For algorithm changes, consider adding or updating property tests.

5. **Run the test suite**

   * `ctest --output-on-failure` must pass.

6. **Submit a PR**

   * Reference the relevant spec/design sections if you’re touching core semantics.

See `CONTRIBUTING.md` for style guidelines and more detailed expectations.

---

## 9. License

`t81lib` is released under a permissive license suitable for open-source and
commercial use.

See `LICENSE` in the repository for details.
