# Cookbook

Small recipes that show how to use t81lib’s primitives for concrete tasks.

## Chain mul_wide for big integers

When building a 192-trit (4-limb) multiply, compute pairwise `mul_wide` results and handle carries manually:

```cpp
std::array<T81Limb, 4> a = {...}, b = {...};
std::array<T81Limb, 8> result{};
for (int i = 0; i < 4; ++i) {
  for (int j = 0; j < 4; ++j) {
    auto [lo, hi] = T81Limb::mul_wide(a[i], b[j]);
    // accumulate lo at result[i+j], hi at i+j+1 with addc
  }
}
```

Use `mul_wide_fast` when available to accelerate inner loops; the fuzz test ensures correctness.

## Packing tryte streams

To compare ternary density with binary:

```cpp
std::array<Trit, 19> trits = {...};
auto packed = packing::pack_trits(trits);
std::cout << "trits/packed bits = " << trits.size() << "/" << packing::packed_bits(trits.size()) << "\n";
```

Combine with `unpack_trits` for round-trippable conversions and feed the packed state into IoT/serialization layers that expect binary payloads.

## CLI helper: checking compare/add performance

Build small CLI that reads ints, converts via `from_int`, performs `operator+` or `compare`, and prints the time using `std::chrono`. The `bench/main.cpp` already shows how to hook into Google Benchmark; borrow its `random_limb()` helper for deterministic inputs.
