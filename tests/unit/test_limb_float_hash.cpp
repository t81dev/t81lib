// tests/unit/test_limb_float_hash.cpp — Tests ensuring limb float hashing consistency.

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>

#include <t81/t81lib.hpp>

int main() {
    bool all_good = true;
    const auto expect = [&](bool condition, const char *message) {
        if (!condition) {
            all_good = false;
            std::cerr << "limit float/hash test failure: " << message << '\n';
        }
    };

    using t81::core::limb;

    const limb a = limb::from_double(42.9);
    expect(a == limb::from_value(42), "from_double should truncate toward zero");

    const limb b = limb::from_double(-7.3);
    expect(b == limb::from_value(-7), "from_double should truncate negatives toward zero");

    const limb c = limb::from_long_double(0.0L);
    expect(c.is_zero(), "from_long_double should accept zero");

    expect(a.to_double() == static_cast<double>(a.to_value()),
           "to_double must match integer value");
    expect(static_cast<double>(b) == -7.0, "operator double must expose the value");

    bool overflowed = false;
    try {
        const long double too_large = static_cast<long double>(limb::max().to_value()) + 1000.0L;
        (void)limb::from_long_double(too_large);
    } catch (const std::overflow_error &) {
        overflowed = true;
    }
    expect(overflowed, "floating-point overflow must throw");

    bool invalid = false;
    try {
        (void)limb::from_double(std::numeric_limits<double>::infinity());
    } catch (const std::invalid_argument &) {
        invalid = true;
    }
    expect(invalid, "inf must be rejected");

    bool nan_rejected = false;
    try {
        (void)limb::from_double(std::numeric_limits<double>::quiet_NaN());
    } catch (const std::invalid_argument &) {
        nan_rejected = true;
    }
    expect(nan_rejected, "NaN must be rejected");

    const auto hash_value = std::hash<limb>{}(a);
    const auto bytes_hash = t81::core::canonical_hash(a);
    expect(hash_value == bytes_hash, "std::hash must match canonical hash helper");

    const limb equivalent = limb::from_bytes(a.to_bytes());
    expect(std::hash<limb>{}(equivalent) == hash_value,
           "Hash must be determined solely by to_bytes()");

    auto mutated_bytes = a.to_bytes();
    mutated_bytes[0] = static_cast<limb::tryte_t>((mutated_bytes[0] + 1) % 27);
    const limb mutated = limb::from_bytes(mutated_bytes);
    expect(std::hash<limb>{}(mutated) != hash_value, "Changing a byte should change the hash");

    if (!all_good) {
        std::cerr << "limb float/hash tests failed\n";
        return 1;
    }
    std::cout << "limb float/hash tests passed\n";
    return 0;
}
