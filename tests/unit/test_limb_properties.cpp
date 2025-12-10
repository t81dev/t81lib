// tests/unit/test_limb_properties.cpp — Unit tests verifying limb invariants.

#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

#include <t81/t81lib.hpp>

namespace {

using t81::core::limb;
namespace core_detail = t81::core::detail;

limb random_limb(std::mt19937_64& rng) {
    std::uniform_int_distribution<int> dist(0, 26);
    std::array<limb::tryte_t, limb::TRYTES> trytes{};
    for (auto& entry : trytes) {
        entry = static_cast<limb::tryte_t>(dist(rng));
    }
    return limb::from_bytes(trytes);
}

std::size_t manual_canonical_hash(const limb& value) {
    constexpr std::uint64_t FNV_OFFSET = 1469598103934665603ULL;
    constexpr std::uint64_t FNV_PRIME = 1099511628211ULL;
    std::uint64_t hash = FNV_OFFSET;
    for (auto byte : value.to_bytes()) {
        hash ^= byte;
        hash *= FNV_PRIME;
    }
    if constexpr (sizeof(std::size_t) >= 8) {
        return static_cast<std::size_t>(hash);
    }
    return static_cast<std::size_t>((hash >> 32) ^ (hash & 0xFFFFFFFFULL));
}

} // namespace

int main() {
    bool all_good = true;
    const auto expect = [&](bool condition, const char* message) {
        if (!condition) {
            all_good = false;
            std::cerr << "limb property test failed: " << message << '\n';
        }
    };

    std::mt19937_64 rng(0xfeedbeef);
    std::uniform_int_distribution<int32_t> small_int(-100000, 100000);
    std::uniform_real_distribution<double> real_dist(-1000.0, 1000.0);

    for (int iteration = 0; iteration < 1000; ++iteration) {
        const limb value = random_limb(rng);
        const auto bytes = value.to_bytes();
        for (auto byte : bytes) {
            expect(byte <= 26, "tryte byte must be in canonical range");
        }
        const limb roundtrip = limb::from_bytes(bytes);
        expect(roundtrip == value, "from_bytes(to_bytes()) should round-trip");

        const auto canonical = t81::core::canonical_hash(value);
        const auto manual = manual_canonical_hash(value);
        expect(canonical == manual, "canonical hash helper must match manual FNV");

        auto mutated_bytes = bytes;
        mutated_bytes[0] = static_cast<limb::tryte_t>((mutated_bytes[0] + 1) % 27);
        const limb mutated = limb::from_bytes(mutated_bytes);
        expect(t81::core::canonical_hash(mutated) != canonical,
               "hash must change when bytes change");
    }

    for (int iteration = 0; iteration < 1000; ++iteration) {
        const double sample = real_dist(rng);
        const double truncated = std::trunc(sample);
        try {
            const limb converted = limb::from_double(sample);
            const core_detail::limb_int128 expected =
                static_cast<core_detail::limb_int128>(truncated);
            if (expected < core_detail::MIN_VALUE || expected > core_detail::MAX_VALUE) {
                expect(false, "out-of-range float must throw overflow_error");
                continue;
            }
            expect(converted.to_value() == expected, "float conversion must truncate toward zero");
        } catch (const std::overflow_error&) {
            const core_detail::limb_int128 expected =
                static_cast<core_detail::limb_int128>(truncated);
            expect(expected < core_detail::MIN_VALUE || expected > core_detail::MAX_VALUE,
                   "overflow only when truncated value is out-of-range");
        }
    }

    for (const auto special : {std::numeric_limits<double>::infinity(),
                              -std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::quiet_NaN()}) {
        try {
            (void)limb::from_double(special);
            expect(false, "infinity/NaN must throw invalid_argument");
        } catch (const std::invalid_argument&) {
            // expected
        }
    }

    if (!all_good) {
        std::cerr << "limb property tests failed\n";
        return 1;
    }
    std::cout << "limb property tests passed\n";
    return 0;
}
