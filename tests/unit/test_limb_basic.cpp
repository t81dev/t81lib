// tests/unit/test_limb_basic.cpp — Basic limb arithmetic unit tests.

#include <array>
#include <iostream>
#include <random>
#include <vector>

#include <t81/t81lib.hpp>
#include <t81/core/detail/simd.hpp>

namespace {

    using t81::core::limb;
    namespace core_detail = t81::core::detail;

    bool compare_tryte_add(const limb &lhs, const limb &rhs) {
        const limb reference = limb::from_value(lhs.to_value() + rhs.to_value());
        if (core_detail::cpu_supports_avx2()) {
            limb avx_result;
            if (core_detail::add_trytes_avx2(lhs, rhs, avx_result)) {
                if (avx_result != reference) {
                    return false;
                }
            }
        }
        if (core_detail::cpu_supports_neon()) {
            limb neon_result;
            if (core_detail::add_trytes_neon(lhs, rhs, neon_result)) {
                if (neon_result != reference) {
                    return false;
                }
            }
        }
        return true;
    }

    bool compare_mul_wide(const limb &lhs, const limb &rhs) {
        const auto reference = core_detail::mul_wide_scalar(lhs, rhs);
        if (core_detail::cpu_supports_avx2()) {
            const auto avx_result = core_detail::mul_wide_avx2(lhs, rhs);
            if (avx_result &&
                (avx_result->first != reference.first || avx_result->second != reference.second)) {
                return false;
            }
        }
        if (core_detail::cpu_supports_neon()) {
            const auto neon_result = core_detail::mul_wide_neon(lhs, rhs);
            if (neon_result && (neon_result->first != reference.first ||
                                neon_result->second != reference.second)) {
                return false;
            }
        }
        return true;
    }

} // namespace

limb
random_limb(std::mt19937_64 &rng) {
    std::uniform_int_distribution<int> dist(0, 26);
    std::array<limb::tryte_t, limb::TRYTES> trytes{};
    for (auto &entry : trytes) {
        entry = static_cast<limb::tryte_t>(dist(rng));
    }
    return limb::from_bytes(trytes);
}

int
main() {
    std::vector<limb> values = {
        limb::zero(),
        limb::one(),
        -limb::one(),
        limb::from_value(42),
        limb::from_value(-42),
        limb::from_value(1234),
        limb::from_value(-1234),
        limb::min(),
        limb::max(),
    };

    bool all_good = true;
    auto addition_in_range = [](const limb &lhs, const limb &rhs) {
        using t81::core::detail::limb_int128;
        const limb_int128 sum = lhs.to_value() + rhs.to_value();
        return sum >= t81::core::detail::MIN_VALUE && sum <= t81::core::detail::MAX_VALUE;
    };

    for (const auto &lhs : values) {
        for (const auto &rhs : values) {
            if (addition_in_range(lhs, rhs) && !compare_tryte_add(lhs, rhs)) {
                std::cerr << "SIMD tryte addition mismatch for " << lhs.to_string() << " + "
                          << rhs.to_string() << "\n";
            }
            if (!compare_mul_wide(lhs, rhs)) {
                all_good = false;
                std::cerr << "SIMD mul_wide mismatch for " << lhs.to_string() << " * "
                          << rhs.to_string() << "\n";
            }
        }
    }

    std::mt19937_64 rng(0xdeadbeef);
    for (int iteration = 0; iteration < 1000; ++iteration) {
        const limb lhs = random_limb(rng);
        const limb rhs = random_limb(rng);
        if (addition_in_range(lhs, rhs) && !compare_tryte_add(lhs, rhs)) {
            std::cerr << "SIMD tryte addition mismatch (random) for " << lhs.to_string() << " + "
                      << rhs.to_string() << "\n";
        }
        if (!compare_mul_wide(lhs, rhs)) {
            all_good = false;
            std::cerr << "SIMD mul_wide mismatch (random) for " << lhs.to_string() << " * "
                      << rhs.to_string() << "\n";
        }
    }

    if (!all_good) {
        std::cerr << "limb SIMD regression failed\n";
        return 1;
    }

    std::cout << "limb SIMD regression passed\n";
    return 0;
}
