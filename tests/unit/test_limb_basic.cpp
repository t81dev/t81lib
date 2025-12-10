#include <array>
#include <iostream>
#include <random>
#include <vector>

#include <t81/t81lib.hpp>
#include <t81/core/detail/simd.hpp>

namespace {

using t81::core::limb;
namespace core_detail = t81::core::detail;

bool compare_tryte_add(const limb& lhs, const limb& rhs) {
    const limb reference = limb::from_value(lhs.to_value() + rhs.to_value());
    limb avx_result;
    limb neon_result;
    if (!core_detail::add_trytes_avx2(lhs, rhs, avx_result) || avx_result != reference) {
        return false;
    }
    if (!core_detail::add_trytes_neon(lhs, rhs, neon_result) || neon_result != reference) {
        return false;
    }
    return true;
}

bool compare_mul_wide(const limb& lhs, const limb& rhs) {
    const auto reference = core_detail::mul_wide_scalar(lhs, rhs);
    const auto avx_result = core_detail::mul_wide_avx2(lhs, rhs);
    const auto neon_result = core_detail::mul_wide_neon(lhs, rhs);
    if (!avx_result || !neon_result) {
        return false;
    }
    return avx_result->first == reference.first && avx_result->second == reference.second &&
           neon_result->first == reference.first && neon_result->second == reference.second;
}

} // namespace

limb random_limb(std::mt19937_64& rng) {
    std::uniform_int_distribution<int> dist(0, 26);
    std::array<limb::tryte_t, limb::TRYTES> trytes{};
    for (auto& entry : trytes) {
        entry = static_cast<limb::tryte_t>(dist(rng));
    }
    return limb::from_bytes(trytes);
}

int main() {
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
    for (const auto& lhs : values) {
        for (const auto& rhs : values) {
            if (!compare_tryte_add(lhs, rhs)) {
                all_good = false;
                std::cerr << "SIMD tryte addition mismatch for "
                          << lhs.to_string() << " + " << rhs.to_string() << "\n";
            }
            if (!compare_mul_wide(lhs, rhs)) {
                all_good = false;
                std::cerr << "SIMD mul_wide mismatch for "
                          << lhs.to_string() << " * " << rhs.to_string() << "\n";
            }
        }
    }

    std::mt19937_64 rng(0xdeadbeef);
    for (int iteration = 0; iteration < 1000; ++iteration) {
        const limb lhs = random_limb(rng);
        const limb rhs = random_limb(rng);
        if (!compare_tryte_add(lhs, rhs)) {
            all_good = false;
            std::cerr << "SIMD tryte addition mismatch (random) for "
                      << lhs.to_string() << " + " << rhs.to_string() << "\n";
        }
        if (!compare_mul_wide(lhs, rhs)) {
            all_good = false;
            std::cerr << "SIMD mul_wide mismatch (random) for "
                      << lhs.to_string() << " * " << rhs.to_string() << "\n";
        }
    }

    if (!all_good) {
        std::cerr << "limb SIMD regression failed\n";
        return 1;
    }

    std::cout << "limb SIMD regression passed\n";
    return 0;
}
