#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <random>

#include "t81/core/T81Limb.hpp"

namespace {

using t81::core::T81Limb;

static constexpr int WIDE_TRITS = 96;

std::pair<T81Limb, T81Limb> canonical_wide(const T81Limb& a, const T81Limb& b) {
    std::array<int, WIDE_TRITS> accum{};
    auto atr = a.to_trits();
    auto btr = b.to_trits();
    for (int i = 0; i < T81Limb::TRITS; ++i) {
        for (int j = 0; j < T81Limb::TRITS; ++j) {
            accum[i + j] += static_cast<int>(atr[i]) * static_cast<int>(btr[j]);
        }
    }

    auto normalize = [](std::array<int, WIDE_TRITS>& values) {
        for (int idx = 0; idx + 1 < WIDE_TRITS; ++idx) {
            int carry = (values[idx] + (values[idx] >= 0 ? 1 : -1)) / 3;
            values[idx] -= carry * 3;
            values[idx + 1] += carry;
        }
        int carry = (values[WIDE_TRITS - 1] + (values[WIDE_TRITS - 1] >= 0 ? 1 : -1)) / 3;
        values[WIDE_TRITS - 1] -= carry * 3;
    };

    normalize(accum);
    normalize(accum);
    normalize(accum);

    std::array<int8_t, WIDE_TRITS> normalized{};
    for (int idx = 0; idx < WIDE_TRITS; ++idx) {
        int value = accum[idx];
        if (value > 1) value = 1;
        if (value < -1) value = -1;
        normalized[idx] = static_cast<int8_t>(value);
    }

    std::array<int8_t, T81Limb::TRITS> low{};
    std::array<int8_t, T81Limb::TRITS> high{};
    std::copy_n(normalized.begin(), T81Limb::TRITS, low.begin());
    std::copy_n(normalized.begin() + T81Limb::TRITS, T81Limb::TRITS, high.begin());

    return { T81Limb::from_trits(low), T81Limb::from_trits(high) };
}

void run_fuzz(int iterations, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> tryte_dist(-3, 3);

    for (int i = 0; i < iterations; ++i) {
        T81Limb lhs, rhs;
        for (int t = 0; t < T81Limb::TRYTES; ++t) {
            lhs.set_tryte(t, static_cast<int8_t>(tryte_dist(rng)));
            rhs.set_tryte(t, static_cast<int8_t>(tryte_dist(rng)));
        }

        auto [sum, carry] = lhs.addc(rhs);
        if (carry != 0) {
            continue;
        }
        auto sum_alt = lhs + rhs;
        if (sum.compare(sum_alt) != 0) {
            throw std::runtime_error("fuzz: addc/operator+ mismatch");
        }
        auto diff = lhs - rhs;
        auto prod = lhs * rhs;
        auto reference = T81Limb::reference_mul(lhs, rhs);

        auto alt_diff = lhs + (-rhs);
        if (diff.compare(alt_diff) != 0) {
            throw std::runtime_error("fuzz: add/sub mismatch");
        }
        if (prod.compare(reference) != 0) {
            throw std::runtime_error("fuzz: mul mismatch");
        }
        auto [lo, hi] = T81Limb::mul_wide(lhs, rhs);
        auto canonical = canonical_wide(lhs, rhs);
        if (lo.compare(canonical.first) != 0 || hi.compare(canonical.second) != 0) {
            throw std::runtime_error("fuzz: mul_wide mismatch");
        }
    }
}

} // namespace

int main() {
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    run_fuzz(1024, 0xBEEFu);
    auto end = high_resolution_clock::now();
    std::cout << "fuzz completed in "
              << duration_cast<milliseconds>(end - start).count()
              << " ms\n";
    return 0;
}
