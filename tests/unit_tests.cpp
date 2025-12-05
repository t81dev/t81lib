#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "t81/core/T81Limb.hpp"

namespace {

using t81::core::T81Limb;
using Triplet = std::array<int8_t, T81Limb::TRITS>;

struct AssertionFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

inline void ensure(bool cond, const char* message) {
    if (!cond) throw AssertionFailure(message);
}

inline void ensure_eq(int lhs, int rhs, const char* message) {
    if (lhs != rhs) throw AssertionFailure(message);
}

constexpr int WIDE_TRITS = 96;

uint32_t advance_seed(uint32_t seed) {
    return seed * 1664525u + 1013904223u;
}

Triplet random_trits(uint32_t& seed) {
    Triplet trits{};
    for (int idx = 0; idx < static_cast<int>(trits.size()); ++idx) {
        seed = advance_seed(seed);
        trits[idx] = static_cast<int8_t>((seed % 3) - 1);
    }
    return trits;
}

T81Limb random_limb(uint32_t& seed) {
    return T81Limb::from_trits(random_trits(seed));
}

std::array<int8_t, T81Limb::TRITS> naive_add(const T81Limb& a, const T81Limb& b) {
    auto at = a.to_trits();
    auto bt = b.to_trits();
    std::array<int8_t, T81Limb::TRITS> result{};
    int carry = 0;
    for (int idx = 0; idx < T81Limb::TRITS; ++idx) {
        int sum = static_cast<int>(at[idx]) + static_cast<int>(bt[idx]) + carry;
        if (sum > 1) {
            result[idx] = static_cast<int8_t>(sum - 3);
            carry = 1;
        } else if (sum < -1) {
            result[idx] = static_cast<int8_t>(sum + 3);
            carry = -1;
        } else {
            result[idx] = static_cast<int8_t>(sum);
            carry = 0;
        }
    }
    return result;
}

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

int run_tests() {
    std::vector<std::pair<std::string, std::function<void()>>> cases;
    cases.emplace_back("Addition matches naive implementation", []() {
        uint32_t seed = 0xC0FFEEu;
        for (int iteration = 0; iteration < 256; ++iteration) {
            auto lhs = random_limb(seed);
            auto rhs = random_limb(seed);

            auto sum = lhs + rhs;
            auto expected = T81Limb::from_trits(naive_add(lhs, rhs));

            ensure(sum.compare(expected) == 0, "kAdd vs naive trit addition mismatch");
            ensure(sum.compare(rhs + lhs) == 0, "addition not commutative");
        }
    });

    cases.emplace_back("Negation and subtraction preserve invariants", []() {
        uint32_t seed = 0xFEEDu;
        T81Limb zero{};
        for (int iteration = 0; iteration < 128; ++iteration) {
            auto value = random_limb(seed);
            auto negated = -value;
            auto restored = -negated;
            ensure(restored.compare(value) == 0, "double negation failed");
            auto difference = value - negated;
            ensure(difference.compare(value + value) == 0, "difference vs add mismatch");
        }
    });

    cases.emplace_back("Comparison is antisymmetric and reflexive", []() {
        uint32_t seed = 0x1234u;
        auto lhs = random_limb(seed);
        auto rhs = random_limb(seed);
        ensure(lhs.compare(lhs) == 0, "compare not zero for identical limbs");
        int forward = lhs.compare(rhs);
        int backward = rhs.compare(lhs);
        ensure(forward == -backward, "compare antisymmetry broken");
    });

    cases.emplace_back("Multiplication equals reference implementation", []() {
        uint32_t seed = 0xDEADBEEFu;
        for (int iteration = 0; iteration < 64; ++iteration) {
            auto lhs = random_limb(seed);
            auto rhs = random_limb(seed);
            auto fast = lhs * rhs;
            auto reference = T81Limb::reference_mul(lhs, rhs);
            ensure(fast.compare(reference) == 0, "fast multiplication diverged from reference");
        }
    });

    cases.emplace_back("mul_wide reproduces canonical wide result", []() {
        uint32_t seed = 0xFACEu;
        for (int iteration = 0; iteration < 48; ++iteration) {
            auto lhs = random_limb(seed);
            auto rhs = random_limb(seed);
            auto fast = T81Limb::mul_wide(lhs, rhs);
            auto canonical = canonical_wide(lhs, rhs);
            ensure(fast.first.compare(canonical.first) == 0, "mul_wide low half mismatch");
            ensure(fast.second.compare(canonical.second) == 0, "mul_wide high half mismatch");
        }
    });

    int failed = 0;
    for (auto& [name, body] : cases) {
        try {
            body();
            std::cout << "[PASS] " << name << '\n';
        } catch (const std::exception& ex) {
            ++failed;
            std::cerr << "[FAIL] " << name << ": " << ex.what() << '\n';
        }
    }
    std::cout << (failed ? "[RESULT] Some unit tests failed\n" : "[RESULT] All unit tests passed\n");
    return failed;
}

} // namespace

int main() {
    return run_tests();
}
