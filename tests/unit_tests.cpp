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
using t81::core::T81Limb27;
using t81::core::T81Limb81;
using Triplet = std::array<int8_t, T81Limb::TRITS>;
using t81::core::MontgomeryContext;
using t81::core::MontgomeryContextTestAccess;

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

std::array<int8_t, T81Limb81::TRITS> random_trits81(uint32_t& seed) {
    std::array<int8_t, T81Limb81::TRITS> trits{};
    for (int idx = 0; idx < static_cast<int>(trits.size()); ++idx) {
        seed = advance_seed(seed);
        trits[idx] = static_cast<int8_t>((seed % 5) - 2);
    }
    return trits;
}

std::array<int8_t, T81Limb81::TRITS * 2> naive_mul81(
    const std::array<int8_t, T81Limb81::TRITS>& a,
    const std::array<int8_t, T81Limb81::TRITS>& b)
{
    constexpr int WIDE_TRITS = T81Limb81::TRITS * 2;
    std::array<int, WIDE_TRITS> accum{};
    for (int i = 0; i < T81Limb81::TRITS; ++i) {
        for (int j = 0; j < T81Limb81::TRITS; ++j) {
            accum[i + j] += static_cast<int>(a[i]) * static_cast<int>(b[j]);
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

    std::array<int8_t, WIDE_TRITS> result{};
    for (int idx = 0; idx < WIDE_TRITS; ++idx) {
        int value = accum[idx];
        if (value > 1) value = 1;
        else if (value < -1) value = -1;
        result[idx] = static_cast<int8_t>(value);
    }
    return result;
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

T81Limb naive_pow_mod(T81Limb base, T81Limb exp, const T81Limb& mod) {
    T81Limb result = T81Limb::one();
    auto two = T81Limb::from_int(2);
    while (!exp.is_zero()) {
        auto [quot, rem] = exp.div_mod(two);
        if (!rem.is_zero()) {
            result = (result * base) % mod;
        }
        base = (base * base) % mod;
        exp = quot;
    }
    return result;
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

    cases.emplace_back("T81Limb81 conversions normalize and round-trip", []() {
        uint32_t seed = 0x1ADBEEFu;
        for (int iteration = 0; iteration < 64; ++iteration) {
            auto digits = random_trits81(seed);
            auto limb81 = T81Limb81::from_trits(digits);
            auto normalized = limb81.to_trits();
            auto round_trip = T81Limb81::from_trits(normalized);
            ensure(normalized == round_trip.to_trits(), "T81Limb81 conversion round-trip mismatch");
            for (auto value : normalized) {
                ensure(value >= -1 && value <= 1, "T81Limb81 trit out of range");
            }
        }
    });

    cases.emplace_back("T81Limb81 slices/stitches T81Limb27 blocks", []() {
        uint32_t seed = 0xABCD1234u;
        for (int iteration = 0; iteration < 64; ++iteration) {
            auto digits = random_trits81(seed);
            auto original = T81Limb81::from_trits(digits);
            auto normalized = original.to_trits();
            auto lo = original.lo();
            auto mid = original.mid();
            auto hi = original.hi();
            auto composed = T81Limb81::from_parts(lo, mid, hi);
            ensure(normalized == composed.to_trits(), "T81Limb81 block recomposition diverged");
            auto lo_trits = lo.to_trits();
            auto mid_trits = mid.to_trits();
            auto hi_trits = hi.to_trits();
            for (int idx = 0; idx < T81Limb27::TRITS; ++idx) {
                ensure(lo_trits[idx] == normalized[idx], "T81Limb81 low block mismatch");
                ensure(mid_trits[idx] == normalized[idx + T81Limb27::TRITS], "T81Limb81 mid block mismatch");
                ensure(hi_trits[idx] == normalized[idx + T81Limb27::TRITS * 2], "T81Limb81 high block mismatch");
            }
        }
    });

    cases.emplace_back("T81Limb81 mul_wide matches naive normalization", []() {
        uint32_t seed = 0xCAFEBABEu;
        for (int iteration = 0; iteration < 48; ++iteration) {
            auto lhs_digits = random_trits81(seed);
            auto rhs_digits = random_trits81(seed);
            auto lhs = T81Limb81::from_trits(lhs_digits);
            auto rhs = T81Limb81::from_trits(rhs_digits);
            auto expected = naive_mul81(lhs_digits, rhs_digits);
            auto [low, high] = T81Limb81::mul_wide(lhs, rhs);
            auto low_trits = low.to_trits();
            auto high_trits = high.to_trits();
            for (int idx = 0; idx < T81Limb81::TRITS; ++idx) {
                ensure(low_trits[idx] == expected[idx], "T81Limb81 mul_wide low half mismatch");
                ensure(high_trits[idx] == expected[idx + T81Limb81::TRITS], "T81Limb81 mul_wide high half mismatch");
            }
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

    cases.emplace_back("pow_mod matches naive Montgomery pow", []() {
        uint32_t seed = 0xBEEFCAFEu;
        for (int iteration = 0; iteration < 64; ++iteration) {
            auto base = random_limb(seed);
            auto exponent = random_limb(seed);
            auto modulus = random_limb(seed);
            if (modulus.is_zero()) continue;
            auto positive_mod = modulus.is_negative() ? -modulus : modulus;
            auto positive_exp = exponent.is_negative() ? -exponent : exponent;
            auto direct = T81Limb::pow_mod(base, positive_exp, positive_mod);
            auto reference = naive_pow_mod(base % positive_mod, positive_exp, positive_mod);
            ensure(direct.compare(reference) == 0, "pow_mod mismatch vs naive");
        }
    });

    cases.emplace_back("Montgomery multiply equals modular product", []() {
        uint32_t seed = 0xCAFEF00Du;
        for (int iteration = 0; iteration < 64; ++iteration) {
            auto lhs = random_limb(seed);
            auto rhs = random_limb(seed);
            auto modulus = random_limb(seed);
            if (modulus.is_zero()) continue;
            auto positive_mod = modulus.is_negative() ? -modulus : modulus;
            auto positive_lhs = lhs % positive_mod;
            if (positive_lhs.is_negative()) positive_lhs = positive_lhs + positive_mod;
            auto positive_rhs = rhs % positive_mod;
            if (positive_rhs.is_negative()) positive_rhs = positive_rhs + positive_mod;
            MontgomeryContext ctx(positive_mod);
            auto mont = ctx.multiply(positive_lhs, positive_rhs);
            auto expected = (positive_lhs * positive_rhs) % positive_mod;
            ensure(mont.compare(expected) == 0, "Montgomery multiply mismatch");
        }
    });

    cases.emplace_back("Montgomery converters round-trip", []() {
        uint32_t seed = 0xDEAD001Fu;
        for (int iteration = 0; iteration < 64; ++iteration) {
            auto value = random_limb(seed);
            auto modulus = random_limb(seed);
            if (modulus.is_zero()) continue;
            auto positive_mod = modulus.is_negative() ? -modulus : modulus;
            auto residue = value % positive_mod;
            if (residue.is_negative()) residue = residue + positive_mod;
            MontgomeryContext ctx(positive_mod);
            auto mont = ctx.to_montgomery(residue);
            auto restored = ctx.from_montgomery(mont);
            ensure(restored.compare(residue) == 0, "Montgomery round-trip mismatch");
        }
    });

    cases.emplace_back("Montgomery reduce matches brute-force wide product", []() {
        uint32_t seed = 0xFEEDBEEFu;
        for (int iteration = 0; iteration < 64; ++iteration) {
            auto lhs = random_limb(seed);
            auto rhs = random_limb(seed);
            auto modulus = random_limb(seed);
            if (modulus.is_zero()) continue;
            auto positive_mod = modulus.is_negative() ? -modulus : modulus;
            auto positive_lhs = lhs % positive_mod;
            if (positive_lhs.is_negative()) positive_lhs = positive_lhs + positive_mod;
            auto positive_rhs = rhs % positive_mod;
            if (positive_rhs.is_negative()) positive_rhs = positive_rhs + positive_mod;
            MontgomeryContext ctx(positive_mod);
            auto wide = T81Limb::mul_wide(positive_lhs, positive_rhs);
            auto reduced = MontgomeryContextTestAccess::reduce(ctx, wide);
            auto expected = (positive_lhs * positive_rhs) % positive_mod;
            ensure(reduced.compare(expected) == 0, "Montgomery reduce mismatch");
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
