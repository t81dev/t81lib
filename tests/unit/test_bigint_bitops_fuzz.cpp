// tests/unit/test_bigint_bitops_fuzz.cpp — Fuzz-style coverage for bigint bitwise operations.

#include <t81/core/bigint_bitops_helpers.hpp>
#include <t81/t81lib.hpp>

#include <iostream>
#include <random>

namespace {

    inline t81::core::bigint random_small_bigint(std::mt19937_64 &rng) {
        static std::uniform_int_distribution<std::int64_t> dist(-1'000'000, 1'000'000);
        return t81::core::bigint(dist(rng));
    }

    inline t81::core::bigint random_bigint(std::mt19937_64 &rng, int limbs) {
        t81::core::bigint value;
        for (int index = 0; index < limbs; ++index) {
            value = value.shift_limbs(1);
            value += random_small_bigint(rng);
        }
        return value;
    }

    template <typename T>
    bool assert_equal(const t81::core::bigint &lhs, const t81::core::bigint &rhs, T label) {
        if (lhs == rhs) {
            return true;
        }
        std::cerr << label << " mismatch: " << t81::io::to_string(lhs)
                  << " != " << t81::io::to_string(rhs) << "\n";
        return false;
    }

    bool run_bitwise_fuzz(std::mt19937_64 &rng, int iterations) {
        std::uniform_int_distribution<int> limb_count(1, 8);
        for (int iteration = 0; iteration < iterations; ++iteration) {
            const auto lhs = random_bigint(rng, limb_count(rng));
            const auto rhs = random_bigint(rng, limb_count(rng));
            if (!assert_equal(
                    lhs & rhs,
                    t81::core::expected_bitwise(
                        lhs, rhs,
                        [](const t81::core::limb &a, const t81::core::limb &b) { return a & b; }),
                    "fuzz bitwise &")) {
                return false;
            }
            if (!assert_equal(
                    lhs | rhs,
                    t81::core::expected_bitwise(
                        lhs, rhs,
                        [](const t81::core::limb &a, const t81::core::limb &b) { return a | b; }),
                    "fuzz bitwise |")) {
                return false;
            }
            if (!assert_equal(
                    lhs ^ rhs,
                    t81::core::expected_bitwise(
                        lhs, rhs,
                        [](const t81::core::limb &a, const t81::core::limb &b) { return a ^ b; }),
                    "fuzz bitwise ^")) {
                return false;
            }
            if (!assert_equal(lhs.consensus(rhs),
                              t81::core::expected_bitwise(
                                  lhs, rhs,
                                  [](const t81::core::limb &a, const t81::core::limb &b) {
                                      return a.consensus(b);
                                  }),
                              "fuzz bitwise consensus")) {
                return false;
            }
            if (!assert_equal(~lhs, t81::core::expected_not(lhs), "fuzz bitwise ~")) {
                return false;
            }
        }
        return true;
    }

    bool run_shift_fuzz(std::mt19937_64 &rng, int iterations) {
        std::uniform_int_distribution<int> tryte_count(0, 12);
        std::uniform_int_distribution<int> trit_count(0, 160);
        std::uniform_int_distribution<int> limb_count(1, 8);
        for (int iteration = 0; iteration < iterations; ++iteration) {
            const auto value = random_bigint(rng, limb_count(rng));
            const int trytes = tryte_count(rng);
            if (!assert_equal(value << trytes, t81::core::expected_tryte_shift_left(value, trytes),
                              "fuzz tryte shift left")) {
                return false;
            }
            if (!assert_equal(value >> trytes, t81::core::expected_tryte_shift_right(value, trytes),
                              "fuzz tryte shift right")) {
                return false;
            }
            const int trits = trit_count(rng);
            if (!assert_equal(value.trit_shift_left(trits),
                              t81::core::expected_trit_shift_left(value, trits),
                              "fuzz trit shift left")) {
                return false;
            }
            if (!assert_equal(value.trit_shift_right(trits),
                              t81::core::expected_trit_shift_right(value, trits),
                              "fuzz trit shift right")) {
                return false;
            }
            if (!assert_equal(value.rotate_left_tbits(trits), value.trit_shift_left(trits),
                              "fuzz rotate left alias")) {
                return false;
            }
            if (!assert_equal(value.rotate_right_tbits(trits), value.trit_shift_right(trits),
                              "fuzz rotate right alias")) {
                return false;
            }
        }
        return true;
    }

} // namespace

int main() {
    std::mt19937_64 rng(0xc0ffee123);
    if (!run_bitwise_fuzz(rng, 8000)) {
        return 1;
    }
    if (!run_shift_fuzz(rng, 4000)) {
        return 1;
    }
    std::cout << "bigint_bitops_fuzz passed\n";
    return 0;
}
