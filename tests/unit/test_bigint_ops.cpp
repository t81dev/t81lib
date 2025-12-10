#include <t81/t81lib.hpp>
#include <t81/io/format.hpp>
#include <t81/io/parse.hpp>
#include <t81/core/bigint_bitops_helpers.hpp>

#include <algorithm>
#include <iostream>
#include <random>
#include <array>
#include <utility>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <numeric>

namespace {

t81::core::bigint random_small_bigint(std::mt19937_64& rng) {
    static std::uniform_int_distribution<std::int64_t> dist(-1'000'000, 1'000'000);
    return t81::core::bigint(dist(rng));
}

t81::core::bigint random_large_bigint(std::mt19937_64& rng) {
    auto value = random_small_bigint(rng);
    for (int i = 0; i < 4; ++i) {
        value = value.shift_limbs(1);
        value += random_small_bigint(rng);
    }
    return value;
}

using t81::core::expected_bitwise;
using t81::core::expected_not;
using t81::core::expected_tryte_shift_left;
using t81::core::expected_tryte_shift_right;
using t81::core::expected_trit_shift_left;
using t81::core::expected_trit_shift_right;

bool check_equal(const t81::core::bigint& lhs,
                 const t81::core::bigint& rhs,
                 std::string_view label) {
    if (lhs == rhs) {
        return true;
    }
    std::cerr << label << " mismatch: " << t81::io::to_string(lhs) << " != "
              << t81::io::to_string(rhs) << "\n";
    return false;
}

bool test_string_roundtrip(std::mt19937_64& rng) {
    const std::vector<int> bases = {3, 7, 10, 16, 27, 81};
    for (int base : bases) {
        for (int iteration = 0; iteration < 16; ++iteration) {
            const auto original = random_small_bigint(rng);
            const std::string text = t81::io::to_string(original, base);
            const auto parsed = t81::io::from_string<t81::core::bigint>(text, base);
            if (!check_equal(original, parsed, "string roundtrip")) {
                return false;
            }
        }
    }
    const auto limb_value =
        t81::io::from_string<t81::core::limb>("42", 10);
    const auto expected = t81::core::limb::from_value(42);
    if (limb_value != expected) {
        std::cerr << "limb parser mismatch\n";
        return false;
    }
    return true;
}

bool test_base81_digits() {
    const std::vector<std::pair<t81::core::bigint, std::string>> cases = {
        {t81::core::bigint(10), "a"},
        {t81::core::bigint(36), "A"},
        {t81::core::bigint(62), "!"},
        {t81::core::bigint(80), "?"}
    };
    for (const auto& [value, repr] : cases) {
        const std::string encoded = t81::io::to_string(value, 81);
        if (encoded != repr) {
            std::cerr << "base81 encoding mismatch for " << t81::io::to_string(value) << "\n";
            return false;
        }
        const auto decoded = t81::io::from_string<t81::core::bigint>(repr, 81);
        if (!check_equal(value, decoded, "base81 digit " + repr)) {
            return false;
        }
    }
    return true;
}

bool test_add_sub_properties(std::mt19937_64& rng) {
    for (int iteration = 0; iteration < 32; ++iteration) {
        const auto a = random_small_bigint(rng);
        const auto b = random_small_bigint(rng);
        const auto sum = a + b;
        const auto diff = sum - b;
        if (!check_equal(diff, a, "addition identity")) {
            return false;
        }
        const auto second = sum - a;
        if (!check_equal(second, b, "addition identity 2")) {
            return false;
        }
        const auto signed_diff = a - b;
        const auto recalc = signed_diff + b;
        if (!check_equal(recalc, a, "subtraction identity")) {
            return false;
        }
    }
    return true;
}

bool test_div_mod(std::mt19937_64& rng) {
    const auto two = t81::core::bigint(2);
    for (int iteration = 0; iteration < 32; ++iteration) {
        auto dividend = random_small_bigint(rng);
        auto divisor = random_small_bigint(rng);
        if (divisor.is_zero()) {
            divisor = t81::core::bigint(1);
        }
        const auto [quotient, remainder] = t81::core::bigint::div_mod(dividend, divisor);
        const auto recomposed = quotient * divisor + remainder;
        if (!check_equal(recomposed, dividend, "div_mod reconstruction")) {
            return false;
        }
        if (!remainder.is_zero()) {
            if (remainder.is_negative() != dividend.is_negative()) {
                std::cerr << "remainder sign does not match dividend\n";
                return false;
            }
            if (!(remainder.abs() < divisor.abs())) {
                std::cerr << "remainder magnitude not less than divisor\n";
                return false;
            }
        }
        const auto [quotient2, remainder2] = t81::core::bigint::div_mod(dividend, two);
        if (quotient2.is_zero() && remainder2.is_zero() && dividend.is_zero()) {
            continue;
        }
        if (!check_equal(quotient2 * two + remainder2, dividend, "div_mod small divisor")) {
            return false;
        }
    }
    return true;
}

bool test_divide_magnitude_large() {
    using bigint = t81::core::bigint;
    const auto divisor = t81::io::from_string<bigint>("121932631112635269");
    auto dividend = divisor;
    for (int iteration = 0; iteration < 3; ++iteration) {
        dividend = dividend.shift_limbs(1);
        dividend += divisor;
    }
    const auto [quotient, remainder] = bigint::div_mod(dividend, divisor);
    if (!check_equal(quotient * divisor + remainder, dividend, "divide_magnitude large recomposition")) {
        return false;
    }
    if (remainder.is_negative()) {
        std::cerr << "divide_magnitude remainder should never be negative\n";
        return false;
    }
    if (!(remainder.abs() < divisor.abs())) {
        std::cerr << "divide_magnitude remainder magnitude not less than divisor\n";
        return false;
    }
    if (quotient.limb_count() <= divisor.limb_count()) {
        std::cerr << "divide_magnitude quotient unexpectedly small\n";
        return false;
    }
    return true;
}

bool test_div_mod_scaled_multiples() {
    using bigint = t81::core::bigint;
    const auto divisor = t81::io::from_string<bigint>("1234567");
    bigint dividend = bigint::zero();
    bigint multiplier = bigint::one();
    bigint expected_quotient = bigint::zero();
    for (int index = 0; index < 6; ++index) {
        dividend += divisor * multiplier;
        expected_quotient += multiplier;
        multiplier += multiplier;
    }
    const auto [quotient, remainder] = bigint::div_mod(dividend, divisor);
    if (!check_equal(quotient, expected_quotient, "div_mod scaled quotient")) {
        return false;
    }
    if (!remainder.is_zero()) {
        std::cerr << "div_mod scaled multiples remainder should be zero\n";
        return false;
    }
    return true;
}

bool test_gcd_cases() {
    using bigint = t81::core::bigint;
    const auto zero = bigint::zero();
    if (!check_equal(bigint::gcd(zero, zero), zero, "gcd(0,0)")) {
        return false;
    }
    const auto positive = t81::io::from_string<bigint>("385");
    if (!check_equal(bigint::gcd(zero, positive), positive, "gcd(0,b)")) {
        return false;
    }
    if (!check_equal(bigint::gcd(positive, zero), positive, "gcd(a,0)")) {
        return false;
    }
    const std::int64_t a_value = 2 * 3 * 5 * 7 * 11 * 13;
    const std::int64_t b_value = 3 * 7 * 17 * 19;
    const auto expect_value = std::gcd(a_value, b_value);
    const bigint a = bigint(a_value);
    const bigint b = bigint(b_value);
    if (!check_equal(bigint::gcd(a, b), bigint(expect_value), "gcd(kusama)")) {
        return false;
    }
    const bigint negative = bigint(-b_value);
    if (!check_equal(bigint::gcd(a, negative), bigint(expect_value), "gcd with negative operand")) {
        return false;
    }
    const auto base_common = t81::io::from_string<bigint>("231");
    const auto multiplier_a =
        t81::io::from_string<bigint>("12345678901234567890");
    const auto multiplier_b =
        t81::io::from_string<bigint>("12345678901234567891");
    const bigint large_a = base_common * multiplier_a;
    const bigint large_b = base_common * multiplier_b;
    if (!check_equal(bigint::gcd(large_a, large_b), base_common, "gcd large multiples")) {
        return false;
    }
    return true;
}

bool test_mod_pow_cases() {
    using bigint = t81::core::bigint;
    const bigint modulus = bigint(1000);
    if (!check_equal(bigint::mod_pow(bigint(2), bigint(10), modulus), bigint(24), "mod_pow small")) {
        return false;
    }
    if (!check_equal(bigint::mod_pow(bigint(5), bigint(0), modulus), bigint(1), "mod_pow exponent zero")) {
        return false;
    }
    if (!check_equal(bigint::mod_pow(bigint(-3), bigint(3), bigint(50)), bigint(23), "mod_pow negative base")) {
        return false;
    }
    const bigint large_mod = t81::io::from_string<bigint>("1000003");
    if (!check_equal(bigint::mod_pow(bigint(2), bigint(20), large_mod), bigint(48573), "mod_pow large modulus")) {
        return false;
    }
    bool caught_negative_exponent = false;
    try {
        (void)bigint::mod_pow(bigint(2), bigint(-1), bigint(17));
    } catch (const std::domain_error&) {
        caught_negative_exponent = true;
    }
    if (!caught_negative_exponent) {
        std::cerr << "mod_pow must reject negative exponent\n";
        return false;
    }
    bool caught_negative_mod = false;
    try {
        (void)bigint::mod_pow(bigint(2), bigint(3), bigint(-5));
    } catch (const std::domain_error&) {
        caught_negative_mod = true;
    }
    if (!caught_negative_mod) {
        std::cerr << "mod_pow must reject negative modulus\n";
        return false;
    }
    return true;
}

bool test_bitwise_ops(std::mt19937_64& rng) {
    for (int iteration = 0; iteration < 32; ++iteration) {
        const auto a = random_small_bigint(rng);
        const auto b = random_small_bigint(rng);
        const auto limb_a = a.to_limb();
        const auto limb_b = b.to_limb();
        if (!check_equal(a & b, t81::core::bigint(limb_a & limb_b), "bitwise &")) {
            return false;
        }
        if (!check_equal(a | b, t81::core::bigint(limb_a | limb_b), "bitwise |")) {
            return false;
        }
        if (!check_equal(a ^ b, t81::core::bigint(limb_a ^ limb_b), "bitwise ^")) {
            return false;
        }
        if (!check_equal(a.consensus(b), t81::core::bigint(limb_a.consensus(limb_b)), "bitwise consensus")) {
            return false;
        }
        if (!check_equal(~a, -(a + t81::core::bigint::one()), "bitwise ~")) {
            return false;
        }

        const auto large_a = random_large_bigint(rng);
        const auto large_b = random_large_bigint(rng);
        if (!check_equal(large_a & large_b,
                         expected_bitwise(large_a, large_b, [](const t81::core::limb& lhs,
                                                                  const t81::core::limb& rhs) {
                             return lhs & rhs;
                         }),
                         "large bitwise &")) {
            return false;
        }
        if (!check_equal(large_a | large_b,
                         expected_bitwise(large_a, large_b, [](const t81::core::limb& lhs,
                                                                  const t81::core::limb& rhs) {
                             return lhs | rhs;
                         }),
                         "large bitwise |")) {
            return false;
        }
        if (!check_equal(large_a ^ large_b,
                         expected_bitwise(large_a, large_b, [](const t81::core::limb& lhs,
                                                                  const t81::core::limb& rhs) {
                             return lhs ^ rhs;
                         }),
                         "large bitwise ^")) {
            return false;
        }
        if (!check_equal(large_a.consensus(large_b),
                         expected_bitwise(large_a, large_b, [](const t81::core::limb& lhs,
                                                                  const t81::core::limb& rhs) {
                             return lhs.consensus(rhs);
                         }),
                         "large bitwise consensus")) {
            return false;
        }
        if (!check_equal(~large_a, expected_not(large_a), "large bitwise ~")) {
            return false;
        }
    }
    return true;
}

bool test_shift_ops(std::mt19937_64& rng) {
    const std::array<int, 5> tryte_shifts = {0, 1, 2, 3, 4};
    const std::array<int, 5> trit_shifts = {0, 1, 5, 12, 30};
    for (int iteration = 0; iteration < 32; ++iteration) {
        const auto value = random_large_bigint(rng);
        for (int count : tryte_shifts) {
            if (!check_equal(value << count,
                             expected_tryte_shift_left(value, count),
                             "tryte shift left multi")) {
                return false;
            }
            if (!check_equal(value >> count,
                             expected_tryte_shift_right(value, count),
                             "tryte shift right multi")) {
                return false;
            }
        }
        for (int count : trit_shifts) {
            if (!check_equal(value.trit_shift_left(count),
                             expected_trit_shift_left(value, count),
                             "trit shift left multi")) {
                return false;
            }
            if (!check_equal(value.trit_shift_right(count),
                             expected_trit_shift_right(value, count),
                             "trit shift right multi")) {
                return false;
            }
            if (!check_equal(value.rotate_left_tbits(count),
                             value.trit_shift_left(count),
                             "rotate left alias")) {
                return false;
            }
            if (!check_equal(value.rotate_right_tbits(count),
                             value.trit_shift_right(count),
                             "rotate right alias")) {
                return false;
            }
        }
    }
    return true;
}

bool test_integral_conversions() {
    const auto positive = t81::core::bigint(1'234'567'890);
    const auto negative = t81::core::bigint(-42);
    if (static_cast<int>(positive) != 1'234'567'890) {
        return false;
    }
    if (static_cast<long long>(positive) != 1'234'567'890LL) {
        return false;
    }
    if (static_cast<int>(negative) != -42) {
        return false;
    }
    bool threw = false;
    try {
        (void)static_cast<unsigned long long>(negative);
    } catch (const std::overflow_error&) {
        threw = true;
    }
    if (!threw) {
        return false;
    }
    bool overflowed = false;
    try {
        (void)static_cast<long long>(positive.shift_limbs(2));
    } catch (const std::overflow_error&) {
        overflowed = true;
    }
    if (!overflowed) {
        return false;
    }
    return true;
}

bool test_signed_limb_reconstruction(std::mt19937_64& rng) {
    for (int iteration = 0; iteration < 32; ++iteration) {
        const auto value = random_large_bigint(rng);
        const auto digits = value.signed_limbs();
        const auto reconstructed = t81::core::bigint::from_signed_limbs(digits);
        if (!check_equal(value, reconstructed, "signed limb reconstruction")) {
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    std::mt19937_64 rng(0xc0ffee123);
    if (!test_string_roundtrip(rng)) {
        return 1;
    }
    if (!test_base81_digits()) {
        return 1;
    }
    if (!test_add_sub_properties(rng)) {
        return 1;
    }
    if (!test_div_mod(rng)) {
        return 1;
    }
    if (!test_divide_magnitude_large()) {
        return 1;
    }
    if (!test_div_mod_scaled_multiples()) {
        return 1;
    }
    if (!test_bitwise_ops(rng)) {
        return 1;
    }
    if (!test_shift_ops(rng)) {
        return 1;
    }
    if (!test_signed_limb_reconstruction(rng)) {
        return 1;
    }
    if (!test_integral_conversions()) {
        return 1;
    }
    if (!test_gcd_cases()) {
        return 1;
    }
    if (!test_mod_pow_cases()) {
        return 1;
    }
    std::cout << "bigint_ops passed\n";
    return 0;
}
