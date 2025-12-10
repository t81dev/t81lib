#include <t81/t81lib.hpp>
#include <t81/io/format.hpp>
#include <t81/io/parse.hpp>

#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

t81::core::bigint random_small_bigint(std::mt19937_64& rng) {
    static std::uniform_int_distribution<std::int64_t> dist(-1'000'000, 1'000'000);
    return t81::core::bigint(dist(rng));
}

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
    const std::vector<int> bases = {3, 7, 10, 16, 27};
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

} // namespace

int main() {
    std::mt19937_64 rng(0xc0ffee123);
    if (!test_string_roundtrip(rng)) {
        return 1;
    }
    if (!test_add_sub_properties(rng)) {
        return 1;
    }
    if (!test_div_mod(rng)) {
        return 1;
    }
    std::cout << "bigint_ops passed\n";
    return 0;
}
