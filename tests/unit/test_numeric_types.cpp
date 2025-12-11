// tests/unit/test_numeric_types.cpp — Exercises the new umbrella numeric helpers.

#include <t81/t81lib.hpp>

#include <iostream>

namespace {

auto limb_from = [](int value) {
    return t81::core::limb::from_value(value);
};

bool test_float() {
    auto f = t81::Float(limb_from(2), 1);
    auto scaled = f.scaled_trytes(1);
    if (scaled.exponent() != 4) {
        std::cerr << "Float scaled exponents" << std::endl;
        return false;
    }
    if (!t81::Float::zero().is_zero()) {
        std::cerr << "Float zero" << std::endl;
        return false;
    }
    return true;
}

bool test_ratio() {
    t81::Ratio half(limb_from(2), limb_from(4));
    if (half.numerator() != t81::core::bigint(1) ||
        half.denominator() != t81::core::bigint(2)) {
        std::cerr << "Ratio normalization" << std::endl;
        return false;
    }
    t81::Ratio sum = half + t81::Ratio(limb_from(1));
    if (sum.numerator() != t81::core::bigint(3) ||
        sum.denominator() != t81::core::bigint(2)) {
        std::cerr << "Ratio addition" << std::endl;
        return false;
    }
    return true;
}

bool test_fixed() {
    t81::Fixed<3> lhs(t81::core::bigint(5));
    t81::Fixed<3> rhs(t81::core::bigint(10));
    auto sum = lhs + rhs;
    if (sum.to_bigint() != t81::core::bigint(-12)) {
        std::cerr << "Fixed wrap" << std::endl;
        return false;
    }
    return true;
}

} // namespace

int main() {
    if (!test_float()) return 1;
    if (!test_ratio()) return 1;
    if (!test_fixed()) return 1;
    return 0;
}
