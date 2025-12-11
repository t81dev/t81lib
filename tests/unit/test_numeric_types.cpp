// tests/unit/test_numeric_types.cpp — Exercises the new umbrella numeric helpers.

#include <t81/t81lib.hpp>

#include <compare>
#include <iostream>
#include <vector>
namespace {

using Limb = t81::core::limb;
using BigInt = t81::core::bigint;

Limb limb_from(int value) {
    return Limb::from_value(value);
}

bool test_float() {
    std::cerr << "test_float start" << std::endl;
    t81::Float normalized(limb_from(9), 0);
    if (normalized.mantissa() != Limb::one() || normalized.exponent() != -2) {
        std::cerr << "Float normalization" << std::endl;
        return false;
    }
    std::cerr << "float normalized ok" << std::endl;
    t81::Float scaled = t81::Float(limb_from(1), 0).scaled_trytes(1);
    if (scaled.exponent() != 3) {
        std::cerr << "Float scaled trytes" << std::endl;
        return false;
    }
    if (!t81::Float::zero().is_zero()) {
        std::cerr << "Float zero" << std::endl;
        return false;
    }
    std::cerr << "test_float end" << std::endl;
    return true;
}

bool test_ratio() {
    std::cerr << "test_ratio start" << std::endl;
    t81::Ratio half(limb_from(2), limb_from(4));
    if (half.numerator() != BigInt(1) || half.denominator() != BigInt(2)) {
        std::cerr << "Ratio normalization" << std::endl;
        return false;
    }
    t81::Ratio sum = half + t81::Ratio(limb_from(1));
    if (sum.numerator() != BigInt(3) || sum.denominator() != BigInt(2)) {
        std::cerr << "Ratio addition" << std::endl;
        return false;
    }
    std::cerr << "ratio arithmetic ok" << std::endl;
    auto comparison = half <=> t81::Ratio(limb_from(3), limb_from(4));
    if (comparison != std::strong_ordering::less) {
        std::cerr << "Ratio comparison" << std::endl;
        return false;
    }
    std::cerr << "test_ratio end" << std::endl;
    return true;
}

bool test_fixed() {
    std::cerr << "test_fixed start" << std::endl;
    t81::Fixed<3> lhs(BigInt(5));
    t81::Fixed<3> rhs(BigInt(10));
    auto sum = lhs + rhs;
    if (sum.to_bigint() != BigInt(-12)) {
        std::cerr << "Fixed wrap" << std::endl;
        return false;
    }
    std::cerr << "test_fixed end" << std::endl;
    return true;
}

bool test_montgomery_int() {
    std::cerr << "test_montgomery start" << std::endl;
    t81::Modulus modulus(limb_from(7));
    std::cerr << "constructed modulus" << std::endl;
    t81::MontgomeryInt left(modulus, limb_from(5));
    std::cerr << "constructed left" << std::endl;
    t81::MontgomeryInt right(modulus, limb_from(6));
    std::cerr << "constructed right" << std::endl;
    left += right;
    std::cerr << "after add" << std::endl;
    if (left.to_limb() != limb_from(4)) {
        std::cerr << "Montgomery addition" << std::endl;
        return false;
    }
    std::cerr << "addition ok" << std::endl;
    left -= right;
    std::cerr << "after subtract" << std::endl;
    if (left.to_limb() != limb_from(5)) {
        std::cerr << "Montgomery subtraction" << std::endl;
        return false;
    }
    std::cerr << "subtraction ok" << std::endl;
    std::cerr << "test_montgomery end" << std::endl;
    return true;
}

bool test_complex() {
    std::cerr << "test_complex start" << std::endl;
    t81::Complex<int> lhs(2, 3);
    t81::Complex<int> rhs(-1, 5);
    auto sum = lhs + rhs;
    if (sum.real() != 1 || sum.imag() != 8) {
        std::cerr << "Complex addition" << std::endl;
        return false;
    }
    auto in_place = lhs;
    in_place += rhs;
    if (in_place.real() != sum.real() || in_place.imag() != sum.imag()) {
        std::cerr << "Complex compound addition" << std::endl;
        return false;
    }
    auto difference = sum - rhs;
    if (difference.real() != lhs.real() || difference.imag() != lhs.imag()) {
        std::cerr << "Complex subtraction" << std::endl;
        return false;
    }
    auto product = difference * rhs;
    if (product.real() != -17 || product.imag() != 7) {
        std::cerr << "Complex multiplication" << std::endl;
        return false;
    }
    std::cerr << "test_complex end" << std::endl;
    return true;
}

bool test_polynomial() {
    std::cerr << "test_polynomial start" << std::endl;
    t81::Polynomial<int> linear(std::vector<int>{1, 2});
    t81::Polynomial<int> quadratic(std::vector<int>{-1, 1, 1});
    auto sum = linear + quadratic;
    const auto& sum_coeffs = sum.coefficients();
    if (sum_coeffs.size() != 3 || sum_coeffs[0] != 0 || sum_coeffs[1] != 3 || sum_coeffs[2] != 1) {
        std::cerr << "Polynomial addition" << std::endl;
        return false;
    }
    auto product = linear * quadratic;
    const auto& product_coeffs = product.coefficients();
    if (product_coeffs.size() != 4 || product_coeffs[0] != -1 || product_coeffs[1] != -1 ||
        product_coeffs[2] != 3 || product_coeffs[3] != 2) {
        std::cerr << "Polynomial multiplication" << std::endl;
        return false;
    }
    if (quadratic.evaluate(2) != 5) {
        std::cerr << "Polynomial evaluation" << std::endl;
        return false;
    }
    std::cerr << "test_polynomial end" << std::endl;
    return true;
}

bool test_f2m() {
    std::cerr << "test_f2m start" << std::endl;
    BigInt modulus(0b1011);
    t81::F2m field(modulus);
    BigInt element1(0b110);
    BigInt element2(0b101);
    auto sum = field.add(element1, element2);
    if (sum != BigInt(0b011)) {
        std::cerr << "F2m addition" << std::endl;
        return false;
    }
    auto product = field.multiply(element1, element2);
    if (product != BigInt(0b11)) {
        std::cerr << "F2m multiplication" << std::endl;
        return false;
    }
    auto square = field.pow(element1, 2);
    if (square != BigInt(0b10)) {
        std::cerr << "F2m powering" << std::endl;
        return false;
    }
    auto reduced = field.reduce(BigInt(0b10000));
    if (reduced != BigInt(0b110)) {
        std::cerr << "F2m reduction" << std::endl;
        return false;
    }
    std::cerr << "test_f2m end" << std::endl;
    return true;
}

} // namespace

int main() {
    if (!test_float()) return 1;
    if (!test_ratio()) return 1;
    if (!test_fixed()) return 1;
    if (!test_montgomery_int()) return 1;
    if (!test_complex()) return 1;
    if (!test_polynomial()) return 1;
    if (!test_f2m()) return 1;
    return 0;
}
