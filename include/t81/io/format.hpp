#pragma once

#include <algorithm>
#include <ostream>
#include <stdexcept>
#include <string>

#include <t81/core/bigint.hpp>

namespace t81::io {

inline std::string to_string(const t81::core::limb& value, int base = 10) {
    return value.to_string(base);
}

inline std::string to_string(const t81::core::bigint& value, int base = 10) {
    if (base < 2 || base > 36) {
        throw std::invalid_argument("supported bases are 2..36");
    }
    if (value.is_zero()) {
        return "0";
    }
    const bool negative = value.is_negative();
    t81::core::bigint cursor = value.abs();
    const t81::core::bigint base_value(base);
    std::string digits;
    while (!cursor.is_zero()) {
        const auto [quotient, remainder] = t81::core::bigint::div_mod(cursor, base_value);
        cursor = quotient;
        const t81::core::bigint absolute_remainder = remainder.abs();
        const auto digit_limb = absolute_remainder.to_limb();
        const int digit_value = digit_limb.to_integer<int>();
        const char digit_char = digit_value < 10
            ? static_cast<char>('0' + digit_value)
            : static_cast<char>('a' + (digit_value - 10));
        digits.push_back(digit_char);
    }
    if (negative) {
        digits.push_back('-');
    }
    std::reverse(digits.begin(), digits.end());
    return digits;
}

inline std::ostream& operator<<(std::ostream& os, const t81::core::limb& value) {
    return os << to_string(value);
}

inline std::ostream& operator<<(std::ostream& os, const t81::core::bigint& value) {
    return os << to_string(value);
}

} // namespace t81::io
