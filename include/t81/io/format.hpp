#pragma once

#include <algorithm>
#include <ostream>
#include <stdexcept>
#include <string>

#include <t81/core/bigint.hpp>
#include <t81/core/detail/base_digits.hpp>

namespace t81::io {

inline std::string to_string(const t81::core::limb& value, int base = 10) {
    return value.to_string(base);
}

inline std::string to_string(const t81::core::bigint& value, int base = 10) {
    if (!t81::core::detail::base81_supports_base(base)) {
        throw std::invalid_argument("supported bases are 2..81");
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
        digits.push_back(t81::core::detail::base81_digit_char(digit_value));
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
