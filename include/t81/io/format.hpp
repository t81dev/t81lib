// include/t81/io/format.hpp — Declarations for formatting bigint values.

#pragma once

#include <algorithm>
#include <ostream>
#include <stdexcept>
#include <string>

#include <t81/core/bigint.hpp>
#include <t81/core/detail/base_digits.hpp>

namespace t81::io {

    inline std::string to_string(const t81::core::limb &value, int base = 10) {
        return value.to_string(base);
    }

    inline std::string to_string(const t81::core::bigint &value, int base = 10) {
        if (base == 81) {
            if (value.is_zero()) {
                return "0";
            }
            const bool negative = value.is_negative();
            const auto digits = value.base81_digits();
            std::string result;
            result.reserve(digits.size() + (negative ? 1 : 0));
            if (negative) {
                result.push_back('-');
            }
            for (auto it = digits.rbegin(); it != digits.rend(); ++it) {
                result.push_back(t81::core::detail::base81_digit_char(*it));
            }
            return result;
        }
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
            const int digit_value = static_cast<int>(remainder.abs().to_limb().to_integer<int>());
            if (digit_value < 10) {
                digits.push_back(static_cast<char>('0' + digit_value));
            } else {
                digits.push_back(static_cast<char>('a' + digit_value - 10));
            }
        }
        if (negative) {
            digits.push_back('-');
        }
        std::reverse(digits.begin(), digits.end());
        return digits;
    }

    inline std::ostream &operator<<(std::ostream &os, const t81::core::limb &value) {
        return os << to_string(value);
    }

    inline std::ostream &operator<<(std::ostream &os, const t81::core::bigint &value) {
        return os << to_string(value);
    }

} // namespace t81::io
