#pragma once

#include <string_view>
#include <stdexcept>
#include <type_traits>

#include <t81/core/bigint.hpp>
#include <t81/core/detail/base_digits.hpp>

namespace t81::io {

template <typename Int>
inline Int from_string(std::string_view text, int base = 10) {
    static_assert(
        std::is_same_v<Int, t81::core::limb> || std::is_same_v<Int, t81::core::bigint>,
        "from_string supports limb and bigint");
    if constexpr (std::is_same_v<Int, t81::core::limb>) {
        return t81::core::limb::from_string(text, base);
    } else {
        if (!t81::core::detail::base81_supports_base(base)) {
            throw std::invalid_argument("supported bases are 2..81");
        }
        if (text.empty()) {
            throw std::invalid_argument("empty string");
        }
        std::size_t index = 0;
        bool negative = false;
        if (text[0] == '+' || text[0] == '-') {
            negative = (text[0] == '-');
            ++index;
            if (index == text.size()) {
                throw std::invalid_argument("string has only a sign");
            }
        }
        t81::core::bigint accumulator;
        const t81::core::bigint base_value(base);
        for (; index < text.size(); ++index) {
            const char ch = text[index];
            const int digit = t81::core::detail::base81_digit_value(ch, base);
            if (digit < 0 || digit >= base) {
                throw std::invalid_argument("invalid digit in string");
            }
            accumulator *= base_value;
            accumulator += t81::core::bigint(digit);
        }
        if (negative) {
            accumulator = -accumulator;
        }
        return accumulator;
    }
}

} // namespace t81::io
