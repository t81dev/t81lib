#pragma once

#include <string_view>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <t81/core/bigint.hpp>
#include <t81/core/detail/base_digits.hpp>

namespace t81::io {

inline t81::core::bigint parse_base81_bigint(std::string_view text) {
    if (text.empty()) {
        throw std::invalid_argument("empty string");
    }
    bool negative = false;
    std::size_t index = 0;
    if (text[0] == '+' || text[0] == '-') {
        negative = (text[0] == '-');
        ++index;
        if (index == text.size()) {
            throw std::invalid_argument("string has only a sign");
        }
    }
    const std::string_view digits = text.substr(index);
    if (digits.empty()) {
        throw std::invalid_argument("empty string");
    }
    constexpr std::size_t chunk_size = t81::core::limb::BASE81_DIGITS_PER_LIMB;
    std::vector<t81::core::limb> limbs;
    limbs.reserve((digits.size() + chunk_size - 1) / chunk_size);
    std::size_t pos = digits.size();
    while (pos > 0) {
        const std::size_t start = pos >= chunk_size ? pos - chunk_size : 0;
        const std::string_view chunk = digits.substr(start, pos - start);
        limbs.push_back(t81::core::limb::from_base81_digits(chunk));
        pos = start;
    }
    while (!limbs.empty() && limbs.back().is_zero()) {
        limbs.pop_back();
    }
    if (limbs.empty()) {
        limbs.push_back(t81::core::limb::zero());
    }
    return t81::core::bigint::from_limbs(std::move(limbs), negative);
}

template <typename Int>
inline Int from_string(std::string_view text, int base = 10) {
    static_assert(
        std::is_same_v<Int, t81::core::limb> || std::is_same_v<Int, t81::core::bigint>,
        "from_string supports limb and bigint");
    if constexpr (std::is_same_v<Int, t81::core::limb>) {
        return t81::core::limb::from_string(text, base);
    } else {
        if (base == 81) {
            return parse_base81_bigint(text);
        }
        if (base < 2 || base > 36) {
            throw std::invalid_argument("supported bases are 2..36");
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
