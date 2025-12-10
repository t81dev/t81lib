#pragma once

#include <array>
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
    t81::core::bigint accumulator = t81::core::bigint::zero();
    const t81::core::bigint base_value(81);
    for (; index < text.size(); ++index) {
        const char ch = text[index];
        const int digit = t81::core::detail::base81_digit_value(ch, 81);
        if (digit < 0) {
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
        if (base == 10) {
            const std::string_view digits = text.substr(index);
            if (digits.empty()) {
                throw std::invalid_argument("empty string");
            }
            constexpr int DECIMAL_CHUNK_DIGITS = 9;
            static const auto decimal_powers = [] {
                std::array<t81::core::bigint, DECIMAL_CHUNK_DIGITS + 1> values{};
                values[0] = t81::core::bigint(1);
                for (int len = 1; len <= DECIMAL_CHUNK_DIGITS; ++len) {
                    values[len] = values[len - 1] * t81::core::bigint(10);
                }
                return values;
            }();
            t81::core::bigint accumulator = t81::core::bigint::zero();
            std::size_t pos = 0;
            const std::size_t total = digits.size();
            while (pos < total) {
                const int remaining = static_cast<int>(total - pos);
                const int chunk_len = (remaining % DECIMAL_CHUNK_DIGITS == 0)
                                          ? DECIMAL_CHUNK_DIGITS
                                          : remaining % DECIMAL_CHUNK_DIGITS;
                std::uint64_t chunk_value = 0;
                for (int offset = 0; offset < chunk_len; ++offset) {
                    const char ch = digits[pos + offset];
                    if (ch < '0' || ch > '9') {
                        throw std::invalid_argument("invalid digit in string");
                    }
                    chunk_value = chunk_value * 10 + static_cast<int>(ch - '0');
                }
                accumulator *= decimal_powers[chunk_len];
                if (chunk_value != 0) {
                    accumulator += t81::core::bigint(static_cast<std::int64_t>(chunk_value));
                }
                pos += static_cast<std::size_t>(chunk_len);
            }
            if (negative) {
                accumulator = -accumulator;
            }
            return accumulator;
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
