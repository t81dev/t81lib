// include/t81/core/detail/base_digits.hpp — Digit traits and base conversion helpers.

// detail/base_digits.hpp - Base-81 digit metadata for parsing/formatting routines.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace t81::core::detail {

    constexpr std::array<char, 19> BASE81_PUNCTUATION = {'!', '#', '$', '%', '&', '(', ')',
                                                         '*', '+', '-', ',', '.', '/', ';',
                                                         ':', '<', '=', '>', '?'};

    constexpr std::array<char, 81> make_base81_digits() {
        std::array<char, 81> digits{};
        std::size_t index = 0;
        for (char value = '0'; value <= '9'; ++value) {
            digits[index++] = value;
        }
        for (char value = 'a'; value <= 'z'; ++value) {
            digits[index++] = value;
        }
        for (char value = 'A'; value <= 'Z'; ++value) {
            digits[index++] = value;
        }
        for (char value : BASE81_PUNCTUATION) {
            digits[index++] = value;
        }
        return digits;
    }

    constexpr std::array<char, 81> BASE81_DIGITS = make_base81_digits();

    constexpr int base81_max_base() noexcept { return static_cast<int>(BASE81_DIGITS.size()); }

    constexpr bool base81_supports_base(int base) noexcept {
        return base >= 2 && base <= base81_max_base();
    }

    inline char base81_digit_char(int digit) noexcept { return BASE81_DIGITS[digit]; }

    inline int punctuation_index(char ch) noexcept {
        for (std::size_t index = 0; index < BASE81_PUNCTUATION.size(); ++index) {
            if (BASE81_PUNCTUATION[index] == ch) {
                return static_cast<int>(index);
            }
        }
        return -1;
    }

    inline int base81_digit_value(char ch, int base) noexcept {
        if (ch >= '0' && ch <= '9') {
            return ch - '0';
        }
        if (ch >= 'a' && ch <= 'z') {
            return 10 + (ch - 'a');
        }
        if (ch >= 'A' && ch <= 'Z') {
            if (base <= 36) {
                return 10 + (ch - 'A');
            }
            const int extended = 36 + (ch - 'A');
            if (extended < base) {
                return extended;
            }
            return 10 + (ch - 'A');
        }
        const int punctuation = punctuation_index(ch);
        if (punctuation >= 0) {
            const int value = 62 + punctuation;
            if (value < base) {
                return value;
            }
        }
        return -1;
    }

    inline constexpr int normalize_base81_sum(int sum) noexcept {
        const int digit = sum % 81;
        return digit < 0 ? digit + 81 : digit;
    }

    constexpr std::array<std::array<std::int8_t, 4>, 81> make_base81_trits() {
        std::array<std::array<std::int8_t, 4>, 81> result{};
        for (int t0 = -1; t0 <= 1; ++t0) {
            for (int t1 = -1; t1 <= 1; ++t1) {
                for (int t2 = -1; t2 <= 1; ++t2) {
                    for (int t3 = -1; t3 <= 1; ++t3) {
                        const int sum = t0 + 3 * t1 + 9 * t2 + 27 * t3;
                        const int digit = normalize_base81_sum(sum);
                        result[digit] = {static_cast<std::int8_t>(t0), static_cast<std::int8_t>(t1),
                                         static_cast<std::int8_t>(t2),
                                         static_cast<std::int8_t>(t3)};
                    }
                }
            }
        }
        return result;
    }

    constexpr auto BASE81_TRITS = make_base81_trits();

    inline int base81_digit_from_trits(const std::array<std::int8_t, 4> &trits) noexcept {
        int sum = 0;
        int weight = 1;
        for (int index = 0; index < 4; ++index) {
            sum += trits[index] * weight;
            weight *= 3;
        }
        return normalize_base81_sum(sum);
    }

    inline const std::array<std::int8_t, 4> &base81_trits_for_digit(int digit) noexcept {
        return BASE81_TRITS[digit];
    }

} // namespace t81::core::detail
