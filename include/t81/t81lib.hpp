// include/t81/t81lib.hpp — Umbrella header that exposes t81lib components.

#pragma once

// Umbrella header for t81lib.
// Users should generally include only this file.

#include <t81/core/bigint.hpp>
#include <t81/core/limb.hpp>
#include <t81/core/montgomery.hpp>
#include <t81/core/montgomery_helpers.hpp>
#include <t81/gf2m.hpp>
#include <t81/io/format.hpp>
#include <t81/io/parse.hpp>
#include <t81/linalg/gemm.hpp>
#include <t81/util/debug.hpp>
#include <t81/util/random.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace t81 {

    namespace detail {
        inline const core::bigint &power_of_three(std::size_t exponent);
        inline std::string format_mantissa(core::bigint mantissa, int exponent, bool negative);
    } // namespace detail

    inline std::string_view trim(std::string_view text) noexcept {
        const auto is_whitespace = [](char ch) {
            return std::isspace(static_cast<unsigned char>(ch));
        };
        std::size_t start = 0;
        while (start < text.size() && is_whitespace(text[start])) {
            ++start;
        }
        std::size_t end = text.size();
        while (end > start && is_whitespace(text[end - 1])) {
            --end;
        }
        return text.substr(start, end - start);
    }

    using Int = core::limb;
    class Float {
      public:
        Float() noexcept = default;
        explicit Float(core::limb mantissa, int exponent = 0) noexcept
            : mantissa_(std::move(mantissa)), exponent_(exponent) {
            normalize();
        }

        static Float zero() noexcept {
            return {};
        }

        static Float from_string(std::string_view text) {
            if (text.empty()) {
                throw std::invalid_argument("float literal cannot be empty");
            }
            std::size_t index = 0;
            bool negative = false;
            if (text[index] == '+' || text[index] == '-') {
                negative = (text[index] == '-');
                ++index;
                if (index == text.size()) {
                    throw std::invalid_argument("float literal missing digits");
                }
            }

            core::bigint mantissa = core::bigint::zero();
            bool seen_decimal = false;
            std::size_t fractional_digits = 0;
            bool has_digit = false;
            const core::bigint three = core::bigint(core::limb::from_value(3));
            constexpr std::string_view OVERLINE_PREFIX("\xC2\xAF", 2);
            constexpr std::string_view PRECOMPOSED_OVERLINE("\xE2\x80\xBE", 3);
            constexpr std::string_view COMBINING_OVERLINE("\xCC\x85", 2);
            constexpr std::string_view TRYTE_EXPONENT_INDICATOR("\xE2\x82\x83", 3);

            const auto prefix_length = [&](std::size_t position) {
                if (position + OVERLINE_PREFIX.size() <= text.size() &&
                    text.substr(position, OVERLINE_PREFIX.size()) == OVERLINE_PREFIX) {
                    return OVERLINE_PREFIX.size();
                }
                if (position + PRECOMPOSED_OVERLINE.size() <= text.size() &&
                    text.substr(position, PRECOMPOSED_OVERLINE.size()) == PRECOMPOSED_OVERLINE) {
                    return PRECOMPOSED_OVERLINE.size();
                }
                return std::size_t{0};
            };

            constexpr std::string_view DECOMPOSED_TRYTE_INDICATOR("3\xCC\xB2", 3);
            const auto tryte_indicator_length = [&](std::size_t position) {
                if (position + TRYTE_EXPONENT_INDICATOR.size() <= text.size() &&
                    text.substr(position, TRYTE_EXPONENT_INDICATOR.size()) ==
                        TRYTE_EXPONENT_INDICATOR) {
                    return TRYTE_EXPONENT_INDICATOR.size();
                }
                if (position + DECOMPOSED_TRYTE_INDICATOR.size() <= text.size() &&
                    text.substr(position, DECOMPOSED_TRYTE_INDICATOR.size()) ==
                        DECOMPOSED_TRYTE_INDICATOR) {
                    return DECOMPOSED_TRYTE_INDICATOR.size();
                }
                return std::size_t{0};
            };

            enum class ExponentIndicator { None, Trit, Tryte };
            ExponentIndicator exponent_indicator = ExponentIndicator::None;

            while (index < text.size()) {
                if (text[index] == 'e' || text[index] == 'E') {
                    exponent_indicator = ExponentIndicator::Trit;
                    ++index;
                    break;
                }
                const std::size_t tryte_len = tryte_indicator_length(index);
                if (tryte_len > 0) {
                    exponent_indicator = ExponentIndicator::Tryte;
                    index += tryte_len;
                    break;
                }
                const char ch = text[index];
                if (ch == '.') {
                    if (seen_decimal) {
                        throw std::invalid_argument("float literal has multiple decimal points");
                    }
                    seen_decimal = true;
                    ++index;
                    continue;
                }
                std::size_t cursor = index;
                bool overlined = false;
                const std::size_t prefix_len = prefix_length(cursor);
                if (prefix_len > 0) {
                    overlined = true;
                    cursor += prefix_len;
                }
                if (cursor >= text.size()) {
                    throw std::invalid_argument("float literal ends after overline");
                }
                const char digit_char = text[cursor];
                if (digit_char < '0' || digit_char > '2') {
                    throw std::invalid_argument("invalid ternary digit");
                }
                ++cursor;
                if (!overlined && cursor + COMBINING_OVERLINE.size() <= text.size() &&
                    text.substr(cursor, COMBINING_OVERLINE.size()) == COMBINING_OVERLINE) {
                    overlined = true;
                    cursor += COMBINING_OVERLINE.size();
                }
                int digit_value = digit_char - '0';
                if (overlined) {
                    if (digit_char == '0') {
                        digit_value = 0;
                    } else {
                        digit_value = -digit_value;
                    }
                }
                index = cursor;
                mantissa *= three;
                if (digit_value != 0) {
                    mantissa += core::bigint(digit_value);
                }
                has_digit = true;
                if (seen_decimal) {
                    ++fractional_digits;
                    if (fractional_digits >
                        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                        throw std::overflow_error("too many fractional trits");
                    }
                }
            }
            if (!has_digit) {
                throw std::invalid_argument("float literal missing digits");
            }
            bool exponent_negative = false;
            std::size_t exponent_value = 0;
            if (exponent_indicator != ExponentIndicator::None) {
                if (index == text.size()) {
                    throw std::invalid_argument("float literal exponent missing digits");
                }
                if (text[index] == '+' || text[index] == '-') {
                    exponent_negative = (text[index] == '-');
                    ++index;
                }
                if (index == text.size()) {
                    throw std::invalid_argument("float literal exponent missing digits");
                }
                const auto max_before_mul =
                    static_cast<std::size_t>(std::numeric_limits<int>::max()) / 10;
                const auto remainder_limit =
                    static_cast<std::size_t>(std::numeric_limits<int>::max()) % 10;
                while (index < text.size()) {
                    const char digit_char = text[index];
                    if (digit_char < '0' || digit_char > '9') {
                        throw std::invalid_argument("invalid float literal exponent digit");
                    }
                    const std::size_t digit = static_cast<std::size_t>(digit_char - '0');
                    if (exponent_value > max_before_mul ||
                        (exponent_value == max_before_mul && digit > remainder_limit)) {
                        throw std::overflow_error("float literal exponent too large");
                    }
                    exponent_value = exponent_value * 10 + digit;
                    ++index;
                }
            } else if (index != text.size()) {
                throw std::invalid_argument("unexpected character in float literal");
            }
            if (negative && !mantissa.is_zero()) {
                mantissa = -mantissa;
            }
            int exponent = static_cast<int>(fractional_digits);
            if (exponent_indicator != ExponentIndicator::None) {
                if (exponent_value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                    throw std::overflow_error("float literal exponent too large");
                }
                int exponent_units = static_cast<int>(exponent_value);
                if (exponent_indicator == ExponentIndicator::Tryte) {
                    if (exponent_units > std::numeric_limits<int>::max() / 3) {
                        throw std::overflow_error("float literal exponent too large");
                    }
                    exponent_units *= 3;
                }
                const int change = exponent_negative ? -exponent_units : exponent_units;
                const long long adjusted_exponent =
                    static_cast<long long>(exponent) - static_cast<long long>(change);
                if (adjusted_exponent < std::numeric_limits<int>::min() ||
                    adjusted_exponent > std::numeric_limits<int>::max()) {
                    throw std::overflow_error("float literal exponent overflow");
                }
                exponent = static_cast<int>(adjusted_exponent);
            }
            return Float(mantissa.to_limb(), exponent);
        }

        core::limb mantissa() const noexcept {
            return mantissa_;
        }
        int exponent() const noexcept {
            return exponent_;
        }
        bool is_zero() const noexcept {
            return mantissa_.is_zero();
        }

        Float scaled_trits(int trits) const noexcept {
            return Float(mantissa_, exponent_ + trits);
        }

        Float scaled_trytes(int trytes) const noexcept {
            return scaled_trits(trytes * 3);
        }

        explicit Float(int value) : Float(core::bigint(value).to_limb()) {
        }
        Float sqrt() const;

        friend Float operator+(Float lhs, const Float &rhs);
        friend Float operator-(Float lhs, const Float &rhs);
        friend Float operator/(Float lhs, const Float &rhs);

        // Ensure mantissa drops trailing zero trits after multiplication.
        friend Float operator*(Float lhs, const Float &rhs) noexcept {
            lhs.mantissa_ *= rhs.mantissa_;
            lhs.exponent_ += rhs.exponent_;
            lhs.normalize();
            return lhs;
        }

        friend bool operator==(const Float &lhs, const Float &rhs) noexcept {
            return lhs.mantissa_ == rhs.mantissa_ && lhs.exponent_ == rhs.exponent_;
        }

        friend std::strong_ordering operator<=>(const Float &lhs, const Float &rhs) noexcept {
            if (lhs.mantissa_ == rhs.mantissa_ && lhs.exponent_ == rhs.exponent_) {
                return std::strong_ordering::equal;
            }
            const long long exponent_delta = static_cast<long long>(lhs.exponent_) - rhs.exponent_;
            core::bigint left = core::bigint(lhs.mantissa_);
            core::bigint right = core::bigint(rhs.mantissa_);
            if (exponent_delta > 0) {
                right *= detail::power_of_three(static_cast<std::size_t>(exponent_delta));
            } else if (exponent_delta < 0) {
                left *= detail::power_of_three(static_cast<std::size_t>(-exponent_delta));
            }
            return left <=> right;
        }

      private:
        void normalize() noexcept {
            if (mantissa_.is_zero()) {
                exponent_ = 0;
                mantissa_ = core::limb::zero();
                return;
            }
            const core::limb divisor = core::limb::from_value(3);
            while (true) {
                const auto [quotient, remainder] = core::limb::div_mod(mantissa_, divisor);
                if (!remainder.is_zero()) {
                    break;
                }
                mantissa_ = quotient;
                --exponent_;
            }
        }
        core::limb mantissa_{core::limb::zero()};
        int exponent_ = 0;
    };

    template <int N> class Fixed;

    namespace detail {

        inline constexpr char OVERLINE_LEAD = '\xC2';
        inline constexpr char OVERLINE_TAIL = '\xAF';

        inline void append_overlined_digit(std::string &output, char digit) {
            output.push_back(OVERLINE_LEAD);
            output.push_back(OVERLINE_TAIL);
            output.push_back(digit);
        }

        inline void append_ternary_digit(std::string &output, int digit) {
            switch (digit) {
            case 0:
                output.push_back('0');
                return;
            case 1:
                output.push_back('1');
                return;
            case 2:
                output.push_back('2');
                return;
            case -1:
                append_overlined_digit(output, '1');
                return;
            case -2:
                append_overlined_digit(output, '2');
                return;
            default:
                throw std::logic_error("unexpected ternary digit");
            }
        }

        inline const core::bigint &power_of_three(std::size_t exponent) {
            static const core::bigint three = core::bigint(core::limb::from_value(3));
            static thread_local std::vector<core::bigint> cache{core::bigint::one()};
            while (cache.size() <= exponent) {
                cache.push_back(cache.back() * three);
            }
            return cache[exponent];
        }

        inline std::vector<int> balanced_digits(core::bigint value) {
            if (value.is_zero()) {
                return {0};
            }
            std::vector<int> digits;
            const core::bigint three = core::bigint(core::limb::from_value(3));
            while (!value.is_zero()) {
                auto [quotient, remainder] = core::bigint::div_mod(value, three);
                int digit = static_cast<int>(remainder.to_limb().to_value());
                if (digit > 1) {
                    digit -= 3;
                    quotient += core::bigint::one();
                } else if (digit < -1) {
                    digit += 3;
                    quotient -= core::bigint::one();
                }
                digits.push_back(digit);
                value = std::move(quotient);
            }
            return digits;
        }

        inline void trim_leading_digits(std::vector<int> &digits) {
            while (digits.size() > 1 && digits.back() == 0) {
                digits.pop_back();
            }
        }

        inline std::string format_mantissa(core::bigint mantissa, int exponent, bool negative) {
            std::vector<int> digits = balanced_digits(std::move(mantissa));
            if (exponent < 0) {
                const std::size_t shift = static_cast<std::size_t>(-exponent);
                digits.insert(digits.begin(), shift, 0);
                exponent = 0;
            }
            trim_leading_digits(digits);
            if (digits.empty()) {
                digits.push_back(0);
            }
            const std::size_t total = digits.size();
            const std::size_t fractional = exponent > 0 ? static_cast<std::size_t>(exponent) : 0;
            const std::size_t integer_digits = total > fractional ? total - fractional : 0;

            std::string output;
            if (negative) {
                output.push_back('-');
            }

            if (integer_digits == 0) {
                output.push_back('0');
            } else {
                for (std::size_t idx = integer_digits; idx-- > 0;) {
                    append_ternary_digit(output, digits[fractional + idx]);
                }
            }

            if (fractional > 0) {
                output.push_back('.');
                if (fractional > total) {
                    for (std::size_t pad = fractional - total; pad-- > 0;) {
                        output.push_back('0');
                    }
                    for (std::size_t idx = total; idx-- > 0;) {
                        append_ternary_digit(output, digits[idx]);
                    }
                } else {
                    for (std::size_t idx = fractional; idx-- > 0;) {
                        append_ternary_digit(output, digits[idx]);
                    }
                }
            }

            return output;
        }

    } // namespace detail

    inline std::string to_string(const Float &value,
                                 std::optional<std::size_t> max_fractional_digits = std::nullopt) {
        if (value.is_zero()) {
            return "0";
        }
        core::bigint mantissa = core::bigint(value.mantissa());
        const bool negative = mantissa.is_negative();
        if (negative) {
            mantissa = -mantissa;
        }
        int exponent = value.exponent();
        if (max_fractional_digits.has_value() && exponent > 0) {
            const std::size_t target_fractional = std::min(
                *max_fractional_digits, static_cast<std::size_t>(std::numeric_limits<int>::max()));
            const int target_exponent = static_cast<int>(target_fractional);
            if (exponent > target_exponent) {
                const std::size_t delta = static_cast<std::size_t>(exponent - target_exponent);
                const core::bigint divisor = detail::power_of_three(delta);
                const auto [quotient, remainder] = core::bigint::div_mod(mantissa, divisor);
                core::bigint rounded = quotient;
                const core::bigint double_remainder = remainder * core::bigint(2);
                if (double_remainder >= divisor) {
                    rounded += core::bigint::one();
                }
                mantissa = std::move(rounded);
                exponent = target_exponent;
            }
        }
        return detail::format_mantissa(std::move(mantissa), exponent, negative);
    }

    namespace literals {

        inline Float operator"" _t3(const char *literal, std::size_t length) {
            return Float::from_string(std::string_view(literal, length));
        }

    } // namespace literals

    template <int N> class FloatN {
      public:
        static_assert(N > 0, "FloatN requires positive trit width");

        using mantissa_type = Fixed<N>;

        constexpr FloatN() noexcept = default;
        constexpr explicit FloatN(mantissa_type mantissa, int exponent = 0)
            : mantissa_(std::move(mantissa)), exponent_(exponent) {
            normalize();
        }

        static constexpr FloatN zero() noexcept {
            return {};
        }

        constexpr const mantissa_type &mantissa() const noexcept {
            return mantissa_;
        }
        constexpr int exponent() const noexcept {
            return exponent_;
        }
        constexpr bool is_zero() const noexcept {
            return mantissa_.is_zero();
        }

        constexpr FloatN scaled_trits(int trits) const noexcept {
            return FloatN(mantissa_, exponent_ + trits, true);
        }

        constexpr FloatN scaled_trytes(int trytes) const noexcept {
            return scaled_trits(trytes * 3);
        }

        explicit FloatN(int value) : FloatN(mantissa_type(core::bigint(value)), 0) {
        }
        explicit FloatN(const Float &value)
            : FloatN(mantissa_type(core::bigint(value.mantissa())), value.exponent()) {
        }
        FloatN sqrt() const;

        friend FloatN operator+(FloatN lhs, const FloatN &rhs);
        friend FloatN operator-(FloatN lhs, const FloatN &rhs);
        friend FloatN operator/(FloatN lhs, const FloatN &rhs);

        friend constexpr FloatN operator*(FloatN lhs, const FloatN &rhs) noexcept {
            lhs.mantissa_ *= rhs.mantissa_;
            lhs.exponent_ += rhs.exponent_;
            lhs.normalize();
            return lhs;
        }

        friend constexpr bool operator==(const FloatN &lhs, const FloatN &rhs) noexcept {
            return lhs.mantissa_ == rhs.mantissa_ && lhs.exponent_ == rhs.exponent_;
        }

        friend constexpr bool operator!=(const FloatN &lhs, const FloatN &rhs) noexcept {
            return !(lhs == rhs);
        }

        friend std::strong_ordering operator<=>(const FloatN &lhs, const FloatN &rhs) noexcept {
            if (lhs.mantissa_ == rhs.mantissa_ && lhs.exponent_ == rhs.exponent_) {
                return std::strong_ordering::equal;
            }
            const long long exponent_delta = static_cast<long long>(lhs.exponent_) - rhs.exponent_;
            core::bigint left = lhs.mantissa().to_bigint();
            core::bigint right = rhs.mantissa().to_bigint();
            if (exponent_delta > 0) {
                right *= detail::power_of_three(static_cast<std::size_t>(exponent_delta));
            } else if (exponent_delta < 0) {
                left *= detail::power_of_three(static_cast<std::size_t>(-exponent_delta));
            }
            return left <=> right;
        }

      private:
        constexpr FloatN(mantissa_type mantissa, int exponent, bool skip_normalize) noexcept
            : mantissa_(std::move(mantissa)), exponent_(exponent) {
            if (!skip_normalize) {
                normalize();
            }
        }

        constexpr void normalize() noexcept {
            if (mantissa_.is_zero()) {
                exponent_ = 0;
                return;
            }
            while (mantissa_.divisible_by_three()) {
                mantissa_.divide_by_three();
                --exponent_;
            }
        }

        mantissa_type mantissa_{};
        int exponent_ = 0;
    };

    class Ratio {
      public:
        Ratio() noexcept = default;
        explicit Ratio(core::bigint numerator) : Ratio(std::move(numerator), core::bigint::one()) {
        }
        explicit Ratio(core::limb numerator) : Ratio(core::bigint(numerator)) {
        }
        Ratio(core::limb numerator, core::limb denominator)
            : Ratio(core::bigint(numerator), core::bigint(denominator)) {
        }
        Ratio(core::bigint numerator, core::bigint denominator) {
            normalize(std::move(numerator), std::move(denominator));
        }

        static Ratio zero() noexcept {
            return Ratio();
        }
        static Ratio from_float(const Float &value) {
            if (value.is_zero()) {
                return Ratio::zero();
            }
            core::bigint numerator = core::bigint(value.mantissa());
            core::bigint denominator = core::bigint::one();
            if (value.exponent() > 0) {
                numerator *= detail::power_of_three(static_cast<std::size_t>(value.exponent()));
            } else if (value.exponent() < 0) {
                denominator *= detail::power_of_three(static_cast<std::size_t>(-value.exponent()));
            }
            return Ratio(std::move(numerator), std::move(denominator));
        }

        const core::bigint &numerator() const noexcept {
            return numerator_;
        }
        const core::bigint &denominator() const noexcept {
            return denominator_;
        }
        bool is_zero() const noexcept {
            return numerator_.is_zero();
        }

        friend Ratio operator+(const Ratio &lhs, const Ratio &rhs) {
            core::bigint numerator =
                lhs.numerator_ * rhs.denominator_ + rhs.numerator_ * lhs.denominator_;
            core::bigint denominator = lhs.denominator_ * rhs.denominator_;
            return Ratio(std::move(numerator), std::move(denominator));
        }

        friend Ratio operator-(const Ratio &lhs, const Ratio &rhs) {
            core::bigint numerator =
                lhs.numerator_ * rhs.denominator_ - rhs.numerator_ * lhs.denominator_;
            core::bigint denominator = lhs.denominator_ * rhs.denominator_;
            return Ratio(std::move(numerator), std::move(denominator));
        }

        friend Ratio operator*(const Ratio &lhs, const Ratio &rhs) {
            core::bigint numerator = lhs.numerator_ * rhs.numerator_;
            core::bigint denominator = lhs.denominator_ * rhs.denominator_;
            return Ratio(std::move(numerator), std::move(denominator));
        }

        friend Ratio operator/(const Ratio &lhs, const Ratio &rhs) {
            if (rhs.is_zero()) {
                throw std::domain_error("ratio division by zero");
            }
            core::bigint numerator = lhs.numerator_ * rhs.denominator_;
            core::bigint denominator = lhs.denominator_ * rhs.numerator_;
            return Ratio(std::move(numerator), std::move(denominator));
        }

        friend std::strong_ordering operator<=>(const Ratio &lhs, const Ratio &rhs) noexcept {
            const core::bigint left = lhs.numerator_ * rhs.denominator_;
            const core::bigint right = rhs.numerator_ * lhs.denominator_;
            return left <=> right;
        }

        friend bool operator==(const Ratio &lhs, const Ratio &rhs) noexcept {
            return lhs.numerator_ == rhs.numerator_ && lhs.denominator_ == rhs.denominator_;
        }

        friend bool operator!=(const Ratio &lhs, const Ratio &rhs) noexcept {
            return !(lhs == rhs);
        }

        Ratio sqrt_exact() const {
            if (is_zero()) {
                return Ratio::zero();
            }
            Ratio approx = *this;
            const Ratio two(core::limb::from_value(2));
            for (int iteration = 0; iteration < 12; ++iteration) {
                approx = (approx + *this / approx) / two;
            }
            return approx;
        }

        explicit operator Float() const {
            if (is_zero()) {
                return Float::zero();
            }
            const core::bigint abs_numerator = numerator_.abs();
            const core::bigint limb_max = core::bigint(core::limb::max());
            const core::bigint three = core::bigint(core::limb::from_value(3));
            core::bigint limit = limb_max * denominator_;
            int exponent = 0;

            const auto increase_limit = [&]() {
                while (abs_numerator > limit) {
                    limit *= three;
                    --exponent;
                }
            };

            const auto decrease_limit = [&]() {
                while (exponent < 0) {
                    const auto [next_limit, remainder] = core::bigint::div_mod(limit, three);
                    if (!remainder.is_zero() || abs_numerator > next_limit) {
                        break;
                    }
                    limit = next_limit;
                    ++exponent;
                }
            };

            increase_limit();
            decrease_limit();

            if (exponent >= 0) {
                core::bigint scaled_abs = abs_numerator;
                while (true) {
                    const core::bigint next_scaled = scaled_abs * three;
                    if (next_scaled > limit) {
                        break;
                    }
                    scaled_abs = next_scaled;
                    ++exponent;
                }
            }

            auto scale_ratio = [&](int exp) {
                core::bigint scaled_num = numerator_;
                core::bigint scaled_den = denominator_;
                if (exp > 0) {
                    scaled_num *= detail::power_of_three(static_cast<std::size_t>(exp));
                } else if (exp < 0) {
                    scaled_den *= detail::power_of_three(static_cast<std::size_t>(-exp));
                }
                return std::pair<core::bigint, core::bigint>{std::move(scaled_num),
                                                             std::move(scaled_den)};
            };

            auto round_scaled = [&](core::bigint scaled_num, const core::bigint &scaled_den) {
                const auto abs_scaled = scaled_num.abs();
                const auto [quotient, remainder] = core::bigint::div_mod(abs_scaled, scaled_den);
                core::bigint result = quotient;
                const core::bigint double_remainder = remainder * core::bigint(2);
                if (double_remainder > scaled_den) {
                    result += core::bigint::one();
                }
                if (scaled_num.is_negative()) {
                    result = -result;
                }
                return result;
            };

            while (true) {
                const auto [scaled_numerator, scaled_denominator] = scale_ratio(exponent);
                const core::bigint mantissa_bigint =
                    round_scaled(scaled_numerator, scaled_denominator);
                try {
                    return Float(mantissa_bigint.to_limb(), exponent);
                } catch (const std::overflow_error &) {
                    if (exponent == std::numeric_limits<int>::min()) {
                        throw;
                    }
                    --exponent;
                }
            }
        }

        Float to_float() const {
            return static_cast<Float>(*this);
        }

      private:
        void normalize(core::bigint numerator, core::bigint denominator) {
            if (denominator.is_zero()) {
                throw std::domain_error("ratio denominator cannot be zero");
            }
            if (denominator.is_negative()) {
                numerator = -numerator;
                denominator = -denominator;
            }
            const core::bigint divisor = core::bigint::gcd(numerator.abs(), denominator);
            if (!divisor.is_zero()) {
                numerator /= divisor;
                denominator /= divisor;
            }
            numerator_ = std::move(numerator);
            denominator_ = std::move(denominator);
        }

        core::bigint numerator_{core::bigint::zero()};
        core::bigint denominator_{core::bigint::one()};
    };

    namespace detail {

        template <int N> inline Ratio floatn_to_ratio(const FloatN<N> &value) {
            if (value.is_zero()) {
                return Ratio::zero();
            }
            core::bigint numerator = value.mantissa().to_bigint();
            core::bigint denominator = core::bigint::one();
            const int exponent = value.exponent();
            if (exponent > 0) {
                numerator *= detail::power_of_three(static_cast<std::size_t>(exponent));
            } else if (exponent < 0) {
                denominator *= detail::power_of_three(static_cast<std::size_t>(-exponent));
            }
            return Ratio(std::move(numerator), std::move(denominator));
        }

    } // namespace detail

    inline Float operator+(Float lhs, const Float &rhs) {
        const Ratio result = Ratio::from_float(lhs) + Ratio::from_float(rhs);
        return result.to_float();
    }

    inline Float operator-(Float lhs, const Float &rhs) {
        const Ratio result = Ratio::from_float(lhs) - Ratio::from_float(rhs);
        return result.to_float();
    }

    inline Float operator/(Float lhs, const Float &rhs) {
        const Ratio result = Ratio::from_float(lhs) / Ratio::from_float(rhs);
        return result.to_float();
    }

    Float Float::sqrt() const {
        if (is_zero()) {
            return zero();
        }
        Float approx = *this;
        const Float two(2);
        for (int iteration = 0; iteration < 12; ++iteration) {
            approx = (approx + *this / approx) / two;
        }
        return approx;
    }

    template <int N> FloatN<N> operator+(FloatN<N> lhs, const FloatN<N> &rhs) {
        const Ratio result = detail::floatn_to_ratio(lhs) + detail::floatn_to_ratio(rhs);
        return FloatN<N>(result.to_float());
    }

    template <int N> FloatN<N> operator-(FloatN<N> lhs, const FloatN<N> &rhs) {
        const Ratio result = detail::floatn_to_ratio(lhs) - detail::floatn_to_ratio(rhs);
        return FloatN<N>(result.to_float());
    }

    template <int N> FloatN<N> operator/(FloatN<N> lhs, const FloatN<N> &rhs) {
        const Ratio result = detail::floatn_to_ratio(lhs) / detail::floatn_to_ratio(rhs);
        return FloatN<N>(result.to_float());
    }

    template <int N> FloatN<N> FloatN<N>::sqrt() const {
        if (is_zero()) {
            return zero();
        }
        FloatN<N> approx = *this;
        const FloatN<N> two(2);
        for (int iteration = 0; iteration < 12; ++iteration) {
            approx = (approx + *this / approx) / two;
        }
        return approx;
    }

    class Modulus {
      public:
        explicit Modulus(core::limb modulus)
            : montgomery_context_(modulus), modulus_(montgomery_context_.modulus()),
              ternary_base_(core::limb::from_value(3)) {
        }

        const core::limb &modulus() const noexcept {
            return modulus_;
        }
        const core::MontgomeryContext<core::limb> &montgomery_context() const noexcept {
            return montgomery_context_;
        }
        core::limb to_montgomery(const core::limb &value) const {
            return montgomery_context_.to_montgomery(value);
        }
        core::limb from_montgomery(const core::limb &value) const {
            return montgomery_context_.from_montgomery(value);
        }
        core::limb mul(const core::limb &lhs, const core::limb &rhs) const {
            return montgomery_context_.mul(lhs, rhs);
        }
        core::limb pow3(std::size_t exponent) const {
            auto &cache = power_of_three_cache_for(modulus_);
            while (exponent >= cache.size()) {
                cache.push_back(modular_mul(cache.back(), ternary_base_));
            }
            return cache[exponent];
        }

      private:
        static std::vector<core::limb> &power_of_three_cache_for(const core::limb &modulus) {
            thread_local std::unordered_map<core::limb, std::vector<core::limb>> cache_by_modulus;
            auto it =
                cache_by_modulus.try_emplace(modulus, std::vector<core::limb>{core::limb::one()})
                    .first;
            return it->second;
        }

        core::limb modular_mul(const core::limb &lhs, const core::limb &rhs) const {
            return montgomery_context_.mul(lhs, rhs);
        }

        core::MontgomeryContext<core::limb> montgomery_context_;
        core::limb modulus_;
        core::limb ternary_base_;
    };

    class MontgomeryInt {
      public:
        explicit MontgomeryInt(const Modulus &modulus)
            : modulus_(&modulus), value_(modulus.to_montgomery(core::limb::zero())) {
        }

        MontgomeryInt(const Modulus &modulus, core::limb plain)
            : modulus_(&modulus), value_(modulus.to_montgomery(reduce(plain))) {
        }

        const Modulus &modulus() const noexcept {
            return *modulus_;
        }

        core::limb to_limb() const {
            return modulus_->from_montgomery(value_);
        }

        MontgomeryInt &operator+=(const MontgomeryInt &other) {
            ensure_same_modulus(other);
            const auto left = value_.to_value();
            const auto right = other.value_.to_value();
            const auto mod_value = modulus_->modulus().to_value();
            auto sum = left + right;
            sum %= mod_value;
            if (sum < 0) {
                sum += mod_value;
            }
            value_ = core::limb::from_value(sum);
            return *this;
        }

        MontgomeryInt &operator-=(const MontgomeryInt &other) {
            ensure_same_modulus(other);
            const auto left = value_.to_value();
            const auto right = other.value_.to_value();
            const auto mod_value = modulus_->modulus().to_value();
            auto diff = left - right;
            diff %= mod_value;
            if (diff < 0) {
                diff += mod_value;
            }
            value_ = core::limb::from_value(diff);
            return *this;
        }

        MontgomeryInt &operator*=(const MontgomeryInt &other) {
            ensure_same_modulus(other);
            value_ = modulus_->mul(value_, other.value_);
            return *this;
        }

        friend MontgomeryInt operator+(MontgomeryInt lhs, const MontgomeryInt &rhs) {
            lhs += rhs;
            return lhs;
        }

        friend MontgomeryInt operator-(MontgomeryInt lhs, const MontgomeryInt &rhs) {
            lhs -= rhs;
            return lhs;
        }

        friend MontgomeryInt operator*(MontgomeryInt lhs, const MontgomeryInt &rhs) {
            lhs *= rhs;
            return lhs;
        }

      private:
        core::limb reduce(core::limb value) const {
            while (!(value < modulus_->modulus())) {
                value -= modulus_->modulus();
            }
            return value;
        }

        void ensure_same_modulus(const MontgomeryInt &other) const {
            if (modulus_ != other.modulus_) {
                throw std::invalid_argument("MontgomeryInt modulus mismatch");
            }
        }

        const Modulus *modulus_;
        core::limb value_;
    };

    template <typename Component> class Complex {
      public:
        Complex() noexcept = default;
        Complex(Component real, Component imag) : real_(real), imag_(imag) {
        }

        const Component &real() const noexcept {
            return real_;
        }
        const Component &imag() const noexcept {
            return imag_;
        }

        Complex &operator+=(const Complex &other) {
            real_ += other.real_;
            imag_ += other.imag_;
            return *this;
        }

        Complex &operator-=(const Complex &other) {
            real_ -= other.real_;
            imag_ -= other.imag_;
            return *this;
        }

        Complex &operator*=(const Complex &other) {
            const Component real_part = real_ * other.real_ - imag_ * other.imag_;
            const Component imag_part = real_ * other.imag_ + imag_ * other.real_;
            real_ = real_part;
            imag_ = imag_part;
            return *this;
        }

        friend Complex operator+(Complex lhs, const Complex &rhs) {
            lhs += rhs;
            return lhs;
        }

        friend Complex operator-(Complex lhs, const Complex &rhs) {
            lhs -= rhs;
            return lhs;
        }

        friend Complex operator*(Complex lhs, const Complex &rhs) {
            lhs *= rhs;
            return lhs;
        }

      private:
        Component real_{};
        Component imag_{};
    };

    template <typename T> class Vector {
      public:
        using value_type = T;

        Vector() noexcept = default;
        explicit Vector(std::size_t size, value_type value = value_type{})
            : elements_(size, value) {
        }
        Vector(std::initializer_list<value_type> init) : elements_(init) {
        }
        explicit Vector(std::vector<value_type> elements) : elements_(std::move(elements)) {
        }

        std::size_t size() const noexcept {
            return elements_.size();
        }
        bool empty() const noexcept {
            return elements_.empty();
        }

        value_type &operator[](std::size_t index) {
            return elements_[index];
        }
        const value_type &operator[](std::size_t index) const {
            return elements_[index];
        }

        std::vector<value_type> &data() noexcept {
            return elements_;
        }
        const std::vector<value_type> &data() const noexcept {
            return elements_;
        }

        Vector &operator+=(const Vector &other) {
            ensure_same_size(other);
            for (std::size_t index = 0; index < elements_.size(); ++index) {
                elements_[index] += other.elements_[index];
            }
            return *this;
        }

        Vector &operator-=(const Vector &other) {
            ensure_same_size(other);
            for (std::size_t index = 0; index < elements_.size(); ++index) {
                elements_[index] -= other.elements_[index];
            }
            return *this;
        }

        Vector &operator*=(const value_type &scalar) {
            for (auto &element : elements_) {
                element *= scalar;
            }
            return *this;
        }

        friend Vector operator+(Vector lhs, const Vector &rhs) {
            lhs += rhs;
            return lhs;
        }

        friend Vector operator-(Vector lhs, const Vector &rhs) {
            lhs -= rhs;
            return lhs;
        }

        friend Vector operator*(Vector lhs, const value_type &scalar) {
            lhs *= scalar;
            return lhs;
        }

        friend Vector operator*(const value_type &scalar, Vector rhs) {
            rhs *= scalar;
            return rhs;
        }

        value_type dot(const Vector &other) const {
            ensure_same_size(other);
            value_type result{};
            for (std::size_t index = 0; index < elements_.size(); ++index) {
                result += elements_[index] * other.elements_[index];
            }
            return result;
        }

      private:
        void ensure_same_size(const Vector &other) const {
            if (elements_.size() != other.elements_.size()) {
                throw std::invalid_argument("Vector size mismatch");
            }
        }

        std::vector<value_type> elements_;
    };

    template <typename Element, std::size_t Rows, std::size_t Cols> class Matrix {
      public:
        static_assert(Rows > 0 && Cols > 0, "Matrix requires positive dimensions");

        using value_type = Element;

        Matrix() noexcept = default;
        explicit Matrix(value_type value) {
            storage_.fill(value);
        }
        Matrix(std::initializer_list<value_type> init) {
            if (init.size() != storage_.size()) {
                throw std::invalid_argument("Matrix initializer size mismatch");
            }
            std::copy(init.begin(), init.end(), storage_.begin());
        }

        value_type &operator()(std::size_t row, std::size_t column) {
            return storage_.at(row * Cols + column);
        }

        const value_type &operator()(std::size_t row, std::size_t column) const {
            return storage_.at(row * Cols + column);
        }

        std::size_t rows() const noexcept {
            return Rows;
        }
        std::size_t cols() const noexcept {
            return Cols;
        }

        Matrix &operator+=(const Matrix &other) {
            for (std::size_t index = 0; index < storage_.size(); ++index) {
                storage_[index] += other.storage_[index];
            }
            return *this;
        }

        Matrix &operator-=(const Matrix &other) {
            for (std::size_t index = 0; index < storage_.size(); ++index) {
                storage_[index] -= other.storage_[index];
            }
            return *this;
        }

        Matrix &operator*=(const value_type &scalar) {
            for (auto &cell : storage_) {
                cell *= scalar;
            }
            return *this;
        }

        friend Matrix operator+(Matrix lhs, const Matrix &rhs) {
            lhs += rhs;
            return lhs;
        }

        friend Matrix operator-(Matrix lhs, const Matrix &rhs) {
            lhs -= rhs;
            return lhs;
        }

        friend Matrix operator*(Matrix lhs, const value_type &scalar) {
            lhs *= scalar;
            return lhs;
        }

        friend Matrix operator*(const value_type &scalar, Matrix rhs) {
            rhs *= scalar;
            return rhs;
        }

        template <std::size_t OtherCols>
        Matrix<Element, Rows, OtherCols>
        operator*(const Matrix<Element, Cols, OtherCols> &other) const {
            Matrix<Element, Rows, OtherCols> result;
            for (std::size_t row = 0; row < Rows; ++row) {
                for (std::size_t col = 0; col < OtherCols; ++col) {
                    Element sum{};
                    for (std::size_t inner = 0; inner < Cols; ++inner) {
                        sum += (*this)(row, inner) * other(inner, col);
                    }
                    result(row, col) = sum;
                }
            }
            return result;
        }

        Vector<Element> multiply(const Vector<Element> &vector) const {
            if (vector.size() != Cols) {
                throw std::invalid_argument("Matrix-vector size mismatch");
            }
            Vector<Element> result(Rows);
            for (std::size_t row = 0; row < Rows; ++row) {
                Element sum{};
                for (std::size_t col = 0; col < Cols; ++col) {
                    sum += (*this)(row, col) * vector[col];
                }
                result[row] = sum;
            }
            return result;
        }

      private:
        std::array<value_type, Rows * Cols> storage_{};
    };

    template <typename Coefficient> class Polynomial {
      public:
        Polynomial() noexcept = default;
        explicit Polynomial(std::vector<Coefficient> coeffs) : coeffs_(std::move(coeffs)) {
            normalize();
        }

        static Polynomial zero() noexcept {
            return Polynomial();
        }
        static Polynomial constant(const Coefficient &value) {
            if (value == Coefficient{}) {
                return zero();
            }
            return Polynomial(std::vector<Coefficient>{value});
        }

        std::size_t degree() const noexcept {
            if (coeffs_.empty()) {
                return 0;
            }
            return coeffs_.size() - 1;
        }

        const std::vector<Coefficient> &coefficients() const noexcept {
            return coeffs_;
        }

        const Coefficient &operator[](std::size_t index) const {
            if (index >= coeffs_.size()) {
                static const Coefficient zero{};
                return zero;
            }
            return coeffs_[index];
        }

        Polynomial &operator+=(const Polynomial &other) {
            coeffs_.resize(std::max(coeffs_.size(), other.coeffs_.size()), Coefficient{});
            for (std::size_t index = 0; index < other.coeffs_.size(); ++index) {
                coeffs_[index] += other.coeffs_[index];
            }
            normalize();
            return *this;
        }

        Polynomial &operator-=(const Polynomial &other) {
            coeffs_.resize(std::max(coeffs_.size(), other.coeffs_.size()), Coefficient{});
            for (std::size_t index = 0; index < other.coeffs_.size(); ++index) {
                coeffs_[index] -= other.coeffs_[index];
            }
            normalize();
            return *this;
        }

        Polynomial &operator*=(const Polynomial &other) {
            if (coeffs_.empty() || other.coeffs_.empty()) {
                coeffs_.clear();
                return *this;
            }
            std::vector<Coefficient> product(coeffs_.size() + other.coeffs_.size() - 1,
                                             Coefficient{});
            for (std::size_t lhs = 0; lhs < coeffs_.size(); ++lhs) {
                for (std::size_t rhs = 0; rhs < other.coeffs_.size(); ++rhs) {
                    product[lhs + rhs] += coeffs_[lhs] * other.coeffs_[rhs];
                }
            }
            coeffs_ = std::move(product);
            normalize();
            return *this;
        }

        Coefficient evaluate(const Coefficient &point) const {
            Coefficient result{};
            for (std::size_t index = coeffs_.size(); index-- > 0;) {
                result = result * point + coeffs_[index];
            }
            return result;
        }

        friend Polynomial operator+(Polynomial lhs, const Polynomial &rhs) {
            lhs += rhs;
            return lhs;
        }

        friend Polynomial operator-(Polynomial lhs, const Polynomial &rhs) {
            lhs -= rhs;
            return lhs;
        }

        friend Polynomial operator*(Polynomial lhs, const Polynomial &rhs) {
            lhs *= rhs;
            return lhs;
        }

      private:
        void normalize() {
            while (!coeffs_.empty() && coeffs_.back() == Coefficient{}) {
                coeffs_.pop_back();
            }
        }

        std::vector<Coefficient> coeffs_;
    };

    namespace ntt {

        inline bool is_power_of_three(std::size_t value) noexcept {
            if (value == 0) {
                return false;
            }
            while (value > 1) {
                if (value % 3 != 0) {
                    return false;
                }
                value /= 3;
            }
            return true;
        }

        inline std::size_t next_power_of_three(std::size_t minimum) {
            if (minimum == 0) {
                return 1;
            }
            std::size_t result = 1;
            while (result < minimum) {
                if (result > std::numeric_limits<std::size_t>::max() / 3) {
                    throw std::overflow_error("NTT size overflow");
                }
                result *= 3;
            }
            return result;
        }

        inline MontgomeryInt montgomery_pow(MontgomeryInt base, std::size_t exponent) {
            const Modulus &modulus = base.modulus();
            MontgomeryInt result(modulus, core::limb::one());
            while (exponent > 0) {
                if ((exponent & 1) != 0) {
                    result *= base;
                }
                base *= base;
                exponent >>= 1;
            }
            return result;
        }

        inline core::limb modular_inverse(core::limb value, const core::limb &modulus) {
            if (value.is_zero()) {
                throw std::invalid_argument("modular inverse of zero");
            }
            core::bigint a = core::bigint(modulus);
            core::bigint b = core::bigint(value);
            if (b.is_negative()) {
                b += a;
            }
            core::bigint x0 = core::bigint::zero();
            core::bigint x1 = core::bigint::one();
            while (!b.is_zero()) {
                const auto [quotient, remainder] = core::bigint::div_mod(a, b);
                a = b;
                b = remainder;
                const auto temp = x0 - quotient * x1;
                x0 = x1;
                x1 = temp;
            }
            if (!(a == core::bigint::one())) {
                throw std::invalid_argument("value not invertible modulo modulus");
            }
            if (x0.is_negative()) {
                x0 += core::bigint(modulus);
            }
            return x0.to_limb();
        }

        inline void apply_ternary_ntt(std::vector<MontgomeryInt> &values,
                                      const Modulus &modulus,
                                      const MontgomeryInt &primitive_root) {
            const std::size_t length = values.size();
            if (length <= 1) {
                return;
            }
            if (!is_power_of_three(length)) {
                throw std::invalid_argument("NTT length must be a power of three");
            }
            for (std::size_t stage = 1; stage < length; stage *= 3) {
                const std::size_t block_size = stage * 3;
                const std::size_t stride = length / block_size;
                const MontgomeryInt omega = montgomery_pow(primitive_root, stride);
                for (std::size_t block_start = 0; block_start < length; block_start += block_size) {
                    MontgomeryInt twiddle(modulus, core::limb::one());
                    for (std::size_t offset = 0; offset < stage; ++offset) {
                        const std::size_t index0 = block_start + offset;
                        const std::size_t index1 = index0 + stage;
                        const std::size_t index2 = index1 + stage;
                        const MontgomeryInt value0 = values[index0];
                        const MontgomeryInt value1 = values[index1];
                        const MontgomeryInt value2 = values[index2];
                        const MontgomeryInt twiddle_sq = twiddle * twiddle;
                        values[index0] = value0 + value1 + value2;
                        values[index1] = value0 + twiddle * value1 + twiddle_sq * value2;
                        values[index2] = value0 + twiddle_sq * value1 + twiddle * value2;
                        twiddle *= omega;
                    }
                }
            }
        }

        inline Polynomial<core::limb> multiply_polynomials(const Polynomial<core::limb> &lhs,
                                                           const Polynomial<core::limb> &rhs,
                                                           const Modulus &modulus,
                                                           core::limb primitive_root) {
            const auto &lhs_coeffs = lhs.coefficients();
            const auto &rhs_coeffs = rhs.coefficients();
            if (lhs_coeffs.empty() || rhs_coeffs.empty()) {
                return Polynomial<core::limb>::zero();
            }
            const std::size_t result_size = lhs_coeffs.size() + rhs_coeffs.size() - 1;
            const std::size_t transform_size = next_power_of_three(result_size);
            MontgomeryInt root(modulus, primitive_root);
            std::vector<MontgomeryInt> values_a(transform_size, MontgomeryInt(modulus));
            std::vector<MontgomeryInt> values_b(transform_size, MontgomeryInt(modulus));
            for (std::size_t index = 0; index < lhs_coeffs.size(); ++index) {
                values_a[index] = MontgomeryInt(modulus, lhs_coeffs[index]);
            }
            for (std::size_t index = 0; index < rhs_coeffs.size(); ++index) {
                values_b[index] = MontgomeryInt(modulus, rhs_coeffs[index]);
            }
            apply_ternary_ntt(values_a, modulus, root);
            apply_ternary_ntt(values_b, modulus, root);
            std::vector<MontgomeryInt> pointwise(transform_size, MontgomeryInt(modulus));
            for (std::size_t index = 0; index < transform_size; ++index) {
                pointwise[index] = values_a[index] * values_b[index];
            }
            const MontgomeryInt inverse_root = montgomery_pow(root, transform_size - 1);
            apply_ternary_ntt(pointwise, modulus, inverse_root);
            const core::detail::limb_int128 transform_value =
                static_cast<core::detail::limb_int128>(transform_size);
            const core::limb transform_limb = core::limb::from_value(transform_value);
            const core::limb inv_transform = modular_inverse(transform_limb, modulus.modulus());
            const MontgomeryInt inv_size(modulus, inv_transform);
            for (auto &value : pointwise) {
                value *= inv_size;
            }
            std::vector<core::limb> result_coeffs(result_size);
            for (std::size_t index = 0; index < result_size; ++index) {
                result_coeffs[index] = pointwise[index].to_limb();
            }
            return Polynomial<core::limb>(std::move(result_coeffs));
        }

    } // namespace ntt

    template <int N> class Fixed {
      public:
        static_assert(N > 0, "Fixed requires positive trit width");

        static constexpr int trits = N;

        constexpr Fixed() noexcept = default;
        explicit Fixed(core::bigint value) : trits_(from_bigint(std::move(value))) {
        }
        explicit constexpr Fixed(std::array<std::int8_t, N> trits)
            : trits_(normalize_trits(std::move(trits))) {
        }

        constexpr const std::array<std::int8_t, N> &digits() const noexcept {
            return trits_;
        }

        core::bigint to_bigint() const {
            core::bigint result = core::bigint::zero();
            core::bigint weight = core::bigint::one();
            const core::bigint three = core::bigint(core::limb::from_value(3));
            for (int index = 0; index < N; ++index) {
                if (trits_[index] != 0) {
                    result += weight * core::bigint(trits_[index]);
                }
                weight *= three;
            }
            return result;
        }

        constexpr bool is_zero() const noexcept {
            for (int index = 0; index < N; ++index) {
                if (trits_[index] != 0) {
                    return false;
                }
            }
            return true;
        }

        constexpr bool divisible_by_three() const noexcept {
            static_assert(N > 0);
            return trits_[0] == 0;
        }

        constexpr void divide_by_three() noexcept {
            for (int index = 0; index < N - 1; ++index) {
                trits_[index] = trits_[index + 1];
            }
            trits_[N - 1] = 0;
        }

        constexpr Fixed &operator+=(const Fixed &other) {
            int carry = 0;
            for (int index = 0; index < N; ++index) {
                const int sum =
                    static_cast<int>(trits_[index]) + static_cast<int>(other.trits_[index]) + carry;
                const auto [digit, next_carry] = core::detail::balanced_digit_and_carry(sum);
                trits_[index] = static_cast<std::int8_t>(digit);
                carry = static_cast<int>(next_carry);
            }
            (void)carry;
            return *this;
        }

        constexpr Fixed &operator-=(const Fixed &other) {
            int carry = 0;
            for (int index = 0; index < N; ++index) {
                const int sum = static_cast<int>(trits_[index]) +
                                static_cast<int>(-other.trits_[index]) + carry;
                const auto [digit, next_carry] = core::detail::balanced_digit_and_carry(sum);
                trits_[index] = static_cast<std::int8_t>(digit);
                carry = static_cast<int>(next_carry);
            }
            (void)carry;
            return *this;
        }

        constexpr Fixed &operator*=(const Fixed &other) {
            std::array<int, N> accum{};
            for (int lhs = 0; lhs < N; ++lhs) {
                for (int rhs = 0; rhs < N; ++rhs) {
                    const int index = lhs + rhs;
                    if (index >= N) {
                        continue;
                    }
                    accum[static_cast<std::size_t>(index)] +=
                        static_cast<int>(trits_[lhs]) * static_cast<int>(other.trits_[rhs]);
                }
            }
            int carry = 0;
            for (int index = 0; index < N; ++index) {
                const int sum = accum[static_cast<std::size_t>(index)] + carry;
                const auto [digit, next_carry] = core::detail::balanced_digit_and_carry(sum);
                trits_[index] = static_cast<std::int8_t>(digit);
                carry = static_cast<int>(next_carry);
            }
            (void)carry;
            return *this;
        }

        Fixed &operator/=(const Fixed &other) {
            const core::bigint divisor = other.to_bigint();
            if (divisor.is_zero()) {
                throw std::domain_error("fixed division by zero");
            }
            const core::bigint dividend = to_bigint();
            const auto [quotient, remainder] = core::bigint::div_mod(dividend, divisor);
            (void)remainder;
            trits_ = from_bigint(std::move(quotient));
            return *this;
        }

        Fixed &operator%=(const Fixed &other) {
            const core::bigint divisor = other.to_bigint();
            if (divisor.is_zero()) {
                throw std::domain_error("fixed modulo by zero");
            }
            const core::bigint dividend = to_bigint();
            const auto [quotient, remainder] = core::bigint::div_mod(dividend, divisor);
            (void)quotient;
            trits_ = from_bigint(std::move(remainder));
            return *this;
        }

        friend constexpr Fixed operator+(Fixed lhs, const Fixed &rhs) {
            lhs += rhs;
            return lhs;
        }

        friend constexpr Fixed operator-(Fixed lhs, const Fixed &rhs) {
            lhs -= rhs;
            return lhs;
        }

        friend constexpr Fixed operator*(Fixed lhs, const Fixed &rhs) {
            lhs *= rhs;
            return lhs;
        }

        friend Fixed operator/(Fixed lhs, const Fixed &rhs) {
            lhs /= rhs;
            return lhs;
        }

        friend Fixed operator%(Fixed lhs, const Fixed &rhs) {
            lhs %= rhs;
            return lhs;
        }

        friend constexpr bool operator==(const Fixed &lhs, const Fixed &rhs) noexcept {
            return lhs.trits_ == rhs.trits_;
        }

        friend constexpr bool operator!=(const Fixed &lhs, const Fixed &rhs) noexcept {
            return !(lhs == rhs);
        }

        friend std::strong_ordering operator<=>(const Fixed &lhs, const Fixed &rhs) noexcept {
            return lhs.to_bigint() <=> rhs.to_bigint();
        }

      private:
        static constexpr std::array<std::int8_t, N>
        normalize_trits(std::array<std::int8_t, N> digits) {
            int carry = 0;
            for (int index = 0; index < N; ++index) {
                const int sum = static_cast<int>(digits[index]) + carry;
                const auto [digit, next_carry] = core::detail::balanced_digit_and_carry(sum);
                digits[static_cast<std::size_t>(index)] = static_cast<std::int8_t>(digit);
                carry = static_cast<int>(next_carry);
            }
            (void)carry;
            return digits;
        }

        static std::array<std::int8_t, N> from_bigint(core::bigint value) {
            std::array<std::int8_t, N> digits{};
            const core::bigint three = core::bigint(core::limb::from_value(3));
            for (int index = 0; index < N; ++index) {
                const auto [quotient, remainder] = core::bigint::div_mod(value, three);
                value = quotient;
                int digit = static_cast<int>(remainder.to_limb().to_value());
                int carry = 0;
                if (digit > 1) {
                    digit -= 3;
                    carry = 1;
                } else if (digit < -1) {
                    digit += 3;
                    carry = -1;
                }
                digits[static_cast<std::size_t>(index)] = static_cast<std::int8_t>(digit);
                if (carry != 0) {
                    value += core::bigint(carry);
                }
            }
            return digits;
        }

        std::array<std::int8_t, N> trits_{};
    };

    template <int N> inline std::string to_string(const FloatN<N> &value) {
        if (value.is_zero()) {
            return "0";
        }
        core::bigint mantissa = value.mantissa().to_bigint();
        const bool negative = mantissa.is_negative();
        if (negative) {
            mantissa = -mantissa;
        }
        return detail::format_mantissa(std::move(mantissa), value.exponent(), negative);
    }

    using Int81 = Fixed<48>;

    inline constexpr int T81LIB_VERSION_MAJOR = 0;
    inline constexpr int T81LIB_VERSION_MINOR = 1;
    inline constexpr int T81LIB_VERSION_PATCH = 0;

} // namespace t81

namespace std {

    template <> struct formatter<t81::Float, char> : std::formatter<std::string_view, char> {
        template <typename FormatContext>
        auto format(const t81::Float &value, FormatContext &ctx) const {
            const std::string text = t81::to_string(value);
            return std::formatter<std::string_view, char>::format(text, ctx);
        }
    };

    template <int N>
    struct formatter<t81::FloatN<N>, char> : std::formatter<std::string_view, char> {
        template <typename FormatContext>
        auto format(const t81::FloatN<N> &value, FormatContext &ctx) const {
            const std::string text = t81::to_string(value);
            return std::formatter<std::string_view, char>::format(text, ctx);
        }
    };

} // namespace std
