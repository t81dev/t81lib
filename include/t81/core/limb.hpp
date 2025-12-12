// include/t81/core/limb.hpp — Limb type definition and low-level helpers.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <compare>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <optional>
#include <vector>

#include <t81/core/detail/lut.hpp>
#include <t81/core/detail/base_digits.hpp>

namespace t81::core {

class limb;

namespace detail {
#if !defined(__SIZEOF_INT128__)
#error "t81::core::limb requires __int128 support"
#endif

using limb_int128 = __int128_t;
using limb_uint128 = unsigned __int128;

inline constexpr int TRITS = 48;
inline constexpr int TRYTES = 16;
inline constexpr int BYTES = TRYTES;
inline constexpr std::uint8_t ZERO_TRYTE = 13;

inline constexpr std::array<limb_int128, TRITS> build_pow3() {
    std::array<limb_int128, TRITS> powers{};
    limb_int128 value{1};
    for (int index = 0; index < TRITS; ++index) {
        powers[index] = value;
        value *= 3;
    }
    return powers;
}

inline constexpr auto POW3 = build_pow3();
inline constexpr limb_int128 RADIX = POW3[TRITS - 1] * 3;
inline constexpr limb_int128 MAX_VALUE = (RADIX - 1) / 2;
inline constexpr limb_int128 MIN_VALUE = -MAX_VALUE;

inline constexpr int decimal_digit_count(limb_int128 value) noexcept {
    if (value == 0) {
        return 1;
    }
    if (value < 0) {
        value = -value;
    }
    int digits = 0;
    while (value != 0) {
        value /= 10;
        ++digits;
    }
    return digits;
}

inline constexpr bool valid_trit(std::int8_t trit) noexcept {
    return trit >= -1 && trit <= 1;
}

inline constexpr std::size_t pack_trit_index(std::int8_t t0,
                                              std::int8_t t1,
                                              std::int8_t t2) noexcept {
    return static_cast<std::size_t>((t0 + 1) + 3 * (t1 + 1) + 9 * (t2 + 1));
}

template <class Int>
inline constexpr std::pair<int, Int> balanced_digit_and_carry(Int value) noexcept {
    const Int carry = value >= 0 ? (value + 1) / 3 : (value - 1) / 3;
    const int digit = static_cast<int>(value - carry * 3);
    return {digit, carry};
}

} // namespace detail

namespace detail {
    bool add_trytes_simd(const limb&, const limb&, limb&);
    std::optional<std::pair<limb, limb>> mul_wide_simd(const limb&, const limb&);
    std::pair<limb, limb> mul_wide_scalar(const limb&, const limb&);
}

class limb {
public:
    static constexpr int TRITS = detail::TRITS;
    static constexpr int TRYTES = detail::TRYTES;
    static constexpr int BYTES = detail::BYTES;
    using tryte_t = std::uint8_t;
    static constexpr int BASE81_DIGITS_PER_LIMB = TRITS / 4;

    constexpr limb() noexcept { trytes_.fill(detail::ZERO_TRYTE); }

    constexpr limb(const limb&) noexcept = default;
    constexpr limb& operator=(const limb&) noexcept = default;

    template <typename Int,
              typename = std::enable_if_t<std::is_integral_v<Int>>>
    constexpr explicit limb(Int value) : limb(from_value(static_cast<detail::limb_int128>(value))) {}

    static constexpr limb zero() noexcept { return limb(); }
    static constexpr limb one() { return from_value(1); }
    static constexpr limb min() { return from_value(detail::MIN_VALUE); }
    static constexpr limb max() { return from_value(detail::MAX_VALUE); }

    static constexpr limb from_value(detail::limb_int128 value) {
        if (value < detail::MIN_VALUE || value > detail::MAX_VALUE) {
            throw std::overflow_error("limb value out of representable range");
        }
        std::array<std::int8_t, TRITS> trits{};
        detail::limb_int128 cursor = value;
        for (int index = 0; index < TRITS; ++index) {
            const auto [digit, carry] = detail::balanced_digit_and_carry(cursor);
            trits[index] = static_cast<std::int8_t>(digit);
            cursor = carry;
        }
        if (cursor != 0) {
            throw std::overflow_error("limb value normalization overflow");
        }
        return from_trits(trits);
    }

    static constexpr limb from_trits(const std::array<std::int8_t, TRITS>& trits) {
        limb result;
        for (std::size_t index = 0; index < TRYTES; ++index) {
            const std::int8_t t0 = trits[3 * index];
            const std::int8_t t1 = trits[3 * index + 1];
            const std::int8_t t2 = trits[3 * index + 2];
            if (!detail::valid_trit(t0) || !detail::valid_trit(t1) || !detail::valid_trit(t2)) {
                throw std::invalid_argument("limb trits must be -1..1");
            }
            const std::size_t packed = detail::pack_trit_index(t0, t1, t2);
            result.trytes_[index] = detail::TRITS_TO_TRYTE[packed];
        }
        return result;
    }

    static constexpr limb from_bytes(std::array<std::uint8_t, BYTES> bytes) {
        limb result;
        for (std::size_t index = 0; index < BYTES; ++index) {
            if (bytes[index] > 26) {
                throw std::invalid_argument("limb bytes must encode canonical trytes");
            }
            result.trytes_[index] = static_cast<tryte_t>(bytes[index]);
        }
        return result;
    }

    constexpr std::array<std::uint8_t, BYTES> to_bytes() const noexcept {
        std::array<std::uint8_t, BYTES> bytes{};
        for (std::size_t index = 0; index < BYTES; ++index) {
            bytes[index] = trytes_[index];
        }
        return bytes;
    }

    constexpr std::array<tryte_t, TRYTES> to_trytes() const noexcept { return trytes_; }

    constexpr tryte_t get_tryte(std::size_t index) const {
        if (index >= TRYTES) {
            throw std::out_of_range("limb tryte index out of range");
        }
        return trytes_[index];
    }

    constexpr limb& set_tryte(std::size_t index, tryte_t value) {
        if (index >= TRYTES) {
            throw std::out_of_range("limb tryte index out of range");
        }
        if (value > 26) {
            throw std::invalid_argument("limb tryte value must be 0..26");
        }
        trytes_[index] = value;
        return *this;
    }

    constexpr std::int8_t get_trit(std::size_t index) const {
        if (index >= TRITS) {
            throw std::out_of_range("limb trit index out of range");
        }
        const std::size_t tryte_index = index / 3;
        const std::size_t offset = index % 3;
        const auto& triple = detail::TRYTE_TO_TRITS[trytes_[tryte_index]];
        return triple[offset];
    }

    constexpr limb& set_trit(std::size_t index, std::int8_t value) {
        if (index >= TRITS) {
            throw std::out_of_range("limb trit index out of range");
        }
        if (!detail::valid_trit(value)) {
            throw std::invalid_argument("limb trit value must be -1..1");
        }
        const std::size_t tryte_index = index / 3;
        auto triple = detail::TRYTE_TO_TRITS[trytes_[tryte_index]];
        triple[index % 3] = value;
        const std::size_t packed = detail::pack_trit_index(triple[0], triple[1], triple[2]);
        trytes_[tryte_index] = detail::TRITS_TO_TRYTE[packed];
        return *this;
    }

    constexpr std::array<std::int8_t, TRITS> to_trits() const noexcept {
        std::array<std::int8_t, TRITS> trits{};
        for (std::size_t index = 0; index < TRYTES; ++index) {
            const auto& triple = detail::TRYTE_TO_TRITS[trytes_[index]];
            trits[3 * index] = triple[0];
            trits[3 * index + 1] = triple[1];
            trits[3 * index + 2] = triple[2];
        }
        return trits;
    }

    constexpr detail::limb_int128 to_value() const noexcept {
        const auto trits = to_trits();
        detail::limb_int128 total = 0;
        for (int index = 0; index < TRITS; ++index) {
            total += static_cast<detail::limb_int128>(trits[index]) * detail::POW3[index];
        }
        return total;
    }

    constexpr bool is_zero() const noexcept { return to_value() == 0; }
    constexpr bool is_negative() const noexcept { return to_value() < 0; }
    constexpr int signum() const noexcept { return (to_value() > 0) - (to_value() < 0); }

    template <typename Int,
              typename = std::enable_if_t<std::is_integral_v<Int>>>
    constexpr Int to_integer() const {
        const auto value = to_value();
        if (value < static_cast<detail::limb_int128>(std::numeric_limits<Int>::min()) ||
            value > static_cast<detail::limb_int128>(std::numeric_limits<Int>::max())) {
            throw std::overflow_error("limb value does not fit in target integer");
        }
        return static_cast<Int>(value);
    }

    constexpr float to_float() const noexcept {
        return static_cast<float>(to_value());
    }

    constexpr double to_double() const noexcept {
        return static_cast<double>(to_value());
    }

    constexpr long double to_long_double() const noexcept {
        return static_cast<long double>(to_value());
    }

    explicit constexpr operator float() const noexcept { return to_float(); }
    explicit constexpr operator double() const noexcept { return to_double(); }
    explicit constexpr operator long double() const noexcept { return to_long_double(); }

    constexpr std::string to_string(int base = 10) const {
        const bool negative = is_negative();
        if (base == 81) {
            const limb magnitude = negative ? -(*this) : *this;
            if (magnitude.is_zero()) {
                return "0";
            }
            const auto trits = magnitude.to_trits();
            std::vector<int> digits;
            digits.reserve(BASE81_DIGITS_PER_LIMB + 4);
            int carry = 0;
            for (int chunk = 0; chunk < BASE81_DIGITS_PER_LIMB; ++chunk) {
                int sum = carry;
                int weight = 1;
                for (int offset = 0; offset < 4; ++offset) {
                    sum += static_cast<int>(trits[chunk * 4 + offset]) * weight;
                    weight *= 3;
                }
                int digit = sum % 81;
                if (digit < 0) {
                    digit += 81;
                }
                carry = (sum - digit) / 81;
                digits.push_back(digit);
            }
            while (carry != 0) {
                int digit = carry % 81;
                if (digit < 0) {
                    digit += 81;
                }
                carry = (carry - digit) / 81;
                digits.push_back(digit);
            }
            while (digits.size() > 1 && digits.back() == 0) {
                digits.pop_back();
            }
            std::string result;
            result.reserve(digits.size() + (negative ? 1 : 0));
            if (negative) {
                result.push_back('-');
            }
            for (auto it = digits.rbegin(); it != digits.rend(); ++it) {
                result.push_back(detail::base81_digit_char(*it));
            }
            return result;
        }
        if (base < 2 || base > 36) {
            throw std::invalid_argument("supported bases are 2..36");
        }
        const auto value = to_value();
        if (value == 0) {
            return "0";
        }
        detail::limb_int128 cursor = value;
        std::string digits;
        const bool legacy_negative = cursor < 0;
        if (legacy_negative) {
            cursor = -cursor;
        }
        while (cursor != 0) {
            const int remainder = static_cast<int>(cursor % base);
            if (remainder < 10) {
                digits.push_back(static_cast<char>('0' + remainder));
            } else {
                digits.push_back(static_cast<char>('a' + remainder - 10));
            }
            cursor /= base;
        }
        if (legacy_negative) {
            digits.push_back('-');
        }
        std::reverse(digits.begin(), digits.end());
        return digits;
    }

    static limb from_base81_digits(std::string_view digits) {
        if (digits.empty()) {
            return zero();
        }
        return from_string_base81(digits);
    }

    static limb from_string(std::string_view text, int base = 10) {
        if (base == 81) {
            return from_string_base81(text);
        }
        if (base < 2 || base > 36) {
            throw std::invalid_argument("supported bases are 2..36");
        }
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
        detail::limb_int128 accumulator = 0;
        for (; index < text.size(); ++index) {
            const char ch = text[index];
            int digit;
            if (ch >= '0' && ch <= '9') {
                digit = ch - '0';
            } else if (ch >= 'a' && ch <= 'z') {
                digit = 10 + ch - 'a';
            } else if (ch >= 'A' && ch <= 'Z') {
                digit = 10 + ch - 'A';
            } else {
                digit = -1;
            }
            if (digit < 0 || digit >= base) {
                throw std::invalid_argument("invalid digit in string");
            }
            accumulator = accumulator * base + digit;
        }
        if (negative) {
            accumulator = -accumulator;
        }
        return from_value(accumulator);
    }
    template <typename Float,
              typename = std::enable_if_t<std::is_floating_point_v<Float>>>
    static limb from_floating(Float value) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("limb floating-point value must be finite");
        }
        const long double truncated = std::trunc(static_cast<long double>(value));
        if (truncated < static_cast<long double>(detail::MIN_VALUE) ||
            truncated > static_cast<long double>(detail::MAX_VALUE)) {
            throw std::overflow_error("limb floating-point value out of range");
        }
        return from_value(static_cast<detail::limb_int128>(truncated));
    }

    static limb from_float(float value) { return from_floating(value); }
    static limb from_double(double value) { return from_floating(value); }
    static limb from_long_double(long double value) { return from_floating(value); }

    constexpr auto operator<=>(const limb& other) const noexcept {
        return to_value() <=> other.to_value();
    }

    constexpr bool operator==(const limb& other) const noexcept {
        return to_value() == other.to_value();
    }

    limb operator-() const { return from_value(-to_value()); }

    limb operator+(const limb& other) const {
        limb result;
        if (detail::add_trytes_simd(*this, other, result)) {
            return result;
        }
        return from_value(to_value() + other.to_value());
    }
    limb operator-(const limb& other) const { return from_value(to_value() - other.to_value()); }

    limb operator*(const limb& other) const {
        const auto [low, high] = mul_wide(*this, other);
        if (!high.is_zero()) {
            throw std::overflow_error("limb multiplication overflow");
        }
        return low;
    }

    static std::pair<limb, limb> mul_wide(const limb& lhs, const limb& rhs) {
        if (const auto simd_result = detail::mul_wide_simd(lhs, rhs)) {
            return *simd_result;
        }
        return detail::mul_wide_scalar(lhs, rhs);
    }

    static std::pair<limb, limb> div_mod(const limb& dividend, const limb& divisor) {
        if (divisor.is_zero()) {
            throw std::domain_error("division by zero");
        }
        const auto num = dividend.to_value();
        const auto den = divisor.to_value();
        const auto quotient = num / den;
        const auto remainder = num % den;
        return {from_value(quotient), from_value(remainder)};
    }

    limb operator/(const limb& other) const {
        return div_mod(*this, other).first;
    }

    limb operator%(const limb& other) const {
        return div_mod(*this, other).second;
    }

    limb& operator+=(const limb& other) { return *this = *this + other; }
    limb& operator-=(const limb& other) { return *this = *this - other; }
    limb& operator*=(const limb& other) { return *this = *this * other; }
    limb& operator/=(const limb& other) { return *this = *this / other; }
    limb& operator%=(const limb& other) { return *this = *this % other; }

    static limb pow_mod(const limb& base, const limb& exponent, const limb& modulus) {
        if (modulus.is_zero()) {
            throw std::domain_error("modulus must be non-zero");
        }
        const auto mod_value = modulus.to_value();
        if (mod_value <= 0) {
            throw std::domain_error("modulus must be positive");
        }
        auto exp_value = exponent.to_value();
        if (exp_value < 0) {
            throw std::domain_error("negative exponent");
        }
        auto result = detail::limb_int128(1);
        auto base_value = base.to_value() % mod_value;
        if (base_value < 0) {
            base_value += mod_value;
        }
        while (exp_value > 0) {
            if ((exp_value & 1) != 0) {
                result = (result * base_value) % mod_value;
            }
            base_value = (base_value * base_value) % mod_value;
            exp_value >>= 1;
        }
        return from_value(result % mod_value);
    }

    limb consensus(const limb& other) const {
        return apply_tritwise(other, [](std::int8_t lhs, std::int8_t rhs) {
            return lhs == rhs ? lhs : 0;
        });
    }

    limb operator&(const limb& other) const {
        return apply_tritwise(other, [](std::int8_t lhs, std::int8_t rhs) {
            return static_cast<std::int8_t>(std::min(lhs, rhs));
        });
    }

    limb operator|(const limb& other) const {
        return apply_tritwise(other, [](std::int8_t lhs, std::int8_t rhs) {
            return static_cast<std::int8_t>(std::max(lhs, rhs));
        });
    }

    limb operator^(const limb& other) const {
        return apply_tritwise(other, [](std::int8_t lhs, std::int8_t rhs) {
            int sum = lhs + rhs;
            if (sum > 1) {
                sum -= 3;
            } else if (sum < -1) {
                sum += 3;
            }
            return static_cast<std::int8_t>(sum);
        });
    }

    limb operator~() const {
        return apply_tritwise(*this, [](std::int8_t lhs, std::int8_t) {
            return static_cast<std::int8_t>(-lhs);
        });
    }

    limb operator<<(int shift) const {
        return tryte_shift_left(shift);
    }

    limb operator>>(int shift) const {
        return tryte_shift_right(shift);
    }

    limb rotate_left_tbits(int count) const {
        return trit_shift_left(count);
    }

    limb rotate_right_tbits(int count) const {
        return trit_shift_right(count);
    }

    limb trit_shift_left(int count) const {
        if (count <= 0) {
            return *this;
        }
        if (count >= TRITS) {
            return zero();
        }
        const auto trits = to_trits();
        std::array<std::int8_t, TRITS> shifted{};
        for (int index = 0; index < TRITS - count; ++index) {
            shifted[index + count] = trits[index];
        }
        return from_trits(shifted);
    }

    limb trit_shift_right(int count) const {
        if (count <= 0) {
            return *this;
        }
        if (count >= TRITS) {
            return zero();
        }
        const auto trits = to_trits();
        std::array<std::int8_t, TRITS> shifted{};
        for (int index = count; index < TRITS; ++index) {
            shifted[index - count] = trits[index];
        }
        return from_trits(shifted);
    }

    limb tryte_shift_left(int count) const {
        if (count <= 0) {
            return *this;
        }
        if (count >= TRYTES) {
            return zero();
        }
        limb result;
        for (int index = TRYTES - 1; index >= count; --index) {
            result.trytes_[index] = trytes_[index - count];
        }
        for (int index = 0; index < count; ++index) {
            result.trytes_[index] = detail::ZERO_TRYTE;
        }
        return result;
    }

    limb tryte_shift_right(int count) const {
        if (count <= 0) {
            return *this;
        }
        if (count >= TRYTES) {
            return zero();
        }
        limb result;
        for (int index = 0; index < TRYTES - count; ++index) {
            result.trytes_[index] = trytes_[index + count];
        }
        for (int index = TRYTES - count; index < TRYTES; ++index) {
            result.trytes_[index] = detail::ZERO_TRYTE;
        }
        return result;
    }

    static limb from_string_base81(std::string_view text) {
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
        std::array<std::int8_t, TRITS> trits{};
        for (; index < text.size(); ++index) {
            const char ch = text[index];
            const int digit = detail::base81_digit_value(ch, 81);
            if (digit < 0) {
                throw std::invalid_argument("invalid digit in string");
            }
            for (int idx = TRITS - 1; idx >= 4; --idx) {
                trits[idx] = trits[idx - 4];
            }
            for (int idx = 0; idx < 4; ++idx) {
                trits[idx] = 0;
            }
            const auto& chunk_trits = detail::base81_trits_for_digit(digit);
            for (int idx = 0; idx < 4; ++idx) {
                trits[idx] += chunk_trits[idx];
            }
            int carry = 0;
            for (int idx = 0; idx < TRITS; ++idx) {
                const auto [digit_val, next_carry] =
                    detail::balanced_digit_and_carry(static_cast<int>(trits[idx]) + carry);
                trits[idx] = static_cast<std::int8_t>(digit_val);
                carry = static_cast<int>(next_carry);
            }
            if (carry != 0) {
                throw std::overflow_error("base81 string overflow");
            }
        }
        limb result = from_trits(trits);
        if (negative) {
            result = -result;
        }
        return result;
    }

private:
    template <typename Fn>
    limb apply_tritwise(const limb& other, Fn&& fn) const {
        const auto lhs_trits = to_trits();
        const auto rhs_trits = other.to_trits();
        std::array<std::int8_t, TRITS> result_trits{};
        for (int index = 0; index < TRITS; ++index) {
            result_trits[index] = fn(lhs_trits[index], rhs_trits[index]);
        }
        return from_trits(result_trits);
    }

    std::array<tryte_t, TRYTES> trytes_{};
};

inline std::size_t canonical_hash(const limb& value) noexcept {
    constexpr std::uint64_t FNV_OFFSET = 1469598103934665603ULL;
    constexpr std::uint64_t FNV_PRIME = 1099511628211ULL;
    std::uint64_t hash = FNV_OFFSET;
    for (auto byte : value.to_bytes()) {
        hash ^= byte;
        hash *= FNV_PRIME;
    }
    if constexpr (sizeof(std::size_t) >= 8) {
        return static_cast<std::size_t>(hash);
    }
    return static_cast<std::size_t>((hash >> 32) ^ (hash & 0xFFFFFFFFULL));
}

} // namespace t81::core

namespace std {

template <>
class numeric_limits<t81::core::limb> {
public:
    using value_type = t81::core::limb;

    static constexpr bool is_specialized = true;

    static constexpr value_type min() noexcept { return value_type::min(); }
    static constexpr value_type max() noexcept { return value_type::max(); }
    static constexpr value_type lowest() noexcept { return value_type::min(); }
    static constexpr value_type epsilon() noexcept { return value_type::zero(); }
    static constexpr value_type round_error() noexcept { return value_type::zero(); }
    static constexpr value_type denorm_min() noexcept { return value_type::zero(); }
    static constexpr value_type infinity() noexcept { return value_type::zero(); }
    static constexpr value_type quiet_NaN() noexcept { return value_type::zero(); }
    static constexpr value_type signaling_NaN() noexcept { return value_type::zero(); }

    static constexpr int digits = value_type::TRITS;
    static constexpr int digits10 =
        t81::core::detail::decimal_digit_count(t81::core::detail::MAX_VALUE);
    static constexpr int max_digits10 = digits10;
    static constexpr int radix = 3;
    static constexpr int min_exponent = 0;
    static constexpr int max_exponent = 0;
    static constexpr int min_exponent10 = 0;
    static constexpr int max_exponent10 = 0;

    static constexpr bool is_signed = true;
    static constexpr bool is_integer = true;
    static constexpr bool is_exact = true;
    static constexpr bool has_infinity = false;
    static constexpr bool has_quiet_NaN = false;
    static constexpr bool has_signaling_NaN = false;
    static constexpr bool has_denorm = false;
    static constexpr bool has_denorm_loss = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr bool traps = true;
    static constexpr bool tinyness_before = false;
    static constexpr bool tinyness_after = false;
    static constexpr float_round_style round_style = round_toward_zero;
    static constexpr bool is_iec559 = false;
};

template <>
struct hash<t81::core::limb> {
    std::size_t operator()(const t81::core::limb& value) const noexcept {
        return t81::core::canonical_hash(value);
    }
};

} // namespace std

#include <t81/core/detail/simd.hpp>
