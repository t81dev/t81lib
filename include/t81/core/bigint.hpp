// include/t81/core/bigint.hpp — Public bigint class declaration and API.

#pragma once

#include <algorithm>
#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include <limits>

#include <t81/core/limb.hpp>

namespace t81::core {

class bigint;

std::vector<limb> signed_limbs(const bigint&);
std::vector<std::int8_t> signed_trits(const bigint&);
bigint from_signed_trits(std::vector<std::int8_t>);
bigint from_signed_limbs(std::vector<limb> digits);
template <typename Fn>
bigint expected_bitwise(const bigint&, const bigint&, Fn);

class bigint {
public:
    bigint() noexcept = default;
    bigint(const bigint&) = default;
    bigint(bigint&&) noexcept = default;
    bigint& operator=(const bigint&) = default;
    bigint& operator=(bigint&&) noexcept = default;

    explicit bigint(limb value) {
        if (!value.is_zero()) {
            negative_ = value.is_negative();
            limb magnitude = value;
            if (negative_) {
                magnitude = -magnitude;
            }
            limbs_.push_back(magnitude);
        }
    }

    template <typename Int, typename = std::enable_if_t<std::is_integral_v<Int>>>
    explicit bigint(Int value) {
        if (value == 0) {
            return;
        }
        negative_ = (value < 0);
        detail::limb_int128 cursor = static_cast<detail::limb_int128>(value);
        if (negative_) {
            cursor = -cursor;
        }
        while (cursor != 0) {
            detail::limb_int128 remainder = cursor % detail::RADIX;
            detail::limb_int128 carry = (cursor - remainder) / detail::RADIX;
            if (remainder > detail::MAX_VALUE) {
                remainder -= detail::RADIX;
                ++carry;
            } else if (remainder < detail::MIN_VALUE) {
                remainder += detail::RADIX;
                --carry;
            }
            limbs_.push_back(limb::from_value(remainder));
            cursor = carry;
        }
    }

    static bigint zero() noexcept { return {}; }
    static bigint one() { return bigint(limb::one()); }

    static bigint from_limbs(std::vector<limb> limbs, bool negative) {
        bigint result;
        result.limbs_ = std::move(limbs);
        result.normalize();
        result.negative_ = negative && !result.is_zero();
        return result;
    }

    std::vector<int> base81_digits() const {
        if (is_zero()) {
            return {0};
        }
        std::vector<int> digits;
        digits.reserve(limbs_.size() * limb::BASE81_DIGITS_PER_LIMB + 4);
        int carry = 0;
        for (const auto& limb_value : limbs_) {
            const auto trits = limb_value.to_trits();
            int block_carry = carry;
            for (int chunk = 0; chunk < limb::BASE81_DIGITS_PER_LIMB; ++chunk) {
                int sum = block_carry;
                int weight = 1;
                for (int offset = 0; offset < 4; ++offset) {
                    sum += static_cast<int>(trits[chunk * 4 + offset]) * weight;
                    weight *= 3;
                }
                int digit = sum % 81;
                if (digit < 0) {
                    digit += 81;
                }
                block_carry = (sum - digit) / 81;
                digits.push_back(digit);
            }
            carry = block_carry;
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
        return digits;
    }

    bool is_zero() const noexcept { return limbs_.empty(); }
    bool is_negative() const noexcept { return negative_; }
    int signum() const noexcept {
        if (is_zero()) {
            return 0;
        }
        return negative_ ? -1 : 1;
    }
    std::size_t limb_count() const noexcept { return limbs_.size(); }

    const limb& limb_at(std::size_t index) const {
        if (index >= limbs_.size()) {
            throw std::out_of_range("bigint limb index out of range");
        }
        return limbs_[index];
    }

    limb to_limb() const {
        if (is_zero()) {
            return limb::zero();
        }
        if (limbs_.size() > 1) {
            throw std::overflow_error("bigint does not fit in a single limb");
        }
        limb value = limbs_[0];
        if (negative_) {
            value = -value;
        }
        return value;
    }

    template <typename Int,
              typename = std::enable_if_t<std::is_integral_v<Int>>>
    explicit operator Int() const {
        if (is_zero()) {
            return static_cast<Int>(0);
        }
        if constexpr (std::is_unsigned_v<Int>) {
            if (negative_) {
                throw std::overflow_error("bigint does not fit in target type");
            }
        }
        using wide = detail::limb_int128;
        const wide max_value =
            static_cast<wide>(std::numeric_limits<Int>::max());
        const wide min_value =
            static_cast<wide>(std::numeric_limits<Int>::min());
        const int limit_limbs = max_limbs_for(max_value);
        if (limb_count() > static_cast<std::size_t>(limit_limbs)) {
            throw std::overflow_error("bigint does not fit in target type");
        }
        wide value = 0;
        for (std::size_t index = limb_count(); index-- > 0;) {
            value = value * detail::RADIX + limbs_[index].to_value();
        }
        if (negative_) {
            value = -value;
        }
        if (value < min_value || value > max_value) {
            throw std::overflow_error("bigint does not fit in target type");
        }
        return static_cast<Int>(value);
    }

    bigint abs() const noexcept {
        bigint copy = *this;
        copy.negative_ = false;
        return copy;
    }

    bigint shift_limbs(std::size_t count) const {
        if (is_zero() || count == 0) {
            return *this;
        }
        bigint result;
        result.negative_ = negative_;
        result.limbs_.assign(count, limb::zero());
        result.limbs_.insert(result.limbs_.end(), limbs_.begin(), limbs_.end());
        result.normalize();
        return result;
    }

    static bigint multiply_by_power_of_three(const bigint& value, int exponent) {
        if (exponent <= 0 || value.is_zero()) {
            return value;
        }
        if (exponent >= limb::TRITS) {
            const int limb_shift = exponent / limb::TRITS;
            const int remainder = exponent % limb::TRITS;
            bigint shifted = value.shift_limbs(static_cast<std::size_t>(limb_shift));
            if (remainder > 0) {
                return multiply_by_power_of_three(shifted, remainder);
            }
            return shifted;
        }
        const detail::limb_int128 multiplier_value = detail::POW3[exponent];
        std::vector<limb> digits = multiply_magnitude_by_small(value.limbs_, multiplier_value);
        bigint result;
        result.negative_ = value.negative_;
        result.limbs_ = std::move(digits);
        result.normalize();
        return result;
    }

    friend std::strong_ordering operator<=>(const bigint& lhs, const bigint& rhs) noexcept {
        return lhs.compare(rhs);
    }

    friend bool operator==(const bigint& lhs, const bigint& rhs) noexcept {
        return lhs.compare(rhs) == std::strong_ordering::equal;
    }

    bigint& operator+=(const bigint& other) {
        if (other.is_zero()) {
            return *this;
        }
        if (is_zero()) {
            *this = other;
            return *this;
        }
        if (negative_ == other.negative_) {
            limbs_ = add_magnitude(limbs_, other.limbs_);
        } else {
            const auto magnitude_cmp = compare_magnitude(other);
            if (magnitude_cmp == std::strong_ordering::equal) {
                limbs_.clear();
                negative_ = false;
                return *this;
            }
            if (magnitude_cmp == std::strong_ordering::greater) {
                limbs_ = subtract_magnitude(limbs_, other.limbs_);
            } else {
                limbs_ = subtract_magnitude(other.limbs_, limbs_);
                negative_ = other.negative_;
            }
        }
        normalize();
        return *this;
    }

    bigint& operator-=(const bigint& other) {
        if (other.is_zero()) {
            return *this;
        }
        if (is_zero()) {
            *this = other;
            negative_ = !other.negative_;
            normalize();
            return *this;
        }
        if (negative_ != other.negative_) {
            limbs_ = add_magnitude(limbs_, other.limbs_);
        } else {
            const auto magnitude_cmp = compare_magnitude(other);
            if (magnitude_cmp == std::strong_ordering::equal) {
                limbs_.clear();
                negative_ = false;
                return *this;
            }
            if (magnitude_cmp == std::strong_ordering::greater) {
                limbs_ = subtract_magnitude(limbs_, other.limbs_);
            } else {
                limbs_ = subtract_magnitude(other.limbs_, limbs_);
                negative_ = !negative_;
            }
        }
        normalize();
        return *this;
    }

    bigint& operator*=(const bigint& other) {
        if (is_zero() || other.is_zero()) {
            limbs_.clear();
            negative_ = false;
            return *this;
        }
        limbs_ = multiply_magnitude(limbs_, other.limbs_);
        negative_ = negative_ != other.negative_;
        normalize();
        return *this;
    }

    bigint operator-() const {
        if (is_zero()) {
            return *this;
        }
        bigint result = *this;
        result.negative_ = !result.negative_;
        return result;
    }

    bigint& operator/=(const bigint& other) {
        const auto [quotient, remainder] = div_mod(*this, other);
        *this = quotient;
        (void)remainder;
        return *this;
    }

    bigint& operator%=(const bigint& other) {
        const auto [quotient, remainder] = div_mod(*this, other);
        *this = remainder;
        return *this;
    }

    friend bigint operator+(bigint lhs, const bigint& rhs) {
        lhs += rhs;
        return lhs;
    }
    friend bigint operator-(bigint lhs, const bigint& rhs) {
        lhs -= rhs;
        return lhs;
    }
    friend bigint operator*(bigint lhs, const bigint& rhs) {
        lhs *= rhs;
        return lhs;
    }
    friend bigint operator/(bigint lhs, const bigint& rhs) {
        lhs /= rhs;
        return lhs;
    }
    friend bigint operator%(bigint lhs, const bigint& rhs) {
        lhs %= rhs;
        return lhs;
    }

    static std::pair<bigint, bigint> div_mod(bigint dividend, const bigint& divisor) {
        if (divisor.is_zero()) {
            throw std::domain_error("division by zero");
        }
        if (dividend.is_zero()) {
            return {bigint::zero(), bigint::zero()};
        }
        const bigint abs_divisor = divisor.abs();
        bigint remainder = dividend.abs();
        bigint quotient;
        if (remainder >= abs_divisor) {
            const auto [quotient_digits, remainder_digits] =
                divide_magnitude(remainder.limbs_, abs_divisor.limbs_);
            quotient.limbs_ = std::move(quotient_digits);
            remainder.limbs_ = std::move(remainder_digits);
        }
        const bool quotient_negative = (dividend.is_negative() != divisor.is_negative());
        const bool remainder_negative = dividend.is_negative();
        quotient.negative_ = quotient.is_zero() ? false : quotient_negative;
        remainder.negative_ = remainder.is_zero() ? false : remainder_negative;
        return {quotient, remainder};
    }

    bigint consensus(const bigint& other) const {
        return apply_limbwise(*this, other, [](const limb& lhs, const limb& rhs) {
            return lhs.consensus(rhs);
        });
    }

    bigint operator&(const bigint& other) const {
        return apply_limbwise(*this, other, [](const limb& lhs, const limb& rhs) {
            return lhs & rhs;
        });
    }

    bigint operator|(const bigint& other) const {
        return apply_limbwise(*this, other, [](const limb& lhs, const limb& rhs) {
            return lhs | rhs;
        });
    }

    bigint operator^(const bigint& other) const {
        return apply_limbwise(*this, other, [](const limb& lhs, const limb& rhs) {
            return lhs ^ rhs;
        });
    }

    bigint bitwise_andnot(const bigint& other) const {
        return apply_limbwise(*this, other, [](const limb& lhs, const limb& rhs) {
            return lhs & ~rhs;
        });
    }

    bigint bitwise_nand(const bigint& other) const {
        return ~(operator&(other));
    }

    bigint bitwise_nor(const bigint& other) const {
        return ~(operator|(other));
    }

    bigint bitwise_xnor(const bigint& other) const {
        return ~(operator^(other));
    }

    bigint operator~() const {
        return -(*this + bigint::one());
    }

    bigint operator<<(int shift) const {
        return tryte_shift_left(shift);
    }

    bigint operator>>(int shift) const {
        return tryte_shift_right(shift);
    }

    bigint rotate_left_tbits(int count) const {
        return trit_shift_left(count);
    }

    bigint rotate_right_tbits(int count) const {
        return trit_shift_right(count);
    }

    bigint trit_shift_left(int count) const {
        if (count <= 0) {
            return *this;
        }
        const int limb_shift = count / limb::TRITS;
        const int remainder = count % limb::TRITS;
        bigint result = shift_limbs(static_cast<std::size_t>(limb_shift));
        if (remainder > 0) {
            result = multiply_by_power_of_three(result, remainder);
        }
        return result;
    }

    bigint trit_shift_right(int count) const {
        if (count <= 0) {
            return *this;
        }
        if (is_zero()) {
            return *this;
        }
        bigint result = *this;
        const bool was_negative = negative_;
        const int limb_shift = count / limb::TRITS;
        const int remainder = count % limb::TRITS;
        if (limb_shift > 0) {
            const std::size_t shift_limbs = static_cast<std::size_t>(limb_shift);
            bool truncated = false;
            if (shift_limbs >= result.limbs_.size()) {
                truncated = !result.limbs_.empty();
                result.limbs_.clear();
                result.negative_ = false;
            } else {
                for (std::size_t index = 0; index < shift_limbs; ++index) {
                    if (!result.limbs_[index].is_zero()) {
                        truncated = true;
                        break;
                    }
                }
                result.limbs_.erase(result.limbs_.begin(),
                                     result.limbs_.begin() + shift_limbs);
                result.normalize();
            }
            if (was_negative && truncated) {
                result -= bigint::one();
            }
        }
        if (remainder > 0) {
            const detail::limb_int128 pow3 = detail::POW3[remainder];
            const bigint divisor(limb::from_value(pow3));
            const bool numerator_negative = result.is_negative();
            const auto [quotient, remainder_value] = div_mod(result, divisor);
            result = quotient;
            if (numerator_negative && !remainder_value.is_zero()) {
                result -= bigint::one();
            }
        }
        return result;
    }

    bigint tryte_shift_left(int count) const {
        if (count <= 0) {
            return *this;
        }
        constexpr int TRITS_PER_TRYTE = 3;
        const long long trit_count = static_cast<long long>(count) * TRITS_PER_TRYTE;
        if (trit_count > std::numeric_limits<int>::max()) {
            throw std::overflow_error("tryte shift count too large");
        }
        return trit_shift_left(static_cast<int>(trit_count));
    }

    bigint tryte_shift_right(int count) const {
        if (count <= 0) {
            return *this;
        }
        constexpr int TRITS_PER_TRYTE = 3;
        const long long trit_count = static_cast<long long>(count) * TRITS_PER_TRYTE;
        if (trit_count > std::numeric_limits<int>::max()) {
            throw std::overflow_error("tryte shift count too large");
        }
        return trit_shift_right(static_cast<int>(trit_count));
    }

    static bigint gcd(bigint lhs, bigint rhs) {
        if (lhs.is_zero()) {
            return rhs.abs();
        }
        if (rhs.is_zero()) {
            return lhs.abs();
        }
        while (!rhs.is_zero()) {
            const bigint remainder = lhs % rhs;
            lhs = rhs;
            rhs = remainder;
        }
        return lhs.abs();
    }

    static bigint mod_pow(bigint base, bigint exponent, const bigint& modulus) {
        if (modulus.is_zero()) {
            throw std::domain_error("modulus must be non-zero");
        }
        if (modulus.is_negative()) {
            throw std::domain_error("modulus must be positive");
        }
        if (exponent.is_negative()) {
            throw std::domain_error("negative exponent");
        }
        bigint result = bigint::one();
        base %= modulus;
        if (base.is_negative()) {
            base += modulus;
        }
        const bigint two = bigint(2);
        while (!exponent.is_zero()) {
            const auto [quotient, remainder] = div_mod(exponent, two);
            if (!remainder.is_zero()) {
                result = (result * base) % modulus;
            }
            base = (base * base) % modulus;
            exponent = quotient;
        }
        return result;
    }

    friend std::vector<limb> signed_limbs(const bigint&);
    friend std::vector<std::int8_t> signed_trits(const bigint&);
    friend bigint from_signed_trits(std::vector<std::int8_t>);
    friend bigint from_signed_limbs(std::vector<limb>);
    template <typename Fn>
    friend bigint expected_bitwise(const bigint&, const bigint&, Fn);

private:
    template <typename Fn>
    static bigint apply_limbwise(const bigint& lhs, const bigint& rhs, Fn fn) {
        const std::size_t size = std::max(lhs.limb_count(), rhs.limb_count());
        const auto lhs_digits = lhs.signed_limbs(size);
        const auto rhs_digits = rhs.signed_limbs(size);
        std::vector<limb> result;
        result.reserve(size);
        for (std::size_t index = 0; index < size; ++index) {
            result.push_back(fn(lhs_digits[index], rhs_digits[index]));
        }
        return from_signed_limbs(result);
    }

    std::vector<limb> signed_limbs(std::size_t min_size = 0) const {
        const std::size_t target = std::max(min_size, limbs_.size());
        std::vector<limb> result;
        result.reserve(target);
        for (std::size_t index = 0; index < target; ++index) {
            limb value = limb::zero();
            if (index < limbs_.size()) {
                value = limbs_[index];
            }
            if (negative_) {
                value = -value;
            }
            result.push_back(value);
        }
        return result;
    }

    std::vector<std::int8_t> signed_trits() const {
        const auto digits = signed_limbs();
        std::vector<std::int8_t> trits;
        trits.reserve(digits.size() * limb::TRITS);
        for (const auto& digit : digits) {
            const auto chunk = digit.to_trits();
            trits.insert(trits.end(), chunk.begin(), chunk.end());
        }
        return trits;
    }

    static bigint from_signed_trits(std::vector<std::int8_t> trits) {
        if (trits.empty()) {
            return bigint::zero();
        }
        constexpr int trits_per_limb = limb::TRITS;
        const std::size_t padded =
            ((trits.size() + trits_per_limb - 1) / trits_per_limb) * trits_per_limb;
        trits.resize(padded, 0);
        std::vector<limb> digits;
        digits.reserve(trits.size() / trits_per_limb);
        for (std::size_t index = 0; index < trits.size(); index += trits_per_limb) {
            std::array<std::int8_t, limb::TRITS> chunk{};
            for (std::size_t offset = 0; offset < trits_per_limb; ++offset) {
                chunk[offset] = trits[index + offset];
            }
            digits.push_back(limb::from_trits(chunk));
        }
        return from_signed_limbs(digits);
    }

    static bigint from_signed_limbs(std::vector<limb> digits) {
        if (digits.empty()) {
            return bigint::zero();
        }
        detail::limb_int128 carry = 0;
        for (std::size_t index = 0; index < digits.size(); ++index) {
            detail::limb_int128 value = digits[index].to_value() + carry;
            const auto [digit, next_carry] = digit_and_carry(value);
            digits[index] = digit;
            carry = next_carry;
        }
        while (carry != 0) {
            const auto [digit, next_carry] = digit_and_carry(carry);
            digits.push_back(digit);
            carry = next_carry;
        }
        normalize_magnitude(digits);
        if (digits.empty()) {
            return bigint::zero();
        }
        bool negative = false;
        for (std::size_t index = digits.size(); index-- > 0;) {
            if (!digits[index].is_zero()) {
                negative = digits[index].is_negative();
                break;
            }
        }
        if (negative) {
            for (auto& digit : digits) {
                digit = -digit;
            }
            normalize_magnitude(digits);
        }
        bigint result;
        result.limbs_ = std::move(digits);
        result.negative_ = negative && !result.limbs_.empty();
        return result;
    }

    static void normalize_magnitude(std::vector<limb>& digits) {
        while (!digits.empty() && digits.back().is_zero()) {
            digits.pop_back();
        }
    }

    static std::strong_ordering compare_magnitude_vectors(const std::vector<limb>& lhs,
                                                          const std::vector<limb>& rhs) noexcept {
        if (lhs.size() != rhs.size()) {
            return lhs.size() < rhs.size() ? std::strong_ordering::less
                                           : std::strong_ordering::greater;
        }
        for (std::size_t index = lhs.size(); index-- > 0;) {
            const auto cmp = lhs[index] <=> rhs[index];
            if (cmp != std::strong_ordering::equal) {
                return cmp;
            }
        }
        return std::strong_ordering::equal;
    }

    static std::pair<std::vector<limb>, std::vector<limb>>
    divide_magnitude(std::vector<limb> dividend, const std::vector<limb>& divisor) {
        normalize_magnitude(dividend);
        std::vector<limb> divisor_digits = divisor;
        normalize_magnitude(divisor_digits);
        if (divisor_digits.empty()) {
            throw std::domain_error("division by zero");
        }
        if (dividend.empty()) {
            return {{}, {}};
        }
        std::vector<std::vector<limb>> scaled;
        std::vector<std::vector<limb>> multiples;
        scaled.push_back(divisor_digits);
        multiples.push_back(std::vector<limb>{limb::one()});
        while (true) {
            auto next_scaled = add_magnitude(scaled.back(), scaled.back());
            normalize_magnitude(next_scaled);
            if (compare_magnitude_vectors(next_scaled, dividend) == std::strong_ordering::greater) {
                break;
            }
            scaled.push_back(std::move(next_scaled));
            auto next_multiple = add_magnitude(multiples.back(), multiples.back());
            normalize_magnitude(next_multiple);
            multiples.push_back(std::move(next_multiple));
        }
        std::vector<limb> quotient;
        std::vector<limb> remainder = std::move(dividend);
        for (int index = static_cast<int>(scaled.size()) - 1; index >= 0; --index) {
            if (!remainder.empty() &&
                compare_magnitude_vectors(remainder, scaled[index]) != std::strong_ordering::less) {
                remainder = subtract_magnitude(remainder, scaled[index]);
                normalize_magnitude(remainder);
                quotient = add_magnitude(quotient, multiples[index]);
                normalize_magnitude(quotient);
            }
        }
        normalize_magnitude(quotient);
        normalize_magnitude(remainder);
        return {quotient, remainder};
    }
    static std::vector<limb> add_magnitude(const std::vector<limb>& lhs,
                                            const std::vector<limb>& rhs) {
        const std::size_t max_len = std::max(lhs.size(), rhs.size());
        std::vector<limb> result;
        result.reserve(max_len + 1);
        detail::limb_int128 carry = 0;
        for (std::size_t index = 0; index < max_len; ++index) {
            detail::limb_int128 a_val = 0;
            detail::limb_int128 b_val = 0;
            if (index < lhs.size()) {
                a_val = lhs[index].to_value();
            }
            if (index < rhs.size()) {
                b_val = rhs[index].to_value();
            }
            const auto [digit, next_carry] = digit_and_carry(a_val + b_val + carry);
            result.push_back(digit);
            carry = next_carry;
        }
        while (carry != 0) {
            const auto [digit, next_carry] = digit_and_carry(carry);
            result.push_back(digit);
            carry = next_carry;
        }
        return result;
    }

    static std::vector<limb> subtract_magnitude(const std::vector<limb>& lhs,
                                                 const std::vector<limb>& rhs) {
        std::vector<limb> result;
        result.reserve(lhs.size());
        detail::limb_int128 carry = 0;
        for (std::size_t index = 0; index < lhs.size(); ++index) {
            detail::limb_int128 a_val = lhs[index].to_value();
            detail::limb_int128 b_val = 0;
            if (index < rhs.size()) {
                b_val = rhs[index].to_value();
            }
            const auto [digit, next_carry] = digit_and_carry(a_val - b_val + carry);
            result.push_back(digit);
            carry = next_carry;
        }
        return result;
    }

    static std::vector<limb> multiply_magnitude_by_small(const std::vector<limb>& digits,
                                                         detail::limb_int128 multiplier) {
        if (digits.empty() || multiplier == 0) {
            return {};
        }
        std::vector<limb> result;
        result.reserve(digits.size() + 2);
        detail::limb_int128 carry = 0;
        for (const auto& digit : digits) {
            const auto product = digit.to_value() * multiplier + carry;
            const auto [value, next_carry] = digit_and_carry(product);
            result.push_back(value);
            carry = next_carry;
        }
        while (carry != 0) {
            const auto [value, next_carry] = digit_and_carry(carry);
            result.push_back(value);
            carry = next_carry;
        }
        return result;
    }

    static std::vector<limb> divide_magnitude_by_small(const std::vector<limb>& digits,
                                                       detail::limb_int128 divisor) {
        if (digits.empty() || divisor == 0) {
            return {};
        }
        const std::vector<limb> divisor_digits{limb::from_value(divisor)};
        const auto [quotient, remainder] = divide_magnitude(digits, divisor_digits);
        (void)remainder;
        return quotient;
    }

    static std::vector<limb> multiply_schoolbook(const std::vector<limb>& lhs,
                                                 const std::vector<limb>& rhs) {
        if (lhs.empty() || rhs.empty()) {
            return {};
        }
        const std::size_t result_size = lhs.size() + rhs.size() + 2;
        thread_local std::vector<detail::limb_int128> accumulation;
        thread_local std::vector<limb> normalized;
        accumulation.assign(result_size, 0);
        for (std::size_t i = 0; i < lhs.size(); ++i) {
            const auto lhs_digit = lhs[i];
            auto* row = accumulation.data() + i;
            std::size_t j = 0;
            for (; j + 1 < rhs.size(); j += 2) {
                const auto rhs0 = rhs[j];
                const auto rhs1 = rhs[j + 1];
                const auto [low0, high0] = limb::mul_wide(lhs_digit, rhs0);
                row[j] += low0.to_value();
                row[j + 1] += high0.to_value();
                const auto [low1, high1] = limb::mul_wide(lhs_digit, rhs1);
                row[j + 1] += low1.to_value();
                row[j + 2] += high1.to_value();
            }
            for (; j < rhs.size(); ++j) {
                const auto rhs_digit = rhs[j];
                const auto [low, high] = limb::mul_wide(lhs_digit, rhs_digit);
                row[j] += low.to_value();
                row[j + 1] += high.to_value();
            }
        }
        normalized.clear();
        normalized.reserve(result_size);
        detail::limb_int128 carry = 0;
        for (std::size_t index = 0; index < result_size; ++index) {
            const auto value = accumulation[index] + carry;
            const auto [digit, next_carry] = digit_and_carry(value);
            normalized.push_back(digit);
            carry = next_carry;
        }
        while (carry != 0) {
            const auto [digit, next_carry] = digit_and_carry(carry);
            normalized.push_back(digit);
            carry = next_carry;
        }
        while (!normalized.empty() && normalized.back().is_zero()) {
            normalized.pop_back();
        }
        std::vector<limb> result = std::move(normalized);
        normalized.clear();
        normalized.reserve(result_size);
        return result;
    }

    static std::vector<limb> multiply_magnitude(const std::vector<limb>& lhs,
                                                 const std::vector<limb>& rhs) {
        if (lhs.empty() || rhs.empty()) {
            return {};
        }
        constexpr std::size_t KARATSUBA_THRESHOLD = 128;
        if (lhs.size() + rhs.size() <= KARATSUBA_THRESHOLD) {
            return multiply_schoolbook(lhs, rhs);
        }
        const std::size_t n = std::max(lhs.size(), rhs.size());
        const std::size_t half = n / 2;
        auto split_low = [&](const std::vector<limb>& value) {
            return std::vector<limb>(value.begin(),
                                     value.begin() + std::min(half, value.size()));
        };
        auto split_high = [&](const std::vector<limb>& value) {
            if (value.size() <= half) {
                return std::vector<limb>{};
            }
            return std::vector<limb>(value.begin() + half, value.end());
        };
        const auto lhs_low = split_low(lhs);
        const auto lhs_high = split_high(lhs);
        const auto rhs_low = split_low(rhs);
        const auto rhs_high = split_high(rhs);
        const auto z0 = multiply_magnitude(lhs_low, rhs_low);
        const auto z2 = multiply_magnitude(lhs_high, rhs_high);
        const auto lhs_sum = add_magnitude(lhs_low, lhs_high);
        const auto rhs_sum = add_magnitude(rhs_low, rhs_high);
        auto z1 = multiply_magnitude(lhs_sum, rhs_sum);
        auto z1_minus_z0 = subtract_magnitude(z1, z0);
        auto z1_final = subtract_magnitude(z1_minus_z0, z2);
        auto shift_and_add = [&](std::vector<limb>& target,
                                 const std::vector<limb>& value,
                                 std::size_t shift) {
            if (value.empty()) {
                return;
            }
            std::vector<limb> shifted;
            shifted.reserve(value.size() + shift);
            shifted.insert(shifted.end(), shift, limb::zero());
            shifted.insert(shifted.end(), value.begin(), value.end());
            target = add_magnitude(target, shifted);
        };
        std::vector<limb> result = z0;
        shift_and_add(result, z1_final, half);
        shift_and_add(result, z2, half * 2);
        return result;
    }

    static std::pair<limb, detail::limb_int128> digit_and_carry(detail::limb_int128 value) {
        detail::limb_int128 remainder = value % detail::RADIX;
        detail::limb_int128 carry = (value - remainder) / detail::RADIX;
        if (remainder > detail::MAX_VALUE) {
            remainder -= detail::RADIX;
            ++carry;
        } else if (remainder < detail::MIN_VALUE) {
            remainder += detail::RADIX;
            --carry;
        }
        return {limb::from_value(remainder), carry};
    }

    static int max_limbs_for(detail::limb_int128 limit) {
        if (limit <= 0) {
            return 1;
        }
        int count = 1;
        while (limit >= detail::RADIX) {
            limit /= detail::RADIX;
            ++count;
        }
        return count;
    }

    std::strong_ordering compare(const bigint& other) const noexcept {
        if (negative_ != other.negative_) {
            return negative_ ? std::strong_ordering::less : std::strong_ordering::greater;
        }
        if (negative_) {
            const auto magnitude_cmp = compare_magnitude(other);
            if (magnitude_cmp == std::strong_ordering::equal) {
                return std::strong_ordering::equal;
            }
            return magnitude_cmp == std::strong_ordering::less ? std::strong_ordering::greater
                                                               : std::strong_ordering::less;
        }
        return compare_magnitude(other);
    }

    std::strong_ordering compare_magnitude(const bigint& other) const noexcept {
        return compare_magnitude_vectors(limbs_, other.limbs_);
    }

    void normalize() {
        while (!limbs_.empty() && limbs_.back().is_zero()) {
            limbs_.pop_back();
        }
        if (limbs_.empty()) {
            negative_ = false;
        }
    }

    std::vector<limb> limbs_;
    bool negative_ = false;
};

} // namespace t81::core
