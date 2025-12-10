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
        while (remainder >= abs_divisor) {
            bigint temp = abs_divisor;
            bigint multiple = bigint::one();
            while (true) {
                bigint doubled = temp;
                doubled += temp;
                if (remainder < doubled) {
                    break;
                }
                temp = std::move(doubled);
                multiple += multiple;
            }
            remainder -= temp;
            quotient += multiple;
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

    bigint operator~() const {
        auto digits = signed_limbs();
        for (auto& digit : digits) {
            digit = -digit;
        }
        return from_signed_limbs(digits);
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
        auto trits = signed_trits();
        trits.insert(trits.begin(), static_cast<std::size_t>(count), 0);
        return from_signed_trits(std::move(trits));
    }

    bigint trit_shift_right(int count) const {
        if (count <= 0) {
            return *this;
        }
        auto trits = signed_trits();
        if (static_cast<std::size_t>(count) >= trits.size()) {
            return bigint::zero();
        }
        trits.erase(trits.begin(), trits.begin() + count);
        return from_signed_trits(std::move(trits));
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
            return bigint::zero();
        }
        return trit_shift_right(static_cast<int>(trit_count));
    }

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

    static bigint from_signed_limbs(const std::vector<limb>& digits) {
        bigint result;
        for (std::size_t index = digits.size(); index-- > 0;) {
            result = result.shift_limbs(1);
            if (!digits[index].is_zero()) {
                result += bigint(digits[index]);
            }
        }
        return result;
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

    static std::vector<limb> multiply_magnitude(const std::vector<limb>& lhs,
                                                const std::vector<limb>& rhs) {
        if (lhs.empty() || rhs.empty()) {
            return {};
        }
        const std::size_t result_size = lhs.size() + rhs.size() + 2;
        std::vector<detail::limb_int128> accumulation(result_size, 0);
        for (std::size_t i = 0; i < lhs.size(); ++i) {
            for (std::size_t j = 0; j < rhs.size(); ++j) {
                const auto [low, high] = limb::mul_wide(lhs[i], rhs[j]);
                accumulation[i + j] += low.to_value();
                accumulation[i + j + 1] += high.to_value();
            }
        }
        std::vector<limb> result;
        result.reserve(result_size);
        detail::limb_int128 carry = 0;
        for (std::size_t index = 0; index < accumulation.size(); ++index) {
            const auto [digit, next_carry] = digit_and_carry(accumulation[index] + carry);
            result.push_back(digit);
            carry = next_carry;
        }
        while (carry != 0) {
            const auto [digit, next_carry] = digit_and_carry(carry);
            result.push_back(digit);
            carry = next_carry;
        }
        while (!result.empty() && result.back().is_zero()) {
            result.pop_back();
        }
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
        if (limbs_.size() != other.limbs_.size()) {
            return limbs_.size() < other.limbs_.size() ? std::strong_ordering::less
                                                       : std::strong_ordering::greater;
        }
        for (std::size_t index = limbs_.size(); index-- > 0;) {
            const auto cmp = limbs_[index] <=> other.limbs_[index];
            if (cmp != std::strong_ordering::equal) {
                return cmp;
            }
        }
        return std::strong_ordering::equal;
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
