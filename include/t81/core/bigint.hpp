#pragma once

#include <algorithm>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

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

private:
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
