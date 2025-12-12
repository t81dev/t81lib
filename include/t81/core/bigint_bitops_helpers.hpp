// include/t81/core/bigint_bitops_helpers.hpp — Bitwise helper APIs for bigint operations.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <t81/core/bigint.hpp>
#include <t81/core/limb.hpp>

namespace t81::core {

    inline std::vector<limb> signed_limbs(const bigint &value) {
        if (value.is_zero()) {
            return {limb::zero()};
        }
        std::vector<limb> result;
        result.reserve(value.limb_count());
        const bool negative = value.is_negative();
        for (std::size_t index = 0; index < value.limb_count(); ++index) {
            auto digit = value.limb_at(index);
            if (negative) {
                digit = -digit;
            }
            result.push_back(digit);
        }
        return result;
    }

    inline std::vector<std::int8_t> signed_trits(const bigint &value) {
        const auto digits = signed_limbs(value);
        std::vector<std::int8_t> trits;
        trits.reserve(digits.size() * limb::TRITS);
        for (const auto &digit : digits) {
            const auto chunk = digit.to_trits();
            trits.insert(trits.end(), chunk.begin(), chunk.end());
        }
        return trits;
    }

    inline bigint from_signed_limbs(std::vector<limb> digits) {
        return bigint::from_signed_limbs(std::move(digits));
    }

    inline bigint from_signed_trits(std::vector<std::int8_t> trits) {
        if (trits.empty()) {
            return bigint::zero();
        }
        constexpr std::size_t trits_per_limb = limb::TRITS;
        const std::size_t padded =
            ((trits.size() + trits_per_limb - 1) / trits_per_limb) * trits_per_limb;
        trits.resize(padded, 0);
        std::vector<limb> digits;
        digits.reserve(padded / trits_per_limb);
        for (std::size_t index = 0; index < padded; index += trits_per_limb) {
            std::array<std::int8_t, trits_per_limb> chunk{};
            for (std::size_t offset = 0; offset < trits_per_limb; ++offset) {
                chunk[offset] = trits[index + offset];
            }
            digits.push_back(limb::from_trits(chunk));
        }
        return bigint::from_signed_limbs(std::move(digits));
    }

    template <typename Fn>
    inline bigint expected_bitwise(const bigint &lhs, const bigint &rhs, Fn fn) {
        auto lhs_digits = signed_limbs(lhs);
        auto rhs_digits = signed_limbs(rhs);
        const std::size_t size = std::max(lhs_digits.size(), rhs_digits.size());
        lhs_digits.resize(size, limb::zero());
        rhs_digits.resize(size, limb::zero());
        std::vector<limb> result;
        result.reserve(size);
        for (std::size_t index = 0; index < size; ++index) {
            result.push_back(fn(lhs_digits[index], rhs_digits[index]));
        }
        return from_signed_limbs(std::move(result));
    }

    inline bigint expected_not(const bigint &value) { return -(value + bigint::one()); }

    inline bigint expected_trit_shift_left(const bigint &value, int count) {
        if (count <= 0) {
            return value;
        }
        auto trits = signed_trits(value);
        trits.insert(trits.begin(), static_cast<std::size_t>(count), 0);
        return from_signed_trits(std::move(trits));
    }

    inline bigint expected_trit_shift_right(const bigint &value, int count) {
        if (count <= 0) {
            return value;
        }
        bigint result = value;
        const std::size_t limb_shift = static_cast<std::size_t>(count / limb::TRITS);
        const int remainder_shift = count % limb::TRITS;
        if (limb_shift > 0) {
            std::vector<limb> limbs;
            limbs.reserve(result.limb_count());
            for (std::size_t index = 0; index < result.limb_count(); ++index) {
                limbs.push_back(result.limb_at(index));
            }
            bool truncated = false;
            if (limb_shift >= limbs.size()) {
                truncated = !limbs.empty();
                limbs.clear();
            } else {
                for (std::size_t index = 0; index < limb_shift; ++index) {
                    if (!limbs[index].is_zero()) {
                        truncated = true;
                        break;
                    }
                }
                limbs.erase(limbs.begin(), limbs.begin() + limb_shift);
            }
            result = bigint::from_limbs(std::move(limbs), value.is_negative());
            if (value.is_negative() && truncated) {
                result -= bigint::one();
            }
        }
        if (remainder_shift > 0) {
            const bool numerator_negative = result.is_negative();
            const bigint divisor =
                bigint::multiply_by_power_of_three(bigint::one(), remainder_shift);
            const auto [quotient, remainder_value] = bigint::div_mod(result, divisor);
            result = quotient;
            if (numerator_negative && !remainder_value.is_zero()) {
                result -= bigint::one();
            }
        }
        return result;
    }

    inline bigint expected_tryte_shift_left(const bigint &value, int count) {
        return expected_trit_shift_left(value, count * 3);
    }

    inline bigint expected_tryte_shift_right(const bigint &value, int count) {
        if (count <= 0) {
            return value;
        }
        constexpr int TRITS_PER_TRYTE = 3;
        const long long trit_count = static_cast<long long>(count) * TRITS_PER_TRYTE;
        if (trit_count > std::numeric_limits<int>::max()) {
            throw std::overflow_error("tryte shift count too large");
        }
        return expected_trit_shift_right(value, static_cast<int>(trit_count));
    }

} // namespace t81::core
