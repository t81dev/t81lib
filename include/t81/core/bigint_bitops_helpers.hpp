// include/t81/core/bigint_bitops_helpers.hpp — Bitwise helper APIs for bigint operations.

// bigint_bitops_helpers.hpp - Testing utilities for bigint bitwise and trit/tryte shift helpers.
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

inline std::vector<limb> signed_limbs(const bigint& value) {
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

inline std::vector<std::int8_t> signed_trits(const bigint& value) {
    const auto digits = signed_limbs(value);
    std::vector<std::int8_t> trits;
    trits.reserve(digits.size() * limb::TRITS);
    for (const auto& digit : digits) {
        const auto chunk = digit.to_trits();
        trits.insert(trits.end(), chunk.begin(), chunk.end());
    }
    return trits;
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
inline bigint expected_bitwise(const bigint& lhs,
                              const bigint& rhs,
                              Fn fn) {
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
        return bigint::from_signed_limbs(std::move(result));
}

inline bigint expected_not(const bigint& value) {
    return -(value + bigint::one());
}

inline bigint expected_trit_shift_left(const bigint& value, int count) {
    if (count <= 0) {
        return value;
    }
    auto trits = signed_trits(value);
    trits.insert(trits.begin(), static_cast<std::size_t>(count), 0);
    return from_signed_trits(std::move(trits));
}

inline bigint expected_trit_shift_right(const bigint& value, int count) {
    if (count <= 0) {
        return value;
    }
    auto trits = signed_trits(value);
    if (trits.empty()) {
        return bigint::zero();
    }
    const std::size_t shift = static_cast<std::size_t>(count);
    const std::size_t limited_shift = std::min(shift, trits.size());
    const bool negative = value.is_negative();
    const bool truncated_nonzero =
        limited_shift > 0 &&
        std::any_of(trits.begin(), trits.begin() + limited_shift,
                    [](std::int8_t trit) { return trit != 0; });
    if (shift >= trits.size()) {
        if (negative && truncated_nonzero) {
            return -bigint::one();
        }
        return bigint::zero();
    }
    trits.erase(trits.begin(), trits.begin() + limited_shift);
    bigint result = from_signed_trits(std::move(trits));
    if (negative && truncated_nonzero) {
        result -= bigint::one();
    }
    return result;
}

inline bigint expected_tryte_shift_left(const bigint& value, int count) {
    return expected_trit_shift_left(value, count * 3);
}

inline bigint expected_tryte_shift_right(const bigint& value, int count) {
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
