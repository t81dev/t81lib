#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
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

inline bigint from_signed_limbs(std::vector<limb> digits) {
    if (digits.empty()) {
        return bigint::zero();
    }
    bigint result;
    for (std::size_t index = digits.size(); index-- > 0;) {
        result = result.shift_limbs(1);
        if (!digits[index].is_zero()) {
            result += bigint(digits[index]);
        }
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
    return from_signed_limbs(digits);
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
    return from_signed_limbs(result);
}

inline bigint expected_not(const bigint& value) {
    auto digits = signed_limbs(value);
    for (auto& digit : digits) {
        digit = -digit;
    }
    return from_signed_limbs(digits);
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
    if (static_cast<std::size_t>(count) >= trits.size()) {
        return bigint::zero();
    }
    trits.erase(trits.begin(), trits.begin() + count);
    return from_signed_trits(std::move(trits));
}

inline bigint expected_tryte_shift_left(const bigint& value, int count) {
    return expected_trit_shift_left(value, count * 3);
}

inline bigint expected_tryte_shift_right(const bigint& value, int count) {
    return expected_trit_shift_right(value, count * 3);
}

} // namespace t81::core
