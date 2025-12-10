#pragma once

#include <cstddef>
#include <random>

#include <t81/core/bigint.hpp>

namespace t81::util {

namespace {

inline const t81::core::bigint& limb_radix() {
    static const t81::core::bigint value = [] {
        t81::core::bigint result = t81::core::bigint::one();
        const t81::core::limb three = t81::core::limb::from_value(3);
        for (int count = 0; count < t81::core::limb::TRITS; ++count) {
            result *= t81::core::bigint(three);
        }
        return result;
    }();
    return value;
}

} // namespace

inline t81::core::limb random_limb(std::mt19937_64& generator) {
    static std::uniform_int_distribution<int> tryte_dist(0, 26);
    t81::core::limb result;
    for (int index = 0; index < t81::core::limb::TRYTES; ++index) {
        result.set_tryte(index, static_cast<t81::core::limb::tryte_t>(tryte_dist(generator)));
    }
    return result;
}

inline t81::core::bigint random_bigint(std::mt19937_64& generator,
                                       std::size_t limb_count,
                                       bool allow_negative = true) {
    if (limb_count == 0) {
        return t81::core::bigint::zero();
    }
    t81::core::bigint value;
    for (std::size_t index = 0; index < limb_count; ++index) {
        value *= limb_radix();
        value += t81::core::bigint(random_limb(generator));
    }
    if (allow_negative) {
        static std::bernoulli_distribution sign_dist(0.5);
        if (sign_dist(generator) && !value.is_zero()) {
            value = -value;
        }
    }
    return value;
}

} // namespace t81::util
