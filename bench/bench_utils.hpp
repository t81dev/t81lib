#pragma once

#include <array>
#include <cstdint>
#include <numeric>
#include <random>

#include "t81/core/T81Limb.hpp"

namespace bench {

inline std::mt19937& rng() {
    static std::mt19937 instance(420);
    return instance;
}

inline t81::core::T81Limb random_limb() {
    t81::core::T81Limb limb;
    std::uniform_int_distribution<int> tryte_dist(-13, 13);
    for (int i = 0; i < t81::core::T81Limb::TRYTES; ++i) {
        limb.set_tryte(i, static_cast<int8_t>(tryte_dist(rng())));
    }
    return limb;
}

inline std::array<int8_t, t81::core::T81Limb::TRITS> random_trits() {
    std::array<int8_t, t81::core::T81Limb::TRITS> digits{};
    std::uniform_int_distribution<int> trit_dist(-1, 1);
    for (auto& digit : digits) {
        digit = static_cast<int8_t>(trit_dist(rng()));
    }
    return digits;
}

inline t81::core::T81Limb shift_left_trytes(const t81::core::T81Limb& limb, int count) {
    auto source = limb.to_trits();
    std::array<int8_t, t81::core::T81Limb::TRITS> shifted{};
    int shift = count * 3;
    for (int idx = t81::core::T81Limb::TRITS - 1; idx >= shift; --idx) {
        shifted[idx] = source[idx - shift];
    }
    return t81::core::T81Limb::from_trits(shifted);
}

inline int64_t to_int64(const t81::core::T81Limb& limb) {
    auto trits = limb.to_trits();
    int64_t value = 0;
    for (int idx = t81::core::T81Limb::TRITS - 1; idx >= 0; --idx) {
        value = value * 3 + static_cast<int>(trits[idx]);
    }
    return value;
}

template <typename Container>
inline Container random_limbs() {
    Container buf;
    for (std::size_t i = 0; i < buf.size(); ++i) {
        buf[i] = random_limb();
    }
    return buf;
}

} // namespace bench
