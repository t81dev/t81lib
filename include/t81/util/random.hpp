#pragma once

#include <random>
#include <t81/core/limb.hpp>

namespace t81::util {

// Placeholder random generator for limbs.
inline t81::core::limb random_limb(std::mt19937_64&) {
    return t81::core::limb{};
}

} // namespace t81::util
