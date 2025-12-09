#pragma once

#include <ostream>
#include <t81/core/limb.hpp>

namespace t81::util {

// Placeholder debug dump for limb.
inline std::ostream& dump(std::ostream& os, const t81::core::limb&) {
    return os << "limb{...}";
}

} // namespace t81::util
