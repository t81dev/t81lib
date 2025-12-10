#pragma once

#include <ostream>

#include <t81/io/format.hpp>
#include <t81/core/bigint.hpp>
#include <t81/core/limb.hpp>

namespace t81::util {

inline std::ostream& dump(std::ostream& os, const t81::core::limb& value) {
    return os << "limb(" << t81::io::to_string(value) << ')';
}

inline std::ostream& dump(std::ostream& os, const t81::core::bigint& value) {
    return os << "bigint(" << t81::io::to_string(value) << ')';
}

} // namespace t81::util
