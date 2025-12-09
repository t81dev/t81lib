#pragma once

#include <string>
#include <ostream>
#include <t81/core/bigint.hpp>

namespace t81::io {

// Placeholder formatted output functions.

inline std::string to_string_ternary(const t81::core::bigint&) {
    return "0t"; // placeholder
}

inline std::ostream& operator<<(std::ostream& os, const t81::core::bigint& v) {
    return os << to_string_ternary(v);
}

} // namespace t81::io
