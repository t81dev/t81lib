#pragma once

#include <string_view>
#include <t81/core/bigint.hpp>

namespace t81::io {

// Placeholder parsing function from ternary string.
inline t81::core::bigint parse_ternary(std::string_view) {
    return t81::core::bigint{};
}

} // namespace t81::io
