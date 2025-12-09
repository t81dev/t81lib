#pragma once

#include <array>
#include <cstdint>
#include <compare>

namespace t81::core {

/// Placeholder fixed-size packed balanced-ternary limb type.
/// Replace with your real implementation.
class limb {
public:
    static constexpr int TRITS = 48; // example size

    constexpr limb() noexcept = default;

    friend constexpr auto operator<=>(const limb&, const limb&) = default;
};

} // namespace t81::core
