#pragma once

#include <vector>
#include <cstdint>
#include <compare>
#include <t81/core/limb.hpp>

namespace t81::core {

/// Placeholder arbitrary-precision balanced-ternary integer,
/// implemented as a vector of limbs.
class bigint {
public:
    constexpr bigint() = default;

    friend constexpr auto operator<=>(const bigint&, const bigint&) = default;

private:
    std::vector<limb> limbs_;
    bool negative_ = false;
};

} // namespace t81::core
