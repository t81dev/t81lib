// include/t81/core/traits.hpp — Compile-time traits for limbs and bigints.

#pragma once

#include <t81/core/bigint.hpp>
#include <t81/core/limb.hpp>
#include <type_traits>

namespace t81::core {

    template <typename T> struct is_ternary_numeric : std::false_type {};

    template <> struct is_ternary_numeric<limb> : std::true_type {};

    template <> struct is_ternary_numeric<bigint> : std::true_type {};

    template <typename T> inline constexpr bool is_ternary_numeric_v = is_ternary_numeric<T>::value;

} // namespace t81::core
