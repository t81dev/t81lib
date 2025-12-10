#pragma once

#include <array>
#include <cstdint>

namespace t81::core::detail {

namespace {

constexpr std::array<std::array<std::int8_t, 3>, 27> build_tryte_to_trits() {
    std::array<std::array<std::int8_t, 3>, 27> table{};
    for (int t2 = -1; t2 <= 1; ++t2) {
        for (int t1 = -1; t1 <= 1; ++t1) {
            for (int t0 = -1; t0 <= 1; ++t0) {
                int tryte_index = (t0 + 3 * t1 + 9 * t2) + 13;
                table[tryte_index] = {
                    static_cast<std::int8_t>(t0),
                    static_cast<std::int8_t>(t1),
                    static_cast<std::int8_t>(t2),
                };
            }
        }
    }
    return table;
}

constexpr std::array<std::uint8_t, 27> build_trits_to_tryte() {
    std::array<std::uint8_t, 27> table{};
    for (int t2 = -1; t2 <= 1; ++t2) {
        for (int t1 = -1; t1 <= 1; ++t1) {
            for (int t0 = -1; t0 <= 1; ++t0) {
                int tri_index = (t0 + 1) + 3 * (t1 + 1) + 9 * (t2 + 1);
                int tryte_index = (t0 + 3 * t1 + 9 * t2) + 13;
                table[static_cast<std::size_t>(tri_index)] =
                    static_cast<std::uint8_t>(tryte_index);
            }
        }
    }
    return table;
}

} // namespace

inline constexpr std::array<std::array<std::int8_t, 3>, 27> TRYTE_TO_TRITS =
    build_tryte_to_trits();
inline constexpr std::array<std::uint8_t, 27> TRITS_TO_TRYTE =
    build_trits_to_tryte();

} // namespace t81::core::detail
