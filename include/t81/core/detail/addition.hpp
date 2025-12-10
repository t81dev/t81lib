// include/t81/core/detail/addition.hpp — Addition helper declarations for bigint.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include <t81/core/detail/lut.hpp>

namespace t81::core {
class limb;
} // namespace t81::core

namespace t81::core::detail {

inline constexpr std::array<int, 27> build_tryte_values() {
    std::array<int, 27> values{};
    for (std::size_t index = 0; index < 27; ++index) {
        const auto& triple = TRYTE_TO_TRITS[index];
        values[index] = triple[0] + 3 * triple[1] + 9 * triple[2];
    }
    return values;
}

inline constexpr std::array<std::uint8_t, 27> build_value_to_tryte() {
    std::array<std::uint8_t, 27> table{};
    for (std::size_t index = 0; index < 27; ++index) {
        const auto& triple = TRYTE_TO_TRITS[index];
        const int value = triple[0] + 3 * triple[1] + 9 * triple[2];
        table[static_cast<std::size_t>(value + 13)] = static_cast<std::uint8_t>(index);
    }
    return table;
}

inline constexpr auto TRYTE_VALUES = build_tryte_values();
inline constexpr auto VALUE_TO_TRYTE = build_value_to_tryte();

struct AddOutcome {
    std::uint8_t tryte_index;
    std::int8_t carry;
};

struct AddEntry {
    std::array<AddOutcome, 3> outcomes; // index 0->carry -1, 1->0, 2->1
};

inline constexpr std::array<std::array<AddEntry, 27>, 27> build_add_table() {
    std::array<std::array<AddEntry, 27>, 27> table{};
    const auto& values = TRYTE_VALUES;
    const auto& rev = VALUE_TO_TRYTE;
    for (std::size_t a = 0; a < 27; ++a) {
        for (std::size_t b = 0; b < 27; ++b) {
            const int sum_base = values[a] + values[b];
            for (int carry_in = -1; carry_in <= 1; ++carry_in) {
                const int sum = sum_base + carry_in;
                int carry_out = 0;
                int digit = sum;
                for (int candidate = -1; candidate <= 1; ++candidate) {
                    const int rem = sum - candidate * 27;
                    if (rem >= -13 && rem <= 13) {
                        digit = rem;
                        carry_out = candidate;
                        break;
                    }
                }
                table[a][b].outcomes[carry_in + 1] = {
                    rev[static_cast<std::size_t>(digit + 13)],
                    static_cast<std::int8_t>(carry_out),
                };
            }
        }
    }
    return table;
}

inline constexpr std::array<std::array<AddEntry, 27>, 27> ADD_TABLE = build_add_table();

inline constexpr std::array<std::array<int, 27>, 27> build_tryte_product_table() {
    std::array<std::array<int, 27>, 27> table{};
    for (std::size_t a = 0; a < 27; ++a) {
        for (std::size_t b = 0; b < 27; ++b) {
            table[a][b] = TRYTE_VALUES[a] * TRYTE_VALUES[b];
        }
    }
    return table;
}

inline constexpr std::array<std::array<int, 27>, 27> TRYTE_PRODUCT = build_tryte_product_table();

inline constexpr int tryte_value(std::uint8_t tryte) noexcept {
    return TRYTE_VALUES[tryte];
}

inline constexpr const AddEntry& add_entry(std::uint8_t lhs, std::uint8_t rhs) noexcept {
    return ADD_TABLE[lhs][rhs];
}

inline constexpr std::uint32_t encode_outcome(const AddOutcome& outcome) noexcept {
    const std::uint32_t tryte = outcome.tryte_index;
    const std::uint32_t carry = static_cast<std::uint32_t>(outcome.carry + 1); // map -1,0,1 -> 0,1,2
    return (tryte & 0x3F) | ((carry & 0x3U) << 6);
}

inline constexpr std::uint32_t pack_add_entry(const AddEntry& entry) noexcept {
    return encode_outcome(entry.outcomes[0]) | (encode_outcome(entry.outcomes[1]) << 8) |
           (encode_outcome(entry.outcomes[2]) << 16);
}

inline constexpr std::array<std::uint32_t, 27 * 27> build_packed_add_map() {
    std::array<std::uint32_t, 27 * 27> packed{};
    for (std::size_t a = 0; a < 27; ++a) {
        for (std::size_t b = 0; b < 27; ++b) {
            packed[a * 27 + b] = pack_add_entry(ADD_TABLE[a][b]);
        }
    }
    return packed;
}

inline constexpr std::array<std::uint32_t, 27 * 27> PACKED_ADD_MAP = build_packed_add_map();

inline constexpr std::uint32_t packed_add_entry(std::uint8_t lhs, std::uint8_t rhs) noexcept {
    return PACKED_ADD_MAP[lhs * 27 + rhs];
}

inline constexpr int PACKED_CHUNK_SIZE = 16;
inline constexpr int PACKED_CHUNK_COUNT =
    (static_cast<int>(PACKED_ADD_MAP.size()) + PACKED_CHUNK_SIZE - 1) / PACKED_CHUNK_SIZE;

struct PackedChunk {
    std::array<std::uint8_t, PACKED_CHUNK_SIZE> bytes[4];
};

inline constexpr PackedChunk build_chunk(std::size_t chunk_index) {
    PackedChunk chunk{};
    const std::size_t base = chunk_index * PACKED_CHUNK_SIZE;
    for (std::size_t lane = 0; lane < PACKED_CHUNK_SIZE; ++lane) {
        const std::size_t entry_index = base + lane;
        std::uint32_t value = 0;
        if (entry_index < PACKED_ADD_MAP.size()) {
            value = PACKED_ADD_MAP[entry_index];
        }
        chunk.bytes[0][lane] = static_cast<std::uint8_t>(value & 0xFFU);
        chunk.bytes[1][lane] = static_cast<std::uint8_t>((value >> 8) & 0xFFU);
        chunk.bytes[2][lane] = static_cast<std::uint8_t>((value >> 16) & 0xFFU);
        chunk.bytes[3][lane] = static_cast<std::uint8_t>((value >> 24) & 0xFFU);
    }
    return chunk;
}

inline constexpr std::array<PackedChunk, PACKED_CHUNK_COUNT> build_packed_chunks() {
    std::array<PackedChunk, PACKED_CHUNK_COUNT> chunks{};
    for (std::size_t chunk_index = 0; chunk_index < PACKED_CHUNK_COUNT; ++chunk_index) {
        chunks[chunk_index] = build_chunk(chunk_index);
    }
    return chunks;
}

inline constexpr std::array<PackedChunk, PACKED_CHUNK_COUNT> PACKED_CHUNKS =
    build_packed_chunks();

inline constexpr std::uint32_t select_packed_outcome(std::uint32_t packed_entry, int carry_in) noexcept {
    const int offset = (carry_in + 1) * 8;
    return (packed_entry >> offset) & 0xFFU;
}

inline constexpr int decode_carry(std::uint32_t outcome_byte) noexcept {
    return static_cast<int>((outcome_byte >> 6) & 0x3U) - 1;
}

inline constexpr std::uint32_t compose_packed_maps(std::uint32_t left, std::uint32_t right) noexcept {
    std::uint32_t result = 0;
    for (int carry_idx = -1; carry_idx <= 1; ++carry_idx) {
        const int offset = (carry_idx + 1);
        const std::uint32_t left_outcome = select_packed_outcome(left, carry_idx);
        const int carry_out = decode_carry(left_outcome);
        const std::uint32_t right_outcome = select_packed_outcome(right, carry_out);
        result |= right_outcome << (offset * 8);
    }
    return result;
}

} // namespace t81::core::detail
