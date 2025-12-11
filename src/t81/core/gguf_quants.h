#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace t81 {
namespace core {
namespace gguf {

constexpr std::size_t TQ1_TRITS_PER_BLOCK = 8;
constexpr std::size_t TQ1_BLOCK_ROWS = 32;
constexpr std::size_t TQ1_BLOCK_BITS = 13;
constexpr std::uint16_t TQ1_BLOCK_MAX = (1u << TQ1_BLOCK_BITS) - 1u;

inline float clamp_threshold(float threshold) {
    return std::clamp(threshold, 0.0f, 0.9999f);
}

inline std::int8_t quantize_trit(float value, float threshold) {
    const float clamped = std::fmax(-1.0f, std::fmin(1.0f, value));
    if (clamped >= threshold) {
        return 1;
    }
    if (clamped <= -threshold) {
        return -1;
    }
    return 0;
}

inline std::uint8_t trit_to_digit(std::int8_t trit) {
    if (trit > 0) {
        return 0;
    }
    if (trit < 0) {
        return 2;
    }
    return 1;
}

inline std::int8_t digit_to_trit(std::uint8_t digit) {
    switch (digit) {
        case 0:
            return 1;
        case 2:
            return -1;
        default:
            return 0;
    }
}

inline float compute_scale(const float* values, std::int64_t length) {
    float max_abs = 0.0f;
    for (std::int64_t index = 0; index < length; ++index) {
        const float candidate = std::fabs(values[index]);
        if (candidate > max_abs) {
            max_abs = candidate;
        }
    }
    return max_abs;
}

inline std::uint16_t pack_block(const std::array<std::uint8_t, TQ1_TRITS_PER_BLOCK>& digits) {
    std::uint16_t result = 0;
    for (std::size_t index = TQ1_TRITS_PER_BLOCK; index > 0; --index) {
        result = static_cast<std::uint16_t>(result * 3u + digits[index - 1]);
    }
    return static_cast<std::uint16_t>(result & TQ1_BLOCK_MAX);
}

inline void unpack_block(std::array<std::uint8_t, TQ1_TRITS_PER_BLOCK>& digits, std::uint16_t value) {
    std::uint16_t cursor = value;
    for (std::size_t index = 0; index < TQ1_TRITS_PER_BLOCK; ++index) {
        digits[index] = static_cast<std::uint8_t>(cursor % 3u);
        cursor /= 3u;
    }
}

inline float half_to_float(std::uint16_t half_bits) {
    const std::uint32_t sign = (half_bits & 0x8000u) << 16;
    std::uint32_t exponent = (half_bits & 0x7C00u) >> 10;
    std::uint32_t mantissa = half_bits & 0x03FFu;

    if (exponent == 0) {
        if (mantissa == 0) {
            std::uint32_t bits = sign;
            float result;
            std::memcpy(&result, &bits, sizeof(result));
            return result;
        }
        while ((mantissa & 0x0400u) == 0u) {
            mantissa <<= 1;
            --exponent;
        }
        mantissa &= 0x03FFu;
        ++exponent;
    } else if (exponent == 0x1Fu) {
        std::uint32_t bits = sign | 0x7F800000u | (mantissa << 13);
        float result;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }

    exponent = exponent + (127 - 15);
    std::uint32_t bits = sign | (exponent << 23) | (mantissa << 13);
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline void quantize_row_tq1_0(const float* values,
                               std::int64_t length,
                               float threshold,
                               float supplied_scale,
                               std::uint16_t* destination) {
    if (length <= 0 || destination == nullptr) {
        return;
    }
    const float scale = supplied_scale > 0.0f ? supplied_scale : compute_scale(values, length);
    const float clamped_threshold = clamp_threshold(threshold);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    const std::size_t blocks = static_cast<std::size_t>((length + TQ1_TRITS_PER_BLOCK - 1) / TQ1_TRITS_PER_BLOCK);

    for (std::size_t block_index = 0; block_index < blocks; ++block_index) {
        std::array<std::uint8_t, TQ1_TRITS_PER_BLOCK> digits{};
        for (std::size_t trit_index = 0; trit_index < TQ1_TRITS_PER_BLOCK; ++trit_index) {
            const std::size_t linear_index = block_index * TQ1_TRITS_PER_BLOCK + trit_index;
            std::int8_t trit = 0;
            if (linear_index < static_cast<std::size_t>(length) && scale > 0.0f) {
                const float normalized = values[linear_index] * inv_scale;
                trit = quantize_trit(normalized, clamped_threshold);
            }
            digits[trit_index] = trit_to_digit(trit);
        }
        destination[block_index] = pack_block(digits);
    }
}

inline void dequantize_row_tq1_0(const std::uint16_t* source,
                                 std::int64_t length,
                                 float scale,
                                 float* destination) {
    if (length <= 0 || destination == nullptr || source == nullptr) {
        return;
    }
    const std::size_t blocks = static_cast<std::size_t>((length + TQ1_TRITS_PER_BLOCK - 1) / TQ1_TRITS_PER_BLOCK);
    if (scale == 0.0f) {
        std::fill_n(destination, static_cast<std::size_t>(length), 0.0f);
        return;
    }
    std::size_t written = 0;
    for (std::size_t block_index = 0; block_index < blocks; ++block_index) {
        std::array<std::uint8_t, TQ1_TRITS_PER_BLOCK> digits{};
        unpack_block(digits, source[block_index]);
        for (std::size_t trit_index = 0;
             trit_index < TQ1_TRITS_PER_BLOCK && written < static_cast<std::size_t>(length);
             ++trit_index, ++written) {
            const std::int8_t trit = digit_to_trit(digits[trit_index]);
            destination[written] = trit * scale;
        }
    }
}

}  // namespace gguf
}  // namespace core
}  // namespace t81
