#pragma once

#include <array>
#include <optional>

#include <t81/core/detail/addition.hpp>
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace t81::core::detail {

inline void compute_prefix(std::array<std::uint32_t, limb::TRYTES>& packed) {
    for (int distance = 1; distance < limb::TRYTES; distance <<= 1) {
        for (int index = distance; index < limb::TRYTES; ++index) {
            packed[index] = compose_packed_maps(packed[index - distance], packed[index]);
        }
    }
}

inline bool finalize_result(const std::array<std::uint32_t, limb::TRYTES>& packed,
                            limb& result) {
    std::array<limb::tryte_t, limb::TRYTES> output{};
    int carry = 0;
    for (std::size_t index = 0; index < limb::TRYTES; ++index) {
        const std::uint32_t outcome = select_packed_outcome(packed[index], carry);
        output[index] = static_cast<limb::tryte_t>(outcome & 0x3FU);
        carry = decode_carry(outcome);
    }
    if (carry != 0) {
        return false;
    }
    result = limb::from_bytes(output);
    return true;
}

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
inline bool gather_packed_additions_neon(const limb& lhs,
                                         const limb& rhs,
                                         std::array<std::uint32_t, limb::TRYTES>& packed) {
    const auto lhs_bytes = lhs.to_bytes();
    const auto rhs_bytes = rhs.to_bytes();
    std::array<int, limb::TRYTES> chunk_ids{};
    std::array<std::uint8_t, limb::TRYTES> lane_offsets{};
    for (int index = 0; index < limb::TRYTES; ++index) {
        const int entry = static_cast<int>(lhs_bytes[index] * 27 + rhs_bytes[index]);
        chunk_ids[index] = entry / PACKED_CHUNK_SIZE;
        lane_offsets[index] = static_cast<std::uint8_t>(entry % PACKED_CHUNK_SIZE);
    }

    for (int chunk = 0; chunk < PACKED_CHUNK_COUNT; ++chunk) {
        alignas(16) std::uint8_t offsets[limb::TRYTES]{};
        std::uint64_t mask = 0;
        for (int lane = 0; lane < limb::TRYTES; ++lane) {
            if (chunk_ids[lane] == chunk) {
                offsets[lane] = lane_offsets[lane];
                mask |= (std::uint64_t{1} << lane);
            }
        }
        if (mask == 0) {
            continue;
        }
        const PackedChunk& chunk_data = PACKED_CHUNKS[chunk];
        const uint8x16_t tbl0 = vld1q_u8(chunk_data.bytes[0].data());
        const uint8x16_t tbl1 = vld1q_u8(chunk_data.bytes[1].data());
        const uint8x16_t tbl2 = vld1q_u8(chunk_data.bytes[2].data());
        const uint8x16_t tbl3 = vld1q_u8(chunk_data.bytes[3].data());

        alignas(16) std::uint8_t out0[limb::TRYTES];
        alignas(16) std::uint8_t out1[limb::TRYTES];
        alignas(16) std::uint8_t out2[limb::TRYTES];
        alignas(16) std::uint8_t out3[limb::TRYTES];
        for (int group = 0; group < 3; ++group) {
            const int base = group * 16;
            const uint8x16_t offset_vec = vld1q_u8(offsets + base);
            const uint8x16_t b0 = vqtbl1q_u8(tbl0, offset_vec);
            const uint8x16_t b1 = vqtbl1q_u8(tbl1, offset_vec);
            const uint8x16_t b2 = vqtbl1q_u8(tbl2, offset_vec);
            const uint8x16_t b3 = vqtbl1q_u8(tbl3, offset_vec);
            vst1q_u8(out0 + base, b0);
            vst1q_u8(out1 + base, b1);
            vst1q_u8(out2 + base, b2);
            vst1q_u8(out3 + base, b3);
        }

        for (int lane = 0; lane < limb::TRYTES; ++lane) {
            if (mask & (std::uint64_t{1} << lane)) {
                packed[lane] = static_cast<std::uint32_t>(out0[lane]) |
                               (static_cast<std::uint32_t>(out1[lane]) << 8) |
                               (static_cast<std::uint32_t>(out2[lane]) << 16) |
                               (static_cast<std::uint32_t>(out3[lane]) << 24);
            }
        }
    }
    return true;
}
#endif

#if defined(__AVX512F__)
inline bool gather_packed_additions_avx512(const limb& lhs,
                                            const limb& rhs,
                                            std::array<std::uint32_t, limb::TRYTES>& packed) {
    const auto lhs_bytes = lhs.to_bytes();
    const auto rhs_bytes = rhs.to_bytes();
    alignas(64) int indices[limb::TRYTES];
    for (int index = 0; index < limb::TRYTES; ++index) {
        indices[index] = static_cast<int>(lhs_bytes[index] * 27 + rhs_bytes[index]);
    }
    const int* base_ptr = reinterpret_cast<const int*>(PACKED_ADD_MAP.data());
    for (int offset = 0; offset < limb::TRYTES; offset += 16) {
        const __m512i index_vec =
            _mm512_loadu_si512(reinterpret_cast<const __m512i*>(indices + offset));
        const __m512i gather =
            _mm512_i32gather_epi32(index_vec, base_ptr, 4);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(packed.data() + offset), gather);
    }
    return true;
}
#endif

#if defined(__AVX2__)
inline bool gather_packed_additions_avx2(const limb& lhs,
                                         const limb& rhs,
                                         std::array<std::uint32_t, limb::TRYTES>& packed) {
    const auto lhs_bytes = lhs.to_bytes();
    const auto rhs_bytes = rhs.to_bytes();
    alignas(32) int indices[limb::TRYTES];
    for (int index = 0; index < limb::TRYTES; ++index) {
        indices[index] = static_cast<int>(lhs_bytes[index] * 27 + rhs_bytes[index]);
    }
    const int* base_ptr = reinterpret_cast<const int*>(PACKED_ADD_MAP.data());
    __m256i gather_lo = _mm256_i32gather_epi32(
        base_ptr, _mm256_load_si256(reinterpret_cast<const __m256i*>(indices)), 4);
    __m256i gather_hi = _mm256_i32gather_epi32(
        base_ptr, _mm256_load_si256(reinterpret_cast<const __m256i*>(indices + 8)), 4);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(packed.data()), gather_lo);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(packed.data() + 8), gather_hi);
    return true;
}
#endif

inline bool add_trytes_avx2(const limb& lhs, const limb& rhs, limb& result) {
#if defined(__AVX2__)
    std::array<std::uint32_t, limb::TRYTES> packed{};
    if (!gather_packed_additions_avx2(lhs, rhs, packed)) {
        return false;
    }
    compute_prefix(packed);
    return finalize_result(packed, result);
#else
    return false;
#endif
}

inline bool add_trytes_avx512(const limb& lhs, const limb& rhs, limb& result) {
#if defined(__AVX512F__)
    std::array<std::uint32_t, limb::TRYTES> packed{};
    if (!gather_packed_additions_avx512(lhs, rhs, packed)) {
        return false;
    }
    compute_prefix(packed);
    return finalize_result(packed, result);
#else
    return false;
#endif
}

inline bool add_trytes_neon(const limb& lhs, const limb& rhs, limb& result) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    std::array<std::uint32_t, limb::TRYTES> packed{};
    if (!gather_packed_additions_neon(lhs, rhs, packed)) {
        return false;
    }
    compute_prefix(packed);
    return finalize_result(packed, result);
#else
    return false;
#endif
}
inline void normalize_accumulator(std::array<int, limb::TRITS * 2>& accum) {
    for (int pass = 0; pass < 6; ++pass) {
        for (std::size_t k = 0; k < accum.size() - 1; ++k) {
            const auto [digit, carry] = balanced_digit_and_carry(accum[k]);
            accum[k] = digit;
            accum[k + 1] += carry;
        }
    }
    for (std::size_t k = accum.size() - 1; k > 0; --k) {
        const auto [digit, carry] = balanced_digit_and_carry(accum[k]);
        accum[k] = digit;
        accum[k - 1] += carry;
    }
    accum[0] = balanced_digit_and_carry(accum[0]).first;
}

inline std::pair<limb, limb> pair_from_accumulator(
    const std::array<int, limb::TRITS * 2>& accum) {
    std::array<std::int8_t, limb::TRITS> low_trits{};
    std::array<std::int8_t, limb::TRITS> high_trits{};
    for (int index = 0; index < limb::TRITS; ++index) {
        low_trits[index] = static_cast<std::int8_t>(accum[index]);
        high_trits[index] = static_cast<std::int8_t>(accum[index + limb::TRITS]);
    }
    return {limb::from_trits(low_trits), limb::from_trits(high_trits)};
}

inline std::pair<limb, limb> mul_wide_scalar(const limb& lhs, const limb& rhs) {
    std::array<int, limb::TRITS * 2> accum{};
    const auto left = lhs.to_trits();
    const auto right = rhs.to_trits();
    for (int i = 0; i < limb::TRITS; ++i) {
        for (int j = 0; j < limb::TRITS; ++j) {
            accum[i + j] += left[i] * right[j];
        }
    }
    normalize_accumulator(accum);
    return pair_from_accumulator(accum);
}

#if defined(__AVX2__)
inline void accumulate_trits_avx2(const std::array<std::int8_t, limb::TRITS>& left_trits,
                                  const std::array<std::int8_t, limb::TRITS>& right_trits,
                                  std::array<int, limb::TRITS * 2>& accum) {
    alignas(32) std::array<int32_t, limb::TRITS> left_values{};
    alignas(32) std::array<int32_t, limb::TRITS> right_values{};
    for (int index = 0; index < limb::TRITS; ++index) {
        left_values[index] = left_trits[index];
        right_values[index] = right_trits[index];
    }
    for (int j = 0; j < limb::TRITS; ++j) {
        const __m256i right_vec = _mm256_set1_epi32(right_values[j]);
        for (int i = 0; i < limb::TRITS; i += 8) {
            const __m256i left_vec =
                _mm256_load_si256(reinterpret_cast<const __m256i*>(left_values.data() + i));
            const __m256i product = _mm256_mullo_epi32(left_vec, right_vec);
            alignas(32) int32_t buffer[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(buffer), product);
            for (int offset = 0; offset < 8; ++offset) {
                accum[i + offset + j] += buffer[offset];
            }
        }
    }
}
#endif

#if defined(__AVX512F__)
inline void accumulate_trits_avx512(const std::array<std::int8_t, limb::TRITS>& left_trits,
                                    const std::array<std::int8_t, limb::TRITS>& right_trits,
                                    std::array<int, limb::TRITS * 2>& accum) {
    alignas(64) std::array<int32_t, limb::TRITS> left_values{};
    alignas(64) std::array<int32_t, limb::TRITS> right_values{};
    for (int index = 0; index < limb::TRITS; ++index) {
        left_values[index] = left_trits[index];
        right_values[index] = right_trits[index];
    }
    for (int j = 0; j < limb::TRITS; ++j) {
        const __m512i right_vec = _mm512_set1_epi32(right_values[j]);
        for (int i = 0; i < limb::TRITS; i += 16) {
            const __m512i left_vec =
                _mm512_loadu_si512(reinterpret_cast<const __m512i*>(left_values.data() + i));
            const __m512i product = _mm512_mullo_epi32(left_vec, right_vec);
            alignas(64) int32_t buffer[16];
            _mm512_store_epi32(reinterpret_cast<__m512i*>(buffer), product);
            for (int offset = 0; offset < 16; ++offset) {
                accum[i + offset + j] += buffer[offset];
            }
        }
    }
}
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
inline void accumulate_trits_neon(const std::array<std::int8_t, limb::TRITS>& left_trits,
                                  const std::array<std::int8_t, limb::TRITS>& right_trits,
                                  std::array<int, limb::TRITS * 2>& accum) {
    alignas(16) std::array<int32_t, limb::TRITS> left_values{};
    alignas(16) std::array<int32_t, limb::TRITS> right_values{};
    for (int index = 0; index < limb::TRITS; ++index) {
        left_values[index] = left_trits[index];
        right_values[index] = right_trits[index];
    }
    for (int j = 0; j < limb::TRITS; ++j) {
        const int32x4_t right_vec = vdupq_n_s32(right_values[j]);
        for (int i = 0; i < limb::TRITS; i += 4) {
            const int32x4_t left_vec = vld1q_s32(left_values.data() + i);
            const int32x4_t product = vmulq_s32(left_vec, right_vec);
            alignas(16) int32_t buffer[4];
            vst1q_s32(buffer, product);
            for (int offset = 0; offset < 4; ++offset) {
                accum[i + offset + j] += buffer[offset];
            }
        }
    }
}
#endif

inline std::optional<std::pair<limb, limb>> mul_wide_avx2(const limb& lhs, const limb& rhs) {
#if defined(__AVX2__)
    std::array<int, limb::TRITS * 2> accum{};
    accumulate_trits_avx2(lhs.to_trits(), rhs.to_trits(), accum);
    normalize_accumulator(accum);
    return pair_from_accumulator(accum);
#else
    (void)lhs;
    (void)rhs;
    return std::nullopt;
#endif
}

#if defined(__AVX512F__)
inline std::optional<std::pair<limb, limb>> mul_wide_avx512(const limb& lhs,
                                                            const limb& rhs) {
    std::array<int, limb::TRITS * 2> accum{};
    accumulate_trits_avx512(lhs.to_trits(), rhs.to_trits(), accum);
    normalize_accumulator(accum);
    return pair_from_accumulator(accum);
}
#else
inline std::optional<std::pair<limb, limb>> mul_wide_avx512(const limb&,
                                                            const limb&) {
    return std::nullopt;
}
#endif

inline std::optional<std::pair<limb, limb>> mul_wide_neon(const limb& lhs, const limb& rhs) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    std::array<int, limb::TRITS * 2> accum{};
    accumulate_trits_neon(lhs.to_trits(), rhs.to_trits(), accum);
    normalize_accumulator(accum);
    return pair_from_accumulator(accum);
#else
    (void)lhs;
    (void)rhs;
    return std::nullopt;
#endif
}

} // namespace t81::core::detail
