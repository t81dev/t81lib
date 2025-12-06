#pragma once

#include <array>
#include <cstdint>
#include <immintrin.h>

namespace t81::core::detail {

/// AVX-512 carrystream carrier map for trit-aligned trytes.
/// These helpers expose the per-lane carry-map composition so the scalar
/// version in `T81Limb` can reuse the same LUTs when we broaden lanes.
#ifdef __AVX512VL__

struct Avx512CarryMap {
    __m512i lanes;

    static constexpr int LANE_TRITS = 6; // 2 trytes per 64-bit lane
    static constexpr int LANES = 8;

    /// Broadcast a pair of trytes (6 trits) onto every lane.
    static inline __m512i broadcast_pair(int8_t tryte_lo, int8_t tryte_hi) noexcept {
        __m128i packed = _mm_set_epi8(0,0,0,0,0,0,0,0, tryte_hi, tryte_lo, 0,0,0,0,0,0);
        return _mm512_broadcast_i32x4(packed);
    }

    /// Compose two carry maps using the precomputed LUT indices.
    static inline __m512i compose(__m512i current, __m512i previous) noexcept {
        // Placeholder: actual implementation would gather from COMPOSITION_TABLE.
        return _mm512_add_epi8(current, previous);
    }
};

#endif // __AVX512VL__

} // namespace t81::core::detail
