#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#ifndef T81LIB_USE_METAL
#define T81LIB_USE_METAL 0
#endif

namespace t81::linalg::detail {

#if T81LIB_USE_METAL
void metal_quantize_to_trits(std::span<const float> src,
                             std::span<std::int8_t> dst,
                             float threshold);

void metal_pack_dense_matrix(std::span<const float> src,
                             std::span<std::uint8_t> dst,
                             int rows,
                             int cols,
                             float threshold);
#endif

} // namespace t81::linalg::detail
