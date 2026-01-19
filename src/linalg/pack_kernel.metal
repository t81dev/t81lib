#include <metal_stdlib>
using namespace metal;

struct PackParams {
    uint rows;
    uint cols;
    uint limbs_per_row;
    uint trits_per_limb;
    uint limb_bytes;
    float threshold;
};

struct QuantParams {
    uint count;
    float threshold;
};

static inline int quantize_trit(float value, float threshold) {
    float clamped = clamp(value, -1.0f, 1.0f);
    if (clamped >= threshold) {
        return 1;
    }
    if (clamped <= -threshold) {
        return -1;
    }
    return 0;
}

kernel void quantize_trits_kernel(device const float *src [[buffer(0)]],
                                  device char *dst [[buffer(1)]],
                                  constant QuantParams &params [[buffer(2)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= params.count) {
        return;
    }
    const int trit = quantize_trit(src[gid], params.threshold);
    dst[gid] = static_cast<char>(trit);
}

kernel void pack_dense_matrix_kernel(device const float *src [[buffer(0)]],
                                     device uchar *dst [[buffer(1)]],
                                     constant PackParams &params [[buffer(2)]],
                                     uint gid [[thread_position_in_grid]]) {
    const uint total_limbs = params.rows * params.limbs_per_row;
    if (gid >= total_limbs) {
        return;
    }

    const uint row = gid / params.limbs_per_row;
    const uint limb = gid % params.limbs_per_row;
    const uint base_col = limb * params.trits_per_limb;
    const uint out_offset = (row * params.limbs_per_row + limb) * params.limb_bytes;

    for (uint tryte_idx = 0; tryte_idx < params.limb_bytes; ++tryte_idx) {
        const uint trit_base = tryte_idx * 3u;
        int t0 = 0;
        int t1 = 0;
        int t2 = 0;

        uint col = base_col + trit_base;
        if (col < params.cols) {
            t0 = quantize_trit(src[row * params.cols + col], params.threshold);
        }
        col = base_col + trit_base + 1u;
        if (col < params.cols) {
            t1 = quantize_trit(src[row * params.cols + col], params.threshold);
        }
        col = base_col + trit_base + 2u;
        if (col < params.cols) {
            t2 = quantize_trit(src[row * params.cols + col], params.threshold);
        }

        const int tryte = t0 + 3 * t1 + 9 * t2 + 13;
        dst[out_offset + tryte_idx] = static_cast<uchar>(tryte);
    }
}
