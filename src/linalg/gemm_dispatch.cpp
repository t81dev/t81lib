#include "t81/linalg/gemm_gpu.hpp"
#include <t81/tensor_metadata.hpp>
#include <t81/core/limb.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#if T81LIB_USE_METAL
#if defined(__APPLE__) && defined(__has_include)
#if __has_include(<Metal/Metal.h>)
#include <Metal/Metal.h>
#define T81LIB_METAL_HEADER_AVAILABLE 1
#else
#define T81LIB_METAL_HEADER_AVAILABLE 0
#endif
#else
#define T81LIB_METAL_HEADER_AVAILABLE 0
#endif
#endif

namespace t81::linalg::detail {

namespace {

#if T81LIB_USE_METAL
bool metal_available() noexcept {
#if defined(T81LIB_METAL_HEADER_AVAILABLE)
    return T81LIB_METAL_HEADER_AVAILABLE != 0;
#else
    return false;
#endif
}
#undef T81LIB_METAL_HEADER_AVAILABLE
#endif

inline std::size_t compute_numel(const TensorMetadata &meta) {
    return meta.numel();
}

inline void ensure_float32_scalar(const TensorMetadata &meta) {
    if (!meta.dtype_is_float32()) {
        throw std::invalid_argument("tensor metadata must describe float32 data");
    }
    if (meta.data_ptr == nullptr) {
        throw std::invalid_argument("tensor metadata must provide a valid data pointer");
    }
}

inline const float *as_const_float_ptr(const TensorMetadata &meta) {
    ensure_float32_scalar(meta);
    const auto *base = reinterpret_cast<const std::uint8_t *>(meta.data_ptr);
    const std::size_t offset =
        static_cast<std::size_t>(std::max<int64_t>(meta.storage_offset, 0LL)) *
        sizeof(float);
    return reinterpret_cast<const float *>(base + offset);
}

inline float *as_float_ptr(TensorMetadata &meta) {
    ensure_float32_scalar(meta);
    auto *base = reinterpret_cast<std::uint8_t *>(meta.data_ptr);
    const std::size_t offset =
        static_cast<std::size_t>(std::max<int64_t>(meta.storage_offset, 0LL)) *
        sizeof(float);
    return reinterpret_cast<float *>(base + offset);
}

inline std::span<const float> metadata_to_const_span(const TensorMetadata &meta) {
    if (!meta.is_contiguous()) {
        throw std::invalid_argument("tensor metadata must describe contiguous storage");
    }
    const auto *ptr = as_const_float_ptr(meta);
    return {ptr, compute_numel(meta)};
}

inline std::span<float> metadata_to_span(TensorMetadata &meta) {
    if (!meta.is_contiguous()) {
        throw std::invalid_argument("tensor metadata must describe contiguous storage");
    }
    auto *ptr = as_float_ptr(meta);
    return {ptr, compute_numel(meta)};
}

inline void ensure_ternary_limb(const TensorMetadata &meta) {
    if (!meta.dtype_is_ternary_limb()) {
        throw std::invalid_argument("tensor metadata must describe ternary limbs");
    }
    if (meta.data_ptr == nullptr) {
        throw std::invalid_argument("tensor metadata must provide a valid data pointer");
    }
}

inline std::span<const core::limb> metadata_to_limb_span(const TensorMetadata &meta) {
    if (!meta.is_contiguous()) {
        throw std::invalid_argument("tensor metadata must describe contiguous storage");
    }
    ensure_ternary_limb(meta);
    const auto *base = reinterpret_cast<const std::uint8_t *>(meta.data_ptr);
    const std::size_t offset =
        static_cast<std::size_t>(std::max<int64_t>(meta.storage_offset, 0LL)) *
        core::limb::BYTES;
    const auto *limb_ptr = reinterpret_cast<const core::limb *>(base + offset);
    return {limb_ptr, compute_numel(meta)};
}

inline int checked_dim(int64_t value) {
    if (value < 0 || value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("tensor dimension exceeds supported range");
    }
    return static_cast<int>(value);
}

inline std::pair<int, int> matrix_shape(const TensorMetadata &meta) {
    if (meta.sizes.size() != 2) {
        throw std::invalid_argument("tensor metadata must describe a rank-2 matrix");
    }
    const int rows = checked_dim(meta.sizes[0]);
    const int cols = checked_dim(meta.sizes[1]);
    return {rows, cols};
}

inline Backend effective_backend(Backend requested) {
#if T81LIB_USE_CUDA
    const Backend backed = requested == Backend::Auto ? get_current_backend() : requested;
    if (backed == Backend::CUDA && backend_available(Backend::CUDA)) {
        return Backend::CUDA;
    }
#else
    const Backend backed = requested == Backend::Auto ? get_current_backend() : requested;
#endif
#if T81LIB_USE_ROCM
    if ((requested == Backend::ROCm || backed == Backend::ROCm) &&
        backend_available(Backend::ROCm)) {
        return Backend::ROCm;
    }
#endif
#if T81LIB_USE_METAL
    if ((requested == Backend::Metal || backed == Backend::Metal) &&
        backend_available(Backend::Metal)) {
        return Backend::Metal;
    }
#endif
    return Backend::CPU;
}

inline void cpu_where(const TensorMetadata &condition,
                      const TensorMetadata &x,
                      const TensorMetadata &y,
                      TensorMetadata &out) {
    const auto cond_span = metadata_to_const_span(condition);
    const auto x_span = metadata_to_const_span(x);
    const auto y_span = metadata_to_const_span(y);
    auto out_span = metadata_to_span(out);
    if (cond_span.size() != out_span.size() || x_span.size() != out_span.size() ||
        y_span.size() != out_span.size()) {
        throw std::invalid_argument("input shapes must match on the CPU path");
    }
    for (std::size_t idx = 0; idx < out_span.size(); ++idx) {
        out_span[idx] = cond_span[idx] != 0.0f ? x_span[idx] : y_span[idx];
    }
}

inline void cpu_clamp(const TensorMetadata &input,
                      TensorMetadata &out,
                      float min_value,
                      float max_value) {
    ensure_float32_scalar(input);
    ensure_float32_scalar(out);
    auto input_span = metadata_to_const_span(input);
    auto out_span = metadata_to_span(out);
    if (input_span.size() != out_span.size()) {
        throw std::invalid_argument("input/output lengths must match for clamp");
    }
    for (std::size_t idx = 0; idx < out_span.size(); ++idx) {
        const float value = input_span[idx];
        out_span[idx] = value < min_value ? min_value : (value > max_value ? max_value : value);
    }
}

inline void cpu_lerp(const TensorMetadata &start,
                     const TensorMetadata &end,
                     const TensorMetadata &weight,
                     TensorMetadata &out) {
    const auto start_span = metadata_to_const_span(start);
    const auto end_span = metadata_to_const_span(end);
    const auto weight_span = metadata_to_const_span(weight);
    auto out_span = metadata_to_span(out);
    if (start_span.size() != out_span.size() || end_span.size() != out_span.size() ||
        weight_span.size() != out_span.size()) {
        throw std::invalid_argument("inputs for lerp must match output length");
    }
    for (std::size_t idx = 0; idx < out_span.size(); ++idx) {
        out_span[idx] =
            start_span[idx] + weight_span[idx] * (end_span[idx] - start_span[idx]);
    }
}

inline void cpu_addcmul(const TensorMetadata &input,
                        const TensorMetadata &tensor1,
                        const TensorMetadata &tensor2,
                        float value,
                        TensorMetadata &out) {
    const auto input_span = metadata_to_const_span(input);
    const auto tensor1_span = metadata_to_const_span(tensor1);
    const auto tensor2_span = metadata_to_const_span(tensor2);
    auto out_span = metadata_to_span(out);
    if (input_span.size() != out_span.size() || tensor1_span.size() != out_span.size() ||
        tensor2_span.size() != out_span.size()) {
        throw std::invalid_argument("inputs for addcmul must align with output");
    }
    for (std::size_t idx = 0; idx < out_span.size(); ++idx) {
        out_span[idx] = input_span[idx] + value * tensor1_span[idx] * tensor2_span[idx];
    }
}

} // namespace

Backend get_current_backend() noexcept {
#if T81LIB_USE_CUDA
    if (cuda_available()) {
        return Backend::CUDA;
    }
#endif
#if T81LIB_USE_ROCM
    if (rocm_available()) {
        return Backend::ROCm;
    }
#endif
#if T81LIB_USE_METAL
    if (metal_available()) {
        return Backend::Metal;
    }
#endif
    return Backend::CPU;
}

void where(const TensorMetadata &condition,
           const TensorMetadata &x,
           const TensorMetadata &y,
           TensorMetadata &out,
           Backend backend) {
    const Backend target = effective_backend(backend);
#if T81LIB_USE_CUDA
    if (target == Backend::CUDA) {
        cuda_where(condition, x, y, out);
        return;
    }
#endif
#if T81LIB_USE_ROCM
    if (target == Backend::ROCm) {
        rocm_where(condition, x, y, out);
        return;
    }
#endif

    cpu_where(condition, x, y, out);
}

void clamp(const TensorMetadata &input,
           TensorMetadata &out,
           float min_value,
           float max_value,
           Backend backend) {
    const Backend target = effective_backend(backend);
#if T81LIB_USE_CUDA
    if (target == Backend::CUDA) {
        cuda_clamp(input, out, min_value, max_value);
        return;
    }
#endif
#if T81LIB_USE_ROCM
    if (target == Backend::ROCm) {
        rocm_clamp(input, out, min_value, max_value);
        return;
    }
#endif
    cpu_clamp(input, out, min_value, max_value);
}

void lerp(const TensorMetadata &start,
          const TensorMetadata &end,
          const TensorMetadata &weight,
          TensorMetadata &out,
          Backend backend) {
    const Backend target = effective_backend(backend);
#if T81LIB_USE_CUDA
    if (target == Backend::CUDA) {
        cuda_lerp(start, end, weight, out);
        return;
    }
#endif
#if T81LIB_USE_ROCM
    if (target == Backend::ROCm) {
        rocm_lerp(start, end, weight, out);
        return;
    }
#endif
    cpu_lerp(start, end, weight, out);
}

void addcmul(const TensorMetadata &input,
             const TensorMetadata &tensor1,
             const TensorMetadata &tensor2,
             float value,
             TensorMetadata &out,
             Backend backend) {
    const Backend target = effective_backend(backend);
#if T81LIB_USE_CUDA
    if (target == Backend::CUDA) {
        cuda_addcmul(input, tensor1, tensor2, value, out);
        return;
    }
#endif
#if T81LIB_USE_ROCM
    if (target == Backend::ROCm) {
        rocm_addcmul(input, tensor1, tensor2, value, out);
        return;
    }
#endif
    cpu_addcmul(input, tensor1, tensor2, value, out);
}

void gemm_ternary(const TensorMetadata &A,
                  const TensorMetadata &B,
                  TensorMetadata &C,
                  float alpha,
                  float beta,
                  Backend backend) {
    const auto [M, K_limbs] = matrix_shape(A);
    const auto [b_rows, N] = matrix_shape(B);
    if (b_rows != K_limbs) {
        throw std::invalid_argument("B shape must match the K/48 dimension of A");
    }
    const auto [c_rows, c_cols] = matrix_shape(C);
    if (c_rows != M || c_cols != N) {
        throw std::invalid_argument("C shape must match the output dimensions");
    }
    const auto a_span = metadata_to_limb_span(A);
    const auto b_span = metadata_to_limb_span(B);
    auto c_span = metadata_to_span(C);
    gemm_ternary_dispatch(a_span, b_span, c_span, M, N, K_limbs * core::limb::TRITS, alpha, beta, backend);
}

} // namespace t81::linalg::detail
