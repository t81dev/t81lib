#include "t81/linalg/gemm_gpu.hpp"
#include <t81/tensor_metadata.hpp>

#if T81LIB_USE_CUDA

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace t81::linalg::detail {

namespace {

inline void cuda_check(cudaError_t result, const char *context) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(context) + ": " + cudaGetErrorString(result));
    }
}

inline const float *read_ptr(const TensorMetadata &meta) {
    if (meta.data_ptr == nullptr) {
        throw std::invalid_argument("tensor metadata lacks a data pointer");
    }
    if (!meta.dtype_is_float32()) {
        throw std::invalid_argument("CUDA kernels only support float32");
    }
    if (!meta.is_contiguous()) {
        throw std::invalid_argument("CUDA kernels require contiguous storage");
    }
    const auto byte_offset =
        static_cast<std::size_t>(std::max<int64_t>(meta.storage_offset, 0LL)) *
        sizeof(float);
    const auto *base = reinterpret_cast<const std::uint8_t *>(meta.data_ptr);
    return reinterpret_cast<const float *>(base + byte_offset);
}

inline float *write_ptr(TensorMetadata &meta) {
    if (meta.data_ptr == nullptr) {
        throw std::invalid_argument("tensor metadata lacks a data pointer");
    }
    if (!meta.dtype_is_float32()) {
        throw std::invalid_argument("CUDA kernels only support float32");
    }
    if (!meta.is_contiguous()) {
        throw std::invalid_argument("CUDA kernels require contiguous storage");
    }
    const auto byte_offset =
        static_cast<std::size_t>(std::max<int64_t>(meta.storage_offset, 0LL)) *
        sizeof(float);
    auto *base = reinterpret_cast<std::uint8_t *>(meta.data_ptr);
    return reinterpret_cast<float *>(base + byte_offset);
}

inline std::size_t numel(const TensorMetadata &meta) {
    return meta.numel();
}

constexpr std::size_t BlockSize = 256;

__global__ void where_kernel(const float *condition,
                             const float *x,
                             const float *y,
                             float *out,
                             std::size_t total) {
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + static_cast<std::size_t>(threadIdx.x);
    if (idx >= total) {
        return;
    }
    out[idx] = condition[idx] != 0.0f ? x[idx] : y[idx];
}

__global__ void clamp_kernel(const float *input,
                             float min_value,
                             float max_value,
                             float *out,
                             std::size_t total) {
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + static_cast<std::size_t>(threadIdx.x);
    if (idx >= total) {
        return;
    }
    const float value = input[idx];
    out[idx] = value < min_value ? min_value : (value > max_value ? max_value : value);
}

__global__ void lerp_kernel(const float *start,
                            const float *end,
                            const float *weight,
                            float *out,
                            std::size_t total) {
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + static_cast<std::size_t>(threadIdx.x);
    if (idx >= total) {
        return;
    }
    out[idx] = start[idx] + weight[idx] * (end[idx] - start[idx]);
}

inline void sync_if_needed(const TensorMetadata &meta) {
    if (meta.requires_sync) {
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }
}

} // namespace

void cuda_where(const TensorMetadata &condition,
                const TensorMetadata &x,
                const TensorMetadata &y,
                TensorMetadata &out) {
    const std::size_t total = numel(condition);
    if (total == 0) {
        return;
    }
    const auto *cond_ptr = read_ptr(condition);
    const auto *x_ptr = read_ptr(x);
    const auto *y_ptr = read_ptr(y);
    auto *out_ptr = write_ptr(out);
    const std::size_t grid = (total + BlockSize - 1) / BlockSize;
    where_kernel<<<grid, BlockSize>>>(cond_ptr, x_ptr, y_ptr, out_ptr, total);
    cuda_check(cudaGetLastError(), "where_kernel");
    sync_if_needed(out);
}

void cuda_clamp(const TensorMetadata &input,
                TensorMetadata &out,
                float min_value,
                float max_value) {
    const std::size_t total = numel(input);
    if (total == 0) {
        return;
    }
    const auto *in_ptr = read_ptr(input);
    auto *out_ptr = write_ptr(out);
    const std::size_t grid = (total + BlockSize - 1) / BlockSize;
    clamp_kernel<<<grid, BlockSize>>>(in_ptr, min_value, max_value, out_ptr, total);
    cuda_check(cudaGetLastError(), "clamp_kernel");
    sync_if_needed(out);
}

void cuda_lerp(const TensorMetadata &start,
               const TensorMetadata &end,
               const TensorMetadata &weight,
               TensorMetadata &out) {
    const std::size_t total = numel(out);
    if (total == 0) {
        return;
    }
    const auto *start_ptr = read_ptr(start);
    const auto *end_ptr = read_ptr(end);
    const auto *weight_ptr = read_ptr(weight);
    auto *out_ptr = write_ptr(out);
    const std::size_t grid = (total + BlockSize - 1) / BlockSize;
    lerp_kernel<<<grid, BlockSize>>>(start_ptr, end_ptr, weight_ptr, out_ptr, total);
    cuda_check(cudaGetLastError(), "lerp_kernel");
    sync_if_needed(out);
}

void cuda_addcmul(const TensorMetadata &input,
                  const TensorMetadata &tensor1,
                  const TensorMetadata &tensor2,
                  float value,
                  TensorMetadata &out) {
    const std::size_t total = numel(out);
    if (total == 0) {
        return;
    }
    const auto *input_ptr = read_ptr(input);
    const auto *t1_ptr = read_ptr(tensor1);
    const auto *t2_ptr = read_ptr(tensor2);
    auto *out_ptr = write_ptr(out);
    const std::size_t grid = (total + BlockSize - 1) / BlockSize;
    addcmul_kernel<<<grid, BlockSize>>>(input_ptr, t1_ptr, t2_ptr, value, out_ptr, total);
    cuda_check(cudaGetLastError(), "addcmul_kernel");
    sync_if_needed(out);
}

} // namespace t81::linalg::detail

#endif // T81LIB_USE_CUDA
