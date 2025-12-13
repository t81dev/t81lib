#include "t81/linalg/gemm_gpu.hpp"
#include <t81/tensor_metadata.hpp>

#if T81LIB_USE_ROCM

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>
#include <t81/core/detail/lut.hpp>

namespace t81::linalg::detail {

namespace {

inline void hip_check(hipError_t result, const char *context) {
    if (result != hipSuccess) {
        throw std::runtime_error(
            std::string(context) + ": " + hipGetErrorString(result));
    }
}

inline const float *read_ptr(const TensorMetadata &meta) {
    if (meta.data_ptr == nullptr) {
        throw std::invalid_argument("tensor metadata lacks a data pointer");
    }
    if (!meta.dtype_is_float32()) {
        throw std::invalid_argument("ROCm kernels only support float32");
    }
    if (!meta.is_contiguous()) {
        throw std::invalid_argument("ROCm kernels require contiguous storage");
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
        throw std::invalid_argument("ROCm kernels only support float32");
    }
    if (!meta.is_contiguous()) {
        throw std::invalid_argument("ROCm kernels require contiguous storage");
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

__global__ void addcmul_kernel(const float *input,
                               const float *tensor1,
                               const float *tensor2,
                               float value,
                               float *out,
                               std::size_t total) {
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + static_cast<std::size_t>(threadIdx.x);
    if (idx >= total) {
        return;
    }
    out[idx] = input[idx] + value * tensor1[idx] * tensor2[idx];
}

inline void sync_if_needed(const TensorMetadata &meta) {
    if (meta.requires_sync) {
        hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize");
    }
}

constexpr auto build_tryte_contributions()
    -> std::array<std::array<double, core::detail::TRYTE_TO_TRITS.size()>,
                  core::limb::TRYTES> {
    std::array<std::array<double, core::detail::TRYTE_TO_TRITS.size()>,
               core::limb::TRYTES>
        table{};
    double weight = 1.0;
    for (std::size_t position = 0; position < core::limb::TRYTES; ++position) {
        for (std::size_t tryte = 0; tryte < core::detail::TRYTE_TO_TRITS.size(); ++tryte) {
            const auto triple = core::detail::TRYTE_TO_TRITS[tryte];
            table[position][tryte] =
                triple[0] * weight + triple[1] * weight * 3.0 + triple[2] * weight * 9.0;
        }
        weight *= 27.0;
    }
    return table;
}

static constexpr auto g_host_tryte_table = build_tryte_contributions();
__device__ __constant__ double g_device_tryte_table[core::limb::TRYTES][27];

inline std::vector<std::uint8_t> pack_limbs(std::span<const core::limb> limbs) {
    constexpr std::size_t TrytesPerLimb = core::limb::TRYTES;
    std::vector<std::uint8_t> buffer(limbs.size() * TrytesPerLimb);
    for (std::size_t index = 0; index < limbs.size(); ++index) {
        const auto values = limbs[index].to_trytes();
        std::memcpy(buffer.data() + index * TrytesPerLimb,
                    values.data(),
                    TrytesPerLimb);
    }
    return buffer;
}

struct HipBuffer {
    void *ptr = nullptr;
    HipBuffer() = default;
    explicit HipBuffer(void *raw) noexcept : ptr(raw) {}
    HipBuffer(HipBuffer &&other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;
    }
    HipBuffer &operator=(HipBuffer &&other) noexcept {
        if (this != &other) {
            if (ptr) {
                hipFree(ptr);
            }
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }
    ~HipBuffer() {
        if (ptr) {
            hipFree(ptr);
        }
    }
    HipBuffer(const HipBuffer &) = delete;
    HipBuffer &operator=(const HipBuffer &) = delete;
};

inline __device__ double decode_limb_value(const std::uint8_t *trytes) {
    double result = 0.0;
    for (int index = 0; index < core::limb::TRYTES; ++index) {
        const int tryte = trytes[index];
        result += g_device_tryte_table[index][tryte];
    }
    return result;
}

__global__ void gemm_kernel(const std::uint8_t *a_trytes,
                            const std::uint8_t *b_trytes,
                            float *out,
                            int M,
                            int N,
                            int K_limbs,
                            float alpha,
                            float beta) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) {
        return;
    }
    double acc = 0.0;
    for (int k = 0; k < K_limbs; ++k) {
        const std::uint8_t *a_ptr =
            a_trytes + (static_cast<std::size_t>(row) * K_limbs + k) * core::limb::TRYTES;
        const std::uint8_t *b_ptr =
            b_trytes + (static_cast<std::size_t>(k) * N + col) * core::limb::TRYTES;
        const double a_value = decode_limb_value(a_ptr);
        const double b_value = decode_limb_value(b_ptr);
        acc += a_value * b_value;
    }
    const std::size_t idx = static_cast<std::size_t>(row) * static_cast<std::size_t>(N) + col;
    const double previous = static_cast<double>(out[idx]);
    out[idx] = static_cast<float>(beta * previous + alpha * acc);
}

} // namespace

void rocm_where(const TensorMetadata &condition,
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
    hipLaunchKernelGGL(where_kernel,
                       dim3(grid),
                       dim3(BlockSize),
                       0,
                       0,
                       cond_ptr,
                       x_ptr,
                       y_ptr,
                       out_ptr,
                       total);
    hip_check(hipGetLastError(), "where_kernel");
    sync_if_needed(out);
}

void rocm_clamp(const TensorMetadata &input,
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
    hipLaunchKernelGGL(clamp_kernel,
                       dim3(grid),
                       dim3(BlockSize),
                       0,
                       0,
                       in_ptr,
                       min_value,
                       max_value,
                       out_ptr,
                       total);
    hip_check(hipGetLastError(), "clamp_kernel");
    sync_if_needed(out);
}

void rocm_lerp(const TensorMetadata &start,
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
    hipLaunchKernelGGL(lerp_kernel,
                       dim3(grid),
                       dim3(BlockSize),
                       0,
                       0,
                       start_ptr,
                       end_ptr,
                       weight_ptr,
                       out_ptr,
                       total);
    hip_check(hipGetLastError(), "lerp_kernel");
    sync_if_needed(out);
}

void rocm_addcmul(const TensorMetadata &input,
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
    hipLaunchKernelGGL(addcmul_kernel,
                       dim3(grid),
                       dim3(BlockSize),
                       0,
                       0,
                       input_ptr,
                       t1_ptr,
                       t2_ptr,
                       value,
                       out_ptr,
                       total);
    hip_check(hipGetLastError(), "addcmul_kernel");
    sync_if_needed(out);
}

void rocm_gemm_ternary(std::span<const core::limb> A,
                       std::span<const core::limb> B,
                       std::span<float> C,
                       int M,
                       int N,
                       int K,
                       float alpha,
                       float beta) {
    const int K_limbs = K / core::limb::TRITS;
    if (M == 0 || N == 0 || K_limbs == 0) {
        return;
    }
    const auto packed_A = pack_limbs(A);
    const auto packed_B = pack_limbs(B);

    HipBuffer a_buffer;
    HipBuffer b_buffer;
    HipBuffer c_buffer;

    hip_check(hipMalloc(&a_buffer.ptr, packed_A.size()));
    hip_check(hipMalloc(&b_buffer.ptr, packed_B.size()));
    hip_check(hipMalloc(&c_buffer.ptr, static_cast<std::size_t>(C.size()) * sizeof(float)));

    hip_check(hipMemcpy(a_buffer.ptr,
                        packed_A.data(),
                        packed_A.size(),
                        hipMemcpyHostToDevice));
    hip_check(hipMemcpy(b_buffer.ptr,
                        packed_B.data(),
                        packed_B.size(),
                        hipMemcpyHostToDevice));
    hip_check(hipMemcpy(c_buffer.ptr,
                        C.data(),
                        static_cast<std::size_t>(C.size()) * sizeof(float),
                        hipMemcpyHostToDevice));

    hip_check(hipMemcpyToSymbol(HIP_SYMBOL(g_device_tryte_table),
                               g_host_tryte_table.data(),
                               sizeof(g_host_tryte_table)));

    constexpr dim3 block(16, 16);
    const dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    hipLaunchKernelGGL(gemm_kernel,
                       grid,
                       block,
                       0,
                       0,
                       static_cast<const std::uint8_t *>(a_buffer.ptr),
                       static_cast<const std::uint8_t *>(b_buffer.ptr),
                       static_cast<float *>(c_buffer.ptr),
                       M,
                       N,
                       K_limbs,
                       alpha,
                       beta);
    hip_check(hipGetLastError());

    hip_check(hipMemcpy(C.data(),
                        c_buffer.ptr,
                        static_cast<std::size_t>(C.size()) * sizeof(float),
                        hipMemcpyDeviceToHost));
}

} // namespace t81::linalg::detail

#endif // T81LIB_USE_ROCM
