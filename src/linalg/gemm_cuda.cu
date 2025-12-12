#include "t81/linalg/gemm_gpu.hpp"

#if T81LIB_USE_CUDA

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace t81::linalg::detail {

namespace {

inline void cuda_check(cudaError_t result, const char *context) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(context) + ": " + cudaGetErrorString(result));
    }
}

struct CUDAFloatBuffer {
    float *data = nullptr;

    explicit CUDAFloatBuffer(std::size_t count) {
        cuda_check(cudaMalloc(&data, count * sizeof(float)), "cudaMalloc");
    }

    CUDAFloatBuffer(CUDAFloatBuffer &&other) noexcept : data(other.data) {
        other.data = nullptr;
    }

    CUDAFloatBuffer(const CUDAFloatBuffer &) = delete;
    CUDAFloatBuffer &operator=(const CUDAFloatBuffer &) = delete;

    ~CUDAFloatBuffer() {
        if (data) {
            cudaFree(data);
        }
    }
};

__global__ void where_kernel(const float *condition,
                             const float *x,
                             const float *y,
                             float *out,
                             std::size_t total) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                            static_cast<std::size_t>(threadIdx.x);
    if (idx >= total) {
        return;
    }
    out[idx] = condition[idx] != 0.0f ? x[idx] : y[idx];
}

} // namespace

bool cuda_available() noexcept {
    int device_count = 0;
    const cudaError_t error = cudaGetDeviceCount(&device_count);
    return error == cudaSuccess && device_count > 0;
}

void cuda_where(std::span<const float> condition,
                std::span<const float> x,
                std::span<const float> y,
                std::span<float> out) {
    const std::size_t total = out.size();
    if (total == 0) {
        return;
    }
    CUDAFloatBuffer device_condition(total);
    CUDAFloatBuffer device_x(total);
    CUDAFloatBuffer device_y(total);
    CUDAFloatBuffer device_out(total);
    const std::size_t byte_count = total * sizeof(float);
    cuda_check(cudaMemcpy(device_condition.data,
                          condition.data(),
                          byte_count,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(condition)");
    cuda_check(cudaMemcpy(device_x.data,
                          x.data(),
                          byte_count,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(x)");
    cuda_check(cudaMemcpy(device_y.data,
                          y.data(),
                          byte_count,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(y)");

    constexpr std::size_t BlockSize = 256;
    const std::size_t grid = (total + BlockSize - 1) / BlockSize;
    where_kernel<<<grid, BlockSize>>>(
        device_condition.data, device_x.data, device_y.data, device_out.data, total);
    cuda_check(cudaGetLastError(), "where_kernel");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    cuda_check(cudaMemcpy(out.data(),
                          device_out.data,
                          byte_count,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(out)");
}

} // namespace t81::linalg::detail

#endif // T81LIB_USE_CUDA
