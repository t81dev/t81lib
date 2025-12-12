#include "t81/linalg/gemm_gpu.hpp"

#if T81LIB_USE_ROCM

#include <hip/hip_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace t81::linalg::detail {

namespace {

inline void hip_check(hipError_t result, const char *context) {
    if (result != hipSuccess) {
        throw std::runtime_error(
            std::string(context) + ": " + hipGetErrorString(result));
    }
}

struct HIPFloatBuffer {
    float *data = nullptr;

    explicit HIPFloatBuffer(std::size_t count) {
        hip_check(hipMalloc(&data, count * sizeof(float)), "hipMalloc");
    }

    HIPFloatBuffer(HIPFloatBuffer &&other) noexcept : data(other.data) {
        other.data = nullptr;
    }

    HIPFloatBuffer(const HIPFloatBuffer &) = delete;
    HIPFloatBuffer &operator=(const HIPFloatBuffer &) = delete;

    ~HIPFloatBuffer() {
        if (data) {
            hipFree(data);
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

bool rocm_available() noexcept {
    int device_count = 0;
    const hipError_t error = hipGetDeviceCount(&device_count);
    return error == hipSuccess && device_count > 0;
}

void rocm_where(std::span<const float> condition,
                std::span<const float> x,
                std::span<const float> y,
                std::span<float> out) {
    const std::size_t total = out.size();
    if (total == 0) {
        return;
    }
    HIPFloatBuffer device_condition(total);
    HIPFloatBuffer device_x(total);
    HIPFloatBuffer device_y(total);
    HIPFloatBuffer device_out(total);
    const std::size_t byte_count = total * sizeof(float);
    hip_check(hipMemcpy(device_condition.data,
                        condition.data(),
                        byte_count,
                        hipMemcpyHostToDevice),
              "hipMemcpy(condition)");
    hip_check(hipMemcpy(device_x.data,
                        x.data(),
                        byte_count,
                        hipMemcpyHostToDevice),
              "hipMemcpy(x)");
    hip_check(hipMemcpy(device_y.data,
                        y.data(),
                        byte_count,
                        hipMemcpyHostToDevice),
              "hipMemcpy(y)");

    constexpr std::size_t BlockSize = 256;
    const std::size_t grid = (total + BlockSize - 1) / BlockSize;
    hipLaunchKernelGGL(where_kernel,
                       dim3(grid),
                       dim3(BlockSize),
                       0,
                       0,
                       device_condition.data,
                       device_x.data,
                       device_y.data,
                       device_out.data,
                       total);
    hip_check(hipGetLastError(), "where_kernel");
    hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize");

    hip_check(hipMemcpy(out.data(),
                        device_out.data,
                        byte_count,
                        hipMemcpyDeviceToHost),
              "hipMemcpy(out)");
}

} // namespace t81::linalg::detail

#endif // T81LIB_USE_ROCM
