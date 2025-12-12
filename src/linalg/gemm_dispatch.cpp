#include "t81/linalg/gemm_gpu.hpp"

#include <cstddef>
#include <stdexcept>

namespace t81::linalg::detail {

namespace {

inline void ensure_matching_size(std::span<const float> left,
                                 std::span<const float> right,
                                 const char *message) {
    if (left.size() != right.size()) {
        throw std::invalid_argument(message);
    }
}

} // namespace

Backend get_current_backend() noexcept {
    // Inspect the compiled backends and runtime availability in priority order.
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
    return Backend::CPU;
}

void where(std::span<const float> condition,
           std::span<const float> x,
           std::span<const float> y,
           std::span<float> out,
           Backend backend) {
    const std::size_t total = out.size();
    ensure_matching_size(condition, out, "condition must align with output length");
    ensure_matching_size(x, out, "x must align with output length");
    ensure_matching_size(y, out, "y must align with output length");

    if (total == 0) {
        return;
    }

#if T81LIB_USE_CUDA
    const Backend effective_backend =
        backend == Backend::Auto ? get_current_backend() : backend;
    if (effective_backend == Backend::CUDA && backend_available(Backend::CUDA)) {
        cuda_where(condition, x, y, out);
        return;
    }
#else
    const Backend effective_backend =
        backend == Backend::Auto ? get_current_backend() : backend;
#endif
#if T81LIB_USE_ROCM
    if (effective_backend == Backend::ROCm && backend_available(Backend::ROCm)) {
        rocm_where(condition, x, y, out);
        return;
    }
#endif

    // Fall back to CPU implementation.
    for (std::size_t index = 0; index < total; ++index) {
        out[index] = condition[index] != 0.0f ? x[index] : y[index];
    }
}

void clamp(std::span<const float> x,
           float min_value,
           float max_value,
           std::span<float> out,
           Backend backend) {
    if (min_value > max_value) {
        throw std::invalid_argument("min_value must be less than or equal to max_value");
    }
    ensure_matching_size(x, out, "input must align with output length");

    for (std::size_t index = 0; index < out.size(); ++index) {
        const float value = x[index];
        out[index] = value < min_value ? min_value : (value > max_value ? max_value : value);
    }
}

void lerp(std::span<const float> start,
          std::span<const float> end,
          std::span<const float> weight,
          std::span<float> out,
          Backend backend) {
    ensure_matching_size(start, out, "start must align with output length");
    ensure_matching_size(end, out, "end must align with output length");
    ensure_matching_size(weight, out, "weight must align with output length");

    for (std::size_t index = 0; index < out.size(); ++index) {
        out[index] =
            start[index] + weight[index] * (end[index] - start[index]);
    }
}

void addcmul(std::span<const float> input,
             std::span<const float> tensor1,
             std::span<const float> tensor2,
             float value,
             std::span<float> out,
             Backend backend) {
    ensure_matching_size(input, out, "input must align with output length");
    ensure_matching_size(tensor1, out, "tensor1 must align with output length");
    ensure_matching_size(tensor2, out, "tensor2 must align with output length");

    for (std::size_t index = 0; index < out.size(); ++index) {
        out[index] = input[index] + value * tensor1[index] * tensor2[index];
    }
}

} // namespace t81::linalg::detail
