#pragma once

#include <cstddef>
#include <span>

#ifndef T81LIB_USE_CUDA
#define T81LIB_USE_CUDA 0
#endif

#ifndef T81LIB_USE_ROCM
#define T81LIB_USE_ROCM 0
#endif

namespace t81::linalg {

enum class Backend { Auto, CPU, CUDA, ROCm };

namespace detail {

inline constexpr bool backend_available(Backend backend) noexcept {
    switch (backend) {
        case Backend::Auto:
            return true;
        case Backend::CPU:
            return true;
    #if T81LIB_USE_CUDA
        case Backend::CUDA:
            return true;
    #else
        case Backend::CUDA:
            return false;
    #endif
    #if T81LIB_USE_ROCM
        case Backend::ROCm:
            return true;
    #else
        case Backend::ROCm:
            return false;
    #endif
    }
    return false;
}

Backend get_current_backend() noexcept;
#if T81LIB_USE_CUDA
bool cuda_available() noexcept;
#endif
#if T81LIB_USE_ROCM
bool rocm_available() noexcept;
#endif

#if T81LIB_USE_CUDA
void cuda_where(std::span<const float> condition,
                std::span<const float> x,
                std::span<const float> y,
                std::span<float> out);
#endif
#if T81LIB_USE_ROCM
void rocm_where(std::span<const float> condition,
                std::span<const float> x,
                std::span<const float> y,
                std::span<float> out);
#endif

void where(std::span<const float> condition,
           std::span<const float> x,
           std::span<const float> y,
           std::span<float> out,
           Backend backend = Backend::Auto);

void clamp(std::span<const float> x,
           float min_value,
           float max_value,
           std::span<float> out,
           Backend backend = Backend::Auto);

void lerp(std::span<const float> start,
          std::span<const float> end,
          std::span<const float> weight,
          std::span<float> out,
          Backend backend = Backend::Auto);

void addcmul(std::span<const float> input,
             std::span<const float> tensor1,
             std::span<const float> tensor2,
             float value,
             std::span<float> out,
             Backend backend = Backend::Auto);

} // namespace detail
} // namespace t81::linalg
