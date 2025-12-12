#pragma once

#include <cstddef>
#include <span>
#include <cstdint>

#include <t81/tensor_metadata.hpp>

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
void cuda_where(const TensorMetadata &condition,
                const TensorMetadata &x,
                const TensorMetadata &y,
                TensorMetadata &out);
#endif
#if T81LIB_USE_ROCM
void rocm_where(const TensorMetadata &condition,
                const TensorMetadata &x,
                const TensorMetadata &y,
                TensorMetadata &out);
#endif

void where(const TensorMetadata &condition,
           const TensorMetadata &x,
           const TensorMetadata &y,
           TensorMetadata &out,
           Backend backend = Backend::Auto);

void clamp(const TensorMetadata &input,
           TensorMetadata &out,
           float min_value,
           float max_value,
           Backend backend = Backend::Auto);

void lerp(const TensorMetadata &start,
          const TensorMetadata &end,
          const TensorMetadata &weight,
          TensorMetadata &out,
          Backend backend = Backend::Auto);

void addcmul(const TensorMetadata &input,
             const TensorMetadata &tensor1,
             const TensorMetadata &tensor2,
             float value,
             TensorMetadata &out,
             Backend backend = Backend::Auto);

#if T81LIB_USE_CUDA
void cuda_clamp(const TensorMetadata &input,
                TensorMetadata &out,
                float min_value,
                float max_value);

void cuda_lerp(const TensorMetadata &start,
               const TensorMetadata &end,
               const TensorMetadata &weight,
               TensorMetadata &out);

void cuda_addcmul(const TensorMetadata &input,
                  const TensorMetadata &tensor1,
                  const TensorMetadata &tensor2,
                  float value,
                  TensorMetadata &out);
#endif
#if T81LIB_USE_ROCM
void rocm_clamp(const TensorMetadata &input,
                TensorMetadata &out,
                float min_value,
                float max_value);

void rocm_lerp(const TensorMetadata &start,
               const TensorMetadata &end,
               const TensorMetadata &weight,
               TensorMetadata &out);

void rocm_addcmul(const TensorMetadata &input,
                  const TensorMetadata &tensor1,
                  const TensorMetadata &tensor2,
                  float value,
                  TensorMetadata &out);
#endif

} // namespace detail
} // namespace t81::linalg
