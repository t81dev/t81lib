#pragma once

#include <cstddef>
#include <span>
#include <cstdint>
#include <stdexcept>

#include <t81/tensor_metadata.hpp>
#include <t81/core/limb.hpp>

#ifndef T81LIB_USE_CUDA
#define T81LIB_USE_CUDA 0
#endif

#ifndef T81LIB_USE_ROCM
#define T81LIB_USE_ROCM 0
#endif

#ifndef T81LIB_USE_METAL
#define T81LIB_USE_METAL 0
#endif

namespace t81::linalg {

enum class Backend { Auto, CPU, CUDA, ROCm, Metal };

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
    #if T81LIB_USE_METAL
        case Backend::Metal:
            return true;
    #else
        case Backend::Metal:
            return false;
    #endif
    }
    return false;
}

#if !defined(T81LIB_DOXYGEN)
Backend get_current_backend() noexcept;
#endif

#if T81LIB_USE_METAL
bool metal_available() noexcept;
#endif
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

#if !defined(T81LIB_DOXYGEN)
void gemm_ternary(const TensorMetadata &A,
                  const TensorMetadata &B,
                  TensorMetadata &C,
                  float alpha,
                  float beta,
                  Backend backend = Backend::Auto);
#endif

#if T81LIB_USE_CUDA
void cuda_gemm_ternary(std::span<const core::limb> A,
                       std::span<const core::limb> B,
                       std::span<float> C,
                       int M,
                       int N,
                       int K,
                       float alpha,
                       float beta);
#endif
#if T81LIB_USE_ROCM
void rocm_gemm_ternary(std::span<const core::limb> A,
                       std::span<const core::limb> B,
                       std::span<float> C,
                       int M,
                       int N,
                       int K,
                       float alpha,
                       float beta);
#endif
#if T81LIB_USE_METAL
void metal_gemm_ternary(std::span<const core::limb> A,
                        std::span<const core::limb> B,
                        std::span<float> C,
                        int M,
                        int N,
                        int K,
                        float alpha,
                        float beta);
#endif

void gemm_ternary_cpu_impl(std::span<const core::limb> A,
                           std::span<const core::limb> B,
                           std::span<float> C,
                           int M,
                           int N,
                           int K,
                           int K_limbs,
                           float alpha,
                           float beta);

inline void gemm_ternary_dispatch(std::span<const core::limb> A,
                                  std::span<const core::limb> B,
                                  std::span<float> C,
                                  int M,
                                  int N,
                                  int K,
                                  float alpha,
                                  float beta,
                                  Backend backend = Backend::Auto) {
    if (M < 0 || N < 0 || K < 0) {
        throw std::invalid_argument("gemm_ternary dimensions must be non-negative");
    }
    if (K % core::limb::TRITS != 0) {
        throw std::invalid_argument("gemm_ternary requires K divisible by 48");
    }
    const int K_limbs = K / core::limb::TRITS;
    if (static_cast<std::size_t>(M) * static_cast<std::size_t>(K_limbs) != A.size()) {
        throw std::invalid_argument("A span size does not match (M, K / 48)");
    }
    if (static_cast<std::size_t>(K_limbs) * static_cast<std::size_t>(N) != B.size()) {
        throw std::invalid_argument("B span size does not match (K / 48, N)");
    }
    if (static_cast<std::size_t>(M) * static_cast<std::size_t>(N) != C.size()) {
        throw std::invalid_argument("C span size does not match (M, N)");
    }

    if (M == 0 || N == 0) {
        return;
    }

    const Backend target = backend == Backend::Auto ? get_current_backend() : backend;
#if T81LIB_USE_CUDA
    if (target == Backend::CUDA && backend_available(Backend::CUDA)) {
        cuda_gemm_ternary(A, B, C, M, N, K, alpha, beta);
        return;
    }
#endif
#if T81LIB_USE_ROCM
    if (target == Backend::ROCm && backend_available(Backend::ROCm)) {
        rocm_gemm_ternary(A, B, C, M, N, K, alpha, beta);
        return;
    }
#endif

#if T81LIB_USE_METAL
    if (target == Backend::Metal && backend_available(Backend::Metal)) {
        metal_gemm_ternary(A, B, C, M, N, K, alpha, beta);
        return;
    }
#endif

    gemm_ternary_cpu_impl(A, B, C, M, N, K, K_limbs, alpha, beta);
}

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
