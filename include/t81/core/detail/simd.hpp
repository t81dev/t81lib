#pragma once

#include <optional>
#include <utility>

namespace t81::core {
class limb;
} // namespace t81::core

namespace t81::core::detail {

inline bool cpu_supports_avx2() noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    #if defined(__has_builtin)
        #if __has_builtin(__builtin_cpu_supports)
            return __builtin_cpu_supports("avx2");
        #else
            return false;
        #endif
    #else
        return false;
    #endif
#else
    return false;
#endif
}

inline bool cpu_supports_neon() noexcept {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #if defined(T81_ENABLE_NEON)
        return true;
    #else
        return false;
    #endif
#else
    return false;
#endif
}

bool add_trytes_avx2(const limb&, const limb&, limb&);
bool add_trytes_neon(const limb&, const limb&, limb&);

std::pair<limb, limb> mul_wide_scalar(const limb&, const limb&);
std::optional<std::pair<limb, limb>> mul_wide_avx2(const limb&, const limb&);
std::optional<std::pair<limb, limb>> mul_wide_neon(const limb&, const limb&);

inline bool add_trytes_simd(const limb& lhs, const limb& rhs, limb& result) {
    if (cpu_supports_avx2()) {
        return add_trytes_avx2(lhs, rhs, result);
    }
    if (cpu_supports_neon()) {
        return add_trytes_neon(lhs, rhs, result);
    }
    return false;
}

inline std::optional<std::pair<limb, limb>> mul_wide_simd(const limb& lhs, const limb& rhs) {
    if (cpu_supports_avx2()) {
        return mul_wide_avx2(lhs, rhs);
    }
    if (cpu_supports_neon()) {
        return mul_wide_neon(lhs, rhs);
    }
    return std::nullopt;
}

} // namespace t81::core::detail

#include <t81/core/detail/simd_impl.hpp>
