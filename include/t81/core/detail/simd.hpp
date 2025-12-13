// include/t81/core/detail/simd.hpp — SIMD helper declarations for core routines.

// detail/simd.hpp - Detects available SIMD instruction sets for limb operations.
#pragma once

#include <optional>
#include <utility>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif
#endif

namespace t81::core {
    class limb;
} // namespace t81::core

namespace t81::core::detail {

namespace {

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
struct cpuid_regs {
    unsigned int eax{};
    unsigned int ebx{};
    unsigned int ecx{};
    unsigned int edx{};
};

inline cpuid_regs read_cpuid(unsigned int leaf, unsigned int subleaf) {
#if defined(_MSC_VER)
    int regs[4];
    __cpuidex(regs, leaf, subleaf);
    return {static_cast<unsigned int>(regs[0]),
            static_cast<unsigned int>(regs[1]),
            static_cast<unsigned int>(regs[2]),
            static_cast<unsigned int>(regs[3])};
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid_count(leaf, subleaf, &eax, &ebx, &ecx, &edx);
    return {eax, ebx, ecx, edx};
#else
    return {};
#endif
}

inline unsigned long long read_xcr0() {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    unsigned int eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<unsigned long long>(edx) << 32) | eax;
#endif
}

inline bool os_supports_xsave() {
    const auto leaf1 = read_cpuid(1, 0);
    return (leaf1.ecx & (1u << 27)) != 0;
}

inline bool os_supports_xcr_states(unsigned long long mask) {
    if (!os_supports_xsave()) {
        return false;
    }
    return (read_xcr0() & mask) == mask;
}

inline bool os_supports_avx_states() {
    constexpr unsigned long long kMask = (1ull << 1) | (1ull << 2);
    return os_supports_xcr_states(kMask);
}

inline bool os_supports_avx512_states() {
    constexpr unsigned long long kMask =
        (1ull << 1) | (1ull << 2) | (1ull << 5) | (1ull << 6) | (1ull << 7);
    return os_supports_xcr_states(kMask);
}

inline bool cpu_reports_avx() {
    const auto leaf1 = read_cpuid(1, 0);
    return (leaf1.ecx & (1u << 28)) != 0;
}

inline bool cpu_reports_avx2() {
    const auto leaf7 = read_cpuid(7, 0);
    return (leaf7.ebx & (1u << 5)) != 0;
}

inline bool cpu_reports_avx512f() {
    const auto leaf7 = read_cpuid(7, 0);
    return (leaf7.ebx & (1u << 16)) != 0;
}

inline bool has_runtime_avx2() {
    if (!os_supports_avx_states() || !cpu_reports_avx()) {
        return false;
    }
    return cpu_reports_avx2();
}

inline bool has_runtime_avx512f() {
    if (!os_supports_avx512_states()) {
        return false;
    }
    return cpu_reports_avx512f();
}
#else
inline bool has_runtime_avx2() {
    return false;
}

inline bool has_runtime_avx512f() {
    return false;
}
#endif

} // namespace

    inline bool cpu_supports_avx2() noexcept {
#if defined(__has_builtin)
#if __has_builtin(__builtin_cpu_supports)
#if defined(__AVX2__)
        if (__builtin_cpu_supports("avx2")) {
            return true;
        }
#endif
#endif
#endif
        return has_runtime_avx2();
    }

    inline bool cpu_supports_avx512f() noexcept {
#if defined(__has_builtin)
#if __has_builtin(__builtin_cpu_supports)
#if defined(__AVX512F__)
        if (__builtin_cpu_supports("avx512f")) {
            return true;
        }
#endif
#endif
#endif
        return has_runtime_avx512f();
    }

    inline bool cpu_supports_neon() noexcept {
#if defined(T81_DISABLE_NEON)
        return false;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(T81_ENABLE_NEON)
        return true;
#else
        return false;
#endif
    }

    // Returns true when SIMD addition completes without an overflow carry.
    bool add_trytes_avx2(const limb &, const limb &, limb &);
    bool add_trytes_avx512(const limb &, const limb &, limb &);
    bool add_trytes_neon(const limb &, const limb &, limb &);

    std::pair<limb, limb> mul_wide_scalar(const limb &, const limb &);
    std::optional<std::pair<limb, limb>> mul_wide_avx2(const limb &, const limb &);
    std::optional<std::pair<limb, limb>> mul_wide_avx512(const limb &, const limb &);
    std::optional<std::pair<limb, limb>> mul_wide_neon(const limb &, const limb &);

    inline bool add_trytes_simd(const limb &lhs, const limb &rhs, limb &result) {
        if (cpu_supports_avx512f()) {
            return add_trytes_avx512(lhs, rhs, result);
        }
        if (cpu_supports_avx2()) {
            return add_trytes_avx2(lhs, rhs, result);
        }
        if (cpu_supports_neon()) {
            return add_trytes_neon(lhs, rhs, result);
        }
        return false;
    }

    inline std::optional<std::pair<limb, limb>> mul_wide_simd(const limb &lhs, const limb &rhs) {
        if (cpu_supports_avx512f()) {
            return mul_wide_avx512(lhs, rhs);
        }
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
