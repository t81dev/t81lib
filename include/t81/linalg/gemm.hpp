// include/t81/linalg/gemm.hpp — Vectorized GEMM over packed ternary limbs.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>

#include <t81/core/limb.hpp>
#include <t81/linalg/gemm_gpu.hpp>

namespace t81::linalg {

    namespace detail {

        inline void prefetch_read(const void *address) noexcept {
#if defined(__has_builtin)
#if __has_builtin(__builtin_prefetch)
            __builtin_prefetch(address, 0, 3);
#endif
#elif defined(__GNUC__) || defined(__clang__)
            __builtin_prefetch(address, 0, 3);
#endif
        }

        inline constexpr auto build_tryte_contributions()
            -> std::array<std::array<double, 27>, core::limb::TRYTES> {
            std::array<std::array<double, 27>, core::limb::TRYTES> table{};
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

        inline const auto &tryte_contribution_table() {
            static const auto table = build_tryte_contributions();
            return table;
        }

        inline double limb_to_double(const core::limb &value) {
            const auto trytes = value.to_trytes();
            const auto &table = tryte_contribution_table();
            double result = 0.0;
            for (std::size_t index = 0; index < core::limb::TRYTES; ++index) {
                result += table[index][trytes[index]];
            }
            return result;
        }

        inline double multiply_to_double(const core::limb &lhs, const core::limb &rhs) {
            const auto [low, high] = core::limb::mul_wide(lhs, rhs);
            const double low_value = limb_to_double(low);
            const double high_value = limb_to_double(high);
            const double radix = static_cast<double>(core::detail::RADIX);
            return low_value + high_value * radix;
        }

    } // namespace detail

    inline void gemm_ternary(std::span<const core::limb> A,
                             std::span<const core::limb> B,
                             std::span<float> C,
                             int M,
                             int N,
                             int K,
                             float alpha,
                             float beta) {
        detail::gemm_ternary_dispatch(A, B, C, M, N, K, alpha, beta);
    }

} // namespace t81::linalg
