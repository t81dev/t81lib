// include/t81/linalg/gemm.hpp — Vectorized GEMM over packed ternary limbs.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>

#include <t81/core/limb.hpp>

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

    inline void gemm_ternary(std::span<const core::limb> A, std::span<const core::limb> B,
                             std::span<float> C, int M, int N, int K, float alpha, float beta) {
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

        constexpr int BlockM = 8;
        constexpr int BlockN = 8;
        constexpr int BlockK = 4;
        const std::size_t N_size = static_cast<std::size_t>(N);
        const auto *const a_data = A.data();
        const auto *const b_data = B.data();
        auto *const c_data = C.data();

        for (int ib = 0; ib < M; ib += BlockM) {
            const int i_end = std::min(M, ib + BlockM);
            for (int jb = 0; jb < N; jb += BlockN) {
                const int j_end = std::min(N, jb + BlockN);
                std::array<std::array<double, BlockN>, BlockM> accum{};
                for (int i = ib; i < i_end; ++i) {
                    const std::size_t row = static_cast<std::size_t>(i) * N_size;
                    for (int j = jb; j < j_end; ++j) {
                        const float existing = c_data[row + static_cast<std::size_t>(j)];
                        accum[i - ib][j - jb] = static_cast<double>(existing) * beta;
                    }
                }

                for (int kb = 0; kb < K_limbs; kb += BlockK) {
                    const int k_end = std::min(K_limbs, kb + BlockK);
                    for (int k = kb; k < k_end; ++k) {
                        const std::size_t b_row = static_cast<std::size_t>(k) * N_size;
                        for (int j = jb; j < j_end; ++j) {
                            const core::limb b_value = b_data[b_row + static_cast<std::size_t>(j)];
                            detail::prefetch_read(b_data + b_row + static_cast<std::size_t>(j) + 1);
                            for (int i = ib; i < i_end; ++i) {
                                const std::size_t a_index = static_cast<std::size_t>(i) *
                                                                static_cast<std::size_t>(K_limbs) +
                                                            static_cast<std::size_t>(k);
                                const core::limb a_value = a_data[a_index];
                                const double product = detail::multiply_to_double(a_value, b_value);
                                accum[i - ib][j - jb] += product * static_cast<double>(alpha);
                                detail::prefetch_read(a_data + a_index + 1);
                            }
                        }
                    }
                }

                for (int i = ib; i < i_end; ++i) {
                    const std::size_t row = static_cast<std::size_t>(i) * N_size;
                    for (int j = jb; j < j_end; ++j) {
                        c_data[row + static_cast<std::size_t>(j)] =
                            static_cast<float>(accum[i - ib][j - jb]);
                    }
                }
            }
        }
    }

} // namespace t81::linalg
