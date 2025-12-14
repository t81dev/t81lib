#include <algorithm>
#include <array>

#include "t81/linalg/gemm.hpp"

namespace t81::linalg::detail {

    void gemm_ternary_cpu_impl(std::span<const core::limb> A,
                               std::span<const core::limb> B,
                               std::span<float> C,
                               int M,
                               int N,
                               int K,
                               int K_limbs,
                               float alpha,
                               float beta) {
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

} // namespace t81::linalg::detail
