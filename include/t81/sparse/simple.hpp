#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <span>

#include <t81/core/limb.hpp>
#include <t81/linalg/gemm.hpp>

namespace t81::sparse {

struct SimpleSparseTernary {
    std::vector<core::limb> values;      // non-zero limbs
    std::vector<int32_t> row_indices;     // row index per non-zero entry
    std::vector<int32_t> col_indices;     // column index per non-zero entry
    int rows = 0;
    int cols = 0;
};

namespace detail {

inline double limb_to_double(const core::limb& value) {
    const auto trits = value.to_trits();
    core::detail::limb_int128 sum = 0;
    for (std::size_t index = 0; index < core::limb::TRITS; ++index) {
        sum += static_cast<core::detail::limb_int128>(trits[index]) *
               core::detail::POW3[index];
    }
    return static_cast<double>(sum);
}

} // namespace detail

inline void spmm_simple(const SimpleSparseTernary& A,
                        std::span<const float> B,
                        std::span<float> C,
                        int N,
                        float alpha = 1.0f,
                        float beta = 0.0f) {
    if (N < 0) {
        throw std::invalid_argument("spmm_simple requires N >= 0");
    }
    if (A.rows < 0 || A.cols < 0) {
        throw std::invalid_argument("SimpleSparseTernary dimensions must be non-negative");
    }
    const std::size_t n_size = static_cast<std::size_t>(N);
    const std::size_t rows_size = static_cast<std::size_t>(A.rows);
    const std::size_t cols_size = static_cast<std::size_t>(A.cols);
    const std::size_t expected_b = cols_size * n_size;
    const std::size_t expected_c = rows_size * n_size;
    if (expected_b != B.size()) {
        throw std::invalid_argument("B span size does not match (cols, N)");
    }
    if (expected_c != C.size()) {
        throw std::invalid_argument("C span size does not match (rows, N)");
    }

    if (beta == 0.0f) {
        std::fill(C.begin(), C.end(), 0.0f);
    } else {
        const double beta_d = static_cast<double>(beta);
        for (std::size_t index = 0; index < C.size(); ++index) {
            C[index] = static_cast<float>(static_cast<double>(C[index]) * beta_d);
        }
    }

    const std::size_t entry_count = A.values.size();
    if (A.row_indices.size() != entry_count || A.col_indices.size() != entry_count) {
        throw std::invalid_argument("Sparse matrix index lengths must match values length");
    }

    const double alpha_d = static_cast<double>(alpha);
    for (std::size_t entry = 0; entry < entry_count; ++entry) {
        const int row = A.row_indices[entry];
        const int col = A.col_indices[entry];
        if (row < 0 || row >= A.rows) {
            throw std::invalid_argument("Row index out of bounds");
        }
        if (col < 0 || col >= A.cols) {
            throw std::invalid_argument("Column index out of bounds");
        }
        const double weight =
            detail::limb_to_double(A.values[entry]) * alpha_d;
        const std::size_t row_offset = static_cast<std::size_t>(row) * n_size;
        const std::size_t col_offset = static_cast<std::size_t>(col) * n_size;
        for (std::size_t k = 0; k < n_size; ++k) {
            const double product =
                weight * static_cast<double>(B[col_offset + k]);
            const double accumulated =
                static_cast<double>(C[row_offset + k]) + product;
            C[row_offset + k] = static_cast<float>(accumulated);
        }
    }
}

} // namespace t81::sparse
