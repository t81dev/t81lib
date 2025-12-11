// python/bindings.cpp — Pybind11 bindings for the t81lib module.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "t81/core/gguf_quants.h"

#include <t81/core/bigint.hpp>
#include <t81/core/bigint_bitops_helpers.hpp>
#include <t81/core/limb.hpp>
#include <t81/core/montgomery.hpp>
#include <t81/io/format.hpp>
#include <t81/linalg/gemm.hpp>
#include <t81/sparse/simple.hpp>
#include <t81/t81lib.hpp>

namespace py = pybind11;
namespace core = t81::core;

static std::string decimal_string(const core::bigint& value) {
    if (value.is_zero()) {
        return "0";
    }
    const bool negative = value.is_negative();
    core::bigint cursor = value.abs();
    const core::bigint base(10);
    std::string result;
    while (!cursor.is_zero()) {
        const auto [quotient, remainder] = core::bigint::div_mod(cursor, base);
        cursor = quotient;
        const long long digit_value = static_cast<long long>(remainder.to_limb().to_value());
        result.push_back(static_cast<char>('0' + digit_value));
    }
    if (negative) {
        result.push_back('-');
    }
    std::reverse(result.begin(), result.end());
    return result;
}

static py::int_ to_python_int(const core::bigint& value) {
    const auto builtins = py::module_::import("builtins");
    return builtins.attr("int")(decimal_string(value));
}

static py::int_ limb_to_python_int(const core::limb& value) {
    return to_python_int(core::bigint(value));
}

namespace {

bool buffer_is_c_contiguous(const py::buffer_info& info) {
    if (info.ndim <= 1) {
        return true;
    }
    std::size_t expected_stride = static_cast<std::size_t>(info.itemsize);
    for (int dimension = info.ndim - 1; dimension >= 0; --dimension) {
        if (info.shape[dimension] < 0) {
            return false;
        }
        const std::size_t actual_stride = static_cast<std::size_t>(info.strides[dimension]);
        if (actual_stride != expected_stride) {
            return false;
        }
        expected_stride *= static_cast<std::size_t>(info.shape[dimension]);
    }
    return true;
}

std::span<const core::limb> make_limb_span(py::buffer& buffer, std::size_t count) {
    const auto info = buffer.request(false);
    if (info.size < 0) {
        throw py::value_error("Packed limb buffer has invalid size");
    }
    if (info.itemsize <= 0) {
        throw py::value_error("Packed limb buffer has invalid item size");
    }
    if (!buffer_is_c_contiguous(info)) {
        throw py::value_error("Packed limb buffers must use C-style contiguous memory");
    }
    const std::size_t total_bytes =
        static_cast<std::size_t>(std::max<py::ssize_t>(info.size, 0)) *
        static_cast<std::size_t>(info.itemsize);
    if (total_bytes != count * sizeof(core::limb)) {
        throw py::value_error("Packed limb buffer byte size mismatch");
    }
    const auto* data = info.ptr ? reinterpret_cast<const core::limb*>(info.ptr) : nullptr;
    return {data, count};
}

std::span<float> make_float_span(py::buffer& buffer, std::size_t count) {
    auto info = buffer.request(true);
    if (info.readonly) {
        throw py::value_error("C buffer must be writable");
    }
    if (info.size < 0) {
        throw py::value_error("C buffer has invalid size");
    }
    if (info.itemsize != static_cast<py::ssize_t>(sizeof(float))) {
        throw py::value_error("C buffer must store float32 values");
    }
    if (!buffer_is_c_contiguous(info)) {
        throw py::value_error("C buffer must use C-style contiguous memory");
    }
    const std::size_t total_bytes =
        static_cast<std::size_t>(std::max<py::ssize_t>(info.size, 0)) *
        static_cast<std::size_t>(info.itemsize);
    if (total_bytes != count * sizeof(float)) {
        throw py::value_error("C buffer byte size mismatch");
    }
    auto* data = info.ptr ? reinterpret_cast<float*>(info.ptr) : nullptr;
    return {data, count};
}

constexpr std::int8_t quantize_trit(float value, float threshold) {
    const float clamped = std::clamp(value, -1.0f, 1.0f);
    if (clamped >= threshold) {
        return 1;
    }
    if (clamped <= -threshold) {
        return -1;
    }
    return 0;
}

py::array_t<std::int8_t> quantize_to_trits(py::array_t<float, py::array::c_style | py::array::forcecast> array,
                                            float threshold) {
    if (threshold <= 0.0f || threshold > 1.0f) {
        throw py::value_error("threshold must be between 0 (exclusive) and 1");
    }
    const auto info = array.request();
    if (info.ndim == 0) {
        throw py::value_error("expected at least one dimension for quantization");
    }
    py::array_t<std::int8_t> output(info.shape);
    const std::size_t total = static_cast<std::size_t>(std::max<py::ssize_t>(info.size, 0));
    const auto src = static_cast<const float*>(info.ptr);
    const auto dst = static_cast<std::int8_t*>(output.request().ptr);
    for (std::size_t index = 0; index < total; ++index) {
        dst[index] = quantize_trit(src[index], threshold);
    }
    return output;
}

py::array_t<float> dequantize_trits(py::array_t<std::int8_t, py::array::c_style | py::array::forcecast> array) {
    const auto info = array.request();
    py::array_t<float> output(info.shape);
    const std::size_t total = static_cast<std::size_t>(std::max<py::ssize_t>(info.size, 0));
    const auto src = static_cast<const std::int8_t*>(info.ptr);
    auto* dst = static_cast<float*>(output.request().ptr);
    for (std::size_t index = 0; index < total; ++index) {
        dst[index] = static_cast<float>(src[index]);
    }
    return output;
}

py::array_t<std::uint8_t> pack_dense_matrix(py::array_t<float, py::array::c_style | py::array::forcecast> array,
                                             float threshold) {
    if (threshold <= 0.0f || threshold > 1.0f) {
        throw py::value_error("threshold must be between 0 (exclusive) and 1");
    }
    const auto info = array.request();
    if (info.ndim != 2) {
        throw py::value_error("pack_dense_matrix expects a 2D array");
    }
    const int rows = static_cast<int>(info.shape[0]);
    const int cols = static_cast<int>(info.shape[1]);
    const int trits_per_limb = core::limb::TRITS;
    const int limbs_per_row = (cols + trits_per_limb - 1) / trits_per_limb;
    const std::size_t limb_bytes = static_cast<std::size_t>(core::limb::BYTES);
    py::array_t<std::uint8_t> packed({static_cast<std::size_t>(rows),
                                      static_cast<std::size_t>(limbs_per_row),
                                      limb_bytes});
    const auto* src = static_cast<const float*>(info.ptr);
    auto* dst = static_cast<std::uint8_t*>(packed.request().ptr);
    const std::size_t row_stride = static_cast<std::size_t>(limbs_per_row) * limb_bytes;
    for (int row = 0; row < rows; ++row) {
        const auto* row_ptr = src + static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
        for (int limb_index = 0; limb_index < limbs_per_row; ++limb_index) {
            std::array<std::int8_t, core::limb::TRITS> trits{};
            for (int trit_index = 0; trit_index < core::limb::TRITS; ++trit_index) {
                const int column = limb_index * trits_per_limb + trit_index;
                if (column < cols) {
                    trits[trit_index] = quantize_trit(row_ptr[column], threshold);
                } else {
                    trits[trit_index] = 0;
                }
            }
            const auto bytes = core::limb::from_trits(trits).to_bytes();
            const std::size_t offset = static_cast<std::size_t>(row) * row_stride +
                                       static_cast<std::size_t>(limb_index) * limb_bytes;
            std::memcpy(dst + offset, bytes.data(), limb_bytes);
        }
    }
    return packed;
}

py::tuple quantize_row_tq1_0(py::array_t<float, py::array::c_style | py::array::forcecast> row,
                             float threshold,
                             float scale) {
    const auto info = row.request();
    if (info.ndim != 1) {
        throw py::value_error("quantize_row_tq1_0 requires a 1-D float array");
    }
    const std::int64_t length = static_cast<std::int64_t>(info.shape[0]);
    const std::size_t blocks = static_cast<std::size_t>(
        (length + static_cast<std::int64_t>(t81::core::gguf::TQ1_TRITS_PER_BLOCK) - 1) /
        static_cast<std::int64_t>(t81::core::gguf::TQ1_TRITS_PER_BLOCK));
    py::array_t<std::uint16_t> packed(static_cast<py::ssize_t>(blocks));
    auto packed_info = packed.request(true);
    auto* dest = packed_info.ptr ? static_cast<std::uint16_t*>(packed_info.ptr) : nullptr;
    float actual_scale = scale;
    if (actual_scale <= 0.0f) {
        actual_scale = t81::core::gguf::compute_scale(
            static_cast<const float*>(info.ptr), length);
    }
    t81::core::gguf::quantize_row_tq1_0(
        static_cast<const float*>(info.ptr),
        length,
        threshold,
        actual_scale,
        dest);
    return py::make_tuple(actual_scale, std::move(packed));
}

py::array_t<float> dequant_row_tq1_0(py::array_t<std::uint16_t, py::array::c_style | py::array::forcecast> packed,
                                     int cols,
                                     float scale) {
    if (cols < 0) {
        throw py::value_error("cols must be non-negative");
    }
    const auto info = packed.request();
    const std::size_t expected_blocks = static_cast<std::size_t>(
        (static_cast<std::size_t>(cols) + t81::core::gguf::TQ1_TRITS_PER_BLOCK - 1) /
        t81::core::gguf::TQ1_TRITS_PER_BLOCK);
    if (static_cast<std::size_t>(info.size) < expected_blocks) {
        throw py::value_error("packed buffer too small for the requested column count");
    }
    py::array_t<float> result(static_cast<py::ssize_t>(cols));
    auto result_info = result.request(true);
    auto* destination = static_cast<float*>(result_info.ptr);
    t81::core::gguf::dequantize_row_tq1_0(
        static_cast<const std::uint16_t*>(info.ptr),
        cols,
        scale,
        destination);
    return result;
}

py::array_t<float> dequant_tensor_tq1_impl(py::buffer buffer,
                                           std::int64_t rows,
                                           std::int64_t cols,
                                           std::int64_t block_rows,
                                           bool has_refinements) {
    if (rows < 0 || cols < 0) {
        throw py::value_error("rows and cols must be non-negative");
    }
    const auto info = buffer.request();
    const std::size_t total_bytes =
        static_cast<std::size_t>(std::max<py::ssize_t>(0, info.size)) *
        static_cast<std::size_t>(info.itemsize);
    if (info.ptr == nullptr && total_bytes > 0) {
        throw py::value_error("buffer pointer is null");
    }
    if (info.itemsize != 1) {
        throw py::value_error("buffer must be observed as bytes");
    }

    py::array_t<float> result({static_cast<std::size_t>(rows), static_cast<std::size_t>(cols)});
    auto* destination = static_cast<float*>(result.request().ptr);
    const std::size_t row_blocks = static_cast<std::size_t>(
        (static_cast<std::size_t>(cols) + t81::core::gguf::TQ1_TRITS_PER_BLOCK - 1) /
        t81::core::gguf::TQ1_TRITS_PER_BLOCK);
    const std::size_t row_bytes = row_blocks * sizeof(std::uint16_t);
    const std::size_t group_rows = static_cast<std::size_t>(
        std::max<std::int64_t>(1, block_rows));
    std::size_t offset = 0;
    std::int64_t current_row = 0;
    std::vector<std::uint16_t> row_buffer(row_blocks);

    while (current_row < rows) {
        if (offset + sizeof(std::uint16_t) > total_bytes) {
            throw py::value_error("buffer truncated while reading scale");
        }
        std::uint16_t scale_bits = 0;
        std::memcpy(&scale_bits, static_cast<const std::uint8_t*>(info.ptr) + offset, sizeof(scale_bits));
        offset += sizeof(std::uint16_t);
        if (has_refinements) {
            constexpr std::size_t kRefinementBytes = 8;
            if (offset + kRefinementBytes > total_bytes) {
                throw py::value_error("buffer truncated while skipping refinement bytes");
            }
            offset += kRefinementBytes;
        }
        const float scale = t81::core::gguf::half_to_float(scale_bits);
        const std::size_t rows_in_group = static_cast<std::size_t>(
            std::min<std::int64_t>(group_rows, rows - current_row));

        for (std::size_t group_index = 0; group_index < rows_in_group; ++group_index) {
            if (offset + row_bytes > total_bytes) {
                throw py::value_error("buffer truncated while reading quantized rows");
            }
            if (row_bytes > 0) {
                std::memcpy(row_buffer.data(),
                            static_cast<const std::uint8_t*>(info.ptr) + offset,
                            row_bytes);
            }
            offset += row_bytes;
            t81::core::gguf::dequantize_row_tq1_0(
                row_buffer.data(),
                cols,
                scale,
                destination + static_cast<std::size_t>(current_row) * static_cast<std::size_t>(cols));
            ++current_row;
        }
    }
    return result;
}

py::array_t<std::int8_t> unpack_packed_limbs(py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> packed,
                                               int rows,
                                               int cols) {
    if (rows < 0 || cols < 0) {
        throw py::value_error("rows and cols must be non-negative");
    }
    const auto info = packed.request();
    if (info.ndim != 3) {
        throw py::value_error("packed limbs must be a 3D array");
    }
    const int actual_rows = static_cast<int>(info.shape[0]);
    const int limbs_per_row = static_cast<int>(info.shape[1]);
    if (rows != actual_rows) {
        throw py::value_error("row count mismatch");
    }
    if (static_cast<int>(info.shape[2]) != static_cast<int>(core::limb::BYTES)) {
        throw py::value_error("packed limb innermost dimension must equal limb byte width");
    }
    const int max_trits = limbs_per_row * core::limb::TRITS;
    if (cols > max_trits) {
        throw py::value_error("cols exceeds packed trit capacity");
    }
    py::array_t<std::int8_t> output({static_cast<std::size_t>(rows), static_cast<std::size_t>(cols)});
    const auto* src = static_cast<const std::uint8_t*>(info.ptr);
    auto* dst = static_cast<std::int8_t*>(output.request().ptr);
    const std::size_t row_stride = static_cast<std::size_t>(limbs_per_row) * core::limb::BYTES;
    for (int row = 0; row < rows; ++row) {
        for (int limb_index = 0; limb_index < limbs_per_row; ++limb_index) {
            const std::size_t offset = static_cast<std::size_t>(row) * row_stride +
                                       static_cast<std::size_t>(limb_index) * core::limb::BYTES;
            std::array<std::uint8_t, core::limb::BYTES> bytes{};
            std::memcpy(bytes.data(), src + offset, core::limb::BYTES);
            const auto limb_value = core::limb::from_bytes(bytes);
            const auto trits = limb_value.to_trits();
            for (int trit_index = 0; trit_index < core::limb::TRITS; ++trit_index) {
                const int column = limb_index * core::limb::TRITS + trit_index;
                if (column >= cols) {
                    break;
                }
                dst[static_cast<std::size_t>(row) * cols + column] = trits[trit_index];
            }
        }
    }
    return output;
}

core::bigint parse_bigint_string(std::string text, int base) {
    if (text.empty()) {
        throw std::invalid_argument("empty string");
    }
    if (base < 2 || base > 36) {
        throw std::invalid_argument("base must be between 2 and 36");
    }
    bool negative = false;
    std::size_t index = 0;
    if (text[0] == '+' || text[0] == '-') {
        negative = (text[0] == '-');
        index = 1;
        if (index == text.size()) {
            throw std::invalid_argument("string has only a sign");
        }
    }
    core::bigint accumulator = core::bigint::zero();
    const core::bigint base_value(static_cast<long long>(base));
    for (; index < text.size(); ++index) {
        const char ch = text[index];
        int digit;
        if (ch >= '0' && ch <= '9') {
            digit = ch - '0';
        } else if (ch >= 'a' && ch <= 'z') {
            digit = 10 + (ch - 'a');
        } else if (ch >= 'A' && ch <= 'Z') {
            digit = 10 + (ch - 'A');
        } else {
            throw std::invalid_argument("invalid digit in string");
        }
        if (digit >= base) {
            throw std::invalid_argument("digit outside of base");
        }
        accumulator = accumulator * base_value + core::bigint(digit);
    }
    if (negative) {
        accumulator = -accumulator;
    }
    return accumulator;
}

template <typename Int>
void bind_montgomery_context(py::module_& module, const char* name) {
    using Context = core::MontgomeryContext<Int>;
    py::class_<Context> cls(module, name, "Montgomery reduction context for modular math");
    cls.def(py::init<const Int&>(), py::arg("modulus"))
        .def_property_readonly("modulus", &Context::modulus,
                               "Modulus used by this Montgomery context")
        .def("to_montgomery", &Context::to_montgomery, py::arg("value"),
             "Map a value into Montgomery form")
        .def("from_montgomery", &Context::from_montgomery, py::arg("value"),
             "Reduce a Montgomery number back to standard representation")
        .def("mod_mul", &Context::mul, py::arg("lhs"), py::arg("rhs"),
             "Perform a Montgomery-aware multiplication")
        .def("mod_pow", &Context::pow, py::arg("base"), py::arg("exponent"),
             "Exponentiate inside the Montgomery domain");
}

} // namespace

PYBIND11_MODULE(t81lib, module) {
    module.doc() = "Pybind11 bindings for the t81lib balanced ternary primitives";
    module.attr("TRITS_PER_LIMB") = core::limb::TRITS;
    module.attr("TRYTES_PER_LIMB") = core::limb::TRYTES;
    module.attr("BYTES_PER_LIMB") = core::limb::BYTES;

    module.def("gemm_ternary",
               [](py::buffer A_packed,
                  py::buffer B_packed,
                  py::buffer C_buffer,
                  int M,
                  int N,
                  int K,
                  float alpha,
                  float beta) {
                   if (M < 0 || N < 0 || K < 0) {
                       throw py::value_error("gemm_ternary dimensions must be non-negative");
                   }
                   if (K % core::limb::TRITS != 0) {
                       throw py::value_error("K must be divisible by the limb trit count");
                   }
                   const int K_limbs = K / core::limb::TRITS;
                   const std::size_t m_size = static_cast<std::size_t>(M);
                   const std::size_t n_size = static_cast<std::size_t>(N);
                   const std::size_t k_size = static_cast<std::size_t>(K_limbs);
                   const std::size_t expected_a = m_size * k_size;
                   const std::size_t expected_b = k_size * n_size;
                   const std::size_t expected_c = m_size * n_size;
                   py::buffer a_view = std::move(A_packed);
                   py::buffer b_view = std::move(B_packed);
                   py::buffer c_view = std::move(C_buffer);
                   const auto a_span = make_limb_span(a_view, expected_a);
                   const auto b_span = make_limb_span(b_view, expected_b);
                   auto c_span = make_float_span(c_view, expected_c);
                   t81::linalg::gemm_ternary(a_span, b_span, c_span, M, N, K, alpha, beta);
               },
               py::arg("A"),
               py::arg("B"),
               py::arg("C"),
               py::arg("M"),
               py::arg("N"),
               py::arg("K"),
               py::arg("alpha") = 1.0f,
               py::arg("beta") = 0.0f,
               "Packed-limb GEMM using AVX/NEON kernels");

    module.def("spmm_simple",
               [](const std::vector<long long>& values,
                  const std::vector<int32_t>& row_indices,
                  const std::vector<int32_t>& col_indices,
                  int rows,
                  int cols,
                  py::buffer B_buffer,
                  py::buffer C_buffer,
                  int N,
                  float alpha,
                  float beta) {
                   if (values.size() != row_indices.size() || values.size() != col_indices.size()) {
                       throw py::value_error("Values and indices lengths must match");
                   }
                   if (rows < 0 || cols < 0) {
                       throw py::value_error("Matrix dimensions must be non-negative");
                   }
                   if (N < 0) {
                       throw py::value_error("Feature dimension N must be non-negative");
                   }

                   t81::sparse::SimpleSparseTernary matrix;
                   matrix.rows = rows;
                   matrix.cols = cols;
                   matrix.row_indices = row_indices;
                   matrix.col_indices = col_indices;
                   matrix.values.reserve(values.size());
                   for (const auto scalar_value : values) {
                       matrix.values.emplace_back(static_cast<long long>(scalar_value));
                   }

                   const std::size_t n_size = static_cast<std::size_t>(N);
                   const std::size_t expected_b = static_cast<std::size_t>(cols) * n_size;
                   const std::size_t expected_c = static_cast<std::size_t>(rows) * n_size;
                   py::buffer B_view = std::move(B_buffer);
                   py::buffer C_view = std::move(C_buffer);
                   const auto b_span = make_float_span(B_view, expected_b);
                   auto c_span = make_float_span(C_view, expected_c);

                   t81::sparse::spmm_simple(matrix, b_span, c_span, N, alpha, beta);
               },
               py::arg("values"),
               py::arg("row_indices"),
               py::arg("col_indices"),
               py::arg("rows"),
               py::arg("cols"),
               py::arg("B"),
               py::arg("C"),
               py::arg("N"),
               py::arg("alpha") = 1.0f,
               py::arg("beta") = 0.0f,
               "COO sparse × dense multiply over ternary weights");

    module.def("quantize_to_trits", &quantize_to_trits, py::arg("values"), py::arg("threshold") = 0.5f,
               "Convert float arrays into {-1,0,1} trit tensors (works with NumPy or buffer-compatible tensors)");

    module.def("dequantize_trits", &dequantize_trits, py::arg("trits"),
               "Turn a trit tensor into its float32 representation for debugging or re-projection");

    module.def("pack_dense_matrix", &pack_dense_matrix,
               py::arg("matrix"),
               py::arg("threshold") = 0.5f,
               "Quantize a 2D float array to balanced ternary and return packed limb bytes in a (rows, limbs, bytes) view");

    module.def("quantize_row_tq1_0", &quantize_row_tq1_0,
               py::arg("row"),
               py::arg("threshold") = 0.45f,
               py::arg("scale") = 0.0f,
               "Pack a single row of floats into TQ1_0 blocks and return (scale, packed blocks)");

    module.def("dequant_row_tq1_0", &dequant_row_tq1_0,
               py::arg("blocks"),
               py::arg("cols"),
               py::arg("scale"),
               "Dequantize a single TQ1_0 row back to float32");

    module.def("dequant_tq1_0",
               [](py::buffer buffer,
                  int64_t rows,
                  int64_t cols,
                  int64_t block_rows) {
                   return dequant_tensor_tq1_impl(buffer, rows, cols, block_rows, false);
               },
               py::arg("buffer"),
               py::arg("rows"),
               py::arg("cols"),
               py::arg("block_rows") = static_cast<int64_t>(t81::core::gguf::TQ1_BLOCK_ROWS),
               "Dequantize an entire TQ1_0 tensor payload");

    module.def("dequant_tq2_0",
               [](py::buffer buffer,
                  int64_t rows,
                  int64_t cols,
                  int64_t block_rows) {
                   return dequant_tensor_tq1_impl(buffer, rows, cols, block_rows, true);
               },
               py::arg("buffer"),
               py::arg("rows"),
               py::arg("cols"),
               py::arg("block_rows") = static_cast<int64_t>(t81::core::gguf::TQ1_BLOCK_ROWS),
               "Dequantize an entire TQ2_0 tensor payload (refinements ignored for now)");

    module.def("unpack_packed_limbs", &unpack_packed_limbs,
               py::arg("packed"),
               py::arg("rows"),
               py::arg("cols"),
               "Recover the quantized trits stored in a (rows, limbs, bytes) packed buffer");

    module.def("limb_from_bytes",
               [](py::buffer buffer) {
                   const auto info = buffer.request();
                   if (info.ndim != 1 || info.shape[0] != static_cast<py::ssize_t>(core::limb::BYTES)) {
                       throw py::value_error(
                           "limb_from_bytes expects a byte array of length " +
                           std::to_string(core::limb::BYTES));
                   }
                   if (!buffer_is_c_contiguous(info)) {
                       throw py::value_error("limb_from_bytes requires a contiguous buffer");
                   }
                   std::array<std::uint8_t, core::limb::BYTES> bytes{};
                   std::memcpy(bytes.data(), info.ptr, core::limb::BYTES);
                   return core::limb::from_bytes(bytes);
               },
               py::arg("buffer"),
               "Decode a single limb from its raw 16 byte representation");

    module.def("bigint_gcd", &core::bigint::gcd,
               py::arg("lhs"),
               py::arg("rhs"),
               "Compute the greatest common divisor of two bigints");

    module.def("bigint_mod_pow", &core::bigint::mod_pow,
               py::arg("base"),
               py::arg("exponent"),
               py::arg("modulus"),
               "Modular exponentiation using repeated squaring");

    py::class_<core::limb> py_limb(module, "Limb", "A 48-trit balanced ternary limb for packing data");
    py_limb.def(py::init<>())
        .def(py::init<long long>(), py::arg("value"))
        .def_static("zero", &core::limb::zero)
        .def_static("one", &core::limb::one)
        .def_static("from_value", &core::limb::from_value)
        .def_static("from_trits", &core::limb::from_trits)
        .def_static("from_bytes", [](py::buffer buffer) {
            const auto info = buffer.request();
            if (info.ndim != 1 || info.shape[0] != static_cast<py::ssize_t>(core::limb::BYTES)) {
                throw py::value_error("from_bytes expects 16 bytes of tryte data");
            }
            if (!buffer_is_c_contiguous(info)) {
                throw py::value_error("from_bytes requires a contiguous buffer");
            }
            std::array<std::uint8_t, core::limb::BYTES> bytes{};
            std::memcpy(bytes.data(), info.ptr, core::limb::BYTES);
            return core::limb::from_bytes(bytes);
        })
        .def_static("from_string", &core::limb::from_string, py::arg("value"), py::arg("base") = 10)
        .def_static("from_float", &core::limb::from_float)
        .def_static("from_double", &core::limb::from_double)
        .def_static("from_long_double", &core::limb::from_long_double)
        .def("to_int", &limb_to_python_int)
        .def("to_float", &core::limb::to_float)
        .def("to_double", &core::limb::to_double)
        .def("to_string", &core::limb::to_string, py::arg("base") = 10)
        .def("get_trit", &core::limb::get_trit)
        .def("set_trit", &core::limb::set_trit, py::arg("index"), py::arg("value"))
        .def("get_tryte", &core::limb::get_tryte)
        .def("to_bytes", [](const core::limb& self) {
            const auto bytes = self.to_bytes();
            return py::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        })
        .def("consensus", &core::limb::consensus)
        .def_static("pow_mod", &core::limb::pow_mod, py::arg("base"), py::arg("exponent"), py::arg("modulus"))
        .def("__add__", [](const core::limb& a, const core::limb& b) { return a + b; })
        .def("__sub__", [](const core::limb& a, const core::limb& b) { return a - b; })
        .def("__mul__", [](const core::limb& a, const core::limb& b) { return a * b; })
        .def("__truediv__", [](const core::limb& a, const core::limb& b) { return a / b; })
        .def("__mod__", [](const core::limb& a, const core::limb& b) { return a % b; })
        .def("__neg__", [](const core::limb& a) { return -a; })
        .def("__and__", [](const core::limb& a, const core::limb& b) { return a & b; })
        .def("__or__", [](const core::limb& a, const core::limb& b) { return a | b; })
        .def("__xor__", [](const core::limb& a, const core::limb& b) { return a ^ b; })
        .def("__lt__", [](const core::limb& a, const core::limb& b) { return a < b; })
        .def("__le__", [](const core::limb& a, const core::limb& b) { return a <= b; })
        .def("__eq__", [](const core::limb& a, const core::limb& b) { return a == b; })
        .def("__ne__", [](const core::limb& a, const core::limb& b) { return a != b; })
        .def("__gt__", [](const core::limb& a, const core::limb& b) { return a > b; })
        .def("__ge__", [](const core::limb& a, const core::limb& b) { return a >= b; })
        .def("rotate_left_tbits", &core::limb::rotate_left_tbits)
        .def("rotate_right_tbits", &core::limb::rotate_right_tbits)
        .def("trit_shift_left", &core::limb::trit_shift_left)
        .def("trit_shift_right", &core::limb::trit_shift_right)
        .def("tryte_shift_left", &core::limb::tryte_shift_left)
        .def("tryte_shift_right", &core::limb::tryte_shift_right)
        .def("__int__", &limb_to_python_int)
        .def("__repr__", [](const core::limb& self) {
            return "<t81lib.Limb " + self.to_string() + ">";
        })
        .def("__str__", &core::limb::to_string);

    py::class_<core::bigint> py_bigint(module, "BigInt", "Arbitrary precision bigint with balanced ternary limbs");
    py_bigint.def(py::init<>())
        .def(py::init<long long>(), py::arg("value"))
        .def_static("from_limbs", &core::bigint::from_limbs, py::arg("limbs"), py::arg("negative"))
        .def_static("from_trits", [](py::array_t<std::int8_t, py::array::c_style | py::array::forcecast> trits) {
            const auto info = trits.request();
            if (info.ndim != 1) {
                throw py::value_error("from_trits expects a flat trit buffer");
            }
            const std::size_t total = static_cast<std::size_t>(std::max<py::ssize_t>(info.size, 0));
            std::vector<std::int8_t> buffer;
            buffer.resize(total);
            std::memcpy(buffer.data(), info.ptr, total * sizeof(std::int8_t));
            return core::from_signed_trits(std::move(buffer));
        })
        .def_static("from_string", [](const std::string& text, int base) {
            return parse_bigint_string(text, base);
        }, py::arg("text"), py::arg("base") = 10)
        .def("__int__", &to_python_int)
        .def("__str__", [](const core::bigint& value) { return decimal_string(value); })
        .def("__repr__", [](const core::bigint& value) {
            return "<t81lib.BigInt " + decimal_string(value) + ">";
        })
        .def("__bool__", [](const core::bigint& value) { return !value.is_zero(); })
        .def("limb_count", &core::bigint::limb_count)
        .def("limb_at", &core::bigint::limb_at)
        .def("to_trits", [](const core::bigint& self) {
            const auto trits = core::signed_trits(self);
            py::array_t<std::int8_t> output(static_cast<std::size_t>(trits.size()));
            std::memcpy(output.request().ptr, trits.data(), trits.size() * sizeof(std::int8_t));
            return output;
        })
        .def("to_limb", &core::bigint::to_limb)
        .def("karatsuba_multiply", [](const core::bigint& lhs, const core::bigint& rhs) {
            return lhs * rhs;
        }, py::arg("other"), "Scale up to large operands using the built-in Karatsuba path")
        .def("__add__", [](const core::bigint& a, const core::bigint& b) { return a + b; })
        .def("__sub__", [](const core::bigint& a, const core::bigint& b) { return a - b; })
        .def("__mul__", [](const core::bigint& a, const core::bigint& b) { return a * b; })
        .def("__truediv__", [](const core::bigint& a, const core::bigint& b) { return a / b; })
        .def("__mod__", [](const core::bigint& a, const core::bigint& b) { return a % b; })
        .def("__neg__", [](const core::bigint& a) { return -a; })
        .def("__lt__", [](const core::bigint& a, const core::bigint& b) { return a < b; })
        .def("__le__", [](const core::bigint& a, const core::bigint& b) { return a <= b; })
        .def("__eq__", [](const core::bigint& a, const core::bigint& b) { return a == b; })
        .def("__ne__", [](const core::bigint& a, const core::bigint& b) { return a != b; })
        .def("__gt__", [](const core::bigint& a, const core::bigint& b) { return a > b; })
        .def("__ge__", [](const core::bigint& a, const core::bigint& b) { return a >= b; })
        .def("consensus", &core::bigint::consensus)
        .def("bitwise_and", &core::bigint::operator&, py::arg("other"))
        .def("bitwise_or", &core::bigint::operator|, py::arg("other"))
        .def("bitwise_xor", &core::bigint::operator^, py::arg("other"))
        .def("bitwise_andnot", &core::bigint::bitwise_andnot, py::arg("other"))
        .def_static("mod_pow", &core::bigint::mod_pow, py::arg("base"), py::arg("exponent"), py::arg("modulus"));

    py::class_<t81::Ratio> py_ratio(module, "Ratio", "Exact ratio built from t81bigint numerators/denominators");
    py_ratio.def(py::init<>())
        .def(py::init([](long long numerator, long long denominator) {
            return t81::Ratio(core::limb::from_value(numerator), core::limb::from_value(denominator));
        }), py::arg("numerator") = 0, py::arg("denominator") = 1)
        .def("numerator", [](const t81::Ratio& value) { return value.numerator(); })
        .def("denominator", [](const t81::Ratio& value) { return value.denominator(); })
        .def("is_zero", &t81::Ratio::is_zero)
        .def("sqrt_exact", &t81::Ratio::sqrt_exact)
        .def("__repr__", [](const t81::Ratio& value) {
            return "<t81lib.Ratio " + t81::io::to_string(value.numerator()) +
                   " / " + t81::io::to_string(value.denominator()) + ">";
        });

    bind_montgomery_context<core::limb>(module, "LimbMontgomeryContext");
    bind_montgomery_context<core::bigint>(module, "BigIntMontgomeryContext");
}
