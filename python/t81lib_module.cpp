// python/t81lib_module.cpp — Pybind11 module entrypoint exposing t81lib.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>

#include <pybind11/stl.h>

#include <t81/core/bigint.hpp>
#include <t81/linalg/gemm.hpp>

#include <t81/sparse/simple.hpp>

namespace py = pybind11;
using t81::core::bigint;
using t81::core::limb;

static std::string decimal_string(const bigint& value) {
    if (value.is_zero()) {
        return "0";
    }
    const bool negative = value.is_negative();
    bigint cursor = value.abs();
    const bigint base(10);
    std::string result;
    while (!cursor.is_zero()) {
        const auto [quotient, remainder] = bigint::div_mod(cursor, base);
        cursor = quotient;
        const long long digit_value = static_cast<long long>(remainder);
        result.push_back(static_cast<char>('0' + digit_value));
    }
    if (negative) {
        result.push_back('-');
    }
    std::reverse(result.begin(), result.end());
    return result;
}

static py::int_ to_python_int(const bigint& value) {
    const auto builtins = py::module_::import("builtins");
    return builtins.attr("int")(decimal_string(value));
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

std::span<const limb> make_limb_span(py::buffer& buffer, std::size_t count) {
    const auto info = buffer.request(false);
    if (info.size < 0) {
        throw py::value_error("Packed limb buffer has invalid size");
    }
    if (info.itemsize <= 0) {
        throw py::value_error("Packed limb buffer has invalid item size");
    }
    if (!buffer_is_c_contiguous(info)) {
        throw py::value_error("A/B buffers must use C-style contiguous memory");
    }
    const std::size_t total_bytes =
        static_cast<std::size_t>(std::max<py::ssize_t>(info.size, 0)) *
        static_cast<std::size_t>(info.itemsize);
    if (total_bytes != count * sizeof(limb)) {
        throw py::value_error("Packed limb buffer byte size mismatch");
    }
    const auto* data = info.ptr ? reinterpret_cast<const limb*>(info.ptr) : nullptr;
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
    if (info.itemsize <= 0) {
        throw py::value_error("C buffer has invalid item size");
    }
    if (!buffer_is_c_contiguous(info)) {
        throw py::value_error("C buffer must use C-style contiguous memory");
    }
    if (info.itemsize != static_cast<py::ssize_t>(sizeof(float))) {
        throw py::value_error("C buffer must store float32 values");
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

} // namespace

PYBIND11_MODULE(t81lib, module) {
    module.doc() = "Python bindings for the t81lib bigint implementation";

    module.def(
        "gemm_ternary",
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
            if (K % limb::TRITS != 0) {
                throw py::value_error("K must be divisible by 48");
            }
            const int K_limbs = K / limb::TRITS;
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
        py::arg("A_packed"),
        py::arg("B_packed"),
        py::arg("C"),
        py::arg("M"),
        py::arg("N"),
        py::arg("K"),
        py::arg("alpha") = 1.0f,
        py::arg("beta") = 0.0f,
        "AVX/NEON-accelerated GEMM over packed ternary limbs");

    module.def(
        "spmm_simple",
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
        "COO-style sparse × dense multiply over ternary weights");

    module.def("zero", &bigint::zero, "Return a bigint representing zero");
    module.def("one", &bigint::one, "Return a bigint representing one");
    module.def("gcd", &bigint::gcd, py::arg("lhs"), py::arg("rhs"),
               "Compute the greatest common divisor of two bigints");
    module.def("mod_pow", &bigint::mod_pow, py::arg("base"), py::arg("exponent"),
               py::arg("modulus"), "Modular exponentiation");

    py::class_<bigint> py_bigint(module, "BigInt");
    py_bigint.def(py::init<>())
        .def(py::init<long long>(), py::arg("value"))
        .def("__int__", &to_python_int)
        .def("__str__", [](const bigint& value) { return decimal_string(value); })
        .def("__repr__", [](const bigint& value) {
            return "<t81lib.BigInt " + decimal_string(value) + ">";
        })
        .def("__bool__", [](const bigint& value) { return !value.is_zero(); })
        .def("__add__", [](const bigint& lhs, const bigint& rhs) { return lhs + rhs; })
        .def("__sub__", [](const bigint& lhs, const bigint& rhs) { return lhs - rhs; })
        .def("__mul__", [](const bigint& lhs, const bigint& rhs) { return lhs * rhs; })
        .def("__floordiv__", [](const bigint& lhs, const bigint& rhs) { return lhs / rhs; })
        .def("__mod__", [](const bigint& lhs, const bigint& rhs) { return lhs % rhs; })
        .def("__neg__", [](const bigint& value) { return -value; })
        .def("__abs__", &bigint::abs)
        .def("__and__", [](const bigint& lhs, const bigint& rhs) { return lhs & rhs; })
        .def("__or__", [](const bigint& lhs, const bigint& rhs) { return lhs | rhs; })
        .def("__xor__", [](const bigint& lhs, const bigint& rhs) { return lhs ^ rhs; })
        .def("__lshift__", [](const bigint& value, int count) {
            return value << count;
        })
        .def("__rshift__", [](const bigint& value, int count) {
            return value >> count;
        })
        .def("__eq__", [](const bigint& lhs, const bigint& rhs) {
            return lhs == rhs;
        })
        .def("__lt__", [](const bigint& lhs, const bigint& rhs) {
            return lhs < rhs;
        })
        .def("consensus", &bigint::consensus);

    py_bigint.def_static("gcd", &bigint::gcd);
}
