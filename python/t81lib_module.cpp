#include <algorithm>
#include <string>

#include <pybind11/pybind11.h>

#include <t81/core/bigint.hpp>

namespace py = pybind11;
using t81::core::bigint;

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

PYBIND11_MODULE(t81lib, module) {
    module.doc() = "Python bindings for the t81lib bigint implementation";

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
