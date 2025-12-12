"""
t81 package root for the PyTorch helpers and helpers that live alongside ``t81lib``'s bindings.

The torch subpackage is deliberately minimal so consumers can ``import t81`` and access
``t81.trit``/``t81.trt`` helpers together with ``t81.torch`` without conflicting with the
system ``torch`` module.
"""

from __future__ import annotations

from typing import Iterable

try:
    import t81lib as _t81lib
except ImportError as exc:  # pragma: no cover - best effort binding availability
    _t81lib = None
    _t81lib_error = exc
else:
    _t81lib_error = None

_binding_exports = [
    "BigInt",
    "Limb",
    "Ratio",
    "LimbMontgomeryContext",
    "BigIntMontgomeryContext",
    "gemm_ternary",
    "spmm_simple",
    "quantize_to_trits",
    "dequantize_trits",
    "pack_dense_matrix",
    "quantize_row_tq1_0",
    "dequant_row_tq1_0",
    "dequant_tq1_0",
    "dequant_tq2_0",
    "unpack_packed_limbs",
    "limb_from_bytes",
    "bigint_gcd",
    "bigint_mod_pow",
]

__all__ = ["torch", "hardware", "Linear", "t81lib"] + _binding_exports

from . import hardware  # noqa: E402 - re-exported helpers
from . import torch as torch_integration  # noqa: E402 - re-exported module
from .nn import Linear as Linear  # noqa: E402 - re-exported layer

torch = torch_integration


def _missing_binding(name: str) -> None:
    raise ImportError(
        f"{name} is provided by the compiled t81lib bindings. "
        "Build the project with -DT81LIB_BUILD_PYTHON_BINDINGS=ON, install it, and rerun your script."
    ) from _t81lib_error


def __getattr__(name: str):
    if name == "t81lib":
        if _t81lib is None:
            _missing_binding("t81lib")
        return _t81lib
    if name in _binding_exports:
        if _t81lib is None:
            _missing_binding(name)
        return getattr(_t81lib, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> Iterable[str]:
    return sorted(set(globals()) | set(__all__))
