"""
PyTorch helpers that bridge ``t81lib``'s packed ternary GEMM into ``torch`` workflows.
This CPU-only prototype keeps a float copy of each parameter so gradients stay in floating
point, re-quantizes right before ``nn.Linear``/``torch.matmul`` runs, and calls
``t81lib.gemm_ternary`` for the heavy matrix multiply.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn

import t81lib


def _build_tryte_tables() -> tuple[np.ndarray, np.ndarray]:
    trits_to_tryte = np.empty(27, dtype=np.uint8)
    tryte_to_trits = np.empty((27, 3), dtype=np.int8)
    for t2 in (-1, 0, 1):
        for t1 in (-1, 0, 1):
            for t0 in (-1, 0, 1):
                tri_index = (t0 + 1) + 3 * (t1 + 1) + 9 * (t2 + 1)
                tryte_index = (t0 + 3 * t1 + 9 * t2) + 13
                trits_to_tryte[tri_index] = tryte_index
                tryte_to_trits[tryte_index] = (t0, t1, t2)
    return trits_to_tryte, tryte_to_trits


_TRITS_TO_TRYTE, _TRYTE_TO_TRITS = _build_tryte_tables()


def _write_tryte_block(destination: np.ndarray, block: np.ndarray) -> None:
    # Write three trits per tryte slot into the packed limb buffer.
    for tryte_index in range(16):
        triple = block[tryte_index]
        tri_index = (int(triple[0]) + 1) + 3 * (int(triple[1]) + 1) + 9 * (int(triple[2]) + 1)
        destination[tryte_index] = _TRITS_TO_TRYTE[tri_index]


def _quantize_tensor(tensor: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    threshold = float(max(0.0, min(threshold, 0.9999)))
    clipped = tensor.clamp(-1.0, 1.0)
    trits = torch.zeros_like(clipped, dtype=torch.int8)
    trits[clipped >= threshold] = 1
    trits[clipped <= -threshold] = -1
    return trits


def _to_cpu_float(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cpu":
        raise NotImplementedError("t81.trit currently only supports CPU tensors")
    return tensor.detach().to(dtype=torch.float32, copy=False)


def _pack_rowwise(trits: np.ndarray, rows: int, k_limbs: int, k_actual: int) -> np.ndarray:
    # Pack row-major trits so each row of the matrix produces k_limbs limbs.
    padded = np.zeros((rows, k_limbs * 48), dtype=np.int8)
    padded[:, :k_actual] = trits[:, :k_actual]
    packed = np.empty((rows, k_limbs), dtype=np.dtype("V16"))
    packed_view = packed.view(np.uint8).reshape(rows * k_limbs, 16)
    triples = padded.reshape(rows, k_limbs, 16, 3)
    write_idx = 0
    for row in range(rows):
        for limb_idx in range(k_limbs):
            _write_tryte_block(packed_view[write_idx], triples[row, limb_idx])
            write_idx += 1
    return packed


def _pack_columnwise(trits: np.ndarray, k_limbs: int, k_actual: int) -> np.ndarray:
    # Pack column-major data so GEMM sees contiguous limbs per column group.
    _, cols = trits.shape
    padded = np.zeros((k_limbs * 48, cols), dtype=np.int8)
    padded[:k_actual, :] = trits[:k_actual, :]
    packed = np.empty((k_limbs, cols), dtype=np.dtype("V16"))
    packed_view = packed.view(np.uint8).reshape(k_limbs * cols, 16)
    idx = 0
    for limb_idx in range(k_limbs):
        chunk = padded[limb_idx * 48 : (limb_idx + 1) * 48]
        triples = chunk.reshape(16, 3, cols)
        for col in range(cols):
            _write_tryte_block(packed_view[idx], triples[:, :, col])
            idx += 1
    return packed


def _limbs_to_trits(packed: np.ndarray, rows: int, k_limbs: int) -> np.ndarray:
    view = packed.view(np.uint8).reshape(rows * k_limbs, 16)
    trits = np.empty((rows, k_limbs * 48), dtype=np.int8)
    cursor = 0
    for row in range(rows):
        for limb_idx in range(k_limbs):
            tryte_values = view[cursor]
            for tryte_idx in range(16):
                trits[row, limb_idx * 48 + tryte_idx * 3 : limb_idx * 48 + tryte_idx * 3 + 3] = (
                    _TRYTE_TO_TRITS[tryte_values[tryte_idx]]
                )
            cursor += 1
    return trits


class TritDtype:
    __slots__ = ()

    def __new__(cls) -> "TritDtype":
        return super().__new__(cls)

    def __repr__(self) -> str:
        return "t81.trit"


trit = TritDtype()


def _extract_dtype_arg(args: Sequence[Any], kwargs: Mapping[str, Any]) -> Optional[torch.dtype]:
    dtype = kwargs.get("dtype")
    if dtype is not None:
        return dtype
    for arg in args:
        if isinstance(arg, torch.dtype):
            return arg
    return None


def _sync_ternary_weights(module: nn.Module, _: Sequence[Any]) -> None:
    cache: Dict[str, "TernaryTensor"] = getattr(module, "_t81_ternary_cache", {})
    for name, ternary in cache.items():
        source = getattr(module, name)
        # Keep the quantized cache consistent with the float weights before a forward.
        ternary.update_from_float(source)


def _apply_ternary_conversion(module: nn.Module) -> None:
    for submodule in module.modules():
        if hasattr(submodule, "_t81_ternary_cache"):
            continue
        cache: Dict[str, "TernaryTensor"] = {}
        for name, parameter in submodule.named_parameters(recurse=False):
            if parameter.ndim != 2:
                continue
            threshold = getattr(
                submodule,
                "ternary_threshold",
                getattr(submodule, "threshold", 0.5),
            )
            cache[name] = TernaryTensor.from_float(parameter.detach(), threshold=threshold)
        if cache:
            setattr(submodule, "_t81_ternary_cache", cache)
            if not hasattr(submodule, "_t81_hook_handle"):
                handle = submodule.register_forward_pre_hook(_sync_ternary_weights)
                setattr(submodule, "_t81_hook_handle", handle)


class TernaryTensor(torch.Tensor):
    __slots__ = (
        "_packed",
        "_packed_view",
        "_rows",
        "_k_actual",
        "_k_limbs",
        "_float_source",
        "_base_device",
        "_threshold",
    )

    def __new__(
        cls,
        packed: np.ndarray,
        shape: tuple[int, int],
        *,
        k_actual: int,
        k_limbs: int,
        float_source: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> "TernaryTensor":
        storage = torch.empty(shape, dtype=torch.float32, device="cpu")
        tensor = torch.Tensor._make_subclass(cls, storage, False)
        tensor._packed = packed
        tensor._packed_view = packed.view(np.uint8).reshape(-1, 16)
        tensor._rows = shape[0]
        tensor._k_actual = k_actual
        tensor._k_limbs = k_limbs
        tensor._float_source = float_source
        tensor._base_device = torch.device("cpu")
        tensor._threshold = float(max(0.0, min(threshold, 0.9999)))
        return tensor

    @property
    def dtype(self) -> TritDtype:  # type: ignore[override]
        return trit

    def __repr__(self) -> str:
        return f"TernaryTensor(shape={self.shape}, trits_per_row={self._k_actual})"

    @staticmethod
    def _ensure_int2d(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 2:
            raise ValueError("t81.trit tensors currently only support 2D matrices")
        return tensor

    @classmethod
    def from_float(
        cls, source: torch.Tensor, quantize: bool = True, threshold: float = 0.5
    ) -> "TernaryTensor":
        tensor = cls._ensure_int2d(source)
        cpu_float = _to_cpu_float(tensor).contiguous()
        rows, cols = cpu_float.shape
        k_limbs = (cols + 47) // 48
        quantized = _quantize_tensor(cpu_float, threshold) if quantize else cpu_float.to(torch.int8)
        packed = _pack_rowwise(quantized.numpy(), rows, k_limbs, cols)
        return cls(
            packed,
            (rows, cols),
            k_actual=cols,
            k_limbs=k_limbs,
            float_source=cpu_float,
            threshold=threshold,
        )

    def update_from_float(self, source: torch.Tensor) -> None:
        cpu_float = _to_cpu_float(self._ensure_int2d(source)).contiguous()
        threshold = getattr(self, "_threshold", 0.5)
        quantized = _quantize_tensor(cpu_float, threshold)
        packed = _pack_rowwise(quantized.numpy(), cpu_float.shape[0], self._k_limbs, self._k_actual)
        self._packed_view[:, :] = packed.view(np.uint8).reshape(-1, 16)
        self._float_source = cpu_float

    def matmul_input(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim != 2:
            raise ValueError("t81.trit linear helpers expect 2D activations")
        rhs = input.transpose(-2, -1)
        output = _TernaryGemmFunction.apply(self, rhs)
        return output.transpose(-2, -1)

    @classmethod
    def __torch_function__(cls, func: Callable, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in {torch.matmul, torch.mm}:
            lhs, rhs = args
            if isinstance(lhs, TernaryTensor):
                return _TernaryGemmFunction.apply(lhs, rhs)
        return NotImplemented

    def _pack_rhs_for_gemm(self, rhs: torch.Tensor) -> np.ndarray:
        cpu_rhs = _to_cpu_float(rhs).contiguous()
        if cpu_rhs.shape[0] != self._k_actual:
            raise ValueError("rhs dimension does not match ternary weight width")
        quantized = _quantize_tensor(cpu_rhs)
        return _pack_columnwise(quantized.numpy(), self._k_limbs, self._k_actual)

    def _compute_gemm(self, rhs: torch.Tensor) -> torch.Tensor:
        rhs_matrix = rhs.contiguous()
        packed_rhs = self._pack_rhs_for_gemm(rhs_matrix)
        output = np.zeros((self._rows * rhs_matrix.shape[1],), dtype=np.float32)
        t81lib.gemm_ternary(
            self._packed,
            packed_rhs,
            output,
            self._rows,
            rhs_matrix.shape[1],
            self._k_limbs * 48,
        )
        return torch.from_numpy(output.reshape(self._rows, rhs_matrix.shape[1]))


class _TernaryGemmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ternary_weight: TernaryTensor, rhs: torch.Tensor) -> torch.Tensor:
        rhs_cpu = _to_cpu_float(rhs)
        ctx.save_for_backward(rhs_cpu)
        ctx.ternary = ternary_weight
        return ternary_weight._compute_gemm(rhs_cpu)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[None, torch.Tensor]:
        rhs, = ctx.saved_tensors
        weight_float = torch.from_numpy(
            _limbs_to_trits(ctx.ternary._packed, ctx.ternary._rows, ctx.ternary._k_limbs)
            .astype(np.float32)
        )[:, : ctx.ternary._k_actual]
        # Gradient for rhs follows the usual matmul gradient formula.
        grad_rhs = weight_float.transpose(-2, -1).matmul(grad_output)
        return None, grad_rhs


_original_module_to = nn.Module.to


def _ternary_module_to(self, *args, **kwargs):
    dtype = _extract_dtype_arg(args, kwargs)
    module = _original_module_to(self, *args, **kwargs)
    if dtype is trit:
        # Whenever a module requests the t81.trit dtype we tag every child with a cache.
        _apply_ternary_conversion(module)
    return module

nn.Module.to = _ternary_module_to

_original_linear_forward = nn.Linear.forward


def _ternary_linear_forward(self, input: torch.Tensor) -> torch.Tensor:
    cache: Dict[str, TernaryTensor] = getattr(self, "_t81_ternary_cache", {})
    ternary = cache.get("weight")
    if ternary is not None:
        # Linear layers with cached ternary copies call the fast GEMM path.
        _sync_ternary_weights(self, ())
        output = ternary.matmul_input(input)
        if self.bias is not None:
            output = output + self.bias
        return output
    return _original_linear_forward(self, input)

nn.Linear.forward = _ternary_linear_forward

__all__ = ["trit", "TernaryTensor", "TritDtype"]
