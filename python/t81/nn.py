# SPDX-License-Identifier: MIT
"""Balanced-ternary drop-in ``torch.nn.Linear`` helper."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

import t81.torch as t81_torch

_DEFAULT_THRESHOLD = 0.45


class Linear(nn.Linear):
    """Drop-in ``torch.nn.Linear`` that reuses packed ternary weights.

    Args:
        in_features: Input dimension for the matrix multiply.
        out_features: Output dimension for the matrix multiply.
        bias: Whether to include a bias term.
        device: Optional target device for the weight/bias.
        dtype: Optional dtype for the initialized parameters.
        threshold: Quantization threshold used by ``TernaryTensor``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.threshold = threshold
        self.ternary_weight: Optional[t81_torch.TernaryTensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that lazily quantizes weights before the GEMM."""
        if self.ternary_weight is None:
            self.ternary_weight = t81_torch.TernaryTensor.from_float(
                self.weight.detach(),
                threshold=self.threshold,
            )
        try:
            return self.ternary_weight.matmul_input(x, bias=self.bias)
        except TypeError as exc:  # pragma: no cover - legacy helper might omit bias
            message = str(exc)
            if "unexpected keyword argument" not in message or "bias" not in message:
                raise
            output = self.ternary_weight.matmul_input(x)
            if self.bias is not None:
                output = output + self.bias
            return output

    def extra_repr(self) -> str:
        """Include the threshold when it differs from the default."""
        base = super().extra_repr()
        if self.threshold != _DEFAULT_THRESHOLD:
            return f"{base}, threshold={self.threshold}"
        return base


__all__ = ["Linear"]
