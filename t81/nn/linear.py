"""Ternary-aware ``nn.Linear`` helper with quantization-aware training hooks."""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

import t81.torch as t81_torch
from t81.qat import ternary

_DEFAULT_THRESHOLD = 0.45


class Linear(nn.Linear):
    """Balanced-ternary ``nn.Linear`` that plays nice with QAT and inference caches."""

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
        self.ternary_threshold = threshold
        self.ternary_weight: Optional[t81_torch.TernaryTensor] = None
        self.register_buffer("training_scale", None)
        self._stochastic_rounding = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            quantized = self._quantize_training_weight()
            return F.linear(x, quantized, self.bias)
        if self.ternary_weight is None:
            self.ternary_weight = t81_torch.TernaryTensor.from_float(
                self.weight.detach(),
                threshold=self.ternary_threshold,
            )
        return self._packed_linear(x)

    def _packed_linear(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            batch, seq, features = x.shape
            flat = x.reshape(batch * seq, features)
            packed = self._packed_linear(flat)
            return packed.reshape(batch, seq, -1)
        if x.ndim != 2:
            raise ValueError("t81.nn.Linear expects 2D or 3D inputs")
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

    def _quantize_training_weight(self) -> torch.Tensor:
        scale = self._update_training_scale()
        scaled = self.weight / scale
        quantized = ternary(
            scaled,
            threshold=self.ternary_threshold,
            stochastic=self._stochastic_rounding,
        )
        return quantized * scale

    def _update_training_scale(self) -> torch.Tensor:
        device = self.weight.device
        raw_max = self.weight.detach().abs().amax()
        if isinstance(raw_max, torch.Tensor):
            candidate = raw_max.to(device=device)
        else:
            candidate = torch.tensor(raw_max, device=device)
        candidate = candidate.clamp(min=1e-6).to(dtype=torch.float32)
        existing = self.training_scale
        if existing is None or existing.device != device:
            self.training_scale = candidate.detach()
            return self.training_scale
        momentum = 0.99
        updated = existing.to(device=device, dtype=torch.float32)
        combined = updated * momentum + candidate * (1.0 - momentum)
        combined = combined.clamp(min=1e-6)
        self.training_scale = combined.detach()
        return self.training_scale

    def extra_repr(self) -> str:
        base = super().extra_repr()
        if self.ternary_threshold != _DEFAULT_THRESHOLD:
            return f"{base}, threshold={self.ternary_threshold}"
        return base

    def train(self, mode: bool = True) -> "Linear":
        if mode:
            self.ternary_weight = None
        return super().train(mode)

    def load_state_dict(self, state_dict: OrderedDict[str, torch.Tensor], strict: bool = True):
        normalized = OrderedDict()
        for key, value in state_dict.items():
            if key in {"weight", "bias"} and value.is_floating_point():
                if value.dtype in {torch.bfloat16, torch.float16}:
                    value = value.to(torch.float32)
            normalized[key] = value
        result = super().load_state_dict(normalized, strict=strict)
        self.ternary_weight = None
        self.training_scale = None
        return result

    def configure_qat(self, *, stochastic_rounding: bool) -> None:
        self._stochastic_rounding = stochastic_rounding
