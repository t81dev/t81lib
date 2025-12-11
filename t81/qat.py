"""Quantization helpers for ternary-aware PyTorch training."""

from __future__ import annotations

import torch


def _clamp_threshold(threshold: float) -> float:
    return float(max(0.0, min(threshold, 0.9999)))


def _quantize_tensor(x: torch.Tensor, threshold: float, stochastic: bool) -> torch.Tensor:
    desc = x.detach()
    clamped = torch.clamp(desc, -1.0, 1.0)
    abs_value = clamped.abs()
    sign = torch.sign(clamped)
    quantized = torch.zeros_like(clamped)
    mask = abs_value >= threshold
    if stochastic:
        denom = max(1.0 - threshold, 1e-6)
        probability = torch.clamp((abs_value - threshold) / denom, 0.0, 1.0)
        noise = torch.rand_like(probability)
        mask = mask & (noise < probability)
    quantized = quantized.masked_scatter(mask, sign[mask])
    return quantized


def dorso_round(x: torch.Tensor) -> torch.Tensor:
    """Dorsal rounding keeps gradient flow around zero while clipping to ternary values."""

    high = torch.ones_like(x)
    low = -high
    scaled = x * 2.0
    return torch.where(x >= 0.5, high, torch.where(x <= -0.5, low, scaled))


class _TernarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, surrogate: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(surrogate)
        return quantized

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        _surrogate, = ctx.saved_tensors
        return grad_output, None


def ternary(x: torch.Tensor, threshold: float = 0.45, stochastic: bool = False) -> torch.Tensor:
    """Quantize ``x`` to balanced ternary with optional stochastic rounding."""

    threshold = _clamp_threshold(threshold)
    quantized = _quantize_tensor(x, threshold, stochastic)
    surrogate = dorso_round(x)
    return _TernarySTE.apply(surrogate, quantized)
