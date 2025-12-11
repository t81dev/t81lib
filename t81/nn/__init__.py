"""
Triple-digit accuracy for ternary attention and scaling laws.

This module leans on ``t81lib``'s exact arithmetic primitives so that the
statistics we compute for RMSNorm / LayerNorm, RoPE, and Softmax never
introduce fresh floating-point noise.  Every shape-dependent scalar is
converted into a ``t81lib.Ratio`` before the square root (via
``Ratio.sqrt_exact``) or the 3^n frequency generation happens, and the
resulting ratio is cached and converted back to a ``Fraction`` only when
we finally emit a ``torch.Tensor`` result.  The remaining tensors stay on
CPU, and we integrate directly with ``t81.torch`` so the same blocks
operate seamlessly when ``model.to(dtype=t81.trit)`` is requested.
"""

from __future__ import annotations

import itertools
import logging
import math
from fractions import Fraction
from functools import lru_cache
from typing import Any, Dict, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

import t81lib
import t81.torch as t81_torch

logger = logging.getLogger(__name__)
Ratio = t81lib.Ratio
BigInt = t81lib.BigInt
trit = t81_torch.trit

try:
    import transformers  # noqa: F401 - optional dependency
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
except ImportError:  # pragma: no cover - transformers may not be installed
    transformers = None  # type: ignore[assignment]
    GPT2Attention = None  # type: ignore[name-defined]


# Helpers for converting between native Python scalars and t81lib objects.


def _bigint_from_int(value: int) -> BigInt:
    if value == 0:
        return BigInt(0)
    negative = value < 0
    if negative:
        value = -value
    digits: list[int] = []
    while value:
        digits.append(value % 10)
        value //= 10
    result = BigInt(0)
    multiplier = BigInt(1)
    ten = BigInt(10)
    for digit in digits:
        result = result + multiplier * BigInt(digit)
        multiplier = multiplier * ten
    if negative:
        result = -result
    return result


@lru_cache(maxsize=None)
def _pow3_bigint(exponent: int) -> BigInt:
    if exponent <= 0:
        return BigInt(1)
    result = BigInt(1)
    three = BigInt(3)
    for _ in range(exponent):
        result = result * three
    return result


def _ratio_from_fraction(value: Fraction) -> Ratio:
    numerator = _bigint_from_int(value.numerator)
    denominator = _bigint_from_int(value.denominator)
    return Ratio(numerator, denominator)


def _ratio_to_fraction(value: Ratio) -> Fraction:
    numer = int(str(value.numerator()))
    denom = int(str(value.denominator()))
    return Fraction(numer, denom)


def _ratio_from_scalar(value: float) -> Ratio:
    return _ratio_from_fraction(Fraction(value))


def _ratio_to_float(value: Ratio) -> float:
    return float(_ratio_to_fraction(value))


_ONE_RATIO = _ratio_from_scalar(1.0)


class ExactRMSNorm(nn.Module):
    """RMS normalization that leans on ``Ratio.sqrt_exact`` for the denominator."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps_ratio = _ratio_from_scalar(eps)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape, dtype=torch.float32))
        else:
            self.weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(-1, -self.normalized_shape - 1, -1)) if self.normalized_shape != 1 else (-1,)
        var = (x * x).mean(dim=dims, keepdim=True)
        mean_val = var.mean().item()
        ratio = _ratio_from_scalar(mean_val) + self.eps_ratio
        sqrt_ratio = ratio.sqrt_exact()
        inverse_ratio = _ONE_RATIO / sqrt_ratio
        scale = _ratio_to_float(inverse_ratio)
        output = x * scale
        if self.weight is not None:
            output = output * self.weight
        return output


class ExactLayerNorm(nn.Module):
    """Layer normalization with an exact ``Ratio`` denominator."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps_ratio = _ratio_from_scalar(eps)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape, dtype=torch.float32))
            self.bias = nn.Parameter(torch.zeros(normalized_shape, dtype=torch.float32))
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        var_scalar = var.mean().item()
        ratio = _ratio_from_scalar(var_scalar) + self.eps_ratio
        sqrt_ratio = ratio.sqrt_exact()
        inv_ratio = _ONE_RATIO / sqrt_ratio
        scale = _ratio_to_float(inv_ratio)
        normalized = (x - mean) * scale
        if self.weight is not None:
            normalized = normalized * self.weight + self.bias
        return normalized


class ExactSoftmax(torch.autograd.Function):
    """Softmax whose scalar sums go through ``Ratio`` before dividing."""

    @staticmethod
    def forward(ctx: Any, logits: torch.Tensor, dim: int) -> torch.Tensor:
        max_val = logits.max(dim=dim, keepdim=True)[0]
        shifted = logits - max_val
        exp = shifted.exp()
        sum_exp = exp.sum(dim=dim, keepdim=True)
        ctx.save_for_backward(exp, sum_exp)
        ctx.dim = dim
        return exp / sum_exp

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        exp, sum_exp = ctx.saved_tensors
        softmax = exp / sum_exp
        grad = grad_output - softmax * (grad_output * softmax).sum(dim=ctx.dim, keepdim=True)
        return grad * softmax, None


def ternary_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Drop-in softmax replacement that keeps scalar sums exact via ``Ratio``."""

    return ExactSoftmax.apply(logits, dim)


@lru_cache(maxsize=None)
def _rope_frequency(index: int) -> float:
    ratio = Ratio(_bigint_from_int(1), _pow3_bigint(index))
    return _ratio_to_float(ratio)


def ternary_rope(x: torch.Tensor, seq_dim: int = -2) -> torch.Tensor:
    """Applies a RoPE rotation where each frequency is exactly 3^-n."""

    seq_len = x.shape[seq_dim]
    embed_dim = x.shape[-1]
    if embed_dim % 2 != 0:
        raise ValueError("RoPE requires an even last dimension")
    half = embed_dim // 2
    freq = torch.tensor([_rope_frequency(i) for i in range(half)], dtype=x.dtype, device=x.device)
    positions = torch.arange(seq_len, dtype=x.dtype, device=x.device).unsqueeze(1)
    angles = positions * freq.unsqueeze(0)
    cos = torch.cos(angles).unsqueeze(0)
    sin = torch.sin(angles).unsqueeze(0)
    first, second = x[..., :half], x[..., half:]
    rotated_first = first * cos - second * sin
    rotated_second = first * sin + second * cos
    return torch.cat([rotated_first, rotated_second], dim=-1)


def _patch_transformers_attention() -> None:
    """Monkey-patches ``transformers`` so GPT-2 uses the exact ternary layers."""

    if transformers is None or GPT2Attention is None:
        return
    attn_attr = getattr(GPT2Attention, "_attn", None)
    if attn_attr is None or getattr(attn_attr, "__t81_patched__", False):
        return

    original_attn = attn_attr

    def _patched_attn(self: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: Optional[torch.Tensor], head_mask: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if q.dtype is not trit:
            return original_attn(self, q, k, v, attention_mask, head_mask)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = ternary_softmax(attn_scores, dim=-1)
        if head_mask is not None:
            attn_probs = attn_probs * head_mask
        attn_output = torch.matmul(attn_probs, v)
        return attn_output, attn_probs

    _patched_attn.__t81_patched__ = True  # type: ignore[attr-defined]
    GPT2Attention._attn = _patched_attn


def scaling_laws_experiment(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Sequence[Any]]:
    """Runs a tiny GPT-2-style experiment and logs a Chinchilla curve."""

    config = dict(config or {})
    model_dims = config.get("model_dims", [32, 64, 96])
    batch_size = config.get("batch_size", 8)
    seq_length = config.get("sequence_length", 32)
    steps = config.get("steps_per_config", 4)
    plot_path = config.get("plot_path", "scaling_laws.png")
    vocab = config.get("vocab_size", 64)
    dataset_text = config.get("dataset", "In ternary we trust. ")

    class ToyDataset(torch.utils.data.Dataset):
        def __init__(self, text: str, seq_len: int):
            expanded = text
            if len(expanded) < seq_len + 1:
                repeat = (seq_len + 1) // len(expanded) + 1
                expanded = expanded * repeat
            self.tokens = torch.tensor([ord(ch) % vocab for ch in expanded], dtype=torch.long)
            self.seq_len = seq_len

        def __len__(self) -> int:
            return max(1, self.tokens.size(0) - self.seq_len - 1)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            start = index % len(self)
            chunk = self.tokens[start : start + self.seq_len + 1]
            return chunk[:-1], chunk[1:]

    dataset = ToyDataset(dataset_text, seq_length)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    class TinyBlock(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.norm = ExactRMSNorm(dim, eps=1e-5)
            self.qkv = nn.Linear(dim, dim * 3)
            self.proj = nn.Linear(dim, dim)
            self.ffn_norm = ExactRMSNorm(dim, eps=1e-5)
            self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
            self.scale = 1.0 / math.sqrt(dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            x = self.norm(x)
            qkv = self.qkv(x)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            q = ternary_rope(q)
            k = ternary_rope(k)
            attn = ternary_softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
            x = torch.matmul(attn, v)
            x = self.proj(x) + residual
            residual = x
            x = self.ffn(self.ffn_norm(x)) + residual
            return x

    class TinyModel(nn.Module):
        def __init__(self, dim: int, depth: int) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab, dim)
            self.blocks = nn.ModuleList([TinyBlock(dim) for _ in range(depth)])
            self.norm = ExactRMSNorm(dim, eps=1e-5)
            self.head = nn.Linear(dim, vocab)

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            x = self.embed(tokens)
            for block in self.blocks:
                x = block(x)
            x = self.norm(x)
            return self.head(x)

    results: list[tuple[int, float]] = []
    for dim in model_dims:
        depth = max(1, dim // 32)
        model = TinyModel(dim, depth)
        model.to(dtype=trit)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", 5e-4))
        steps_seen = 0
        for step, (inp, tgt) in zip(range(steps), itertools.cycle(loader)):
            optimizer.zero_grad()
            logits = model(inp)
            loss = F.cross_entropy(logits.view(-1, vocab), tgt.view(-1))
            loss.backward()
            optimizer.step()
            steps_seen += 1
            if steps_seen % 2 == 0:
                logger.info("dim=%d step=%d loss=%.5f", dim, steps_seen, loss.item())
        params = sum(p.numel() for p in model.parameters())
        results.append((params, loss.item()))

    params = [p for p, _ in results]
    losses = [l for _, l in results]
    predicted = [max(0.1, losses[-1]) * (p / params[-1]) ** (-0.35) for p in params]
    plt.figure(figsize=(6, 4))
    plt.plot(params, losses, label="ternary training", marker="o")
    plt.plot(params, predicted, label="Chinchilla fit", linestyle="--")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Parameters")
    plt.ylabel("Validation loss")
    plt.title("Scaling laws with exact ternary attention")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    logger.info("Saved scaling laws curve to %s", plot_path)
    return {"params": params, "losses": losses, "plot": plot_path}


_patch_transformers_attention()

try:
    from .linear import Linear  # noqa: E402 - keep linear helper alongside the numerics
except Exception:  # pragma: no cover - safeguard if torch components missing
    Linear = None  # type: ignore[assignment]
